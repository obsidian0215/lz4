/*
 * LZ4 GPU Compressor Host Implementation (C version)
 * Pure C implementation for C projects
 */

#include "lz4_gpu_host.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h> /* for readlink */

/* Host-side debug macro. Define LZ4_GPU_HOST_DEBUG to enable verbose host
 * debug prints. By default these are compiled out to keep logs concise. Use
 * printf for errors unconditionally. */
#ifdef LZ4_GPU_HOST_DEBUG
#define HDEBUG(...) printf(__VA_ARGS__)
#else
#define HDEBUG(...) do {} while(0)
#endif

// Error messages
static const char* LZ4_GPU_ERRORS[] = {
    "Success",
    "Failed to get OpenCL platform",
    "Failed to get OpenCL device",
    "Failed to create OpenCL context",
    "Failed to create command queue",
    "Failed to create program",
    "Failed to build program",
    "Failed to create kernel",
    "Failed to create buffer",
    "Failed to upload data",
    "Failed to download data",
    "Failed to set kernel arguments",
    "Failed to launch kernel",
    "Compression failed",
    "Decompression failed",
    "Invalid input parameters",
    "Buffer too small",
    "GPU not available"
};

// LZ4 Memory functions (for host code)
unsigned int LZ4_readLE32(const unsigned char* ptr) {
    return (unsigned int)ptr[0] | ((unsigned int)ptr[1] << 8) | ((unsigned int)ptr[2] << 16) | ((unsigned int)ptr[3] << 24);
}

void LZ4_writeLE32(unsigned char* ptr, unsigned int value) {
    ptr[0] = (unsigned char)value;
    ptr[1] = (unsigned char)(value >> 8);
    ptr[2] = (unsigned char)(value >> 16);
    ptr[3] = (unsigned char)(value >> 24);
}

void LZ4_writeLE64(unsigned char* ptr, unsigned long long value) {
    ptr[0] = (unsigned char)value;
    ptr[1] = (unsigned char)(value >> 8);
    ptr[2] = (unsigned char)(value >> 16);
    ptr[3] = (unsigned char)(value >> 24);
    ptr[4] = (unsigned char)(value >> 32);
    ptr[5] = (unsigned char)(value >> 40);
    ptr[6] = (unsigned char)(value >> 48);
    ptr[7] = (unsigned char)(value >> 56);
}

// LZ4F_headerChecksum function - simplified version matching original LZ4 source
unsigned char LZ4F_headerChecksum(const void* header, size_t length) {
    // Use the same algorithm as the original LZ4F_headerChecksum function
    unsigned int xxh = XXH32(header, length, 0);
    return (unsigned char)(xxh >> 8);
}

LZ4GPUCompressor* lz4_gpu_create_compressor(void) {
    LZ4GPUCompressor* compressor = (LZ4GPUCompressor*)malloc(sizeof(LZ4GPUCompressor));
    if (!compressor) {
        return NULL;
    }

    memset(compressor, 0, sizeof(LZ4GPUCompressor));
    compressor->last_error = LZ4_GPU_SUCCESS;
        compressor->enable_kernel_debug = 0; // Default: keep logs quiet (only enable via CLI -g or debug build)
        compressor->prefer_precompiled = 1; /* default to prefer using precompiled clbin when available */
        compressor->precompiled_path[0] = '\0';
        compressor->kernel_src_path[0] = '\0';
        compressor->enable_profiling = 0; // By default profiling is disabled; user may enable via API or CLI (-p/--profile)
        compressor->last_timing.total_ms = 0.0;
        compressor->last_timing.init_ms = 0.0;
        compressor->last_timing.alloc_ms = 0.0;
        compressor->last_timing.h2d_ms = 0.0;
        compressor->last_timing.kernel_ms = -1.0;
        compressor->last_timing.kernel_ms_device = -1.0;
        compressor->last_timing.d2h_ms = 0.0;
        compressor->last_timing.frame_ms = 0.0;
        compressor->last_timing.input_bytes = 0;
        compressor->last_timing.output_bytes = 0;
        compressor->last_timing.num_blocks = 0;
        compressor->compress_block_size = 0; /* 0 -> use dynamic block sizing */
        compressor->decompress_block_size = 0; /* 0 -> use frame's block layout */
        compressor->default_local = LZ4_GPU_DEFAULT_LOCAL_SIZE;
        compressor->default_acceleration = LZ4_GPU_DEFAULT_ACCELERATION;
        compressor->use_pinned_memory = LZ4_GPU_DEFAULT_PINNED;
        compressor->enable_io_overlap = 0; /* default: disable overlapping transfers (can enable via --io-overlap) */

    return compressor;
}

void lz4_gpu_destroy_compressor(LZ4GPUCompressor* compressor) {
    if (!compressor) return;

    if (compressor->compress_kernel) clReleaseKernel(compressor->compress_kernel);
    if (compressor->decompress_kernel) clReleaseKernel(compressor->decompress_kernel);
    if (compressor->frame_decompress_kernel) clReleaseKernel(compressor->frame_decompress_kernel);

    /* Unmap pinned pointers (if present) before releasing buffers */
    if (compressor->input_buffer) {
        if (compressor->pinned_input_ptr) {
            clEnqueueUnmapMemObject(compressor->queue, compressor->input_buffer, compressor->pinned_input_ptr, 0, NULL, NULL);
            compressor->pinned_input_ptr = NULL;
            compressor->pinned_input_size = 0;
        }
        clReleaseMemObject(compressor->input_buffer);
    }
    if (compressor->output_buffer) {
        if (compressor->pinned_output_ptr) {
            clEnqueueUnmapMemObject(compressor->queue, compressor->output_buffer, compressor->pinned_output_ptr, 0, NULL, NULL);
            compressor->pinned_output_ptr = NULL;
            compressor->pinned_output_size = 0;
        }
        clReleaseMemObject(compressor->output_buffer);
    }
    if (compressor->block_offsets_buffer) clReleaseMemObject(compressor->block_offsets_buffer);
    if (compressor->compressed_sizes_buffer) clReleaseMemObject(compressor->compressed_sizes_buffer);
    if (compressor->output_offsets_buffer) clReleaseMemObject(compressor->output_offsets_buffer);
    if (compressor->max_output_sizes_buffer) clReleaseMemObject(compressor->max_output_sizes_buffer);
        /* persistent device buffers are reused and freed at compressor destruction */

    if (compressor->program) clReleaseProgram(compressor->program);
    if (compressor->queue) clReleaseCommandQueue(compressor->queue);
    if (compressor->context) clReleaseContext(compressor->context);

    free(compressor);
}

static void lz4_gpu_set_error(LZ4GPUCompressor* compressor, LZ4GPUErrorCode code, const char* message) {
    compressor->last_error = code;
    strncpy(compressor->error_message, message, sizeof(compressor->error_message) - 1);
    compressor->error_message[sizeof(compressor->error_message) - 1] = '\0';
}

LZ4GPUErrorCode lz4_gpu_get_last_error(LZ4GPUCompressor* compressor) {
    return compressor->last_error;
}

const char* lz4_gpu_get_error_message(LZ4GPUCompressor* compressor) {
    if (!compressor) return "(null)";
    if (compressor->error_message[0] != '\0') return compressor->error_message;
    LZ4GPUErrorCode code = compressor->last_error;
    size_t n = sizeof(LZ4_GPU_ERRORS)/sizeof(LZ4_GPU_ERRORS[0]);
    if ((int)code >= 0 && (size_t)code < n) return LZ4_GPU_ERRORS[code];
    return "Unknown LZ4 GPU error";
}

/* Helper to ensure persistent input buffer is large enough.
 * Returns 1 on success, 0 on failure. Buffer is only reallocated if needed.
 * Uses a growth factor to reduce frequency of reallocations. */
static int ensure_input_buffer(LZ4GPUCompressor* compressor, size_t needed_size) {
    if (compressor->input_buffer && compressor->input_buffer_capacity >= needed_size) {
        return 1; /* Already large enough */
    }
        /* Release old buffer if exists (and ensure we unmap pinned pointers first) */
    if (compressor->input_buffer) {
        if (compressor->pinned_input_ptr) {
            clEnqueueUnmapMemObject(compressor->queue, compressor->input_buffer, compressor->pinned_input_ptr, 0, NULL, NULL);
            compressor->pinned_input_ptr = NULL;
            compressor->pinned_input_size = 0;
        }
        clReleaseMemObject(compressor->input_buffer);
        compressor->input_buffer = NULL;
        compressor->input_buffer_capacity = 0;
    }
    /* Allocate with 20% headroom to reduce future reallocations.
     * When I/O overlap is enabled, allocate double space for ping-pong slots. */
    size_t alloc_size = needed_size + needed_size / 5;
    if (compressor->enable_io_overlap) alloc_size *= 2;
    cl_int err;
    cl_mem_flags in_flags = CL_MEM_READ_ONLY;
    if (compressor->use_pinned_memory) in_flags |= CL_MEM_ALLOC_HOST_PTR;
    compressor->input_buffer = clCreateBuffer(compressor->context, in_flags,
                                              alloc_size, NULL, &err);
    if (err != CL_SUCCESS) {
        HDEBUG("DEBUG: Failed to allocate input buffer of %zu bytes (err=%d)\n", alloc_size, err);
        return 0;
    }
    compressor->input_buffer_capacity = alloc_size;
    /* If pinned memory enabled, (re)map host pointer for easier pinned uploads */
    if (compressor->use_pinned_memory) {
        /* Unmap previous mapping if any */
        if (compressor->pinned_input_ptr) {
            /* We map the buffer in blocking mode; unmap to clear */
            clEnqueueUnmapMemObject(compressor->queue, compressor->input_buffer, compressor->pinned_input_ptr, 0, NULL, NULL);
            compressor->pinned_input_ptr = NULL;
            compressor->pinned_input_size = 0;
        }
        /* Map the freshly created input buffer for host access */
        cl_int merr = CL_SUCCESS;
        void* p = clEnqueueMapBuffer(compressor->queue, compressor->input_buffer, CL_TRUE, CL_MAP_WRITE, 0, alloc_size, 0, NULL, NULL, &merr);
        if (merr == CL_SUCCESS && p) {
            compressor->pinned_input_ptr = p;
            compressor->pinned_input_size = alloc_size;
        } else {
            compressor->pinned_input_ptr = NULL;
            compressor->pinned_input_size = 0;
        }
    }
    HDEBUG("DEBUG: Allocated input buffer: %zu bytes (requested %zu)\n", alloc_size, needed_size);
    return 1;
}

/* Helper to ensure persistent output buffer is large enough.
 * Returns 1 on success, 0 on failure. Buffer is only reallocated if needed. */
static int ensure_output_buffer(LZ4GPUCompressor* compressor, size_t needed_size) {
    if (compressor->output_buffer && compressor->output_buffer_capacity >= needed_size) {
        return 1; /* Already large enough */
    }
    /* Release old buffer if exists (and unmap pinned pointers first) */
    if (compressor->output_buffer) {
        if (compressor->pinned_output_ptr) {
            clEnqueueUnmapMemObject(compressor->queue, compressor->output_buffer, compressor->pinned_output_ptr, 0, NULL, NULL);
            compressor->pinned_output_ptr = NULL;
            compressor->pinned_output_size = 0;
        }
        clReleaseMemObject(compressor->output_buffer);
        compressor->output_buffer = NULL;
        compressor->output_buffer_capacity = 0;
    }
    /* Allocate with 20% headroom to reduce future reallocations.
     * When I/O overlap is enabled, allocate double space for ping-pong slots. */
    size_t alloc_size = needed_size + needed_size / 5;
    if (compressor->enable_io_overlap) alloc_size *= 2;
    cl_int err;
    cl_mem_flags out_flags = CL_MEM_READ_WRITE;
    if (compressor->use_pinned_memory) out_flags |= CL_MEM_ALLOC_HOST_PTR;
    compressor->output_buffer = clCreateBuffer(compressor->context, out_flags,
                                               alloc_size, NULL, &err);
    if (err != CL_SUCCESS) {
        HDEBUG("DEBUG: Failed to allocate output buffer of %zu bytes (err=%d)\n", alloc_size, err);
        return 0;
    }
    compressor->output_buffer_capacity = alloc_size;
    /* If pinned memory enabled, (re)map host pointer for easier pinned downloads */
    if (compressor->use_pinned_memory) {
        if (compressor->pinned_output_ptr) {
            clEnqueueUnmapMemObject(compressor->queue, compressor->output_buffer, compressor->pinned_output_ptr, 0, NULL, NULL);
            compressor->pinned_output_ptr = NULL;
            compressor->pinned_output_size = 0;
        }
        cl_int merr = CL_SUCCESS;
        void* p = clEnqueueMapBuffer(compressor->queue, compressor->output_buffer, CL_TRUE, CL_MAP_READ, 0, alloc_size, 0, NULL, NULL, &merr);
        if (merr == CL_SUCCESS && p) {
            compressor->pinned_output_ptr = p;
            compressor->pinned_output_size = alloc_size;
        } else {
            compressor->pinned_output_ptr = NULL;
            compressor->pinned_output_size = 0;
        }
    }
    HDEBUG("DEBUG: Allocated output buffer: %zu bytes (requested %zu)\n", alloc_size, needed_size);
    return 1;
}

static char* load_kernel_source(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(*size + 1);
    if (!source) {
        fclose(file);
        return NULL;
    }

    fread(source, 1, *size, file);
    source[*size] = '\0';
    fclose(file);

    return source;
}

/* Build OpenCL program with current compressor build options, and create kernels. */
static int lz4_gpu_build_program_with_options(LZ4GPUCompressor* compressor) {
    cl_int err;
    /* Timing variables removed (not used in build step) */
    /* Profiling event placeholders used when compressor->enable_profiling == 1 */
    cl_event upload_ev = NULL;
    cl_event offsets_ev = NULL;
    cl_event outoffs_ev = NULL;
    cl_event kernel_ev = NULL; // top-level kernel event for profiling
    cl_event maxouts_ev = NULL;

    // 1) Optional: try precompiled binary if configured or if a default clbin exists
    //    Prefer precompiled binary when available (even if prefer_precompiled is not explicitly set).
    //    Order of attempts: compressor->precompiled_path (if set) -> default './lz4_gpu.clbin'
    {
        /* Candidate resolution order for precompiled binary (prefer):
         * 1) explicit compressor->precompiled_path when set
         * 2) environment variable LZ4_GPU_CLBIN
         * 3) default binary `lz4_gpu.clbin` in current working directory
         * 4) default binary `lz4_gpu.clbin` in executable directory (/proc/self/exe)
         */
        const char* candidate_paths[4];
        int candidate_count = 0;
        if (compressor->precompiled_path[0]) candidate_paths[candidate_count++] = compressor->precompiled_path;
        const char* env_clbin = getenv("LZ4_GPU_CLBIN");
        if (env_clbin && env_clbin[0]) candidate_paths[candidate_count++] = env_clbin;
        candidate_paths[candidate_count++] = "lz4_gpu.clbin"; /* current working dir */
        /* try to find exec dir lz4_gpu.clbin
           resolve /proc/self/exe -> /path/to/exec, strip filename and append lz4_gpu.clbin */
        char execdir[4096] = {0};
        char exe_path[4096] = {0};
        ssize_t rl = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
        if (rl > 0) {
            exe_path[rl] = '\0';
            char* last_sep = strrchr(exe_path, '/');
            if (last_sep) {
                size_t dir_len = (size_t)(last_sep - exe_path);
                if (dir_len + 1 + 13 < sizeof(execdir)) {
                    memcpy(execdir, exe_path, dir_len);
                    execdir[dir_len] = '/';
                    memcpy(execdir + dir_len + 1, "lz4_gpu.clbin", 13);
                    execdir[dir_len + 1 + 13] = '\0';
                    candidate_paths[candidate_count++] = execdir;
                }
            }
        }
        const char* path = NULL;
        int try_precompiled = 0;
        for (int ci = 0; ci < candidate_count; ++ci) {
            if (!candidate_paths[ci]) continue;
            FILE* ftest = fopen(candidate_paths[ci], "rb");
            if (ftest) { fclose(ftest); path = candidate_paths[ci]; try_precompiled = 1; break; }
        }
        if (try_precompiled && path) {
            HDEBUG("DEBUG: Candidate precompiled path chosen: %s\n", path);
            // Heuristic check: ensure filename looks valid; skip if it seems wrong
            int skip_binary = 0;
            const char* fname = strrchr(path, '\\');
            if (!fname) fname = strrchr(path, '/');
            if (!fname) fname = path; else fname++;
            // If user passes a precompiled binary, accept it unless filename looks unrelated
            if (!strstr(fname, "lz4_gpu") && !strstr(fname, ".clbin")) {
                HDEBUG("DEBUG: Precompiled binary '%s' does not look like expected clbin; skipping to source build\n", fname);
                skip_binary = 1;
            }

            if (!skip_binary) {
                FILE* bf = fopen(path, "rb");
                if (bf) {
                    fseek(bf, 0, SEEK_END);
                    size_t bsz = (size_t)ftell(bf);
                    fseek(bf, 0, SEEK_SET);
                    unsigned char* bdata = (unsigned char*)malloc(bsz);
                    if (bdata && fread(bdata, 1, bsz, bf) == bsz) {
                        const unsigned char* binaries[1] = { bdata };
                        size_t lengths[1] = { bsz };
                        if (compressor->program) { clReleaseProgram(compressor->program); compressor->program = NULL; }
                        HDEBUG("DEBUG: Loading precompiled OpenCL binary '%s' (%zu bytes)\n", path, bsz);
                        compressor->program = clCreateProgramWithBinary(
                            compressor->context, 1, &compressor->device,
                            lengths, binaries, NULL, &err);
                            if (err == CL_SUCCESS) {
                                err = clBuildProgram(compressor->program, 1, &compressor->device, NULL, NULL, NULL);
                                if (err == CL_SUCCESS) {
                                    // Try creating kernels to validate this binary
                                    cl_int kerr;
                                    cl_kernel ktest = clCreateKernel(compressor->program, "lz4_decompress_block", &kerr);
                                    if (kerr == CL_SUCCESS) clReleaseKernel(ktest);
                                    if (kerr == CL_SUCCESS) {
                                        HDEBUG("DEBUG: Precompiled binary validated successfully: %s\n", path);
                                        if (compressor->verbose || compressor->enable_kernel_debug) {
                                            fprintf(stderr, "NOTE: Using precompiled OpenCL binary '%s'\n", path);
                                        }
                                        free(bdata);
                                        // Proceed to create real kernels later below
                                        goto create_kernels;
                                    } else {
                                        if (compressor->verbose || compressor->enable_kernel_debug) {
                                            fprintf(stderr, "NOTE: Precompiled binary '%s' missing required kernels; falling back to source build\n", path);
                                        }
                                        HDEBUG("DEBUG: Precompiled binary missing required kernels; falling back to source build\n");
                                        clReleaseProgram(compressor->program); compressor->program = NULL; err = CL_INVALID_PROGRAM;
                                    }
                                } else {
                                    if (compressor->verbose || compressor->enable_kernel_debug) {
                                        fprintf(stderr, "NOTE: clBuildProgram on precompiled binary '%s' failed (%d); will fall back to source build\n", path, err);
                                    }
                                    HDEBUG("DEBUG: clBuildProgram on binary failed (%d); will fall back to source\n", err);
                                    clReleaseProgram(compressor->program); compressor->program = NULL;
                                }
                        } else {
                            HDEBUG("DEBUG: clCreateProgramWithBinary failed (%d); will fall back to source\n", err);
                        }
                    }
                    if (bdata) free(bdata);
                    fclose(bf);
                } else {
                    if (compressor->verbose || compressor->enable_kernel_debug) {
                        fprintf(stderr, "NOTE: Could not open precompiled binary at '%s' - will try building from source\n", path);
                    }
                    HDEBUG("DEBUG: Could not open precompiled binary at '%s'\n", path);
                }
            }
            }
        }

    // 2) Load kernel source and build with options
    size_t kernel_size = 0;
    const char* kernel_src_file = compressor->kernel_src_path[0] ? compressor->kernel_src_path : "lz4_gpu.cl";
    char* kernel_source = load_kernel_source(kernel_src_file, &kernel_size);
    if (!kernel_source) {
        lz4_gpu_set_error(compressor, LZ4_GPU_PROGRAM_ERROR, "Failed to load kernel source");
        return 0;
    }

    if (compressor->program) {
        clReleaseProgram(compressor->program);
        compressor->program = NULL;
    }

    compressor->program = clCreateProgramWithSource(compressor->context, 1,
                                                   (const char**)&kernel_source, &kernel_size, &err);
    free(kernel_source);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_PROGRAM_ERROR, "Failed to create program");
        return 0;
    }

    // Compose build options string
    char build_opts[256];
    build_opts[0] = '\0';
    /* vector mode removed: do not pass LZ4_GPU_VECTOR_IO define */
    if (compressor->enable_kernel_debug) {
        strcat(build_opts, " -DLZ4_GPU_KERNEL_DEBUG=1");
    }

    HDEBUG("DEBUG: Building OpenCL program with options:%s\n", build_opts[0] ? build_opts : " <none>");
    err = clBuildProgram(compressor->program, 1, &compressor->device,
                         build_opts[0] ? build_opts : NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Retrieve and report build log
        size_t log_size = 0;
        clGetProgramBuildInfo(compressor->program, compressor->device,
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = (char*)malloc(log_size + 1);
        if (build_log) {
            clGetProgramBuildInfo(compressor->program, compressor->device,
                                  CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
            build_log[log_size] = '\0';
            printf("DEBUG: Build failed with error %d\n", err);
            printf("DEBUG: Build log (%zu bytes):\n%s\n", log_size, build_log);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUILD_ERROR, build_log);
            free(build_log);
        } else {
            lz4_gpu_set_error(compressor, LZ4_GPU_BUILD_ERROR, "Failed to build program (no log)\n");
        }
        return 0;
    }
    HDEBUG("DEBUG: Program built successfully\n");
    if (compressor->verbose || compressor->enable_kernel_debug) {
        fprintf(stderr, "NOTE: Built OpenCL program from source successfully\n");
    }

    // After a successful source build, export the OpenCL program binary once for reuse
    // Filename convention: include "vec" when vector IO is enabled
    {
        cl_uint numDevices = 0;
        if (clGetProgramInfo(compressor->program, CL_PROGRAM_NUM_DEVICES, sizeof(numDevices), &numDevices, NULL) == CL_SUCCESS && numDevices > 0) {
            size_t* binSizes = (size_t*)malloc(sizeof(size_t) * numDevices);
            if (binSizes && clGetProgramInfo(compressor->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * numDevices, binSizes, NULL) == CL_SUCCESS) {
                unsigned char** binaries = (unsigned char**)malloc(sizeof(unsigned char*) * numDevices);
                if (binaries) {
                    for (cl_uint i = 0; i < numDevices; ++i) {
                        binaries[i] = (unsigned char*)malloc(binSizes[i]);
                    }
                    if (clGetProgramInfo(compressor->program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices, binaries, NULL) == CL_SUCCESS) {
                        const char* fname = "lz4_gpu.clbin";
                        FILE* fbin = fopen(fname, "wb");
                        if (fbin) {
                            size_t written = fwrite(binaries[0], 1, binSizes[0], fbin);
                            fclose(fbin);
                            HDEBUG("DEBUG: Exported OpenCL binary to '%s' (%zu bytes written)\n", fname, written);
                            if (compressor->verbose || compressor->enable_kernel_debug) {
                                fprintf(stderr, "NOTE: Exported OpenCL program binary to '%s' (%zu bytes)\n", fname, written);
                            }
                            (void)written;
                        } else {
                            HDEBUG("DEBUG: Warning: failed to open '%s' for writing program binary\n", fname);
                        }
                    }
                    for (cl_uint i = 0; i < numDevices; ++i) {
                        free(binaries[i]);
                    }
                    free(binaries);
                }
            }
            if (binSizes) free(binSizes);
        }
    }

create_kernels:
    // (Re)create frequently used kernels
    if (compressor->compress_kernel) { clReleaseKernel(compressor->compress_kernel); compressor->compress_kernel = NULL; }
    if (compressor->decompress_kernel) { clReleaseKernel(compressor->decompress_kernel); compressor->decompress_kernel = NULL; }

    cl_int kerr;
    compressor->compress_kernel = clCreateKernel(compressor->program, "lz4_compress_block_accelerated", &kerr);
    if (kerr != CL_SUCCESS) {
        // Fallback to non-accelerated name if not found
        compressor->compress_kernel = clCreateKernel(compressor->program, "lz4_compress_block", &kerr);
        if (kerr != CL_SUCCESS) {
            lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create compression kernel");
            return 0;
        }
    }

    compressor->decompress_kernel = clCreateKernel(compressor->program, "lz4_decompress_block", &kerr);
    if (kerr != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create block decompression kernel");
        return 0;
    }

    /* Silence unused variable warnings when profiling is disabled */
    (void)upload_ev; (void)offsets_ev; (void)outoffs_ev; (void)maxouts_ev; (void)kernel_ev;
    return 1;
}

int lz4_gpu_initialize(LZ4GPUCompressor* compressor) {
    cl_int err;
    /* Measure initialization time if desired */
    clock_t t_init_start = 0, t_init_end = 0;
    t_init_start = clock();

    // Get platform
    err = clGetPlatformIDs(1, &compressor->platform, NULL);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_PLATFORM_ERROR, "Failed to get OpenCL platform");
        return 0;
    }

    // Get GPU device (fallback to CPU if no GPU)
    err = clGetDeviceIDs(compressor->platform, CL_DEVICE_TYPE_GPU, 1, &compressor->device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(compressor->platform, CL_DEVICE_TYPE_CPU, 1, &compressor->device, NULL);
        if (err != CL_SUCCESS) {
            lz4_gpu_set_error(compressor, LZ4_GPU_DEVICE_ERROR, "Failed to get OpenCL device");
            return 0;
        }
    }

    // Query device capabilities
    compressor->device_compute_units = 0;
    compressor->device_max_work_group_size = 0;
    clGetDeviceInfo(compressor->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compressor->device_compute_units, NULL);
    clGetDeviceInfo(compressor->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &compressor->device_max_work_group_size, NULL);
    // Query device local memory size
    cl_ulong device_local_mem = 0;
    clGetDeviceInfo(compressor->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device_local_mem, NULL);
    // Set a sane default dynamic block size (can be adjusted per-input).
    // Tuned default for our 5-metric composite ranking: 16KB
    compressor->dynamic_block_size = LZ4_GPU_DEFAULT_BLOCK_SIZE; /* 16KB default */
    HDEBUG("DEBUG: Device compute units=%u, max_work_group_size=%zu, default_block_size=%zu\n",
           (unsigned int)compressor->device_compute_units, compressor->device_max_work_group_size, compressor->dynamic_block_size);

    /* Default local work-group sizes. These are chosen conservatively based on device
     * capabilities and the local memory requirements of the accelerated kernels.
     * The compress kernel uses a per-work-group local hash table sized by LZ4_HASHLOG.
     * If the device doesn't have enough local mem, keep defaults at 1. */
    size_t default_comp_local = 1;
    size_t default_decomp_local = 1;
    size_t local_table_bytes = ((size_t)1 << LZ4_GPU_HASH_LOG) * sizeof(cl_uint);
    if (device_local_mem >= (cl_ulong)local_table_bytes && compressor->device_max_work_group_size >= 16) {
        /* If the device can support our preferred default local size, pick it; otherwise
           fallback to heuristic based on device max work_group size. */
        if (compressor->device_max_work_group_size >= LZ4_GPU_DEFAULT_LOCAL_SIZE) {
            default_comp_local = LZ4_GPU_DEFAULT_LOCAL_SIZE;
        } else if (compressor->device_max_work_group_size >= 256) default_comp_local = 256;
        else if (compressor->device_max_work_group_size >= 128) default_comp_local = 128;
        else if (compressor->device_max_work_group_size >= 64) default_comp_local = 64;
        else default_comp_local = 1; /* conservative fallback */
    } else {
        default_comp_local = 1; /* fallback if local memory insufficient */
    }
    default_decomp_local = 1; /* tuning shows decompress kernel-only is best with local=1 */

    /* Respect a global API clamp for local sizes (avoid vendor-specific huge locals) */
    if (default_comp_local > LZ4_GPU_MAX_LOCAL_SIZE) default_comp_local = LZ4_GPU_MAX_LOCAL_SIZE;
    compressor->default_local = default_comp_local; /* unified default local size */

    // Create context
    compressor->context = clCreateContext(NULL, 1, &compressor->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_CONTEXT_ERROR, "Failed to create OpenCL context");
        return 0;
    }

    // Create command queue (enable profiling if requested)
    cl_command_queue_properties qprops = 0;
    if (compressor->enable_profiling) qprops |= CL_QUEUE_PROFILING_ENABLE;
#if defined(CL_VERSION_2_0)
    cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, (cl_queue_properties)qprops, 0 };
    compressor->queue = clCreateCommandQueueWithProperties(compressor->context, compressor->device, queue_props, &err);
#else
    compressor->queue = clCreateCommandQueue(compressor->context, compressor->device, qprops, &err);
#endif
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_QUEUE_ERROR, "Failed to create command queue");
        return 0;
    }

    // Load and build program with current build options
    if (!lz4_gpu_build_program_with_options(compressor)) {
        return 0;
    }
        /* persistent device buffers are reused and freed at compressor destruction */

    // Now create the actual compression kernel - use the accelerated version
    HDEBUG("DEBUG: Creating accelerated compression kernel...\n");
    compressor->compress_kernel = clCreateKernel(compressor->program, "lz4_compress_block_accelerated", &err);
    if (err != CL_SUCCESS) {
        printf("DEBUG: Failed to create accelerated compression kernel, error %d\n", err);
        // Try fallback to original kernel
        printf("DEBUG: Trying fallback to original kernel...\n");
        compressor->compress_kernel = clCreateKernel(compressor->program, "lz4_compress_block", &err);
        if (err != CL_SUCCESS) {
            printf("DEBUG: Failed to create compression kernel, error %d\n", err);
            lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create compression kernel");
            return 0;
        }
    }

    // Get kernel info for debugging
        /* Done with initialization - record init time */
        t_init_end = clock();
        compressor->last_timing.init_ms = (double)(t_init_end - t_init_start) / CLOCKS_PER_SEC * 1000.0;
    cl_uint num_args;
    err = clGetKernelInfo(compressor->compress_kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &num_args, NULL);
    // if (err == CL_SUCCESS) {
    //     printf("DEBUG: Compression kernel has %u arguments\n", num_args);
    // }

    HDEBUG("DEBUG: Creating decompression kernel...\n");
    compressor->decompress_kernel = clCreateKernel(compressor->program, "lz4_decompress_block", &err);
    if (err != CL_SUCCESS) {
        printf("DEBUG: Failed to create block decompression kernel, error %d\n", err);
        lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create block decompression kernel");
        return 0;
    }

    // Create initial buffers with reasonable default sizes
    // These will be resized as needed during compression
    compressor->input_buffer_size = 64 * 1024 * 1024; // 64MB default
    compressor->output_buffer_size = 128 * 1024 * 1024; // 128MB default

    cl_mem_flags init_in_flags = CL_MEM_READ_ONLY;
    if (compressor->use_pinned_memory) init_in_flags |= CL_MEM_ALLOC_HOST_PTR;
    compressor->input_buffer = clCreateBuffer(compressor->context, init_in_flags,
                                             compressor->input_buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create input buffer");
        return 0;
    }

    cl_mem_flags init_out_flags = CL_MEM_READ_WRITE;
    if (compressor->use_pinned_memory) init_out_flags |= CL_MEM_ALLOC_HOST_PTR;
    compressor->output_buffer = clCreateBuffer(compressor->context, init_out_flags,
                                              compressor->output_buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output buffer");
        return 0;
    }

    // Note: We don't create block_offsets_buffer and block_sizes_buffer here anymore
    // They are created dynamically in compression functions as needed and reused.
    compressor->block_offsets_buffer = NULL;
    compressor->block_offsets_capacity = 0;
    compressor->compressed_sizes_buffer = NULL;
    compressor->compressed_sizes_capacity = 0;
    compressor->output_offsets_buffer = NULL;
    compressor->output_offsets_capacity = 0;
    compressor->max_output_sizes_buffer = NULL;
    compressor->max_output_sizes_capacity = 0;
    /* Print pinned memory usage state for clarity when profiling/verbose */
    if (compressor->enable_profiling || compressor->enable_kernel_debug) {
        if (compressor->use_pinned_memory) fprintf(stderr, "NOTE: Pinned host memory will be used if supported\n");
        else fprintf(stderr, "NOTE: Pinned host memory disabled; using standard host memory\n");
    }
    return 1;
}

void lz4_gpu_set_kernel_debug(LZ4GPUCompressor* compressor, int enabled) {
    if (!compressor) return;
    compressor->enable_kernel_debug = enabled ? 1 : 0;
}

int lz4_gpu_rebuild_program(LZ4GPUCompressor* compressor) {
    if (!compressor || !compressor->context || !compressor->device) return 0;
    return lz4_gpu_build_program_with_options(compressor);
}

void lz4_gpu_use_precompiled(LZ4GPUCompressor* compressor, int enabled) {
    if (!compressor) return;
    compressor->prefer_precompiled = enabled ? 1 : 0;
}

void lz4_gpu_set_precompiled_binary(LZ4GPUCompressor* compressor, const char* path) {
    if (!compressor) return;
    if (!path) { compressor->precompiled_path[0] = '\0'; return; }
    strncpy(compressor->precompiled_path, path, sizeof(compressor->precompiled_path)-1);
    compressor->precompiled_path[sizeof(compressor->precompiled_path)-1] = '\0';
}

void lz4_gpu_set_kernel_source(LZ4GPUCompressor* compressor, const char* path) {
    if (!compressor) return;
    if (!path) { compressor->kernel_src_path[0] = '\0'; return; }
    strncpy(compressor->kernel_src_path, path, sizeof(compressor->kernel_src_path)-1);
    compressor->kernel_src_path[sizeof(compressor->kernel_src_path)-1] = '\0';
}

void lz4_gpu_set_workgroup_size(LZ4GPUCompressor* compressor, size_t local) {
    if (!compressor) return;
    /* Clamp API-provided workgroup size to a sane maximum to avoid vendor-specific surprises */
    if (local > LZ4_GPU_MAX_LOCAL_SIZE) local = LZ4_GPU_MAX_LOCAL_SIZE;
    if (compressor->device_max_work_group_size > 0 && local > compressor->device_max_work_group_size) {
        local = compressor->device_max_work_group_size;
    }
    if (local > 0) compressor->default_local = local;
}

void lz4_gpu_set_block_sizes(LZ4GPUCompressor* compressor, size_t compress_block_size, size_t decompress_block_size) {
    if (!compressor) return;
    /* Clamp and align block sizes for safety */
    const size_t ALIGN = 4 * 1024;
    const size_t MIN_BLOCK = 16 * 1024;
    if (compress_block_size > 0) {
        size_t bs = compress_block_size;
        if (bs < MIN_BLOCK) bs = MIN_BLOCK;
        if (bs > LZ4_GPU_MAX_BLOCK_SIZE) bs = LZ4_GPU_MAX_BLOCK_SIZE;
        bs = ((bs + ALIGN - 1) / ALIGN) * ALIGN;
        compressor->compress_block_size = bs;
    }
    if (decompress_block_size > 0) {
        size_t bs2 = decompress_block_size;
        if (bs2 < MIN_BLOCK) bs2 = MIN_BLOCK;
        if (bs2 > LZ4_GPU_MAX_BLOCK_SIZE) bs2 = LZ4_GPU_MAX_BLOCK_SIZE;
        bs2 = ((bs2 + ALIGN - 1) / ALIGN) * ALIGN;
        compressor->decompress_block_size = bs2;
    }
}

void lz4_gpu_set_pinned_memory(LZ4GPUCompressor* compressor, int enabled) {
    if (!compressor) return;
    compressor->use_pinned_memory = enabled ? 1 : 0;
}

void lz4_gpu_set_io_overlap(LZ4GPUCompressor* compressor, int enabled) {
    if (!compressor) return;
    compressor->enable_io_overlap = enabled ? 1 : 0;
}


static void split_into_blocks(LZ4GPUCompressor* compressor, const void* input, size_t input_size,
                              cl_uint* block_offsets, size_t* num_blocks) {
    // Use dynamic block size if available
    (void)input; /* unused by split_into_blocks - placeholder for API parity */
    size_t offset = 0;
    size_t block_count = 0;
    size_t block_size = compressor ? compressor->dynamic_block_size : LZ4_GPU_MAX_BLOCK_SIZE;

    while (offset < input_size) {
        size_t current_block_size = (input_size - offset < block_size) ?
                                  (input_size - offset) : block_size;

        block_offsets[block_count * 2] = (cl_uint)offset;
        block_offsets[block_count * 2 + 1] = (cl_uint)current_block_size;

        offset += current_block_size;
        block_count++;
    }

    *num_blocks = block_count;
}

// Compute dynamic block size based on input size and device capabilities.
// Strategy: Default to 16KB (optimal per benchmarks). Only increase block size
// if it creates too few blocks for effective GPU parallelism.
static size_t compute_dynamic_block_size(LZ4GPUCompressor* compressor, size_t input_size) {
    const size_t DEFAULT_BLOCK = LZ4_GPU_DEFAULT_BLOCK_SIZE; // 16KB - optimal per 5-metric ranking
    const size_t MAX_BLOCK = LZ4_GPU_MAX_BLOCK_SIZE; // 512KB max

    // For small inputs, use default
    if (input_size <= DEFAULT_BLOCK * 4) {
        return DEFAULT_BLOCK;
    }

    // Calculate how many blocks 16KB would create
    size_t blocks_at_default = (input_size + DEFAULT_BLOCK - 1) / DEFAULT_BLOCK;

    // Only increase block size if we have very few blocks (less than 16)
    // This ensures good parallelism: even 1MB creates 64 blocks at 16KB
    const size_t MIN_BLOCKS = 16;
    if (blocks_at_default >= MIN_BLOCKS) {
        return DEFAULT_BLOCK;
    }

    // For very small inputs creating <16 blocks, compute larger block size
    size_t block_size = (input_size + MIN_BLOCKS - 1) / MIN_BLOCKS;

    // Round up to 4KB alignment
    const size_t ALIGN = 4 * 1024;
    block_size = ((block_size + ALIGN - 1) / ALIGN) * ALIGN;

    // Clamp to valid range
    if (block_size < DEFAULT_BLOCK) block_size = DEFAULT_BLOCK;
    if (block_size > MAX_BLOCK) block_size = MAX_BLOCK;

    return block_size;
}

/* Map an uncompressed block size to LZ4 frame Block Size ID (4=64KB,5=256KB,6=1MB,7=4MB) */
static unsigned char lz4f_blockSizeID_from_uncompressed_size(size_t size) {
    if (size <= 64 * 1024) return 4;
    if (size <= 256 * 1024) return 5;
    if (size <= 1024 * 1024) return 6;
    /* cap to 4MB per spec */
    return 7;
}

// Query device capabilities for an initialized compressor.
// Fills output pointers when non-NULL. Returns 1 on success, 0 on failure.
int lz4_gpu_query_device_capabilities(LZ4GPUCompressor* compressor, cl_uint* out_compute_units, size_t* out_max_work_group_size) {
    if (!compressor) return 0;
    if (compressor->device == 0) return 0;

    if (compressor->device_compute_units == 0 || compressor->device_max_work_group_size == 0) {
        cl_uint cu = 0;
        size_t wg = 0;
        clGetDeviceInfo(compressor->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cu, NULL);
        clGetDeviceInfo(compressor->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &wg, NULL);
        compressor->device_compute_units = cu;
        compressor->device_max_work_group_size = wg;
    }

    if (out_compute_units) *out_compute_units = compressor->device_compute_units;
    if (out_max_work_group_size) *out_max_work_group_size = compressor->device_max_work_group_size;
    /* Debug output: print the queried device capabilities for troubleshooting */
    HDEBUG("DEBUG: lz4_gpu_query_device_capabilities -> compute_units=%u, max_work_group_size=%zu, dynamic_block_size=%zu\n",
           (unsigned int)compressor->device_compute_units, compressor->device_max_work_group_size, compressor->dynamic_block_size);
    return 1;
}

size_t lz4_gpu_compress_frame(LZ4GPUCompressor* compressor, const void* input, size_t input_size, void* output, size_t output_capacity) {
    int accel = LZ4_GPU_DEFAULT_ACCELERATION;
    if (compressor && compressor->default_acceleration > 0) accel = compressor->default_acceleration;
    return lz4_gpu_compress_frame_accelerated(compressor, input, input_size, output, output_capacity, accel);
}

size_t lz4_gpu_compress_frame_accelerated(LZ4GPUCompressor* compressor,
                                    const void* input, size_t input_size,
                                    void* output, size_t output_capacity,
                                    int acceleration) {
    HDEBUG("DEBUG: lz4_gpu_compress_frame_accelerated called with input_size=%zu, acceleration=%d\n",
        input_size, acceleration);

    /* Record wall-clock start time for total operation timing */
    clock_t tstart = clock();
    /* Local timing variables for this compression call */
    clock_t t_alloc_start = 0, t_alloc_end = 0;
    clock_t t_h2d_start = 0, t_h2d_end = 0;
    clock_t t_setup_start = 0, t_setup_end = 0;
    clock_t t_kernel_host_start = 0, t_kernel_host_end = 0;
    clock_t t_d2h_start = 0, t_d2h_end = 0;
    clock_t t_frame_start = 0, t_frame_end = 0;

    double alloc_ms = 0.0, h2d_ms = 0.0, setup_ms = 0.0, kernel_ms_host = -1.0, kernel_ms_dev = -1.0, d2h_ms = 0.0, frame_ms = 0.0, event_profile_ms = 0.0;

    if (!compressor || !input || !output || input_size == 0) {
        if (compressor) {
            lz4_gpu_set_error(compressor, LZ4_GPU_INVALID_PARAMS, "Invalid input parameters");
        }
        return 0;
    }

    if (!compressor->context) {
        lz4_gpu_set_error(compressor, LZ4_GPU_NOT_INITIALIZED, "Compressor not initialized");
        return 0;
    }

    // Validate acceleration parameter
    if (acceleration < 1) acceleration = 1;
    if (acceleration > LZ4_GPU_MAX_ACCELERATION) acceleration = LZ4_GPU_MAX_ACCELERATION;
    /* persistent device buffers are reused and freed at compressor destruction */

    HDEBUG("DEBUG: Using acceleration=%d (clamped to valid range)\n", acceleration);

    cl_int err;
    /* Event placeholders for asynchronous transfers used in this function */
    cl_event upload_ev = NULL;
    cl_event offsets_ev = NULL;
    cl_event outoffs_ev = NULL;
    /* per-block max_output_sizes will be persisted on compressor->max_output_sizes_buffer */
    cl_event maxouts_ev = NULL;
    cl_event kernel_ev = NULL; /* top-level kernel event for profiling */
    /* Event arrays used by overlapped per-chunk pipeline (declared here so they are visible in the whole function) */
    cl_event *upload_ev_slots = NULL;
    cl_event *kernel_ev_slots = NULL;
    cl_event *read_sizes_ev_slots = NULL;
    cl_event *read_blocks_ev_slots = NULL;

    // Determine dynamic block size (smaller when acceleration > 1) and split input into blocks

    size_t computed_block = compute_dynamic_block_size(compressor, input_size);
    /* If caller provided an explicit block size, honor it (clamp/align). */
    if (compressor->compress_block_size > 0) {
        size_t bs = compressor->compress_block_size;
        const size_t MIN_BLOCK = 16 * 1024;
        const size_t MAX_BLOCK = 4 * 1024 * 1024;
        const size_t ALIGN = 4 * 1024;
        if (bs < MIN_BLOCK) bs = MIN_BLOCK;
        if (bs > MAX_BLOCK) bs = MAX_BLOCK;
        /* align to 4KB */
        bs = ((bs + ALIGN - 1) / ALIGN) * ALIGN;
        compressor->dynamic_block_size = bs;
    } else {
        compressor->dynamic_block_size = computed_block;
    }

    HDEBUG("DEBUG: Using dynamic block size = %zu bytes for input_size=%zu (acceleration path)\n",
        compressor->dynamic_block_size, input_size);

    size_t max_blocks = (input_size + compressor->dynamic_block_size - 1) / compressor->dynamic_block_size + 1;
    cl_uint* block_offsets = (cl_uint*)malloc(max_blocks * 2 * sizeof(cl_uint));
    if (!block_offsets) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate block offsets buffer");
        return 0;
    }

    size_t num_blocks = 0;
    split_into_blocks(compressor, input, input_size, block_offsets, &num_blocks);

    HDEBUG("DEBUG: Split input into %zu blocks for parallel processing\n", num_blocks);

    // Ensure persistent buffers are large enough (reused across calls)
    /* Allocation phase: start timing */
    t_alloc_start = clock();

    /* Use persistent input buffer */
    if (!ensure_input_buffer(compressor, input_size)) {
        free(block_offsets);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to ensure input buffer");
        return 0;
    }

    /* NOTE: creation/upload of max_output_sizes is deferred until after
    * the host computes output_offsets and max_output_sizes.  The kernel
    * expects an array of per-block max output sizes — we don't have that
    * data yet here. This avoids referencing output_buffer or max_output_sizes
    * before they are allocated/populated. */

    // Ensure device buffer for block offsets exists and is large enough (reuse if possible)
    size_t needed_block_off_bytes = sizeof(cl_uint) * num_blocks * 2;
    if (!compressor->block_offsets_buffer || compressor->block_offsets_capacity < needed_block_off_bytes) {
        if (compressor->block_offsets_buffer) { clReleaseMemObject(compressor->block_offsets_buffer); compressor->block_offsets_buffer = NULL; compressor->block_offsets_capacity = 0; }
        compressor->block_offsets_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                                        needed_block_off_bytes, NULL, &err);
        if (err != CL_SUCCESS) {
            free(block_offsets);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create block offsets buffer");
            return 0;
        }
        compressor->block_offsets_capacity = needed_block_off_bytes;
    }

    // Ensure device buffer for output offsets exists and is large enough (reuse if possible)
    size_t needed_out_off_bytes = sizeof(cl_uint) * num_blocks;
    if (!compressor->output_offsets_buffer || compressor->output_offsets_capacity < needed_out_off_bytes) {
        if (compressor->output_offsets_buffer) { clReleaseMemObject(compressor->output_offsets_buffer); compressor->output_offsets_buffer = NULL; compressor->output_offsets_capacity = 0; }
        compressor->output_offsets_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                                        needed_out_off_bytes, NULL, &err);
        if (err != CL_SUCCESS) {
            free(block_offsets);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output offsets buffer");
            return 0;
        }
        compressor->output_offsets_capacity = needed_out_off_bytes;
    }

    // Upload input data (use pinned staging pointer if available)
    void* upload_src_ptr = (void*)input;
    if (compressor->use_pinned_memory && compressor->pinned_input_ptr && compressor->pinned_input_size >= input_size) {
        memcpy(compressor->pinned_input_ptr, input, input_size);
        upload_src_ptr = compressor->pinned_input_ptr;
    }
    if (compressor->enable_profiling) {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->input_buffer, CL_FALSE, 0,
                                input_size, upload_src_ptr, 0, NULL, &upload_ev);
    } else {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->input_buffer, CL_TRUE, 0,
                                input_size, upload_src_ptr, 0, NULL, NULL);
    }
    if (err != CL_SUCCESS) {
        if (upload_ev) { clReleaseEvent(upload_ev); }
        lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload input data");
        return 0;
    }

    // Calculate required GPU output buffer size
    size_t required_gpu_buffer_size = 0;
    for (size_t i = 0; i < num_blocks; i++) {
        size_t block_size = block_offsets[i * 2 + 1];
        required_gpu_buffer_size += block_size + (block_size / 255) + 64; // Conservative estimate
    }

    HDEBUG("DEBUG: Required GPU buffer size: %zu bytes, output_capacity: %zu bytes\n",
        required_gpu_buffer_size, output_capacity);

    if (required_gpu_buffer_size > output_capacity) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Output buffer too small for GPU processing");
        return 0;
    }

    /* Use persistent output buffer */
    if (!ensure_output_buffer(compressor, output_capacity)) {
        free(block_offsets);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to ensure output buffer");
        return 0;
    }

    // Upload block offsets
    if (compressor->enable_profiling) {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->block_offsets_buffer, CL_FALSE, 0,
                                sizeof(cl_uint) * num_blocks * 2, block_offsets, 0, NULL, &offsets_ev);
    } else {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->block_offsets_buffer, CL_TRUE, 0,
                                sizeof(cl_uint) * num_blocks * 2, block_offsets, 0, NULL, NULL);
    }
    if (err != CL_SUCCESS) {
        if (upload_ev) { clReleaseEvent(upload_ev); }
        if (offsets_ev) { clReleaseEvent(offsets_ev); }
        free(block_offsets);
        lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload block offsets");
        return 0;
    }

    // Ensure device buffer for compressed sizes exists and is large enough (reuse if possible)
    size_t needed_csizes_bytes = sizeof(cl_uint) * num_blocks;
    if (!compressor->compressed_sizes_buffer || compressor->compressed_sizes_capacity < needed_csizes_bytes) {
        if (compressor->compressed_sizes_buffer) { clReleaseMemObject(compressor->compressed_sizes_buffer); compressor->compressed_sizes_buffer = NULL; compressor->compressed_sizes_capacity = 0; }
        compressor->compressed_sizes_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE,
                                                            needed_csizes_bytes, NULL, &err);
        if (err != CL_SUCCESS) {
            free(block_offsets);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create compressed sizes buffer");
            return 0;
        }
        compressor->compressed_sizes_capacity = needed_csizes_bytes;
    }

    // Calculate output offsets for each block
    cl_uint* output_offsets = (cl_uint*)malloc(num_blocks * sizeof(cl_uint));
        /* Per-block maximum output size for accelerated kernel */
        cl_uint* max_output_sizes = (cl_uint*)malloc(num_blocks * sizeof(cl_uint));
    if (!output_offsets || !max_output_sizes) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate output offsets buffer");
        if (output_offsets) free(output_offsets);
        if (max_output_sizes) free(max_output_sizes);
        free(block_offsets);
        return 0;
    }

    size_t gpu_buffer_offset = 0;
    for (size_t i = 0; i < num_blocks; i++) {
        output_offsets[i] = (cl_uint)gpu_buffer_offset;
        size_t block_size = block_offsets[i * 2 + 1];
        size_t block_buffer_size = block_size + (block_size / 255) + 64;
        max_output_sizes[i] = (cl_uint)block_buffer_size;
        gpu_buffer_offset += block_buffer_size;

        if (i < 3) {
            HDEBUG("DEBUG: Block %zu: input_size=%zu, buffer_size=%zu, output_offset=%zu\n",
                i, block_size, block_buffer_size, output_offsets[i]);
        }
    }

    // Upload output offsets
    if (compressor->enable_profiling) {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->output_offsets_buffer, CL_FALSE, 0,
                                sizeof(cl_uint) * num_blocks, output_offsets, 0, NULL, &outoffs_ev);
    } else {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->output_offsets_buffer, CL_TRUE, 0,
                                sizeof(cl_uint) * num_blocks, output_offsets, 0, NULL, NULL);
    }
    if (err != CL_SUCCESS) {
        if (upload_ev) { clReleaseEvent(upload_ev); }
        if (offsets_ev) { clReleaseEvent(offsets_ev); }
        if (outoffs_ev) { clReleaseEvent(outoffs_ev); }
        lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload output offsets");
        return 0;
    }

    /* Ensure device buffer for per-block max output sizes exists and upload host array */
    size_t needed_max_out_bytes = sizeof(cl_uint) * num_blocks;
    if (!compressor->max_output_sizes_buffer || compressor->max_output_sizes_capacity < needed_max_out_bytes) {
        if (compressor->max_output_sizes_buffer) {
            clReleaseMemObject(compressor->max_output_sizes_buffer);
            compressor->max_output_sizes_buffer = NULL;
            compressor->max_output_sizes_capacity = 0;
        }
        compressor->max_output_sizes_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                                            needed_max_out_bytes, NULL, &err);
        if (err != CL_SUCCESS) {
            if (upload_ev) { clReleaseEvent(upload_ev); }
            if (offsets_ev) { clReleaseEvent(offsets_ev); }
            if (outoffs_ev) { clReleaseEvent(outoffs_ev); }
            free(output_offsets);
            free(max_output_sizes);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create max_output_sizes buffer");
            return 0;
        }
            compressor->max_output_sizes_capacity = needed_max_out_bytes;
        }
        /* Allocation phase: end timing */
        t_alloc_end = clock(); alloc_ms += (double)(t_alloc_end - t_alloc_start) / CLOCKS_PER_SEC * 1000.0;

    if (compressor->enable_profiling) {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->max_output_sizes_buffer, CL_FALSE, 0,
                                sizeof(cl_uint) * num_blocks, max_output_sizes, 0, NULL, &maxouts_ev);
    } else {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->max_output_sizes_buffer, CL_TRUE, 0,
                                sizeof(cl_uint) * num_blocks, max_output_sizes, 0, NULL, NULL);
    }
    if (err != CL_SUCCESS) {
        if (upload_ev) { clReleaseEvent(upload_ev); }
        if (offsets_ev) { clReleaseEvent(offsets_ev); }
        if (outoffs_ev) { clReleaseEvent(outoffs_ev); }
        if (maxouts_ev) { clReleaseEvent(maxouts_ev); }


        free(output_offsets);
        free(max_output_sizes);
        lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload max_output_sizes");
        return 0;
    }

    // Set kernel arguments for accelerated compression kernel
    int table_type = 1; // Use byU32 table type for better performance
    cl_uint num_blocks_u32 = (cl_uint)num_blocks;
    cl_uint input_size_u32 = (cl_uint)input_size;

    HDEBUG("DEBUG: Setting kernel args for accelerated compression: num_blocks=%u, input_size=%u, table_type=%d, acceleration=%d\n",
        num_blocks_u32, input_size_u32, table_type, acceleration);

    // Set kernel arguments for the accelerated kernel
    // Measure setup time for kernel args and enqueue
    t_setup_start = clock();
    // Kernel signature: lz4_compress_block_accelerated(input, output, blockSizes, blockOffsets, outputOffsets, totalBlocks, inputSize, tableType, acceleration)
    err = clSetKernelArg(compressor->compress_kernel, 0, sizeof(cl_mem), &compressor->input_buffer);              // input
    err |= clSetKernelArg(compressor->compress_kernel, 1, sizeof(cl_mem), &compressor->output_buffer);            // output
    err |= clSetKernelArg(compressor->compress_kernel, 2, sizeof(cl_mem), &compressor->compressed_sizes_buffer);  // blockSizes
    err |= clSetKernelArg(compressor->compress_kernel, 3, sizeof(cl_mem), &compressor->block_offsets_buffer);      // blockOffsets
    err |= clSetKernelArg(compressor->compress_kernel, 4, sizeof(cl_mem), &compressor->output_offsets_buffer);     // outputOffsets
    err |= clSetKernelArg(compressor->compress_kernel, 5, sizeof(cl_mem), &compressor->max_output_sizes_buffer);   // maxOutputSizes
    err |= clSetKernelArg(compressor->compress_kernel, 6, sizeof(cl_uint), &num_blocks_u32);           // totalBlocks
    err |= clSetKernelArg(compressor->compress_kernel, 7, sizeof(cl_uint), &input_size_u32);           // inputSize
    err |= clSetKernelArg(compressor->compress_kernel, 8, sizeof(int), &table_type);                  // tableType
    err |= clSetKernelArg(compressor->compress_kernel, 9, sizeof(int), &acceleration);               // acceleration parameter
        /* Allocate per-work-group local hash table */
        size_t local_table_bytes_accel = ((size_t)1 << LZ4_GPU_HASH_LOG) * sizeof(cl_uint);
        /* Clamp local table size to device local memory to avoid kernel arg failures */
        {
            cl_ulong device_local_mem = 0;
            if (compressor && compressor->device) {
                clGetDeviceInfo(compressor->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_local_mem), &device_local_mem, NULL);
            }
            if (device_local_mem > 0 && local_table_bytes_accel > (size_t)device_local_mem) {
                HDEBUG("DEBUG: Requested local table size %zu > device local mem %llu, limiting to device value\n", local_table_bytes_accel, (unsigned long long)device_local_mem);
                local_table_bytes_accel = (size_t)device_local_mem;
            }
        }
        err |= clSetKernelArg(compressor->compress_kernel, 10, local_table_bytes_accel, NULL);

    if (err != CL_SUCCESS) {
        printf("DEBUG: clSetKernelArg failed with error %d\n", err);
        if (upload_ev) clReleaseEvent(upload_ev);
        if (offsets_ev) clReleaseEvent(offsets_ev);
        if (outoffs_ev) clReleaseEvent(outoffs_ev);
        if (maxouts_ev) clReleaseEvent(maxouts_ev);
        free(output_offsets);
        free(max_output_sizes);
        lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set accelerated kernel arguments");
        return 0;
    }

    // Launch parallel compression kernel
    // Allow multi-thread work-groups per tuning defaults while preserving a
    // safe fallback to a single work-item per block. We round up the global
    // size to a multiple of the local work-group size to satisfy strict
    // OpenCL implementations.
    size_t global_work_size = num_blocks;
    size_t local_work_size = compressor->default_local;
    if (local_work_size == 0) local_work_size = 1;
    if (compressor->device_max_work_group_size > 0 && local_work_size > compressor->device_max_work_group_size) {
        local_work_size = compressor->device_max_work_group_size;
    }
    if (local_work_size > global_work_size) local_work_size = global_work_size;
    size_t global_work_size_rounded = global_work_size;
    if (local_work_size > 0 && (global_work_size % local_work_size) != 0) {
        global_work_size_rounded = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
    }
    HDEBUG("DEBUG: Launching accelerated compression kernel with global_size=%zu (rounded=%zu), local_size=%zu, acceleration=%d\n",
        global_work_size, global_work_size_rounded, local_work_size, acceleration);

    // Launch parallel compression kernel
    if (!compressor->enable_io_overlap) {
        cl_event kernel_ev = NULL;
        {
            cl_event wait_list_local[4];
            cl_uint num_waits = 0;
            if (compressor->enable_profiling) {
                if (upload_ev) wait_list_local[num_waits++] = upload_ev;
                if (offsets_ev) wait_list_local[num_waits++] = offsets_ev;
                if (outoffs_ev) wait_list_local[num_waits++] = outoffs_ev;
                if (maxouts_ev) wait_list_local[num_waits++] = maxouts_ev;
            }
            err = clEnqueueNDRangeKernel(compressor->queue, compressor->compress_kernel, 1, NULL,
                                &global_work_size_rounded, &local_work_size,
                                num_waits, (num_waits ? wait_list_local : NULL),
                                (compressor->enable_profiling ? &kernel_ev : NULL));
            t_setup_end = clock(); setup_ms += (double)(t_setup_end - t_setup_start) / CLOCKS_PER_SEC * 1000.0;
        }
    } else {
        // IO overlap enabled - actual overlapped pipeline runs later after per-frame buffers are ready
    }
    if (err != CL_SUCCESS) {
        printf("DEBUG: Accelerated kernel launch failed with error %d\n", err);
        if (upload_ev) clReleaseEvent(upload_ev);
        if (offsets_ev) clReleaseEvent(offsets_ev);
        if (outoffs_ev) clReleaseEvent(outoffs_ev);
        if (maxouts_ev) clReleaseEvent(maxouts_ev);
        free(output_offsets);
        free(max_output_sizes);
        lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch accelerated compression kernel");
        return 0;
    }

    HDEBUG("DEBUG: Kernel launched\n");

    // Wait for kernel completion (if not using profiling we keep previous behavior of blocking finish)
    if (compressor->enable_profiling) {
        // we'll wait on the download event later before memcpy; do not clFinish here to allow profiling
    } else {
        HDEBUG("DEBUG: Waiting for accelerated kernel completion...\n");
        /* Measure host-side kernel time when profiling is disabled */
        t_kernel_host_start = clock();
        err = clFinish(compressor->queue);
        t_kernel_host_end = clock();
        kernel_ms_host += (double)(t_kernel_host_end - t_kernel_host_start) / CLOCKS_PER_SEC * 1000.0;
        if (err != CL_SUCCESS) {
            printf("DEBUG: clFinish failed with error %d\n", err);
            if (upload_ev) clReleaseEvent(upload_ev);
            if (offsets_ev) clReleaseEvent(offsets_ev);
            if (outoffs_ev) clReleaseEvent(outoffs_ev);
            if (maxouts_ev) clReleaseEvent(maxouts_ev);
            free(output_offsets);
            free(max_output_sizes);
            lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to wait for accelerated kernel completion");
            return 0;
        }
        HDEBUG("DEBUG: Accelerated kernel completed\n");
    }

    // Download compressed sizes
    cl_event read_sizes_ev = NULL;
        cl_uint* compressed_sizes = (cl_uint*)malloc(num_blocks * sizeof(cl_uint));
        /* Total compressed frame size estimate includes header; we'll accumulate block sizes next */
        size_t total_compressed_size = LZ4F_HEADER_SIZE_MAX;
    /* Prepare final frame pointer and assembly position early for overlapped pipeline */
    unsigned char* frame_output = (unsigned char*)output;
    size_t frame_pos = 0;

    /* chunk size in blocks used by overlapped IO pipeline */
    int chunk_blocks = 8;
    int num_chunks = 0;
    unsigned char** tmp_chunk_bufs = NULL;
    size_t* tmp_chunk_buf_sizes = NULL;
    if (!compressed_sizes) {
        printf("ERROR: Failed to allocate compressed_sizes array for %zu blocks\n", num_blocks);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate compressed sizes array");
        return 0;
    }

    // Wait for kernel completion before reading results
    cl_uint num_wait_events = 0;
    cl_event wait_events[1];
    if (compressor->enable_profiling && kernel_ev) {
        wait_events[0] = kernel_ev;
        num_wait_events = 1;
    }
    else {
        /* If profiling not enabled, the kernel host wait time is the measurement.
        * We already recorded t_kernel_host_end/t_kernel_host_start earlier if not profiling; use that.
        */
        if (kernel_ms_host >= 0.0) {
            kernel_ms_dev = kernel_ms_host; /* fallback when no device-profiling present */
        }
    }

    /* If overlap is enabled and there are enough chunks, use overlapped pipeline.
     * Otherwise fallback to the non-overlap path to avoid small-chunk races.
     */
    if (compressor->enable_io_overlap) {
        if ((int)num_blocks < chunk_blocks) chunk_blocks = (int)num_blocks;
        num_chunks = (int)((num_blocks + chunk_blocks - 1) / chunk_blocks);
    } else {
        num_chunks = 0;
    }

    int use_io_overlap_local = compressor->enable_io_overlap && (num_chunks > 1);
    if (!use_io_overlap_local) {
        if (compressor->enable_profiling) {
            err = clEnqueueReadBuffer(compressor->queue, compressor->compressed_sizes_buffer, CL_TRUE, 0,
                                    sizeof(cl_uint) * num_blocks, compressed_sizes,
                                    num_wait_events, (num_wait_events ? wait_events : NULL), &read_sizes_ev);
        } else {
            t_d2h_start = clock();
            err = clEnqueueReadBuffer(compressor->queue, compressor->compressed_sizes_buffer, CL_TRUE, 0,
                                    sizeof(cl_uint) * num_blocks, compressed_sizes,
                                    num_wait_events, (num_wait_events ? wait_events : NULL), NULL);
            t_d2h_end = clock(); d2h_ms += (double)(t_d2h_end - t_d2h_start) / CLOCKS_PER_SEC * 1000.0;
        }
    } else {
        /* Overlapped IO per-chunk pipeline: upload per-chunk, kernel per-chunk, download per-chunk */
        const int IO_SLOTS = 2;
        /* chunk_blocks and num_chunks were computed above; re-evaluate chunk_blocks
         * just in case num_blocks changed; keep same semantics. */
        if ((int)num_blocks < chunk_blocks) chunk_blocks = (int)num_blocks;
        num_chunks = (int)((num_blocks + chunk_blocks - 1) / chunk_blocks);

        tmp_chunk_bufs = (unsigned char**)malloc(sizeof(unsigned char*) * num_chunks);
        tmp_chunk_buf_sizes = (size_t*)malloc(sizeof(size_t) * num_chunks);
        if (!tmp_chunk_bufs || !tmp_chunk_buf_sizes) {
            if (tmp_chunk_bufs) free(tmp_chunk_bufs);
            if (tmp_chunk_buf_sizes) free(tmp_chunk_buf_sizes);
            /* chunk_input_sizes and chunk_output_sizes are not yet allocated at this point */
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate temporary chunk arrays");
            return 0;
        }
        for (int ci = 0; ci < num_chunks; ++ci) { tmp_chunk_bufs[ci] = NULL; tmp_chunk_buf_sizes[ci] = 0; }

        // Precompute chunk sizes
        size_t* chunk_input_sizes = (size_t*)malloc(sizeof(size_t) * num_chunks);
        size_t* chunk_output_sizes = (size_t*)malloc(sizeof(size_t) * num_chunks);
        if (!chunk_input_sizes || !chunk_output_sizes) {
            if (chunk_input_sizes) free(chunk_input_sizes);
            if (chunk_output_sizes) free(chunk_output_sizes);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate chunk arrays");
            return 0;
        }
        size_t max_chunk_input = 0, max_chunk_output = 0;
        for (int ci = 0; ci < num_chunks; ++ci) {
            size_t start_block = (size_t)ci * chunk_blocks;
            size_t end_block = start_block + chunk_blocks;
            if (end_block > num_blocks) end_block = num_blocks;
            size_t in_sz = 0, out_sz = 0;
            for (size_t j = start_block; j < end_block; ++j) {
                size_t u = (size_t)block_offsets[j * 2 + 1];
                in_sz += u;
                out_sz += (u + (u / 255) + 64);
            }
            chunk_input_sizes[ci] = in_sz;
            chunk_output_sizes[ci] = out_sz;
            if (in_sz > max_chunk_input) max_chunk_input = in_sz;
            if (out_sz > max_chunk_output) max_chunk_output = out_sz;
        }

        if (!ensure_input_buffer(compressor, max_chunk_input * IO_SLOTS)) {
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to ensure overlapped input buffer");
            free(chunk_input_sizes); free(chunk_output_sizes);
            return 0;
        }
        if (!ensure_output_buffer(compressor, max_chunk_output * IO_SLOTS)) {
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to ensure overlapped output buffer");
            free(chunk_input_sizes); free(chunk_output_sizes);
            return 0;
        }

        cl_uint* h_offsets = (cl_uint*)malloc(sizeof(cl_uint) * chunk_blocks * 2);
        cl_uint* h_outoffs = (cl_uint*)malloc(sizeof(cl_uint) * chunk_blocks);
        cl_uint* h_maxouts = (cl_uint*)malloc(sizeof(cl_uint) * chunk_blocks);
        cl_uint* h_csizes = (cl_uint*)malloc(sizeof(cl_uint) * chunk_blocks);
        if (!h_offsets || !h_outoffs || !h_maxouts || !h_csizes) {
            if (h_offsets) free(h_offsets);
            if (h_outoffs) free(h_outoffs);
            if (h_maxouts) free(h_maxouts);
            if (h_csizes) free(h_csizes);
            free(chunk_input_sizes); free(chunk_output_sizes);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate per-chunk host arrays");
            return 0;
        }

        cl_event *upload_ev_slots = (cl_event*)malloc(sizeof(cl_event) * num_chunks);
        cl_event *kernel_ev_slots = (cl_event*)malloc(sizeof(cl_event) * num_chunks);
        cl_event *read_sizes_ev_slots = (cl_event*)malloc(sizeof(cl_event) * num_chunks);
        read_blocks_ev_slots = (cl_event*)malloc(sizeof(cl_event) * num_chunks);
        if (!upload_ev_slots || !kernel_ev_slots || !read_sizes_ev_slots || !read_blocks_ev_slots) {
            if (upload_ev_slots) free(upload_ev_slots);
            if (kernel_ev_slots) free(kernel_ev_slots);
            if (read_sizes_ev_slots) free(read_sizes_ev_slots);
            if (read_blocks_ev_slots) free(read_blocks_ev_slots);
            free(tmp_chunk_bufs); free(tmp_chunk_buf_sizes); free(chunk_input_sizes); free(chunk_output_sizes);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate per-chunk event arrays");
            return 0;
        }
        for (int e = 0; e < num_chunks; ++e) { upload_ev_slots[e] = NULL; kernel_ev_slots[e] = NULL; read_sizes_ev_slots[e] = NULL; read_blocks_ev_slots[e] = NULL; }

        frame_pos = 0; // reset pos
        for (int ci = 0; ci < num_chunks; ++ci) {
            int slot = ci % IO_SLOTS;
            /* When reusing ping-pong slots ensure previous transfer on same slot is complete
             * to avoid clobbering device buffers which are still being read by the host. */
            if (ci >= IO_SLOTS) {
                int prev_ci = ci - IO_SLOTS;
                if (read_blocks_ev_slots && read_blocks_ev_slots[prev_ci]) {
                    clWaitForEvents(1, &read_blocks_ev_slots[prev_ci]);
                    clReleaseEvent(read_blocks_ev_slots[prev_ci]);
                    read_blocks_ev_slots[prev_ci] = NULL;
                }
                if (kernel_ev_slots && kernel_ev_slots[prev_ci]) {
                    /* Wait for any outstanding kernel events that may still be writing into this slot */
                    clWaitForEvents(1, &kernel_ev_slots[prev_ci]);
                    /* kernel_ev_slots[prev_ci] will be released earlier - ensure it's null before reuse */
                }
            }
            size_t start_block = (size_t)ci * chunk_blocks;
            size_t end_block = start_block + chunk_blocks;
            if (end_block > num_blocks) end_block = num_blocks;
            size_t chunk_n = end_block - start_block;
            size_t chunk_input_off = block_offsets[start_block * 2];
            size_t chunk_input_sz = chunk_input_sizes[ci];
            size_t slot_input_base = (size_t)slot * max_chunk_input;
            size_t slot_output_base = (size_t)slot * max_chunk_output;

            if (compressor->use_pinned_memory && compressor->pinned_input_ptr && compressor->pinned_input_size >= slot_input_base + chunk_input_sz) {
                memcpy((unsigned char*)compressor->pinned_input_ptr + slot_input_base, (const unsigned char*)input + chunk_input_off, chunk_input_sz);
            }
            void* up_src = compressor->use_pinned_memory && compressor->pinned_input_ptr ? (void*)((unsigned char*)compressor->pinned_input_ptr + slot_input_base) : (void*)((unsigned char*)input + chunk_input_off);
            err = clEnqueueWriteBuffer(compressor->queue, compressor->input_buffer, CL_FALSE, slot_input_base, chunk_input_sz, up_src, 0, NULL, &upload_ev_slots[ci]);
            if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload chunk input"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }

            size_t acc_out = 0;
            for (size_t bi = start_block, idx = 0; bi < end_block; ++bi, ++idx) {
                size_t u = (size_t)block_offsets[bi * 2 + 1];
                h_offsets[idx * 2] = (cl_uint)( (size_t)block_offsets[bi * 2] - chunk_input_off + slot_input_base );
                h_offsets[idx * 2 + 1] = (cl_uint)u;
                size_t block_out_sz = u + (u / 255) + 64;
                h_outoffs[idx] = (cl_uint)(acc_out + slot_output_base);
                h_maxouts[idx] = (cl_uint)block_out_sz;
                acc_out += block_out_sz;
            }

            cl_event offsets_ev_local = NULL;
            /* Create chunk-specific device buffers for offsets/out offsets/max outs to avoid races when reusing
             * the global per-frame buffer across chunks. This prevents data races where later writes to the
             * global buffer can clobber data used by kernels still running for previous chunks. */
            cl_mem d_chunk_block_offsets = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, sizeof(cl_uint) * chunk_n * 2, NULL, &err);
            if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create chunk block_offsets buffer"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }
            err = clEnqueueWriteBuffer(compressor->queue, d_chunk_block_offsets, CL_FALSE, 0, sizeof(cl_uint) * chunk_n * 2, h_offsets, 1, &upload_ev_slots[ci], &offsets_ev_local);
            if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload chunk offsets"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }
            cl_event outoffs_ev_local = NULL;
            cl_mem d_chunk_output_offsets = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, sizeof(cl_uint) * chunk_n, NULL, &err);
            if (err != CL_SUCCESS) { clReleaseMemObject(d_chunk_block_offsets); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create chunk output_offsets buffer"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }
            err = clEnqueueWriteBuffer(compressor->queue, d_chunk_output_offsets, CL_FALSE, 0, sizeof(cl_uint) * chunk_n, h_outoffs, 1, &upload_ev_slots[ci], &outoffs_ev_local);
            if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload chunk out offsets"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }
            cl_event maxouts_ev_local = NULL;
            cl_mem d_chunk_maxouts = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, sizeof(cl_uint) * chunk_n, NULL, &err);
            if (err != CL_SUCCESS) { clReleaseMemObject(d_chunk_block_offsets); clReleaseMemObject(d_chunk_output_offsets); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create chunk max_outsizes buffer"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }
            err = clEnqueueWriteBuffer(compressor->queue, d_chunk_maxouts, CL_FALSE, 0, sizeof(cl_uint) * chunk_n, h_maxouts, 1, &upload_ev_slots[ci], &maxouts_ev_local);
            if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload chunk maxouts"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }

            cl_uint chunk_n_u32 = (cl_uint)chunk_n;
            cl_uint slot_input_size_u32 = (cl_uint)(slot_input_base + chunk_input_sz);
            /* Set kernel args and use chunk-specific offsets buffers */
            err = clSetKernelArg(compressor->compress_kernel, 6, sizeof(cl_uint), &chunk_n_u32);
            err |= clSetKernelArg(compressor->compress_kernel, 7, sizeof(cl_uint), &slot_input_size_u32);
            err |= clSetKernelArg(compressor->compress_kernel, 3, sizeof(cl_mem), &d_chunk_block_offsets);
            err |= clSetKernelArg(compressor->compress_kernel, 4, sizeof(cl_mem), &d_chunk_output_offsets);
            err |= clSetKernelArg(compressor->compress_kernel, 5, sizeof(cl_mem), &d_chunk_maxouts);
            if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set chunk kernel args"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }

            size_t cglobal = chunk_n; size_t clocal = local_work_size;
            if (clocal > cglobal) clocal = cglobal;
            size_t cglobal_r = cglobal;
            if (clocal > 0 && (cglobal % clocal) != 0) cglobal_r = ((cglobal + clocal - 1) / clocal) * clocal;
            cl_event wait_list_local[4]; cl_uint num_waits = 0;
            if (upload_ev_slots[ci]) wait_list_local[num_waits++] = upload_ev_slots[ci];
            if (offsets_ev_local) wait_list_local[num_waits++] = offsets_ev_local;
            if (outoffs_ev_local) wait_list_local[num_waits++] = outoffs_ev_local;
            if (maxouts_ev_local) wait_list_local[num_waits++] = maxouts_ev_local;
            err = clEnqueueNDRangeKernel(compressor->queue, compressor->compress_kernel, 1, NULL, &cglobal_r, &clocal, num_waits, (num_waits ? wait_list_local : NULL), &kernel_ev_slots[ci]);
            if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch chunk kernel"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }

            // read compressed sizes for this chunk (offset in buffer = start_block * sizeof(cl_uint))
            err = clEnqueueReadBuffer(compressor->queue, compressor->compressed_sizes_buffer, CL_FALSE,
                                    (size_t)start_block * sizeof(cl_uint), sizeof(cl_uint) * chunk_n, h_csizes, 1, &kernel_ev_slots[ci], &read_sizes_ev_slots[ci]);
            if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to enqueue read of chunk sizes"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }
            // Wait for sizes to be available and accumulate total compressed size
            clWaitForEvents(1, &read_sizes_ev_slots[ci]);
            for (size_t bi = 0; bi < chunk_n; ++bi) {
                compressed_sizes[start_block + bi] = h_csizes[bi];
                size_t block_uncompressed = (size_t)block_offsets[(start_block + bi) * 2 + 1];
                size_t csz = (size_t)h_csizes[bi];
                total_compressed_size += LZ4F_BLOCK_HEADER_SIZE;
                if (csz == 0 || csz >= block_uncompressed) total_compressed_size += block_uncompressed; else total_compressed_size += csz;
            }

            // Schedule a single read of the full chunk output region into temp chunk buffer; we'll assemble later
            if (chunk_output_sizes[ci] > 0) {
                tmp_chunk_bufs[ci] = (unsigned char*)malloc(chunk_output_sizes[ci]);
                tmp_chunk_buf_sizes[ci] = chunk_output_sizes[ci];
                if (!tmp_chunk_bufs[ci]) {
                    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate tmp chunk buffer");
                    free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes);
                    return 0;
                }
                HDEBUG("DEBUG: Enqueue chunk read for chunk %d slot=%d slot_input_base=%zu slot_output_base=%zu chunk_input_sz=%zu chunk_out_sz=%zu\n", ci, slot, slot_input_base, slot_output_base, chunk_input_sz, chunk_output_sizes[ci]);
                err = clEnqueueReadBuffer(compressor->queue, compressor->output_buffer, CL_FALSE, slot_output_base, chunk_output_sizes[ci], tmp_chunk_bufs[ci], 1, &kernel_ev_slots[ci], &read_blocks_ev_slots[ci]);
                if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to enqueue chunk read"); free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes); return 0; }
            } else {
                read_blocks_ev_slots[ci] = NULL;
                tmp_chunk_bufs[ci] = NULL;
                tmp_chunk_buf_sizes[ci] = 0;
            }
            if (upload_ev_slots[ci]) { clReleaseEvent(upload_ev_slots[ci]); upload_ev_slots[ci] = NULL; }
            if (kernel_ev_slots[ci]) { clReleaseEvent(kernel_ev_slots[ci]); kernel_ev_slots[ci] = NULL; }
            if (read_sizes_ev_slots[ci]) { clReleaseEvent(read_sizes_ev_slots[ci]); read_sizes_ev_slots[ci] = NULL; }
            /* read_blocks_ev_slots[ci] will be released after final assembly */
                if (offsets_ev_local) { clReleaseEvent(offsets_ev_local); offsets_ev_local = NULL; }
                if (outoffs_ev_local) { clReleaseEvent(outoffs_ev_local); outoffs_ev_local = NULL; }
                if (maxouts_ev_local) { clReleaseEvent(maxouts_ev_local); maxouts_ev_local = NULL; }
                /* Release chunk-specific device buffers; they will remain valid for queued kernels
                    as OpenCL retains object references internally until kernels finish using them. */
                if (d_chunk_block_offsets) { clReleaseMemObject(d_chunk_block_offsets); d_chunk_block_offsets = NULL; }
                if (d_chunk_output_offsets) { clReleaseMemObject(d_chunk_output_offsets); d_chunk_output_offsets = NULL; }
                if (d_chunk_maxouts) { clReleaseMemObject(d_chunk_maxouts); d_chunk_maxouts = NULL; }
        }
        free(h_offsets); free(h_outoffs); free(h_maxouts); free(h_csizes); free(chunk_input_sizes); free(chunk_output_sizes);
    }
    if (err != CL_SUCCESS) {
        printf("DEBUG: Failed to download compressed sizes, error %d\n", err);
        if (upload_ev) clReleaseEvent(upload_ev);
        if (offsets_ev) clReleaseEvent(offsets_ev);
        if (outoffs_ev) clReleaseEvent(outoffs_ev);
        if (maxouts_ev) clReleaseEvent(maxouts_ev);
        if (kernel_ev) clReleaseEvent(kernel_ev);
        free(output_offsets);
        free(max_output_sizes);
        lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download compressed sizes");
        return 0;
    }

    // Debug: Print compressed sizes for each block
    HDEBUG("DEBUG: Compressed sizes for %zu blocks (acceleration=%d):\n", num_blocks, acceleration);
    for (size_t _i = 0; _i < num_blocks; ++_i) {
        HDEBUG("DEBUG: compressed_sizes[%zu] = %u\n", _i, (unsigned int)compressed_sizes[_i]);
    }
    // Debug: print block offsets, output_offsets and max_output_sizes arrays
    HDEBUG("DEBUG: Block offsets (pairs):\n");
    for (size_t _i = 0; _i < num_blocks; ++_i) {
        HDEBUG("  block %zu -> offset=%u, size=%u\n", _i, (unsigned int)block_offsets[_i*2], (unsigned int)block_offsets[_i*2+1]);
    }
    HDEBUG("DEBUG: Output offsets (per block):\n");
    for (size_t _i = 0; _i < num_blocks; ++_i) {
        HDEBUG("  output_offsets[%zu] = %u (max_out %u)\n", _i, (unsigned int)output_offsets[_i], (unsigned int)max_output_sizes[_i]);
    }
    total_compressed_size += LZ4F_CONTENT_CHECKSUM_SIZE + LZ4F_ENDMARK_SIZE;

    HDEBUG("DEBUG: Total compressed size calculation: header=%d, blocks=%d, content_checksum=%d, endmark=%d\n",
        LZ4F_HEADER_SIZE_MAX, (int)(num_blocks * LZ4F_BLOCK_HEADER_SIZE),
        LZ4F_CONTENT_CHECKSUM_SIZE, LZ4F_ENDMARK_SIZE);

    if (total_compressed_size > output_capacity) {
        printf("DEBUG: Total compressed size %zu exceeds output capacity %zu\n", total_compressed_size, output_capacity);
        free(output_offsets);
        free(max_output_sizes);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Total compressed data too large for output buffer");
        return 0;
    }

    // Build final LZ4 frame (same as original function) -- frame_output and frame_pos declared earlier

    /* Start measuring frame assembly time (exclude D2H block reads, which are measured separately)
    * We'll stop the timer around per-block reads to avoid double counting D2H time in frame_ms.
    */
    t_frame_start = clock();

    // Write frame header manually (following LZ4F_compressBegin_internal)
    unsigned int magic = LZ4F_MAGICNUMBER;
    unsigned char* header_start;
    unsigned char FLG, BD, HC;
    /* content_size is intentionally unused; update FLG/BD to include if needed */

    // Write frame header
    LZ4_writeLE32(frame_output + frame_pos, magic);
    frame_pos += 4;
    header_start = frame_output + frame_pos;

    // FLG Byte - Version 01, Independent blocks (no linking), Content size, Content checksum, no block checksum, no dict
    // Use independent blocks so GPU can decompress blocks in parallel without relying on prior block state
    FLG = (BYTE)(((1 & 0x03) << 6)    /* Version('01') */
        + ((0 & 0x01) << 5)            /* Block Linked mode: 0 => independent blocks */
        + ((0 & 0x01) << 4)            /* No block checksum */
        + ((0 & 0x01) << 3)            /* No content size flag (match standard LZ4) */
        + ((1 & 0x01) << 2)            /* Content checksum flag */
        +  (0) );                      /* No dict ID */

    // BD Byte: choose Block Size ID based on the maximum uncompressed block size used
    size_t max_uncompressed_block = 0;
    for (size_t i = 0; i < num_blocks; ++i) {
        size_t bsz = (size_t)block_offsets[i * 2 + 1];
        if (bsz > max_uncompressed_block) max_uncompressed_block = bsz;
    }
    unsigned char blockSizeID = lz4f_blockSizeID_from_uncompressed_size(max_uncompressed_block);
    BD = (BYTE)((blockSizeID & 0x07) << 4);  /* Block size ID, no reserved bits */

    // Write FLG and BD
    frame_output[frame_pos++] = FLG;
    frame_output[frame_pos++] = BD;

    // Header CRC Byte - calculate checksum for entire header from FLG to end
    HC = LZ4F_headerChecksum(header_start, (size_t)(frame_output + frame_pos - header_start));
    frame_output[frame_pos++] = HC;

    HDEBUG("DEBUG: Frame header size: %zu\n", frame_pos);
    HDEBUG("DEBUG: FLG=0x%02x, BD=0x%02x, HC=0x%02x\n", FLG, BD, HC);

    // Write block headers and data (compressed or uncompressed fallback per block)
    for (size_t i = 0; i < num_blocks; i++) {
        HDEBUG("DEBUG: Start block %zu, initial frame_pos=%zu\n", i, frame_pos);
        size_t block_start = (size_t)block_offsets[i * 2];
        size_t block_uncompressed = (size_t)block_offsets[i * 2 + 1];
        size_t csz = (size_t)compressed_sizes[i];
        if (csz == 0 || csz >= block_uncompressed) {
            unsigned int header = (unsigned int)block_uncompressed | 0x80000000U;
            LZ4_writeLE32(frame_output + frame_pos, header);
            HDEBUG("DEBUG: Wrote uncompressed block header 0x%08x at pos=%zu\n", header, frame_pos);
            frame_pos += LZ4F_BLOCK_HEADER_SIZE;
            HDEBUG("DEBUG: Block %zu stored uncompressed: %zu bytes at frame offset %zu (input offset %zu)\n",
                i, block_uncompressed, frame_pos, block_start);
            memcpy(frame_output + frame_pos, (const unsigned char*)input + block_start, block_uncompressed);
            frame_pos += block_uncompressed;
            HDEBUG("DEBUG: Copied uncompressed block %zu payload, new frame_pos=%zu\n", i, frame_pos);
            HDEBUG("DEBUG: After writing uncompressed block %zu, frame_pos=%zu\n", i, frame_pos);
        } else {
            LZ4_writeLE32(frame_output + frame_pos, (unsigned int)csz);
            frame_pos += LZ4F_BLOCK_HEADER_SIZE;
            HDEBUG("DEBUG: Copying compressed block %zu: size=%zu\n", i, csz);

            /* Stop frame assembly timer while performing device->host read (D2H)
            * to avoid double counting D2H time in frame_ms.
            */
            t_frame_end = clock(); frame_ms += (double)(t_frame_end - t_frame_start) / CLOCKS_PER_SEC * 1000.0;
            if (!compressor->enable_io_overlap) {
                if (compressor->enable_profiling) {
                    cl_event read_block_ev = NULL;
                    void* d2h_dest_ptr = frame_output + frame_pos;
                    if (compressor->use_pinned_memory && compressor->pinned_output_ptr && compressor->pinned_output_size >= (size_t)output_offsets[i] + csz) {
                        d2h_dest_ptr = (unsigned char*)compressor->pinned_output_ptr + output_offsets[i];
                    }
                          HDEBUG("DEBUG: D2H (no-overlap, profiling enabled): block %zu output_off=%u, csz=%zu, d2h_dest_ptr=%p, frame_dest=%p\n",
                              i, (unsigned int)output_offsets[i], csz, d2h_dest_ptr, frame_output + frame_pos);
                          err = clEnqueueReadBuffer(compressor->queue, compressor->output_buffer, CL_TRUE,
                                      (size_t)output_offsets[i], csz, d2h_dest_ptr, 0,
                                      NULL, &read_block_ev);
                    if (read_block_ev) {
                        cl_ulong rs=0, re=0;
                        if (clGetEventProfilingInfo(read_block_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &rs, NULL) == CL_SUCCESS &&
                            clGetEventProfilingInfo(read_block_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &re, NULL) == CL_SUCCESS) {
                            double block_d2h_ms = (double)(re - rs) / 1e6;
                            d2h_ms += block_d2h_ms;
                            event_profile_ms += block_d2h_ms;
                        }
                        clReleaseEvent(read_block_ev);
                    }
                    /* If we're using pinned staging read, copy into frame output buffer */
                    if (d2h_dest_ptr == (void*)(frame_output + frame_pos)) {
                        HDEBUG("DEBUG: D2H wrote directly into frame_output for block %zu; first bytes: ", i);
                        size_t pda = csz < 8 ? csz : 8;
                        for (size_t bb = 0; bb < pda; ++bb) HDEBUG("%02x ", (unsigned int)frame_output[frame_pos + bb]);
                        HDEBUG("\n");
                    } else {
                        if (d2h_dest_ptr != (void*)(frame_output + frame_pos)) {
                            HDEBUG("DEBUG: D2H wrote to pinned ptr; copying into frame_output for block %zu\n", i);
                            memcpy(frame_output + frame_pos, d2h_dest_ptr, csz);
                            HDEBUG("DEBUG: After memcpy, first bytes at frame_output: ");
                            size_t pda = csz < 8 ? csz : 8;
                            for (size_t bb = 0; bb < pda; ++bb) HDEBUG("%02x ", (unsigned int)frame_output[frame_pos + bb]);
                            HDEBUG("\n");
                        }
                    }
                    /* Resume frame assembly timer */
                            t_frame_start = clock();
                            /* Advance frame_pos by csz for compressed block */
                            frame_pos += csz;
                            HDEBUG("DEBUG: After copying compressed block %zu, frame_pos=%zu\n", i, frame_pos);
                } else {
                    t_d2h_start = clock();
                    void* d2h_dest_ptr = frame_output + frame_pos;
                    if (compressor->use_pinned_memory && compressor->pinned_output_ptr && compressor->pinned_output_size >= (size_t)output_offsets[i] + csz) {
                        d2h_dest_ptr = (unsigned char*)compressor->pinned_output_ptr + output_offsets[i];
                    }
                          HDEBUG("DEBUG: D2H (no-overlap, blocking): block %zu output_off=%u, csz=%zu, d2h_dest_ptr=%p, frame_dest=%p\n",
                              i, (unsigned int)output_offsets[i], csz, d2h_dest_ptr, frame_output + frame_pos);
                          err = clEnqueueReadBuffer(compressor->queue, compressor->output_buffer, CL_TRUE,
                                      (size_t)output_offsets[i], csz, d2h_dest_ptr, 0, NULL, NULL);
                    if (d2h_dest_ptr == (void*)(frame_output + frame_pos)) {
                        HDEBUG("DEBUG: D2H (blocking) wrote directly into frame_output for block %zu; first bytes: ", i);
                        size_t pda = csz < 8 ? csz : 8;
                        for (size_t bb = 0; bb < pda; ++bb) HDEBUG("%02x ", (unsigned int)frame_output[frame_pos + bb]);
                        HDEBUG("\n");
                    } else {
                        HDEBUG("DEBUG: D2H (blocking) into pinned; copying into frame_output for block %zu\n", i);
                        memcpy(frame_output + frame_pos, d2h_dest_ptr, csz);
                        size_t pda = csz < 8 ? csz : 8;
                        HDEBUG("DEBUG: After memcpy (blocking), first bytes at frame_output: ");
                        for (size_t bb = 0; bb < pda; ++bb) HDEBUG("%02x ", (unsigned int)frame_output[frame_pos + bb]);
                        HDEBUG("\n");
                    }
                    t_d2h_end = clock(); d2h_ms += (double)(t_d2h_end - t_d2h_start) / CLOCKS_PER_SEC * 1000.0;
                    /* Resume frame assembly timer */
                    t_frame_start = clock();
                    /* Advance frame_pos by csz for compressed block */
                    frame_pos += csz;
                }
            } else {
                /* Overlapped path uses temporary chunk buffers already populated.
                * Wait for chunk D2H read to complete then copy compressed data from tmp_chunk_bufs to frame output. */
                int chunk_index = (int)(i / chunk_blocks);
                size_t start_block = (size_t)chunk_index * chunk_blocks;
                size_t local_pos = 0;
                /* tmp_chunk_bufs is laid out using per-block buffer sizes (max output sizes per block),
                 * not compacted packed compressed sizes. For each prior block in the chunk we must
                 * skip the full block buffer region (computed as block_uncomp + block_uncomp/255 + 64)
                 * to find the start of the current block's data inside the tmp buffer. Using the
                 * actual compressed size (csz) would incorrectly compact the offsets and lead to
                 * misaligned copies (e.g., reading zeros or previous block padding). */
                for (size_t k = start_block; k < i; ++k) {
                    size_t prev_uncomp = (size_t)block_offsets[k * 2 + 1];
                    size_t prev_block_bufsz = prev_uncomp + (prev_uncomp / 255) + 64;
                    local_pos += prev_block_bufsz;
                }
                if (read_blocks_ev_slots && read_blocks_ev_slots[chunk_index]) {
                    clWaitForEvents(1, &read_blocks_ev_slots[chunk_index]);
                    clReleaseEvent(read_blocks_ev_slots[chunk_index]);
                    read_blocks_ev_slots[chunk_index] = NULL;
                }
                /* Debug: dump first bytes of the tmp chunk buffer after D2H read completes */
                if (tmp_chunk_bufs[chunk_index] && tmp_chunk_buf_sizes[chunk_index] > 0) {
                    size_t dump_n = tmp_chunk_buf_sizes[chunk_index] < 32 ? tmp_chunk_buf_sizes[chunk_index] : 32;
                    HDEBUG("DEBUG: Chunk %d tmp buffer first %zu bytes: ", chunk_index, dump_n);
                    for (size_t db = 0; db < dump_n; ++db) HDEBUG("%02x ", tmp_chunk_bufs[chunk_index][db]);
                    HDEBUG("\n");
                }
                /* Debug: log some bytes prior to copy (to detect any corruption) */
                if (tmp_chunk_bufs[chunk_index]) {
                    size_t p_dump = csz < 8 ? csz : 8;
                    HDEBUG("DEBUG: Copying compressed block %zu from tmp_chunk[%d] + %zu -> frame_pos=%zu, size=%zu (first %zu bytes): ", i, chunk_index, local_pos, frame_pos, csz, p_dump);
                    for (size_t bb = 0; bb < p_dump; ++bb) HDEBUG("%02x ", tmp_chunk_bufs[chunk_index][local_pos + bb]);
                    HDEBUG("\n");
                }
                /* Pre-copy bytes at destination for comparison */
                {
                    size_t pdd = csz < 8 ? csz : 8;
                    HDEBUG("DEBUG: Frame bytes at dest before copy (first %zu bytes): ", pdd);
                    for (size_t bb = 0; bb < pdd; ++bb) HDEBUG("%02x ", (unsigned int)frame_output[frame_pos + bb]);
                    HDEBUG("\n");
                }
                memcpy(frame_output + frame_pos, tmp_chunk_bufs[chunk_index] + local_pos, csz);
                /* Debug: print few bytes after copy */
                {
                    size_t pda = csz < 8 ? csz : 8;
                    HDEBUG("DEBUG: Frame bytes at dest after copy (first %zu bytes): ", pda);
                    for (size_t bb = 0; bb < pda; ++bb) HDEBUG("%02x ", (unsigned int)frame_output[frame_pos + bb]);
                    HDEBUG("\n");
                }
                /* Resume frame assembly timer */
                t_frame_start = clock();
                /* Advance frame_pos by csz for compressed block (overlapped path) */
                frame_pos += csz;
            }
                /* (No-op) D2H actions handled above depending on profiling or overlap mode */
                t_frame_start = clock();
            }
            if (err != CL_SUCCESS) {
                printf("DEBUG: Failed to download compressed block %zu, error %d\n", i, err);
                if (upload_ev) clReleaseEvent(upload_ev);
                if (offsets_ev) clReleaseEvent(offsets_ev);
                if (outoffs_ev) clReleaseEvent(outoffs_ev);
                if (maxouts_ev) clReleaseEvent(maxouts_ev);
                if (kernel_ev) clReleaseEvent(kernel_ev);
                free(output_offsets);
                free(max_output_sizes);
                lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download compressed block data");
                return 0;
            }

            /* Only advance frame_pos by compressed size for compressed blocks (we already updated for uncompressed above) */
            /* frame_pos increment moved into compressed branch to avoid double counting for uncompressed blocks */
        }

    HDEBUG("DEBUG: Writing end mark at frame_pos=%zu\n", frame_pos);
    /* Dump some of the next bytes in the frame buffer to help debug any unexpected data */
    {
        size_t dump_bytes = 16;
        HDEBUG("DEBUG: Bytes at frame_pos (hex): ");
        for (size_t di = 0; di < dump_bytes; ++di) {
            size_t idx = frame_pos + di;
            if (idx < output_capacity) HDEBUG("%02x ", (unsigned int)frame_output[idx]);
            else HDEBUG("-- ");
        }
        HDEBUG("\n");
    }
    // Write end mark (4 zero bytes) - this MUST be the last part before checksum
    LZ4_writeLE32(frame_output + frame_pos, 0);
    frame_pos += LZ4F_ENDMARK_SIZE;

    // Add content checksum AFTER end mark if enabled (XXH32 hash)
    /* Finish frame assembly timing */
    t_frame_end = clock(); frame_ms += (double)(t_frame_end - t_frame_start) / CLOCKS_PER_SEC * 1000.0;
    if ((FLG >> 2) & 1) {
        // Calculate proper XXH32 hash of input data
        unsigned int checksum = XXH32(input, input_size, 0);
        LZ4_writeLE32(frame_output + frame_pos, checksum);
        frame_pos += LZ4F_CONTENT_CHECKSUM_SIZE;
        HDEBUG("DEBUG: Added content checksum after end mark: 0x%08x\n", checksum);
    }

    HDEBUG("DEBUG: Final frame size: %zu bytes (acceleration=%d)\n", frame_pos, acceleration);
    /* Dump first bytes of frame to help debug any unexpected inserted data */
    {
        size_t dump_n = frame_pos < 256 ? frame_pos : 256;
        HDEBUG("DEBUG: Frame (first %zu bytes):\n", dump_n);
        for (size_t j = 0; j < dump_n; ++j) {
            if ((j % 16) == 0) HDEBUG("%04zx: ", j);
            HDEBUG("%02x ", (unsigned int)frame_output[j]);
            if ((j % 16) == 15) HDEBUG("\n");
        }
        if ((dump_n % 16) != 0) HDEBUG("\n");
    }
    // Dump the full frame first 128 bytes for debugging
    size_t dump_sz = frame_pos < 128 ? frame_pos : 128;
    HDEBUG("DEBUG: Frame first %zu bytes: ", dump_sz);
    for (size_t di = 0; di < dump_sz; ++di) {
        HDEBUG("%02x ", (unsigned int)frame_output[di]);
    }
    HDEBUG("\n");

    // Populate profiling timings
    if (compressor->enable_profiling) {
        cl_ulong s = 0, e = 0;
        /* H2D upload event profiling */
        if (upload_ev) {
            if (clGetEventProfilingInfo(upload_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(upload_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                double ms = (double)(e - s) / 1e6;
                h2d_ms += ms;
                event_profile_ms += ms;
            }
        }
        if (offsets_ev) {
            if (clGetEventProfilingInfo(offsets_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(offsets_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                double ms = (double)(e - s) / 1e6;
                h2d_ms += ms;
                event_profile_ms += ms;
            }
        }
        /* Use outer kernel_ms_dev variable; do not shadow */

        if (upload_ev) {
            if (clGetEventProfilingInfo(upload_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(upload_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                h2d_ms += (double)(e - s) / 1e6;
                event_profile_ms += (double)(e - s) / 1e6;
            }
        }
        if (offsets_ev) {
            if (clGetEventProfilingInfo(offsets_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(offsets_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                h2d_ms += (double)(e - s) / 1e6;
                event_profile_ms += (double)(e - s) / 1e6;
            }
        }
        if (outoffs_ev) {
            if (clGetEventProfilingInfo(outoffs_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(outoffs_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                h2d_ms += (double)(e - s) / 1e6;
                event_profile_ms += (double)(e - s) / 1e6;
            }
        }
        if (maxouts_ev) {
            if (clGetEventProfilingInfo(maxouts_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(maxouts_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                h2d_ms += (double)(e - s) / 1e6;
                event_profile_ms += (double)(e - s) / 1e6;
            }
        }
        if (kernel_ev) {
            if (clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                kernel_ms_dev = (double)(e - s) / 1e6;
                event_profile_ms += kernel_ms_dev;
            }
        }

        // Also include read_sizes_ev profiling info if present
        if (read_sizes_ev) {
            cl_ulong rs = 0, re = 0;
            if (clGetEventProfilingInfo(read_sizes_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &rs, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(read_sizes_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &re, NULL) == CL_SUCCESS) {
                double read_sizes_ms = (double)(re - rs) / 1e6;
                d2h_ms += read_sizes_ms;
                event_profile_ms += read_sizes_ms;
            }
        }
        /* compute total wall time and fill last_timing */
        clock_t tend = clock();
        double total_ms = (double)(tend - tstart) / CLOCKS_PER_SEC * 1000.0;

        compressor->last_timing.total_ms = total_ms;
        compressor->last_timing.alloc_ms = alloc_ms;
        compressor->last_timing.h2d_ms = h2d_ms;
        compressor->last_timing.setup_ms = setup_ms;
        compressor->last_timing.kernel_ms_device = (kernel_ms_dev > 0.0) ? kernel_ms_dev : -1.0;
        compressor->last_timing.kernel_ms = (kernel_ms_dev > 0.0) ? kernel_ms_dev : kernel_ms_host;
        compressor->last_timing.event_profile_ms = event_profile_ms;
        compressor->last_timing.d2h_ms = d2h_ms;
        compressor->last_timing.frame_ms = frame_ms;
        compressor->last_timing.map_ms = 0.0;
        compressor->last_timing.input_bytes = input_size;
        compressor->last_timing.output_bytes = frame_pos;
        compressor->last_timing.block_size = compressor->dynamic_block_size;
        compressor->last_timing.num_blocks = (int)num_blocks;
    } else {
        /* Non-profiling path: fill last_timing from host-measured values */
        clock_t tend = clock();
        double total_ms = (double)(tend - tstart) / CLOCKS_PER_SEC * 1000.0;
        compressor->last_timing.total_ms = total_ms;
        compressor->last_timing.alloc_ms = alloc_ms;
        compressor->last_timing.h2d_ms = h2d_ms;
        compressor->last_timing.setup_ms = setup_ms;
        compressor->last_timing.kernel_ms_device = -1.0;
        compressor->last_timing.kernel_ms = (kernel_ms_host >= 0.0) ? kernel_ms_host : -1.0;
        compressor->last_timing.event_profile_ms = event_profile_ms;
        compressor->last_timing.d2h_ms = d2h_ms;
        compressor->last_timing.frame_ms = frame_ms;
        compressor->last_timing.map_ms = 0.0;
        compressor->last_timing.input_bytes = input_size;
        compressor->last_timing.output_bytes = frame_pos;
        compressor->last_timing.block_size = compressor->dynamic_block_size;
        compressor->last_timing.num_blocks = (int)num_blocks;
    }

    // Cleanup
    if (upload_ev) clReleaseEvent(upload_ev);
    if (offsets_ev) clReleaseEvent(offsets_ev);
    if (outoffs_ev) clReleaseEvent(outoffs_ev);
    if (maxouts_ev) clReleaseEvent(maxouts_ev);
    if (kernel_ev) clReleaseEvent(kernel_ev);
    /* persistent device buffers (input_buffer, output_buffer) are released when compressor is destroyed */
    // Free dynamically allocated block offsets
    free(block_offsets);
    free(output_offsets);
    free(max_output_sizes);

    // Free temporary chunk buffers used in overlapped mode
    if (compressor->enable_io_overlap) {
        if (tmp_chunk_bufs) {
            for (int ci = 0; ci < num_chunks; ++ci) {
                if (tmp_chunk_bufs[ci]) free(tmp_chunk_bufs[ci]);
            }
            free(tmp_chunk_bufs);
            tmp_chunk_bufs = NULL;
        }
        if (tmp_chunk_buf_sizes) { free(tmp_chunk_buf_sizes); tmp_chunk_buf_sizes = NULL; }
        // Release per-chunk events
        if (upload_ev_slots) { free(upload_ev_slots); upload_ev_slots = NULL; }
        if (kernel_ev_slots) { free(kernel_ev_slots); kernel_ev_slots = NULL; }
        if (read_sizes_ev_slots) { free(read_sizes_ev_slots); read_sizes_ev_slots = NULL; }
        if (read_blocks_ev_slots) { free(read_blocks_ev_slots); read_blocks_ev_slots = NULL; }
    }

return frame_pos;
}

// Block information structure for parallel decompression
typedef struct {
    size_t compressed_offset;    // Offset in compressed frame
    size_t compressed_size;      // Size of compressed block
    size_t output_offset;        // Offset in output buffer
    size_t max_output_size;      // Maximum decompressed size for this block
    int is_compressed;           // 1 if compressed, 0 if uncompressed
} LZ4BlockInfo;

// GPU-accelerated LZ4 frame decompression - Parallel block processing
size_t lz4_gpu_decompress_frame(LZ4GPUCompressor* compressor,
                                const void* input, size_t input_size,
                                void* output, size_t output_capacity) {
    if (!compressor || !input || !output || input_size == 0) {
        if (compressor) {
            lz4_gpu_set_error(compressor, LZ4_GPU_INVALID_PARAMS, "Invalid input parameters");
        }
        return 0;
    }

    if (!compressor->context) {
        lz4_gpu_set_error(compressor, LZ4_GPU_NOT_INITIALIZED, "Compressor not initialized");
        return 0;
    }

    HDEBUG("DEBUG: lz4_gpu_decompress_frame starting - parallel block processing\n");

    /* Start wall clock for operation timing */
    clock_t tstart = clock();

    /* timing variables for decompression path */
    clock_t t_alloc_start = 0, t_alloc_end = 0;
    clock_t t_h2d_start = 0, t_h2d_end = 0;
    clock_t t_setup_start = 0, t_setup_end = 0;
    clock_t t_kernel_host_start = 0, t_kernel_host_end = 0;
    clock_t t_d2h_start = 0, t_d2h_end = 0;
    clock_t t_frame_start = 0, t_frame_end = 0;
    double alloc_ms = 0.0, h2d_ms = 0.0, setup_ms = 0.0, kernel_ms_host = -1.0, kernel_ms_dev = -1.0, d2h_ms = 0.0, frame_ms = 0.0, event_profile_ms = 0.0;

    const unsigned char* ip = (const unsigned char*)input;
    const unsigned char* const iend = ip + input_size;

    // Phase 1: Parse frame header and collect block information
    HDEBUG("DEBUG: Phase 1 - Parsing frame header and collecting block info...\n");

    // Check magic number
    if (LZ4_readLE32(ip) != LZ4F_MAGICNUMBER) {
        lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Invalid LZ4 frame magic number");
        return 0;
    }
    ip += 4;

    // Parse FLG and BD bytes
    unsigned char FLG = *ip++;
    unsigned char BD = *ip++;

    // Parse content size if present
    size_t contentSize = 0;
    if (FLG & (1 << 3)) { // Content size flag
        if (ip + 8 > iend) {
            lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame header too short for content size");
            return 0;
        }
        contentSize = (size_t)LZ4_readLE32(ip) | ((size_t)LZ4_readLE32(ip + 4) << 32);
        ip += 8;
        HDEBUG("DEBUG: Content size: %zu bytes\n", contentSize);
    }

    // Skip header checksum
    ip++;

    unsigned int blockChecksumFlag = (FLG >> 4) & 1;
    unsigned int contentChecksumFlag = (FLG >> 2) & 1;
    /* Suppress unused variable warnings — flags are parsed for future use */
    (void)contentSize;
    (void)blockChecksumFlag;

    /* Avoid unused variable warnings when features are not enabled/checked yet */
    (void)contentSize;
    (void)blockChecksumFlag;

    /*
     * Calculate maximum possible blocks for dynamic allocation.
     * Important: do NOT rely on compressor->dynamic_block_size here because the
     * compressor instance's dynamic_block_size reflects a runtime tuning value
     * used during compression and may not match the block layout in the frame
     * being parsed (or may be the default). Using it can truncate parsing
     * (e.g. if dynamic_block_size > actual block size used when compressing).
     *
     * Use a conservative minimum block size for parsing (4KB) to compute an
     * upper bound for the number of blocks. This avoids missing blocks while
     * keeping memory allocation reasonable.
     */
    const size_t parse_min_block = LZ4_GPU_MIN_BLOCK_SIZE; /* 4KB */

    /* Determine an initial capacity for blocks array. Prefer using the
     * content size (uncompressed) when present; otherwise use a sane default
     * to avoid underestimating the number of blocks when input_size is the
     * compressed frame size. We also enforce a minimum capacity.
     */
    size_t initial_capacity;
    if (contentSize > 0) {
        initial_capacity = (contentSize + parse_min_block - 1) / parse_min_block + 4;
    } else {
        /* No content size available in header: start with a reasonable default */
        initial_capacity = 256;
    }

    if (initial_capacity < 16) initial_capacity = 16;

    // Collect block information
    size_t capacity = initial_capacity; /* initial conservative capacity */
    LZ4BlockInfo* blocks = (LZ4BlockInfo*)malloc(capacity * sizeof(LZ4BlockInfo));
    if (!blocks) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate blocks array");
        return 0;
    }
    size_t num_blocks = 0;
    size_t num_compressed_blocks = 0;

    // Calculate where the content checksum and endmark should be
    const unsigned char* contentChecksumPos = NULL;
    if (contentChecksumFlag) {
        if ((size_t)(iend - ip) < (LZ4F_CONTENT_CHECKSUM_SIZE + LZ4F_ENDMARK_SIZE)) {
            free(blocks);
            lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame too small for content checksum / endmark");
            return 0;
        }
        contentChecksumPos = iend - (LZ4F_CONTENT_CHECKSUM_SIZE + LZ4F_ENDMARK_SIZE);
    } else {
        if ((size_t)(iend - ip) < LZ4F_ENDMARK_SIZE) {
            free(blocks);
            lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame too small for endmark");
            return 0;
        }
        contentChecksumPos = iend - LZ4F_ENDMARK_SIZE;
    }
    (void)contentSize; (void)blockChecksumFlag;

    /* Parse until the content checksum / endmark region. We do not limit by
     * a precomputed max_blocks here because the blocks array grows as needed.
     * A hard safety cap prevents pathological frames from exhausting memory.
     */
    const size_t HARD_MAX_BLOCKS = 1 << 20; /* ~1M blocks safety cap */
    while (ip < contentChecksumPos) {
        if (ip + LZ4F_BLOCK_HEADER_SIZE > iend) {
            lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Block header truncated");
            return 0;
        }

    unsigned int blockHeader = LZ4_readLE32(ip);
    ip += LZ4F_BLOCK_HEADER_SIZE;

    unsigned int blockSize = blockHeader & 0x7FFFFFFFU;
    int isCompressed = !(blockHeader & LZ4F_BLOCKUNCOMPRESSED_FLAG);

    HDEBUG("DEBUG: Raw block header at offset %zu: 0x%08x, size=%u, compressed=%d\n",
         (size_t)(ip - (const unsigned char*)input) - LZ4F_BLOCK_HEADER_SIZE,
         blockHeader, blockSize, isCompressed);

    if (blockSize == 0) {
    HDEBUG("DEBUG: Found end mark after %zu blocks\n", num_blocks);
        break; // End mark
    }

    /* Basic bounds checks before accepting block */
    if (ip + blockSize > iend) {
        free(blocks);
        lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Block data truncated or blockSize invalid");
        return 0;
    }

    /* Grow blocks array if needed */
    if (num_blocks >= capacity) {
        size_t new_capacity = capacity * 2;
        LZ4BlockInfo* tmp = (LZ4BlockInfo*)realloc(blocks, new_capacity * sizeof(LZ4BlockInfo));
        if (!tmp) {
            free(blocks);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to grow blocks array");
            return 0;
        }
        blocks = tmp;
        capacity = new_capacity;
    }

    blocks[num_blocks].compressed_offset = (size_t)(ip - (const unsigned char*)input);
    blocks[num_blocks].compressed_size = blockSize;
    blocks[num_blocks].is_compressed = isCompressed;

    if (isCompressed) num_compressed_blocks++;

    HDEBUG("DEBUG: Block %zu: offset=%zu, size=%u, compressed=%d\n",
         num_blocks, blocks[num_blocks].compressed_offset, blockSize, isCompressed);

    ip += blockSize;
    num_blocks++;

    if (num_blocks >= HARD_MAX_BLOCKS) {
        free(blocks);
        lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Too many blocks in frame (safety limit reached)");
        return 0;
    }

    // Skip block checksum if present
    if (blockChecksumFlag) {
        if (ip + LZ4F_BLOCK_CHECKSUM_SIZE > iend) {
            lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Block checksum truncated");
            return 0;
        }
        ip += LZ4F_BLOCK_CHECKSUM_SIZE;
    }
    }

    // Skip content checksum if present
    if (contentChecksumFlag) {
        if (ip + LZ4F_CONTENT_CHECKSUM_SIZE > iend) {
            lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Content checksum truncated");
            return 0;
        }
        ip += LZ4F_CONTENT_CHECKSUM_SIZE;
    }

    if (num_blocks == 0) {
    HDEBUG("DEBUG: No blocks to process\n");
        free(blocks);
        return 0;
    }

    if (contentSize > 0 && contentSize > output_capacity) {
        free(blocks);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Output buffer too small for declared content size");
        return 0;
    }

    // Phase 2: Process blocks - uncompressed blocks on CPU, compressed blocks on GPU
    HDEBUG("DEBUG: Phase 2 - Processing %zu blocks (CPU for uncompressed, GPU for compressed)...\n", num_blocks);
    HDEBUG("DEBUG: %zu blocks total, %zu compressed blocks require GPU processing\n", num_blocks, num_compressed_blocks);

    if (num_compressed_blocks == 0) {
        HDEBUG("DEBUG: All blocks are uncompressed, processing on CPU...\n");

        size_t stream_offset = 0;
        for (size_t i = 0; i < num_blocks; i++) {
            size_t block_size = blocks[i].compressed_size;
            if (stream_offset + block_size > output_capacity) {
                free(blocks);
                lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Output buffer too small for uncompressed blocks");
                return 0;
            }
            memcpy((unsigned char*)output + stream_offset,
                   (const unsigned char*)input + blocks[i].compressed_offset,
                   block_size);
            stream_offset += block_size;
            HDEBUG("DEBUG: CPU copied block %zu: %zu bytes at offset %zu\n", i, block_size, stream_offset - block_size);
        }

        free(blocks);
        return stream_offset;
    }

    // We have compressed blocks, do 2-pass parallel decompression
    cl_int err;
    /* Profiling events */
    cl_event upload_ev = NULL, offsets_ev = NULL, sizes_ev = NULL, copy_ev = NULL;
    cl_event kernel_ev = NULL; /* top-level kernel event for non-overlap profiling */
    /* Multi-block kernel and profiling events for pass 2 */
    cl_kernel k_multi = NULL;
    cl_event k_multi_ev = NULL;
    cl_event read_stream_ev = NULL;

    // Create/reuse buffers for GPU processing
    /* Allocation phase: start timing */
    t_alloc_start = clock();
    if (!ensure_input_buffer(compressor, input_size)) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create input buffer");
        return 0;
    }

    if (!ensure_output_buffer(compressor, output_capacity)) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output buffer");
        return 0;
    }

    // Upload input frame once using pinned staging buffer if available
    void* upload_src = (void*)input;
    if (compressor->use_pinned_memory && compressor->pinned_input_ptr && compressor->pinned_input_size >= input_size) {
        memcpy(compressor->pinned_input_ptr, input, input_size);
        upload_src = compressor->pinned_input_ptr;
    }
    if (compressor->enable_profiling) {
        err = clEnqueueWriteBuffer(compressor->queue, compressor->input_buffer, CL_FALSE, 0, input_size, upload_src, 0, NULL, &upload_ev);
    } else {
        t_h2d_start = clock();
        err = clEnqueueWriteBuffer(compressor->queue, compressor->input_buffer, CL_TRUE, 0, input_size, upload_src, 0, NULL, NULL);
        t_h2d_end = clock(); h2d_ms += (double)(t_h2d_end - t_h2d_start) / CLOCKS_PER_SEC * 1000.0;
    }
    if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload input data"); return 0; }

    // Prepare arrays for compressed blocks
    size_t ncb = num_compressed_blocks;
    cl_ulong* h_comp_offsets = (cl_ulong*)malloc(ncb * sizeof(cl_ulong));
    cl_ulong* h_comp_sizes   = (cl_ulong*)malloc(ncb * sizeof(cl_ulong));
    cl_uint*  h_sizes_out    = (cl_uint*)malloc(ncb * sizeof(cl_uint));
    size_t*   h_comp_block_index = (size_t*)malloc(ncb * sizeof(size_t)); // map compressed idx -> global block idx
    if (!h_comp_offsets || !h_comp_sizes || !h_sizes_out || !h_comp_block_index) {
        if (h_comp_offsets) free(h_comp_offsets);
        if (h_comp_sizes) free(h_comp_sizes);
        if (h_sizes_out) free(h_sizes_out);
        if (h_comp_block_index) free(h_comp_block_index);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate host arrays for parallel decompression");
        return 0;
    }

    // Fill arrays
    size_t ci = 0;
    for (size_t i = 0; i < num_blocks; ++i) {
        if (blocks[i].is_compressed) {
            h_comp_offsets[ci] = (cl_ulong)blocks[i].compressed_offset;
            h_comp_sizes[ci]   = (cl_ulong)blocks[i].compressed_size;
            h_comp_block_index[ci] = i;
            ci++;
        }
    }

    // Derive max block output size from BD (block size ID)
    unsigned char blockSizeID = (BD >> 4) & 0x07;
    cl_ulong max_block_out_size = 64ULL * 1024ULL;
    if (blockSizeID == 5) max_block_out_size = 256ULL * 1024ULL;
    else if (blockSizeID == 6) max_block_out_size = 1024ULL * 1024ULL;
    else if (blockSizeID == 7) max_block_out_size = 4ULL * 1024ULL * 1024ULL;

    // Device buffers for pass 1
    cl_mem d_comp_offsets = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create comp_offsets buffer"); return 0; }
    cl_mem d_comp_sizes   = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create comp_sizes buffer"); return 0; }
    cl_mem d_sizes_out    = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE, ncb * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create sizes_out buffer"); return 0; }

    /* Allocation phase: end timing */
    t_alloc_end = clock(); alloc_ms += (double)(t_alloc_end - t_alloc_start) / CLOCKS_PER_SEC * 1000.0;

    // Upload arrays
    if (compressor->enable_profiling) {
        err  = clEnqueueWriteBuffer(compressor->queue, d_comp_offsets, CL_FALSE, 0, ncb * sizeof(cl_ulong), h_comp_offsets, 0, NULL, &offsets_ev);
        err |= clEnqueueWriteBuffer(compressor->queue, d_comp_sizes,   CL_FALSE, 0, ncb * sizeof(cl_ulong), h_comp_sizes,   0, NULL, &sizes_ev);
    } else {
        t_h2d_start = clock();
        err  = clEnqueueWriteBuffer(compressor->queue, d_comp_offsets, CL_TRUE, 0, ncb * sizeof(cl_ulong), h_comp_offsets, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(compressor->queue, d_comp_sizes,   CL_TRUE, 0, ncb * sizeof(cl_ulong), h_comp_sizes,   0, NULL, NULL);
        t_h2d_end = clock(); h2d_ms += (double)(t_h2d_end - t_h2d_start) / CLOCKS_PER_SEC * 1000.0;
    }
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload comp arrays"); return 0; }

    // Create and run size-only kernel
    cl_event k_size_ev = NULL;
    cl_event read_sizes_ev = NULL;
    cl_kernel k_size = clCreateKernel(compressor->program, "lz4_decompress_blocks_sizeonly", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create size-only kernel"); return 0; }

    cl_uint ncb_u32 = (cl_uint)ncb;
    /* Measure setup time for size-only kernel args and enqueue */
    t_setup_start = clock();
    err  = clSetKernelArg(k_size, 0, sizeof(cl_mem), &compressor->input_buffer);
    err |= clSetKernelArg(k_size, 1, sizeof(cl_mem), &d_comp_offsets);
    err |= clSetKernelArg(k_size, 2, sizeof(cl_mem), &d_comp_sizes);
    err |= clSetKernelArg(k_size, 3, sizeof(cl_ulong), &max_block_out_size);
    err |= clSetKernelArg(k_size, 4, sizeof(cl_mem), &d_sizes_out);
    err |= clSetKernelArg(k_size, 5, sizeof(cl_uint), &ncb_u32);
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set size-only kernel args"); return 0; }

    size_t gws = ncb;
    size_t lws = compressor->default_local;
    if (lws == 0) lws = 1;
    if (compressor->device_max_work_group_size > 0 && lws > compressor->device_max_work_group_size) lws = compressor->device_max_work_group_size;
    if (lws > gws) lws = gws;
    size_t gws_rounded = gws;
    if (lws > 0 && (gws % lws) != 0) gws_rounded = ((gws + lws - 1) / lws) * lws;
    {
        cl_event wait_list_local[4]; cl_uint num_waits = 0;
        if (upload_ev) wait_list_local[num_waits++] = upload_ev;
        if (offsets_ev) wait_list_local[num_waits++] = offsets_ev;
        if (sizes_ev) wait_list_local[num_waits++] = sizes_ev;
        if (compressor->enable_profiling) {
            err = clEnqueueNDRangeKernel(compressor->queue, k_size, 1, NULL, &gws_rounded, &lws, num_waits, (num_waits ? wait_list_local : NULL), &k_size_ev);
        } else {
            err = clEnqueueNDRangeKernel(compressor->queue, k_size, 1, NULL, &gws_rounded, &lws, num_waits, (num_waits ? wait_list_local : NULL), NULL);
        }
    }
    t_setup_end = clock(); setup_ms += (double)(t_setup_end - t_setup_start) / CLOCKS_PER_SEC * 1000.0;
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch size-only kernel"); return 0; }

    // Read back sizes
    if (compressor->enable_profiling && k_size_ev) {
        cl_event waits[1]; waits[0] = k_size_ev;
        err = clEnqueueReadBuffer(compressor->queue, d_sizes_out, CL_TRUE, 0, ncb * sizeof(cl_uint), h_sizes_out, 1, waits, &read_sizes_ev);
    } else {
        /* When profiling is disabled, we measure host-side kernel time around clFinish */
        t_kernel_host_start = clock();
        err = clEnqueueReadBuffer(compressor->queue, d_sizes_out, CL_TRUE, 0, ncb * sizeof(cl_uint), h_sizes_out, 0, NULL, NULL);
        t_kernel_host_end = clock();
        kernel_ms_host += (double)(t_kernel_host_end - t_kernel_host_start) / CLOCKS_PER_SEC * 1000.0;
    }
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to read sizes_out"); return 0; }

    clReleaseKernel(k_size);

    // Compute prefix sum across all blocks to get output offsets
    size_t* h_output_offsets = (size_t*)malloc(num_blocks * sizeof(size_t));
    if (!h_output_offsets) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate output_offsets array"); return 0; }

    size_t stream_offset = 0; ci = 0;
    for (size_t i = 0; i < num_blocks; ++i) {
        h_output_offsets[i] = stream_offset;
        size_t bsize;
        if (blocks[i].is_compressed) {
            cl_uint s = h_sizes_out[ci++];
            if (s == 0xFFFFFFFFu) { // error from kernel
                size_t failed_ci = (ci > 0) ? (ci - 1) : 0;
                fprintf(stderr, "DEBUG: size-only kernel reported error for global block %zu (comp_idx=%zu), comp_offset=%llu comp_size=%llu\n",
                        i, failed_ci,
                        (unsigned long long)h_comp_offsets[failed_ci], (unsigned long long)h_comp_sizes[failed_ci]);
                clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
                lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "GPU reported decompression error in size-only pass");
                return 0;
            }
            bsize = (size_t)s;
        } else {
            bsize = blocks[i].compressed_size; // raw block
        }
        if (stream_offset + bsize > output_capacity) {
            clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
            free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets);
            lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Output buffer too small for decompressed stream");
            return 0;
        }
        stream_offset += bsize;
    }

    // Prepare per-compressed-block output offsets and max sizes (use the exact sizes we measured)
    cl_ulong* h_out_offsets  = (cl_ulong*)malloc(ncb * sizeof(cl_ulong));
    cl_ulong* h_max_outsizes = (cl_ulong*)malloc(ncb * sizeof(cl_ulong));
    if (!h_out_offsets || !h_max_outsizes) {
        clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
        free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets);
        if (h_out_offsets) free(h_out_offsets);
        if (h_max_outsizes) free(h_max_outsizes);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate per-block arrays");
        return 0;
    }

    ci = 0;
    for (size_t i = 0; i < num_blocks; ++i) {
        if (blocks[i].is_compressed) {
            h_out_offsets[ci]  = (cl_ulong)h_output_offsets[i];
            h_max_outsizes[ci] = (cl_ulong)h_sizes_out[ci]; // exact size as max cap
            ci++;
        }
    }

    // Pass 2: copy raw blocks device->device, and launch parallel decompression for compressed blocks
    // Device buffers for pass 2
    cl_mem d_out_offsets   = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,  ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create out_offsets buffer"); return 0; }
    cl_mem d_max_outsizes  = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,  ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create max_outsizes buffer"); return 0; }
    cl_mem d_sizes_out2    = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE, ncb * sizeof(cl_uint),  NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create sizes_out2 buffer"); return 0; }

    // Upload per-block outputs
    err  = clEnqueueWriteBuffer(compressor->queue, d_out_offsets,  CL_TRUE, 0, ncb * sizeof(cl_ulong), h_out_offsets,  0, NULL, NULL);
    err |= clEnqueueWriteBuffer(compressor->queue, d_max_outsizes, CL_TRUE, 0, ncb * sizeof(cl_ulong), h_max_outsizes, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload out arrays"); return 0; }

    // Copy raw blocks device->device (uncompressed blocks) so the decompression
    // kernel (`k_multi`) can write only the compressed blocks in pass 2.
    for (size_t i = 0; i < num_blocks; ++i) {
        if (!blocks[i].is_compressed) {
            size_t sz = blocks[i].compressed_size;
            size_t src_off = (size_t)blocks[i].compressed_offset;
            size_t dst_off = (size_t)h_output_offsets[i];
            if (sz > 0) {
                /* Wait for input upload to finish (upload_ev) before copying raw blocks */
                cl_event local_waits[2]; cl_uint local_wait_count = 0;
                if (upload_ev) local_waits[local_wait_count++] = upload_ev;
                if (offsets_ev) local_waits[local_wait_count++] = offsets_ev;
                if (compressor->enable_profiling) {
                    err = clEnqueueCopyBuffer(compressor->queue, compressor->input_buffer, compressor->output_buffer, src_off, dst_off, sz, local_wait_count, (local_wait_count ? local_waits : NULL), &copy_ev);
                } else {
                    err = clEnqueueCopyBuffer(compressor->queue, compressor->input_buffer, compressor->output_buffer, src_off, dst_off, sz, local_wait_count, (local_wait_count ? local_waits : NULL), NULL);
                }
                if (err != CL_SUCCESS) {
                    clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
                    clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2);
                    free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes);
                    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to enqueue device->device copy for uncompressed block");
                    return 0;
                }
            }
        }
    }

    /* Create multi-block kernel and measure setup time for multi-block kernel args and enqueue */
    k_multi = clCreateKernel(compressor->program, "lz4_decompress_blocks", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create multi-block decompression kernel"); return 0; }
    /* k_multi_ev is declared earlier; will be used to collect kernel profiling info when enabled */
    t_setup_start = clock();
    err  = clSetKernelArg(k_multi, 0, sizeof(cl_mem), &compressor->input_buffer);
    err |= clSetKernelArg(k_multi, 1, sizeof(cl_mem), &compressor->output_buffer);
    err |= clSetKernelArg(k_multi, 2, sizeof(cl_mem), &d_comp_offsets);
    err |= clSetKernelArg(k_multi, 3, sizeof(cl_mem), &d_comp_sizes);
    err |= clSetKernelArg(k_multi, 4, sizeof(cl_mem), &d_out_offsets);
    err |= clSetKernelArg(k_multi, 5, sizeof(cl_mem), &d_max_outsizes);
    err |= clSetKernelArg(k_multi, 6, sizeof(cl_mem), &d_sizes_out2);
    err |= clSetKernelArg(k_multi, 7, sizeof(cl_uint), &ncb_u32);
    if (err != CL_SUCCESS) { clReleaseKernel(k_multi); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set multi-block kernel args"); return 0; }

    gws = ncb;
    lws = compressor->default_local;
    if (lws == 0) lws = 1;
    if (compressor->device_max_work_group_size > 0 && lws > compressor->device_max_work_group_size) lws = compressor->device_max_work_group_size;
    if (lws > gws) lws = gws;
    gws_rounded = gws;
    if (lws > 0 && (gws % lws) != 0) gws_rounded = ((gws + lws - 1) / lws) * lws;
    {
        cl_event wait_list_local[4]; cl_uint num_waits = 0;
        if (upload_ev) wait_list_local[num_waits++] = upload_ev;
        if (offsets_ev) wait_list_local[num_waits++] = offsets_ev;
        if (sizes_ev) wait_list_local[num_waits++] = sizes_ev;
        if (compressor->enable_profiling) {
            err = clEnqueueNDRangeKernel(compressor->queue, k_multi, 1, NULL, &gws_rounded, &lws, num_waits, (num_waits ? wait_list_local : NULL), &k_multi_ev);
        } else {
            err = clEnqueueNDRangeKernel(compressor->queue, k_multi, 1, NULL, &gws_rounded, &lws, num_waits, (num_waits ? wait_list_local : NULL), NULL);
        }
    }
    t_setup_end = clock(); setup_ms += (double)(t_setup_end - t_setup_start) / CLOCKS_PER_SEC * 1000.0;
    if (err != CL_SUCCESS) { clReleaseKernel(k_multi); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch multi-block kernel"); return 0; }

    // Wait for completion
    if (compressor->enable_profiling) {
        // don't clFinish here; we'll read and wait on events in the D2H read when needed
    } else {
        t_kernel_host_start = clock();
        err = clFinish(compressor->queue);
        t_kernel_host_end = clock();
        kernel_ms_host += (double)(t_kernel_host_end - t_kernel_host_start) / CLOCKS_PER_SEC * 1000.0;
    }
    if (err != CL_SUCCESS) { clReleaseKernel(k_multi); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to wait for multi-block completion"); return 0; }

    clReleaseKernel(k_multi);

    // Bulk download the entire decompressed stream
    if (stream_offset > 0) {
        void* d2h_dest_ptr = (void*)output;
        if (compressor->use_pinned_memory && compressor->pinned_output_ptr && compressor->pinned_output_size >= stream_offset) {
            d2h_dest_ptr = compressor->pinned_output_ptr;
        }
        if (compressor->enable_profiling) {
            err = clEnqueueReadBuffer(compressor->queue, compressor->output_buffer, CL_TRUE, 0, stream_offset, (unsigned char*)d2h_dest_ptr, 0, NULL, &read_stream_ev);
        } else {
            t_d2h_start = clock();
            err = clEnqueueReadBuffer(compressor->queue, compressor->output_buffer, CL_TRUE, 0, stream_offset, (unsigned char*)d2h_dest_ptr, 0, NULL, NULL);
            t_d2h_end = clock(); d2h_ms += (double)(t_d2h_end - t_d2h_start) / CLOCKS_PER_SEC * 1000.0;
        }
        if (d2h_dest_ptr != (void*)output) {
            memcpy(output, d2h_dest_ptr, stream_offset);
        }
        if (err != CL_SUCCESS) {

            clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
            clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2);
            free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes);
            lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download decompressed stream");
            return 0;
        }
    }

    // Cleanup


    clReleaseMemObject(d_comp_offsets);
    clReleaseMemObject(d_comp_sizes);
    clReleaseMemObject(d_sizes_out);
    clReleaseMemObject(d_out_offsets);
    clReleaseMemObject(d_max_outsizes);
    clReleaseMemObject(d_sizes_out2);

    free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes);

    HDEBUG("DEBUG: Frame decompression completed successfully (parallel), total output: %zu bytes\n", stream_offset);

    /* Finalize timing values and populate last_timing */
    if (compressor->enable_profiling) {
        cl_ulong s = 0, e = 0;
        /* size-only kernel event */
        if (k_size_ev) {
            if (clGetEventProfilingInfo(k_size_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(k_size_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                double ms = (double)(e - s) / 1e6;
                kernel_ms_dev += ms;
                event_profile_ms += ms;
            }
        }
        /* size read event */
        if (read_sizes_ev) {
            cl_ulong rs = 0, re = 0;
            if (clGetEventProfilingInfo(read_sizes_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &rs, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(read_sizes_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &re, NULL) == CL_SUCCESS) {
                double ms = (double)(re - rs) / 1e6;
                d2h_ms += ms;
                event_profile_ms += ms;
            }
        }
        /* multi-block kernel event */
        if (k_multi_ev) {
            if (clGetEventProfilingInfo(k_multi_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(k_multi_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                double ms = (double)(e - s) / 1e6;
                kernel_ms_dev += ms;
                event_profile_ms += ms;
            }
        }
        /* final read event */
        if (read_stream_ev) {
            cl_ulong rs = 0, re = 0;
            if (clGetEventProfilingInfo(read_stream_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &rs, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(read_stream_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &re, NULL) == CL_SUCCESS) {
                double ms = (double)(re - rs) / 1e6;
                d2h_ms += ms;
                event_profile_ms += ms;
            }
        }
        /* H2D array uploads events */
        if (upload_ev) {
            if (clGetEventProfilingInfo(upload_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(upload_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                double ms = (double)(e - s) / 1e6;
                h2d_ms += ms; event_profile_ms += ms;
            }
        }
        if (offsets_ev) {
            if (clGetEventProfilingInfo(offsets_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(offsets_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                double ms = (double)(e - s) / 1e6;
                h2d_ms += ms; event_profile_ms += ms;
            }
        }
        if (sizes_ev) {
            if (clGetEventProfilingInfo(sizes_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(sizes_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                double ms = (double)(e - s) / 1e6;
                h2d_ms += ms; event_profile_ms += ms;
            }
        }
        if (copy_ev) {
            if (clGetEventProfilingInfo(copy_ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(copy_ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL) == CL_SUCCESS) {
                double ms = (double)(e - s) / 1e6;
                event_profile_ms += ms;
            }
        }
    } else {
        /* Ensure kernel_ms_host is set by prior clFinish calls */
    }

    clock_t tend = clock();
    double total_ms = (double)(tend - tstart) / CLOCKS_PER_SEC * 1000.0;
    compressor->last_timing.total_ms = total_ms;
    compressor->last_timing.alloc_ms = alloc_ms;
    compressor->last_timing.h2d_ms = h2d_ms;
    compressor->last_timing.setup_ms = setup_ms;
    compressor->last_timing.kernel_ms_device = (kernel_ms_dev > 0.0) ? kernel_ms_dev : -1.0;
    compressor->last_timing.kernel_ms = (kernel_ms_dev > 0.0) ? kernel_ms_dev : kernel_ms_host;
    compressor->last_timing.event_profile_ms = event_profile_ms;
    compressor->last_timing.d2h_ms = d2h_ms;
    compressor->last_timing.frame_ms = frame_ms;
    compressor->last_timing.map_ms = 0.0;
    compressor->last_timing.input_bytes = input_size;
    compressor->last_timing.output_bytes = stream_offset;
    compressor->last_timing.num_blocks = (int)num_blocks;

    /* Release events if present */
    if (k_size_ev) clReleaseEvent(k_size_ev);
    if (read_sizes_ev) clReleaseEvent(read_sizes_ev);
    if (k_multi_ev) clReleaseEvent(k_multi_ev);
    if (read_stream_ev) clReleaseEvent(read_stream_ev);
    if (upload_ev) clReleaseEvent(upload_ev);
    if (offsets_ev) clReleaseEvent(offsets_ev);
    if (sizes_ev) clReleaseEvent(sizes_ev);
    if (copy_ev) clReleaseEvent(copy_ev);

    free(blocks);
    return stream_offset;
}

// Estimate decompressed stream size by running the size-only kernel path.
// This mirrors the size-only pass in lz4_gpu_decompress_frame but does not
// allocate or write the final output buffer. Returns required stream size
// on success, or 0 on error (and sets compressor->last_error).
size_t lz4_gpu_estimate_decompressed_size(LZ4GPUCompressor* compressor,
                                          const void* input, size_t input_size) {
    if (!compressor || !input || input_size == 0) {
        if (compressor) lz4_gpu_set_error(compressor, LZ4_GPU_INVALID_PARAMS, "Invalid input parameters");
        return 0;
    }
    if (!compressor->context) {
        lz4_gpu_set_error(compressor, LZ4_GPU_NOT_INITIALIZED, "Compressor not initialized");
        return 0;
    }

    const unsigned char* ip = (const unsigned char*)input;
    const unsigned char* const iend = ip + input_size;

    // Basic frame header checks (magic + FLG + BD)
    if (ip + 4 > iend) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Input too small for header"); return 0; }
    if (LZ4_readLE32(ip) != LZ4F_MAGICNUMBER) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Invalid LZ4 frame magic number"); return 0; }
    ip += 4;
    if (ip + 2 > iend) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Input too small for FLG/BD"); return 0; }
    unsigned char FLG = *ip++;
    unsigned char BD  = *ip++;

    if (FLG & (1 << 3)) {
        if (ip + 8 > iend) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame header too short for content size"); return 0; }
        /* Skip 8-byte content size field in header; contentSize not used in this path */
        ip += 8;
    }
    // skip header checksum
    if (ip + 1 > iend) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame header too short for HC"); return 0; }
    ip++;

    /* Note: blockChecksumFlag currently unused; header indicates per-block checksum presence */
    /* unsigned int blockChecksumFlag = (FLG >> 4) & 1; */
    unsigned int contentChecksumFlag = (FLG >> 2) & 1;

    // compute content checksum/endmark position
    const unsigned char* contentChecksumPos = NULL;
    if (contentChecksumFlag) {
        if ((size_t)(iend - ip) < (LZ4F_CONTENT_CHECKSUM_SIZE + LZ4F_ENDMARK_SIZE)) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame too small for content checksum / endmark"); return 0; }
        contentChecksumPos = iend - (LZ4F_CONTENT_CHECKSUM_SIZE + LZ4F_ENDMARK_SIZE);
    } else {
        if ((size_t)(iend - ip) < LZ4F_ENDMARK_SIZE) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame too small for endmark"); return 0; }
        contentChecksumPos = iend - LZ4F_ENDMARK_SIZE;
    }

    // Parse blocks to collect compressed offsets/sizes and count compressed blocks
    size_t capacity = 256;
    if (capacity < 16) capacity = 16;
    size_t num_blocks = 0;
    size_t num_compressed_blocks = 0;

    // We'll store temporary lists for compressed blocks
    cl_ulong* h_comp_offsets = NULL;
    cl_ulong* h_comp_sizes = NULL;

    while (ip < contentChecksumPos) {
        if (ip + LZ4F_BLOCK_HEADER_SIZE > iend) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Block header truncated"); goto fail; }
        unsigned int blockHeader = LZ4_readLE32(ip); ip += LZ4F_BLOCK_HEADER_SIZE;
        unsigned int blockSize = blockHeader & 0x7FFFFFFFU;
        int isCompressed = !(blockHeader & LZ4F_BLOCKUNCOMPRESSED_FLAG);
        if (blockSize == 0) break; // end mark
        if (ip + blockSize > iend) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Block data truncated or blockSize invalid"); goto fail; }
        // if compressed, append to arrays
        if (isCompressed) {
            cl_ulong* new_offsets = (cl_ulong*)realloc(h_comp_offsets, (num_compressed_blocks+1)*sizeof(cl_ulong));
            cl_ulong* new_sizes   = (cl_ulong*)realloc(h_comp_sizes,   (num_compressed_blocks+1)*sizeof(cl_ulong));
            if (!new_offsets || !new_sizes) { lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate comp arrays"); goto fail; }
            h_comp_offsets = new_offsets; h_comp_sizes = new_sizes;
            h_comp_offsets[num_compressed_blocks] = (cl_ulong)(ip - (const unsigned char*)input);
            h_comp_sizes[num_compressed_blocks] = (cl_ulong)blockSize;
            num_compressed_blocks++;
        }
        ip += blockSize;
        num_blocks++;
    }

    if (num_blocks == 0) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "No blocks to process"); goto fail; }

    // If no compressed blocks, total size is sum of raw block sizes (we can compute)
    if (num_compressed_blocks == 0) {
        // Need to reparse to compute total (we can reuse stored data via pointer arithmetic but simpler to reparse)
        const unsigned char* qp = (const unsigned char*)input + 4; // after magic
        qp += 2; // FLG BD
        if (FLG & (1<<3)) qp += 8;
        qp += 1; // skip HC
        size_t total = 0;
        while (qp < contentChecksumPos) {
            unsigned int bh = LZ4_readLE32(qp); qp += 4;
            unsigned int bsz = bh & 0x7FFFFFFFU;
            if (bsz == 0) break;
            total += bsz;
            qp += bsz;
        }
        if (h_comp_offsets) free(h_comp_offsets);
        if (h_comp_sizes) free(h_comp_sizes);
        return total;
    }

    // Run size-only kernel to compute per-compressed-block decompressed sizes
    cl_int err;

    // Use persistent input buffer
    if (!ensure_input_buffer(compressor, input_size)) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create input buffer");
        goto fail;
    }

    // Upload input
    err = clEnqueueWriteBuffer(compressor->queue, compressor->input_buffer, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload input data"); goto fail; }

    size_t ncb = num_compressed_blocks;
    cl_mem d_comp_offsets = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create comp_offsets buffer"); goto fail; }
    cl_mem d_comp_sizes   = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create comp_sizes buffer"); goto fail; }
    cl_mem d_sizes_out    = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE, ncb * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create sizes_out buffer"); goto fail; }

    // Upload comp arrays
    err = clEnqueueWriteBuffer(compressor->queue, d_comp_offsets, CL_TRUE, 0, ncb * sizeof(cl_ulong), h_comp_offsets, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(compressor->queue, d_comp_sizes,   CL_TRUE, 0, ncb * sizeof(cl_ulong), h_comp_sizes,   0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload comp arrays"); goto fail; }

    // Create size-only kernel
    cl_kernel k_size = clCreateKernel(compressor->program, "lz4_decompress_blocks_sizeonly", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create size-only kernel"); goto fail; }

    // Determine max_block_out_size from BD
    unsigned char blockSizeID = (BD >> 4) & 0x07;
    cl_ulong max_block_out_size = 64ULL * 1024ULL;
    if (blockSizeID == 5) max_block_out_size = 256ULL * 1024ULL;
    else if (blockSizeID == 6) max_block_out_size = 1024ULL * 1024ULL;
    else if (blockSizeID == 7) max_block_out_size = 4ULL * 1024ULL * 1024ULL;

    cl_uint ncb_u32 = (cl_uint)ncb;
    err  = clSetKernelArg(k_size, 0, sizeof(cl_mem), &compressor->input_buffer);
    err |= clSetKernelArg(k_size, 1, sizeof(cl_mem), &d_comp_offsets);
    err |= clSetKernelArg(k_size, 2, sizeof(cl_mem), &d_comp_sizes);
    err |= clSetKernelArg(k_size, 3, sizeof(cl_ulong), &max_block_out_size);
    err |= clSetKernelArg(k_size, 4, sizeof(cl_mem), &d_sizes_out);
    err |= clSetKernelArg(k_size, 5, sizeof(cl_uint), &ncb_u32);
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set size-only kernel args"); goto fail; }

    size_t gws = ncb;
    size_t lws = compressor->default_local;
    if (lws == 0) lws = 1;
    if (compressor->device_max_work_group_size > 0 && lws > compressor->device_max_work_group_size) lws = compressor->device_max_work_group_size;
    if (lws > gws) lws = gws;
    size_t gws_rounded = gws;
    if (lws > 0 && (gws % lws) != 0) gws_rounded = ((gws + lws - 1) / lws) * lws;
    err = clEnqueueNDRangeKernel(compressor->queue, k_size, 1, NULL, &gws_rounded, &lws, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch size-only kernel"); goto fail; }

    // Read back sizes
    cl_uint* h_sizes_out = (cl_uint*)malloc(ncb * sizeof(cl_uint));
    if (!h_sizes_out) { clReleaseKernel(k_size); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate sizes array"); goto fail; }
    err = clEnqueueReadBuffer(compressor->queue, d_sizes_out, CL_TRUE, 0, ncb * sizeof(cl_uint), h_sizes_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) { free(h_sizes_out); clReleaseKernel(k_size); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to read sizes_out"); goto fail; }

    clReleaseKernel(k_size);

    // Sum sizes to compute stream_offset
    size_t stream_offset = 0;
    size_t ci = 0;
    const unsigned char* qp = (const unsigned char*)input + 4; // reparse from header
    qp += 2; // FLG BD
    if (FLG & (1<<3)) qp += 8;
    qp += 1; // skip HC
    for (size_t i = 0; i < num_blocks; ++i) {
        size_t bsize;
        unsigned int blockHeader = LZ4_readLE32(qp); qp += LZ4F_BLOCK_HEADER_SIZE;
        unsigned int blockSize = blockHeader & 0x7FFFFFFFU;
        int isCompressed = !(blockHeader & LZ4F_BLOCKUNCOMPRESSED_FLAG);
        if (isCompressed) {
            cl_uint s = h_sizes_out[ci++];
            if (s == 0xFFFFFFFFu) {
                size_t failed_ci = (ci > 0) ? (ci - 1) : 0;
                fprintf(stderr, "DEBUG: size-only kernel reported error for global block %zu (comp_idx=%zu), comp_offset=%llu comp_size=%llu\n",
                        i, failed_ci,
                        (unsigned long long)h_comp_offsets[failed_ci], (unsigned long long)h_comp_sizes[failed_ci]);
                free(h_sizes_out);
                clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
                lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "GPU reported decompression error in size-only pass"); goto fail;
            }
            bsize = (size_t)s;
            qp += blockSize;
        } else {
            bsize = blockSize;
            qp += blockSize;
        }
        stream_offset += bsize;
    }

    free(h_sizes_out);
    clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
    if (h_comp_offsets) free(h_comp_offsets);
    if (h_comp_sizes) free(h_comp_sizes);
    return stream_offset;

fail:
    if (h_comp_offsets) free(h_comp_offsets);
    if (h_comp_sizes) free(h_comp_sizes);
    return 0;
}

// Decompress single LZ4 block using GPU - Fixed kernel signature
size_t lz4_gpu_decompress_block(LZ4GPUCompressor* compressor,
                                 const void* compressed_block, size_t compressed_size,
                                 void* output, size_t max_output_size) {
    if (!compressor || !compressed_block || !output || compressed_size == 0) {
        if (compressor) {
            lz4_gpu_set_error(compressor, LZ4_GPU_INVALID_PARAMS, "Invalid input parameters");
        }
        return 0;
    }

    if (!compressor->context) {
        lz4_gpu_set_error(compressor, LZ4_GPU_NOT_INITIALIZED, "Compressor not initialized");
        return 0;
    }

    cl_int err;

    // Create buffers
    cl_mem_flags comp_in_flags = CL_MEM_READ_ONLY;
    if (compressor->use_pinned_memory) comp_in_flags |= CL_MEM_ALLOC_HOST_PTR;
    cl_mem compressed_block_buffer = clCreateBuffer(compressor->context, comp_in_flags,
                                                    compressed_size, NULL, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create compressed block buffer");
        return 0;
    }

    // Use persistent output buffer
    if (!ensure_output_buffer(compressor, max_output_size)) {
        clReleaseMemObject(compressed_block_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output buffer");
        return 0;
    }

    cl_mem_flags outsz_flags = CL_MEM_READ_WRITE;
    if (compressor->use_pinned_memory) outsz_flags |= CL_MEM_ALLOC_HOST_PTR;
    cl_mem output_sizes_buffer = clCreateBuffer(compressor->context, outsz_flags,
                                               sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);

        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output size buffer");
        return 0;
    }

    // Upload compressed block data
    err = clEnqueueWriteBuffer(compressor->queue, compressed_block_buffer, CL_TRUE, 0,
                               compressed_size, compressed_block, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);

        clReleaseMemObject(output_sizes_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload compressed block");
        return 0;
    }

    // Set kernel arguments - Fixed order to match kernel signature
    cl_ulong compressed_offset_u64 = (cl_ulong)0;
    cl_ulong compressed_size_u64 = (cl_ulong)compressed_size;
    cl_ulong output_offset_u64 = (cl_ulong)0;
    cl_ulong max_output_size_u64 = (cl_ulong)max_output_size;
    cl_uint block_index_u32 = (cl_uint)0;

    // Kernel signature: lz4_decompress_block(const __global uchar* input, __global uchar* output,
    //      unsigned long compressed_offset, unsigned long compressed_size,
    //      unsigned long output_offset, unsigned long max_output_size,
    //      __global uint* output_sizes, uint block_index)
    err = clSetKernelArg(compressor->decompress_kernel, 0, sizeof(cl_mem), &compressed_block_buffer);  // input
    err |= clSetKernelArg(compressor->decompress_kernel, 1, sizeof(cl_mem), &compressor->output_buffer);            // output
    err |= clSetKernelArg(compressor->decompress_kernel, 2, sizeof(cl_ulong), &compressed_offset_u64);  // compressed_offset
    err |= clSetKernelArg(compressor->decompress_kernel, 3, sizeof(cl_ulong), &compressed_size_u64);    // compressed_size
    err |= clSetKernelArg(compressor->decompress_kernel, 4, sizeof(cl_ulong), &output_offset_u64);      // output_offset
    err |= clSetKernelArg(compressor->decompress_kernel, 5, sizeof(cl_ulong), &max_output_size_u64);    // max_output_size
    err |= clSetKernelArg(compressor->decompress_kernel, 6, sizeof(cl_mem), &output_sizes_buffer);      // output_sizes
    err |= clSetKernelArg(compressor->decompress_kernel, 7, sizeof(cl_uint), &block_index_u32);         // block_index

    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);

        clReleaseMemObject(output_sizes_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set kernel arguments");
        return 0;
    }

    // Launch kernel
    size_t work_size = 1; // global size (1 block)
    size_t local = compressor->default_local;
    if (local == 0) local = 1;
    if (compressor->device_max_work_group_size > 0 && local > compressor->device_max_work_group_size) local = compressor->device_max_work_group_size;
    if (local > work_size) local = work_size;
    size_t global_rounded = work_size;
    if (local > 0 && (work_size % local) != 0) global_rounded = ((work_size + local - 1) / local) * local;
    err = clEnqueueNDRangeKernel(compressor->queue, compressor->decompress_kernel, 1, NULL,
                                 &global_rounded, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);

        clReleaseMemObject(output_sizes_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch decompression kernel");
        return 0;
    }

    // Download output size
    cl_uint actual_output_size;
    err = clEnqueueReadBuffer(compressor->queue, output_sizes_buffer, CL_TRUE, 0,
                               sizeof(cl_uint), &actual_output_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);

        clReleaseMemObject(output_sizes_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download output size");
        return 0;
    }

    // Download decompressed data
    if (actual_output_size <= max_output_size) {
        void* d2h_dest_ptr = (void*)output;
        if (compressor->use_pinned_memory && compressor->pinned_output_ptr && compressor->pinned_output_size >= actual_output_size) {
            d2h_dest_ptr = compressor->pinned_output_ptr;
        }
        err = clEnqueueReadBuffer(compressor->queue, compressor->output_buffer, CL_TRUE, 0,
                                   actual_output_size, d2h_dest_ptr, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            actual_output_size = 0;
            lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download decompressed data");
        }
        if (d2h_dest_ptr != (void*)output) {
            memcpy(output, d2h_dest_ptr, actual_output_size);
        }
    } else {
        actual_output_size = 0;
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Decompressed data too large for output buffer");
    }

    // Cleanup
    clReleaseMemObject(compressed_block_buffer);

    clReleaseMemObject(output_sizes_buffer);

    return actual_output_size;
}

/* runtime host debug control (no-op when LZ4_GPU_HOST_DEBUG not compiled in) */
void lz4_gpu_set_host_debug(LZ4GPUCompressor* compressor, int enabled) {
    (void)compressor; (void)enabled; /* placeholder, compile-time HDEBUG controls printing */
}

/* Return the last timing summary filled by the compressor. */
int lz4_gpu_get_last_timing(LZ4GPUCompressor* compressor, LZ4GPUTiming* timing) {
    if (!compressor || !timing) return 0;
    *timing = compressor->last_timing;
    return 1;
}

/* Print a compact timing summary for CLI use */
void lz4_gpu_print_timing(const LZ4GPUTiming* timing) {
    if (!timing || timing->total_ms == 0.0) { printf("No timing data available\n"); return; }
    printf("=== LZ4 GPU Performance Timing ===\n");
    printf("  Total:    %8.3f ms\n", timing->total_ms);
    printf("  Alloc:    %8.3f ms\n", timing->alloc_ms);
    printf("  H2D:      %8.3f ms\n", timing->h2d_ms);
    if (timing->kernel_ms >= 0.0) printf("  Kernel:   %8.3f ms\n", timing->kernel_ms);
    if (timing->kernel_ms_device > 0.0) printf("  Kernel(dev): %8.3f ms\n", timing->kernel_ms_device);
    printf("  D2H:      %8.3f ms\n", timing->d2h_ms);
    printf("  Frame:    %8.3f ms\n", timing->frame_ms);
}
