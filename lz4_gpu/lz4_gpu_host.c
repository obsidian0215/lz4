/*
 * LZ4 GPU Compressor Host Implementation (C version)
 * Pure C implementation for C projects
 */

#include "lz4_gpu_host.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
    // Default: keep logs quiet and vector IO off unless enabled by user
    compressor->enable_vector_io = 0;
    compressor->enable_kernel_debug = 0;
    compressor->prefer_precompiled = 0;
    compressor->precompiled_path[0] = '\0';
    /* By default profiling is disabled; user may enable via API or CLI (-p/--profile) */
    compressor->enable_profiling = 0;

    return compressor;
}

void lz4_gpu_destroy_compressor(LZ4GPUCompressor* compressor) {
    if (!compressor) return;

    if (compressor->compress_kernel) clReleaseKernel(compressor->compress_kernel);
    if (compressor->decompress_kernel) clReleaseKernel(compressor->decompress_kernel);
    if (compressor->frame_decompress_kernel) clReleaseKernel(compressor->frame_decompress_kernel);

    if (compressor->input_buffer) clReleaseMemObject(compressor->input_buffer);
    if (compressor->output_buffer) clReleaseMemObject(compressor->output_buffer);

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
    return compressor->error_message;
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
    /* Profiling event placeholders used when compressor->enable_profiling == 1 */
    cl_event upload_ev = NULL;
    cl_event offsets_ev = NULL;
    cl_event outoffs_ev = NULL;
    cl_event kernel_ev = NULL;

    // 1) Optional: try precompiled binary if configured
    if (compressor->prefer_precompiled && compressor->precompiled_path[0]) {
        const char* path = compressor->precompiled_path;
        // Heuristic check: ensure filename matches expected vectorization if user cares
        // If enable_vector_io=1 but filename doesn't contain "vec", warn and skip binary
        int skip_binary = 0;
        const char* fname = strrchr(path, '\\');
        if (!fname) fname = strrchr(path, '/');
        if (!fname) fname = path; else fname++;
        if (compressor->enable_vector_io) {
            if (!strstr(fname, "vec")) {
                HDEBUG("DEBUG: Precompiled binary '%s' doesn't look vectorized; skipping to source build\n", fname);
                skip_binary = 1;
            }
        } else {
            if (strstr(fname, "vec")) {
                HDEBUG("DEBUG: Precompiled binary '%s' seems vectorized but vector_io disabled; skipping to source build\n", fname);
                skip_binary = 1;
            }
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
                                HDEBUG("DEBUG: Precompiled binary validated successfully\n");
                                free(bdata);
                                // Proceed to create real kernels later below
                                goto create_kernels;
                            } else {
                                HDEBUG("DEBUG: Precompiled binary missing required kernels; falling back to source build\n");
                                clReleaseProgram(compressor->program); compressor->program = NULL; err = CL_INVALID_PROGRAM;
                            }
                        } else {
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
                HDEBUG("DEBUG: Could not open precompiled binary at '%s'\n", path);
            }
        }
    }

    // 2) Load kernel source and build with options
    size_t kernel_size = 0;
    char* kernel_source = load_kernel_source("lz4_gpu.cl", &kernel_size);
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
    if (compressor->enable_vector_io) {
        strcat(build_opts, " -DLZ4_GPU_VECTOR_IO=1");
    }
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
                        const char* fname = compressor->enable_vector_io ? "lz4_gpu_vec.clbin" : "lz4_gpu.clbin";
                        FILE* fbin = fopen(fname, "wb");
                        if (fbin) {
                            size_t written = fwrite(binaries[0], 1, binSizes[0], fbin);
                            fclose(fbin);
                            HDEBUG("DEBUG: Exported OpenCL binary to '%s' (%zu bytes written)\n", fname, written);
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

    return 1;
}

int lz4_gpu_initialize(LZ4GPUCompressor* compressor) {
    cl_int err;

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
    // Set a sane default dynamic block size (can be adjusted per-input)
    compressor->dynamic_block_size = LZ4_GPU_MAX_BLOCK_SIZE;
    HDEBUG("DEBUG: Device compute units=%u, max_work_group_size=%zu, default_block_size=%zu\n",
           (unsigned int)compressor->device_compute_units, compressor->device_max_work_group_size, compressor->dynamic_block_size);

    // Create context
    compressor->context = clCreateContext(NULL, 1, &compressor->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_CONTEXT_ERROR, "Failed to create OpenCL context");
        return 0;
    }

    // Create command queue (enable profiling if requested)
    cl_command_queue_properties qprops = 0;
    if (compressor->enable_profiling) qprops |= CL_QUEUE_PROFILING_ENABLE;
    compressor->queue = clCreateCommandQueue(compressor->context, compressor->device, qprops, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_QUEUE_ERROR, "Failed to create command queue");
        return 0;
    }

    // Load and build program with current build options
    if (!lz4_gpu_build_program_with_options(compressor)) {
        return 0;
    }

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

    compressor->input_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                             compressor->input_buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create input buffer");
        return 0;
    }

    compressor->output_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE,
                                              compressor->output_buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output buffer");
        return 0;
    }

    // Note: We don't create block_offsets_buffer and block_sizes_buffer here anymore
    // They are created dynamically in compression functions as needed
    return 1;
}

// Public setters for build options and rebuild API
void lz4_gpu_set_vector_io(LZ4GPUCompressor* compressor, int enabled) {
    if (!compressor) return;
    compressor->enable_vector_io = enabled ? 1 : 0;
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


static void split_into_blocks(LZ4GPUCompressor* compressor, const void* input, size_t input_size,
                              cl_uint* block_offsets, size_t* num_blocks) {
    // Use dynamic block size if available
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

// Compute a dynamic block size based on device capabilities and input size.
// Goal: create at least a few blocks per compute unit to keep the GPU busy,
// but keep block sizes between 64KB and 1MB.
static size_t compute_dynamic_block_size(LZ4GPUCompressor* compressor, size_t input_size) {
    // Adjusted block size range based on device variability and user feedback:
    // make MIN_BLOCK smaller (4KB) to better handle small inputs / finer-grained parallelism,
    // and MAX_BLOCK larger (8MB) for high-throughput devices.
    const size_t MIN_BLOCK = 4 * 1024; // 4KB
    const size_t MAX_BLOCK = 4 * 1024 * 1024; // 4MB (LZ4 frame spec maximum)

    cl_uint cu = compressor->device_compute_units ? compressor->device_compute_units : 1;
    // Aim for 4 blocks per compute unit to provide scheduling flexibility
    size_t target_blocks = (size_t)cu * 4;
    if (target_blocks == 0) target_blocks = 1;

    size_t block_size = (input_size + target_blocks - 1) / target_blocks;

    // Round up to multiple of MIN_BLOCK for alignment and predictable memory layout
    if (block_size < MIN_BLOCK) block_size = MIN_BLOCK;
    else {
        block_size = ((block_size + MIN_BLOCK - 1) / MIN_BLOCK) * MIN_BLOCK;
    }

    if (block_size > MAX_BLOCK) block_size = MAX_BLOCK;

    return block_size;
}

// Compute a smaller dynamic block size when acceleration > 1.
// Strategy: start from base size, then divide by min(acceleration, 8).
// Keep alignment to LZ4_GPU_MIN_BLOCK_SIZE and within [MIN_BLOCK, MAX_BLOCK].
static size_t compute_dynamic_block_size_accel(LZ4GPUCompressor* compressor, size_t input_size, int acceleration) {
    // Start from base sizing tuned for device
    size_t base = compute_dynamic_block_size(compressor, input_size);
    if (acceleration <= 1) return base;

    // Gentler downscale: at most 4x smaller, roughly log2-based steps
    // acc=1 -> x1, acc>=4 -> x2, acc>=8 -> x3, acc>=16 -> x4
    size_t scale;
    if (acceleration >= 16) scale = 4;
    else if (acceleration >= 8) scale = 3;
    else if (acceleration >= 4) scale = 2;
    else scale = 1;

    size_t block_size = (scale > 1) ? (base / scale) : base;

    // Compression path minimum block size: keep >=256KB to amortize per-block overhead better
    const size_t MIN_BLOCK = 256 * 1024; // override parsing min (4KB) for compression throughput
    const size_t MAX_BLOCK = 4 * 1024 * 1024; // LZ4 frame spec max

    if (block_size < MIN_BLOCK) block_size = MIN_BLOCK;
    // round up to multiple of 4KB for alignment and predictable layout
    const size_t ALIGN = 4 * 1024;
    block_size = ((block_size + ALIGN - 1) / ALIGN) * ALIGN;
    if (block_size > MAX_BLOCK) block_size = MAX_BLOCK;

    // Additionally, avoid creating an excessive number of blocks: cap by compute units
    if (compressor && compressor->device_compute_units) {
        size_t cu = compressor->device_compute_units;
    size_t maxBlocks = cu * 4; // heuristic: up to 4 blocks per CU for higher throughput
        size_t estBlocks = (input_size + block_size - 1) / block_size;
        if (estBlocks > maxBlocks && maxBlocks > 0) {
            block_size = (input_size + maxBlocks - 1) / maxBlocks;
            // align again
            block_size = ((block_size + ALIGN - 1) / ALIGN) * ALIGN;
            if (block_size < MIN_BLOCK) block_size = MIN_BLOCK;
            if (block_size > MAX_BLOCK) block_size = MAX_BLOCK;
        }
    }

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

size_t lz4_gpu_compress_frame(LZ4GPUCompressor* compressor, const void* input, size_t input_size, void* output, size_t output_capacity) { return lz4_gpu_compress_frame_accelerated(compressor, input, input_size, output, output_capacity, 1); }
size_t lz4_gpu_compress_frame_accelerated(LZ4GPUCompressor* compressor,
                                    const void* input, size_t input_size,
                                    void* output, size_t output_capacity,
                                    int acceleration) {
HDEBUG("DEBUG: lz4_gpu_compress_frame_accelerated called with input_size=%zu, acceleration=%d\n",
       input_size, acceleration);

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
if (acceleration > 65537) acceleration = 65537;

HDEBUG("DEBUG: Using acceleration=%d (clamped to valid range)\n", acceleration);

cl_int err;
    /* Accelerated per-block max output sizes buffer (created later) */
    cl_mem max_output_sizes_buffer = NULL;

// Determine dynamic block size (smaller when acceleration > 1) and split input into blocks
compressor->dynamic_block_size = compute_dynamic_block_size_accel(compressor, input_size, acceleration);
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

// Create temporary buffers for parallel operation
cl_mem input_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                    input_size, NULL, &err);
if (err != CL_SUCCESS) {
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create input buffer");
    return 0;
}

    /* (max_output_sizes_buffer will be created after output_offsets are uploaded) */

    /* (max_output_sizes_buffer will be created after output_offsets are uploaded) */

// Create buffer for block offsets (offset, size pairs)
cl_mem block_offsets_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                             sizeof(cl_uint) * num_blocks * 2, NULL, &err);
if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buffer);
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create block offsets buffer");
    return 0;
}

// Create buffer for output offsets
cl_mem output_offsets_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                              sizeof(cl_uint) * num_blocks, NULL, &err);
if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(block_offsets_buffer);
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output offsets buffer");
    return 0;
}

// Upload input data
cl_event upload_ev = NULL;
if (compressor->enable_profiling) {
    err = clEnqueueWriteBuffer(compressor->queue, input_buffer, CL_FALSE, 0,
                               input_size, input, 0, NULL, &upload_ev);
} else {
    err = clEnqueueWriteBuffer(compressor->queue, input_buffer, CL_TRUE, 0,
                               input_size, input, 0, NULL, NULL);
}
if (err != CL_SUCCESS) {
    if (upload_ev) { clReleaseEvent(upload_ev); }
    clReleaseMemObject(input_buffer);
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
    clReleaseMemObject(input_buffer);
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Output buffer too small for GPU processing");
    return 0;
}

// Create output buffer for compressed data
cl_mem output_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE,
                                      output_capacity, NULL, &err);
if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buffer);
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output buffer");
    return 0;
}

// Upload block offsets
cl_event offsets_ev = NULL;
if (compressor->enable_profiling) {
    err = clEnqueueWriteBuffer(compressor->queue, block_offsets_buffer, CL_FALSE, 0,
                               sizeof(cl_uint) * num_blocks * 2, block_offsets, 0, NULL, &offsets_ev);
} else {
    err = clEnqueueWriteBuffer(compressor->queue, block_offsets_buffer, CL_TRUE, 0,
                               sizeof(cl_uint) * num_blocks * 2, block_offsets, 0, NULL, NULL);
}
if (err != CL_SUCCESS) {
    if (upload_ev) { clReleaseEvent(upload_ev); }
    if (offsets_ev) { clReleaseEvent(offsets_ev); }
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(output_offsets_buffer);
    lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload block offsets");
    return 0;
}

// Create buffer to store compressed sizes
cl_mem compressed_sizes_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE,
                                                sizeof(cl_uint) * num_blocks, NULL, &err);
if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create compressed sizes buffer");
    return 0;
}

// Calculate output offsets for each block
cl_uint* output_offsets = (cl_uint*)malloc(num_blocks * sizeof(cl_uint));
    /* Per-block maximum output size for accelerated kernel */
    cl_uint* max_output_sizes = (cl_uint*)malloc(num_blocks * sizeof(cl_uint));
if (!output_offsets || !max_output_sizes) {
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate output offsets buffer");
    if (output_offsets) free(output_offsets);
    if (max_output_sizes) free(max_output_sizes);
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
cl_event outoffs_ev = NULL;
if (compressor->enable_profiling) {
    err = clEnqueueWriteBuffer(compressor->queue, output_offsets_buffer, CL_FALSE, 0,
                               sizeof(cl_uint) * num_blocks, output_offsets, 0, NULL, &outoffs_ev);
} else {
    err = clEnqueueWriteBuffer(compressor->queue, output_offsets_buffer, CL_TRUE, 0,
                               sizeof(cl_uint) * num_blocks, output_offsets, 0, NULL, NULL);
}
if (err != CL_SUCCESS) {
    if (upload_ev) { clReleaseEvent(upload_ev); }
    if (offsets_ev) { clReleaseEvent(offsets_ev); }
    if (outoffs_ev) { clReleaseEvent(outoffs_ev); }
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(compressed_sizes_buffer);
    clReleaseMemObject(block_offsets_buffer);
    lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload output offsets");
    return 0;
}

// Set kernel arguments for accelerated compression kernel
int table_type = 1; // Use byU32 table type for better performance
cl_uint num_blocks_u32 = (cl_uint)num_blocks;
cl_uint input_size_u32 = (cl_uint)input_size;

HDEBUG("DEBUG: Setting kernel args for accelerated compression: num_blocks=%u, input_size=%u, table_type=%d, acceleration=%d\n",
       num_blocks_u32, input_size_u32, table_type, acceleration);

// Set kernel arguments for the accelerated kernel
// Kernel signature: lz4_compress_block_accelerated(input, output, blockSizes, blockOffsets, outputOffsets, totalBlocks, inputSize, tableType, acceleration)
err = clSetKernelArg(compressor->compress_kernel, 0, sizeof(cl_mem), &input_buffer);              // input
err |= clSetKernelArg(compressor->compress_kernel, 1, sizeof(cl_mem), &output_buffer);            // output
err |= clSetKernelArg(compressor->compress_kernel, 2, sizeof(cl_mem), &compressed_sizes_buffer);  // blockSizes
err |= clSetKernelArg(compressor->compress_kernel, 3, sizeof(cl_mem), &block_offsets_buffer);      // blockOffsets
err |= clSetKernelArg(compressor->compress_kernel, 4, sizeof(cl_mem), &output_offsets_buffer);     // outputOffsets
err |= clSetKernelArg(compressor->compress_kernel, 5, sizeof(cl_mem), &max_output_sizes_buffer);   // maxOutputSizes
err |= clSetKernelArg(compressor->compress_kernel, 6, sizeof(cl_uint), &num_blocks_u32);           // totalBlocks
err |= clSetKernelArg(compressor->compress_kernel, 7, sizeof(cl_uint), &input_size_u32);           // inputSize
err |= clSetKernelArg(compressor->compress_kernel, 8, sizeof(int), &table_type);                  // tableType
err |= clSetKernelArg(compressor->compress_kernel, 9, sizeof(int), &acceleration);               // acceleration parameter
/* Allocate per-work-group local hash table */
size_t local_table_bytes_accel = ((size_t)1 << LZ4_GPU_HASH_LOG) * sizeof(cl_uint);
err |= clSetKernelArg(compressor->compress_kernel, 10, local_table_bytes_accel, NULL);

if (err != CL_SUCCESS) {
    printf("DEBUG: clSetKernelArg failed with error %d\n", err);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(compressed_sizes_buffer);
    clReleaseMemObject(block_offsets_buffer);
    clReleaseMemObject(output_offsets_buffer);
    if (max_output_sizes_buffer) clReleaseMemObject(max_output_sizes_buffer);
    free(output_offsets);
    free(max_output_sizes);
    lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set accelerated kernel arguments");
    return 0;
}

// Launch parallel compression kernel
// Use one work-item per block to avoid local memory sharing issues between blocks
size_t global_work_size = num_blocks;
size_t local_work_size = 1; // single work-item per block
HDEBUG("DEBUG: Launching accelerated compression kernel with global_size=%zu, local_size=%zu, acceleration=%d\n",
    global_work_size, local_work_size, acceleration);

// Launch parallel compression kernel
cl_event kernel_ev = NULL;
{
    const cl_uint num_waits = (compressor->enable_profiling && upload_ev) ? 1u : 0u;
    const cl_event wait_list[1] = { upload_ev };
    err = clEnqueueNDRangeKernel(compressor->queue, compressor->compress_kernel, 1, NULL,
                     &global_work_size, &local_work_size,
                     num_waits, (num_waits ? wait_list : NULL),
                     (compressor->enable_profiling ? &kernel_ev : NULL));
}
if (err != CL_SUCCESS) {
    printf("DEBUG: Accelerated kernel launch failed with error %d\n", err);
    if (upload_ev) clReleaseEvent(upload_ev);
    if (offsets_ev) clReleaseEvent(offsets_ev);
    if (outoffs_ev) clReleaseEvent(outoffs_ev);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(compressed_sizes_buffer);
    clReleaseMemObject(block_offsets_buffer);
    clReleaseMemObject(output_offsets_buffer);
    if (max_output_sizes_buffer) clReleaseMemObject(max_output_sizes_buffer);
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
    err = clFinish(compressor->queue);
    if (err != CL_SUCCESS) {
        printf("DEBUG: clFinish failed with error %d\n", err);
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(compressed_sizes_buffer);
        clReleaseMemObject(block_offsets_buffer);
        clReleaseMemObject(output_offsets_buffer);
        if (max_output_sizes_buffer) clReleaseMemObject(max_output_sizes_buffer);
        free(output_offsets);
        free(max_output_sizes);
        lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to wait for accelerated kernel completion");
        return 0;
    }
    HDEBUG("DEBUG: Accelerated kernel completed\n");
}

// Download compressed sizes
cl_uint* compressed_sizes = (cl_uint*)malloc(num_blocks * sizeof(cl_uint));
if (!compressed_sizes) {
    printf("ERROR: Failed to allocate compressed_sizes array for %zu blocks\n", num_blocks);
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate compressed sizes array");
    return 0;
}

err = clEnqueueReadBuffer(compressor->queue, compressed_sizes_buffer, CL_TRUE, 0,
                           sizeof(cl_uint) * num_blocks, compressed_sizes, 0, NULL, NULL);
if (err != CL_SUCCESS) {
    printf("DEBUG: Failed to download compressed sizes, error %d\n", err);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(compressed_sizes_buffer);
    clReleaseMemObject(block_offsets_buffer);
    clReleaseMemObject(output_offsets_buffer);
    if (max_output_sizes_buffer) clReleaseMemObject(max_output_sizes_buffer);
    free(output_offsets);
    free(max_output_sizes);
    lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download compressed sizes");
    return 0;
}

// Debug: Print compressed sizes for each block
HDEBUG("DEBUG: Compressed sizes for %zu blocks (acceleration=%d):\n", num_blocks, acceleration);
size_t total_compressed_size = LZ4F_HEADER_SIZE_MAX;
for (size_t i = 0; i < num_blocks; i++) {
    size_t block_uncompressed = (size_t)block_offsets[i * 2 + 1];
    size_t csz = (size_t)compressed_sizes[i];
    total_compressed_size += LZ4F_BLOCK_HEADER_SIZE;
    if (csz == 0 || csz >= block_uncompressed) {
        total_compressed_size += block_uncompressed;
    } else {
        total_compressed_size += csz;
    }
}
total_compressed_size += LZ4F_CONTENT_CHECKSUM_SIZE + LZ4F_ENDMARK_SIZE;

HDEBUG("DEBUG: Total compressed size calculation: header=%d, blocks=%d, content_checksum=%d, endmark=%d\n",
       LZ4F_HEADER_SIZE_MAX, (int)(num_blocks * LZ4F_BLOCK_HEADER_SIZE),
       LZ4F_CONTENT_CHECKSUM_SIZE, LZ4F_ENDMARK_SIZE);

if (total_compressed_size > output_capacity) {
    printf("DEBUG: Total compressed size %zu exceeds output capacity %zu\n", total_compressed_size, output_capacity);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(compressed_sizes_buffer);
    clReleaseMemObject(block_offsets_buffer);
    clReleaseMemObject(output_offsets_buffer);
    if (max_output_sizes_buffer) clReleaseMemObject(max_output_sizes_buffer);
    free(output_offsets);
    free(max_output_sizes);
    lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Total compressed data too large for output buffer");
    return 0;
}

// Build final LZ4 frame (same as original function)
unsigned char* frame_output = (unsigned char*)output;
size_t frame_pos = 0;

// Write frame header manually (following LZ4F_compressBegin_internal)
unsigned int magic = LZ4F_MAGICNUMBER;
unsigned char* header_start;
unsigned char FLG, BD, HC;
unsigned long long content_size = (unsigned long long)input_size;

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
        size_t block_start = (size_t)block_offsets[i * 2];
        size_t block_uncompressed = (size_t)block_offsets[i * 2 + 1];
        size_t csz = (size_t)compressed_sizes[i];
        if (csz == 0 || csz >= block_uncompressed) {
            unsigned int header = (unsigned int)block_uncompressed | 0x80000000U;
            LZ4_writeLE32(frame_output + frame_pos, header);
            frame_pos += LZ4F_BLOCK_HEADER_SIZE;
            HDEBUG("DEBUG: Block %zu stored uncompressed: %zu bytes at frame offset %zu (input offset %zu)\n",
                   i, block_uncompressed, frame_pos, block_start);
            memcpy(frame_output + frame_pos, (const unsigned char*)input + block_start, block_uncompressed);
            frame_pos += block_uncompressed;
        } else {
            LZ4_writeLE32(frame_output + frame_pos, (unsigned int)csz);
            frame_pos += LZ4F_BLOCK_HEADER_SIZE;
            HDEBUG("DEBUG: Copying compressed block %zu: size=%zu, from GPU offset %u to frame offset %zu\n",
                   i, csz, output_offsets[i], frame_pos);

            err = clEnqueueReadBuffer(compressor->queue, output_buffer, CL_TRUE,
                                     (size_t)output_offsets[i], csz, frame_output + frame_pos, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                printf("DEBUG: Failed to download compressed block %zu, error %d\n", i, err);
                clReleaseMemObject(input_buffer);
                clReleaseMemObject(output_buffer);
                clReleaseMemObject(compressed_sizes_buffer);
                clReleaseMemObject(block_offsets_buffer);
                clReleaseMemObject(output_offsets_buffer);
                if (max_output_sizes_buffer) clReleaseMemObject(max_output_sizes_buffer);
                free(output_offsets);
                free(max_output_sizes);
                lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download compressed block data");
                return 0;
            }

            frame_pos += csz;
        }
    }

// Write end mark (4 zero bytes) - this MUST be the last part before checksum
LZ4_writeLE32(frame_output + frame_pos, 0);
frame_pos += LZ4F_ENDMARK_SIZE;

// Add content checksum AFTER end mark if enabled (XXH32 hash)
if ((FLG >> 2) & 1) {
    // Calculate proper XXH32 hash of input data
    unsigned int checksum = XXH32(input, input_size, 0);
    LZ4_writeLE32(frame_output + frame_pos, checksum);
    frame_pos += LZ4F_CONTENT_CHECKSUM_SIZE;
    HDEBUG("DEBUG: Added content checksum after end mark: 0x%08x\n", checksum);
}

HDEBUG("DEBUG: Final frame size: %zu bytes (acceleration=%d)\n", frame_pos, acceleration);

// Cleanup
clReleaseMemObject(input_buffer);
clReleaseMemObject(output_buffer);
clReleaseMemObject(compressed_sizes_buffer);
clReleaseMemObject(block_offsets_buffer);
clReleaseMemObject(output_offsets_buffer);
    /* Free the accelerated per-block max_output_sizes resources */
    if (max_output_sizes_buffer) clReleaseMemObject(max_output_sizes_buffer);

// Free dynamically allocated block offsets
free(block_offsets);
free(output_offsets);
free(max_output_sizes);

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

    /*
     * Calculate maximum possible blocks for dynamic allocation.
     * Important: do NOT rely on compressor->dynamic_block_size here because the
     * compressor instance's dynamic_block_size reflects a runtime tuning value
     * used during compression and may not match the block layout in the frame
     * being parsed (or may be the default). Using it can truncate parsing
     * (e.g. if dynamic_block_size > actual block size used when compressing).
     *
     * Use a conservative minimum block size for parsing (4KB) to compute an
     * upper bound for number of blocks. This avoids missing blocks while
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
    size_t contentChecksumPos = 0;
    if (contentChecksumFlag) {
        if (iend < LZ4F_CONTENT_CHECKSUM_SIZE + LZ4F_ENDMARK_SIZE) {
            free(blocks);
            lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame too small for content checksum / endmark");
            return 0;
        }
        contentChecksumPos = iend - LZ4F_CONTENT_CHECKSUM_SIZE - LZ4F_ENDMARK_SIZE;
    } else {
        if (iend < LZ4F_ENDMARK_SIZE) {
            free(blocks);
            lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame too small for endmark");
            return 0;
        }
        contentChecksumPos = iend - LZ4F_ENDMARK_SIZE;
    }

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

    // Create buffers for GPU processing
    cl_mem input_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                         input_size, NULL, &err);
    if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create input buffer"); return 0; }

    cl_mem output_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE,
                                          output_capacity, NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output buffer"); return 0; }

    // Upload input frame once
    err = clEnqueueWriteBuffer(compressor->queue, input_buffer, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload input data"); return 0; }

    // Prepare arrays for compressed blocks
    size_t ncb = num_compressed_blocks;
    cl_ulong* h_comp_offsets = (cl_ulong*)malloc(ncb * sizeof(cl_ulong));
    cl_ulong* h_comp_sizes   = (cl_ulong*)malloc(ncb * sizeof(cl_ulong));
    cl_uint*  h_sizes_out    = (cl_uint*)malloc(ncb * sizeof(cl_uint));
    size_t*   h_comp_block_index = (size_t*)malloc(ncb * sizeof(size_t)); // map compressed idx -> global block idx
    if (!h_comp_offsets || !h_comp_sizes || !h_sizes_out || !h_comp_block_index) {
        clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer);
        if (h_comp_offsets) free(h_comp_offsets); if (h_comp_sizes) free(h_comp_sizes);
        if (h_sizes_out) free(h_sizes_out); if (h_comp_block_index) free(h_comp_block_index);
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
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create comp_offsets buffer"); return 0; }
    cl_mem d_comp_sizes   = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create comp_sizes buffer"); return 0; }
    cl_mem d_sizes_out    = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE, ncb * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create sizes_out buffer"); return 0; }

    // Upload arrays
    err  = clEnqueueWriteBuffer(compressor->queue, d_comp_offsets, CL_TRUE, 0, ncb * sizeof(cl_ulong), h_comp_offsets, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(compressor->queue, d_comp_sizes,   CL_TRUE, 0, ncb * sizeof(cl_ulong), h_comp_sizes,   0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload comp arrays"); return 0; }

    // Create and run size-only kernel
    cl_kernel k_size = clCreateKernel(compressor->program, "lz4_decompress_blocks_sizeonly", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create size-only kernel"); return 0; }

    cl_uint ncb_u32 = (cl_uint)ncb;
    err  = clSetKernelArg(k_size, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(k_size, 1, sizeof(cl_mem), &d_comp_offsets);
    err |= clSetKernelArg(k_size, 2, sizeof(cl_mem), &d_comp_sizes);
    err |= clSetKernelArg(k_size, 3, sizeof(cl_ulong), &max_block_out_size);
    err |= clSetKernelArg(k_size, 4, sizeof(cl_mem), &d_sizes_out);
    err |= clSetKernelArg(k_size, 5, sizeof(cl_uint), &ncb_u32);
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set size-only kernel args"); return 0; }

    size_t gws = ncb; size_t lws = 1;
    err = clEnqueueNDRangeKernel(compressor->queue, k_size, 1, NULL, &gws, &lws, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch size-only kernel"); return 0; }

    // Read back sizes
    err = clEnqueueReadBuffer(compressor->queue, d_sizes_out, CL_TRUE, 0, ncb * sizeof(cl_uint), h_sizes_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to read sizes_out"); return 0; }

    clReleaseKernel(k_size);

    // Compute prefix sum across all blocks to get output offsets
    size_t* h_output_offsets = (size_t*)malloc(num_blocks * sizeof(size_t));
    if (!h_output_offsets) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate output_offsets array"); return 0; }

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
                clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
                free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets);
                lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "GPU reported decompression error in size-only pass");
                return 0;
            }
            bsize = (size_t)s;
        } else {
            bsize = blocks[i].compressed_size; // raw block
        }
        if (stream_offset + bsize > output_capacity) {
            clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
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
        clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
        free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets);
        if (h_out_offsets) free(h_out_offsets); if (h_max_outsizes) free(h_max_outsizes);
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
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create out_offsets buffer"); return 0; }
    cl_mem d_max_outsizes  = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,  ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create max_outsizes buffer"); return 0; }
    cl_mem d_sizes_out2    = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE, ncb * sizeof(cl_uint),  NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create sizes_out2 buffer"); return 0; }

    // Upload per-block outputs
    err  = clEnqueueWriteBuffer(compressor->queue, d_out_offsets,  CL_TRUE, 0, ncb * sizeof(cl_ulong), h_out_offsets,  0, NULL, NULL);
    err |= clEnqueueWriteBuffer(compressor->queue, d_max_outsizes, CL_TRUE, 0, ncb * sizeof(cl_ulong), h_max_outsizes, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload out arrays"); return 0; }

    // Copy raw blocks device->device
    for (size_t i = 0; i < num_blocks; ++i) {
        if (!blocks[i].is_compressed) {
            size_t sz = blocks[i].compressed_size;
            size_t src_off = blocks[i].compressed_offset;
            size_t dst_off = h_output_offsets[i];
            if (sz) {
                err = clEnqueueCopyBuffer(compressor->queue, input_buffer, output_buffer, src_off, dst_off, sz, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer);
                    clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
                    clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2);
                    free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes);
                    lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to device-copy raw block");
                    return 0;
                }
            }
        }
    }

    // Create and run multi-block decompression kernel
    cl_kernel k_multi = clCreateKernel(compressor->program, "lz4_decompress_blocks", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create multi-block kernel"); return 0; }

    err  = clSetKernelArg(k_multi, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(k_multi, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(k_multi, 2, sizeof(cl_mem), &d_comp_offsets);
    err |= clSetKernelArg(k_multi, 3, sizeof(cl_mem), &d_comp_sizes);
    err |= clSetKernelArg(k_multi, 4, sizeof(cl_mem), &d_out_offsets);
    err |= clSetKernelArg(k_multi, 5, sizeof(cl_mem), &d_max_outsizes);
    err |= clSetKernelArg(k_multi, 6, sizeof(cl_mem), &d_sizes_out2);
    err |= clSetKernelArg(k_multi, 7, sizeof(cl_uint), &ncb_u32);
    if (err != CL_SUCCESS) { clReleaseKernel(k_multi); clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set multi-block kernel args"); return 0; }

    gws = ncb; lws = 1;
    err = clEnqueueNDRangeKernel(compressor->queue, k_multi, 1, NULL, &gws, &lws, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k_multi); clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch multi-block kernel"); return 0; }

    // Wait for completion
    err = clFinish(compressor->queue);
    if (err != CL_SUCCESS) { clReleaseKernel(k_multi); clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2); free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to wait for multi-block completion"); return 0; }

    clReleaseKernel(k_multi);

    // Bulk download the entire decompressed stream
    if (stream_offset > 0) {
        err = clEnqueueReadBuffer(compressor->queue, output_buffer, CL_TRUE, 0, stream_offset, (unsigned char*)output, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(input_buffer); clReleaseMemObject(output_buffer);
            clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
            clReleaseMemObject(d_out_offsets); clReleaseMemObject(d_max_outsizes); clReleaseMemObject(d_sizes_out2);
            free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes);
            lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download decompressed stream");
            return 0;
        }
    }

    // Cleanup
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(d_comp_offsets);
    clReleaseMemObject(d_comp_sizes);
    clReleaseMemObject(d_sizes_out);
    clReleaseMemObject(d_out_offsets);
    clReleaseMemObject(d_max_outsizes);
    clReleaseMemObject(d_sizes_out2);

    free(h_comp_offsets); free(h_comp_sizes); free(h_sizes_out); free(h_comp_block_index); free(h_output_offsets); free(h_out_offsets); free(h_max_outsizes);

    HDEBUG("DEBUG: Frame decompression completed successfully (parallel), total output: %zu bytes\n", stream_offset);
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

    size_t contentSize = 0;
    if (FLG & (1 << 3)) {
        if (ip + 8 > iend) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame header too short for content size"); return 0; }
        contentSize = (size_t)LZ4_readLE32(ip) | ((size_t)LZ4_readLE32(ip + 4) << 32);
        ip += 8;
    }
    // skip header checksum
    if (ip + 1 > iend) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame header too short for HC"); return 0; }
    ip++;

    unsigned int blockChecksumFlag = (FLG >> 4) & 1;
    unsigned int contentChecksumFlag = (FLG >> 2) & 1;

    // compute content checksum/endmark position
    size_t contentChecksumPos = 0;
    if (contentChecksumFlag) {
        if (iend < LZ4F_CONTENT_CHECKSUM_SIZE + LZ4F_ENDMARK_SIZE) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame too small for content checksum / endmark"); return 0; }
        contentChecksumPos = iend - LZ4F_CONTENT_CHECKSUM_SIZE - LZ4F_ENDMARK_SIZE;
    } else {
        if (iend < LZ4F_ENDMARK_SIZE) { lz4_gpu_set_error(compressor, LZ4_GPU_DECOMPRESS_ERROR, "Frame too small for endmark"); return 0; }
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
        if (FLG & (1<<3)) qp += 8; qp += 1; // skip HC
        size_t total = 0;
        while (qp < contentChecksumPos) {
            unsigned int bh = LZ4_readLE32(qp); qp += 4;
            unsigned int bsz = bh & 0x7FFFFFFFU;
            if (bsz == 0) break;
            total += bsz;
            qp += bsz;
        }
        if (h_comp_offsets) free(h_comp_offsets); if (h_comp_sizes) free(h_comp_sizes);
        return total;
    }

    // Run size-only kernel to compute per-compressed-block decompressed sizes
    cl_int err;
    cl_mem input_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, input_size, NULL, &err);
    if (err != CL_SUCCESS) { lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create input buffer"); goto fail; }

    // Upload input
    err = clEnqueueWriteBuffer(compressor->queue, input_buffer, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload input data"); goto fail; }

    size_t ncb = num_compressed_blocks;
    cl_mem d_comp_offsets = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create comp_offsets buffer"); goto fail; }
    cl_mem d_comp_sizes   = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY, ncb * sizeof(cl_ulong), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create comp_sizes buffer"); goto fail; }
    cl_mem d_sizes_out    = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE, ncb * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create sizes_out buffer"); goto fail; }

    // Upload comp arrays
    err = clEnqueueWriteBuffer(compressor->queue, d_comp_offsets, CL_TRUE, 0, ncb * sizeof(cl_ulong), h_comp_offsets, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(compressor->queue, d_comp_sizes,   CL_TRUE, 0, ncb * sizeof(cl_ulong), h_comp_sizes,   0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_UPLOAD_ERROR, "Failed to upload comp arrays"); goto fail; }

    // Create size-only kernel
    cl_kernel k_size = clCreateKernel(compressor->program, "lz4_decompress_blocks_sizeonly", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ERROR, "Failed to create size-only kernel"); goto fail; }

    // Determine max_block_out_size from BD
    unsigned char blockSizeID = (BD >> 4) & 0x07;
    cl_ulong max_block_out_size = 64ULL * 1024ULL;
    if (blockSizeID == 5) max_block_out_size = 256ULL * 1024ULL;
    else if (blockSizeID == 6) max_block_out_size = 1024ULL * 1024ULL;
    else if (blockSizeID == 7) max_block_out_size = 4ULL * 1024ULL * 1024ULL;

    cl_uint ncb_u32 = (cl_uint)ncb;
    err  = clSetKernelArg(k_size, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(k_size, 1, sizeof(cl_mem), &d_comp_offsets);
    err |= clSetKernelArg(k_size, 2, sizeof(cl_mem), &d_comp_sizes);
    err |= clSetKernelArg(k_size, 3, sizeof(cl_ulong), &max_block_out_size);
    err |= clSetKernelArg(k_size, 4, sizeof(cl_mem), &d_sizes_out);
    err |= clSetKernelArg(k_size, 5, sizeof(cl_uint), &ncb_u32);
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set size-only kernel args"); goto fail; }

    size_t gws = ncb; size_t lws = 1;
    err = clEnqueueNDRangeKernel(compressor->queue, k_size, 1, NULL, &gws, &lws, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k_size); clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_LAUNCH_ERROR, "Failed to launch size-only kernel"); goto fail; }

    // Read back sizes
    cl_uint* h_sizes_out = (cl_uint*)malloc(ncb * sizeof(cl_uint));
    if (!h_sizes_out) { clReleaseKernel(k_size); clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to allocate sizes array"); goto fail; }
    err = clEnqueueReadBuffer(compressor->queue, d_sizes_out, CL_TRUE, 0, ncb * sizeof(cl_uint), h_sizes_out, 0, NULL, NULL);
    if (err != CL_SUCCESS) { free(h_sizes_out); clReleaseKernel(k_size); clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out); lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to read sizes_out"); goto fail; }

    clReleaseKernel(k_size);

    // Sum sizes to compute stream_offset
    size_t stream_offset = 0;
    size_t ci = 0;
    const unsigned char* qp = (const unsigned char*)input + 4; // reparse from header
    qp += 2; if (FLG & (1<<3)) qp += 8; qp += 1; // skip HC
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
                clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
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
    clReleaseMemObject(input_buffer); clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_sizes_out);
    if (h_comp_offsets) free(h_comp_offsets); if (h_comp_sizes) free(h_comp_sizes);
    return stream_offset;

fail:
    if (h_comp_offsets) free(h_comp_offsets); if (h_comp_sizes) free(h_comp_sizes);
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
    cl_mem compressed_block_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_ONLY,
                                                    compressed_size, NULL, &err);
    if (err != CL_SUCCESS) {
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create compressed block buffer");
        return 0;
    }

    cl_mem output_buffer = clCreateBuffer(compressor->context, CL_MEM_WRITE_ONLY,
                                         max_output_size, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output buffer");
        return 0;
    }

    cl_mem output_sizes_buffer = clCreateBuffer(compressor->context, CL_MEM_READ_WRITE,
                                               sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);
        clReleaseMemObject(output_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_ERROR, "Failed to create output size buffer");
        return 0;
    }

    // Upload compressed block data
    err = clEnqueueWriteBuffer(compressor->queue, compressed_block_buffer, CL_TRUE, 0,
                               compressed_size, compressed_block, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);
        clReleaseMemObject(output_buffer);
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
    err |= clSetKernelArg(compressor->decompress_kernel, 1, sizeof(cl_mem), &output_buffer);            // output
    err |= clSetKernelArg(compressor->decompress_kernel, 2, sizeof(cl_ulong), &compressed_offset_u64);  // compressed_offset
    err |= clSetKernelArg(compressor->decompress_kernel, 3, sizeof(cl_ulong), &compressed_size_u64);    // compressed_size
    err |= clSetKernelArg(compressor->decompress_kernel, 4, sizeof(cl_ulong), &output_offset_u64);      // output_offset
    err |= clSetKernelArg(compressor->decompress_kernel, 5, sizeof(cl_ulong), &max_output_size_u64);    // max_output_size
    err |= clSetKernelArg(compressor->decompress_kernel, 6, sizeof(cl_mem), &output_sizes_buffer);      // output_sizes
    err |= clSetKernelArg(compressor->decompress_kernel, 7, sizeof(cl_uint), &block_index_u32);         // block_index

    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(output_sizes_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_KERNEL_ARGS_ERROR, "Failed to set kernel arguments");
        return 0;
    }

    // Launch kernel
    size_t work_size = 1;
    err = clEnqueueNDRangeKernel(compressor->queue, compressor->decompress_kernel, 1, NULL,
                                 &work_size, &work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(compressed_block_buffer);
        clReleaseMemObject(output_buffer);
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
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(output_sizes_buffer);
        lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download output size");
        return 0;
    }

    // Download decompressed data
    if (actual_output_size <= max_output_size) {
        err = clEnqueueReadBuffer(compressor->queue, output_buffer, CL_TRUE, 0,
                                   actual_output_size, output, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            actual_output_size = 0;
            lz4_gpu_set_error(compressor, LZ4_GPU_DOWNLOAD_ERROR, "Failed to download decompressed data");
        }
    } else {
        actual_output_size = 0;
        lz4_gpu_set_error(compressor, LZ4_GPU_BUFFER_TOO_SMALL, "Decompressed data too large for output buffer");
    }

    // Cleanup
    clReleaseMemObject(compressed_block_buffer);
    clReleaseMemObject(output_buffer);
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
