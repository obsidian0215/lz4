/*
 * Minimal OpenCL host test harness for backup_vector lz4_gpu kernels.
 * - Builds from `lz4_gpu.cl` or uses precompiled `lz4_gpu.clbin` if present
 * - Splits input into 64KB blocks, runs lz4_compress_block_accelerated per block
 * - Runs lz4_decompress_blocks to reconstruct original and validates
 * - Reports kernel execution times via OpenCL event profiling
 *
 * Build (Linux):
 *   gcc -O2 gpu_roundtrip_test.c -o gpu_roundtrip_test -lOpenCL
 * Run:
 *   ./gpu_roundtrip_test <input-file>
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <libgen.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <CL/cl.h>
#include <execinfo.h>
#include <signal.h>
#include <fcntl.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (64*1024)
#endif
#define EST_DST_CAPACITY(src) ((src) + (src)/255 + 64)

static char* read_file_text(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, size, f) != (size_t)size) { free(buf); fclose(f); return NULL; }
    buf[size] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)size;
    return buf;
}

static void write_binary_file(const char* path, const unsigned char* data, size_t len) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(data, 1, len, f);
    fclose(f);
}

static void die(const char* msg) { perror(msg); exit(1); }

static void crash_handler(int sig) {
    void *bt[64];
    int n = backtrace(bt, 64);
    fprintf(stderr, "Critical: caught signal %d, printing backtrace (%d frames)\n", sig, n);
    backtrace_symbols_fd(bt, n, STDERR_FILENO);
    _exit(128 + sig);
}

/* Utility: join two paths (dir + name), result allocated with malloc.
 * Caller must free.
 */
static char* join_path(const char* dir, const char* name) {
    if (!dir || !name) return NULL;
    size_t dlen = strlen(dir);
    size_t nlen = strlen(name);
    int need_sep = (dlen > 0 && dir[dlen-1] != '/');
    char* out = malloc(dlen + nlen + (need_sep ? 2 : 1));
    if (!out) return NULL;
    strcpy(out, dir);
    if (need_sep) strcat(out, "/");
    strcat(out, name);
    return out;
}

/* Utility: get the directory of the running executable.
 * Returns a malloc'd string or NULL. Works on Linux via /proc/self/exe.
 */
static char* get_exec_dir(const char* argv0) {
    char buf[PATH_MAX+1];
    ssize_t r = readlink("/proc/self/exe", buf, PATH_MAX);
    if (r <= 0) {
        // fallback to argv0 path
        if (!argv0) return NULL;
        char* tmp = strdup(argv0);
        if (!tmp) return NULL;
        char* d = dirname(tmp);
        char* out = strdup(d);
        free(tmp);
        return out;
    }
    buf[r] = '\0';
    // truncate to directory
    char* tmp = strdup(buf);
    if (!tmp) return NULL;
    char* d = dirname(tmp); // note: dirname modifies tmp
    char* out = strdup(d);
    free(tmp);
    return out;
}

/* Utility: validate file exists and is readable */
static int file_readable(const char* path) {
    if (!path) return 0;
    return (access(path, R_OK) == 0);
}

/* Try to open binary using several candidate paths; prints attempts.
 * Candidates (in order):
 *  - as given (if absolute or contains slash)
 *  - cwd/<name> (if name only)
 *  - exec_dir/<name>
 *  - exec_dir/.. / exec_dir/../.. (try up to 3 parent levels)
 */
static FILE* try_open_bin_candidates(const char* bin_path, const char* exec_dir) {
    if (!bin_path || strlen(bin_path) == 0) return NULL;
    FILE* f = NULL;
    // 1) as-is
    f = fopen(bin_path, "rb");
    if (f) return f;
    // 2) if bin_path contains '/', assume as-is already tried; skip
    if (strchr(bin_path, '/') == NULL) {
        // Try CWD
        f = fopen(bin_path, "rb");
        if (f) return f;
        // Try exec_dir/bin_path
        if (exec_dir) {
            char* p = join_path(exec_dir, bin_path);
            if (p) {
                f = fopen(p, "rb");
                free(p);
                if (f) return f;
            }
            // Try a few parent levels
            char* cur = strdup(exec_dir);
            if (cur) {
                for (int i = 0; i < 3; ++i) {
                    // go up one level
                    char* nd = strdup(cur);
                    if (!nd) break;
                    char* parent = dirname(nd);
                    // join parent/bin_path
                    char* cand = join_path(parent, bin_path);
                    if (cand) {
                        f = fopen(cand, "rb");
                        free(cand);
                        if (f) { free(nd); break; }
                    }
                    free(nd);
                    // try to shorten cur to parent for next iteration
                    char* tmp = strdup(cur);
                    if (!tmp) break;
                    char* pdir = dirname(tmp);
                    free(cur); cur = strdup(pdir);
                    free(tmp);
                    if (!cur) break;
                }
                free(cur);
            }
        }
    }
    return NULL;
}

// Execute build_clbin if present to create a precompiled clbin from source. Returns 1 on success.
static int build_clbin_if_present(const char* exec_dir, const char* source_path, const char* out_clbin, const char* build_flags) {
    // locate build_clbin in exec_dir or in PATH
    char build_clbin_path[PATH_MAX+1]; build_clbin_path[0] = '\0';
    if (exec_dir) {
        snprintf(build_clbin_path, sizeof(build_clbin_path), "%s/build_clbin", exec_dir);
        if (access(build_clbin_path, X_OK) != 0) build_clbin_path[0] = '\0';
    }
    // Also check directory of source_path (if provided)
    if (build_clbin_path[0] == '\0' && source_path) {
        char* tmp = strdup(source_path);
        if (tmp) {
            char* d = dirname(tmp);
            char tpath[PATH_MAX+1]; tpath[0] = '\0';
            snprintf(tpath, sizeof(tpath), "%s/build_clbin", d);
            if (access(tpath, X_OK) == 0) {
                strncpy(build_clbin_path, tpath, sizeof(build_clbin_path)-1);
                build_clbin_path[sizeof(build_clbin_path)-1] = '\0';
            }
            free(tmp);
        }
    }
    if (build_clbin_path[0] == '\0') {
        // try PATH
        const char* fallback = "build_clbin";
        // check if available by running which
        char *which_cmd = NULL;
        size_t space = strlen("which ") + strlen(fallback) + 1;
        which_cmd = malloc(space);
        snprintf(which_cmd, space, "which %s", fallback);
        FILE* pw = popen(which_cmd, "r");
        free(which_cmd);
        if (pw) {
            char buf[PATH_MAX+1];
            if (fgets(buf, sizeof(buf), pw)) {
                // trim newline
                size_t ln = strlen(buf);
                if (ln && buf[ln-1] == '\n') buf[ln-1] = '\0';
                strncpy(build_clbin_path, buf, sizeof(build_clbin_path)-1);
                build_clbin_path[sizeof(build_clbin_path)-1] = '\0';
            }
            pclose(pw);
        }
        if (build_clbin_path[0] == '\0') {
            // No build_clbin binary found. Try to compile build_clbin.c in exec_dir or source_path directory
            char build_src_path[PATH_MAX+1]; build_src_path[0] = '\0';
            if (exec_dir) {
                snprintf(build_src_path, sizeof(build_src_path), "%s/build_clbin.c", exec_dir);
                if (access(build_src_path, R_OK) != 0) build_src_path[0] = '\0';
            }
            if (build_src_path[0] == '\0' && source_path) {
                char* tmp = strdup(source_path);
                if (tmp) {
                    char* d = dirname(tmp);
                    snprintf(build_src_path, sizeof(build_src_path), "%s/build_clbin.c", d);
                    if (access(build_src_path, R_OK) != 0) build_src_path[0] = '\0';
                    free(tmp);
                }
            }
            if (build_src_path[0] == '\0') return 0;
            // compile build_clbin.c to build_clbin in exec_dir
            char build_out[PATH_MAX+1]; build_out[0] = '\0';
            snprintf(build_out, sizeof(build_out), "%s/build_clbin", exec_dir ? exec_dir : ".");
            pid_t cpid = fork();
            if (cpid == 0) {
                // child: exec gcc
                execlp("gcc", "gcc", "-O2", build_src_path, "-o", build_out, "-lOpenCL", (char*)NULL);
                _exit(127);
            }
            int cstatus = 0; waitpid(cpid, &cstatus, 0);
            if (WIFEXITED(cstatus) && WEXITSTATUS(cstatus) == 0) {
                // check generated binary
                if (access(build_out, X_OK) == 0) {
                    strncpy(build_clbin_path, build_out, sizeof(build_clbin_path)-1);
                    build_clbin_path[sizeof(build_clbin_path)-1] = '\0';
                } else {
                    return 0;
                }
            } else {
                return 0;
            }
        }
    }

    // invoke: build_clbin -o <out_clbin> "<build_flags>"
    pid_t pid = fork();
    if (pid == 0) {
        // child; ensure we execute build_clbin in the directory containing the source
        char* d = NULL;
        if (source_path) { char* tmp = strdup(source_path); if (tmp) { d = dirname(tmp); } }
        // build the command safely: cd "<d>" && "<build_clbin_path>" -o "<out_clbin>" '<build_flags>'
        char cmd[PATH_MAX*3]; cmd[0] = '\0';
        if (d) {
            if (build_flags && build_flags[0]) snprintf(cmd, sizeof(cmd), "cd '%s' && '%s' -o '%s' '%s'", d, build_clbin_path, out_clbin, build_flags);
            else snprintf(cmd, sizeof(cmd), "cd '%s' && '%s' -o '%s'", d, build_clbin_path, out_clbin);
        } else {
            if (build_flags && build_flags[0]) snprintf(cmd, sizeof(cmd), "'%s' -o '%s' '%s'", build_clbin_path, out_clbin, build_flags);
            else snprintf(cmd, sizeof(cmd), "'%s' -o '%s'", build_clbin_path, out_clbin);
        }
        execlp("sh", "sh", "-c", cmd, (char*)NULL);
        _exit(127);
    }
    // parent
    int status = 0; waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return 1;
    return 0;
}

// Compile OpenCL source in a sub-process (isolated child) and write out a binary at out_clbin.
// Returns 1 on success, 0 on failure.
static int compile_source_subproc(const char* sourceToRead, const char* out_clbin, const char* build_flags) {
    pid_t pid = fork();
    if (pid == 0) {
        // child: compile source using OpenCL runtime, then write binary to out_clbin
        cl_int err;
        cl_uint numPlatforms = 0; if (clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS || numPlatforms == 0) _exit(2);
        cl_platform_id platform; clGetPlatformIDs(1, &platform, NULL);
        cl_uint numDevices = 0; clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices); if (numDevices == 0) _exit(3);
        cl_device_id device; clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
        cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err); if (err != CL_SUCCESS || !ctx) _exit(4);
        size_t src_len = 0; char* srccl = read_file_text(sourceToRead, &src_len); if (!srccl) _exit(5);
        const char* s = srccl; cl_program program = clCreateProgramWithSource(ctx, 1, &s, &src_len, &err); if (err != CL_SUCCESS) { free(srccl); clReleaseContext(ctx); _exit(6); }
        err = clBuildProgram(program, 1, &device, build_flags && build_flags[0] ? build_flags : NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t logsz = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
            char* log = malloc(logsz+1); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL); log[logsz] = 0;
            fprintf(stderr, "[subproc build] Build log:\n%s\n", log); free(log);
            free(srccl); clReleaseProgram(program); clReleaseContext(ctx); _exit(7);
        }
        size_t num_devices=0; clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_devices, NULL);
        size_t* binary_sizes = malloc(sizeof(size_t) * num_devices);
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*num_devices, binary_sizes, NULL);
        unsigned char** binaries = malloc(sizeof(unsigned char*) * num_devices);
        for (size_t i=0;i<num_devices;i++) binaries[i] = malloc(binary_sizes[i]);
        err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*)*num_devices, binaries, NULL);
        if (err == CL_SUCCESS) {
            FILE* f = fopen(out_clbin, "wb");
            if (!f) {
                // try to write to tmp but fail
                for (size_t i=0;i<num_devices;i++) {
                    free(binaries[i]);
                }
                free(binaries); free(binary_sizes); free(srccl);
                clReleaseProgram(program); clReleaseContext(ctx); _exit(8);
            }
            fwrite(binaries[0], 1, binary_sizes[0], f); fclose(f);
        } else {
            for (size_t i=0;i<num_devices;i++) {
                free(binaries[i]);
            }
            free(binaries); free(binary_sizes); free(srccl);
            clReleaseProgram(program); clReleaseContext(ctx); _exit(9);
        }
        for (size_t i=0;i<num_devices;i++) {
            free(binaries[i]);
        }
        free(binaries); free(binary_sizes); free(srccl);
        clReleaseProgram(program); clReleaseContext(ctx);
        _exit(0);
    }
    // parent wait
    int status = 0; waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    /* Install SIGSEGV handler to capture stack traces when a crash happens */
    struct sigaction sa; memset(&sa, 0, sizeof(sa)); sa.sa_handler = crash_handler; sigemptyset(&sa.sa_mask); sa.sa_flags = SA_RESTART | SA_ONSTACK; sigaction(SIGSEGV, &sa, NULL);
    if (argc < 2) {
        printf("Usage: %s <input-file> [--accel=<n>] [--local=<n>]\n", argv[0]);
        printf("       (ENVIRONMENT VARIABLES)\n");
        printf("         LZ4_GPU_CLBIN : Path to precompiled .clbin file to use (preferred)\n");
        printf("         LZ4_GPU_CLSRC : Path to kernel source file to use when building from source\n");
        printf("  (NOTE) --clbin and --clsrc CLI options removed; set LZ4_GPU_CLBIN/LZ4_GPU_CLSRC instead.\n");
        printf("  --accel     optional integer passed to compression kernel (default: 1)\n");
        printf("  --local     optional local work-group size override\n");
        return 1;
    }

    const char* src_path = argv[1];
    const char* default_bin = "lz4_gpu.clbin"; /* default precompiled binary */
    const char* kernel_bin_path = NULL; /* optional explicit binary path (from env LZ4_GPU_CLBIN) */
    const char* kernel_src_path = NULL; /* optional kernel source (from env LZ4_GPU_CLSRC) */
    int acceleration = 1; /* default acceleration */
    int local_override = 0; /* optional fourth parameter: local work-group size */
    int allow_inproc_build = 0; /* default: disallow in-process clBuildProgram unless explicitly allowed */
    /* Allow flags: --clbin <path> or --clbin=<path>, --clsrc <path>, --accel, --local
     * And support legacy positional args: <input-file> [clbin_path] [acceleration] [local_override]
     */
    char* exec_dir = get_exec_dir(argv[0]);
    /* Read environment variables for kernel binary/source. CLI flags --clbin/--clsrc removed. */
    kernel_bin_path = getenv("LZ4_GPU_CLBIN");
    kernel_src_path = getenv("LZ4_GPU_CLSRC");
    for (int i = 2; i < argc; ++i) {
        const char* a = argv[i];
        if (!a) continue;
        if (strncmp(a, "--accel=", 8) == 0) { acceleration = atoi(a + 8); continue; }
        if (strcmp(a, "--accel") == 0 && i+1 < argc) { acceleration = atoi(argv[++i]); continue; }
        if (strncmp(a, "--local=", 8) == 0) { local_override = atoi(a + 8); continue; }
        if (strcmp(a, "--local") == 0 && i+1 < argc) { local_override = atoi(argv[++i]); continue; }
        /* positional args are only used for filenames or unknown args; do not accept clbin/clsrc as positional */
        const char* ext = strrchr(a, '.');
        if (ext) {
            /* keep detection for .cl/.clbin as a convenience only if passed as a positional (not recommended) */
            if (strcmp(ext, ".cl") == 0) { kernel_src_path = a; continue; }
            if (strcmp(ext, ".clbin") == 0) { kernel_bin_path = a; continue; }
        }
        char* endptr = NULL; long v = strtol(a, &endptr, 10);
        if (endptr != a && *endptr == '\0') {
            if (acceleration == 1 && v > 0) { acceleration = (int)v; continue; }
            else if (local_override == 0 && v > 0) { local_override = (int)v; continue; }
        }
        // New flags to control compile behavior
        if (strcmp(a, "--allow-clsrc-build") == 0) { allow_inproc_build = 1; continue; }
    }
    if (!kernel_bin_path) kernel_bin_path = default_bin;
    // If explicit source path requested, prefer it and skip binary probing
    if (kernel_src_path) {
        kernel_bin_path = NULL;
    }
    size_t src_len = 0;
    FILE* f = fopen(src_path, "rb");
    if (!f) die("fopen");
    fseek(f, 0, SEEK_END);
    src_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* src = malloc(src_len);
    if (!src) die("malloc src");
    if (fread(src, 1, src_len, f) != src_len) die("fread");
    fclose(f);

    // Partition into blocks
    size_t totalBlocks = (src_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Input %zu bytes => %zu blocks\n", src_len, totalBlocks);

    // Prepare blockOffsets (pairs) and outputOffsets
    uint32_t* blockOffsets = calloc(totalBlocks * 2, sizeof(uint32_t));
    uint32_t* outputOffsets = calloc(totalBlocks, sizeof(uint32_t));
    uint32_t* initialOutputOffsets = calloc(totalBlocks, sizeof(uint32_t));
    size_t max_compressed_buffer = 0;
    for (size_t i = 0; i < totalBlocks; ++i) {
        uint32_t start = (uint32_t)(i * BLOCK_SIZE);
        uint32_t sz = (uint32_t)((i == totalBlocks - 1) ? (src_len - i*BLOCK_SIZE) : BLOCK_SIZE);
        blockOffsets[i*2 + 0] = start;
        blockOffsets[i*2 + 1] = sz;
        uint32_t dstCap = (uint32_t)EST_DST_CAPACITY(sz);
        /* Store per-block max output sizes for the compress kernel to use */
        // (max compressed size allocation for each block)
        // Create array on host to pass to kernel as __global maxOutputSizes

        outputOffsets[i] = initialOutputOffsets[i];
        max_compressed_buffer += dstCap;

        /* lazily allocate per-block maxOutputSizes array and fill values */
        /* We'll allocate after loop once total size known */
    }

    printf("Allocating device buffers: compressed buffer approx %zu bytes\n", max_compressed_buffer);

    /* per-block maximum output size array for compression kernel */
    uint32_t* maxOutputSizes = calloc(totalBlocks, sizeof(uint32_t));
    for (size_t i = 0; i < totalBlocks; ++i) {
        uint32_t sz = blockOffsets[i*2 + 1];
        maxOutputSizes[i] = (uint32_t)EST_DST_CAPACITY(sz);
    }

    /* Assign initial output offsets non-overlapping using per-block max sizes */
    uint32_t offset_cursor = 0;
    for (size_t i = 0; i < totalBlocks; ++i) {
        initialOutputOffsets[i] = offset_cursor;
        outputOffsets[i] = initialOutputOffsets[i];
        offset_cursor += maxOutputSizes[i];
    }

    // Setup OpenCL
    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) die("clGetPlatformIDs");
    cl_platform_id* platforms = malloc(sizeof(cl_platform_id) * numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms, NULL);
    cl_platform_id platform = platforms[0];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        // fallback to CPU device
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, NULL, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) die("clGetDeviceIDs");
    }
    cl_device_id* devices = malloc(sizeof(cl_device_id) * numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) die("clCreateContext");
#if defined(CL_VERSION_2_0)
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, (cl_queue_properties)CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    if (err != CL_SUCCESS) die("clCreateCommandQueueWithProperties");
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) die("clCreateCommandQueue");
#endif

    /* Print device info helpful for tuning */
    cl_uint compute_units = 0;
    size_t max_wg = 0;
    /* Effective hashlog used for local hash table memory sizing. 14 by default. */
    int effective_hashlog = 14;
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL) == CL_SUCCESS) {
        /* ok */
    }
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg, NULL) == CL_SUCCESS) {
        /* ok */
    }
    printf("Device compute units: %u, max work-group size: %zu\n", (unsigned int)compute_units, max_wg);

    // Try to load precompiled binary first
    if (argc >= 3) {
        /* kernel_bin_path may have been set above via flags or legacy positional args */
    }
    printf("Attempting to open precompiled binary: %s, source: %s\n", kernel_bin_path ? kernel_bin_path : "<none>", kernel_src_path ? kernel_src_path : "<none>");
    fflush(stdout);
    cl_program program = NULL;
    /* Decide whether the provided path is a source (.cl) or a binary, based on extension.
     * If it ends with .cl, treat it as a source file. Otherwise, attempt binary first.
     */
    int is_source_path = 0;
    if (kernel_bin_path) {
        const char* ext = strrchr(kernel_bin_path, '.');
        if (ext && strcmp(ext, ".cl") == 0) is_source_path = 1;
    }
    FILE* fb = NULL;
    if (!is_source_path && kernel_bin_path) {
        // print candidate list (cwd, exec_dir and parents) for debug
        printf("  Candidate binary path: %s\n", kernel_bin_path);
        if (exec_dir) printf("  Exec dir candidate: %s/%s\n", exec_dir, strrchr(kernel_bin_path, '/') ? strrchr(kernel_bin_path, '/')+1 : kernel_bin_path);
        fflush(stdout);
        fb = try_open_bin_candidates(kernel_bin_path, exec_dir);
        if (!fb) {
            int save_errno = errno;
            printf("Could not open binary '%s' (errno=%d), falling back to cl source (if any)\n", kernel_bin_path, save_errno);
            fflush(stdout);
        }
        if (fb) {
        fseek(fb, 0, SEEK_END);
        long bsize = ftell(fb);
        fseek(fb, 0, SEEK_SET);
        unsigned char* bin = malloc(bsize);
        if (fread(bin, 1, bsize, fb) != (size_t)bsize) die("read bin");
        fclose(fb);
        cl_int bin_status;
        program = clCreateProgramWithBinary(context, 1, &device, (const size_t*)&bsize, (const unsigned char**)&bin, &bin_status, &err);
        free(bin);
            if (err != CL_SUCCESS) die("clCreateProgramWithBinary");
        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            // print build log
            size_t logsz = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
            char* log = malloc(logsz+1); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL); log[logsz]=0;
            printf("Build log:\n%s\n", log); free(log);
            die("clBuildProgram (binary)");
        }
            printf("Loaded program from binary %s\n", kernel_bin_path);
        }
    }
    if (!program) {
        /* Fallback: try to load source from a provided source path (if given) or from CWD's lz4_gpu.cl */
        const char* sourceToRead = NULL;
        char* alloc_sourceToRead = NULL;
        if (is_source_path) sourceToRead = kernel_bin_path;
        else if (kernel_src_path) sourceToRead = kernel_src_path;
        else if (exec_dir) {
            // try exec dir candidate first
            char* cand = join_path(exec_dir, "lz4_gpu.cl");
            if (cand && file_readable(cand)) { alloc_sourceToRead = cand; sourceToRead = alloc_sourceToRead; }
            else if (cand) free(cand);
        }
        if (!sourceToRead) sourceToRead = "lz4_gpu.cl";
        printf("  Source chosen: %s\n", sourceToRead);
        fflush(stdout);
        size_t srccl_len = 0;
        char* srccl = read_file_text(sourceToRead, &srccl_len);
        if (!srccl) {
            char msg[512];
            snprintf(msg, sizeof(msg), "read %s", sourceToRead);
            die(msg);
        }
        printf("  Successfully read source (%zu bytes)\n", srccl_len);
        fflush(stdout);
        // Before building from source, determine device local mem and set build flags
        cl_ulong dev_local_mem = 0;
        if (clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(dev_local_mem), &dev_local_mem, NULL) != CL_SUCCESS) dev_local_mem = 0;
        // default LZ4_HASHLOG is 14; compute max hashlog supported by device local mem
        int default_hashlog = 14;
        int max_hashlog = default_hashlog;
        if (dev_local_mem > 0) {
            size_t max_entries = (size_t)dev_local_mem / sizeof(uint32_t);
            // compute floor(log2(max_entries))
            int candidate = 0;
            while (((size_t)1 << (candidate+1)) <= max_entries) candidate++;
            if (candidate < max_hashlog) max_hashlog = candidate;
        }
        char build_flags[256]; build_flags[0] = '\0';
        if (max_hashlog != default_hashlog) {
            snprintf(build_flags, sizeof(build_flags), "-DLZ4_HASHLOG=%d", max_hashlog);
            printf("Building from source with build flags: %s (device local mem %llu bytes)\n", build_flags, (unsigned long long)dev_local_mem);
        }
        /* Set effective hashlog so later we allocate proper local table size or limit on local mem args */
        effective_hashlog = max_hashlog;
        // If a build_clbin helper is available, prefer generating a binary via build_clbin first
        char tmp_clbin[PATH_MAX] = "";
        // export_path logic copied from above
        char export_path[1024]; export_path[0] = '\0';
        if (kernel_bin_path && strlen(kernel_bin_path) > 0) {
            strncpy(export_path, kernel_bin_path, sizeof(export_path)-1); export_path[sizeof(export_path)-1] = '\0';
        } else if (kernel_src_path && strlen(kernel_src_path) > 0) {
            strncpy(export_path, kernel_src_path, sizeof(export_path)-1); export_path[sizeof(export_path)-1] = '\0';
            char* dot = strrchr(export_path, '.');
            if (dot) strcpy(dot, ".clbin"); else strncat(export_path, ".clbin", sizeof(export_path)-strlen(export_path)-1);
        } else if (exec_dir) {
            char* tmp = join_path(exec_dir, "lz4_gpu.clbin"); if (tmp) { strncpy(export_path, tmp, sizeof(export_path)-1); export_path[sizeof(export_path)-1] = '\0'; free(tmp); }
        } else {
            strncpy(export_path, "lz4_gpu.clbin", sizeof(export_path)-1); export_path[sizeof(export_path)-1] = '\0';
        }
        snprintf(tmp_clbin, sizeof(tmp_clbin), "%s", export_path);
        int built_ok = 0;
        // Prefer using the repository build_clbin helper to prebuild a binary
        if (build_clbin_if_present(exec_dir, sourceToRead, tmp_clbin, build_flags)) {
            // Attempt to load binary produced by build_clbin
            printf("Using build_clbin produced binary %s\n", tmp_clbin);
            FILE* fb2 = try_open_bin_candidates(tmp_clbin, exec_dir);
            if (fb2) {
                fseek(fb2, 0, SEEK_END);
                long bsize = ftell(fb2);
                fseek(fb2, 0, SEEK_SET);
                unsigned char* bin = malloc(bsize);
                if (fread(bin, 1, bsize, fb2) == (size_t)bsize) {
                    fclose(fb2);
                    cl_int bin_status;
                    program = clCreateProgramWithBinary(context, 1, &device, (const size_t*)&bsize, (const unsigned char**)&bin, &bin_status, &err);
                    free(bin);
                    if (err == CL_SUCCESS) {
                        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
                        if (err == CL_SUCCESS) built_ok = 1;
                        else {
                            // print build log
                            size_t logsz = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
                            char* log = malloc(logsz+1); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL); log[logsz]=0;
                            printf("Binary build log:\n%s\n", log); free(log);
                        }
                    }
                } else {
                    fclose(fb2);
                    free(bin);
                }
            }
        }
        if (!built_ok) {
            if (!allow_inproc_build) {
                // Not allowed to build in-process; suggest using precompiled CLBIN or build_clbin helper
                printf("Error: build_clbin not available and in-process build is disabled.\n");
                printf("Please set LZ4_GPU_CLBIN=<path-to-clbin> environment variable or enable in-process build using --allow-clsrc-build\n");
                die("Cannot proceed with kernel build");
            }
            // Try to compile the source in a subprocess and write tmp_clbin
            printf("Attempting to compile source in subprocess to generate binary: %s\n", tmp_clbin);
            fflush(stdout);
            if (!compile_source_subproc(sourceToRead, tmp_clbin, build_flags)) {
                die("Subprocess compile failed to generate a binary");
            }
            FILE* fb2 = try_open_bin_candidates(tmp_clbin, exec_dir);
            if (fb2) {
                fseek(fb2, 0, SEEK_END);
                long bsize = ftell(fb2);
                fseek(fb2, 0, SEEK_SET);
                unsigned char* bin = malloc(bsize);
                if (fread(bin, 1, bsize, fb2) == (size_t)bsize) {
                    fclose(fb2);
                    cl_int bin_status;
                    program = clCreateProgramWithBinary(context, 1, &device, (const size_t*)&bsize, (const unsigned char**)&bin, &bin_status, &err);
                    free(bin);
                    if (err == CL_SUCCESS) {
                        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
                        if (err == CL_SUCCESS) built_ok = 1;
                    }
                } else { fclose(fb2); free(bin); }
            }
            if (!built_ok) {
                die("Failed to load binary generated by subprocess");
            }
        }

        if (err != CL_SUCCESS) {
            size_t logsz = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
            char* log = malloc(logsz+1); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL); log[logsz]=0;
            printf("Build log (clBuildProgram):\n%s\n", log); free(log);
            die("clBuildProgram");
        }
        printf("Built program from source (%s)\n", sourceToRead);
        fflush(stdout);

        // Export binary for future runs
        size_t num_devices = 0; clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_devices, NULL);
        /* size_t binary_sizes_count = num_devices; // not used */
        unsigned char** binaries = malloc(sizeof(unsigned char*) * num_devices);
        size_t* binary_sizes = malloc(sizeof(size_t) * num_devices);
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*num_devices, binary_sizes, NULL);
        for (size_t i = 0; i < num_devices; ++i) {
            binaries[i] = malloc(binary_sizes[i]);
        }
        err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*)*num_devices, binaries, NULL);
        if (err == CL_SUCCESS) {
            // Determine export path for compiled binary:
            // Prefer explicit kernel_bin_path if provided; if not, derive from kernel_src_path; else fallback to exe dir default.
            char export_path[1024]; export_path[0] = '\0';
            if (kernel_bin_path && strlen(kernel_bin_path) > 0) {
                strncpy(export_path, kernel_bin_path, sizeof(export_path)-1); export_path[sizeof(export_path)-1] = '\0';
            } else if (kernel_src_path && strlen(kernel_src_path) > 0) {
                // convert e.g. /path/lz4_gpu.cl -> /path/lz4_gpu.clbin
                strncpy(export_path, kernel_src_path, sizeof(export_path)-1); export_path[sizeof(export_path)-1] = '\0';
                char* dot = strrchr(export_path, '.');
                if (dot) {
                    strcpy(dot, ".clbin");
                } else {
                    strncat(export_path, ".clbin", sizeof(export_path)-strlen(export_path)-1);
                }
            } else if (exec_dir) {
                // fallback to exec_dir/lz4_gpu.clbin
                char* tmp = join_path(exec_dir, "lz4_gpu.clbin");
                if (tmp) { strncpy(export_path, tmp, sizeof(export_path)-1); export_path[sizeof(export_path)-1] = '\0'; free(tmp); }
            } else {
                // Last resort: try to write to a relative name
                strncpy(export_path, "lz4_gpu.clbin", sizeof(export_path)-1); export_path[sizeof(export_path)-1] = '\0';
            }
            // write first binary to export_path
            write_binary_file(export_path, binaries[0], binary_sizes[0]);
            printf("Wrote precompiled binary %s (%zu bytes)\n", export_path, binary_sizes[0]);
        }
        for (size_t i = 0; i < num_devices; ++i) {
            free(binaries[i]);
        }
        free(binaries); free(binary_sizes);
        free(srccl);
        if (alloc_sourceToRead) free(alloc_sourceToRead);
    }

    // Create kernels
    cl_kernel k_compress = clCreateKernel(program, "lz4_compress_block_accelerated", &err);
    if (err != CL_SUCCESS) die("clCreateKernel compress");
    cl_kernel k_decompress = clCreateKernel(program, "lz4_decompress_blocks", &err);
    if (err != CL_SUCCESS) die("clCreateKernel decompress");

    // Create buffers
    // Create input buffer and explicitly enqueue write to measure host->device time
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_len, NULL, &err); if (err!=CL_SUCCESS) die("clCreateBuffer input");
    cl_event ev_write_input = NULL;
    err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, src_len, src, 0, NULL, &ev_write_input); if (err!=CL_SUCCESS) die("clEnqueueWriteBuffer input");
    // read profiling for host->device
    cl_ulong write_start=0, write_end=0;
    if (ev_write_input) {
        clGetEventProfilingInfo(ev_write_input, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_start, NULL);
        clGetEventProfilingInfo(ev_write_input, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write_end, NULL);
    }
    size_t compressed_buf_sz = max_compressed_buffer + 1024;
    cl_mem d_compressed = clCreateBuffer(context, CL_MEM_READ_WRITE, compressed_buf_sz, NULL, &err); if (err!=CL_SUCCESS) die("clCreateBuffer compressed");

    cl_mem d_blockOffsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*totalBlocks*2, blockOffsets, &err); if (err!=CL_SUCCESS) die("clCreateBuffer blockOffsets");
    cl_mem d_outputOffsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*totalBlocks, outputOffsets, &err); if (err!=CL_SUCCESS) die("clCreateBuffer outputOffsets");
    cl_mem d_blockSizes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t)*totalBlocks, NULL, &err); if (err!=CL_SUCCESS) die("clCreateBuffer blockSizes");
    cl_mem d_maxOutputSizes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*totalBlocks, maxOutputSizes, &err); if (err!=CL_SUCCESS) die("clCreateBuffer maxOutputSizes");

    // set args for compress kernel
    err  = clSetKernelArg(k_compress, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(k_compress, 1, sizeof(cl_mem), &d_compressed);
    err |= clSetKernelArg(k_compress, 2, sizeof(cl_mem), &d_blockSizes);
    err |= clSetKernelArg(k_compress, 3, sizeof(cl_mem), &d_blockOffsets);
    err |= clSetKernelArg(k_compress, 4, sizeof(cl_mem), &d_outputOffsets);
    /* Provide maxOutputSizes buffer expected by kernel at arg index 5 */
    err |= clSetKernelArg(k_compress, 5, sizeof(cl_mem), &d_maxOutputSizes);
    int totalBlocks_i = (int)totalBlocks;
    err |= clSetKernelArg(k_compress, 6, sizeof(int), &totalBlocks_i);
    int inputSize_i = (int)src_len;
    err |= clSetKernelArg(k_compress, 7, sizeof(int), &inputSize_i);
    int tableType = 1;
    err |= clSetKernelArg(k_compress, 8, sizeof(int), &tableType);
    /* pass configured acceleration into kernel */
    err |= clSetKernelArg(k_compress, 9, sizeof(int), &acceleration);
    /* Provide local hash table buffer per work-group (size must match kernel's LZ4_HASHLOG). */
    /* If build flags provided and reduced hashlog was used, extract it from build_flags where applicable. */
    /* Note: If device doesn't support default local memory, the compiled kernel will override hashlog accordingly (see compile section). */
    long local_table_bytes = ((long)1 << effective_hashlog) * sizeof(uint32_t); /* default 65536 bytes */
    /* Query device local mem and make sure we do not pass excessive local table size in case of mismatch. */
    cl_ulong dev_local_mem = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(dev_local_mem), &dev_local_mem, NULL) != CL_SUCCESS) dev_local_mem = 0;
    if (dev_local_mem > 0 && (unsigned long)local_table_bytes > (unsigned long)dev_local_mem) {
        fprintf(stdout, "Requested local hash table bytes %ld > device local mem %llu, limiting to device local mem.\n", local_table_bytes, (unsigned long long)dev_local_mem);
        local_table_bytes = (long)dev_local_mem;
    }
    err |= clSetKernelArg(k_compress, 10, (size_t)local_table_bytes, NULL);
    if (err != CL_SUCCESS) die("clSetKernelArg compress");

    size_t global = totalBlocks;
    size_t local = 1; /* default to single work-item per work-group */
    if (local_override > 0) {
        /* allow optional override of local size via 4th CLI arg */
        if ((size_t)local_override <= global) local = (size_t)local_override;
        else local = global;
    } else {
        /* If no explicit override, choose a sane default local size based on device
         * capabilities and local memory availability (mirror host heuristic). */
        size_t default_local = 1;
        long local_table_bytes = ((long)1 << effective_hashlog) * sizeof(uint32_t);
        if (dev_local_mem > 0 && (unsigned long)local_table_bytes <= (unsigned long)dev_local_mem && max_wg >= 16) {
            if (max_wg >= 256) default_local = 256;
            else if (max_wg >= 128) default_local = 128;
            else if (max_wg >= 64) default_local = 64;
            else default_local = 1;
        }
        local = default_local;
    }
    fprintf(stdout, "Selected local work-group size: %zu\n", local);
    /* Round up global to a multiple of local to satisfy drivers that require it */
    size_t global_rounded = global;
    if (local > 0 && (global % local) != 0) {
        global_rounded = ((global + local - 1) / local) * local;
    }
    cl_event ev_comp;
    err = clEnqueueNDRangeKernel(queue, k_compress, 1, NULL, &global_rounded, &local, 0, NULL, &ev_comp); if (err!=CL_SUCCESS) die("clEnqueueNDRangeKernel compress");
    clFinish(queue);
    // profile
    cl_ulong startt=0, endt=0; clGetEventProfilingInfo(ev_comp, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startt, NULL); clGetEventProfilingInfo(ev_comp, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endt, NULL);
    double comp_ms = (endt - startt) * 1e-6;
    double host_to_device_ms = 0.0;
    if (ev_write_input) host_to_device_ms = (write_end - write_start) * 1e-6;
    printf("Host->Device upload time: %.3f ms\n", host_to_device_ms);
    printf("Compress kernel time: %.3f ms\n", comp_ms);

    // read blockSizes
    uint32_t* blockSizes = malloc(sizeof(uint32_t)*totalBlocks);
    // Read back blockSizes with profiling
    cl_event ev_read_blockSizes = NULL;
    err = clEnqueueReadBuffer(queue, d_blockSizes, CL_FALSE, 0, sizeof(uint32_t)*totalBlocks, blockSizes, 0, NULL, &ev_read_blockSizes); if (err!=CL_SUCCESS) die("clEnqueueReadBuffer blockSizes");
    clFinish(queue);
    cl_ulong read_bs_start=0, read_bs_end=0;
    if (ev_read_blockSizes) {
        clGetEventProfilingInfo(ev_read_blockSizes, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_bs_start, NULL);
        clGetEventProfilingInfo(ev_read_blockSizes, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_bs_end, NULL);
    }
    double device_to_host_blocksizes_ms = (read_bs_end - read_bs_start) * 1e-6;
    printf("Device->Host read blockSizes time: %.3f ms\n", device_to_host_blocksizes_ms);

    // Create arrays for comp offsets/sizes as 64-bit for decompress kernel
    uint64_t* comp_offsets = malloc(sizeof(uint64_t)*totalBlocks);
    uint64_t* comp_sizes = malloc(sizeof(uint64_t)*totalBlocks);
    uint64_t* out_offsets = malloc(sizeof(uint64_t)*totalBlocks);
    uint64_t* max_out_sizes = malloc(sizeof(uint64_t)*totalBlocks);
    uint32_t* sizes_out = calloc(totalBlocks, sizeof(uint32_t));

    for (size_t i = 0; i < totalBlocks; ++i) {
        comp_offsets[i] = (uint64_t)initialOutputOffsets[i];
        comp_sizes[i] = blockSizes[i];
        out_offsets[i] = (uint64_t)(i * BLOCK_SIZE);
        max_out_sizes[i] = (uint64_t)BLOCK_SIZE;
    }

    cl_mem d_comp_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t)*totalBlocks, comp_offsets, &err); if (err!=CL_SUCCESS) die("d_comp_offsets");
    cl_mem d_comp_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t)*totalBlocks, comp_sizes, &err); if (err!=CL_SUCCESS) die("d_comp_sizes");
    cl_mem d_out_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t)*totalBlocks, out_offsets, &err); if (err!=CL_SUCCESS) die("d_out_offsets");
    cl_mem d_max_out_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t)*totalBlocks, max_out_sizes, &err); if (err!=CL_SUCCESS) die("d_max_out_sizes");
    cl_mem d_sizes_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t)*totalBlocks, NULL, &err); if (err!=CL_SUCCESS) die("d_sizes_out");

    // Allocate decompressed output buffer
    unsigned char* decompressed = malloc(totalBlocks * BLOCK_SIZE);
    cl_mem d_decompressed = clCreateBuffer(context, CL_MEM_READ_WRITE, totalBlocks * BLOCK_SIZE, NULL, &err); if (err!=CL_SUCCESS) die("d_decompressed");

    /* Note: per-block alignment gating was removed; alignment flags are not used. */

    // set args for decompress
    err  = clSetKernelArg(k_decompress, 0, sizeof(cl_mem), &d_compressed);
    err |= clSetKernelArg(k_decompress, 1, sizeof(cl_mem), &d_decompressed);
    err |= clSetKernelArg(k_decompress, 2, sizeof(cl_mem), &d_comp_offsets);
    err |= clSetKernelArg(k_decompress, 3, sizeof(cl_mem), &d_comp_sizes);
    err |= clSetKernelArg(k_decompress, 4, sizeof(cl_mem), &d_out_offsets);
    err |= clSetKernelArg(k_decompress, 5, sizeof(cl_mem), &d_max_out_sizes);
    err |= clSetKernelArg(k_decompress, 6, sizeof(cl_mem), &d_sizes_out);
    err |= clSetKernelArg(k_decompress, 7, sizeof(uint32_t), &totalBlocks_i);
    if (err != CL_SUCCESS) die("clSetKernelArg decompress");

    cl_event ev_decomp;
    /* Use same rounding for decompress kernel */
    size_t global_rounded_decomp = global;
    if (local > 0 && (global % local) != 0) {
        global_rounded_decomp = ((global + local - 1) / local) * local;
    }
    err = clEnqueueNDRangeKernel(queue, k_decompress, 1, NULL, &global_rounded_decomp, &local, 0, NULL, &ev_decomp); if (err!=CL_SUCCESS) die("clEnqueueNDRangeKernel decompress");
    clFinish(queue);
    clGetEventProfilingInfo(ev_decomp, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startt, NULL);
    clGetEventProfilingInfo(ev_decomp, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endt, NULL);
    double decomp_ms = (endt - startt) * 1e-6;
    printf("Decompress kernel time: %.3f ms\n", decomp_ms);

    // read decompressed output
    cl_event ev_read_decomp = NULL;
    err = clEnqueueReadBuffer(queue, d_decompressed, CL_FALSE, 0, totalBlocks * BLOCK_SIZE, decompressed, 0, NULL, &ev_read_decomp); if (err!=CL_SUCCESS) die("read decompressed");
    clFinish(queue);
    cl_ulong read_decomp_start=0, read_decomp_end=0;
    if (ev_read_decomp) {
        clGetEventProfilingInfo(ev_read_decomp, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_decomp_start, NULL);
        clGetEventProfilingInfo(ev_read_decomp, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_decomp_end, NULL);
    }
    double device_to_host_decomp_ms = (read_decomp_end - read_decomp_start) * 1e-6;
    printf("Device->Host read decompressed time: %.3f ms\n", device_to_host_decomp_ms);

    /* Read per-block produced sizes (output sizes) from device to host so we can
     * include them in the per-block CSV and sanity check the decompress operation.
     */
    cl_event ev_read_sizes = NULL;
    err = clEnqueueReadBuffer(queue, d_sizes_out, CL_FALSE, 0, sizeof(uint32_t)*totalBlocks, sizes_out, 0, NULL, &ev_read_sizes); if (err!=CL_SUCCESS) die("read sizes_out");
    clFinish(queue);
    cl_ulong read_sizes_start=0, read_sizes_end=0;
    if (ev_read_sizes) {
        clGetEventProfilingInfo(ev_read_sizes, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_sizes_start, NULL);
        clGetEventProfilingInfo(ev_read_sizes, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_sizes_end, NULL);
    }
    double device_to_host_sizes_ms = (read_sizes_end - read_sizes_start) * 1e-6;
    printf("Device->Host read sizes_out time: %.3f ms\n", device_to_host_sizes_ms);

    // Validate
    int ok = 1;
    for (size_t i = 0; i < src_len; ++i) {
        if (src[i] != decompressed[i]) {
            ok = 0;
            printf("Mismatch at %zu: src=%02x got=%02x\n", i, src[i], decompressed[i]);
            // create ab_results dir if missing and dump failing block for offline analysis
            const char* default_outdir = "/root/lz4/lz4_gpu/ab_results";
            const char* env_out = getenv("LZ4_GPU_AB_RESULTS");
            const char* outdir = env_out ? env_out : default_outdir;
            const char* create_dirs_env = getenv("CREATE_DIRS");
            int allow_create_dirs = (create_dirs_env && strcmp(create_dirs_env, "1") == 0) ? 1 : 0;
            if (allow_create_dirs) {
                #ifdef _WIN32
                _mkdir(outdir);
                #else
                mkdir(outdir, 0755);
                #endif
            } else {
                printf("Not creating %s (CREATE_DIRS=0 or unset). Skipping write of failing block files.\n", outdir);
            }
            size_t block_idx = i / BLOCK_SIZE;
            uint32_t orig_size = blockOffsets[block_idx*2 + 1];
            uint32_t comp_size = blockSizes[block_idx];
            uint32_t comp_offset = initialOutputOffsets[block_idx];
            if (allow_create_dirs) {
                // read compressed bytes from device
                if (comp_size > 0) {
                    unsigned char* compbuf = malloc(comp_size);
                    if (compbuf) {
                        // blocking read
                        cl_int r = clEnqueueReadBuffer(queue, d_compressed, CL_TRUE, (size_t)comp_offset, comp_size, compbuf, 0, NULL, NULL);
                        if (r == CL_SUCCESS) {
                            char path_comp[512];
                            snprintf(path_comp, sizeof(path_comp), "%s/fail_block_%zu_comp.bin", outdir, block_idx);
                            write_binary_file(path_comp, compbuf, comp_size);
                            printf("Wrote compressed failing block to %s\n", path_comp);
                        } else {
                            printf("Failed to read compressed block from device (err %d)\n", r);
                        }
                        free(compbuf);
                    }
                }
            // dump expected and actual decompressed block

                size_t block_src_off = blockOffsets[block_idx*2 + 0];
                char path_exp[512]; char path_got[512];
                snprintf(path_exp, sizeof(path_exp), "%s/fail_block_%zu_exp.bin", outdir, block_idx);
                snprintf(path_got, sizeof(path_got), "%s/fail_block_%zu_got.bin", outdir, block_idx);
                write_binary_file(path_exp, src + block_src_off, orig_size);
                write_binary_file(path_got, decompressed + block_idx * BLOCK_SIZE, orig_size);
                printf("Wrote expected -> %s and actual -> %s\n", path_exp, path_got);
            }
            break;
        }
    }
    printf("Round-trip %s\n", ok ? "OK" : "FAILED");
    printf("Total kernel time: %.3f ms (compress + decompress)\n", comp_ms + decomp_ms);

    // Write per-block CSV stats
    FILE* csv = fopen("gpu_roundtrip_stats.csv", "w");
    if (csv) {
        fprintf(csv, "block,orig_size,comp_size,ratio,comp_offset,decomp_size\n");
        size_t total_orig = 0, total_comp = 0;
        for (size_t i = 0; i < totalBlocks; ++i) {
            uint32_t orig = blockOffsets[i*2 + 1];
            uint32_t comp = blockSizes[i];
            uint32_t decomp = sizes_out[i];
            double ratio = comp ? ((double)orig / (double)comp) : 0.0;
            uint32_t comp_offset = initialOutputOffsets[i];
            fprintf(csv, "%zu,%u,%u,%.3f,%u,%u\n", i, orig, comp, ratio, comp_offset, decomp);
            total_orig += orig; total_comp += comp;
        }
        fprintf(csv, "TOTAL,%zu,%zu,%.3f,\n", total_orig, total_comp, total_comp ? ((double)total_orig/total_comp) : 0.0);
        fclose(csv);
        printf("Wrote per-block stats to gpu_roundtrip_stats.csv\n");
    }

    // cleanup
    clReleaseMemObject(d_input); clReleaseMemObject(d_compressed); clReleaseMemObject(d_blockOffsets);
    clReleaseMemObject(d_outputOffsets); clReleaseMemObject(d_blockSizes);
    clReleaseMemObject(d_maxOutputSizes);
    clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_out_offsets);
    clReleaseMemObject(d_max_out_sizes); clReleaseMemObject(d_decompressed); clReleaseMemObject(d_sizes_out);
    clReleaseKernel(k_compress); clReleaseKernel(k_decompress); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);

    free(src); free(blockOffsets); free(outputOffsets); free(initialOutputOffsets); free(blockSizes);
    if (exec_dir) free(exec_dir);
    free(comp_offsets); free(comp_sizes); free(out_offsets); free(max_out_sizes); free(decompressed); free(sizes_out);
    return ok ? 0 : 2;
}
