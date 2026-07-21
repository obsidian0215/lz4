#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#define access _access
#define getpid _getpid
#define F_OK 0
#define strcasecmp _stricmp
#else
#include <strings.h>
#include <unistd.h>
#include <signal.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "lz4_gpu_core.h"
#include "lz4_gpu_debug.h"
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_utils.h"
#include "lz4_tp_profile.h"
#include "timing.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int lz4_set_env(const char* name, const char* value) {
#if defined(_WIN32)
    return _putenv_s(name, value);
#else
    return setenv(name, value, 1);
#endif
}

static int lz4_unset_env(const char* name) {
#if defined(_WIN32)
    return _putenv_s(name, "");
#else
    return unsetenv(name);
#endif
}

/* Forward declarations */
#if defined(_WIN32)
static int run_daemon(void) {
    fprintf(stderr, "Daemon mode is not supported in Windows builds. Use standalone or bench mode.\n");
    return 1;
}
static int run_lz4_client(int mode, const char* input_path, const char* output_path, int block_size, int acceleration, int local_size, int hash_log, int raw_buffer, int twophase) {
    (void)mode; (void)input_path; (void)output_path; (void)block_size; (void)acceleration; (void)local_size; (void)hash_log; (void)raw_buffer; (void)twophase;
    fprintf(stderr, "--use-daemon is not supported in Windows builds. Use standalone mode.\n");
    return 1;
}
#else
int run_daemon(void);
int run_lz4_client(int mode, const char* input_path, const char* output_path, int block_size, int acceleration, int local_size, int hash_log, int raw_buffer, int twophase);
#endif

int g_verbose = 0;
static size_t g_cli_local_size = 1;
static size_t g_cli_fixed_block_bytes = 64 * 1024;
static int g_cli_acceleration = 1;
static int g_cli_tp_n = 4;
static int g_cli_hash_log = 14;
static const char* g_cli_metrics_path = NULL;

static cl_context ctx;
static cl_command_queue queue;
static cl_device_id dev;
static cl_platform_id platform;

static void ocl_release_runtime(void) {
    if (queue) { clReleaseCommandQueue(queue); queue = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
    lz4_release_opencl_device(&dev);
    platform = NULL;
}

static void ocl_init() {
    cl_int err;
    cl_platform_id selected_pf = NULL;
    dev = NULL;
    platform = NULL;
    cl_int r = lz4_select_opencl_platform_device(&selected_pf, &dev);

    if (r != CL_SUCCESS || dev == NULL) {
        if (r == CL_INVALID_VALUE) {
            fprintf(stderr, "OpenCL init failed: invalid FORCE_OPENCL_DEVICE value\n");
        } else {
            fprintf(stderr, "OpenCL init failed: requested device type is unavailable\n");
        }
        ctx = NULL;
        queue = NULL;
        return;
    }

    {
        char pfname[256] = {0};
        char devname[256] = {0};
        cl_device_type devtype = 0;
        clGetPlatformInfo(selected_pf, CL_PLATFORM_NAME, sizeof(pfname), pfname, NULL);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devname), devname, NULL);
        clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
        fprintf(stderr, "[OpenCL DEBUG] Selected platform=%s, device=%s (type=%s)\n",
                pfname,
                devname,
                (devtype & CL_DEVICE_TYPE_GPU) ? "GPU" :
                (devtype & CL_DEVICE_TYPE_CPU) ? "CPU" :
                (devtype & CL_DEVICE_TYPE_DEFAULT) ? "DEFAULT" : "UNKNOWN");
    }

    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || ctx == NULL) {
        fprintf(stderr, "OpenCL init failed: clCreateContext err=%d\n", err);
        lz4_release_opencl_device(&dev);
        ctx = NULL;
        return;
    }
    platform = selected_pf;
#if defined(CL_VERSION_2_0)
    {
        const cl_queue_properties qprops[] = {
            CL_QUEUE_PROPERTIES,
            (cl_queue_properties)CL_QUEUE_PROFILING_ENABLE,
            0
        };
        queue = clCreateCommandQueueWithProperties(ctx, dev, qprops, &err);
    }
#else
    queue = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (err != CL_SUCCESS || queue == NULL) {
        fprintf(stderr, "OpenCL init failed: command queue creation err=%d\n", err);
        if (ctx) clReleaseContext(ctx);
        lz4_release_opencl_device(&dev);
        ctx = NULL;
        queue = NULL;
        return;
    }
}

static void lz4_print_json_string(FILE* output, const char* value) {
    fputc('"', output);
    for (const unsigned char* p = (const unsigned char*)(value ? value : ""); *p; p++) {
        switch (*p) {
            case '"': fputs("\\\"", output); break;
            case '\\': fputs("\\\\", output); break;
            case '\b': fputs("\\b", output); break;
            case '\f': fputs("\\f", output); break;
            case '\n': fputs("\\n", output); break;
            case '\r': fputs("\\r", output); break;
            case '\t': fputs("\\t", output); break;
            default:
                if (*p < 0x20) fprintf(output, "\\u%04x", (unsigned)*p);
                else fputc(*p, output);
        }
    }
    fputc('"', output);
}

static int lz4_print_device_info(void) {
    ocl_init();
    if (!ctx || !queue || !dev || !platform) return 1;
    char platform_name[256] = {0}, platform_vendor[256] = {0}, platform_version[256] = {0};
    char device_name[256] = {0}, device_vendor[256] = {0}, driver[256] = {0};
    char device_version[256] = {0}, opencl_c_version[256] = {0};
    cl_device_type device_type = 0;
    cl_ulong global_mem = 0, max_alloc = 0;
    cl_uint compute_units = 0, parent_compute_units = 0, requested_cpu_threads = 0;
    int partitioned = 0;
    char profile_key[320] = {0};
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, NULL);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(driver), driver, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, sizeof(opencl_c_version), opencl_c_version, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    partitioned = lz4_opencl_device_partition_info(dev, &parent_compute_units);
    if (!partitioned) parent_compute_units = compute_units;
    if (device_type & CL_DEVICE_TYPE_CPU) {
        const char* limit = getenv("HETEROLZ_CPU_THREADS");
        if (limit && *limit) requested_cpu_threads = (cl_uint)strtoul(limit, NULL, 10);
    }
    lz4_device_profile_key(dev, profile_key, sizeof(profile_key));
    const char* type_name = (device_type & CL_DEVICE_TYPE_GPU) ? "GPU" :
                            (device_type & CL_DEVICE_TYPE_CPU) ? "CPU" :
                            (device_type & CL_DEVICE_TYPE_ACCELERATOR) ? "ACCELERATOR" :
                            (device_type & CL_DEVICE_TYPE_DEFAULT) ? "DEFAULT" : "OTHER";
    fputs("{\n  \"schema\": \"heterolz.device-info.v1\",\n  \"platform_name\": ", stdout);
    lz4_print_json_string(stdout, platform_name);
    fputs(",\n  \"platform_vendor\": ", stdout); lz4_print_json_string(stdout, platform_vendor);
    fputs(",\n  \"platform_version\": ", stdout); lz4_print_json_string(stdout, platform_version);
    fputs(",\n  \"device_name\": ", stdout); lz4_print_json_string(stdout, device_name);
    fputs(",\n  \"device_vendor\": ", stdout); lz4_print_json_string(stdout, device_vendor);
    fputs(",\n  \"device_type\": ", stdout); lz4_print_json_string(stdout, type_name);
    fputs(",\n  \"driver_version\": ", stdout); lz4_print_json_string(stdout, driver);
    fputs(",\n  \"device_version\": ", stdout); lz4_print_json_string(stdout, device_version);
    fputs(",\n  \"opencl_c_version\": ", stdout); lz4_print_json_string(stdout, opencl_c_version);
    fputs(",\n  \"profile_key\": ", stdout); lz4_print_json_string(stdout, profile_key);
    fprintf(stdout, ",\n  \"compute_units\": %u,\n  \"parent_compute_units\": %u,\n"
                    "  \"requested_cpu_threads\": %u,\n  \"partitioned\": %s,\n"
                    "  \"global_mem_bytes\": %llu,\n"
                    "  \"max_alloc_bytes\": %llu\n}\n",
            compute_units, parent_compute_units, requested_cpu_threads,
            partitioned ? "true" : "false",
            (unsigned long long)global_mem, (unsigned long long)max_alloc);
    fflush(stdout);
    ocl_release_runtime();
    return 0;
}

static void show_help(const char* prog_name) {
    fprintf(stderr, "Unified LZ4 GPU Tool\n");
    fprintf(stderr, "Usage Modes:\n");
    fprintf(stderr, "  1. Standalone:   %s [options] <input_file|->\n", prog_name);
    fprintf(stderr, "  2. Run Daemon:   %s --daemon [options]\n", prog_name);
    fprintf(stderr, "  3. Use Daemon:   %s --use-daemon [options] <input_file>\n", prog_name);
    fprintf(stderr, "                    add --raw-buffer to stream stdin/stdout through daemon raw-buffer protocol\n");
    fprintf(stderr, "  4. Stop Daemon:  %s --stop-daemon\n", prog_name);

    fprintf(stderr, "\nBasic Options:\n");
    fprintf(stderr, "  -c                   Compress mode (default)\n");
    fprintf(stderr, "  -d, --decompress     Decompress mode\n");
    fprintf(stderr, "  -o, --output FILE    Output file (use '-' for stdout)\n");
    fprintf(stderr, "  -B, --block-size N   Block size in bytes (default: 64KB)\n");
    fprintf(stderr, "  -a, --acceleration N Acceleration factor (default: 1)\n");
    fprintf(stderr, "  --d-bits N           Hash dictionary bits, 11..15 (default: 14)\n");
    fprintf(stderr, "  --local N            Local work-group size (default: 1)\n");
    fprintf(stderr, "  -v, --verbose        Enable performance statistics\n");
    fprintf(stderr, "  --bench [N]          Stable benchmark (compress+decompress+verify), optional N seconds (default: 3)\n");
    fprintf(stderr, "\nTwo-phase LZ4TP1:\n");
    fprintf(stderr, "  --twophase           Use the segmented LZ4TP1 container\n");
    fprintf(stderr, "  -N 1|2|4|8          Segments per block for --twophase compression\n");
    fprintf(stderr, "  --tp-bench           Two-phase kernel sweep with 2 warmups and 9 measurements\n");
    fprintf(stderr, "  --calibrate          Calibrate from the input file; -o optionally selects the profile\n");
    fprintf(stderr, "  --auto               Select N from the stored device calibration profile\n");
    fprintf(stderr, "  --metrics-json FILE  Write structured timings for one LZ4TP1 operation\n");
    fprintf(stderr, "  --device-info        Print the selected OpenCL platform/device as JSON\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Streaming:\n");
    fprintf(stderr, "  input '-'            Read input from stdin (standalone mode only)\n");
    fprintf(stderr, "  output '-'           Write output to stdout (standalone mode only)\n");
    fprintf(stderr, "\nEnvironment:\n");
    fprintf(stderr, "  LZ4_STANDARD_COPY=0|1         0=map/unmap, 1=clEnqueueRead/WriteBuffer (default: auto)\n");
    fprintf(stderr, "  LZ4_GPU_DECOMP_FORCE_CHUNKED=0|1  Force decompression chunked readback/write\n");
    fprintf(stderr, "  LZ4_GPU_DECOMP_DISABLE_CHUNKED=1  Disable decompression chunked readback/write\n");
    fprintf(stderr, "  LZ4_GPU_DECOMP_CHUNKED_THRESHOLD_KB=N  Chunked output threshold (default: 16384)\n");
    fprintf(stderr, "  LZ4_GPU_DECOMP_READBACK_KB=N     Chunked readback size (default: 8192)\n");
    fprintf(stderr, "  LZ4_GPU_DEBUG=0|1             Enable host/device debug counters (forces source build)\n");
    fprintf(stderr, "  LZ4_GPU_DEBUG_BLOCK_LIMIT=N   Print first N blocks of debug counters\n");
    fprintf(stderr, "  LZ4TP_PROFILE=FILE            Device-keyed calibration profile (default: lz4tp.profile)\n");
    fprintf(stderr, "  LZ4_TP_LEARN=0|1             Daemon live learning; experimental and disabled by default\n");
    fprintf(stderr, "  FORCE_OPENCL_DEVICE=GPU|CPU|DEFAULT|ALL  Strict device type when explicitly set\n");
}

static int stop_daemon_cmd() {
#if defined(_WIN32)
    fprintf(stderr, "--stop-daemon is not supported in Windows builds.\n");
    return 1;
#else
    const char* pid_path = "/tmp/lz4_gpu_daemon.pid";
    FILE* f = fopen(pid_path, "r");
    if (!f) {
        printf("Daemon not running (PID file not found)\n");
        return 0;
    }
    pid_t pid;
    if (fscanf(f, "%d", &pid) != 1) {
        fclose(f);
        return 1;
    }
    fclose(f);
    printf("Stopping daemon (PID: %d)...\n", pid);
    kill(pid, SIGTERM);
    unlink(pid_path);
    unlink(SOCKET_PATH);
    return 0;
#endif
}

static size_t parse_size_bytes(const char* s) {
    if (!s || !*s || *s == '-') return 0;
    char* endptr = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(s, &endptr, 10);
    if (errno != 0 || endptr == s) return 0;
    size_t multiplier = 1;
    if (*endptr != '\0') {
        if (endptr[1] != '\0') return 0;
        if (*endptr == 'k' || *endptr == 'K') multiplier = 1024;
        else if (*endptr == 'm' || *endptr == 'M') multiplier = 1024 * 1024;
        else return 0;
    }
    if (parsed > (unsigned long long)SIZE_MAX / multiplier) return 0;
    return (size_t)parsed * multiplier;
}

static int parse_int_arg(const char* s, int* out) {
    if (!s || !*s || !out) return -1;
    char* end = NULL;
    errno = 0;
    long value = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || value < INT_MIN || value > INT_MAX) return -1;
    *out = (int)value;
    return 0;
}

static int parse_positive_double_arg(const char* s, double* out) {
    if (!s || !*s || !out) return -1;
    char* end = NULL;
    errno = 0;
    double value = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(value) || value <= 0.0) return -1;
    *out = value;
    return 0;
}

static int copy_cli_path(char* destination, size_t capacity, const char* source) {
    if (!destination || capacity == 0 || !source) return -1;
    size_t length = strlen(source);
    if (length >= capacity) return -1;
    memcpy(destination, source, length + 1);
    return 0;
}

static int append_cli_suffix(char* destination, size_t capacity,
                             const char* source, const char* suffix) {
    if (!destination || capacity == 0 || !source || !suffix) return -1;
    size_t source_length = strlen(source);
    size_t suffix_length = strlen(suffix);
    if (source_length >= capacity || suffix_length >= capacity - source_length) return -1;
    memcpy(destination, source, source_length);
    memcpy(destination + source_length, suffix, suffix_length + 1);
    return 0;
}

static int validate_cli_config(int twophase_mode) {
    if (g_cli_fixed_block_bytes == 0 || g_cli_fixed_block_bytes > INT_MAX) {
        fprintf(stderr, "Error: block size must be in 1..%d bytes\n", INT_MAX);
        return -1;
    }
    if (g_cli_hash_log < 11 || g_cli_hash_log > 15) {
        fprintf(stderr, "Error: --d-bits must be in 11..15\n");
        return -1;
    }
    if (g_cli_acceleration < 1) {
        fprintf(stderr, "Error: acceleration must be positive\n");
        return -1;
    }
    if (g_cli_local_size < 1 || g_cli_local_size > INT_MAX) {
        fprintf(stderr, "Error: local size must be in 1..%d\n", INT_MAX);
        return -1;
    }
    if (twophase_mode && g_cli_tp_n != 1 && g_cli_tp_n != 2 &&
        g_cli_tp_n != 4 && g_cli_tp_n != 8) {
        fprintf(stderr, "Error: two-phase N must be one of 1, 2, 4, or 8\n");
        return -1;
    }
    return 0;
}

static int path_is_dash(const char* path) {
    return path && strcmp(path, "-") == 0;
}

static int paths_identify_same_file(const char* left, const char* right) {
    if (!left || !right) return 0;
    if (strcmp(left, right) == 0) return 1;
#if defined(_WIN32)
    char* left_full = _fullpath(NULL, left, 0);
    char* right_full = _fullpath(NULL, right, 0);
    int same = left_full && right_full && _stricmp(left_full, right_full) == 0;
    free(left_full);
    free(right_full);
    if (same) return 1;
    HANDLE left_handle = CreateFileA(left, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                     NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    HANDLE right_handle = CreateFileA(right, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (left_handle == INVALID_HANDLE_VALUE || right_handle == INVALID_HANDLE_VALUE) {
        if (left_handle != INVALID_HANDLE_VALUE) CloseHandle(left_handle);
        if (right_handle != INVALID_HANDLE_VALUE) CloseHandle(right_handle);
        return 0;
    }
    BY_HANDLE_FILE_INFORMATION left_info, right_info;
    same = GetFileInformationByHandle(left_handle, &left_info) &&
           GetFileInformationByHandle(right_handle, &right_info) &&
           left_info.dwVolumeSerialNumber == right_info.dwVolumeSerialNumber &&
           left_info.nFileIndexHigh == right_info.nFileIndexHigh &&
           left_info.nFileIndexLow == right_info.nFileIndexLow;
    CloseHandle(left_handle);
    CloseHandle(right_handle);
    return same;
#else
    struct stat left_stat, right_stat;
    if (stat(left, &left_stat) != 0 || stat(right, &right_stat) != 0) return 0;
    return left_stat.st_dev == right_stat.st_dev && left_stat.st_ino == right_stat.st_ino;
#endif
}

static int create_temp_path(char* path_buf, size_t path_buf_size, const char* templ) {
    if (!path_buf || path_buf_size == 0 || !templ) return -1;
#if defined(_WIN32)
    (void)path_buf_size;
    (void)templ;
    return -1;
#else
    {
        int fd;
        size_t n = strlen(templ);
        if (n + 1 > path_buf_size) return -1;
        memcpy(path_buf, templ, n + 1);
        fd = mkstemp(path_buf);
        if (fd < 0) return -1;
        close(fd);
        unlink(path_buf);
        return 0;
    }
#endif
}

static int copy_stream_to_path(FILE* in, const char* path) {
    FILE* out;
    unsigned char buf[1 << 20];
    size_t nread;

    if (!in || !path) return -1;
    out = fopen(path, "wb");
    if (!out) return -1;

    while ((nread = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, nread, out) != nread) {
            fclose(out);
            return -1;
        }
    }
    if (ferror(in)) {
        fclose(out);
        return -1;
    }

    if (fclose(out) != 0) return -1;
    return 0;
}

static int copy_path_to_stream(const char* path, FILE* out) {
    FILE* in;
    unsigned char buf[1 << 20];
    size_t nread;

    if (!path || !out) return -1;
    in = fopen(path, "rb");
    if (!in) return -1;

    while ((nread = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, nread, out) != nread) {
            fclose(in);
            return -1;
        }
    }
    if (ferror(in)) {
        fclose(in);
        return -1;
    }

    if (fclose(in) != 0) return -1;
    if (fflush(out) != 0) return -1;
    return 0;
}

static unsigned parse_unsigned_env_with_default(const char* name, unsigned defv) {
    const char* env = getenv(name);
    char* end = NULL;
    unsigned long parsed;
    if (!env || !*env) return defv;
    parsed = strtoul(env, &end, 10);
    if (end == env || *end != '\0') return defv;
    return (unsigned)parsed;
}

static int cmp_double_asc(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double median_double(const double* vals, size_t n) {
    if (!vals || n == 0) return 0.0;
    double* tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return 0.0;
    memcpy(tmp, vals, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double_asc);
    double out = (n % 2 == 0) ? (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0 : tmp[n / 2];
    free(tmp);
    return out;
}

static double event_elapsed_us(cl_event ev) {
    cl_ulong st = 0, en = 0;
    if (!ev) return 0.0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) != CL_SUCCESS) return 0.0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, NULL) != CL_SUCCESS) return 0.0;
    if (en <= st) return 0.0;
    return (double)(en - st) / 1000.0;
}

static double elapsed_sec(const struct timespec* start, const struct timespec* end) {
    return (double)(end->tv_sec - start->tv_sec) + (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

static int run_lz4_bench(const char* input_path,
                         int block_size,
                         int acceleration,
                         int local_size,
                         double bench_seconds) {
    struct stat st;
    if (!input_path || stat(input_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "bench error: invalid input file\n");
        return 1;
    }
    if (bench_seconds <= 0.0) bench_seconds = 3.0;

    size_t in_size_ref = (size_t)st.st_size;
    unsigned char* input_ref = (unsigned char*)malloc(in_size_ref);
    unsigned long ref_read_us = 0;
    if (!input_ref || in_size_ref == 0 ||
        lz4_read_file_to_buf(input_path, input_ref, in_size_ref, &ref_read_us) != 0) {
        free(input_ref);
        fprintf(stderr, "bench error: failed to read input\n");
        return 1;
    }
    (void)ref_read_us;

    ocl_init();
    if (!ctx || !queue) {
        fprintf(stderr, "bench error: OpenCL init failed\n");
        free(input_ref);
        return 1;
    }

    cl_program prog = lz4_load_program(ctx, dev, g_cli_hash_log, (size_t)block_size);
    if (!prog) {
        fprintf(stderr, "bench error: kernel program load failed\n");
        ocl_release_runtime();
        free(input_ref);
        return 1;
    }

    cl_int err = CL_SUCCESS;
    cl_kernel kcomp = clCreateKernel(prog, "lz4_compress_block", &err);
    if (err != CL_SUCCESS || !kcomp) {
        fprintf(stderr, "bench error: create compress kernel failed (%d)\n", err);
        clReleaseProgram(prog);
        ocl_release_runtime();
        free(input_ref);
        return 1;
    }
    cl_kernel kdec = clCreateKernel(prog, "lz4_decompress_blocks", &err);
    if (err != CL_SUCCESS || !kdec) {
        fprintf(stderr, "bench error: create decompress kernel failed (%d)\n", err);
        clReleaseKernel(kcomp);
        clReleaseProgram(prog);
        ocl_release_runtime();
        free(input_ref);
        return 1;
    }
    lz4_gpu_workspace_t ws;
    lz4_gpu_workspace_init(&ws);

    size_t cap = 16, n = 0;
    double* comp_tp = (double*)malloc(cap * sizeof(double));
    double* dec_tp = (double*)malloc(cap * sizeof(double));
    double* ratio_pct = (double*)malloc(cap * sizeof(double));
    int verify_ok = 1;

    cl_mem d_out = NULL;
    size_t d_out_capacity = 0;

    cl_uint* h_sizes = NULL;
    cl_uint* h_comp_off = NULL;
    cl_uint* h_sizes_out = NULL;
    size_t h_blocks_capacity = 0;
    cl_uint* h_prev_sizes = NULL;
    cl_uint* h_prev_comp_off = NULL;
    size_t h_prev_blocks_capacity = 0;
    size_t prev_meta_nblk = 0;
    size_t prev_meta_blk = 0;
    size_t prev_meta_in_size = 0;
    int prev_meta_valid = 0;

    cl_uint kernel_num_args_dec = 0;
    int kernel_has_dbg_dec = 0;
    lz4_gpu_debug_config_t dbg_cfg = lz4_gpu_get_debug_config();
    if (clGetKernelInfo(kdec, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args_dec), &kernel_num_args_dec, NULL) == CL_SUCCESS) {
        kernel_has_dbg_dec = (kernel_num_args_dec >= 9U);
    }

    if (!comp_tp || !dec_tp || !ratio_pct) {
        fprintf(stderr, "bench error: malloc failed\n");
        verify_ok = 0;
    }

    if (verify_ok) {
        ws.decomp_comp_off_buf = ensure_buffer(ctx, ws.decomp_comp_off_buf, sizeof(cl_uint),
                                               &ws.current_decomp_comp_off_capacity, &err);
        if (!ws.decomp_comp_off_buf || err != CL_SUCCESS) verify_ok = 0;
    }
    if (verify_ok) {
        ws.decomp_comp_size_buf = ensure_buffer(ctx, ws.decomp_comp_size_buf, sizeof(cl_uint),
                                                &ws.current_decomp_comp_size_capacity, &err);
        if (!ws.decomp_comp_size_buf || err != CL_SUCCESS) verify_ok = 0;
    }
    if (verify_ok) {
        ws.decomp_sizes_out_buf = ensure_buffer(ctx, ws.decomp_sizes_out_buf, sizeof(cl_uint),
                                                &ws.current_decomp_sizes_out_capacity, &err);
        if (!ws.decomp_sizes_out_buf || err != CL_SUCCESS) verify_ok = 0;
    }
    if (!verify_ok) {
        fprintf(stderr, "bench error: failed to allocate reusable decomp buffers\n");
    }

    unsigned warmup_rounds = parse_unsigned_env_with_default("LZ4_GPU_BENCH_WARMUP_ROUNDS", 1U);
    unsigned warmup_done = 0;
    int input_uploaded_once = 0;
    int timer_started = 0;

    struct timespec ts0, ts1;
    if (warmup_rounds == 0) {
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        timer_started = 1;
    }

    while (verify_ok) {
        timing_t tc;
        memset(&tc, 0, sizeof(tc));

        int rc = lz4_compress_core(ctx,
                                   queue,
                                   kcomp,
                                   input_path,
                                   NULL,
                                   (size_t)block_size,
                                   acceleration,
                                   g_cli_hash_log,
                                   &ws,
                                   &tc,
                                   local_size,
                                   input_uploaded_once ? 1 : 0);
        if (rc != 0) {
            verify_ok = 0;
            break;
        }
        input_uploaded_once = 1;

        size_t nblk = (size_t)tc.nblk;
        size_t blk = (size_t)tc.blk_size_bytes;
        size_t worst_blk = (size_t)((double)blk * 1.1 + 64.0);
        size_t comp_total = (size_t)tc.out_size;
        if (nblk == 0 || blk == 0 || comp_total == 0) {
            verify_ok = 0;
            break;
        }

        if (h_blocks_capacity < nblk) {
            cl_uint* nh_sizes = (cl_uint*)realloc(h_sizes, nblk * sizeof(cl_uint));
            if (!nh_sizes) {
                verify_ok = 0;
                break;
            }
            h_sizes = nh_sizes;

            cl_uint* nh_comp_off = (cl_uint*)realloc(h_comp_off, nblk * sizeof(cl_uint));
            if (!nh_comp_off) {
                verify_ok = 0;
                break;
            }
            h_comp_off = nh_comp_off;

            cl_uint* nh_sizes_out = (cl_uint*)realloc(h_sizes_out, nblk * sizeof(cl_uint));
            if (!nh_sizes_out) {
                verify_ok = 0;
                break;
            }
            h_sizes_out = nh_sizes_out;

            cl_uint* nh_prev_sizes = (cl_uint*)realloc(h_prev_sizes, nblk * sizeof(cl_uint));
            if (!nh_prev_sizes) {
                verify_ok = 0;
                break;
            }
            h_prev_sizes = nh_prev_sizes;

            cl_uint* nh_prev_comp_off = (cl_uint*)realloc(h_prev_comp_off, nblk * sizeof(cl_uint));
            if (!nh_prev_comp_off) {
                verify_ok = 0;
                break;
            }
            h_prev_comp_off = nh_prev_comp_off;

            h_blocks_capacity = nblk;
            h_prev_blocks_capacity = nblk;
            prev_meta_valid = 0;
        }
        {
            void* map_sizes = clEnqueueMapBuffer(queue, ws.output_size_buf, CL_TRUE, CL_MAP_READ,
                                                 0, nblk * sizeof(cl_uint), 0, NULL, NULL, &err);
            if (err != CL_SUCCESS || !map_sizes) {
                verify_ok = 0;
                break;
            }
            memcpy(h_sizes, map_sizes, nblk * sizeof(cl_uint));
            clEnqueueUnmapMemObject(queue, ws.output_size_buf, map_sizes, 0, NULL, NULL);

            for (size_t i = 0; i < nblk; ++i) {
                size_t csz = (size_t)h_sizes[i];
                h_comp_off[i] = (cl_uint)(i * worst_blk);
                if (csz > worst_blk) {
                    verify_ok = 0;
                    break;
                }
            }
            if (!verify_ok) {
                verify_ok = 0;
                break;
            }
        }

        d_out = ensure_buffer(ctx, d_out, (size_t)tc.in_size, &d_out_capacity, &err);
        if (!d_out || err != CL_SUCCESS) {
            verify_ok = 0;
            break;
        }

        size_t prev_comp_off_cap = ws.current_decomp_comp_off_capacity;
        size_t prev_comp_size_cap = ws.current_decomp_comp_size_capacity;
        ws.decomp_comp_off_buf = ensure_buffer(ctx, ws.decomp_comp_off_buf, nblk * sizeof(cl_uint),
                                               &ws.current_decomp_comp_off_capacity, &err);
        ws.decomp_comp_size_buf = ensure_buffer(ctx, ws.decomp_comp_size_buf, nblk * sizeof(cl_uint),
                                                &ws.current_decomp_comp_size_capacity, &err);
        ws.decomp_sizes_out_buf = ensure_buffer(ctx, ws.decomp_sizes_out_buf, nblk * sizeof(cl_uint),
                                                &ws.current_decomp_sizes_out_capacity, &err);

        cl_mem d_comp_off = ws.decomp_comp_off_buf;
        cl_mem d_comp_sz = ws.decomp_comp_size_buf;
        cl_mem d_sizes_out = ws.decomp_sizes_out_buf;
        cl_mem d_dbg_dec = NULL;

        if (!d_comp_off || !d_comp_sz || !d_sizes_out || err != CL_SUCCESS) {
            verify_ok = 0;
            break;
        }

        int meta_buffers_recreated =
            (ws.current_decomp_comp_off_capacity != prev_comp_off_cap) ||
            (ws.current_decomp_comp_size_capacity != prev_comp_size_cap);

        size_t meta_bytes = nblk * sizeof(cl_uint);
        int dec_meta_changed = 1;
        if (!meta_buffers_recreated &&
            prev_meta_valid &&
            h_prev_sizes &&
            h_prev_comp_off &&
            h_prev_blocks_capacity >= nblk &&
            prev_meta_nblk == nblk &&
            prev_meta_blk == blk &&
            prev_meta_in_size == (size_t)tc.in_size &&
            memcmp(h_prev_sizes, h_sizes, meta_bytes) == 0 &&
            memcmp(h_prev_comp_off, h_comp_off, meta_bytes) == 0) {
            dec_meta_changed = 0;
        }

        if (dec_meta_changed) {
            memcpy(h_prev_sizes, h_sizes, meta_bytes);
            memcpy(h_prev_comp_off, h_comp_off, meta_bytes);
            prev_meta_nblk = nblk;
            prev_meta_blk = blk;
            prev_meta_in_size = (size_t)tc.in_size;
            prev_meta_valid = 1;
        }

        if (dbg_cfg.enabled && kernel_has_dbg_dec) {
            size_t dbg_bytes = nblk * LZ4_DBG_DEC_N * sizeof(cl_uint);
            d_dbg_dec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, dbg_bytes, NULL, &err);
            if (err == CL_SUCCESS && d_dbg_dec) {
                cl_uint zero = 0;
                err = clEnqueueFillBuffer(queue, d_dbg_dec, &zero, sizeof(zero), 0, dbg_bytes, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    clReleaseMemObject(d_dbg_dec);
                    d_dbg_dec = NULL;
                }
            } else {
                d_dbg_dec = NULL;
            }
        }

        cl_event write_ev[2] = { NULL, NULL };
        cl_uint write_event_count = 0;
        if (dec_meta_changed) {
            cl_int ew0 = clEnqueueWriteBuffer(queue, d_comp_off, CL_FALSE, 0, meta_bytes, h_comp_off, 0, NULL, &write_ev[0]);
            cl_int ew1 = clEnqueueWriteBuffer(queue, d_comp_sz, CL_FALSE, 0, meta_bytes, h_sizes, 0, NULL, &write_ev[1]);
            if (ew0 != CL_SUCCESS || ew1 != CL_SUCCESS) {
                for (int wi = 0; wi < 2; ++wi) if (write_ev[wi]) clReleaseEvent(write_ev[wi]);
                if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
                verify_ok = 0;
                break;
            }
            write_event_count = 2;
        }

        cl_uint totalBlocks = (cl_uint)nblk;
        cl_uint block_size_u32 = (cl_uint)blk;
        err  = clSetKernelArg(kdec, 0, sizeof(cl_mem), &ws.out_buf);
        err |= clSetKernelArg(kdec, 1, sizeof(cl_mem), &d_out);
        err |= clSetKernelArg(kdec, 2, sizeof(cl_mem), &d_comp_off);
        err |= clSetKernelArg(kdec, 3, sizeof(cl_mem), &d_comp_sz);
        err |= clSetKernelArg(kdec, 4, sizeof(cl_mem), &d_sizes_out);
        err |= clSetKernelArg(kdec, 5, sizeof(cl_uint), &block_size_u32);
        err |= clSetKernelArg(kdec, 6, sizeof(cl_uint), &totalBlocks);
        if (kernel_has_dbg_dec) {
            cl_mem dbg_arg = d_dbg_dec ? d_dbg_dec : d_sizes_out;
            cl_uint dbg_flag = d_dbg_dec ? 1U : 0U;
            err |= clSetKernelArg(kdec, 7, sizeof(cl_mem), &dbg_arg);
            err |= clSetKernelArg(kdec, 8, sizeof(cl_uint), &dbg_flag);
        }
        if (err != CL_SUCCESS) {
            if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
            verify_ok = 0;
            break;
        }

        size_t lsz = (local_size > 0) ? (size_t)local_size : 1;
        if (lsz > nblk) lsz = 1;
        size_t gsz = ((nblk + lsz - 1) / lsz) * lsz;
        if (gsz == 0) gsz = 1;

        cl_event dec_kernel_evt = NULL;
        err = clEnqueueNDRangeKernel(queue, kdec, 1, NULL, &gsz, &lsz,
                         write_event_count,
                         (write_event_count > 0) ? write_ev : NULL,
                         &dec_kernel_evt);
        double dec_upload_us = 0.0;
        for (int wi = 0; wi < 2; ++wi) {
            if (write_ev[wi]) {
                clWaitForEvents(1, &write_ev[wi]);
                dec_upload_us += event_elapsed_us(write_ev[wi]);
                clReleaseEvent(write_ev[wi]);
                write_ev[wi] = NULL;
            }
        }
        double dec_kernel_us = 0.0;
        if (err == CL_SUCCESS && dec_kernel_evt) {
            clWaitForEvents(1, &dec_kernel_evt);
            dec_kernel_us = event_elapsed_us(dec_kernel_evt);
            clReleaseEvent(dec_kernel_evt);
            dec_kernel_evt = NULL;
        }
        if (err != CL_SUCCESS) {
            if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
            verify_ok = 0;
            break;
        }

        cl_event read_sizes_evt = NULL;
        err = clEnqueueReadBuffer(queue, d_sizes_out, CL_FALSE, 0, nblk * sizeof(cl_uint), h_sizes_out, 0, NULL, &read_sizes_evt);
        if (err != CL_SUCCESS) {
            if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
            verify_ok = 0;
            break;
        }
        clWaitForEvents(1, &read_sizes_evt);
        clReleaseEvent(read_sizes_evt);

        size_t out_total = 0;
        for (size_t i = 0; i < nblk; ++i) {
            if (h_sizes_out[i] == 0xFFFFFFFFU) {
                verify_ok = 0;
                break;
            }
            out_total += (size_t)h_sizes_out[i];
        }

        void* map_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, (size_t)tc.in_size, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !map_out) {
            verify_ok = 0;
        } else {
            if (out_total != (size_t)tc.in_size || memcmp(map_out, input_ref, (size_t)tc.in_size) != 0) {
                verify_ok = 0;
            }
            clEnqueueUnmapMemObject(queue, d_out, map_out, 0, NULL, NULL);
        }

        if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
        if (!verify_ok) break;
        if (warmup_done < warmup_rounds) {
            warmup_done++;
            if (warmup_done == warmup_rounds) {
                clock_gettime(CLOCK_MONOTONIC, &ts0);
                timer_started = 1;
            }
            continue;
        }

        if (n == cap) {
            size_t new_cap = cap * 2;
            double* nc = (double*)realloc(comp_tp, new_cap * sizeof(double));
            if (!nc) {
                verify_ok = 0;
                break;
            }
            comp_tp = nc;

            double* nd = (double*)realloc(dec_tp, new_cap * sizeof(double));
            if (!nd) {
                verify_ok = 0;
                break;
            }
            dec_tp = nd;

            double* nr = (double*)realloc(ratio_pct, new_cap * sizeof(double));
            if (!nr) {
                verify_ok = 0;
                break;
            }
            ratio_pct = nr;
            cap = new_cap;
        }

        double in_mb = (double)tc.in_size / (1024.0 * 1024.0);
        /* pure-kernel via event profiling (excludes OpenCL dispatch/wakeup latency); fall back to wall-clock */
        unsigned long comp_kus = (tc.kernel_prof_us > 0) ? tc.kernel_prof_us : tc.kernel_exec_us;
        comp_tp[n] = (comp_kus > 0) ? (in_mb * 1000000.0 / (double)comp_kus) : 0.0;
        dec_tp[n] = (dec_kernel_us > 0.0) ? (in_mb * 1000000.0 / dec_kernel_us) : 0.0;
        ratio_pct[n] = (tc.in_size > 0) ? (100.0 * (double)tc.out_size / (double)tc.in_size) : 0.0;
        n++;

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (timer_started && elapsed_sec(&ts0, &ts1) >= bench_seconds && n > 0) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    if (n > 0) {
        printf("Bench Compress : kernel_tp=%.2f MB/s ratio=%.2f%%\n",
            median_double(comp_tp, n), median_double(ratio_pct, n));
        printf("Bench Decompress : kernel_tp=%.2f MB/s verify=%s\n",
            median_double(dec_tp, n), verify_ok ? "OK" : "FAIL");
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_tp);
    free(dec_tp);
    free(ratio_pct);
    free(h_sizes);
    free(h_comp_off);
    free(h_sizes_out);
    free(h_prev_sizes);
    free(h_prev_comp_off);
    if (d_out) clReleaseMemObject(d_out);
    free(input_ref);
    lz4_gpu_workspace_free(&ws);
    clReleaseKernel(kdec);
    clReleaseKernel(kcomp);
    clReleaseProgram(prog);
    ocl_release_runtime();
    return verify_ok ? 0 : 1;
}

/* ============================================================================
 * Two-phase (intra-block parallel) compress bench.
 * Purpose: characterize whether splitting one 64K block across N work-items
 * (build cumulative-prefix dicts + per-segment scan) recovers throughput on
 * occupancy-starved inputs, versus the production single-work-item-per-block
 * baseline. Reuses the trusted ocl_init() (GPU-selecting, device-printing) so
 * it CANNOT silently run on the CPU device. All kernels event-profiled; every
 * two-phase result roundtrip byte-verified before it is reported.
 * ==========================================================================*/
static int run_lz4_tp_bench(const char* input_path, int block_size) {
    struct stat st;
    if (!input_path || stat(input_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "tp-bench error: invalid input file\n");
        return 1;
    }
    if (block_size <= 0) block_size = 64 * 1024;

    size_t in_size = (size_t)st.st_size;
    unsigned char* input_ref = (unsigned char*)malloc(in_size);
    unsigned long rd_us = 0;
    if (!input_ref || lz4_read_file_to_buf(input_path, input_ref, in_size, &rd_us) != 0) {
        free(input_ref);
        fprintf(stderr, "tp-bench error: read failed\n");
        return 1;
    }

    ocl_init();
    if (!ctx || !queue) {
        free(input_ref);
        fprintf(stderr, "tp-bench error: OpenCL init failed\n");
        return 1;
    }

    int hash_log = g_cli_hash_log;
    size_t dict_entries = (size_t)1u << hash_log;
    size_t nblk = (in_size + (size_t)block_size - 1) / (size_t)block_size;

    cl_ulong max_alloc = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);

    cl_int err = CL_SUCCESS;
    const int MEAS = 9, WARM = 2;

    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_size, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "tp-bench: d_input alloc %d\n", err); free(input_ref); return 1; }
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, in_size, input_ref, 0, NULL, NULL);

    /* ---------- BASE: single work-item per 64K block, clear16 (production-best) ---------- */
    double base_comp_mbps = 0.0, base_ratio = 0.0;
    {
        lz4_unset_env("LZ4_GPU_EPOCH32");
        cl_program prog = lz4_load_program(ctx, dev, hash_log, (size_t)block_size);
        if (!prog) { fprintf(stderr, "tp-bench: base prog load failed\n"); return 1; }
        cl_kernel k = clCreateKernel(prog, "lz4_compress_block", &err);
        if (err != CL_SUCCESS || !k) { fprintf(stderr, "tp-bench: base kernel %d\n", err); return 1; }

        int baseMaxOut = block_size + block_size / 255 + 64;
        size_t out_bytes = nblk * (size_t)baseMaxOut;
        size_t pool_bytes = nblk * dict_entries * sizeof(unsigned short); /* clear16 -> ushort entries */
        cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_bytes, NULL, &err);
        cl_mem d_sizes = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nblk * sizeof(cl_uint), NULL, &err);
        cl_mem d_pool = clCreateBuffer(ctx, CL_MEM_READ_WRITE, pool_bytes, NULL, &err);
        if (!d_out || !d_sizes || !d_pool) { fprintf(stderr, "tp-bench: base buffers failed\n"); return 1; }

        cl_int i_nblk = (cl_int)nblk, i_insize = (cl_int)in_size, i_blk = block_size, i_bmax = baseMaxOut,
               i_accel = (g_cli_acceleration > 0 ? g_cli_acceleration : 1), i_gib = 0;
        cl_uint u_lanes = (cl_uint)nblk, u_epoch = 0;
        clSetKernelArg(k, 0, sizeof(cl_mem), &d_input);
        clSetKernelArg(k, 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(k, 2, sizeof(cl_mem), &d_sizes);
        clSetKernelArg(k, 3, sizeof(cl_int), &i_nblk);
        clSetKernelArg(k, 4, sizeof(cl_int), &i_insize);
        clSetKernelArg(k, 5, sizeof(cl_int), &i_blk);
        clSetKernelArg(k, 6, sizeof(cl_int), &i_bmax);
        clSetKernelArg(k, 7, sizeof(cl_int), &i_accel);
        clSetKernelArg(k, 8, sizeof(cl_int), &i_gib);
        clSetKernelArg(k, 9, sizeof(cl_mem), &d_pool);
        clSetKernelArg(k, 10, sizeof(cl_uint), &u_lanes);
        clSetKernelArg(k, 11, sizeof(cl_uint), &u_epoch);

        double us[32]; int nu = 0;
        size_t gws = nblk; size_t lws1 = 1;
        for (int it = 0; it < WARM + MEAS; ++it) {
            cl_event ev;
            err = clEnqueueNDRangeKernel(queue, k, 1, NULL, &gws, &lws1, 0, NULL, &ev);
            if (err != CL_SUCCESS) { fprintf(stderr, "tp-bench: base enqueue %d\n", err); return 1; }
            clWaitForEvents(1, &ev);
            double us1 = event_elapsed_us(ev);
            clReleaseEvent(ev);
            if (it >= WARM && nu < 32) us[nu++] = us1;
        }
        double med = median_double(us, nu);
        base_comp_mbps = (med > 0) ? (double)in_size / med : 0.0;

        cl_uint* hs = (cl_uint*)malloc(nblk * sizeof(cl_uint));
        clEnqueueReadBuffer(queue, d_sizes, CL_TRUE, 0, nblk * sizeof(cl_uint), hs, 0, NULL, NULL);
        size_t csum = 0; for (size_t b = 0; b < nblk; b++) csum += hs[b];
        base_ratio = csum > 0 ? (double)in_size / (double)csum : 0.0;
        free(hs);
        clReleaseMemObject(d_out); clReleaseMemObject(d_sizes); clReleaseMemObject(d_pool);
        clReleaseKernel(k); clReleaseProgram(prog);
    }

    printf("[TP] file=%s size=%zu blocks=%zu block=%dK hashlog=%d\n",
           input_path, in_size, nblk, block_size / 1024, hash_log);
    printf("     BASE(N=1): comp=%.3f MB/s  ratio=%.3f  (occ=%zu WI)\n",
           base_comp_mbps, base_ratio, nblk);

    /* ---------- TWO-PHASE sweep (epoch32) ---------- */
    lz4_set_env("LZ4_GPU_EPOCH32", "1");
    cl_program progtp = lz4_load_program(ctx, dev, hash_log, (size_t)block_size);
    if (!progtp) { fprintf(stderr, "tp-bench: tp prog load failed\n"); return 1; }
    cl_kernel kbuild = clCreateKernel(progtp, "lz4_tp_build_prefix", &err);
    cl_kernel kscan  = clCreateKernel(progtp, "lz4_tp_scan_seg", &err);
    cl_kernel kdec   = clCreateKernel(progtp, "lz4_tp_decompress_segmented", &err);
    if (!kbuild || !kscan || !kdec) { fprintf(stderr, "tp-bench: tp kernels failed %d\n", err); return 1; }

    int Ns[] = {2, 4, 8};
    for (int ni = 0; ni < 3; ++ni) {
        int N = Ns[ni];
        int nk = N - 1;
        int segLenMax = (block_size + N - 1) / N;
        int segMaxOut = segLenMax + segLenMax / 255 + 64;
        size_t prefix_bytes = (size_t)nblk * (size_t)nk * dict_entries * sizeof(cl_uint);
        size_t own_bytes    = (size_t)nblk * (size_t)N  * dict_entries * sizeof(cl_uint);
        size_t out_bytes    = (size_t)nblk * (size_t)N  * (size_t)segMaxOut;
        size_t dec_bytes    = (size_t)nblk * (size_t)block_size;
        size_t meta_cnt     = (size_t)nblk * (size_t)N;

        if (max_alloc && ((cl_ulong)prefix_bytes > max_alloc || (cl_ulong)own_bytes > max_alloc ||
                          (cl_ulong)out_bytes > max_alloc)) {
            size_t big = prefix_bytes > own_bytes ? prefix_bytes : own_bytes;
            printf("     N=%d: SKIPPED (largest buffer %zuMB > CL_DEVICE_MAX_MEM_ALLOC_SIZE %zuMB)\n",
                   N, big / (1024 * 1024), (size_t)(max_alloc / (1024 * 1024)));
            continue;
        }

        cl_mem d_prefix = clCreateBuffer(ctx, CL_MEM_READ_WRITE, prefix_bytes ? prefix_bytes : 4, NULL, &err);
        cl_mem d_own    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, own_bytes, NULL, &err);
        cl_mem d_out    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
        cl_mem d_sizes  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, meta_cnt * sizeof(cl_uint), NULL, &err);
        cl_mem d_coff   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, meta_cnt * sizeof(cl_uint), NULL, &err);
        cl_mem d_decout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, dec_bytes, NULL, &err);
        cl_mem d_sout   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nblk * sizeof(cl_uint), NULL, &err);
        if (!d_prefix || !d_own || !d_out || !d_sizes || !d_coff || !d_decout || !d_sout) {
            printf("     N=%d: buffer alloc failed (%d)\n", N, err);
            if (d_prefix) clReleaseMemObject(d_prefix);
            if (d_own) clReleaseMemObject(d_own);
            if (d_out) clReleaseMemObject(d_out);
            if (d_sizes) clReleaseMemObject(d_sizes);
            if (d_coff) clReleaseMemObject(d_coff);
            if (d_decout) clReleaseMemObject(d_decout);
            if (d_sout) clReleaseMemObject(d_sout);
            continue;
        }

        { /* zero epoch pools (fresh CL buffers are uninitialized) */
            cl_uint zero = 0;
            if (prefix_bytes) clEnqueueFillBuffer(queue, d_prefix, &zero, sizeof(zero), 0, prefix_bytes, 0, NULL, NULL);
            clEnqueueFillBuffer(queue, d_own, &zero, sizeof(zero), 0, own_bytes, 0, NULL, NULL);
            clFinish(queue);
        }
        { /* comp_offsets static: segment gg lives at gg*segMaxOut in d_out */
            cl_uint* coff = (cl_uint*)malloc(meta_cnt * sizeof(cl_uint));
            for (size_t g = 0; g < meta_cnt; ++g) coff[g] = (cl_uint)(g * (size_t)segMaxOut);
            clEnqueueWriteBuffer(queue, d_coff, CL_TRUE, 0, meta_cnt * sizeof(cl_uint), coff, 0, NULL, NULL);
            free(coff);
        }

        cl_int i_nblk = (cl_int)nblk, i_insize = (cl_int)in_size, i_blk = block_size,
               i_segmax = segMaxOut, i_N = N;
        cl_uint u_blk = (cl_uint)block_size, u_tot = (cl_uint)nblk;

        clSetKernelArg(kbuild, 0, sizeof(cl_mem), &d_input);
        clSetKernelArg(kbuild, 1, sizeof(cl_mem), &d_prefix);
        clSetKernelArg(kbuild, 2, sizeof(cl_int), &i_nblk);
        clSetKernelArg(kbuild, 3, sizeof(cl_int), &i_insize);
        clSetKernelArg(kbuild, 4, sizeof(cl_int), &i_blk);
        clSetKernelArg(kbuild, 5, sizeof(cl_int), &i_N);

        clSetKernelArg(kscan, 0, sizeof(cl_mem), &d_input);
        clSetKernelArg(kscan, 1, sizeof(cl_mem), &d_out);
        clSetKernelArg(kscan, 2, sizeof(cl_mem), &d_sizes);
        clSetKernelArg(kscan, 3, sizeof(cl_mem), &d_own);
        clSetKernelArg(kscan, 4, sizeof(cl_mem), &d_prefix);
        clSetKernelArg(kscan, 5, sizeof(cl_int), &i_nblk);
        clSetKernelArg(kscan, 6, sizeof(cl_int), &i_insize);
        clSetKernelArg(kscan, 7, sizeof(cl_int), &i_blk);
        clSetKernelArg(kscan, 8, sizeof(cl_int), &i_segmax);
        clSetKernelArg(kscan, 9, sizeof(cl_int), &i_N);

        clSetKernelArg(kdec, 0, sizeof(cl_mem), &d_out);
        clSetKernelArg(kdec, 1, sizeof(cl_mem), &d_decout);
        clSetKernelArg(kdec, 2, sizeof(cl_mem), &d_coff);
        clSetKernelArg(kdec, 3, sizeof(cl_mem), &d_sizes);
        clSetKernelArg(kdec, 4, sizeof(cl_mem), &d_sout);
        clSetKernelArg(kdec, 5, sizeof(cl_uint), &u_blk);
        clSetKernelArg(kdec, 6, sizeof(cl_int), &i_N);
        clSetKernelArg(kdec, 7, sizeof(cl_uint), &u_tot);

        size_t g_build = (size_t)nblk * (size_t)nk * 4; /* LZ4_TP_BUILD_CHUNKS=4 */
        size_t g_scan  = (size_t)nblk * (size_t)N;
        size_t g_dec   = nblk;
        size_t lws1 = 1;

        double comp_us[32]; int ncu = 0;
        cl_uint epoch = 1;
        double dec_us = 0.0;
        int verify_ok = -1;

        for (int it = 0; it < WARM + MEAS; ++it, ++epoch) {
            cl_uint ep = epoch;
            double bu = 0.0, sc = 0.0;
            if (nk >= 1) {
                clSetKernelArg(kbuild, 6, sizeof(cl_uint), &ep);
                cl_event evb;
                err = clEnqueueNDRangeKernel(queue, kbuild, 1, NULL, &g_build, &lws1, 0, NULL, &evb);
                if (err != CL_SUCCESS) { fprintf(stderr, "N=%d build enqueue %d\n", N, err); verify_ok = 0; break; }
                clWaitForEvents(1, &evb); bu = event_elapsed_us(evb); clReleaseEvent(evb);
            }
            clSetKernelArg(kscan, 10, sizeof(cl_uint), &ep);
            cl_event evs;
            err = clEnqueueNDRangeKernel(queue, kscan, 1, NULL, &g_scan, &lws1, 0, NULL, &evs);
            if (err != CL_SUCCESS) { fprintf(stderr, "N=%d scan enqueue %d\n", N, err); verify_ok = 0; break; }
            clWaitForEvents(1, &evs); sc = event_elapsed_us(evs); clReleaseEvent(evs);
            if (it >= WARM && ncu < 32) comp_us[ncu++] = bu + sc;

            if (it == WARM) { /* decode + roundtrip verify once */
                cl_event evd;
                err = clEnqueueNDRangeKernel(queue, kdec, 1, NULL, &g_dec, &lws1, 0, NULL, &evd);
                if (err != CL_SUCCESS) { fprintf(stderr, "N=%d dec enqueue %d\n", N, err); verify_ok = 0; break; }
                clWaitForEvents(1, &evd); dec_us = event_elapsed_us(evd); clReleaseEvent(evd);
                unsigned char* dec = (unsigned char*)malloc(dec_bytes);
                cl_uint* sout = (cl_uint*)malloc(nblk * sizeof(cl_uint));
                clEnqueueReadBuffer(queue, d_decout, CL_TRUE, 0, dec_bytes, dec, 0, NULL, NULL);
                clEnqueueReadBuffer(queue, d_sout, CL_TRUE, 0, nblk * sizeof(cl_uint), sout, 0, NULL, NULL);
                verify_ok = 1;
                for (size_t b = 0; b < nblk; b++) {
                    size_t start = b * (size_t)block_size;
                    size_t L = in_size - start; if (L > (size_t)block_size) L = block_size;
                    if (sout[b] != (cl_uint)L) { verify_ok = 0; break; }
                    if (memcmp(dec + b * (size_t)block_size, input_ref + start, L) != 0) { verify_ok = 0; break; }
                }
                free(dec); free(sout);
            }
        }

        double tp_comp_mbps = 0.0, tp_ratio = 0.0, tp_dec_mbps = 0.0, speedup = 0.0;
        if (ncu > 0) {
            double med = median_double(comp_us, ncu);
            tp_comp_mbps = med > 0 ? (double)in_size / med : 0.0;
            tp_dec_mbps  = dec_us > 0 ? (double)in_size / dec_us : 0.0;
            speedup = base_comp_mbps > 0 ? tp_comp_mbps / base_comp_mbps : 0.0;
            cl_uint* sz = (cl_uint*)malloc(meta_cnt * sizeof(cl_uint));
            clEnqueueReadBuffer(queue, d_sizes, CL_TRUE, 0, meta_cnt * sizeof(cl_uint), sz, 0, NULL, NULL);
            size_t csum = 0; for (size_t g = 0; g < meta_cnt; g++) csum += sz[g];
            tp_ratio = csum > 0 ? (double)in_size / (double)csum : 0.0;
            free(sz);
        }
        printf("     N=%d: comp=%.3f MB/s  ratio=%.3f  speedup=%.2fx  dec=%.3f MB/s  verify=%s  (occ=%zu WI)\n",
               N, tp_comp_mbps, tp_ratio, speedup, tp_dec_mbps,
               verify_ok == 1 ? "OK" : (verify_ok == 0 ? "FAIL" : "?"), meta_cnt);

        clReleaseMemObject(d_prefix); clReleaseMemObject(d_own); clReleaseMemObject(d_out);
        clReleaseMemObject(d_sizes); clReleaseMemObject(d_coff);
        clReleaseMemObject(d_decout); clReleaseMemObject(d_sout);
    }

    clReleaseKernel(kbuild); clReleaseKernel(kscan); clReleaseKernel(kdec);
    clReleaseProgram(progtp);
    clReleaseMemObject(d_input);
    lz4_unset_env("LZ4_GPU_EPOCH32");
    free(input_ref);
    ocl_release_runtime();
    return 0;
}

/* ============================================================================
 * I1: two-phase as a REAL codec — compress/decompress files to/from a
 * self-describing frame. Unlike run_lz4_tp_bench (in-memory measurement only),
 * these produce a real .lz4tp file that any device can decode.
 *
 * Frame layout (little-endian; all our platforms x86-64 + aarch64 are LE):
 *   char   magic[8] = "LZ4TP1\0\0"
 *   u32    N                 (segments per 64K block)
 *   u32    hash_log
 *   u32    block_size
 *   u32    nblk
 *   u64    orig_size         (uncompressed length)
 *   u32    comp_sizes[nblk*N] (per-segment compressed size, gg = blk*N+seg)
 *   bytes  payload           (all segments concatenated in gg order, compacted)
 * ==========================================================================*/
#define LZ4TP_MAGIC "LZ4TP1\0\0"

static int lz4_tp_replace_path(const char* temp_path, const char* output_path);

static int lz4_tp_write_metrics(const char* operation, size_t input_bytes, size_t output_bytes,
                                int N, int block_size, int hash_log,
                                size_t chunk_blocks,
                                uint64_t kernel_us, uint64_t no_ocl_us,
                                uint64_t total_us, uint64_t ocl_setup_us) {
    if (!g_cli_metrics_path) return 0;
    char temp_path[PATH_MAX];
    if (snprintf(temp_path, sizeof(temp_path), "%s.tmp.%lu", g_cli_metrics_path,
                 (unsigned long)getpid()) >= (int)sizeof(temp_path)) {
        fprintf(stderr, "metrics path is too long\n");
        return -1;
    }
    FILE* f = fopen(temp_path, "w");
    if (!f) { fprintf(stderr, "cannot write metrics file %s\n", temp_path); return -1; }
    double ratio_pct = 0.0;
    if (strcmp(operation, "compress") == 0) {
        ratio_pct = input_bytes ? 100.0 * (double)output_bytes / (double)input_bytes : 0.0;
    } else {
        ratio_pct = output_bytes ? 100.0 * (double)input_bytes / (double)output_bytes : 0.0;
    }
    int ok = fprintf(
        f,
        "{\n"
        "  \"schema\": \"heterolz.operation-metric.v1\",\n"
        "  \"operation\": \"%s\",\n"
        "  \"input_bytes\": %zu,\n"
        "  \"output_bytes\": %zu,\n"
        "  \"n\": %d,\n"
        "  \"block_size\": %d,\n"
        "  \"hash_log\": %d,\n"
        "  \"chunk_blocks\": %zu,\n"
        "  \"kernel_us\": %llu,\n"
        "  \"no_ocl_us\": %llu,\n"
        "  \"total_us\": %llu,\n"
        "  \"ocl_setup_us\": %llu,\n"
        "  \"ratio_pct\": %.9f\n"
        "}\n",
        operation, input_bytes, output_bytes, N, block_size, hash_log, chunk_blocks,
        (unsigned long long)kernel_us, (unsigned long long)no_ocl_us,
        (unsigned long long)total_us, (unsigned long long)ocl_setup_us, ratio_pct) >= 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok) { remove(temp_path); return -1; }
    if (lz4_tp_replace_path(temp_path, g_cli_metrics_path) != 0) {
        fprintf(stderr, "cannot finalize metrics file %s\n", g_cli_metrics_path);
        remove(temp_path);
        return -1;
    }
    return 0;
}

static int lz4_tp_size_mul(size_t a, size_t b, size_t* out) {
    if (!out || (a != 0 && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int lz4_tp_size_add(size_t a, size_t b, size_t* out) {
    if (!out || b > SIZE_MAX - a) return -1;
    *out = a + b;
    return 0;
}

static int lz4_tp_write_exact(FILE* f, const void* data, size_t size) {
    return size == 0 || (f && fwrite(data, 1, size, f) == size);
}

static int lz4_tp_read_exact(FILE* f, void* data, size_t size) {
    return size == 0 || (f && fread(data, 1, size, f) == size);
}

static int lz4_tp_make_temp_path(const char* output_path, char* temp_path, size_t capacity) {
    if (!output_path || !*output_path || !temp_path || capacity == 0 ||
        snprintf(temp_path, capacity, "%s.tmp.%lu", output_path,
                 (unsigned long)getpid()) >= (int)capacity) {
        fprintf(stderr, "LZ4TP1 output path is too long\n");
        return -1;
    }
    remove(temp_path);
    return 0;
}

static int lz4_tp_replace_path(const char* temp_path, const char* output_path) {
    if (!temp_path || !output_path) return -1;
#if defined(_WIN32)
    return MoveFileExA(temp_path, output_path,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(temp_path, output_path);
#endif
}

static int lz4_tp_finalize_temp_output(FILE** fp, const char* temp_path, const char* output_path) {
    if (!fp || !*fp) return -1;
    if (fflush(*fp) != 0 || fclose(*fp) != 0) {
        *fp = NULL;
        remove(temp_path);
        fprintf(stderr, "cannot finalize temporary output %s\n", temp_path);
        return -1;
    }
    *fp = NULL;
    if (lz4_tp_replace_path(temp_path, output_path) != 0) {
        remove(temp_path);
        fprintf(stderr, "cannot rename temporary output to %s\n", output_path);
        return -1;
    }
    return 0;
}

static size_t lz4_tp_apply_test_chunk_override(size_t safe_limit) {
    const char* value = getenv("LZ4TP_TEST_CHUNK_BLOCKS");
    if (!value || !*value || safe_limit == 0) return safe_limit;
    char* end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed == 0 || parsed > (unsigned long long)SIZE_MAX)
        return safe_limit;
    if ((size_t)parsed < safe_limit) return (size_t)parsed;
    return safe_limit;
}

static size_t lz4_tp_device_alloc_budget(cl_ulong max_alloc, size_t fallback) {
    if (!max_alloc) return fallback;
    cl_ulong budget = (max_alloc / 4) * 3;
    if (budget > (cl_ulong)SIZE_MAX) return SIZE_MAX;
    return (size_t)budget;
}

static size_t lz4_tp_device_global_budget(cl_ulong global_mem, size_t fallback) {
    size_t cap = (size_t)512 * 1024 * 1024;
    if (!global_mem) return fallback < cap ? fallback : cap;
    cl_ulong budget = global_mem / 4;
    if (budget > (cl_ulong)cap) return cap;
    return (size_t)budget;
}

static size_t lz4_tp_choose_chunk_blocks(size_t total_blocks, int N, int block_size,
                                         size_t dict_entries) {
    if (total_blocks == 0 || N < 1 || block_size < 1) return 0;
    const size_t hard_cap = 512;
    size_t seg_len_max = ((size_t)block_size + (size_t)N - 1) / (size_t)N;
    size_t seg_max_out = 0, per_prefix = 0, per_own = 0, per_output = 0;
    size_t per_total = 0, tmp = 0;
    if (lz4_tp_size_add(seg_len_max, seg_len_max / 255, &seg_max_out) != 0 ||
        lz4_tp_size_add(seg_max_out, 64, &seg_max_out) != 0 ||
        lz4_tp_size_mul((size_t)(N - 1), dict_entries, &tmp) != 0 ||
        lz4_tp_size_mul(tmp, sizeof(cl_uint), &per_prefix) != 0 ||
        lz4_tp_size_mul((size_t)N, dict_entries, &tmp) != 0 ||
        lz4_tp_size_mul(tmp, sizeof(cl_uint), &per_own) != 0 ||
        lz4_tp_size_mul((size_t)N, seg_max_out, &per_output) != 0 ||
        lz4_tp_size_add((size_t)block_size, per_prefix, &per_total) != 0 ||
        lz4_tp_size_add(per_total, per_own, &per_total) != 0 ||
        lz4_tp_size_add(per_total, per_output, &per_total) != 0 ||
        lz4_tp_size_mul((size_t)N, sizeof(cl_uint), &tmp) != 0 ||
        lz4_tp_size_add(per_total, tmp, &per_total) != 0 || per_output == 0) {
        return 0;
    }

    size_t limit = total_blocks < hard_cap ? total_blocks : hard_cap;
    size_t by_offset = (size_t)UINT_MAX / per_output;
    if (by_offset < limit) limit = by_offset;

    cl_ulong max_alloc = 0, global_mem = 0;
    if (dev) {
        clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
        clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    }
    size_t alloc_budget = lz4_tp_device_alloc_budget(max_alloc, (size_t)256 * 1024 * 1024);
    size_t components[] = {(size_t)block_size, per_prefix, per_own, per_output};
    for (size_t i = 0; i < sizeof(components) / sizeof(components[0]); i++) {
        if (components[i] == 0) continue;
        size_t by_alloc = alloc_budget / components[i];
        if (by_alloc < limit) limit = by_alloc;
    }
    size_t global_budget = lz4_tp_device_global_budget(global_mem, (size_t)512 * 1024 * 1024);
    size_t by_global = global_budget / per_total;
    if (by_global < limit) limit = by_global;
    return lz4_tp_apply_test_chunk_override(limit);
}

static size_t lz4_tp_choose_decode_chunk_blocks(size_t total_blocks, int N, int block_size,
                                                size_t seg_max_out) {
    if (total_blocks == 0 || N < 1 || block_size < 1 || seg_max_out == 0) return 0;
    const size_t hard_cap = 512;
    size_t per_payload = 0, per_meta = 0, per_total = 0, tmp = 0;
    if (lz4_tp_size_mul((size_t)N, seg_max_out, &per_payload) != 0 ||
        lz4_tp_size_mul((size_t)N, 2 * sizeof(cl_uint), &per_meta) != 0 ||
        lz4_tp_size_add(per_payload, (size_t)block_size, &per_total) != 0 ||
        lz4_tp_size_add(per_total, per_meta, &per_total) != 0 ||
        lz4_tp_size_add(per_total, sizeof(cl_uint), &per_total) != 0 || per_payload == 0) {
        return 0;
    }
    size_t limit = total_blocks < hard_cap ? total_blocks : hard_cap;
    size_t by_offset = (size_t)UINT_MAX / per_payload;
    if (by_offset < limit) limit = by_offset;

    cl_ulong max_alloc = 0, global_mem = 0;
    if (dev) {
        clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
        clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    }
    size_t alloc_budget = lz4_tp_device_alloc_budget(max_alloc, (size_t)256 * 1024 * 1024);
    size_t components[] = {per_payload, (size_t)block_size, per_meta, sizeof(cl_uint)};
    for (size_t i = 0; i < sizeof(components) / sizeof(components[0]); i++) {
        if (components[i] == 0) continue;
        tmp = alloc_budget / components[i];
        if (tmp < limit) limit = tmp;
    }
    size_t global_budget = lz4_tp_device_global_budget(global_mem, (size_t)512 * 1024 * 1024);
    tmp = global_budget / per_total;
    if (tmp < limit) limit = tmp;
    return lz4_tp_apply_test_chunk_override(limit);
}

static int lz4_tp_compress_chunk(const unsigned char* input, size_t input_size,
                                 size_t chunk_nblk, int N, int block_size,
                                 size_t dict_entries, int seg_max_out,
                                 cl_kernel kbuild, cl_kernel kscan,
                                 cl_uint* sizes_out, unsigned char* payload,
                                 size_t payload_capacity, size_t* payload_used,
                                 uint64_t* kernel_us) {
    int status = 1;
    int nk = N - 1;
    size_t meta_cnt = 0, prefix_bytes = 0, own_bytes = 0, out_bytes = 0, tmp = 0;
    cl_int err = CL_SUCCESS;
    cl_mem d_input = NULL, d_prefix = NULL, d_own = NULL, d_out = NULL, d_sizes = NULL;
    unsigned char* padded = NULL;
    cl_event build_event = NULL, scan_event = NULL;

    if (!input || input_size == 0 || input_size > (size_t)INT_MAX ||
        chunk_nblk == 0 || chunk_nblk > (size_t)INT_MAX || N < 1 ||
        block_size < 1 || seg_max_out < 1 || !sizes_out || !payload ||
        !payload_used || !kernel_us ||
        lz4_tp_size_mul(chunk_nblk, (size_t)N, &meta_cnt) != 0 ||
        lz4_tp_size_mul(chunk_nblk, (size_t)nk, &tmp) != 0 ||
        lz4_tp_size_mul(tmp, dict_entries, &tmp) != 0 ||
        lz4_tp_size_mul(tmp, sizeof(cl_uint), &prefix_bytes) != 0 ||
        lz4_tp_size_mul(meta_cnt, dict_entries, &tmp) != 0 ||
        lz4_tp_size_mul(tmp, sizeof(cl_uint), &own_bytes) != 0 ||
        lz4_tp_size_mul(meta_cnt, (size_t)seg_max_out, &out_bytes) != 0 ||
        out_bytes > payload_capacity) {
        fprintf(stderr, "tp-compress: invalid chunk dimensions\n");
        goto done;
    }

    d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY, input_size, NULL, &err);
    d_prefix = clCreateBuffer(ctx, CL_MEM_READ_WRITE, prefix_bytes ? prefix_bytes : 4, NULL, &err);
    d_own = clCreateBuffer(ctx, CL_MEM_READ_WRITE, own_bytes, NULL, &err);
    d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    d_sizes = clCreateBuffer(ctx, CL_MEM_READ_WRITE, meta_cnt * sizeof(cl_uint), NULL, &err);
    if (!d_input || !d_prefix || !d_own || !d_out || !d_sizes) {
        fprintf(stderr, "tp-compress: chunk buffer alloc failed (%d)\n", err);
        goto done;
    }
    if (clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL) != CL_SUCCESS) goto done;
    {
        cl_uint zero = 0;
        if (prefix_bytes && clEnqueueFillBuffer(queue, d_prefix, &zero, sizeof(zero), 0,
                                                prefix_bytes, 0, NULL, NULL) != CL_SUCCESS) goto done;
        if (clEnqueueFillBuffer(queue, d_own, &zero, sizeof(zero), 0,
                                own_bytes, 0, NULL, NULL) != CL_SUCCESS) goto done;
        if (clFinish(queue) != CL_SUCCESS) goto done;
    }

    cl_int i_nblk = (cl_int)chunk_nblk, i_insize = (cl_int)input_size, i_blk = block_size,
           i_segmax = seg_max_out, i_N = N;
    cl_uint epoch = 1;
    size_t lws = 1;
    cl_int arg_err = CL_SUCCESS;
    arg_err |= clSetKernelArg(kbuild, 0, sizeof(cl_mem), &d_input);
    arg_err |= clSetKernelArg(kbuild, 1, sizeof(cl_mem), &d_prefix);
    arg_err |= clSetKernelArg(kbuild, 2, sizeof(cl_int), &i_nblk);
    arg_err |= clSetKernelArg(kbuild, 3, sizeof(cl_int), &i_insize);
    arg_err |= clSetKernelArg(kbuild, 4, sizeof(cl_int), &i_blk);
    arg_err |= clSetKernelArg(kbuild, 5, sizeof(cl_int), &i_N);
    arg_err |= clSetKernelArg(kbuild, 6, sizeof(cl_uint), &epoch);
    arg_err |= clSetKernelArg(kscan, 0, sizeof(cl_mem), &d_input);
    arg_err |= clSetKernelArg(kscan, 1, sizeof(cl_mem), &d_out);
    arg_err |= clSetKernelArg(kscan, 2, sizeof(cl_mem), &d_sizes);
    arg_err |= clSetKernelArg(kscan, 3, sizeof(cl_mem), &d_own);
    arg_err |= clSetKernelArg(kscan, 4, sizeof(cl_mem), &d_prefix);
    arg_err |= clSetKernelArg(kscan, 5, sizeof(cl_int), &i_nblk);
    arg_err |= clSetKernelArg(kscan, 6, sizeof(cl_int), &i_insize);
    arg_err |= clSetKernelArg(kscan, 7, sizeof(cl_int), &i_blk);
    arg_err |= clSetKernelArg(kscan, 8, sizeof(cl_int), &i_segmax);
    arg_err |= clSetKernelArg(kscan, 9, sizeof(cl_int), &i_N);
    arg_err |= clSetKernelArg(kscan, 10, sizeof(cl_uint), &epoch);
    if (arg_err != CL_SUCCESS) goto done;

    if (nk >= 1) {
        size_t g_build = chunk_nblk * (size_t)nk * 4;
        err = clEnqueueNDRangeKernel(queue, kbuild, 1, NULL, &g_build, &lws, 0, NULL, &build_event);
        if (err != CL_SUCCESS) goto done;
    }
    size_t g_scan = chunk_nblk * (size_t)N;
    err = clEnqueueNDRangeKernel(queue, kscan, 1, NULL, &g_scan, &lws, 0, NULL, &scan_event);
    if (err != CL_SUCCESS) goto done;
    if (clFinish(queue) != CL_SUCCESS) goto done;
    if (build_event) *kernel_us += (uint64_t)(event_elapsed_us(build_event) + 0.5);
    if (scan_event) *kernel_us += (uint64_t)(event_elapsed_us(scan_event) + 0.5);

    padded = (unsigned char*)malloc(out_bytes);
    if (!padded) goto done;
    if (clEnqueueReadBuffer(queue, d_sizes, CL_TRUE, 0, meta_cnt * sizeof(cl_uint),
                            sizes_out, 0, NULL, NULL) != CL_SUCCESS) goto done;
    if (clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, out_bytes,
                            padded, 0, NULL, NULL) != CL_SUCCESS) goto done;
    for (size_t g = 0; g < meta_cnt; g++) {
        size_t size = sizes_out[g];
        if (size == 0 || size > (size_t)seg_max_out || size > payload_capacity ||
            *payload_used > payload_capacity - size) {
            fprintf(stderr, "tp-compress: invalid chunk segment size\n");
            goto done;
        }
        memcpy(payload + *payload_used, padded + g * (size_t)seg_max_out, size);
        *payload_used += size;
    }
    status = 0;

done:
    if (build_event) clReleaseEvent(build_event);
    if (scan_event) clReleaseEvent(scan_event);
    free(padded);
    if (d_input) clReleaseMemObject(d_input);
    if (d_prefix) clReleaseMemObject(d_prefix);
    if (d_own) clReleaseMemObject(d_own);
    if (d_out) clReleaseMemObject(d_out);
    if (d_sizes) clReleaseMemObject(d_sizes);
    return status;
}

static int lz4_tp_decompress_chunk(const unsigned char* payload, size_t payload_size,
                                   const cl_uint* comp_off, const cl_uint* comp_sizes,
                                   size_t chunk_nblk, int N, int block_size,
                                   cl_kernel kdec, unsigned char* dec, cl_uint* sizes_out,
                                   uint64_t* kernel_us) {
    int status = 1;
    size_t meta_cnt = 0, dec_bytes = 0;
    cl_int err = CL_SUCCESS;
    cl_mem d_payload = NULL, d_coff = NULL, d_csz = NULL, d_decout = NULL, d_sout = NULL;
    cl_event dec_event = NULL;
    if (!payload || payload_size == 0 || payload_size > (size_t)UINT_MAX ||
        !comp_off || !comp_sizes || !dec || !sizes_out || !kernel_us ||
        chunk_nblk == 0 || chunk_nblk > (size_t)INT_MAX || N < 1 || block_size < 1 ||
        lz4_tp_size_mul(chunk_nblk, (size_t)N, &meta_cnt) != 0 ||
        lz4_tp_size_mul(chunk_nblk, (size_t)block_size, &dec_bytes) != 0) {
        fprintf(stderr, "tp-decompress: invalid chunk dimensions\n");
        goto done;
    }

    d_payload = clCreateBuffer(ctx, CL_MEM_READ_ONLY, payload_size, NULL, &err);
    d_coff = clCreateBuffer(ctx, CL_MEM_READ_ONLY, meta_cnt * sizeof(cl_uint), NULL, &err);
    d_csz = clCreateBuffer(ctx, CL_MEM_READ_ONLY, meta_cnt * sizeof(cl_uint), NULL, &err);
    d_decout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, dec_bytes, NULL, &err);
    d_sout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, chunk_nblk * sizeof(cl_uint), NULL, &err);
    if (!d_payload || !d_coff || !d_csz || !d_decout || !d_sout) {
        fprintf(stderr, "tp-decompress: chunk buffer alloc failed (%d)\n", err);
        goto done;
    }
    if (clEnqueueWriteBuffer(queue, d_payload, CL_TRUE, 0, payload_size, payload, 0, NULL, NULL) != CL_SUCCESS ||
        clEnqueueWriteBuffer(queue, d_coff, CL_TRUE, 0, meta_cnt * sizeof(cl_uint), comp_off, 0, NULL, NULL) != CL_SUCCESS ||
        clEnqueueWriteBuffer(queue, d_csz, CL_TRUE, 0, meta_cnt * sizeof(cl_uint), comp_sizes, 0, NULL, NULL) != CL_SUCCESS) {
        goto done;
    }

    cl_uint u_blk = (cl_uint)block_size, u_tot = (cl_uint)chunk_nblk;
    cl_int i_N = N;
    cl_int arg_err = CL_SUCCESS;
    arg_err |= clSetKernelArg(kdec, 0, sizeof(cl_mem), &d_payload);
    arg_err |= clSetKernelArg(kdec, 1, sizeof(cl_mem), &d_decout);
    arg_err |= clSetKernelArg(kdec, 2, sizeof(cl_mem), &d_coff);
    arg_err |= clSetKernelArg(kdec, 3, sizeof(cl_mem), &d_csz);
    arg_err |= clSetKernelArg(kdec, 4, sizeof(cl_mem), &d_sout);
    arg_err |= clSetKernelArg(kdec, 5, sizeof(cl_uint), &u_blk);
    arg_err |= clSetKernelArg(kdec, 6, sizeof(cl_int), &i_N);
    arg_err |= clSetKernelArg(kdec, 7, sizeof(cl_uint), &u_tot);
    if (arg_err != CL_SUCCESS) goto done;

    size_t g_dec = chunk_nblk, lws = 1;
    err = clEnqueueNDRangeKernel(queue, kdec, 1, NULL, &g_dec, &lws, 0, NULL, &dec_event);
    if (err != CL_SUCCESS || clFinish(queue) != CL_SUCCESS) goto done;
    *kernel_us += (uint64_t)(event_elapsed_us(dec_event) + 0.5);
    if (clEnqueueReadBuffer(queue, d_decout, CL_TRUE, 0, dec_bytes, dec, 0, NULL, NULL) != CL_SUCCESS ||
        clEnqueueReadBuffer(queue, d_sout, CL_TRUE, 0, chunk_nblk * sizeof(cl_uint), sizes_out, 0, NULL, NULL) != CL_SUCCESS) {
        goto done;
    }
    status = 0;

done:
    if (dec_event) clReleaseEvent(dec_event);
    if (d_payload) clReleaseMemObject(d_payload);
    if (d_coff) clReleaseMemObject(d_coff);
    if (d_csz) clReleaseMemObject(d_csz);
    if (d_decout) clReleaseMemObject(d_decout);
    if (d_sout) clReleaseMemObject(d_sout);
    return status;
}

static int lz4_tp_compress_to_file(const char* input_path, const char* output_path,
                                   int N, int block_size,
                                   uint64_t pre_total_us, uint64_t pre_ocl_setup_us) {
    uint64_t total_start = get_us();
    int status = 1, env_set = 0, temp_active = 0;
    struct stat st;
    FILE* input_file = NULL;
    FILE* output_file = NULL;
    cl_program prog = NULL;
    cl_kernel kbuild = NULL, kscan = NULL;
    cl_uint* sizes = NULL;
    unsigned char* input_chunk = NULL;
    unsigned char* payload_chunk = NULL;
    char temp_path[PATH_MAX] = {0};
    if (!input_path || !output_path || !*output_path || paths_identify_same_file(input_path, output_path) ||
        (g_cli_metrics_path && (paths_identify_same_file(g_cli_metrics_path, input_path) ||
                                paths_identify_same_file(g_cli_metrics_path, output_path))) ||
        stat(input_path, &st) != 0 || st.st_size < 0 ||
        (uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "tp-compress: invalid input\n"); return 1;
    }
    if ((N != 1 && N != 2 && N != 4 && N != 8) || block_size < 1 ||
        g_cli_hash_log < 11 || g_cli_hash_log > 15) {
        fprintf(stderr, "tp-compress: unsupported frame parameters\n"); return 1;
    }

    size_t orig_size = (size_t)st.st_size;
    if (lz4_tp_make_temp_path(output_path, temp_path, sizeof(temp_path)) != 0) return 1;
    temp_active = 1;
    if (orig_size == 0) {
        output_file = fopen(temp_path, "wb");
        if (!output_file) { fprintf(stderr, "tp-compress: cannot open %s\n", temp_path); goto done; }
        cl_uint u_N = 1, u_hl = (cl_uint)g_cli_hash_log;
        cl_uint u_bs = (cl_uint)block_size, u_nblk = 0;
        unsigned long long u_orig = 0;
        if (!lz4_tp_write_exact(output_file, LZ4TP_MAGIC, 8) ||
            !lz4_tp_write_exact(output_file, &u_N, 4) ||
            !lz4_tp_write_exact(output_file, &u_hl, 4) ||
            !lz4_tp_write_exact(output_file, &u_bs, 4) ||
            !lz4_tp_write_exact(output_file, &u_nblk, 4) ||
            !lz4_tp_write_exact(output_file, &u_orig, 8) ||
            lz4_tp_finalize_temp_output(&output_file, temp_path, output_path) != 0) {
            fprintf(stderr, "tp-compress: write failed\n"); goto done;
        }
        temp_active = 0;
        uint64_t total_end = get_us();
        uint64_t total_us = total_end - total_start + pre_total_us;
        uint64_t host_pre_us = pre_total_us >= pre_ocl_setup_us ?
                               pre_total_us - pre_ocl_setup_us : 0;
        if (lz4_tp_write_metrics("compress", 0, 32, 1, block_size, g_cli_hash_log,
                                 0, 0, total_end - total_start + host_pre_us,
                                 total_us, pre_ocl_setup_us) != 0) goto done;
        fprintf(stderr, "[tp-compress] %s -> %s : empty LZ4TP1 frame\n", input_path, output_path);
        status = 0;
        goto done;
    }

    uint64_t ocl_start = get_us();
    if (lz4_set_env("LZ4_GPU_EPOCH32", "1") != 0) {
        fprintf(stderr, "tp-compress: cannot set epoch mode\n"); goto done;
    }
    env_set = 1;
    ocl_init();
    if (!ctx || !queue) { fprintf(stderr, "tp-compress: OpenCL init failed\n"); goto done; }

    int hash_log = g_cli_hash_log;
    size_t dict_entries = (size_t)1u << hash_log;
    size_t nblk = orig_size / (size_t)block_size + (orig_size % (size_t)block_size != 0);
    if (nblk > (size_t)UINT_MAX) { fprintf(stderr, "tp-compress: too many blocks\n"); goto done; }
    size_t seg_len_max = ((size_t)block_size + (size_t)N - 1) / (size_t)N;
    size_t seg_max_out = 0, meta_cnt = 0;
    if (lz4_tp_size_add(seg_len_max, seg_len_max / 255, &seg_max_out) != 0 ||
        lz4_tp_size_add(seg_max_out, 64, &seg_max_out) != 0 || seg_max_out > (size_t)INT_MAX ||
        lz4_tp_size_mul(nblk, (size_t)N, &meta_cnt) != 0 ||
        meta_cnt > SIZE_MAX / sizeof(cl_uint)) {
        fprintf(stderr, "tp-compress: frame dimensions overflow\n"); goto done;
    }

    cl_int err = CL_SUCCESS;
    prog = lz4_load_program(ctx, dev, hash_log, (size_t)block_size);
    if (!prog) { fprintf(stderr, "tp-compress: prog load failed\n"); goto done; }
    kbuild = clCreateKernel(prog, "lz4_tp_build_prefix", &err);
    kscan = clCreateKernel(prog, "lz4_tp_scan_seg", &err);
    if (!kbuild || !kscan) { fprintf(stderr, "tp-compress: kernels failed %d\n", err); goto done; }
    uint64_t work_start = get_us();

    size_t chunk_blocks = lz4_tp_choose_chunk_blocks(nblk, N, block_size, dict_entries);
    size_t chunk_input_capacity = 0, chunk_meta_capacity = 0, chunk_payload_capacity = 0;
    if (chunk_blocks == 0 ||
        lz4_tp_size_mul(chunk_blocks, (size_t)block_size, &chunk_input_capacity) != 0 ||
        chunk_input_capacity > (size_t)INT_MAX ||
        lz4_tp_size_mul(chunk_blocks, (size_t)N, &chunk_meta_capacity) != 0 ||
        lz4_tp_size_mul(chunk_meta_capacity, seg_max_out, &chunk_payload_capacity) != 0 ||
        chunk_payload_capacity > (size_t)UINT_MAX) {
        fprintf(stderr, "tp-compress: no safe OpenCL chunk size\n"); goto done;
    }
    sizes = (cl_uint*)calloc(meta_cnt, sizeof(cl_uint));
    input_chunk = (unsigned char*)malloc(chunk_input_capacity);
    payload_chunk = (unsigned char*)malloc(chunk_payload_capacity);
    input_file = fopen(input_path, "rb");
    output_file = fopen(temp_path, "w+b");
    if (!sizes || !input_chunk || !payload_chunk || !input_file || !output_file) {
        fprintf(stderr, "tp-compress: host allocation or file open failed\n"); goto done;
    }

    cl_uint u_N = (cl_uint)N, u_hl = (cl_uint)hash_log, u_bs = (cl_uint)block_size, u_nblk = (cl_uint)nblk;
    unsigned long long u_orig = (unsigned long long)orig_size;
    if (!lz4_tp_write_exact(output_file, LZ4TP_MAGIC, 8) ||
        !lz4_tp_write_exact(output_file, &u_N, 4) ||
        !lz4_tp_write_exact(output_file, &u_hl, 4) ||
        !lz4_tp_write_exact(output_file, &u_bs, 4) ||
        !lz4_tp_write_exact(output_file, &u_nblk, 4) ||
        !lz4_tp_write_exact(output_file, &u_orig, 8) ||
        !lz4_tp_write_exact(output_file, sizes, meta_cnt * sizeof(cl_uint))) {
        fprintf(stderr, "tp-compress: frame header write failed\n"); goto done;
    }

    size_t payload_total = 0;
    uint64_t kernel_us = 0;
    for (size_t base = 0; base < nblk; base += chunk_blocks) {
        size_t this_blocks = nblk - base;
        if (this_blocks > chunk_blocks) this_blocks = chunk_blocks;
        size_t nominal_bytes = this_blocks * (size_t)block_size;
        size_t consumed = base * (size_t)block_size;
        size_t remaining = orig_size - consumed;
        size_t input_bytes = remaining < nominal_bytes ? remaining : nominal_bytes;
        if (!lz4_tp_read_exact(input_file, input_chunk, input_bytes)) {
            fprintf(stderr, "tp-compress: input changed or read failed\n"); goto done;
        }
        size_t payload_used = 0;
        if (lz4_tp_compress_chunk(input_chunk, input_bytes, this_blocks, N, block_size,
                                  dict_entries, (int)seg_max_out,
                                  kbuild, kscan, sizes + base * (size_t)N,
                                  payload_chunk, chunk_payload_capacity, &payload_used,
                                  &kernel_us) != 0 ||
            !lz4_tp_write_exact(output_file, payload_chunk, payload_used) ||
            lz4_tp_size_add(payload_total, payload_used, &payload_total) != 0) {
            fprintf(stderr, "tp-compress: chunk processing failed\n"); goto done;
        }
    }
    if (fgetc(input_file) != EOF || ferror(input_file)) {
        fprintf(stderr, "tp-compress: input size changed during compression\n"); goto done;
    }
    if (fseek(output_file, 32L, SEEK_SET) != 0 ||
        !lz4_tp_write_exact(output_file, sizes, meta_cnt * sizeof(cl_uint)) ||
        lz4_tp_finalize_temp_output(&output_file, temp_path, output_path) != 0) {
        fprintf(stderr, "tp-compress: frame finalization failed\n"); goto done;
    }
    temp_active = 0;

    size_t frame_total = 0, meta_bytes = meta_cnt * sizeof(cl_uint);
    if (lz4_tp_size_add(32, meta_bytes, &frame_total) != 0 ||
        lz4_tp_size_add(frame_total, payload_total, &frame_total) != 0) {
        fprintf(stderr, "tp-compress: frame size overflow\n"); goto done;
    }
    uint64_t work_end = get_us();
    uint64_t host_pre_us = pre_total_us >= pre_ocl_setup_us ?
                           pre_total_us - pre_ocl_setup_us : 0;
    uint64_t no_ocl_us = (ocl_start - total_start) + (work_end - work_start) + host_pre_us;
    uint64_t total_us = work_end - total_start + pre_total_us;
    uint64_t ocl_setup_us = work_start - ocl_start + pre_ocl_setup_us;
    if (lz4_tp_write_metrics("compress", orig_size, frame_total, N, block_size, hash_log,
                             chunk_blocks, kernel_us, no_ocl_us, total_us,
                             ocl_setup_us) != 0) goto done;
    fprintf(stderr, "[tp-compress] %s -> %s : %zu -> %zu (%.3f:1) N=%d blocks=%zu chunk=%zu\n",
            input_path, output_path, orig_size, frame_total,
            frame_total > 0 ? (double)orig_size / (double)frame_total : 0.0, N, nblk, chunk_blocks);
    status = 0;

done:
    if (input_file) fclose(input_file);
    if (output_file) fclose(output_file);
    if (temp_active) remove(temp_path);
    free(sizes);
    free(input_chunk);
    free(payload_chunk);
    if (kbuild) clReleaseKernel(kbuild);
    if (kscan) clReleaseKernel(kscan);
    if (prog) clReleaseProgram(prog);
    if (env_set) lz4_unset_env("LZ4_GPU_EPOCH32");
    ocl_release_runtime();
    return status;
}

static int lz4_tp_decompress_from_file(const char* input_path, const char* output_path) {
    uint64_t total_start = get_us();
    int status = 1, env_set = 0, temp_active = 0;
    struct stat frame_st;
    FILE* input_file = NULL;
    FILE* output_file = NULL;
    cl_program prog = NULL;
    cl_kernel kdec = NULL;
    cl_uint* comp_sizes = NULL;
    cl_uint* comp_off = NULL;
    cl_uint* chunk_sizes_out = NULL;
    unsigned char* payload = NULL;
    unsigned char* dec = NULL;
    char temp_path[PATH_MAX] = {0};
    if (!input_path || !output_path || !*output_path || paths_identify_same_file(input_path, output_path) ||
        (g_cli_metrics_path && (paths_identify_same_file(g_cli_metrics_path, input_path) ||
                                paths_identify_same_file(g_cli_metrics_path, output_path))) ||
        stat(input_path, &frame_st) != 0 || frame_st.st_size < 0 ||
        (uint64_t)frame_st.st_size > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "tp-decompress: invalid input\n"); return 1;
    }
    size_t frame_size = (size_t)frame_st.st_size;
    if (frame_size < 32) { fprintf(stderr, "tp-decompress: truncated header\n"); return 1; }
    input_file = fopen(input_path, "rb");
    if (!input_file) { fprintf(stderr, "tp-decompress: cannot open %s\n", input_path); return 1; }
    char magic[8] = {0};
    cl_uint u_N = 0, u_hl = 0, u_bs = 0, u_nblk = 0;
    unsigned long long u_orig = 0;
    if (!lz4_tp_read_exact(input_file, magic, 8) || memcmp(magic, LZ4TP_MAGIC, 8) != 0 ||
        !lz4_tp_read_exact(input_file, &u_N, 4) ||
        !lz4_tp_read_exact(input_file, &u_hl, 4) ||
        !lz4_tp_read_exact(input_file, &u_bs, 4) ||
        !lz4_tp_read_exact(input_file, &u_nblk, 4) ||
        !lz4_tp_read_exact(input_file, &u_orig, 8)) {
        fprintf(stderr, "tp-decompress: invalid or truncated LZ4TP1 header\n"); goto done;
    }
    int N = (int)u_N, hash_log = (int)u_hl, block_size = (int)u_bs;
    if ((N != 1 && N != 2 && N != 4 && N != 8) || hash_log < 11 || hash_log > 15 ||
        block_size < 1 || u_orig > (unsigned long long)SIZE_MAX) {
        fprintf(stderr, "tp-decompress: unsupported frame parameters\n"); goto done;
    }
    size_t nblk = (size_t)u_nblk, orig_size = (size_t)u_orig;
    size_t expected_nblk = orig_size / (size_t)block_size + (orig_size % (size_t)block_size != 0);
    size_t meta_cnt = 0, meta_bytes = 0, header_bytes = 0;
    size_t seg_len_max = ((size_t)block_size + (size_t)N - 1) / (size_t)N;
    size_t seg_max_out = 0;
    if (nblk != expected_nblk || lz4_tp_size_mul(nblk, (size_t)N, &meta_cnt) != 0 ||
        lz4_tp_size_mul(meta_cnt, sizeof(cl_uint), &meta_bytes) != 0 ||
        lz4_tp_size_add(32, meta_bytes, &header_bytes) != 0 || header_bytes > frame_size ||
        lz4_tp_size_add(seg_len_max, seg_len_max / 255, &seg_max_out) != 0 ||
        lz4_tp_size_add(seg_max_out, 64, &seg_max_out) != 0 || seg_max_out > (size_t)INT_MAX) {
        fprintf(stderr, "tp-decompress: invalid frame dimensions\n"); goto done;
    }
    if (orig_size == 0) {
        if (frame_size != 32 || fgetc(input_file) != EOF) {
            fprintf(stderr, "tp-decompress: trailing data in empty frame\n"); goto done;
        }
        if (lz4_tp_make_temp_path(output_path, temp_path, sizeof(temp_path)) != 0) goto done;
        temp_active = 1;
        output_file = fopen(temp_path, "wb");
        if (!output_file || lz4_tp_finalize_temp_output(&output_file, temp_path, output_path) != 0) {
            fprintf(stderr, "tp-decompress: cannot write %s\n", output_path); goto done;
        }
        temp_active = 0;
        uint64_t total_end = get_us();
        if (lz4_tp_write_metrics("decompress", frame_size, 0, N, block_size, hash_log,
                                 0, 0, total_end - total_start, total_end - total_start, 0) != 0) goto done;
        fprintf(stderr, "[tp-decompress] %s -> %s : empty frame OK\n", input_path, output_path);
        status = 0;
        goto done;
    }

    comp_sizes = (cl_uint*)malloc(meta_bytes);
    if (!comp_sizes || !lz4_tp_read_exact(input_file, comp_sizes, meta_bytes)) {
        fprintf(stderr, "tp-decompress: cannot read sizes\n"); goto done;
    }
    size_t payload_total = 0;
    for (size_t g = 0; g < meta_cnt; g++) {
        if (comp_sizes[g] == 0 || comp_sizes[g] > seg_max_out ||
            lz4_tp_size_add(payload_total, (size_t)comp_sizes[g], &payload_total) != 0) {
            fprintf(stderr, "tp-decompress: invalid compressed segment size\n"); goto done;
        }
    }
    size_t expected_frame_size = 0;
    if (lz4_tp_size_add(header_bytes, payload_total, &expected_frame_size) != 0 ||
        expected_frame_size != frame_size) {
        fprintf(stderr, "tp-decompress: payload length or trailing data mismatch\n"); goto done;
    }

    uint64_t ocl_start = get_us();
    if (lz4_set_env("LZ4_GPU_EPOCH32", "1") != 0) {
        fprintf(stderr, "tp-decompress: cannot set epoch mode\n"); goto done;
    }
    env_set = 1;
    ocl_init();
    if (!ctx || !queue) { fprintf(stderr, "tp-decompress: OpenCL init failed\n"); goto done; }

    cl_int err = CL_SUCCESS;
    prog = lz4_load_program(ctx, dev, hash_log, (size_t)block_size);
    if (!prog) { fprintf(stderr, "tp-decompress: prog load failed\n"); goto done; }
    kdec = clCreateKernel(prog, "lz4_tp_decompress_segmented", &err);
    if (!kdec) { fprintf(stderr, "tp-decompress: kernel failed %d\n", err); goto done; }
    uint64_t work_start = get_us();

    size_t chunk_blocks = lz4_tp_choose_decode_chunk_blocks(nblk, N, block_size, seg_max_out);
    size_t chunk_meta_capacity = 0, chunk_payload_capacity = 0, chunk_dec_capacity = 0;
    if (chunk_blocks == 0 ||
        lz4_tp_size_mul(chunk_blocks, (size_t)N, &chunk_meta_capacity) != 0 ||
        lz4_tp_size_mul(chunk_meta_capacity, seg_max_out, &chunk_payload_capacity) != 0 ||
        chunk_payload_capacity > (size_t)UINT_MAX ||
        lz4_tp_size_mul(chunk_blocks, (size_t)block_size, &chunk_dec_capacity) != 0) {
        fprintf(stderr, "tp-decompress: no safe OpenCL chunk size\n"); goto done;
    }
    payload = (unsigned char*)malloc(chunk_payload_capacity);
    comp_off = (cl_uint*)malloc(chunk_meta_capacity * sizeof(cl_uint));
    chunk_sizes_out = (cl_uint*)malloc(chunk_blocks * sizeof(cl_uint));
    dec = (unsigned char*)malloc(chunk_dec_capacity);
    if (!payload || !comp_off || !chunk_sizes_out || !dec ||
        lz4_tp_make_temp_path(output_path, temp_path, sizeof(temp_path)) != 0) {
        fprintf(stderr, "tp-decompress: host allocation failed\n"); goto done;
    }
    temp_active = 1;
    output_file = fopen(temp_path, "wb");
    if (!output_file) { fprintf(stderr, "tp-decompress: cannot open %s\n", temp_path); goto done; }

    uint64_t kernel_us = 0;
    size_t written_total = 0;
    for (size_t base = 0; base < nblk; base += chunk_blocks) {
        size_t this_blocks = nblk - base;
        if (this_blocks > chunk_blocks) this_blocks = chunk_blocks;
        size_t this_meta = this_blocks * (size_t)N;
        size_t chunk_payload_size = 0;
        for (size_t g = 0; g < this_meta; g++) {
            cl_uint csize = comp_sizes[base * (size_t)N + g];
            if (chunk_payload_size > (size_t)UINT_MAX ||
                csize > (cl_uint)((size_t)UINT_MAX - chunk_payload_size)) {
                fprintf(stderr, "tp-decompress: chunk offset overflow\n"); goto done;
            }
            comp_off[g] = (cl_uint)chunk_payload_size;
            chunk_payload_size += csize;
        }
        if (chunk_payload_size > chunk_payload_capacity ||
            !lz4_tp_read_exact(input_file, payload, chunk_payload_size)) {
            fprintf(stderr, "tp-decompress: payload read failed\n"); goto done;
        }
        for (size_t b = 0; b < this_blocks; b++) {
            size_t global_block = base + b;
            size_t block_start = global_block * (size_t)block_size;
            size_t block_len = orig_size - block_start;
            if (block_len > (size_t)block_size) block_len = (size_t)block_size;
            size_t seg_len = (block_len + (size_t)N - 1) / (size_t)N;
            for (int s = 0; s < N; s++) {
                size_t local_g = b * (size_t)N + (size_t)s;
                size_t lo = (size_t)s * seg_len;
                if (lo >= block_len &&
                    (comp_sizes[base * (size_t)N + local_g] != 1 || payload[comp_off[local_g]] != 0)) {
                    fprintf(stderr, "tp-decompress: non-canonical empty segment\n"); goto done;
                }
            }
        }
        if (lz4_tp_decompress_chunk(payload, chunk_payload_size, comp_off,
                                    comp_sizes + base * (size_t)N, this_blocks, N,
                                    block_size, kdec, dec, chunk_sizes_out, &kernel_us) != 0) {
            fprintf(stderr, "tp-decompress: chunk decode failed\n"); goto done;
        }
        for (size_t b = 0; b < this_blocks; b++) {
            size_t global_block = base + b;
            size_t block_start = global_block * (size_t)block_size;
            size_t expected = orig_size - block_start;
            if (expected > (size_t)block_size) expected = (size_t)block_size;
            if (chunk_sizes_out[b] != (cl_uint)expected ||
                !lz4_tp_write_exact(output_file, dec + b * (size_t)block_size, expected) ||
                lz4_tp_size_add(written_total, expected, &written_total) != 0) {
                fprintf(stderr, "tp-decompress: block %zu length or write failure\n", global_block); goto done;
            }
        }
    }
    if (written_total != orig_size || fgetc(input_file) != EOF || ferror(input_file) ||
        lz4_tp_finalize_temp_output(&output_file, temp_path, output_path) != 0) {
        fprintf(stderr, "tp-decompress: final length or output failure\n"); goto done;
    }
    temp_active = 0;
    uint64_t work_end = get_us();
    uint64_t no_ocl_us = (ocl_start - total_start) + (work_end - work_start);
    if (lz4_tp_write_metrics("decompress", frame_size, orig_size, N, block_size, hash_log,
                             chunk_blocks, kernel_us, no_ocl_us, work_end - total_start,
                             work_start - ocl_start) != 0) goto done;
    fprintf(stderr, "[tp-decompress] %s -> %s : %zu bytes N=%d blocks=%zu chunk=%zu OK\n",
            input_path, output_path, orig_size, N, nblk, chunk_blocks);
    status = 0;

done:
    if (input_file) fclose(input_file);
    if (output_file) fclose(output_file);
    if (temp_active) remove(temp_path);
    free(comp_sizes);
    free(comp_off);
    free(chunk_sizes_out);
    free(payload);
    free(dec);
    if (kdec) clReleaseKernel(kdec);
    if (prog) clReleaseProgram(prog);
    if (env_set) lz4_unset_env("LZ4_GPU_EPOCH32");
    ocl_release_runtime();
    return status;
}
/* ============================================================================
 * I2: the selector, IN the binary (not Python).
 *   --calibrate <file> [-o profile] : measure this device's W_sat, store it.
 *   -c --auto <in> -o <out>         : choose the largest N in {1,2,4,8} with
 *                                     min(file_blocks, safe_chunk_N)*N <=
 *                                     0.75*W_sat, then dispatch
 *                                     two-phase (N=1 == base via seg path).
 * Consistency with deployment: calibration measures the SAME tp kernels that
 * --auto dispatches (base = tp N=1), so the fitted W_sat matches what --auto uses.
 * ==========================================================================*/

/* measure two-phase compress throughput (event-profiled, best-of-K) for one
 * (prefix of input, N). Returns MB/s, or 0 on failure. Allocates+frees per call
 * (calibration is one-time). */
static double lz4_tp_measure_one(const unsigned char* input_ref, size_t size, int N,
                                 int block_size, int hash_log,
                                 cl_kernel kbuild, cl_kernel kscan) {
    cl_int err = CL_SUCCESS;
    size_t dict_entries = (size_t)1u << hash_log;
    size_t nblk = size / (size_t)block_size + (size % (size_t)block_size != 0);
    int nk = N - 1;
    size_t seg_len_max = ((size_t)block_size + (size_t)N - 1) / (size_t)N;
    size_t seg_max_out = 0, meta_cnt = 0, prefix_bytes = 0, own_bytes = 0, out_bytes = 0, tmp = 0;
    if (!input_ref || size == 0 || size > INT_MAX || !kbuild || !kscan ||
        lz4_tp_size_add(seg_len_max, seg_len_max / 255, &seg_max_out) != 0 ||
        lz4_tp_size_add(seg_max_out, 64, &seg_max_out) != 0 || seg_max_out > INT_MAX ||
        lz4_tp_size_mul(nblk, (size_t)N, &meta_cnt) != 0 ||
        lz4_tp_size_mul(nblk, (size_t)nk, &tmp) != 0 ||
        lz4_tp_size_mul(tmp, dict_entries, &tmp) != 0 ||
        lz4_tp_size_mul(tmp, sizeof(cl_uint), &prefix_bytes) != 0 ||
        lz4_tp_size_mul(meta_cnt, dict_entries, &tmp) != 0 ||
        lz4_tp_size_mul(tmp, sizeof(cl_uint), &own_bytes) != 0 ||
        lz4_tp_size_mul(meta_cnt, seg_max_out, &out_bytes) != 0) {
        return 0.0;
    }

    cl_mem d_input  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size, NULL, &err);
    cl_mem d_prefix = clCreateBuffer(ctx, CL_MEM_READ_WRITE, prefix_bytes ? prefix_bytes : 4, NULL, &err);
    cl_mem d_own    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, own_bytes, NULL, &err);
    cl_mem d_out    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    cl_mem d_sizes  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, meta_cnt * sizeof(cl_uint), NULL, &err);
    if (!d_input || !d_prefix || !d_own || !d_out || !d_sizes) {
        if (d_input) clReleaseMemObject(d_input);
        if (d_prefix) clReleaseMemObject(d_prefix);
        if (d_own) clReleaseMemObject(d_own);
        if (d_out) clReleaseMemObject(d_out);
        if (d_sizes) clReleaseMemObject(d_sizes);
        return 0.0;
    }
    if (clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, size, input_ref, 0, NULL, NULL) != CL_SUCCESS) {
        clReleaseMemObject(d_input); clReleaseMemObject(d_prefix); clReleaseMemObject(d_own);
        clReleaseMemObject(d_out); clReleaseMemObject(d_sizes);
        return 0.0;
    }
    { cl_uint zero = 0;
      if ((prefix_bytes && clEnqueueFillBuffer(queue, d_prefix, &zero, 4, 0, prefix_bytes, 0, NULL, NULL) != CL_SUCCESS) ||
          clEnqueueFillBuffer(queue, d_own, &zero, 4, 0, own_bytes, 0, NULL, NULL) != CL_SUCCESS ||
          clFinish(queue) != CL_SUCCESS) {
          clReleaseMemObject(d_input); clReleaseMemObject(d_prefix); clReleaseMemObject(d_own);
          clReleaseMemObject(d_out); clReleaseMemObject(d_sizes);
          return 0.0;
      } }

    cl_int i_nblk = (cl_int)nblk, i_insize = (cl_int)size, i_blk = block_size, i_segmax = (cl_int)seg_max_out, i_N = N;
    size_t lws1 = 1, g_build = (size_t)nblk * (size_t)nk * 4, g_scan = (size_t)nblk * (size_t)N;
    cl_int arg_err = CL_SUCCESS;
    arg_err |= clSetKernelArg(kbuild,0,sizeof(cl_mem),&d_input); arg_err |= clSetKernelArg(kbuild,1,sizeof(cl_mem),&d_prefix);
    arg_err |= clSetKernelArg(kbuild,2,sizeof(cl_int),&i_nblk); arg_err |= clSetKernelArg(kbuild,3,sizeof(cl_int),&i_insize);
    arg_err |= clSetKernelArg(kbuild,4,sizeof(cl_int),&i_blk); arg_err |= clSetKernelArg(kbuild,5,sizeof(cl_int),&i_N);
    arg_err |= clSetKernelArg(kscan,0,sizeof(cl_mem),&d_input); arg_err |= clSetKernelArg(kscan,1,sizeof(cl_mem),&d_out);
    arg_err |= clSetKernelArg(kscan,2,sizeof(cl_mem),&d_sizes); arg_err |= clSetKernelArg(kscan,3,sizeof(cl_mem),&d_own);
    arg_err |= clSetKernelArg(kscan,4,sizeof(cl_mem),&d_prefix); arg_err |= clSetKernelArg(kscan,5,sizeof(cl_int),&i_nblk);
    arg_err |= clSetKernelArg(kscan,6,sizeof(cl_int),&i_insize); arg_err |= clSetKernelArg(kscan,7,sizeof(cl_int),&i_blk);
    arg_err |= clSetKernelArg(kscan,8,sizeof(cl_int),&i_segmax); arg_err |= clSetKernelArg(kscan,9,sizeof(cl_int),&i_N);
    if (arg_err != CL_SUCCESS) {
        clReleaseMemObject(d_input); clReleaseMemObject(d_prefix); clReleaseMemObject(d_own);
        clReleaseMemObject(d_out); clReleaseMemObject(d_sizes);
        return 0.0;
    }

    double us[16]; int nu = 0; cl_uint epoch = 1;
    const int WARM = 2, MEAS = 5;
    for (int it = 0; it < WARM + MEAS; ++it, ++epoch) {
        double t = 0.0;
        if (nk >= 1) {
            if (clSetKernelArg(kbuild, 6, sizeof(cl_uint), &epoch) != CL_SUCCESS) break;
            cl_event e = NULL;
            if (clEnqueueNDRangeKernel(queue,kbuild,1,NULL,&g_build,&lws1,0,NULL,&e)!=CL_SUCCESS ||
                clWaitForEvents(1,&e) != CL_SUCCESS) { if (e) clReleaseEvent(e); break; }
            t += event_elapsed_us(e); clReleaseEvent(e);
        }
        if (clSetKernelArg(kscan, 10, sizeof(cl_uint), &epoch) != CL_SUCCESS) break;
        cl_event e2 = NULL;
        if (clEnqueueNDRangeKernel(queue,kscan,1,NULL,&g_scan,&lws1,0,NULL,&e2)!=CL_SUCCESS ||
            clWaitForEvents(1,&e2) != CL_SUCCESS) { if (e2) clReleaseEvent(e2); break; }
        t += event_elapsed_us(e2); clReleaseEvent(e2);
        if (it >= WARM && nu < 16) us[nu++] = t;
    }
    cl_uint* sizes = (cl_uint*)malloc(meta_cnt * sizeof(cl_uint));
    int sizes_ok = sizes && nu == MEAS &&
                   clEnqueueReadBuffer(queue, d_sizes, CL_TRUE, 0, meta_cnt * sizeof(cl_uint),
                                       sizes, 0, NULL, NULL) == CL_SUCCESS;
    if (sizes_ok) {
        for (size_t g = 0; g < meta_cnt; g++) {
            if (sizes[g] == 0 || sizes[g] > seg_max_out) { sizes_ok = 0; break; }
        }
    }
    free(sizes);
    clReleaseMemObject(d_input); clReleaseMemObject(d_prefix); clReleaseMemObject(d_own);
    clReleaseMemObject(d_out); clReleaseMemObject(d_sizes);
    if (!sizes_ok) return 0.0;
    double med = median_double(us, nu);
    return med > 0 ? (double)size / med : 0.0;
}

static int lz4_tp_pick_N_chunked(size_t total_blocks, int W_sat,
                                 int block_size, int hash_log,
                                 size_t* selected_chunk_blocks) {
    int cand[4] = {1, 2, 4, 8};
    int best = 1;
    size_t best_chunk = 0;
    double threshold = 0.75 * (double)W_sat;
    size_t dict_entries = (size_t)1u << hash_log;
    for (int i = 0; i < 4; i++) {
        size_t chunk = lz4_tp_choose_chunk_blocks(total_blocks, cand[i], block_size, dict_entries);
        if (chunk == 0) continue;
        size_t effective_blocks = total_blocks < chunk ? total_blocks : chunk;
        if ((double)effective_blocks * (double)cand[i] <= threshold) {
            best = cand[i];
            best_chunk = chunk;
        } else if (cand[i] == 1 && best_chunk == 0) {
            best_chunk = chunk;
        }
    }
    if (selected_chunk_blocks) *selected_chunk_blocks = best_chunk;
    return best;
}

static void lz4_device_name(char* buf, size_t n) {
    lz4_device_profile_key(dev, buf, n);
}

static const char* lz4_profile_path(void) {
    const char* p = getenv("LZ4TP_PROFILE");
    return (p && *p) ? p : "lz4tp.profile";
}

static int lz4_calibrate_device(const char* cal_file, int block_size) {
    struct stat st;
    if (!cal_file || stat(cal_file, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "calibrate: invalid calibration file\n"); return 1;
    }
    if (block_size <= 0) block_size = 64 * 1024;
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "calibrate: input is too large for this host\n"); return 1;
    }
    size_t full = (size_t)st.st_size;
    if (lz4_set_env("LZ4_GPU_EPOCH32", "1") != 0) {
        fprintf(stderr, "calibrate: cannot set epoch mode\n"); return 1;
    }
    ocl_init();
    if (!ctx || !queue) {
        lz4_unset_env("LZ4_GPU_EPOCH32");
        fprintf(stderr, "calibrate: OpenCL init failed\n"); return 1;
    }
    char devname[256]; lz4_device_name(devname, sizeof(devname));
    int hash_log = g_cli_hash_log;
    size_t dict_entries = (size_t)1u << hash_log;
    size_t full_nblk = full / (size_t)block_size;
    size_t safe_nblk = full_nblk;
    int cand[4] = {1,2,4,8};
    for (int ci = 0; ci < 4; ci++) {
        size_t safe = lz4_tp_choose_chunk_blocks(full_nblk, cand[ci], block_size, dict_entries);
        if (safe < safe_nblk) safe_nblk = safe;
    }
    if (safe_nblk < 4) {
        fprintf(stderr, "calibrate: device and input must support at least four complete blocks\n");
        lz4_unset_env("LZ4_GPU_EPOCH32");
        ocl_release_runtime();
        return 1;
    }
    size_t calibration_bytes = safe_nblk * (size_t)block_size;
    unsigned char* input_ref = (unsigned char*)malloc(calibration_bytes);
    unsigned long rd = 0;
    if (!input_ref || lz4_read_file_to_buf(cal_file, input_ref, calibration_bytes, &rd) != 0) {
        free(input_ref);
        lz4_unset_env("LZ4_GPU_EPOCH32");
        ocl_release_runtime();
        fprintf(stderr, "calibrate: read failed\n"); return 1;
    }
    cl_int err = CL_SUCCESS;
    cl_program prog = lz4_load_program(ctx, dev, hash_log, (size_t)block_size);
    cl_kernel kbuild = prog ? clCreateKernel(prog, "lz4_tp_build_prefix", &err) : NULL;
    cl_kernel kscan  = prog ? clCreateKernel(prog, "lz4_tp_scan_seg", &err) : NULL;
    if (!kbuild || !kscan) {
        fprintf(stderr, "calibrate: kernels failed\n");
        free(input_ref);
        if (kbuild) clReleaseKernel(kbuild);
        if (kscan) clReleaseKernel(kscan);
        if (prog) clReleaseProgram(prog);
        lz4_unset_env("LZ4_GPU_EPOCH32");
        ocl_release_runtime();
        return 1;
    }

    /* Block-count ladder, capped at the largest point all candidate N values can
     * execute as one deployment-equivalent chunk on this device. */
    int ladder[] = {2,4,8,16,32,64,128,256,384,512,633};
    int NL = (int)(sizeof(ladder)/sizeof(ladder[0]));
    /* per-ladder-point throughput for N in {1,2,4,8}; store to fit W_sat */
    double tp[16][4]; int nblk_of[16]; int L = 0;
    fprintf(stderr, "[calibrate] device=%s  block=%dK\n", devname, block_size/1024);
    for (int li = 0; li < NL && L < 16; li++) {
        size_t nb = (size_t)ladder[li];
        if (nb > safe_nblk) break;
        size_t size = nb * (size_t)block_size;
        nblk_of[L] = (int)nb;
        for (int ci = 0; ci < 4; ci++)
            tp[L][ci] = lz4_tp_measure_one(input_ref, size, cand[ci], block_size, hash_log, kbuild, kscan);
        fprintf(stderr, "  nblk=%-4d N1=%.0f N2=%.0f N4=%.0f N8=%.0f MB/s\n",
                (int)nb, tp[L][0], tp[L][1], tp[L][2], tp[L][3]);
        L++;
    }
    if (L < 2) {
        fprintf(stderr, "calibrate: input must contain at least four complete blocks\n");
        free(input_ref);
        clReleaseKernel(kbuild); clReleaseKernel(kscan); clReleaseProgram(prog);
        lz4_unset_env("LZ4_GPU_EPOCH32");
        ocl_release_runtime();
        return 1;
    }

    /* fit W_sat: grid W in 1..8192, maximize geomean over ladder of achieved/oracle,
     * where achieved = tp[ pick_raw(nblk,W) ] and pick_raw uses W directly (no 0.75). */
    int bestW = 1; double bestScore = -1e18; int wlo = 1, whi = 1;
    for (int W = 1; W <= 8192; W++) {
        double s = 0; int used = 0;
        for (int i = 0; i < L; i++) {
            double orc = 0; for (int ci=0;ci<4;ci++) if (tp[i][ci] > orc) orc = tp[i][ci];
            if (orc <= 0) continue;
            int pick = 1; for (int ci=0;ci<4;ci++) if ((double)nblk_of[i]*cand[ci] <= (double)W) pick = cand[ci];
            int idx = pick==1?0:pick==2?1:pick==4?2:3;
            double got = tp[i][idx] > 0 ? tp[i][idx] : tp[i][0];
            if (got <= 0) continue;
            s += log(got/orc); used++;
        }
        if (used == 0) continue;
        s /= used;
        if (s > bestScore + 1e-9) { bestScore = s; wlo = whi = W; }
        else if (s > bestScore - 1e-9) { whi = W; }
    }
    bestW = (wlo + whi) / 2;   /* midpoint of the flat optimum plateau (sparse ladder => wide ties) */
    fprintf(stderr, "[calibrate] fitted W_sat = %d  (device=%s)\n", bestW, devname);
    if (lz4_tp_profile_store(lz4_profile_path(), devname, block_size, hash_log, bestW) != 0) {
        fprintf(stderr, "calibrate: cannot store profile %s\n", lz4_profile_path());
        free(input_ref);
        clReleaseKernel(kbuild); clReleaseKernel(kscan); clReleaseProgram(prog);
        lz4_unset_env("LZ4_GPU_EPOCH32");
        ocl_release_runtime();
        return 1;
    }
    fprintf(stderr, "[calibrate] stored to %s\n", lz4_profile_path());

    free(input_ref);
    clReleaseKernel(kbuild); clReleaseKernel(kscan); clReleaseProgram(prog);
    lz4_unset_env("LZ4_GPU_EPOCH32");
    ocl_release_runtime();
    return 0;
}

/* -c --auto : pick N from the stored W_sat, then compress. Needs device name,
 * so we peek the device via a throwaway ocl_init, read profile, release, then
 * call the real compressor (which re-inits). Cheap relative to compression. */
static int lz4_tp_compress_auto(const char* input_path, const char* output_path, int block_size) {
    struct stat st;
    if (!input_path || stat(input_path, &st) != 0 || st.st_size < 0) {
        fprintf(stderr, "auto: invalid input\n"); return 1;
    }
    if (block_size <= 0) block_size = 64 * 1024;
    if (st.st_size == 0)
        return lz4_tp_compress_to_file(input_path, output_path, 1, block_size, 0, 0);
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "auto: input is too large for this host\n"); return 1;
    }
    size_t input_size = (size_t)st.st_size;
    size_t nblk = input_size / (size_t)block_size + (input_size % (size_t)block_size != 0);

    uint64_t selector_start = get_us();
    uint64_t selector_ocl_start = get_us();
    ocl_init();
    if (!ctx) { fprintf(stderr, "auto: OpenCL init failed\n"); return 1; }
    char devname[256]; lz4_device_name(devname, sizeof(devname));
    uint64_t selector_ocl_end = get_us();
    int W_sat = lz4_tp_profile_lookup(lz4_profile_path(), devname, block_size, g_cli_hash_log);
    int N;
    size_t selected_chunk = 0;
    if (W_sat <= 0) {
        N = 1;
        selected_chunk = lz4_tp_choose_chunk_blocks(nblk, N, block_size, (size_t)1u << g_cli_hash_log);
        fprintf(stderr, "[auto] no profile for device \"%s\" (run --calibrate); using N=1 (safe, no gain)\n", devname);
    } else {
        N = lz4_tp_pick_N_chunked(nblk, W_sat, block_size, g_cli_hash_log, &selected_chunk);
        fprintf(stderr, "[auto] device=%s W_sat=%d nblk=%zu chunk=%zu -> N=%d\n",
                devname, W_sat, nblk, selected_chunk, N);
    }
    ocl_release_runtime();
    if (selected_chunk == 0) {
        fprintf(stderr, "auto: no safe OpenCL chunk size\n"); return 1;
    }
    uint64_t selector_end = get_us();
    return lz4_tp_compress_to_file(input_path, output_path, N, block_size,
                                   selector_end - selector_start,
                                   selector_ocl_end - selector_ocl_start);
}
int run_lz4_standalone(int argc, char** argv) {
    int mode = mode_compress;
    int bench_mode = 0;
    int tp_bench_mode = 0;
    int twophase_mode = 0;
    int calibrate_mode = 0;
    int auto_mode = 0;
    double bench_seconds = 3.0;
    const char* input_path = NULL;
    const char* effective_input_path = NULL;
    const char* effective_output_path = NULL;
    char output_path[512] = {0};
    char temp_input_path[PATH_MAX] = {0};
    char temp_output_path[PATH_MAX] = {0};
    int output_explicit = 0;
    int input_from_stdin;
    int output_to_stdout;
    int have_temp_input = 0;
    int have_temp_output = 0;
    int ret = -1;
    cl_program prog = NULL;
    cl_kernel kernel = NULL;
    cl_int err = CL_SUCCESS;
    lz4_gpu_workspace_t ws;
    int ws_inited = 0;
    timing_t t_out;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-c") == 0) {
            mode = mode_compress;
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) {
            mode = mode_decompress;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_verbose = 1;
        } else if (strcmp(argv[i], "--calibrate") == 0) {
            calibrate_mode = 1;
        } else if (strcmp(argv[i], "--auto") == 0) {
            auto_mode = 1;
        } else if (strcmp(argv[i], "--twophase") == 0) {
            twophase_mode = 1;
        } else if (strcmp(argv[i], "-N") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &g_cli_tp_n) != 0) {
                fprintf(stderr, "Error: -N requires an integer argument\n"); return 1;
            }
        } else if (strcmp(argv[i], "--tp-bench") == 0) {
            tp_bench_mode = 1;
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                if (parse_positive_double_arg(argv[++i], &bench_seconds) != 0) {
                    fprintf(stderr, "Error: --bench duration must be positive\n"); return 1;
                }
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                if (copy_cli_path(output_path, sizeof(output_path), argv[++i]) != 0) {
                    fprintf(stderr, "Error: output path is too long\n"); return 1;
                }
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: -o requires an argument\n");
                show_help(argv[0]); return 1;
            }
        } else if (strcmp(argv[i], "-B") == 0 || strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) {
            if (i + 1 >= argc || (g_cli_fixed_block_bytes = parse_size_bytes(argv[++i])) == 0) {
                fprintf(stderr, "Error: -B requires a positive byte size\n"); return 1;
            }
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--acceleration") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &g_cli_acceleration) != 0) {
                fprintf(stderr, "Error: -a requires an integer argument\n"); return 1;
            }
        } else if (strcmp(argv[i], "--d-bits") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &g_cli_hash_log) != 0) {
                fprintf(stderr, "Error: --d-bits requires an integer argument\n"); return 1;
            }
        } else if (strcmp(argv[i], "--local") == 0) {
            int parsed_local = 0;
            if (i + 1 >= argc || parse_int_arg(argv[++i], &parsed_local) != 0 || parsed_local < 1) {
                fprintf(stderr, "Error: --local requires a positive integer argument\n"); return 1;
            }
            g_cli_local_size = (size_t)parsed_local;
        } else if (strcmp(argv[i], "--metrics-json") == 0) {
            if (i + 1 < argc) g_cli_metrics_path = argv[++i];
            else { fprintf(stderr, "Error: --metrics-json requires an argument\n"); return 1; }
        } else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0) {
            fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
            show_help(argv[0]); return 1;
        } else {
            if (!input_path) input_path = argv[i];
            else if (!output_explicit) {
                if (copy_cli_path(output_path, sizeof(output_path), argv[i]) != 0) {
                    fprintf(stderr, "Error: output path is too long\n"); return 1;
                }
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: Too many positional arguments\n");
                show_help(argv[0]); return 1;
            }
        }
    }

    if (validate_cli_config(twophase_mode) != 0) return 1;
    int special_modes = calibrate_mode + auto_mode + twophase_mode + tp_bench_mode + bench_mode;
    if (special_modes > 1) {
        fprintf(stderr, "Error: --calibrate, --auto, --twophase, --tp-bench, and --bench are mutually exclusive\n");
        return 1;
    }
    if (g_cli_metrics_path && (!*g_cli_metrics_path || path_is_dash(g_cli_metrics_path) ||
                               (!twophase_mode && !auto_mode))) {
        fprintf(stderr, "Error: --metrics-json requires --twophase or --auto and a regular output path\n");
        return 1;
    }

    if (!input_path) {
        show_help(argv[0]);
        return 1;
    }

    if (calibrate_mode) {
        if (mode != mode_compress || path_is_dash(input_path)) {
            fprintf(stderr, "Error: --calibrate requires a regular input file in compress mode\n");
            return 1;
        }
        if (output_explicit && paths_identify_same_file(input_path, output_path)) {
            fprintf(stderr, "Error: calibration profile must not replace its input\n");
            return 1;
        }
        if (output_explicit && lz4_set_env("LZ4TP_PROFILE", output_path) != 0) {
            fprintf(stderr, "Error: cannot set calibration profile path\n");
            return 1;
        }
        int result = lz4_calibrate_device(input_path, (int)g_cli_fixed_block_bytes);
        if (output_explicit) lz4_unset_env("LZ4TP_PROFILE");
        return result;
    }
    if (auto_mode) {
        if (mode != mode_compress || path_is_dash(input_path) ||
            (output_explicit && path_is_dash(output_path))) {
            fprintf(stderr, "Error: --auto requires regular input and output files in compress mode\n");
            return 1;
        }
        char au_out[512];
        const char* outp = output_explicit ? output_path : NULL;
        if (!outp) {
            if (append_cli_suffix(au_out, sizeof(au_out), input_path, ".lz4tp") != 0) {
                fprintf(stderr, "Error: derived output path is too long\n"); return 1;
            }
            outp = au_out;
        }
        if (paths_identify_same_file(lz4_profile_path(), input_path) ||
            paths_identify_same_file(lz4_profile_path(), outp)) {
            fprintf(stderr, "Error: calibration profile must differ from auto input and output\n");
            return 1;
        }
        return lz4_tp_compress_auto(input_path, outp, (int)g_cli_fixed_block_bytes);
    }

    if (twophase_mode) {
        if (path_is_dash(input_path) || (output_explicit && path_is_dash(output_path))) {
            fprintf(stderr, "Error: LZ4TP1 currently requires regular input and output files\n");
            return 1;
        }
        char tp_out[512];
        const char* outp = output_explicit ? output_path : NULL;
        if (!outp) {
            const char* suffix = mode == mode_compress ? ".lz4tp" : ".tpout";
            if (append_cli_suffix(tp_out, sizeof(tp_out), input_path, suffix) != 0) {
                fprintf(stderr, "Error: derived output path is too long\n"); return 1;
            }
            outp = tp_out;
        }
        if (mode == mode_compress)
            return lz4_tp_compress_to_file(input_path, outp, g_cli_tp_n,
                                           (int)g_cli_fixed_block_bytes, 0, 0);
        return lz4_tp_decompress_from_file(input_path, outp);
    }

    if (tp_bench_mode) {
        if (path_is_dash(input_path)) {
            fprintf(stderr, "Error: --tp-bench does not support stdin input ('-')\n");
            return 1;
        }
        return run_lz4_tp_bench(input_path, (int)g_cli_fixed_block_bytes);
    }

    if (bench_mode) {
        if (mode != mode_compress) {
            fprintf(stderr, "Error: --bench only supports compress mode input (it runs compress+decompress internally)\n");
            return 1;
        }
        if (path_is_dash(input_path)) {
            fprintf(stderr, "Error: --bench does not support stdin input ('-')\n");
            return 1;
        }
        return run_lz4_bench(input_path,
                            (int)g_cli_fixed_block_bytes,
                            g_cli_acceleration,
                            (int)g_cli_local_size,
                            bench_seconds);
    }

    if (!output_explicit) {
        if (path_is_dash(input_path)) {
            if (copy_cli_path(output_path, sizeof(output_path), "-") != 0) return 1;
            output_explicit = 1;
        } else if (mode == mode_compress) {
            if (append_cli_suffix(output_path, sizeof(output_path), input_path, ".lz4") != 0) {
                fprintf(stderr, "Error: derived output path is too long\n"); return 1;
            }
        } else {
            if (append_cli_suffix(output_path, sizeof(output_path), input_path, ".dec") != 0) {
                fprintf(stderr, "Error: derived output path is too long\n"); return 1;
            }
        }
    }

    input_from_stdin = path_is_dash(input_path);
    output_to_stdout = path_is_dash(output_path);
    effective_input_path = input_path;
    effective_output_path = output_path;

    if (input_from_stdin) {
        if (create_temp_path(temp_input_path, sizeof(temp_input_path), "/tmp/lz4_gpu_stdin_XXXXXX") != 0) {
            fprintf(stderr, "Error: failed to create temporary input path for stdin stream\n");
            return 1;
        }
        if (copy_stream_to_path(stdin, temp_input_path) != 0) {
            fprintf(stderr, "Error: failed to capture stdin into temporary input file\n");
            unlink(temp_input_path);
            return 1;
        }
        effective_input_path = temp_input_path;
        have_temp_input = 1;
    }

    if (output_to_stdout) {
        if (create_temp_path(temp_output_path, sizeof(temp_output_path), "/tmp/lz4_gpu_stdout_XXXXXX") != 0) {
            fprintf(stderr, "Error: failed to create temporary output path for stdout stream\n");
            if (have_temp_input) unlink(temp_input_path);
            return 1;
        }
        effective_output_path = temp_output_path;
        have_temp_output = 1;
    }

    uint64_t t_total_start = get_us();
    uint64_t t1, t2;

    t1 = get_us();
    ocl_init();
    if (!ctx || !queue) {
        fprintf(stderr, "Failed to initialize OpenCL runtime\n");
        ret = 1;
        goto cleanup;
    }
    t2 = get_us();
    g_ocl_init_us = t2 - t1;

    t1 = get_us();
    prog = lz4_load_program(ctx, dev, g_cli_hash_log, g_cli_fixed_block_bytes);
    t2 = get_us();
    g_kernel_load_us = t2 - t1;

    if (!prog) {
        fprintf(stderr, "Failed to load OCL program\n");
        ret = 1;
        goto cleanup;
    }

    if (mode == mode_compress) {
        kernel = clCreateKernel(prog, "lz4_compress_block", &err);
        if (err != CL_SUCCESS || !kernel) { fprintf(stderr, "Failed to create kernel: %d\n", err); ret = 1; goto cleanup; }
    } else {
        kernel = clCreateKernel(prog, "lz4_decompress_blocks", &err);
        if (err != CL_SUCCESS || !kernel) { fprintf(stderr, "Failed to create kernel: %d\n", err); ret = 1; goto cleanup; }
    }

    lz4_gpu_workspace_init(&ws);
    ws_inited = 1;
    memset(&t_out, 0, sizeof(t_out));

    if (mode == mode_compress) {
        ret = lz4_compress_core(ctx, queue, kernel,
                                effective_input_path, effective_output_path,
                                (int)g_cli_fixed_block_bytes, g_cli_acceleration, g_cli_hash_log,
                                &ws, &t_out, (int)g_cli_local_size, 0);
    } else {
        ret = lz4_decompress_core(ctx, queue, kernel,
                                  effective_input_path, effective_output_path,
                                  &ws, &t_out, (int)g_cli_local_size);
    }

    uint64_t t_total_end = get_us();

    if (ret == 0) {
        FILE* msg = output_to_stdout ? stderr : stdout;
        t_out.ocl_setup_us = (unsigned long)(g_ocl_init_us + g_kernel_load_us);
        response_t resp;
        memset(&resp, 0, sizeof(resp));
        resp.status = 0;
        resp.out_size = t_out.out_size;
        resp.timing = t_out;
        resp.time_us = (unsigned long)(t_total_end - t_total_start);
        if (g_verbose && !output_to_stdout) {
            print_response_stats(&resp, input_path, mode);
        } else if (g_verbose && output_to_stdout) {
            double ratio = (double)t_out.in_size / (t_out.out_size > 0 ? t_out.out_size : 1);
            fprintf(msg, "%s : %zu -> %zu (%.2f:1) in %.2f ms\n",
                    input_path,
                    (size_t)t_out.in_size,
                    (size_t)resp.out_size,
                    ratio,
                    resp.time_us / 1000.0);
        } else {
            double ratio = (double)t_out.in_size / (t_out.out_size > 0 ? t_out.out_size : 1);
            fprintf(msg, "%s : %zu -> %zu (%.2f:1) in %.2f ms\n",
                    input_path,
                    (size_t)t_out.in_size,
                    (size_t)resp.out_size,
                    ratio,
                    resp.time_us / 1000.0);
        }

        if (output_to_stdout) {
            if (copy_path_to_stream(effective_output_path, stdout) != 0) {
                fprintf(stderr, "Error: failed to emit streamed output to stdout\n");
                ret = 1;
            }
        }
    }

cleanup:
    if (ws_inited) lz4_gpu_workspace_free(&ws);
    if (kernel) clReleaseKernel(kernel);
    if (prog) clReleaseProgram(prog);
    ocl_release_runtime();
    if (have_temp_input) unlink(temp_input_path);
    if (have_temp_output) unlink(temp_output_path);
    return ret;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        show_help(argv[0]);
        return 0;
    }
    if (argc >= 2) {
        if (strcmp(argv[1], "--device-info") == 0) {
            if (argc != 2) {
                fprintf(stderr, "Error: --device-info does not accept additional arguments\n");
                return 1;
            }
            return lz4_print_device_info();
        }
        if (strcmp(argv[1], "--daemon") == 0) {
            return run_daemon();
        }
        if (strcmp(argv[1], "--stop-daemon") == 0) return stop_daemon_cmd();
        if (strcmp(argv[1], "--use-daemon") == 0) {
            int mode = mode_compress;
            int bench_mode = 0;
            double bench_seconds = 3.0;
            const char* input = NULL;
            char output[512] = {0};
            int output_explicit = 0;
            int raw_buffer = 0;
            int twophase = 0;
            for (int i = 2; i < argc; i++) {
                if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                    show_help(argv[0]);
                    return 0;
                }
                else if (strcmp(argv[i], "-c") == 0) mode = mode_compress;
                else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) mode = mode_decompress;
                else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) g_verbose = 1;
                else if (strcmp(argv[i], "--raw-buffer") == 0) raw_buffer = 1;
                else if (strcmp(argv[i], "--twophase") == 0) twophase = 1;
                else if (strcmp(argv[i], "--bench") == 0) {
                    bench_mode = 1;
                    if (i + 1 < argc && argv[i + 1][0] != '-') {
                        if (parse_positive_double_arg(argv[++i], &bench_seconds) != 0) {
                            fprintf(stderr, "Error: --bench duration must be positive\n");
                            return 1;
                        }
                    }
                }
                else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                    if (copy_cli_path(output, sizeof(output), argv[++i]) != 0) {
                        fprintf(stderr, "Error: output path is too long\n"); return 1;
                    }
                    output_explicit = 1;
                } else if ((strcmp(argv[i], "-B") == 0 || strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) && i + 1 < argc) {
                    g_cli_fixed_block_bytes = parse_size_bytes(argv[++i]);
                } else if ((strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--acceleration") == 0) && i + 1 < argc) {
                    if (parse_int_arg(argv[++i], &g_cli_acceleration) != 0) {
                        fprintf(stderr, "Error: acceleration requires an integer\n"); return 1;
                    }
                } else if (strcmp(argv[i], "--d-bits") == 0 && i + 1 < argc) {
                    if (parse_int_arg(argv[++i], &g_cli_hash_log) != 0) {
                        fprintf(stderr, "Error: --d-bits requires an integer\n"); return 1;
                    }
                } else if ((strcmp(argv[i], "--local") == 0) && i + 1 < argc) {
                    int parsed_local = 0;
                    if (parse_int_arg(argv[++i], &parsed_local) != 0 || parsed_local < 1) {
                        fprintf(stderr, "Error: --local requires a positive integer\n"); return 1;
                    }
                    g_cli_local_size = (size_t)parsed_local;
                } else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0) {
                    fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
                    return 1;
                } else {
                    if (!input) input = argv[i];
                    else if (!output_explicit) {
                        if (copy_cli_path(output, sizeof(output), argv[i]) != 0) {
                            fprintf(stderr, "Error: output path is too long\n"); return 1;
                        }
                        output_explicit = 1;
                    } else {
                        fprintf(stderr, "Error: Too many positional arguments\n");
                        return 1;
                    }
                }
            }
            if (!input && !raw_buffer) {
                fprintf(stderr, "Error: No input file specified\n");
                return 1;
            }
            if (validate_cli_config(twophase) != 0) return 1;
            if (twophase && mode != mode_compress) {
                fprintf(stderr, "Error: daemon two-phase currently supports compression only; use standalone mode to decode LZ4TP1\n");
                return 1;
            }
            if (twophase && raw_buffer) {
                fprintf(stderr, "Error: daemon two-phase does not support raw-buffer transport\n");
                return 1;
            }
            if (twophase && g_cli_fixed_block_bytes != 64 * 1024) {
                fprintf(stderr, "Error: daemon two-phase currently requires a 64K block size\n");
                return 1;
            }
            if (bench_mode) {
                if (twophase) {
                    fprintf(stderr, "Error: use standalone --tp-bench for two-phase measurements\n");
                    return 1;
                }
                if (mode != mode_compress) {
                    fprintf(stderr, "Error: --bench only supports compress mode input (it runs compress+decompress internally)\n");
                    return 1;
                }
                if (path_is_dash(input)) {
                    fprintf(stderr, "Error: --bench does not support stdin input ('-')\n");
                    return 1;
                }
                return run_lz4_bench(input,
                                    (int)g_cli_fixed_block_bytes,
                                    g_cli_acceleration,
                                    (int)g_cli_local_size,
                                    bench_seconds);
            }
            if (raw_buffer) {
                return run_lz4_client(mode, NULL, NULL, (int)g_cli_fixed_block_bytes, g_cli_acceleration, (int)g_cli_local_size, g_cli_hash_log, 1, twophase);
            }
            if (!output_explicit) {
                const char* suffix = mode == mode_compress ? (twophase ? ".lz4tp" : ".lz4") : ".dec";
                if (append_cli_suffix(output, sizeof(output), input, suffix) != 0) {
                    fprintf(stderr, "Error: derived output path is too long\n"); return 1;
                }
            }
            if (path_is_dash(input) || path_is_dash(output)) {
                fprintf(stderr, "Error: '-' stream I/O is only supported in standalone mode\n");
                return 1;
            }
            return run_lz4_client(mode, input, output, (int)g_cli_fixed_block_bytes, g_cli_acceleration, (int)g_cli_local_size, g_cli_hash_log, 0, twophase);
        }
    }
    return run_lz4_standalone(argc, argv);
}
