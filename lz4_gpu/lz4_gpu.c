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
#define access _access
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
#include <time.h>
#include <math.h>
#include "lz4_gpu_core.h"
#include "lz4_gpu_debug.h"
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_utils.h"
#include "timing.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

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

static cl_context ctx;
static cl_command_queue queue;
static cl_device_id dev;

static void ocl_init() {
    cl_int err;
    cl_platform_id selected_pf = NULL;
    dev = NULL;
    cl_int r = lz4_select_opencl_platform_device(&selected_pf, &dev);

    if (r != CL_SUCCESS || dev == NULL) {
        fprintf(stderr, "OpenCL init failed: clGetDeviceIDs failed for all type/plat combos\n");
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
        ctx = NULL;
        return;
    }
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
        ctx = NULL;
        queue = NULL;
        return;
    }
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
    char* endptr;
    size_t val = strtoul(s, &endptr, 10);
    if (*endptr == 'k' || *endptr == 'K') val *= 1024;
    else if (*endptr == 'm' || *endptr == 'M') val *= 1024 * 1024;
    return val;
}

static int path_is_dash(const char* path) {
    return path && strcmp(path, "-") == 0;
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
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(input_ref);
        return 1;
    }

    cl_int err = CL_SUCCESS;
    cl_kernel kcomp = clCreateKernel(prog, "lz4_compress_block", &err);
    if (err != CL_SUCCESS || !kcomp) {
        fprintf(stderr, "bench error: create compress kernel failed (%d)\n", err);
        clReleaseProgram(prog);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        free(input_ref);
        return 1;
    }
    cl_kernel kdec = clCreateKernel(prog, "lz4_decompress_blocks", &err);
    if (err != CL_SUCCESS || !kdec) {
        fprintf(stderr, "bench error: create decompress kernel failed (%d)\n", err);
        clReleaseKernel(kcomp);
        clReleaseProgram(prog);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
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
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
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
static size_t tp_roundup64(size_t x) { return ((x + 63) / 64) * 64; }

static int run_lz4_tp_bench(const char* input_path, int block_size, double bench_seconds) {
    struct stat st;
    if (!input_path || stat(input_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "tp-bench error: invalid input file\n");
        return 1;
    }
    if (block_size <= 0) block_size = 64 * 1024;
    (void)bench_seconds;

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
        unsetenv("LZ4_GPU_EPOCH32");
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
    setenv("LZ4_GPU_EPOCH32", "1", 1);
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
            if (d_prefix) clReleaseMemObject(d_prefix); if (d_own) clReleaseMemObject(d_own);
            if (d_out) clReleaseMemObject(d_out); if (d_sizes) clReleaseMemObject(d_sizes);
            if (d_coff) clReleaseMemObject(d_coff); if (d_decout) clReleaseMemObject(d_decout);
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
    unsetenv("LZ4_GPU_EPOCH32");
    free(input_ref);
    if (queue) clReleaseCommandQueue(queue);
    if (ctx) clReleaseContext(ctx);
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

static int lz4_tp_compress_to_file(const char* input_path, const char* output_path,
                                   int N, int block_size) {
    struct stat st;
    if (!input_path || stat(input_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "tp-compress: invalid input\n"); return 1;
    }
    if (block_size <= 0) block_size = 64 * 1024;
    if (N < 1) N = 1; if (N > 64) N = 64;

    size_t orig_size = (size_t)st.st_size;
    unsigned char* input_ref = (unsigned char*)malloc(orig_size);
    unsigned long rd_us = 0;
    if (!input_ref || lz4_read_file_to_buf(input_path, input_ref, orig_size, &rd_us) != 0) {
        free(input_ref); fprintf(stderr, "tp-compress: read failed\n"); return 1;
    }

    setenv("LZ4_GPU_EPOCH32", "1", 1);   /* two-phase prefix dicts require epoch32 */
    ocl_init();
    if (!ctx || !queue) { free(input_ref); fprintf(stderr, "tp-compress: OpenCL init failed\n"); return 1; }

    int hash_log = g_cli_hash_log;
    size_t dict_entries = (size_t)1u << hash_log;
    size_t nblk = (orig_size + (size_t)block_size - 1) / (size_t)block_size;
    int nk = N - 1;
    int segLenMax = (block_size + N - 1) / N;
    int segMaxOut = segLenMax + segLenMax / 255 + 64;
    size_t meta_cnt = nblk * (size_t)N;

    cl_int err = CL_SUCCESS;
    cl_program prog = lz4_load_program(ctx, dev, hash_log, (size_t)block_size);
    if (!prog) { fprintf(stderr, "tp-compress: prog load failed\n"); free(input_ref); return 1; }
    cl_kernel kbuild = clCreateKernel(prog, "lz4_tp_build_prefix", &err);
    cl_kernel kscan  = clCreateKernel(prog, "lz4_tp_scan_seg", &err);
    if (!kbuild || !kscan) { fprintf(stderr, "tp-compress: kernels failed %d\n", err); free(input_ref); return 1; }

    size_t prefix_bytes = (size_t)nblk * (size_t)nk * dict_entries * sizeof(cl_uint);
    size_t own_bytes    = (size_t)nblk * (size_t)N  * dict_entries * sizeof(cl_uint);
    size_t out_bytes    = (size_t)nblk * (size_t)N  * (size_t)segMaxOut;

    cl_mem d_input  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, orig_size, NULL, &err);
    cl_mem d_prefix = clCreateBuffer(ctx, CL_MEM_READ_WRITE, prefix_bytes ? prefix_bytes : 4, NULL, &err);
    cl_mem d_own    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, own_bytes, NULL, &err);
    cl_mem d_out    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    cl_mem d_sizes  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, meta_cnt * sizeof(cl_uint), NULL, &err);
    if (!d_input || !d_prefix || !d_own || !d_out || !d_sizes) {
        fprintf(stderr, "tp-compress: buffer alloc failed (%d)\n", err); free(input_ref); return 1;
    }
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, orig_size, input_ref, 0, NULL, NULL);
    { cl_uint zero = 0;
      if (prefix_bytes) clEnqueueFillBuffer(queue, d_prefix, &zero, sizeof(zero), 0, prefix_bytes, 0, NULL, NULL);
      clEnqueueFillBuffer(queue, d_own, &zero, sizeof(zero), 0, own_bytes, 0, NULL, NULL);
      clFinish(queue);
    }

    cl_int i_nblk = (cl_int)nblk, i_insize = (cl_int)orig_size, i_blk = block_size,
           i_segmax = segMaxOut, i_N = N;
    cl_uint epoch = 1;
    size_t lws1 = 1;
    clSetKernelArg(kbuild, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kbuild, 1, sizeof(cl_mem), &d_prefix);
    clSetKernelArg(kbuild, 2, sizeof(cl_int), &i_nblk);
    clSetKernelArg(kbuild, 3, sizeof(cl_int), &i_insize);
    clSetKernelArg(kbuild, 4, sizeof(cl_int), &i_blk);
    clSetKernelArg(kbuild, 5, sizeof(cl_int), &i_N);
    clSetKernelArg(kbuild, 6, sizeof(cl_uint), &epoch);
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
    clSetKernelArg(kscan, 10, sizeof(cl_uint), &epoch);

    if (nk >= 1) {
        size_t g_build = (size_t)nblk * (size_t)nk * 4; /* LZ4_TP_BUILD_CHUNKS=4 */
        err = clEnqueueNDRangeKernel(queue, kbuild, 1, NULL, &g_build, &lws1, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "tp-compress: build enqueue %d\n", err); return 1; }
    }
    size_t g_scan = (size_t)nblk * (size_t)N;
    err = clEnqueueNDRangeKernel(queue, kscan, 1, NULL, &g_scan, &lws1, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "tp-compress: scan enqueue %d\n", err); return 1; }
    clFinish(queue);

    cl_uint* sizes = (cl_uint*)malloc(meta_cnt * sizeof(cl_uint));
    unsigned char* padded = (unsigned char*)malloc(out_bytes);
    if (!sizes || !padded) { fprintf(stderr, "tp-compress: host oom\n"); return 1; }
    clEnqueueReadBuffer(queue, d_sizes, CL_TRUE, 0, meta_cnt * sizeof(cl_uint), sizes, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out,   CL_TRUE, 0, out_bytes, padded, 0, NULL, NULL);

    /* compact + sanity */
    size_t payload_total = 0;
    for (size_t g = 0; g < meta_cnt; g++) {
        if (sizes[g] == 0 || sizes[g] > (cl_uint)segMaxOut) {
            fprintf(stderr, "tp-compress: bad segment size seg=%zu sz=%u (overflow?)\n", g, sizes[g]);
            return 1;
        }
        payload_total += sizes[g];
    }
    unsigned char* payload = (unsigned char*)malloc(payload_total);
    if (!payload) { fprintf(stderr, "tp-compress: payload oom\n"); return 1; }
    size_t off = 0;
    for (size_t g = 0; g < meta_cnt; g++) {
        memcpy(payload + off, padded + g * (size_t)segMaxOut, sizes[g]);
        off += sizes[g];
    }

    /* write frame */
    FILE* f = fopen(output_path, "wb");
    if (!f) { fprintf(stderr, "tp-compress: cannot open %s\n", output_path); return 1; }
    cl_uint u_N = (cl_uint)N, u_hl = (cl_uint)hash_log, u_bs = (cl_uint)block_size, u_nblk = (cl_uint)nblk;
    unsigned long long u_orig = (unsigned long long)orig_size;
    fwrite(LZ4TP_MAGIC, 1, 8, f);
    fwrite(&u_N, 4, 1, f); fwrite(&u_hl, 4, 1, f); fwrite(&u_bs, 4, 1, f); fwrite(&u_nblk, 4, 1, f);
    fwrite(&u_orig, 8, 1, f);
    fwrite(sizes, sizeof(cl_uint), meta_cnt, f);
    fwrite(payload, 1, payload_total, f);
    fclose(f);

    size_t frame_total = 8 + 4*4 + 8 + meta_cnt * sizeof(cl_uint) + payload_total;
    fprintf(stderr, "[tp-compress] %s -> %s : %zu -> %zu (%.3f:1) N=%d blocks=%zu\n",
            input_path, output_path, orig_size, frame_total,
            frame_total > 0 ? (double)orig_size / (double)frame_total : 0.0, N, nblk);

    free(sizes); free(padded); free(payload); free(input_ref);
    clReleaseMemObject(d_input); clReleaseMemObject(d_prefix); clReleaseMemObject(d_own);
    clReleaseMemObject(d_out); clReleaseMemObject(d_sizes);
    clReleaseKernel(kbuild); clReleaseKernel(kscan); clReleaseProgram(prog);
    unsetenv("LZ4_GPU_EPOCH32");
    if (queue) clReleaseCommandQueue(queue);
    if (ctx) clReleaseContext(ctx);
    return 0;
}

static int lz4_tp_decompress_from_file(const char* input_path, const char* output_path) {
    FILE* f = fopen(input_path, "rb");
    if (!f) { fprintf(stderr, "tp-decompress: cannot open %s\n", input_path); return 1; }
    char magic[8] = {0};
    cl_uint u_N = 0, u_hl = 0, u_bs = 0, u_nblk = 0;
    unsigned long long u_orig = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, LZ4TP_MAGIC, 8) != 0) {
        fprintf(stderr, "tp-decompress: not a LZ4TP frame\n"); fclose(f); return 1;
    }
    fread(&u_N, 4, 1, f); fread(&u_hl, 4, 1, f); fread(&u_bs, 4, 1, f); fread(&u_nblk, 4, 1, f);
    fread(&u_orig, 8, 1, f);
    int N = (int)u_N, hash_log = (int)u_hl, block_size = (int)u_bs;
    size_t nblk = (size_t)u_nblk, orig_size = (size_t)u_orig, meta_cnt = nblk * (size_t)N;
    if (N < 1 || block_size <= 0 || nblk == 0) { fprintf(stderr, "tp-decompress: bad header\n"); fclose(f); return 1; }

    cl_uint* comp_sizes = (cl_uint*)malloc(meta_cnt * sizeof(cl_uint));
    if (!comp_sizes || fread(comp_sizes, sizeof(cl_uint), meta_cnt, f) != meta_cnt) {
        fprintf(stderr, "tp-decompress: cannot read sizes\n"); fclose(f); return 1;
    }
    size_t payload_total = 0;
    for (size_t g = 0; g < meta_cnt; g++) payload_total += comp_sizes[g];
    unsigned char* payload = (unsigned char*)malloc(payload_total);
    if (!payload || fread(payload, 1, payload_total, f) != payload_total) {
        fprintf(stderr, "tp-decompress: cannot read payload\n"); fclose(f); return 1;
    }
    fclose(f);

    /* comp_offsets = prefix sum of comp_sizes (compacted layout) */
    cl_uint* comp_off = (cl_uint*)malloc(meta_cnt * sizeof(cl_uint));
    { size_t acc = 0; for (size_t g = 0; g < meta_cnt; g++) { comp_off[g] = (cl_uint)acc; acc += comp_sizes[g]; } }

    setenv("LZ4_GPU_EPOCH32", "1", 1);   /* keep build identical to compress side */
    ocl_init();
    if (!ctx || !queue) { fprintf(stderr, "tp-decompress: OpenCL init failed\n"); return 1; }

    cl_int err = CL_SUCCESS;
    cl_program prog = lz4_load_program(ctx, dev, hash_log > 0 ? hash_log : g_cli_hash_log, (size_t)block_size);
    if (!prog) { fprintf(stderr, "tp-decompress: prog load failed\n"); return 1; }
    cl_kernel kdec = clCreateKernel(prog, "lz4_tp_decompress_segmented", &err);
    if (!kdec) { fprintf(stderr, "tp-decompress: kernel failed %d\n", err); return 1; }

    size_t dec_bytes = nblk * (size_t)block_size;
    cl_mem d_payload = clCreateBuffer(ctx, CL_MEM_READ_ONLY, payload_total ? payload_total : 4, NULL, &err);
    cl_mem d_coff    = clCreateBuffer(ctx, CL_MEM_READ_ONLY, meta_cnt * sizeof(cl_uint), NULL, &err);
    cl_mem d_csz     = clCreateBuffer(ctx, CL_MEM_READ_ONLY, meta_cnt * sizeof(cl_uint), NULL, &err);
    cl_mem d_decout  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, dec_bytes, NULL, &err);
    cl_mem d_sout    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nblk * sizeof(cl_uint), NULL, &err);
    if (!d_payload || !d_coff || !d_csz || !d_decout || !d_sout) {
        fprintf(stderr, "tp-decompress: buffer alloc failed (%d)\n", err); return 1;
    }
    if (payload_total) clEnqueueWriteBuffer(queue, d_payload, CL_TRUE, 0, payload_total, payload, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_coff, CL_TRUE, 0, meta_cnt * sizeof(cl_uint), comp_off, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_csz,  CL_TRUE, 0, meta_cnt * sizeof(cl_uint), comp_sizes, 0, NULL, NULL);

    cl_uint u_blk = (cl_uint)block_size, u_tot = (cl_uint)nblk; cl_int i_N = N;
    clSetKernelArg(kdec, 0, sizeof(cl_mem), &d_payload);
    clSetKernelArg(kdec, 1, sizeof(cl_mem), &d_decout);
    clSetKernelArg(kdec, 2, sizeof(cl_mem), &d_coff);
    clSetKernelArg(kdec, 3, sizeof(cl_mem), &d_csz);
    clSetKernelArg(kdec, 4, sizeof(cl_mem), &d_sout);
    clSetKernelArg(kdec, 5, sizeof(cl_uint), &u_blk);
    clSetKernelArg(kdec, 6, sizeof(cl_int), &i_N);
    clSetKernelArg(kdec, 7, sizeof(cl_uint), &u_tot);

    size_t g_dec = nblk, lws1 = 1;
    err = clEnqueueNDRangeKernel(queue, kdec, 1, NULL, &g_dec, &lws1, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "tp-decompress: dec enqueue %d\n", err); return 1; }
    clFinish(queue);

    unsigned char* dec = (unsigned char*)malloc(dec_bytes);
    cl_uint* sout = (cl_uint*)malloc(nblk * sizeof(cl_uint));
    clEnqueueReadBuffer(queue, d_decout, CL_TRUE, 0, dec_bytes, dec, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_sout, CL_TRUE, 0, nblk * sizeof(cl_uint), sout, 0, NULL, NULL);

    /* validate per-block decoded lengths sum to orig_size */
    size_t total = 0;
    for (size_t b = 0; b < nblk; b++) {
        if (sout[b] == 0xFFFFFFFFu) { fprintf(stderr, "tp-decompress: block %zu decode error\n", b); return 1; }
        total += sout[b];
    }
    if (total != orig_size) {
        fprintf(stderr, "tp-decompress: length mismatch got=%zu expect=%zu\n", total, orig_size); return 1;
    }

    FILE* of = fopen(output_path, "wb");
    if (!of) { fprintf(stderr, "tp-decompress: cannot open %s\n", output_path); return 1; }
    fwrite(dec, 1, orig_size, of);   /* full blocks are exactly block_size, last is remainder */
    fclose(of);
    fprintf(stderr, "[tp-decompress] %s -> %s : %zu bytes N=%d blocks=%zu OK\n",
            input_path, output_path, orig_size, N, nblk);

    free(comp_sizes); free(comp_off); free(payload); free(dec); free(sout);
    clReleaseMemObject(d_payload); clReleaseMemObject(d_coff); clReleaseMemObject(d_csz);
    clReleaseMemObject(d_decout); clReleaseMemObject(d_sout);
    clReleaseKernel(kdec); clReleaseProgram(prog);
    unsetenv("LZ4_GPU_EPOCH32");
    if (queue) clReleaseCommandQueue(queue);
    if (ctx) clReleaseContext(ctx);
    return 0;
}
/* ============================================================================
 * I2: the selector, IN the binary (not Python).
 *   --calibrate <file> [-o profile] : measure this device's W_sat, store it.
 *   -c --auto <in> -o <out>         : runtime pick N* = clamp(round(0.75*W_sat/nblk),1,8),
 *                                     dispatch two-phase (N=1 == base via seg path).
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
    size_t nblk = (size + (size_t)block_size - 1) / (size_t)block_size;
    int nk = N - 1;
    int segLenMax = (block_size + N - 1) / N;
    int segMaxOut = segLenMax + segLenMax / 255 + 64;
    size_t meta_cnt = nblk * (size_t)N;
    size_t prefix_bytes = (size_t)nblk * (size_t)nk * dict_entries * sizeof(cl_uint);
    size_t own_bytes    = (size_t)nblk * (size_t)N  * dict_entries * sizeof(cl_uint);
    size_t out_bytes    = (size_t)nblk * (size_t)N  * (size_t)segMaxOut;

    cl_mem d_input  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size, NULL, &err);
    cl_mem d_prefix = clCreateBuffer(ctx, CL_MEM_READ_WRITE, prefix_bytes ? prefix_bytes : 4, NULL, &err);
    cl_mem d_own    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, own_bytes, NULL, &err);
    cl_mem d_out    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    cl_mem d_sizes  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, meta_cnt * sizeof(cl_uint), NULL, &err);
    if (!d_input || !d_prefix || !d_own || !d_out || !d_sizes) {
        if (d_input) clReleaseMemObject(d_input); if (d_prefix) clReleaseMemObject(d_prefix);
        if (d_own) clReleaseMemObject(d_own); if (d_out) clReleaseMemObject(d_out);
        if (d_sizes) clReleaseMemObject(d_sizes);
        return 0.0;
    }
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, size, input_ref, 0, NULL, NULL);
    { cl_uint zero = 0;
      if (prefix_bytes) clEnqueueFillBuffer(queue, d_prefix, &zero, 4, 0, prefix_bytes, 0, NULL, NULL);
      clEnqueueFillBuffer(queue, d_own, &zero, 4, 0, own_bytes, 0, NULL, NULL);
      clFinish(queue); }

    cl_int i_nblk = (cl_int)nblk, i_insize = (cl_int)size, i_blk = block_size, i_segmax = segMaxOut, i_N = N;
    size_t lws1 = 1, g_build = (size_t)nblk * (size_t)nk * 4, g_scan = (size_t)nblk * (size_t)N;
    clSetKernelArg(kbuild,0,sizeof(cl_mem),&d_input); clSetKernelArg(kbuild,1,sizeof(cl_mem),&d_prefix);
    clSetKernelArg(kbuild,2,sizeof(cl_int),&i_nblk); clSetKernelArg(kbuild,3,sizeof(cl_int),&i_insize);
    clSetKernelArg(kbuild,4,sizeof(cl_int),&i_blk); clSetKernelArg(kbuild,5,sizeof(cl_int),&i_N);
    clSetKernelArg(kscan,0,sizeof(cl_mem),&d_input); clSetKernelArg(kscan,1,sizeof(cl_mem),&d_out);
    clSetKernelArg(kscan,2,sizeof(cl_mem),&d_sizes); clSetKernelArg(kscan,3,sizeof(cl_mem),&d_own);
    clSetKernelArg(kscan,4,sizeof(cl_mem),&d_prefix); clSetKernelArg(kscan,5,sizeof(cl_int),&i_nblk);
    clSetKernelArg(kscan,6,sizeof(cl_int),&i_insize); clSetKernelArg(kscan,7,sizeof(cl_int),&i_blk);
    clSetKernelArg(kscan,8,sizeof(cl_int),&i_segmax); clSetKernelArg(kscan,9,sizeof(cl_int),&i_N);

    double us[16]; int nu = 0; cl_uint epoch = 1;
    const int WARM = 2, MEAS = 5;
    for (int it = 0; it < WARM + MEAS; ++it, ++epoch) {
        double t = 0.0;
        if (nk >= 1) {
            clSetKernelArg(kbuild, 6, sizeof(cl_uint), &epoch);
            cl_event e; if (clEnqueueNDRangeKernel(queue,kbuild,1,NULL,&g_build,&lws1,0,NULL,&e)!=CL_SUCCESS){break;}
            clWaitForEvents(1,&e); t += event_elapsed_us(e); clReleaseEvent(e);
        }
        clSetKernelArg(kscan, 10, sizeof(cl_uint), &epoch);
        cl_event e2; if (clEnqueueNDRangeKernel(queue,kscan,1,NULL,&g_scan,&lws1,0,NULL,&e2)!=CL_SUCCESS){break;}
        clWaitForEvents(1,&e2); t += event_elapsed_us(e2); clReleaseEvent(e2);
        if (it >= WARM && nu < 16) us[nu++] = t;
    }
    clReleaseMemObject(d_input); clReleaseMemObject(d_prefix); clReleaseMemObject(d_own);
    clReleaseMemObject(d_out); clReleaseMemObject(d_sizes);
    if (nu == 0) return 0.0;
    double med = median_double(us, nu);
    return med > 0 ? (double)size / med : 0.0;
}

static int lz4_tp_pick_N(size_t nblk, int W_sat) {
    int cand[4] = {1,2,4,8}; int best = 1;
    double thresh = 0.75 * (double)W_sat;       /* conservative margin (validated) */
    for (int i = 0; i < 4; i++) if ((double)nblk * cand[i] <= thresh) best = cand[i];
    return best;
}

static void lz4_device_name(char* buf, size_t n) {
    buf[0] = 0;
    if (dev) clGetDeviceInfo(dev, CL_DEVICE_NAME, n, buf, NULL);
}

static const char* lz4_profile_path(void) {
    const char* p = getenv("LZ4TP_PROFILE");
    return (p && *p) ? p : "lz4tp.profile";
}

/* read W_sat for the current device from the profile; returns W_sat or 0 if absent */
static int lz4_profile_lookup(const char* devname) {
    FILE* f = fopen(lz4_profile_path(), "r");
    if (!f) return 0;
    char line[512]; int w = 0;
    while (fgets(line, sizeof(line), f)) {
        char* tab = strrchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        if (strcmp(line, devname) == 0) { w = atoi(tab + 1); break; }
    }
    fclose(f);
    return w;
}

static void lz4_profile_store(const char* devname, int W_sat) {
    /* rewrite file: keep other devices, replace/append this one */
    const char* path = lz4_profile_path();
    char (*names)[256] = NULL; int* ws = NULL; int cnt = 0, cap = 0, found = 0;
    FILE* f = fopen(path, "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            char* nl = strchr(line, '\n'); if (nl) *nl = 0;
            char* tab = strrchr(line, '\t'); if (!tab) continue; *tab = 0;
            if (cnt == cap) { cap = cap ? cap*2 : 8;
                names = realloc(names, cap*sizeof(*names)); ws = realloc(ws, cap*sizeof(int)); }
            strncpy(names[cnt], line, 255); names[cnt][255]=0; ws[cnt] = atoi(tab+1); cnt++;
        }
        fclose(f);
    }
    f = fopen(path, "w");
    if (!f) { fprintf(stderr, "calibrate: cannot write %s\n", path); free(names); free(ws); return; }
    for (int i = 0; i < cnt; i++) {
        if (strcmp(names[i], devname) == 0) { fprintf(f, "%s\t%d\n", devname, W_sat); found = 1; }
        else fprintf(f, "%s\t%d\n", names[i], ws[i]);
    }
    if (!found) fprintf(f, "%s\t%d\n", devname, W_sat);
    fclose(f); free(names); free(ws);
}

static int lz4_calibrate_device(const char* cal_file, int block_size) {
    struct stat st;
    if (!cal_file || stat(cal_file, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "calibrate: invalid calibration file\n"); return 1;
    }
    if (block_size <= 0) block_size = 64 * 1024;
    size_t full = (size_t)st.st_size;
    unsigned char* input_ref = (unsigned char*)malloc(full);
    unsigned long rd = 0;
    if (!input_ref || lz4_read_file_to_buf(cal_file, input_ref, full, &rd) != 0) {
        free(input_ref); fprintf(stderr, "calibrate: read failed\n"); return 1;
    }
    setenv("LZ4_GPU_EPOCH32", "1", 1);
    ocl_init();
    if (!ctx || !queue) { free(input_ref); fprintf(stderr, "calibrate: OpenCL init failed\n"); return 1; }
    char devname[256]; lz4_device_name(devname, sizeof(devname));
    int hash_log = g_cli_hash_log;
    cl_int err = CL_SUCCESS;
    cl_program prog = lz4_load_program(ctx, dev, hash_log, (size_t)block_size);
    cl_kernel kbuild = prog ? clCreateKernel(prog, "lz4_tp_build_prefix", &err) : NULL;
    cl_kernel kscan  = prog ? clCreateKernel(prog, "lz4_tp_scan_seg", &err) : NULL;
    if (!kbuild || !kscan) { fprintf(stderr, "calibrate: kernels failed\n"); return 1; }

    /* block-count ladder, capped at the cal file's block count */
    size_t full_nblk = full / (size_t)block_size;
    int ladder[] = {2,4,8,16,32,64,128,256,384,512,633};
    int NL = (int)(sizeof(ladder)/sizeof(ladder[0]));
    int cand[4] = {1,2,4,8};
    /* per-ladder-point throughput for N in {1,2,4,8}; store to fit W_sat */
    double tp[16][4]; int nblk_of[16]; int L = 0;
    fprintf(stderr, "[calibrate] device=%s  block=%dK\n", devname, block_size/1024);
    for (int li = 0; li < NL && L < 16; li++) {
        size_t nb = (size_t)ladder[li];
        if (nb > full_nblk) break;
        size_t size = nb * (size_t)block_size;
        nblk_of[L] = (int)nb;
        for (int ci = 0; ci < 4; ci++)
            tp[L][ci] = lz4_tp_measure_one(input_ref, size, cand[ci], block_size, hash_log, kbuild, kscan);
        fprintf(stderr, "  nblk=%-4d N1=%.0f N2=%.0f N4=%.0f N8=%.0f MB/s\n",
                (int)nb, tp[L][0], tp[L][1], tp[L][2], tp[L][3]);
        L++;
    }
    if (L < 2) { fprintf(stderr, "calibrate: cal file too small (need >=8MB for a useful W_sat)\n"); }

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
    lz4_profile_store(devname, bestW);
    fprintf(stderr, "[calibrate] stored to %s\n", lz4_profile_path());

    free(input_ref);
    clReleaseKernel(kbuild); clReleaseKernel(kscan); clReleaseProgram(prog);
    unsetenv("LZ4_GPU_EPOCH32");
    if (queue) clReleaseCommandQueue(queue);
    if (ctx) clReleaseContext(ctx);
    return 0;
}

/* -c --auto : pick N from the stored W_sat, then compress. Needs device name,
 * so we peek the device via a throwaway ocl_init, read profile, release, then
 * call the real compressor (which re-inits). Cheap relative to compression. */
static int lz4_tp_compress_auto(const char* input_path, const char* output_path, int block_size) {
    struct stat st;
    if (!input_path || stat(input_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "auto: invalid input\n"); return 1;
    }
    if (block_size <= 0) block_size = 64 * 1024;
    size_t nblk = ((size_t)st.st_size + (size_t)block_size - 1) / (size_t)block_size;

    ocl_init();
    if (!ctx) { fprintf(stderr, "auto: OpenCL init failed\n"); return 1; }
    char devname[256]; lz4_device_name(devname, sizeof(devname));
    if (queue) clReleaseCommandQueue(queue);
    if (ctx) clReleaseContext(ctx);
    ctx = NULL; queue = NULL;

    int W_sat = lz4_profile_lookup(devname);
    int N;
    if (W_sat <= 0) {
        N = 1;
        fprintf(stderr, "[auto] no profile for device \"%s\" (run --calibrate); using N=1 (safe, no gain)\n", devname);
    } else {
        N = lz4_tp_pick_N(nblk, W_sat);
        fprintf(stderr, "[auto] device=%s W_sat=%d nblk=%zu -> N=%d\n", devname, W_sat, nblk, N);
    }
    return lz4_tp_compress_to_file(input_path, output_path, N, block_size);
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
            if (i + 1 < argc) g_cli_tp_n = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -N requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "--tp-bench") == 0) {
            tp_bench_mode = 1;
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                bench_seconds = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                strncpy(output_path, argv[++i], sizeof(output_path) - 1);
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: -o requires an argument\n");
                show_help(argv[0]); return 1;
            }
        } else if (strcmp(argv[i], "-B") == 0 || strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) {
            if (i + 1 < argc) g_cli_fixed_block_bytes = parse_size_bytes(argv[++i]);
            else { fprintf(stderr, "Error: -B requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--acceleration") == 0) {
            if (i + 1 < argc) g_cli_acceleration = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -a requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "--d-bits") == 0) {
            if (i + 1 < argc) g_cli_hash_log = atoi(argv[++i]);
            else { fprintf(stderr, "Error: --d-bits requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "--local") == 0) {
            if (i + 1 < argc) g_cli_local_size = atoi(argv[++i]);
            else { fprintf(stderr, "Error: --local requires an argument\n"); return 1; }
        } else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0) {
            fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
            show_help(argv[0]); return 1;
        } else {
            if (!input_path) input_path = argv[i];
            else if (!output_explicit) {
                strncpy(output_path, argv[i], sizeof(output_path)-1);
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: Too many positional arguments\n");
                show_help(argv[0]); return 1;
            }
        }
    }

    if (!input_path) {
        show_help(argv[0]);
        return 1;
    }

    if (calibrate_mode) {
        return lz4_calibrate_device(input_path, (int)g_cli_fixed_block_bytes);
    }
    if (auto_mode) {
        char au_out[512];
        const char* outp = output_explicit ? output_path : NULL;
        if (!outp) { snprintf(au_out, sizeof(au_out), "%s.lz4tp", input_path); outp = au_out; }
        return lz4_tp_compress_auto(input_path, outp, (int)g_cli_fixed_block_bytes);
    }

    if (twophase_mode) {
        char tp_out[512];
        const char* outp = output_explicit ? output_path : NULL;
        if (!outp) {
            if (mode == mode_compress) snprintf(tp_out, sizeof(tp_out), "%s.lz4tp", input_path);
            else snprintf(tp_out, sizeof(tp_out), "%s.tpout", input_path);
            outp = tp_out;
        }
        if (mode == mode_compress)
            return lz4_tp_compress_to_file(input_path, outp, g_cli_tp_n, (int)g_cli_fixed_block_bytes);
        return lz4_tp_decompress_from_file(input_path, outp);
    }

    if (tp_bench_mode) {
        if (path_is_dash(input_path)) {
            fprintf(stderr, "Error: --tp-bench does not support stdin input ('-')\n");
            return 1;
        }
        return run_lz4_tp_bench(input_path, (int)g_cli_fixed_block_bytes, bench_seconds);
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
            strncpy(output_path, "-", sizeof(output_path) - 1);
            output_explicit = 1;
        } else if (mode == mode_compress) {
            snprintf(output_path, sizeof(output_path), "%s.lz4", input_path);
        } else {
            snprintf(output_path, sizeof(output_path), "%s.dec", input_path);
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
    if (queue) { clReleaseCommandQueue(queue); queue = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
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
                        bench_seconds = atof(argv[++i]);
                    }
                }
                else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                    strncpy(output, argv[++i], sizeof(output)-1);
                    output_explicit = 1;
                } else if ((strcmp(argv[i], "-B") == 0 || strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) && i + 1 < argc) {
                    g_cli_fixed_block_bytes = parse_size_bytes(argv[++i]);
                } else if ((strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--acceleration") == 0) && i + 1 < argc) {
                    g_cli_acceleration = atoi(argv[++i]);
                } else if (strcmp(argv[i], "--d-bits") == 0 && i + 1 < argc) {
                    g_cli_hash_log = atoi(argv[++i]);
                } else if ((strcmp(argv[i], "--local") == 0) && i + 1 < argc) {
                    g_cli_local_size = atoi(argv[++i]);
                } else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0) {
                    fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
                    return 1;
                } else {
                    if (!input) input = argv[i];
                    else if (!output_explicit) {
                        strncpy(output, argv[i], sizeof(output)-1);
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
            if (bench_mode) {
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
                if (mode == mode_compress) snprintf(output, sizeof(output), "%s%s", input, twophase ? ".lz4tp" : ".lz4");
                else snprintf(output, sizeof(output), "%s.dec", input);
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
