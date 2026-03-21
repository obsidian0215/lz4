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
#include <time.h>
#include "lz4_gpu_core.h"
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_utils.h"
#include "timing.h"

/* Forward declarations */
#if defined(_WIN32)
static int run_daemon(void) {
    fprintf(stderr, "Daemon mode is not supported in Windows builds. Use standalone or bench mode.\n");
    return 1;
}
static int run_lz4_client(int mode, const char* input_path, const char* output_path, int block_size, int acceleration, int local_size) {
    (void)mode; (void)input_path; (void)output_path; (void)block_size; (void)acceleration; (void)local_size;
    fprintf(stderr, "--use-daemon is not supported in Windows builds. Use standalone mode.\n");
    return 1;
}
#else
int run_daemon(void);
int run_lz4_client(int mode, const char* input_path, const char* output_path, int block_size, int acceleration, int local_size);
#endif

int g_verbose = 0;
static size_t g_cli_local_size = 1;
static size_t g_cli_fixed_block_bytes = 32 * 1024;
static int g_cli_acceleration = 1;

static cl_context ctx;
static cl_command_queue queue;
static cl_device_id dev;

static cl_int lz4_try_get_device(cl_platform_id* platforms,
                                 cl_uint num_platforms,
                                 cl_device_type dtype,
                                 cl_device_id* out_dev,
                                 cl_platform_id* out_pf) {
    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        cl_device_id tmp_dev = NULL;
        cl_int r = clGetDeviceIDs(platforms[pi], dtype, 1, &tmp_dev, NULL);
        if (r == CL_SUCCESS && tmp_dev != NULL) {
            *out_dev = tmp_dev;
            *out_pf = platforms[pi];
            return CL_SUCCESS;
        }
    }
    return CL_DEVICE_NOT_FOUND;
}

static cl_device_type preferred_opencl_device_type(void) {
    const char* pref = getenv("FORCE_OPENCL_DEVICE");
    if (!pref || !*pref) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "CPU") == 0) return CL_DEVICE_TYPE_CPU;
    if (strcasecmp(pref, "GPU") == 0) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "DEFAULT") == 0) return CL_DEVICE_TYPE_DEFAULT;
    if (strcasecmp(pref, "ALL") == 0) return CL_DEVICE_TYPE_ALL;
    return CL_DEVICE_TYPE_GPU;
}

static void ocl_init() {
    cl_int err;
    cl_device_type pref_type = preferred_opencl_device_type();

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "OpenCL init failed: clGetPlatformIDs err=%d\n", err);
        ctx = NULL;
        queue = NULL;
        return;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
        fprintf(stderr, "OpenCL init failed: malloc platforms\n");
        ctx = NULL;
        queue = NULL;
        return;
    }

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL init failed: clGetPlatformIDs err=%d\n", err);
        free(platforms);
        ctx = NULL;
        queue = NULL;
        return;
    }

    dev = NULL;
    cl_platform_id selected_pf = NULL;
    cl_int r = CL_DEVICE_NOT_FOUND;

    if (pref_type == CL_DEVICE_TYPE_GPU) {
        r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
    } else if (pref_type == CL_DEVICE_TYPE_CPU) {
        r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
    } else if (pref_type == CL_DEVICE_TYPE_DEFAULT) {
        r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
    } else {
        r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
    }

    free(platforms);

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
    queue = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || queue == NULL) {
        fprintf(stderr, "OpenCL init failed: clCreateCommandQueue err=%d\n", err);
        if (ctx) clReleaseContext(ctx);
        ctx = NULL;
        queue = NULL;
        return;
    }
}

static void show_help(const char* prog_name) {
    fprintf(stderr, "Unified LZ4 GPU Tool\n");
    fprintf(stderr, "Usage Modes:\n");
    fprintf(stderr, "  1. Standalone:   %s [options] <input_file>\n", prog_name);
    fprintf(stderr, "  2. Run Daemon:   %s --daemon [options]\n", prog_name);
    fprintf(stderr, "  3. Use Daemon:   %s --use-daemon [options] <input_file>\n", prog_name);
    fprintf(stderr, "  4. Stop Daemon:  %s --stop-daemon\n", prog_name);

    fprintf(stderr, "\nBasic Options:\n");
    fprintf(stderr, "  -c                   Compress mode (default)\n");
    fprintf(stderr, "  -d, --decompress     Decompress mode\n");
    fprintf(stderr, "  -o, --output FILE    Output file\n");
    fprintf(stderr, "  -B, --block-size N   Block size in bytes (default: 64KB)\n");
    fprintf(stderr, "  -a, --acceleration N Acceleration factor (default: 1)\n");
    fprintf(stderr, "  --local N            Local work-group size (default: 1)\n");
    fprintf(stderr, "  -v, --verbose        Enable performance statistics\n");
    fprintf(stderr, "  --bench [N]          Stable benchmark (compress+decompress+verify), optional N seconds (default: 3)\n");
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

    cl_program prog = lz4_load_program(ctx, dev);
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
    cl_kernel kpack = clCreateKernel(prog, "lz4_pack_blocks", &err);
    if (err != CL_SUCCESS || !kpack) {
        fprintf(stderr, "bench error: create pack kernel failed (%d)\n", err);
        clReleaseKernel(kdec);
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
    double* comp_total_tp = (double*)malloc(cap * sizeof(double));
    double* dec_tp = (double*)malloc(cap * sizeof(double));
    double* dec_total_tp = (double*)malloc(cap * sizeof(double));
    double* ratio_pct = (double*)malloc(cap * sizeof(double));
    int verify_ok = 1;

    cl_mem d_out = NULL;
    size_t d_out_capacity = 0;

    cl_uint* h_sizes = NULL;
    cl_uint* h_comp_off = NULL;
    cl_uint* h_out_off = NULL;
    cl_uint* h_max_out = NULL;
    size_t h_blocks_capacity = 0;

    cl_uint kernel_num_args_dec = 0;
    int kernel_has_dbg_dec = 0;
    if (clGetKernelInfo(kdec, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args_dec), &kernel_num_args_dec, NULL) == CL_SUCCESS) {
        kernel_has_dbg_dec = (kernel_num_args_dec >= 10U);
    }

    if (!comp_tp || !comp_total_tp || !dec_tp || !dec_total_tp || !ratio_pct) {
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
        ws.decomp_out_off_buf = ensure_buffer(ctx, ws.decomp_out_off_buf, sizeof(cl_uint),
                                              &ws.current_decomp_out_off_capacity, &err);
        if (!ws.decomp_out_off_buf || err != CL_SUCCESS) verify_ok = 0;
    }
    if (verify_ok) {
        ws.decomp_max_out_buf = ensure_buffer(ctx, ws.decomp_max_out_buf, sizeof(cl_uint),
                                              &ws.current_decomp_max_out_capacity, &err);
        if (!ws.decomp_max_out_buf || err != CL_SUCCESS) verify_ok = 0;
    }
    if (verify_ok) {
        ws.decomp_sizes_out_buf = ensure_buffer(ctx, ws.decomp_sizes_out_buf, sizeof(cl_uint),
                                                &ws.current_decomp_sizes_out_capacity, &err);
        if (!ws.decomp_sizes_out_buf || err != CL_SUCCESS) verify_ok = 0;
    }
    if (!verify_ok) {
        fprintf(stderr, "bench error: failed to allocate reusable decomp buffers\n");
    }

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    while (verify_ok) {
        timing_t tc;
        memset(&tc, 0, sizeof(tc));

        uint64_t tcomp0 = get_us();

        int rc = lz4_compress_core(ctx, queue, kcomp, kpack, input_path, "/dev/null",
                                   (size_t)block_size, acceleration, &ws, &tc,
                                   local_size, (n > 0) ? 1 : 0);
        uint64_t tcomp1 = get_us();
        if (rc != 0) {
            verify_ok = 0;
            break;
        }

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

            cl_uint* nh_out_off = (cl_uint*)realloc(h_out_off, nblk * sizeof(cl_uint));
            if (!nh_out_off) {
                verify_ok = 0;
                break;
            }
            h_out_off = nh_out_off;

            cl_uint* nh_max_out = (cl_uint*)realloc(h_max_out, nblk * sizeof(cl_uint));
            if (!nh_max_out) {
                verify_ok = 0;
                break;
            }
            h_max_out = nh_max_out;

            h_blocks_capacity = nblk;
        }
        uint64_t tdec_total0 = get_us();

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
            h_out_off[i] = (cl_uint)(i * blk);
            h_max_out[i] = (cl_uint)((i + 1 == nblk) ? (tc.in_size - i * blk) : blk);
            if (csz > worst_blk) {
                verify_ok = 0;
                break;
            }
        }
        if (!verify_ok) {
            verify_ok = 0;
            break;
        }

        d_out = ensure_buffer(ctx, d_out, (size_t)tc.in_size, &d_out_capacity, &err);
        if (!d_out || err != CL_SUCCESS) {
            verify_ok = 0;
            break;
        }

        ws.decomp_comp_off_buf = ensure_buffer(ctx, ws.decomp_comp_off_buf, nblk * sizeof(cl_uint),
                                               &ws.current_decomp_comp_off_capacity, &err);
        ws.decomp_comp_size_buf = ensure_buffer(ctx, ws.decomp_comp_size_buf, nblk * sizeof(cl_uint),
                                                &ws.current_decomp_comp_size_capacity, &err);
        ws.decomp_out_off_buf = ensure_buffer(ctx, ws.decomp_out_off_buf, nblk * sizeof(cl_uint),
                                              &ws.current_decomp_out_off_capacity, &err);
        ws.decomp_max_out_buf = ensure_buffer(ctx, ws.decomp_max_out_buf, nblk * sizeof(cl_uint),
                                              &ws.current_decomp_max_out_capacity, &err);
        ws.decomp_sizes_out_buf = ensure_buffer(ctx, ws.decomp_sizes_out_buf, nblk * sizeof(cl_uint),
                                                &ws.current_decomp_sizes_out_capacity, &err);

        cl_mem d_comp_off = ws.decomp_comp_off_buf;
        cl_mem d_comp_sz = ws.decomp_comp_size_buf;
        cl_mem d_out_off = ws.decomp_out_off_buf;
        cl_mem d_max_out = ws.decomp_max_out_buf;
        cl_mem d_sizes_out = ws.decomp_sizes_out_buf;
        cl_mem d_dbg_dec = NULL;

        if (!d_comp_off || !d_comp_sz || !d_out_off || !d_max_out || !d_sizes_out || err != CL_SUCCESS) {
            verify_ok = 0;
            break;
        }

        if (kernel_has_dbg_dec) {
            size_t dbg_bytes = nblk * 5U * sizeof(cl_uint);
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

        if (write_buffer_mapped(queue, d_comp_off, h_comp_off, nblk * sizeof(cl_uint)) != 0 ||
            write_buffer_mapped(queue, d_comp_sz, h_sizes, nblk * sizeof(cl_uint)) != 0 ||
            write_buffer_mapped(queue, d_out_off, h_out_off, nblk * sizeof(cl_uint)) != 0 ||
            write_buffer_mapped(queue, d_max_out, h_max_out, nblk * sizeof(cl_uint)) != 0) {
            if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
            verify_ok = 0;
            break;
        }

        cl_uint totalBlocks = (cl_uint)nblk;
        err  = clSetKernelArg(kdec, 0, sizeof(cl_mem), &ws.out_buf);
        err |= clSetKernelArg(kdec, 1, sizeof(cl_mem), &d_out);
        err |= clSetKernelArg(kdec, 2, sizeof(cl_mem), &d_comp_off);
        err |= clSetKernelArg(kdec, 3, sizeof(cl_mem), &d_comp_sz);
        err |= clSetKernelArg(kdec, 4, sizeof(cl_mem), &d_out_off);
        err |= clSetKernelArg(kdec, 5, sizeof(cl_mem), &d_max_out);
        err |= clSetKernelArg(kdec, 6, sizeof(cl_mem), &d_sizes_out);
        err |= clSetKernelArg(kdec, 7, sizeof(cl_uint), &totalBlocks);
        if (kernel_has_dbg_dec) {
            cl_mem dbg_arg = d_dbg_dec ? d_dbg_dec : d_sizes_out;
            cl_uint dbg_flag = d_dbg_dec ? 1U : 0U;
            err |= clSetKernelArg(kdec, 8, sizeof(cl_mem), &dbg_arg);
            err |= clSetKernelArg(kdec, 9, sizeof(cl_uint), &dbg_flag);
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

        uint64_t td0 = get_us();
        cl_event dec_kernel_evt = NULL;
        err = clEnqueueNDRangeKernel(queue, kdec, 1, NULL, &gsz, &lsz, 0, NULL, &dec_kernel_evt);
        if (err == CL_SUCCESS && dec_kernel_evt) {
            clWaitForEvents(1, &dec_kernel_evt);
            clReleaseEvent(dec_kernel_evt);
            dec_kernel_evt = NULL;
        }
        uint64_t td1 = get_us();
        if (err != CL_SUCCESS) {
            if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
            verify_ok = 0;
            break;
        }

        cl_uint* map_sizes_out = (cl_uint*)clEnqueueMapBuffer(queue, d_sizes_out, CL_TRUE, CL_MAP_READ,
                                                              0, nblk * sizeof(cl_uint), 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !map_sizes_out) {
            if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
            verify_ok = 0;
            break;
        }

        size_t out_total = 0;
        for (size_t i = 0; i < nblk; ++i) {
            if (map_sizes_out[i] == 0xFFFFFFFFU) {
                verify_ok = 0;
                break;
            }
            out_total += (size_t)map_sizes_out[i];
        }
        clEnqueueUnmapMemObject(queue, d_sizes_out, map_sizes_out, 0, NULL, NULL);

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

        uint64_t tdec_total1 = get_us();

        if (n == cap) {
            size_t new_cap = cap * 2;
            double* nc = (double*)realloc(comp_tp, new_cap * sizeof(double));
            if (!nc) {
                verify_ok = 0;
                break;
            }
            comp_tp = nc;

            double* nct = (double*)realloc(comp_total_tp, new_cap * sizeof(double));
            if (!nct) {
                verify_ok = 0;
                break;
            }
            comp_total_tp = nct;

            double* nd = (double*)realloc(dec_tp, new_cap * sizeof(double));
            if (!nd) {
                verify_ok = 0;
                break;
            }
            dec_tp = nd;

            double* ndt = (double*)realloc(dec_total_tp, new_cap * sizeof(double));
            if (!ndt) {
                verify_ok = 0;
                break;
            }
            dec_total_tp = ndt;

            double* nr = (double*)realloc(ratio_pct, new_cap * sizeof(double));
            if (!nr) {
                verify_ok = 0;
                break;
            }
            ratio_pct = nr;
            cap = new_cap;
        }

        double in_mb = (double)tc.in_size / (1024.0 * 1024.0);
        comp_tp[n] = (tc.kernel_exec_us > 0) ? (in_mb * 1000000.0 / (double)tc.kernel_exec_us) : 0.0;
        /* Exclude host file I/O from total throughput (keep device transfer + compute path). */
        double comp_total_us = (double)(tcomp1 - tcomp0);
        {
            double io_read_us = (double)tc.file_read_us;
            double io_write_us = (double)tc.file_write_us;
            if (io_write_us > (double)tc.download_total_us) {
                io_write_us -= (double)tc.download_total_us;
            } else {
                io_write_us = 0.0;
            }
            if (comp_total_us > io_read_us + io_write_us) {
                comp_total_us -= (io_read_us + io_write_us);
            } else {
                comp_total_us = 0.0;
            }
        }
        comp_total_tp[n] = (comp_total_us > 0.0) ? (in_mb * 1000000.0 / comp_total_us) : 0.0;
        double dec_kernel_us = (double)(td1 - td0);
        dec_tp[n] = (dec_kernel_us > 0.0) ? (in_mb * 1000000.0 / dec_kernel_us) : 0.0;
        double dec_total_us = (double)(tdec_total1 - tdec_total0);
        dec_total_tp[n] = (dec_total_us > 0.0) ? (in_mb * 1000000.0 / dec_total_us) : 0.0;
        ratio_pct[n] = (tc.in_size > 0) ? (100.0 * (double)tc.out_size / (double)tc.in_size) : 0.0;
        n++;

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (elapsed_sec(&ts0, &ts1) >= bench_seconds && n > 0) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double sec = elapsed_sec(&ts0, &ts1);

    if (n > 0) {
        printf("Bench Compress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s ratio=%.2f%%\n",
               median_double(comp_tp, n), median_double(comp_total_tp, n), median_double(ratio_pct, n));
        printf("Bench Decompress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s verify=%s\n",
               median_double(dec_tp, n), median_double(dec_total_tp, n), verify_ok ? "OK" : "FAIL");
        printf("Bench Summary : iterations=%zu seconds=%.2f\n", n, sec);
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_tp);
    free(comp_total_tp);
    free(dec_tp);
    free(dec_total_tp);
    free(ratio_pct);
    free(h_sizes);
    free(h_comp_off);
    free(h_out_off);
    free(h_max_out);
    if (d_out) clReleaseMemObject(d_out);
    free(input_ref);
    lz4_gpu_workspace_free(&ws);
    clReleaseKernel(kpack);
    clReleaseKernel(kdec);
    clReleaseKernel(kcomp);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return verify_ok ? 0 : 1;
}

int run_lz4_standalone(int argc, char** argv) {
    int mode = mode_compress;
    int bench_mode = 0;
    double bench_seconds = 3.0;
    const char* input_path = NULL;
    char output_path[512] = {0};
    int output_explicit = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) {
            mode = mode_decompress;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_verbose = 1;
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
        } else if (strcmp(argv[i], "--local") == 0) {
            if (i + 1 < argc) g_cli_local_size = atoi(argv[++i]);
            else { fprintf(stderr, "Error: --local requires an argument\n"); return 1; }
        } else if (argv[i][0] == '-') {
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

    if (bench_mode) {
        if (mode != mode_compress) {
            fprintf(stderr, "Error: --bench only supports compress mode input (it runs compress+decompress internally)\n");
            return 1;
        }
        return run_lz4_bench(input_path,
                            (int)g_cli_fixed_block_bytes,
                            g_cli_acceleration,
                            (int)g_cli_local_size,
                            bench_seconds);
    }

    if (!output_explicit) {
        if (mode == mode_compress) snprintf(output_path, sizeof(output_path), "%s.lz4", input_path);
        else snprintf(output_path, sizeof(output_path), "%s.dec", input_path);
    }

    uint64_t t_total_start = get_us();
    uint64_t t1, t2;

    t1 = get_us();
    ocl_init();
    if (!ctx || !queue) {
        fprintf(stderr, "Failed to initialize OpenCL runtime\n");
        return 1;
    }
    t2 = get_us();
    g_ocl_init_us = t2 - t1;

    t1 = get_us();
    cl_program prog = lz4_load_program(ctx, dev);
    t2 = get_us();
    g_kernel_load_us = t2 - t1;

    if (!prog) { fprintf(stderr, "Failed to load OCL program\n"); return 1; }

    cl_kernel kernel;
    cl_kernel pack_kernel = NULL;
    cl_int err;
    if (mode == mode_compress) {
        kernel = clCreateKernel(prog, "lz4_compress_block", &err);
        if (err == CL_SUCCESS && kernel) {
            pack_kernel = clCreateKernel(prog, "lz4_pack_blocks", &err);
        }
    } else {
        kernel = clCreateKernel(prog, "lz4_decompress_blocks", &err);
    }

    if (err != CL_SUCCESS || !kernel) {
        fprintf(stderr, "Failed to create kernel: %d\n", err);
        return 1;
    }

    lz4_gpu_workspace_t ws;
    lz4_gpu_workspace_init(&ws);
    timing_t t_out;
    memset(&t_out, 0, sizeof(t_out));

    int ret = -1;
    if (mode == mode_compress) {
        ret = lz4_compress_core(ctx, queue, kernel, pack_kernel, input_path, output_path, (int)g_cli_fixed_block_bytes, g_cli_acceleration, &ws, &t_out, (int)g_cli_local_size, 0);
    } else {
        ret = lz4_decompress_core(ctx, queue, kernel, input_path, output_path, &ws, &t_out, (int)g_cli_local_size);
    }

    uint64_t t_total_end = get_us();

    if (ret == 0) {
        t_out.ocl_setup_us = (unsigned long)(g_ocl_init_us + g_kernel_load_us);
        response_t resp;
        memset(&resp, 0, sizeof(resp));
        resp.status = 0;
        resp.out_size = t_out.out_size;
        resp.timing = t_out;
        resp.time_us = (unsigned long)(t_total_end - t_total_start);
        if (g_verbose) {
            print_response_stats(&resp, input_path, mode);
        } else {
            double ratio = (double)t_out.in_size / (t_out.out_size > 0 ? t_out.out_size : 1);
            printf("%s : %zu -> %zu (%.2f:1) in %.2f ms\n", input_path, (size_t)t_out.in_size, (size_t)resp.out_size, ratio, resp.time_us / 1000.0);
        }
    }

    lz4_gpu_workspace_free(&ws);
    if (pack_kernel) clReleaseKernel(pack_kernel);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return ret;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        show_help(argv[0]);
        return 0;
    }
    if (argc >= 2) {
        if (strcmp(argv[1], "--daemon") == 0) return run_daemon();
        if (strcmp(argv[1], "--stop-daemon") == 0) return stop_daemon_cmd();
        if (strcmp(argv[1], "--use-daemon") == 0) {
            int mode = mode_compress;
            int bench_mode = 0;
            double bench_seconds = 3.0;
            const char* input = NULL;
            char output[512] = {0};
            int output_explicit = 0;
            for (int i = 2; i < argc; i++) {
                if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                    show_help(argv[0]);
                    return 0;
                }
                else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) mode = mode_decompress;
                else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) g_verbose = 1;
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
                } else if ((strcmp(argv[i], "--local") == 0) && i + 1 < argc) {
                    g_cli_local_size = atoi(argv[++i]);
                } else if (argv[i][0] == '-') {
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
            if (!input) {
                fprintf(stderr, "Error: No input file specified\n");
                return 1;
            }
            if (bench_mode) {
                if (mode != mode_compress) {
                    fprintf(stderr, "Error: --bench only supports compress mode input (it runs compress+decompress internally)\n");
                    return 1;
                }
                return run_lz4_bench(input,
                                    (int)g_cli_fixed_block_bytes,
                                    g_cli_acceleration,
                                    (int)g_cli_local_size,
                                    bench_seconds);
            }
            if (!output_explicit) {
                if (mode == mode_compress) snprintf(output, sizeof(output), "%s.lz4", input);
                else snprintf(output, sizeof(output), "%s.dec", input);
            }
            return run_lz4_client(mode, input, output, (int)g_cli_fixed_block_bytes, g_cli_acceleration, (int)g_cli_local_size);
        }
    }
    return run_lz4_standalone(argc, argv);
}
