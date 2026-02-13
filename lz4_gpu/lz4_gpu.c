#define _POSIX_C_SOURCE 200809L
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <errno.h>
#include "lz4_gpu_core.h"
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_utils.h"
#include "timing.h"

/* Forward declarations */
int run_daemon();
int run_lz4_client(int mode, const char* input_path, const char* output_path, int block_size, int acceleration, int local_size, int hash_log);

int g_verbose = 0;
static size_t g_cli_local_size = 1;
static size_t g_cli_fixed_block_bytes = 64 * 1024;
static int g_cli_acceleration = 1;
static int g_cli_hash_log = 14;

static cl_context ctx;
static cl_command_queue queue;
static cl_device_id dev;

static void ocl_init() {
    cl_platform_id pf;
    clGetPlatformIDs(1, &pf, NULL);
    clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
    queue = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, NULL);
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
    fprintf(stderr, "  -b, --block-size N   Block size in bytes (default: 64KB)\n");
    fprintf(stderr, "  -a, --acceleration N Acceleration factor (default: 1)\n");
    fprintf(stderr, "  -H, --hash N         Hash log bits (default: 14)\n");
    fprintf(stderr, "  -l, --local N        Local work-group size (default: 1)\n");
    fprintf(stderr, "  -v, --verbose        Enable performance statistics\n");
}

static int stop_daemon_cmd() {
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
}

static size_t parse_size_bytes(const char* s) {
    char* endptr;
    size_t val = strtoul(s, &endptr, 10);
    if (*endptr == 'k' || *endptr == 'K') val *= 1024;
    else if (*endptr == 'm' || *endptr == 'M') val *= 1024 * 1024;
    return val;
}

int run_lz4_standalone(int argc, char** argv) {
    int mode = mode_compress;
    const char* input_path = NULL;
    char output_path[512] = {0};
    int output_explicit = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) {
            mode = mode_decompress;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_verbose = 1;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                strncpy(output_path, argv[++i], sizeof(output_path) - 1);
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: -o requires an argument\n");
                show_help(argv[0]); return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) {
            if (i + 1 < argc) g_cli_fixed_block_bytes = parse_size_bytes(argv[++i]);
            else { fprintf(stderr, "Error: -b requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--acceleration") == 0) {
            if (i + 1 < argc) g_cli_acceleration = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -a requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--hash") == 0) {
            if (i + 1 < argc) g_cli_hash_log = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -H requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--local") == 0) {
            if (i + 1 < argc) g_cli_local_size = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -l requires an argument\n"); return 1; }
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

    if (!output_explicit) {
        if (mode == mode_compress) snprintf(output_path, sizeof(output_path), "%s.lz4", input_path);
        else snprintf(output_path, sizeof(output_path), "%s.dec", input_path);
    }

    uint64_t t_total_start = get_us();
    uint64_t t1, t2;

    t1 = get_us();
    ocl_init();
    t2 = get_us();
    g_ocl_init_us = t2 - t1;

    t1 = get_us();
    cl_program prog = lz4_load_program(ctx, dev, g_cli_hash_log);
    t2 = get_us();
    g_kernel_load_us = t2 - t1;

    if (!prog) { fprintf(stderr, "Failed to load OCL program\n"); return 1; }

    cl_kernel kernel;
    cl_int err;
    if (mode == mode_compress) {
        kernel = clCreateKernel(prog, "lz4_compress_block", &err);
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
        ret = lz4_compress_core(ctx, queue, kernel, input_path, output_path, (int)g_cli_fixed_block_bytes, g_cli_acceleration, &ws, &t_out, (int)g_cli_local_size, g_cli_hash_log);
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
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return ret;
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        if (strcmp(argv[1], "--daemon") == 0) return run_daemon();
        if (strcmp(argv[1], "--stop-daemon") == 0) return stop_daemon_cmd();
        if (strcmp(argv[1], "--use-daemon") == 0) {
            int mode = mode_compress;
            const char* input = NULL;
            char output[512] = {0};
            int output_explicit = 0;
            for (int i = 2; i < argc; i++) {
                if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) mode = mode_decompress;
                else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) g_verbose = 1;
                else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                    strncpy(output, argv[++i], sizeof(output)-1);
                    output_explicit = 1;
                } else if ((strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) && i + 1 < argc) {
                    g_cli_fixed_block_bytes = parse_size_bytes(argv[++i]);
                } else if ((strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--acceleration") == 0) && i + 1 < argc) {
                    g_cli_acceleration = atoi(argv[++i]);
                } else if ((strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--hash") == 0) && i + 1 < argc) {
                    g_cli_hash_log = atoi(argv[++i]);
                } else if ((strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--local") == 0) && i + 1 < argc) {
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
            if (!output_explicit) {
                if (mode == mode_compress) snprintf(output, sizeof(output), "%s.lz4", input);
                else snprintf(output, sizeof(output), "%s.dec", input);
            }
            return run_lz4_client(mode, input, output, (int)g_cli_fixed_block_bytes, g_cli_acceleration, (int)g_cli_local_size, g_cli_hash_log);
        }
    }
    return run_lz4_standalone(argc, argv);
}
