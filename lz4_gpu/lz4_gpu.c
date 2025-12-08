/*
 * lz4_gpu.c - CLI wrapper for GPU-accelerated LZ4
 * Usage: lz4_gpu -c|-d [-l level] [-o out] [--bench] [-v] <input>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lz4_gpu_host.h"
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <math.h>

#define DEFAULT_ACCELERATION LZ4_GPU_DEFAULT_ACCELERATION

static int ends_with(const char* str, const char* suf) {
    if (!str || !suf) return 0;
    size_t lstr = strlen(str);
    size_t lsuf = strlen(suf);
    if (lsuf > lstr) return 0;
    return strcmp(str + lstr - lsuf, suf) == 0;
}

// Helper functions for daemon IPC
static ssize_t read_all(int fd, void* buf, size_t count) {
    size_t offset = 0;
    while (offset < count) {
        ssize_t r = read(fd, (char*)buf + offset, count - offset);
        if (r <= 0) return r; // zero or negative indicates EOF or error
        offset += (size_t)r;
    }
    return (ssize_t)offset;
}

static ssize_t write_all(int fd, const void* buf, size_t count) {
    size_t offset = 0;
    while (offset < count) {
        ssize_t r = write(fd, (const char*)buf + offset, count - offset);
        if (r <= 0) return r;
        offset += (size_t)r;
    }
    return (ssize_t)offset;
}

// Run the daemon server loop - this blocks until terminated by signal
static volatile int daemon_should_stop = 0;
static void daemon_signal_handler(int sig) {
    (void)sig;
    daemon_should_stop = 1;
}

// Daemon context to manage persistent buffers and statistics
typedef struct {
    unsigned char* input_buf;
    size_t input_capacity;
    unsigned char* output_buf;
    size_t output_capacity;
    uint64_t total_requests;
    uint64_t total_compress_bytes;
    uint64_t total_decompress_bytes;
    double total_compress_time_ms;
    double total_decompress_time_ms;
} DaemonContext;

static int run_daemon_server(LZ4GPUCompressor* ctx, const char* socket_path, int enable_pinned) {
    int listen_fd = -1;
    struct sockaddr_un addr;
    if (!socket_path) socket_path = "/tmp/lz4_gpu_daemon.sock";

    // If pinned requested, enable pinned memory before init
    if (enable_pinned >= 0) {
        lz4_gpu_set_pinned_memory(ctx, enable_pinned);
    } else {
        // Daemon default: prefer pinned memory for performance
        lz4_gpu_set_pinned_memory(ctx, 1);
    }

    if (!lz4_gpu_initialize(ctx)) {
        fprintf(stderr, "Daemon: GPU init failed: %s\n", lz4_gpu_get_error_message(ctx));
        return 1;
    }

    // Initialize daemon context with modest initial buffers
    DaemonContext dctx = {0};
    dctx.input_capacity = 64 * 1024 * 1024;  // 64MB initial
    dctx.output_capacity = 128 * 1024 * 1024; // 128MB initial
    dctx.input_buf = (unsigned char*)malloc(dctx.input_capacity);
    dctx.output_buf = (unsigned char*)malloc(dctx.output_capacity);
    if (!dctx.input_buf || !dctx.output_buf) {
        fprintf(stderr, "Daemon: failed to allocate persistent buffers\n");
        free(dctx.input_buf);
        free(dctx.output_buf);
        return 1;
    }

    // Optional micro-benchmark to prefer smaller acceleration if similar performance
    const char* tune_env = getenv("LZ4_GPU_AUTO_TUNE_ACCEL");
    if (tune_env && tune_env[0] == '1') {
        // Run a tiny benchmark comparing accel=1 and accel=4 to prefer smaller if within 3%
        size_t sample_size = 1024 * 1024; // 1MB
        unsigned char* sbuf = (unsigned char*)malloc(sample_size);
        unsigned char* outbuf = (unsigned char*)malloc(sample_size + sample_size/10 + 65536);
        if (sbuf && outbuf) {
            // Fill with pseudo-random data
            for (size_t i = 0; i < sample_size; ++i) sbuf[i] = (unsigned char)(i & 0xFF);
            int reps = 3;
            double best1 = 0.0, best4 = 0.0;
            for (int r = 0; r < reps; ++r) {
                clock_t t0 = clock();
                size_t sz = lz4_gpu_compress_frame_accelerated(ctx, sbuf, sample_size, outbuf, sample_size + sample_size/10, 1);
                clock_t t1 = clock();
                if (sz > 0) best1 += (double)(t1 - t0) / CLOCKS_PER_SEC;
                t0 = clock();
                sz = lz4_gpu_compress_frame_accelerated(ctx, sbuf, sample_size, outbuf, sample_size + sample_size/10, 4);
                t1 = clock();
                if (sz > 0) best4 += (double)(t1 - t0) / CLOCKS_PER_SEC;
            }
            if (best1 > 0 && best4 > 0) {
                double avg1 = best1 / (double)reps;
                double avg4 = best4 / (double)reps;
                double diff = fabs(avg1 - avg4) / ((avg1+avg4) * 0.5);
                if (diff < 0.03) {
                    // Similar performance: prefer smaller acceleration
                    ctx->default_acceleration = 1;
                    fprintf(stderr, "Daemon: micro-benchmark: accel 1 and 4 similar; preferring smaller accel=1\n");
                }
            }
        }
        if (sbuf) free(sbuf);
        if (outbuf) free(outbuf);
    }

    if ((listen_fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
        perror("daemon: socket");
        return 1;
    }

    memset(&addr, 0, sizeof(struct sockaddr_un));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    // If another socket file exists, try to connect to verify if a daemon is running.
    if (access(socket_path, F_OK) == 0) {
        int tmpfd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (tmpfd >= 0) {
            if (connect(tmpfd, (struct sockaddr*)&addr, sizeof(struct sockaddr_un)) == 0) {
                fprintf(stderr, "Daemon: socket %s already in use by a running daemon\n", socket_path);
                close(tmpfd);
                close(listen_fd);
                return 1;
            }
            close(tmpfd);
        }
        unlink(socket_path); // not running: remove stale socket
    }

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(struct sockaddr_un)) < 0) {
        perror("daemon: bind");
        close(listen_fd);
        return 1;
    }
    if (listen(listen_fd, 4) < 0) {
        perror("daemon: listen");
        close(listen_fd);
        unlink(socket_path);
        return 1;
    }
    // Set socket perms to allow local users connecting
    chmod(socket_path, 0666);

    fprintf(stderr, "Daemon: Listening on %s\n", socket_path);

    // Install signal handler for graceful shutdown
    signal(SIGINT, daemon_signal_handler);
    signal(SIGTERM, daemon_signal_handler);

    while (!daemon_should_stop) {
        int client_fd = accept(listen_fd, NULL, NULL);
        if (client_fd < 0) {
            if (errno == EINTR) continue; // interrupted by signal, try again
            perror("daemon: accept");
            break;
        }
        dctx.total_requests++;

        // Protocol: header = 16 bytes: op(1) + accel(1) + local(1) + pinned(1) + input_size(8) + block_size(4)
        uint8_t header[16];
        if (read_all(client_fd, header, sizeof(header)) != sizeof(header)) {
            fprintf(stderr, "Daemon: failed to read header from client\n");
            close(client_fd);
            continue;
        }
        uint8_t op = header[0];
        int requested_accel = (int)header[1];
        int requested_local = (int)header[2];
        int requested_pinned = (int)header[3];
        uint64_t input_size = 0;
        memcpy(&input_size, &header[4], sizeof(uint64_t));
        uint32_t requested_block_size = 0;
        memcpy(&requested_block_size, &header[12], sizeof(uint32_t));

        // Expand persistent input buffer if needed
        if ((size_t)input_size > dctx.input_capacity) {
            dctx.input_capacity = (size_t)input_size + (size_t)input_size / 5; // 20% headroom
            unsigned char* new_buf = (unsigned char*)realloc(dctx.input_buf, dctx.input_capacity);
            if (!new_buf) {
                fprintf(stderr, "Daemon: failed to expand input buffer to %zu bytes\n", dctx.input_capacity);
                close(client_fd);
                continue;
            }
            dctx.input_buf = new_buf;
        }

        // Read input into persistent buffer
        if (read_all(client_fd, dctx.input_buf, (size_t)input_size) != (ssize_t)input_size) {
            fprintf(stderr, "Daemon: failed to read input payload\n");
            close(client_fd);
            continue;
        }
        if (requested_pinned <= 1) lz4_gpu_set_pinned_memory(ctx, requested_pinned);
        if (requested_local > 0) lz4_gpu_set_workgroup_size(ctx, (size_t)requested_local);
        if (requested_block_size > 0) lz4_gpu_set_block_sizes(ctx, (size_t)requested_block_size, (size_t)requested_block_size);

        // Estimate output buffer size based on operation
        size_t needed_size = dctx.output_capacity;
        if (op == 1) { // compress
            needed_size = (size_t)input_size + (size_t)input_size/10 + 65536;
        } else if (op == 2) { // decompress - estimate from frame
            size_t estimated = lz4_gpu_estimate_decompressed_size(ctx, dctx.input_buf, (size_t)input_size);
            if (estimated > 0) {
                needed_size = estimated + 65536;
            } else {
                // Fallback: conservative 10x multiplier
                needed_size = (size_t)input_size * 10 + 65536;
            }
        }

        // Expand output buffer if needed with some headroom
        if (needed_size > dctx.output_capacity) {
            dctx.output_capacity = needed_size + (size_t)(needed_size * 0.1); // 10% headroom
            unsigned char* new_buf = (unsigned char*)realloc(dctx.output_buf, dctx.output_capacity);
            if (!new_buf) {
                fprintf(stderr, "Daemon: failed to expand output buffer to %zu bytes\n", dctx.output_capacity);
                close(client_fd);
                continue;
            }
            dctx.output_buf = new_buf;
        }

        // Clear init_ms from the context's last_timing since daemon init is one-time only.
        // For each request, we should not include the one-time OpenCL setup time.
        ctx->last_timing.init_ms = 0.0;

        // Time the operation
        clock_t t_op_start = clock();
        size_t out_sz = 0;
        if (op == 1) { // compress
            if (requested_accel <= 0) requested_accel = DEFAULT_ACCELERATION;
            out_sz = lz4_gpu_compress_frame_accelerated(ctx, dctx.input_buf, (size_t)input_size, dctx.output_buf, dctx.output_capacity, requested_accel);
            if (out_sz > 0) {
                dctx.total_compress_bytes += input_size;
            }
        } else if (op == 2) { // decompress
            out_sz = lz4_gpu_decompress_frame(ctx, dctx.input_buf, (size_t)input_size, dctx.output_buf, dctx.output_capacity);
            if (out_sz > 0) {
                dctx.total_decompress_bytes += out_sz;
            }
        } else {
            fprintf(stderr, "Daemon: unknown op %d\n", op);
        }
        clock_t t_op_end = clock();
        double op_time_ms = (double)(t_op_end - t_op_start) / CLOCKS_PER_SEC * 1000.0;
        if (op == 1) {
            dctx.total_compress_time_ms += op_time_ms;
        } else if (op == 2) {
            dctx.total_decompress_time_ms += op_time_ms;
        }

        // Send result
        if (out_sz == 0) {
            uint8_t status = 1;
            write_all(client_fd, &status, 1);
            const char* err = lz4_gpu_get_error_message(ctx);
            uint32_t len = (uint32_t)strlen(err);
            uint32_t len_le = len;
            write_all(client_fd, &len_le, sizeof(len_le));
            write_all(client_fd, err, len);
        } else {
            uint8_t status = 0;
            write_all(client_fd, &status, 1);
            uint64_t sz_le = (uint64_t)out_sz;
            write_all(client_fd, &sz_le, sizeof(sz_le));
            write_all(client_fd, dctx.output_buf, out_sz);

            // Send timing data for detailed statistics
            LZ4GPUTiming timing;
            memset(&timing, 0, sizeof(timing));
            if (lz4_gpu_get_last_timing(ctx, &timing)) {
                write_all(client_fd, &timing, sizeof(timing));
            } else {
                // Send empty timing if unavailable
                write_all(client_fd, &timing, sizeof(timing));
            }
        }
        close(client_fd);
        // continue accepting new clients
    }

    // Print daemon statistics on shutdown
    fprintf(stderr, "Daemon: Shutting down. Statistics:\n");
    fprintf(stderr, "  Total requests: %llu\n", (unsigned long long)dctx.total_requests);
    if (dctx.total_compress_time_ms > 0 && dctx.total_compress_bytes > 0) {
        double comp_mb = (double)dctx.total_compress_bytes / (1024.0 * 1024.0);
        double comp_sec = dctx.total_compress_time_ms / 1000.0;
        fprintf(stderr, "  Compress: %.2f MB, %.2f ms, %.2f MB/s\n", comp_mb, dctx.total_compress_time_ms, comp_mb / comp_sec);
    }
    if (dctx.total_decompress_time_ms > 0 && dctx.total_decompress_bytes > 0) {
        double decomp_mb = (double)dctx.total_decompress_bytes / (1024.0 * 1024.0);
        double decomp_sec = dctx.total_decompress_time_ms / 1000.0;
        fprintf(stderr, "  Decompress: %.2f MB, %.2f ms, %.2f MB/s\n", decomp_mb, dctx.total_decompress_time_ms, decomp_mb / decomp_sec);
    }
    fprintf(stderr, "  Final buffer capacities - input: %zu bytes, output: %zu bytes\n", dctx.input_capacity, dctx.output_capacity);

    close(listen_fd);
    free(dctx.input_buf);
    free(dctx.output_buf);
    unlink(socket_path);
    return 0;
}

static int run_daemon_client(const char* socket_path, const unsigned char* input_buf, size_t input_size,
                             unsigned char** out_buf, size_t* out_size,
                             int compress, int accel, int local, int pinned, size_t block_size, double* elapsed_ms, LZ4GPUTiming* timing_out) {
    if (!socket_path) socket_path = "/tmp/lz4_gpu_daemon.sock";
    clock_t t_start = clock();
    int sd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sd < 0) { perror("daemon client: socket"); return 1; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(struct sockaddr_un));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    if (connect(sd, (struct sockaddr*)&addr, sizeof(struct sockaddr_un)) < 0) {
        perror("daemon client: connect");
        close(sd);
        return 2; // unable to connect
    }
    // Protocol: 16 byte header = op(1) + accel(1) + local(1) + pinned(1) + input_size(8) + block_size(4)
    uint8_t header[16];
    header[0] = compress ? 1 : 2; // op
    header[1] = (uint8_t)(accel & 0xff);
    header[2] = (uint8_t)(local & 0xff);
    header[3] = (uint8_t)((pinned >= 0) ? pinned : 255);
    uint64_t in_sz_le = (uint64_t)input_size;
    memcpy(&header[4], &in_sz_le, sizeof(in_sz_le));
    uint32_t block_sz_le = (uint32_t)(block_size & 0xffffffff);
    memcpy(&header[12], &block_sz_le, sizeof(block_sz_le));
    if (write_all(sd, header, sizeof(header)) != sizeof(header)) { close(sd); return 3; }
    if (write_all(sd, input_buf, input_size) != (ssize_t)input_size) { close(sd); return 4; }
    uint8_t status = 1;
    if (read_all(sd, &status, 1) != 1) { close(sd); return 5; }
    if (status != 0) {
        uint32_t msglen = 0;
        if (read_all(sd, &msglen, sizeof(msglen)) != sizeof(msglen)) { close(sd); return 6; }
        char* msg = (char*)malloc((size_t)msglen + 1);
        if (!msg) { close(sd); return 7; }
        if (read_all(sd, msg, msglen) != (ssize_t)msglen) { free(msg); close(sd); return 8; }
        msg[msglen] = '\0';
        fprintf(stderr, "Daemon error: %s\n", msg);
        free(msg);
        close(sd);
        return 9;
    }
    uint64_t out_sz = 0;
    if (read_all(sd, &out_sz, sizeof(out_sz)) != sizeof(out_sz)) { close(sd); return 10; }
    unsigned char* out = (unsigned char*)malloc((size_t)out_sz);
    if (!out) { close(sd); return 11; }
    if (read_all(sd, out, (size_t)out_sz) != (ssize_t)out_sz) { free(out); close(sd); return 12; }
    *out_buf = out; *out_size = (size_t)out_sz;

    // Read timing data from daemon
    LZ4GPUTiming timing;
    memset(&timing, 0, sizeof(timing));
    if (read_all(sd, &timing, sizeof(timing)) == sizeof(timing)) {
        if (timing_out) {
            memcpy(timing_out, &timing, sizeof(timing));
        }
    }

    clock_t t_end = clock();
    if (elapsed_ms) *elapsed_ms = ((double)(t_end - t_start) * 1000.0) / CLOCKS_PER_SEC;
    close(sd);
    return 0;
}

/* Try to parse LZ4 frame header and extract the original content size if present.
 * Returns 1 and sets *out_size on success, 0 if not present or cannot parse.
 */
static int parse_lz4f_content_size(const unsigned char* buf, size_t len, size_t* out_size) {
    if (!buf || len < 7) return 0; /* need at least magic + FLG + BD */
    /* magic is little-endian 32-bit */
    unsigned int magic = (unsigned int)buf[0] | ((unsigned int)buf[1] << 8) | ((unsigned int)buf[2] << 16) | ((unsigned int)buf[3] << 24);
    if (magic != LZ4F_MAGICNUMBER) return 0;
    unsigned char FLG = buf[4];
    /* content size flag is bit 3 of FLG (0x08) */
    if ((FLG & 0x08) == 0) return 0;
    /* content size field (8 bytes LE) starts at offset 6 (after magic(4) + FLG(1) + BD(1)) */
    if (len < 4 + 1 + 1 + 8) return 0; /* not enough bytes available in provided buffer */
    const unsigned char* p = buf + 6;
    unsigned long long cs = 0;
    for (int i = 0; i < 8; ++i) cs |= ((unsigned long long)p[i]) << (8 * i);
    *out_size = (size_t)cs;
    return 1;
}

static void usage(const char* prog) {
    fprintf(stderr,
    "Usage: %s [-c|-d] [-l level] [-B blocksize] [-o outfile|-] [--bench] [-v] <input>\n"
        "  -c        compress (default)\n"
        "  -d        decompress\n"
        "  -l LEVEL  compression acceleration level (higher -> faster, less ratio). Allowed: 1..12\n"
        "  -B|--blocksize SIZE  block size for compression (e.g. 16k, 64k, 256k). Default: 64k\n"
    "  -o FILE   output file (default: input.lz4 for compress, input.out for decompress). Use '-' to write to stdout.\n"
        "  -g|--kernel-debug Enable kernel-side debug prints (build with LZ4_GPU_KERNEL_DEBUG)\n"
    "  -p|--profile  Enable OpenCL event profiling and print upload/kernel/download CSV\n"
        "  --local N  Optional override for local work-group size (applies to both compression and decompression kernels)\n"
    "  --bench   print throughput and compression ratio summary\n"
        "  -v        verbose logging\n"
        "  --pinned  Use pinned host memory (enabled if supported).\n"
        "  --no-pinned  Disable pinned host memory.\n"
        "  --daemon  Run as a persistent lz4_gpu daemon process (accepts IPC requests via unix domain socket)\n"
        "  --use-daemon  Send a compress/decompress request to a running lz4_gpu daemon if one is available\n"
        "  --daemon-socket PATH  Unix domain socket path for daemon (default: /tmp/lz4_gpu_daemon.sock)\n"
         "ENVIRONMENT VARIABLES:\n"
         "  LZ4_GPU_CLBIN : Optional path to precompiled .clbin file (preferred).\n"
         "  LZ4_GPU_CLSRC : Optional path to kernel source file to use when building from source.\n",
         prog);
}

/* parse size string like 16k, 64K, 1m into bytes */
static size_t parse_size_arg(const char* s) {
    if (!s) return 0;
    char* endptr;
    long long v = strtoll(s, &endptr, 10);
    if (v <= 0) return 0;
    if (*endptr == 'k' || *endptr == 'K') v *= 1024LL;
    else if (*endptr == 'm' || *endptr == 'M') v *= 1024LL * 1024LL;
    return (size_t)v;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    int compress = 1;
    int accel = DEFAULT_ACCELERATION;
    const char* outpath = NULL;
    int bench = 0; /* default: bench disabled; enable only when --bench passed */
    int verbose = 0;
    int kernel_debug = 0;
    int host_debug = 0;
    int enable_profile = 0;
    int cli_pinned = -1;
    const char* infile = NULL;
    size_t cli_blocksize = 0;
    int cli_local = 0;
    const char* env_clbin_path = NULL;
    const char* env_clsrc_path = NULL;
    int daemon_mode = 0;
    int use_daemon = 0;
    const char* daemon_socket = NULL;
    // int daemon_pinned = -1; /* 移除单独的daemon_pinned参数 */

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0) { compress = 1; }
        else if (strcmp(argv[i], "-d") == 0) { compress = 0; }
        else if (strcmp(argv[i], "-v") == 0) { verbose = 1; }
        else if (strcmp(argv[i], "--bench") == 0) { bench = 1; }
        else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--kernel-debug") == 0) { kernel_debug = 1; }
        else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--host-debug") == 0) { host_debug = 1; }
        else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--profile") == 0) { enable_profile = 1; }
        else if (strcmp(argv[i], "-l") == 0 && i+1 < argc) { accel = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-B") == 0 || strcmp(argv[i], "--blocksize") == 0) && i+1 < argc) { cli_blocksize = parse_size_arg(argv[++i]); }
        else if (strncmp(argv[i], "--local=", 8) == 0) { cli_local = atoi(argv[i] + 8); }
        else if (strcmp(argv[i], "--local") == 0 && i+1 < argc) { cli_local = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--pinned") == 0) { cli_pinned = 1; }
        else if (strcmp(argv[i], "--no-pinned") == 0) { cli_pinned = 0; }
        else if (strcmp(argv[i], "--daemon") == 0) { daemon_mode = 1; }
        else if (strcmp(argv[i], "--use-daemon") == 0) { use_daemon = 1; }
        // 移除 --daemon-pinned 和 --daemon-no-pinned 参数
        else if (strcmp(argv[i], "--daemon-socket") == 0 && i+1 < argc) { daemon_socket = argv[++i]; }
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) { outpath = argv[++i]; }
        else if (argv[i][0] == '-') { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
        else { infile = argv[i]; }
    }

    if (!infile && !daemon_mode) { usage(argv[0]); return 1; }

    /* create compressor */
    LZ4GPUCompressor* ctx = lz4_gpu_create_compressor();
    if (!ctx) { fprintf(stderr, "Failed to create GPU compressor context\n"); return 1; }

    // If user requested daemon mode, run server loop now and exit.
    if (daemon_mode) {
        // 直接用cli_pinned参数
        if (cli_pinned >= 0) lz4_gpu_set_pinned_memory(ctx, cli_pinned);
        if (kernel_debug) lz4_gpu_set_kernel_debug(ctx, 1);
        if (host_debug) lz4_gpu_set_host_debug(ctx, 1);
        if (enable_profile) ctx->enable_profiling = 1;
        if (verbose) ctx->verbose = 1;
        if (cli_local > 0) lz4_gpu_set_workgroup_size(ctx, (size_t)cli_local);
        if (!daemon_socket) daemon_socket = getenv("LZ4_GPU_DAEMON_SOCKET");
        // 读取环境变量设置预编译二进制或内核源路径（daemon模式也需要）
        env_clbin_path = getenv("LZ4_GPU_CLBIN");
        env_clsrc_path = getenv("LZ4_GPU_CLSRC");
        if (env_clbin_path && env_clbin_path[0] != '\0') {
            lz4_gpu_use_precompiled(ctx, 1);
            lz4_gpu_set_precompiled_binary(ctx, env_clbin_path);
            if (verbose) fprintf(stderr, "Daemon: Using precompiled OpenCL binary from LZ4_GPU_CLBIN=%s\n", env_clbin_path);
        } else if (env_clsrc_path && env_clsrc_path[0] != '\0') {
            lz4_gpu_set_kernel_source(ctx, env_clsrc_path);
            if (verbose) fprintf(stderr, "Daemon: Using kernel source from LZ4_GPU_CLSRC=%s\n", env_clsrc_path);
        }
        int rc = run_daemon_server(ctx, daemon_socket, cli_pinned);
        lz4_gpu_destroy_compressor(ctx);
        return rc;
    }

    /* read input file (needed for both local and daemon modes) */
    FILE* f = fopen(infile, "rb");
    if (!f) { perror("open input"); lz4_gpu_destroy_compressor(ctx); return 1; }
    fseek(f, 0, SEEK_END);
    long input_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* input_buf = malloc(input_size);
    if (!input_buf) { fprintf(stderr, "Out of memory\n"); fclose(f); lz4_gpu_destroy_compressor(ctx); return 1; }
    if (fread(input_buf, 1, input_size, f) != (size_t)input_size) { perror("read"); free(input_buf); fclose(f); lz4_gpu_destroy_compressor(ctx); return 1; }
    fclose(f);

    char default_out[1024];
    if (!outpath) {
        if (compress) {
            snprintf(default_out, sizeof(default_out), "%s.lz4", infile);
        } else {
            /* If input ends with .lz4, strip it for default output name, otherwise use .out */
            if (ends_with(infile, ".lz4")) {
                size_t len = strlen(infile) - 4;
                if (len >= sizeof(default_out)) len = sizeof(default_out) - 1;
                memcpy(default_out, infile, len);
                default_out[len] = '\0';
            } else {
                snprintf(default_out, sizeof(default_out), "%s.out", infile);
            }
        }
        outpath = default_out;
    }

    /* honor CLI request for vector IO and kernel debug before initializing/building kernels */
    if (kernel_debug) lz4_gpu_set_kernel_debug(ctx, 1);
    if (host_debug) lz4_gpu_set_host_debug(ctx, 1);
    if (enable_profile) ctx->enable_profiling = 1;
    if (verbose) ctx->verbose = 1;
    if (cli_pinned >= 0) {
        lz4_gpu_set_pinned_memory(ctx, cli_pinned);
    }

    // If the user requested to use the daemon, attempt to connect and let the daemon process the request.
    if (use_daemon) {
        if (!daemon_socket) daemon_socket = getenv("LZ4_GPU_DAEMON_SOCKET");
        unsigned char* d_out = NULL; size_t d_out_sz = 0;
        double daemon_elapsed_ms = 0.0;
        LZ4GPUTiming daemon_timing;
        memset(&daemon_timing, 0, sizeof(daemon_timing));
        // Client determines block size and sends to daemon
        size_t client_block_size = 0;
        if (cli_blocksize > 0) {
            client_block_size = cli_blocksize;
        } else {
            // If no explicit block size, use dynamic calculation (same logic as local mode)
            // For now, use 16KB default for daemon
            client_block_size = 16 * 1024;
        }
        int d_rc = run_daemon_client(daemon_socket, input_buf, (size_t)input_size, &d_out, &d_out_sz, compress, accel, cli_local, cli_pinned, client_block_size, &daemon_elapsed_ms, &daemon_timing);
        if (d_rc == 0) {
            // Successful; print statistics in same detailed format as local mode
            if (verbose) {
                double in_mb = input_size / (1024.0 * 1024.0);
                double out_mb = d_out_sz / (1024.0 * 1024.0);
                double ipc_overhead_ms = daemon_elapsed_ms - daemon_timing.total_ms;
                if (ipc_overhead_ms < 0) ipc_overhead_ms = 0.0;

                if (compress) {
                    double ratio = (input_size > 0) ? (double)input_size / (double)d_out_sz : 0.0;
                    double pct = (input_size > 0) ? (double)d_out_sz / (double)input_size * 100.0 : 0.0;
                    long long saved = (long long)input_size - (long long)d_out_sz;

                    fprintf(stderr, "\n=== Compression Statistics (Daemon Mode) ===\n");
                    fprintf(stderr, "Input size       : %zu bytes (%.2f MB)\n", input_size, in_mb);
                    fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", d_out_sz, out_mb);
                    fprintf(stderr, "Compression ratio: %.2f:1 (%.2f%% of original)\n", ratio, pct);
                    fprintf(stderr, "Space saved      : %lld bytes (%.2f MB, %.2f%%)\n", saved, saved/1024.0/1024.0, 100.0 - pct);
                    fprintf(stderr, "Block size       : %zu bytes\n", daemon_timing.block_size);
                    fprintf(stderr, "Number of blocks : %d\n", daemon_timing.num_blocks);

                    double total_s = daemon_timing.total_ms / 1000.0;
                    double throughput_total = total_s > 0.0 ? in_mb / total_s : 0.0;
                    double kernel_s = daemon_timing.kernel_ms > 0.0 ? daemon_timing.kernel_ms / 1000.0 : 0.0;
                    double throughput_kernel = kernel_s > 0.0 ? in_mb / kernel_s : 0.0;
                    if (kernel_s > 0.0) {
                        fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n", throughput_total, throughput_kernel);
                    } else {
                        fprintf(stderr, "Throughput       : %.2f MB/s\n", throughput_total);
                    }
                    fprintf(stderr, "-------------------------------\n");
                    fprintf(stderr, "  Total time:        %8.3f ms\n", daemon_timing.total_ms);
                    fprintf(stderr, "  Buffer Alloc:      %8.3f ms\n", daemon_timing.alloc_ms);
                    fprintf(stderr, "  OpenCL Init:       %8.3f ms\n", daemon_timing.init_ms);
                    fprintf(stderr, "  Host→Device:       %8.3f ms\n", daemon_timing.h2d_ms);
                    fprintf(stderr, "  Setup Args:        %8.3f ms\n", daemon_timing.setup_ms);
                    fprintf(stderr, "  Kernel:            %8.3f ms\n", daemon_timing.kernel_ms);
                    fprintf(stderr, "  Device→Host:       %8.3f ms\n", daemon_timing.d2h_ms);
                    fprintf(stderr, "  Frame Assembly:    %8.3f ms\n", daemon_timing.frame_ms);
                    if (daemon_timing.map_ms > 0.0) {
                        fprintf(stderr, "  Map/Unmap:         %8.3f ms\n", daemon_timing.map_ms);
                    }
                } else {
                    fprintf(stderr, "\n=== Decompression Statistics (Daemon Mode) ===\n");
                    fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", input_size, in_mb);
                    fprintf(stderr, "Decompressed size: %zu bytes (%.2f MB)\n", d_out_sz, out_mb);
                    fprintf(stderr, "Number of blocks : %d\n", daemon_timing.num_blocks);

                    double total_s = daemon_timing.total_ms / 1000.0;
                    double throughput_total = total_s > 0.0 ? out_mb / total_s : 0.0;
                    double kernel_s = daemon_timing.kernel_ms > 0.0 ? daemon_timing.kernel_ms / 1000.0 : 0.0;
                    double throughput_kernel = kernel_s > 0.0 ? out_mb / kernel_s : 0.0;
                    if (kernel_s > 0.0) {
                        fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n", throughput_total, throughput_kernel);
                    } else {
                        fprintf(stderr, "Throughput       : %.2f MB/s\n", throughput_total);
                    }
                    fprintf(stderr, "-------------------------------\n");
                    fprintf(stderr, "  Total time:        %8.3f ms\n", daemon_timing.total_ms);
                    fprintf(stderr, "  Buffer Alloc:      %8.3f ms\n", daemon_timing.alloc_ms);
                    fprintf(stderr, "  Host→Device:       %8.3f ms\n", daemon_timing.h2d_ms);
                    fprintf(stderr, "  Kernel:            %8.3f ms\n", daemon_timing.kernel_ms);
                    fprintf(stderr, "  Device→Host:       %8.3f ms\n", daemon_timing.d2h_ms);
                    if (daemon_timing.map_ms > 0.0) {
                        fprintf(stderr, "  Map/Unmap:         %8.3f ms\n", daemon_timing.map_ms);
                    }
                }
            }
            // write result and exit
            FILE* fo = NULL;
            int close_fo = 1;
            if (outpath[0] == '-' && outpath[1] == '\0') { fo = stdout; close_fo = 0; }
            else { fo = fopen(outpath, "wb"); if (!fo) { perror("open out"); free(d_out); lz4_gpu_destroy_compressor(ctx); free(input_buf); return 1; } }
            if (fwrite(d_out, 1, d_out_sz, fo) != d_out_sz) { perror("write out"); if (close_fo) fclose(fo); free(d_out); lz4_gpu_destroy_compressor(ctx); free(input_buf); return 1; }
            if (close_fo) fclose(fo);
            free(d_out);
            lz4_gpu_destroy_compressor(ctx);
            free(input_buf);
            return 0;
        } else if (d_rc == 2) {
            if (verbose) fprintf(stderr, "Daemon: not available, falling back to local processing\n");
            // fallthrough to local processing
        } else {
            if (verbose) fprintf(stderr, "Daemon: failed to process request (code %d), falling back to local processing\n", d_rc);
            // fallthrough to local processing
        }
    }
    /* Apply any CLI-specified workgroup or block size override before initialize to influence kernel builds/decisions */
    if (cli_local > 0) {
        lz4_gpu_set_workgroup_size(ctx, (size_t)cli_local);
    }
    if (cli_blocksize > 0) {
        /* Clamp and align CLI-specified block size to GPU-friendly bounds */
        size_t bs = cli_blocksize;
        const size_t ALIGN = 4 * 1024;
        const size_t MIN_BLOCK = 16 * 1024;
        if (bs < MIN_BLOCK) bs = MIN_BLOCK;
        if (bs > LZ4_GPU_MAX_BLOCK_SIZE) bs = LZ4_GPU_MAX_BLOCK_SIZE;
        bs = ((bs + ALIGN - 1) / ALIGN) * ALIGN;
          /* Set both compress and decompress block sizes to the same CLI-specified value.
              Users typically expect --blocksize (-B) to apply to both directions. */
          lz4_gpu_set_block_sizes(ctx, bs, bs);
        if (bs != cli_blocksize && verbose) fprintf(stderr, "Note: blocksize clamped/rounded to %zu\n", bs);
    }
    /* Clamp user-specified local sizes to keep kernel launches reasonable */
    if (cli_local > 0 && cli_local > (int)LZ4_GPU_MAX_LOCAL_SIZE) {
        if (verbose) fprintf(stderr, "Note: local size clamped to %d\n", (int)LZ4_GPU_MAX_LOCAL_SIZE);
        cli_local = (int)LZ4_GPU_MAX_LOCAL_SIZE;
    }
    if (cli_local > 0) {
        lz4_gpu_set_workgroup_size(ctx, (size_t)cli_local);
    }
    /* If env variable provided, use it.
       Priority: CLBIN > CLSRC (if both are provided, prefer a precompiled binary). */
    env_clbin_path = getenv("LZ4_GPU_CLBIN");
    env_clsrc_path = getenv("LZ4_GPU_CLSRC");
    if (env_clbin_path && env_clbin_path[0] != '\0') {
        lz4_gpu_use_precompiled(ctx, 1);
        lz4_gpu_set_precompiled_binary(ctx, env_clbin_path);
        if (verbose) fprintf(stderr, "NOTE: Using precompiled OpenCL binary from LZ4_GPU_CLBIN=%s\n", env_clbin_path);
    } else if (env_clsrc_path && env_clsrc_path[0] != '\0') {
        lz4_gpu_set_kernel_source(ctx, env_clsrc_path);
        if (verbose) fprintf(stderr, "NOTE: Using kernel source from LZ4_GPU_CLSRC=%s\n", env_clsrc_path);
    }
    /* Also print environment variables section on verbose to make debugging easier */
    if (verbose) {
        fprintf(stderr, "ENV: LZ4_GPU_CLBIN=%s\n", env_clbin_path ? env_clbin_path : "<unset>");
        fprintf(stderr, "ENV: LZ4_GPU_CLSRC=%s\n", env_clsrc_path ? env_clsrc_path : "<unset>");
        fprintf(stderr, "DEFAULTS: accel=%d block=%zu local=%zu pinned=%d\n", DEFAULT_ACCELERATION, ctx->dynamic_block_size, ctx->default_local, ctx->use_pinned_memory);
    }
    /* Clamp acceleration to maximum allowed */
    if (accel < 1) accel = 1;
    if (accel > LZ4_GPU_MAX_ACCELERATION) {
        if (verbose) fprintf(stderr, "Note: acceleration clamped to %d\n", LZ4_GPU_MAX_ACCELERATION);
        accel = LZ4_GPU_MAX_ACCELERATION;
    }
    if (!lz4_gpu_initialize(ctx)) { fprintf(stderr, "GPU init failed: %s\n", lz4_gpu_get_error_message(ctx)); lz4_gpu_destroy_compressor(ctx); free(input_buf); return 1; }

    int rc = 0;
    if (compress) {
        size_t out_capacity = (size_t)input_size + (size_t)input_size/10 + 65536;
        unsigned char* outbuf = malloc(out_capacity);
            if (!outbuf) { fprintf(stderr, "alloc outbuf failed\n"); rc = 1; goto cleanup; }

        clock_t t0 = clock();
        size_t out_sz = lz4_gpu_compress_frame_accelerated(ctx, input_buf, (size_t)input_size, outbuf, out_capacity, accel);
        clock_t t1 = clock();
        if (out_sz == 0) { fprintf(stderr, "Compression failed: %s\n", lz4_gpu_get_error_message(ctx)); free(outbuf); rc = 1; goto cleanup; }
        double seconds = (double)(t1 - t0) / CLOCKS_PER_SEC;

        // Get and print timing information
        LZ4GPUTiming timing;
        if (lz4_gpu_get_last_timing(ctx, &timing)) {
            if (bench || verbose) {
                /* Suppress early verbose summary; we'll print a single consolidated
                   Compression Statistics + detailed timing block after the file is
                   written (so file-write time is included in totals). */

                /* CLI-level additional breakdown including file write time if available */
                /* Note: file write timing will be printed separately below if verbose is set */
            }
        }

        if (bench) {
            double mb = (double)input_size / (1024.0*1024.0);
            fprintf(stderr, "COMPRESS: in=%zu out=%zu time=%.3fs throughput=%.2fMB/s ratio=%.3f\n", (size_t)input_size, out_sz, seconds, mb/seconds, (double)out_sz / (double)input_size);
        }
        double t_write_ms = 0.0;
        if (verbose) {
            /* Measure file write duration for more detailed breakdown */
            clock_t tw0 = clock();
            if (verbose) fprintf(stderr, "Writing output %s (%zu bytes)\n", outpath, out_sz);
            clock_t tw1 = clock();
            t_write_ms = (double)(tw1 - tw0) / CLOCKS_PER_SEC * 1000.0; /* will be tiny since we didn't include fwrite here */
        }
        FILE* fo = NULL;
        int close_fo = 1;
        if (outpath[0] == '-' && outpath[1] == '\0') {
            fo = stdout;
            close_fo = 0;
#if defined(_WIN32)
            _setmode(_fileno(stdout), _O_BINARY);
#endif
        } else {
            fo = fopen(outpath, "wb");
            if (!fo) { perror("open out"); free(outbuf); rc = 1; goto cleanup; }
        }
        /* Measure actual file write time for verbose breakdown */
        clock_t tw0 = clock();
        if (fwrite(outbuf, 1, out_sz, fo) != out_sz) { perror("write out"); if (close_fo) fclose(fo); free(outbuf); rc = 1; goto cleanup; }
        clock_t tw1 = clock();
        t_write_ms = (double)(tw1 - tw0) / CLOCKS_PER_SEC * 1000.0;
        if (close_fo) fclose(fo);
        free(outbuf);

        if (verbose && lz4_gpu_get_last_timing(ctx, &timing)) {
            /* Consolidated Compression Statistics + Timing (includes file write) */
            double total_ms = timing.total_ms + t_write_ms;
            double denom_ms = total_ms + timing.init_ms; /* include OpenCL init in percentage denominator */
            if (denom_ms <= 0.0) denom_ms = total_ms;
            if (total_ms <= 0.0) total_ms = timing.total_ms; /* fallback */

            double in_mb = (double)input_size / (1024.0*1024.0);
            double out_mb = (double)out_sz / (1024.0*1024.0);
            double ratio = in_mb > 0.0 ? ((double)input_size) / ((double)out_sz) : 0.0;
            double pct = (double)out_sz / (double)input_size * 100.0;
            long long saved = (long long)input_size - (long long)out_sz;

            printf("=== Compression Statistics ===\n");
            printf("Input size       : %zu bytes (%.2f MB)\n", (size_t)input_size, in_mb);
            printf("Compressed size  : %zu bytes (%.2f MB)\n", out_sz, out_mb);
            printf("Compression ratio: %.2f:1 (%.2f%% of original)\n", ratio, pct);
            printf("Space saved      : %lld bytes (%.2f MB, %.2f%%)\n", saved, (double)saved/1024.0/1024.0, 100.0 * ((double)saved) / (double)input_size);
            printf("Block size       : %zu bytes (%.2f KB)\n", ctx->dynamic_block_size, (double)ctx->dynamic_block_size / 1024.0);
            printf("Number of blocks : %d\n", timing.num_blocks);
            printf("Compression level: %d\n", accel);
            printf("Work groups      : global=%d, local=auto\n", timing.num_blocks);

            /* throughput using total_ms; kernel throughput only if kernel_ms available */
            double total_s = total_ms / 1000.0;
            double throughput_total = total_s > 0.0 ? in_mb / total_s : 0.0;
            double kernel_s = -1.0;
            double kernel_ms_display = timing.kernel_ms;
            int kernel_is_approx = 0;
            /* If we don't have device profiling info, estimate kernel time by
               subtracting other known parts from the recorded compressor total. */
            if (kernel_ms_display < 0.0) {
                double sum_known = timing.init_ms + timing.alloc_ms + timing.h2d_ms + timing.setup_ms + timing.d2h_ms + timing.frame_ms;
                double approx = timing.total_ms - sum_known; /* approximate kernel within timing.total_ms */
                if (approx < 0.0) approx = 0.0;
                /* Accept any non-negative approximation — even small values —
                   because users asked to always show a kernel duration when
                   profiling is disabled. We'll still avoid computing kernel
                   throughput if the approximated kernel time is zero. */
                kernel_ms_display = approx;
                kernel_is_approx = 1;
            }
            if (kernel_ms_display > 0.0) kernel_s = kernel_ms_display / 1000.0;
            if (kernel_s > 0.0) {
                double throughput_kernel = in_mb / kernel_s;
                        printf("Throughput       : %.2f MB/s (kernel: %.2f MB/s%s)\n", throughput_total, throughput_kernel, kernel_is_approx ? " (approx)" : "");
            } else {
                printf("Throughput       : %.2f MB/s (kernel: N/A)\n", throughput_total);
            }

            printf("-------------------------------\n");
            printf("  Total time:      %8.3f ms\n", total_ms);
            printf("  Buffer Alloc:    %8.3f ms\n", timing.alloc_ms);
            printf("  OpenCL Init:     %8.3f ms\n", timing.init_ms);
            printf("  Host→Device:     %8.3f ms\n", timing.h2d_ms);
            printf("  Setup Args:      %8.3f ms\n", timing.setup_ms);
            if (kernel_ms_display < 0.0) {
                /* previously tried to set an approximate kernel part; recompute
                   and use it only if it's meaningful (>= 1ms) */
                double sum_known = timing.init_ms + timing.alloc_ms + timing.h2d_ms + timing.setup_ms + timing.d2h_ms + timing.frame_ms;
                double approx = timing.total_ms - sum_known;
                if (approx < 0.0) approx = 0.0;
                kernel_ms_display = approx;
                kernel_is_approx = 1;
            }
            if (kernel_ms_display >= 0.0) printf("  Kernel Exec:     %8.3f ms%s\n", kernel_ms_display, kernel_is_approx ? " (approx)" : "");
            else printf("  Kernel Exec:         N/A\n");
            if (timing.event_profile_ms > 0.0) printf("   (event profiling) : %8.3f ms\n", timing.event_profile_ms);
            printf("  Device→Host:     %8.3f ms\n", timing.d2h_ms);
            printf("  Frame assembly:  %8.3f ms\n", timing.frame_ms);
            printf("  File Write:      %8.3f ms\n", t_write_ms);
            printf("-------------------------------\n");

                 printf("=== Percentage Breakdown ===\n");
                if (kernel_ms_display >= 0.0) printf("Kernel Exec     : %6.2f%%\n", 100.0 * kernel_ms_display / denom_ms);
            else printf("Kernel Exec     : N/A\n");
            double data_transfer = timing.h2d_ms + timing.d2h_ms;
                 printf("OpenCL Init     : %6.2f%%\n", 100.0 * timing.init_ms / denom_ms);
                 printf("Data Transfer   : %6.2f%% (upload=%4.2f%% + download=%4.2f%%)\n",
                     100.0 * data_transfer / denom_ms, 100.0 * timing.h2d_ms / denom_ms, 100.0 * timing.d2h_ms / denom_ms);
                 printf("File I/O        : %6.2f%%\n", 100.0 * t_write_ms / denom_ms);
                 printf("Buffer Alloc    : %6.2f%%\n", 100.0 * timing.alloc_ms / denom_ms);
                 printf("Setup Args      : %6.2f%%\n", 100.0 * timing.setup_ms / denom_ms);
            printf("===============================\n");
        }
    } else {
        /* Try to determine exact output size from frame header if present. */
        size_t out_capacity = 0;
        size_t parsed_content_size = 0;
        if (parse_lz4f_content_size(input_buf, (size_t)input_size, &parsed_content_size)) {
            if (parsed_content_size > 0) {
                /* allocate exact size + small margin */
                out_capacity = parsed_content_size + 64;
                if (verbose) fprintf(stderr, "Detected frame content size: %zu, allocating %zu bytes\n", parsed_content_size, out_capacity);
            }
        }

        /* Fallback: start with heuristic capacity and grow if needed */
        if (out_capacity == 0) {
            out_capacity = (size_t)input_size * 4 + 65536;
            if (out_capacity < 65536) out_capacity = 65536;
        }

    /* Ask GPU to estimate decompressed size first. If that succeeds we can allocate exactly. */
    size_t required_size = lz4_gpu_estimate_decompressed_size(ctx, input_buf, (size_t)input_size);
    unsigned char* outbuf = NULL;
    size_t out_sz = 0;
    clock_t t0 = 0, t1 = 0;

    if (required_size != 0) {
        out_capacity = required_size + 64;
        if (verbose) fprintf(stderr, "Estimated decompressed size: %zu, allocating %zu bytes\n", required_size, out_capacity);
        outbuf = (unsigned char*)malloc(out_capacity);
        if (!outbuf) { fprintf(stderr, "alloc outbuf failed (requested %zu bytes)\n", out_capacity); rc = 1; goto cleanup; }
        t0 = clock();
        out_sz = lz4_gpu_decompress_frame(ctx, input_buf, (size_t)input_size, outbuf, out_capacity);
        t1 = clock();
        if (out_sz == 0) {
            const char* emsg = lz4_gpu_get_error_message(ctx);
            fprintf(stderr, "Decompression failed after estimate allocation: %s\n", emsg ? emsg : "unknown error");
            free(outbuf); rc = 1; goto cleanup;
        }
    } else {
        /* Estimate failed; fall back to header parsing + heuristic and growth strategy. */
        if (parsed_content_size > 0) out_capacity = parsed_content_size + 64;
        if (out_capacity == 0) {
            out_capacity = (size_t)input_size * 4 + 65536;
            if (out_capacity < 65536) out_capacity = 65536;
        }
        outbuf = (unsigned char*)malloc(out_capacity);
        if (!outbuf) { fprintf(stderr, "alloc outbuf failed (requested %zu bytes)\n", out_capacity); rc = 1; goto cleanup; }
        t0 = clock();
        out_sz = lz4_gpu_decompress_frame(ctx, input_buf, (size_t)input_size, outbuf, out_capacity);
        t1 = clock();
        if (out_sz == 0) {
            const char* emsg = lz4_gpu_get_error_message(ctx);
            if (parsed_content_size > 0 && emsg && (strstr(emsg, "Output buffer too small") || strstr(emsg, "buffer too small"))) {
                free(outbuf);
                outbuf = NULL;
                const size_t MAX_OUT_CAP = (size_t)8ULL * 1024ULL * 1024ULL * 1024ULL; /* 8GB cap */
                size_t grow_cap = parsed_content_size + 64;
                while (grow_cap <= MAX_OUT_CAP) {
                    unsigned char* tmp = (unsigned char*)realloc(outbuf, grow_cap);
                    if (!tmp) { fprintf(stderr, "alloc outbuf failed (requested %zu bytes)\n", grow_cap); free(outbuf); rc = 1; goto cleanup; }
                    outbuf = tmp;
                    t0 = clock();
                    out_sz = lz4_gpu_decompress_frame(ctx, input_buf, (size_t)input_size, outbuf, grow_cap);
                    t1 = clock();
                    if (out_sz != 0) break;
                    emsg = lz4_gpu_get_error_message(ctx);
                    if (!(emsg && (strstr(emsg, "Output buffer too small") || strstr(emsg, "buffer too small")))) break;
                    grow_cap *= 2;
                }
                if (out_sz == 0) { fprintf(stderr, "Decompression failed after growth attempts: %s\n", emsg ? emsg : "unknown error"); free(outbuf); rc = 1; goto cleanup; }
            } else { fprintf(stderr, "Decompression failed: %s\n", emsg ? emsg : "unknown error"); free(outbuf); rc = 1; goto cleanup; }
        }
    }

    double seconds = (double)(t1 - t0) / CLOCKS_PER_SEC;
    if (bench) {
        double mb = (double)out_sz / (1024.0*1024.0);
        fprintf(stderr, "DECOMPRESS: out=%zu time=%.3fs throughput=%.2fMB/s\n", out_sz, seconds, mb/seconds);
    }
    if (verbose) fprintf(stderr, "Writing output %s (%zu bytes)\n", outpath, out_sz);
    FILE* fo = NULL;
    int close_fo = 1;
    if (outpath[0] == '-' && outpath[1] == '\0') {
        fo = stdout;
        close_fo = 0;
#if defined(_WIN32)
        _setmode(_fileno(stdout), _O_BINARY);
#endif
    } else {
        fo = fopen(outpath, "wb");
        if (!fo) { perror("open out"); free(outbuf); rc = 1; goto cleanup; }
    }
    /* Measure file write time for detailed breakdown */
    clock_t tw0 = clock();
    if (fwrite(outbuf, 1, out_sz, fo) != out_sz) { perror("write out"); if (close_fo) fclose(fo); free(outbuf); rc = 1; goto cleanup; }
    clock_t tw1 = clock();
    double t_write_ms = (double)(tw1 - tw0) / CLOCKS_PER_SEC * 1000.0;
    if (close_fo) fclose(fo);
    free(outbuf);

    if (verbose) {
        LZ4GPUTiming timing;
        if (lz4_gpu_get_last_timing(ctx, &timing)) {
            double total_ms = timing.total_ms + t_write_ms;
            if (total_ms <= 0.0) total_ms = timing.total_ms;
            double denom_ms = total_ms + timing.init_ms; /* include OpenCL init in percentage denominator */
            if (denom_ms <= 0.0) denom_ms = total_ms;

            double out_mb = (double)out_sz / (1024.0*1024.0);
            double total_s = total_ms / 1000.0;
            double throughput_total = total_s > 0.0 ? out_mb / total_s : 0.0;

            /* Kernel display logic mirrors compression: if kernel_ms < 0 use an approx
               computed from remaining time; mark approximate values explicitly. */
            double kernel_ms_display = timing.kernel_ms;
            int kernel_is_approx = 0;
            if (kernel_ms_display < 0.0) {
                double sum_known = timing.init_ms + timing.alloc_ms + timing.h2d_ms + timing.setup_ms + timing.d2h_ms + timing.frame_ms;
                double approx = timing.total_ms - sum_known;
                if (approx < 0.0) approx = 0.0;
                kernel_ms_display = approx;
                kernel_is_approx = 1;
            }

            double kernel_s = -1.0;
            if (kernel_ms_display > 0.0) kernel_s = kernel_ms_display / 1000.0;

            printf("=== Decompression Statistics ===\n");
            printf("Compressed size  : %zu bytes (%.2f MB)\n", (size_t)input_size, (double)input_size / (1024.0*1024.0));
            printf("Output size      : %zu bytes (%.2f MB)\n", out_sz, out_mb);
            printf("Number of blocks : %d\n", timing.num_blocks);
            printf("Work groups      : global=%d, local=1\n", timing.num_blocks);
            if (kernel_s > 0.0) {
                double throughput_kernel = out_mb / kernel_s;
                printf("Throughput       : %.2f MB/s (kernel: %.2f MB/s%s)\n", throughput_total, throughput_kernel, kernel_is_approx ? " (approx)" : "");
            } else {
                printf("Throughput       : %.2f MB/s (kernel: N/A)\n", throughput_total);
            }

            printf("-------------------------------\n");
            printf("  Total time:      %8.3f ms\n", total_ms);
            printf("  Buffer Alloc:    %8.3f ms\n", timing.alloc_ms);
            printf("  OpenCL Init:     %8.3f ms\n", timing.init_ms);
            printf("  Host→Device:     %8.3f ms\n", timing.h2d_ms);
            printf("  Setup Args:      %8.3f ms\n", timing.setup_ms);
            if (kernel_ms_display >= 0.0) printf("  Kernel Exec:     %8.3f ms%s\n", kernel_ms_display, kernel_is_approx ? " (approx)" : "");
            else printf("  Kernel Exec:         N/A\n");
            if (timing.event_profile_ms > 0.0) printf("   (event profiling) : %8.3f ms\n", timing.event_profile_ms);
            printf("  Device→Host:     %8.3f ms\n", timing.d2h_ms);
            printf("  Frame assembly:  %8.3f ms\n", timing.frame_ms);
            printf("  File Write:      %8.3f ms\n", t_write_ms);
            printf("-------------------------------\n");

                 printf("=== Percentage Breakdown ===\n");
                 if (kernel_ms_display >= 0.0) printf("Kernel Exec     : %6.2f%%\n", 100.0 * kernel_ms_display / denom_ms);
                 else printf("Kernel Exec     : N/A\n");
                 double data_transfer = timing.h2d_ms + timing.d2h_ms;
                 printf("OpenCL Init     : %6.2f%%\n", 100.0 * timing.init_ms / denom_ms);
                 printf("Data Transfer   : %6.2f%% (upload=%4.2f%% + download=%4.2f%%)\n",
                     100.0 * data_transfer / denom_ms, 100.0 * timing.h2d_ms / denom_ms, 100.0 * timing.d2h_ms / denom_ms);
                 printf("File I/O        : %6.2f%%\n", 100.0 * t_write_ms / denom_ms);
                 printf("Buffer Alloc    : %6.2f%%\n", 100.0 * timing.alloc_ms / denom_ms);
                 printf("Setup Args      : %6.2f%%\n", 100.0 * timing.setup_ms / denom_ms);
            printf("===============================\n");
        }
    }
    }

cleanup:
    lz4_gpu_destroy_compressor(ctx);
    free(input_buf);
    return rc;
}
