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

#define DEFAULT_ACCELERATION 1

static int ends_with(const char* str, const char* suf) {
    if (!str || !suf) return 0;
    size_t lstr = strlen(str);
    size_t lsuf = strlen(suf);
    if (lsuf > lstr) return 0;
    return strcmp(str + lstr - lsuf, suf) == 0;
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
    "Usage: %s [-c|-d] [-l level] [-o outfile|-] [--bench] [-v] <input>\n"
        "  -c        compress (default)\n"
        "  -d        decompress\n"
        "  -l LEVEL  compression acceleration level (higher -> faster, less ratio)\n"
    "  -o FILE   output file (default: input.lz4 for compress, input.out for decompress). Use '-' to write to stdout.\n"
        "  -g|--kernel-debug Enable kernel-side debug prints (build with LZ4_GPU_KERNEL_DEBUG)\n"
        "  -p|--profile  Enable OpenCL event profiling and print upload/kernel/download CSV\n"
        "  --bench   print throughput and compression ratio summary\n"
        "  -v        verbose logging\n",
        prog);
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
    const char* infile = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0) { compress = 1; }
        else if (strcmp(argv[i], "-d") == 0) { compress = 0; }
    else if (strcmp(argv[i], "-v") == 0) { verbose = 1; }
        else if (strcmp(argv[i], "--bench") == 0) { bench = 1; }
    else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--kernel-debug") == 0) { kernel_debug = 1; }
    else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--host-debug") == 0) { host_debug = 1; }
    else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--profile") == 0) { enable_profile = 1; }
        else if (strcmp(argv[i], "-l") == 0 && i+1 < argc) { accel = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) { outpath = argv[++i]; }
        else if (argv[i][0] == '-') { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
        else { infile = argv[i]; }
    }

    if (!infile) { usage(argv[0]); return 1; }

    /* read input file */
    FILE* f = fopen(infile, "rb");
    if (!f) { perror("open input"); return 1; }
    fseek(f, 0, SEEK_END);
    long input_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* input_buf = malloc(input_size);
    if (!input_buf) { fprintf(stderr, "Out of memory\n"); fclose(f); return 1; }
    if (fread(input_buf, 1, input_size, f) != (size_t)input_size) { perror("read"); free(input_buf); fclose(f); return 1; }
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

    /* create compressor */
    LZ4GPUCompressor* ctx = lz4_gpu_create_compressor();
    if (!ctx) { fprintf(stderr, "Failed to create GPU compressor context\n"); free(input_buf); return 1; }
    /* honor CLI request for vector IO and kernel debug before initializing/building kernels */
    if (kernel_debug) lz4_gpu_set_kernel_debug(ctx, 1);
    if (host_debug) lz4_gpu_set_host_debug(ctx, 1);
    if (enable_profile) ctx->enable_profiling = 1;
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
                double sum_known = timing.alloc_ms + timing.h2d_ms + timing.setup_ms + timing.d2h_ms + timing.frame_ms;
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
            printf("  Host→Device:     %8.3f ms\n", timing.h2d_ms);
            printf("  Setup Args:      %8.3f ms\n", timing.setup_ms);
            if (kernel_ms_display < 0.0) {
                /* previously tried to set an approximate kernel part; recompute
                   and use it only if it's meaningful (>= 1ms) */
                double sum_known = timing.alloc_ms + timing.h2d_ms + timing.setup_ms + timing.d2h_ms + timing.frame_ms;
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
            if (kernel_ms_display >= 0.0) printf("Kernel Exec     : %6.2f%%\n", 100.0 * kernel_ms_display / total_ms);
            else printf("Kernel Exec     : N/A\n");
            double data_transfer = timing.h2d_ms + timing.d2h_ms;
            printf("Data Transfer   : %6.2f%% (upload=%4.2f%% + download=%4.2f%%)\n",
                   100.0 * data_transfer / total_ms, 100.0 * timing.h2d_ms / total_ms, 100.0 * timing.d2h_ms / total_ms);
            printf("File I/O        : %6.2f%%\n", 100.0 * t_write_ms / total_ms);
            printf("Buffer Alloc    : %6.2f%%\n", 100.0 * timing.alloc_ms / total_ms);
            printf("Setup Args      : %6.2f%%\n", 100.0 * timing.setup_ms / total_ms);
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

            double out_mb = (double)out_sz / (1024.0*1024.0);
            double total_s = total_ms / 1000.0;
            double throughput_total = total_s > 0.0 ? out_mb / total_s : 0.0;

            /* Kernel display logic mirrors compression: if kernel_ms < 0 use an approx
               computed from remaining time; mark approximate values explicitly. */
            double kernel_ms_display = timing.kernel_ms;
            int kernel_is_approx = 0;
            if (kernel_ms_display < 0.0) {
                double sum_known = timing.alloc_ms + timing.h2d_ms + timing.setup_ms + timing.d2h_ms + timing.frame_ms;
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
            if (kernel_ms_display >= 0.0) printf("Kernel Exec     : %6.2f%%\n", 100.0 * kernel_ms_display / total_ms);
            else printf("Kernel Exec     : N/A\n");
            double data_transfer = timing.h2d_ms + timing.d2h_ms;
            printf("Data Transfer   : %6.2f%% (upload=%4.2f%% + download=%4.2f%%)\n",
                   100.0 * data_transfer / total_ms, 100.0 * timing.h2d_ms / total_ms, 100.0 * timing.d2h_ms / total_ms);
            printf("File I/O        : %6.2f%%\n", 100.0 * t_write_ms / total_ms);
            printf("Buffer Alloc    : %6.2f%%\n", 100.0 * timing.alloc_ms / total_ms);
            printf("Setup Args      : %6.2f%%\n", 100.0 * timing.setup_ms / total_ms);
            printf("===============================\n");
        }
    }
    }

cleanup:
    lz4_gpu_destroy_compressor(ctx);
    free(input_buf);
    return rc;
}
