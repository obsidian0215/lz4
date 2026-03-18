#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <limits.h>
#include <sys/stat.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#else
#include <windows.h>
#endif
#include "lz4_gpu_core.h"
#include "lz4_gpu_utils.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

enum {
    LZ4_DBG_COMP_SEARCH_ITERS = 0,
    LZ4_DBG_COMP_FP_CHECKS,
    LZ4_DBG_COMP_MATCH_FOUND,
    LZ4_DBG_COMP_LITERAL_BYTES,
    LZ4_DBG_COMP_MATCH_BYTES,
    LZ4_DBG_COMP_LASTLIT_BYTES,
    LZ4_DBG_COMP_N
};

enum {
    LZ4_DBG_DEC_TOKENS = 0,
    LZ4_DBG_DEC_LITERAL_BYTES,
    LZ4_DBG_DEC_MATCH_BYTES,
    LZ4_DBG_DEC_SMALL_OFFSETS,
    LZ4_DBG_DEC_OUTPUT_ERROR,
    LZ4_DBG_DEC_N
};

void lz4_gpu_workspace_init(lz4_gpu_workspace_t* ws) {
    memset(ws, 0, sizeof(*ws));
    ws->comp_epoch_base = 1;
}

void lz4_gpu_workspace_free(lz4_gpu_workspace_t* ws) {
    if (ws->in_buf) clReleaseMemObject(ws->in_buf);
    if (ws->comp_in_buf) clReleaseMemObject(ws->comp_in_buf);
    if (ws->out_buf) clReleaseMemObject(ws->out_buf);
    if (ws->packed_out_buf) clReleaseMemObject(ws->packed_out_buf);
    if (ws->block_info_buf) clReleaseMemObject(ws->block_info_buf);
    if (ws->out_offsets_buf) clReleaseMemObject(ws->out_offsets_buf);
    if (ws->packed_offsets_buf) clReleaseMemObject(ws->packed_offsets_buf);
    if (ws->output_size_buf) clReleaseMemObject(ws->output_size_buf);
    if (ws->dict_buf) clReleaseMemObject(ws->dict_buf);
    if (ws->decomp_comp_off_buf) clReleaseMemObject(ws->decomp_comp_off_buf);
    if (ws->decomp_comp_size_buf) clReleaseMemObject(ws->decomp_comp_size_buf);
    if (ws->decomp_out_off_buf) clReleaseMemObject(ws->decomp_out_off_buf);
    if (ws->decomp_max_out_buf) clReleaseMemObject(ws->decomp_max_out_buf);
    if (ws->decomp_sizes_out_buf) clReleaseMemObject(ws->decomp_sizes_out_buf);
    memset(ws, 0, sizeof(*ws));
}

static int lz4_device_host_unified_memory(cl_command_queue queue) {
    cl_device_id qdev = NULL;
    cl_bool unified = CL_FALSE;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) != CL_SUCCESS || !qdev) {
        return 1;
    }
    if (clGetDeviceInfo(qdev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL) != CL_SUCCESS) {
        return 1;
    }
    return unified == CL_TRUE;
}

static int lz4_prefers_standard_copy(cl_command_queue queue) {
    const char* env = getenv("LZ4_STANDARD_COPY");
    if (env && *env) {
        if (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0 || strcasecmp(env, "yes") == 0) return 1;
        if (strcmp(env, "0") == 0 || strcasecmp(env, "false") == 0 || strcasecmp(env, "no") == 0) return 0;
    }
    return lz4_device_host_unified_memory(queue) ? 0 : 1;
}

static int lz4_env_flag_value(const char* name, int* is_set) {
    const char* env = getenv(name);
    if (is_set) *is_set = 0;
    if (!env || !*env) return 0;
    if (is_set) *is_set = 1;
    if (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0 || strcasecmp(env, "yes") == 0 || strcasecmp(env, "on") == 0) return 1;
    if (strcmp(env, "0") == 0 || strcasecmp(env, "false") == 0 || strcasecmp(env, "no") == 0 || strcasecmp(env, "off") == 0) return 0;
    return atoi(env) != 0;
}

static unsigned lz4_env_unsigned_value(const char* name, unsigned defv) {
    const char* env = getenv(name);
    char* end = NULL;
    unsigned long parsed;
    if (!env || !*env) return defv;
    parsed = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || parsed > UINT_MAX) return defv;
    return (unsigned)parsed;
}

static int lz4_should_use_device_compaction(size_t packed_bytes, size_t sparse_bytes, size_t num_blocks, cl_kernel pack_kernel) {
    int force_set = 0;
    int force_value = lz4_env_flag_value("LZ4_GPU_FORCE_COMPACTION", &force_set);
    int enable_set = 0;
    int enable_value = lz4_env_flag_value("LZ4_GPU_ENABLE_COMPACTION", &enable_set);
    if (!pack_kernel) return 0;
    if (force_set) return force_value;
    if (!enable_set || !enable_value) return 0;
    if (packed_bytes == 0 || sparse_bytes == 0 || packed_bytes >= sparse_bytes) return 0;
    {
        unsigned min_blocks = lz4_env_unsigned_value("LZ4_GPU_COMPACTION_MIN_BLOCKS", 8U);
        unsigned min_gain_pct = lz4_env_unsigned_value("LZ4_GPU_COMPACTION_MIN_GAIN_PCT", 5U);
        size_t saved_bytes;
        if (num_blocks < (size_t)min_blocks) return 0;
        saved_bytes = sparse_bytes - packed_bytes;
        return saved_bytes * 100U >= sparse_bytes * (size_t)min_gain_pct;
    }
}

cl_mem ensure_buffer_ex(cl_context context, cl_mem buf, size_t size, size_t* current_capacity, cl_mem_flags flags, int alloc_host_ptr, cl_int* err) {
    if (buf && *current_capacity >= size) {
        if (err) *err = CL_SUCCESS;
        return buf;
    }
    if (buf) clReleaseMemObject(buf);

    {
        cl_mem_flags create_flags = flags;
        if (alloc_host_ptr) create_flags |= CL_MEM_ALLOC_HOST_PTR;
        cl_mem nbuf = clCreateBuffer(context, create_flags, size, NULL, err);
        if (nbuf) {
            *current_capacity = size;
        } else {
            *current_capacity = 0;
        }
        return nbuf;
    }
}

cl_mem ensure_buffer(cl_context context, cl_mem buf, size_t size, size_t* current_capacity, cl_int* err) {
    return ensure_buffer_ex(context, buf, size, current_capacity, CL_MEM_READ_WRITE, 1, err);
}

int write_buffer_auto(cl_command_queue queue, cl_mem buf, const void* src, size_t bytes, int standard_copy) {
    cl_int err;
    if (!buf || !src || bytes == 0) return 0;

    if (standard_copy) {
        err = clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, src, 0, NULL, NULL);
        return (err == CL_SUCCESS) ? 0 : -1;
    }

    void* mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    if (err == CL_SUCCESS && mapped) {
        memcpy(mapped, src, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
        return 0;
    }

    err = clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, src, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

int write_buffer_mapped(cl_command_queue queue, cl_mem buf, const void* src, size_t bytes) {
    return write_buffer_auto(queue, buf, src, bytes, 0);
}

int read_buffer_auto(cl_command_queue queue, cl_mem buf, void* dst, size_t bytes, int standard_copy) {
    cl_int err;
    if (!buf || !dst || bytes == 0) return 0;

    if (standard_copy) {
        err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
        return (err == CL_SUCCESS) ? 0 : -1;
    }

    void* mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &err);
    if (err == CL_SUCCESS && mapped) {
        memcpy(dst, mapped, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
        return 0;
    }

    err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

static int zero_buffer(cl_command_queue queue, cl_mem buf, size_t bytes) {
    if (!buf || bytes == 0) return 0;
#if defined(CL_VERSION_1_2)
    {
        static const cl_uint z = 0;
        cl_int ferr = clEnqueueFillBuffer(queue, buf, &z, sizeof(z), 0, bytes, 0, NULL, NULL);
        if (ferr == CL_SUCCESS) {
            clFinish(queue);
            return 0;
        }
    }
#endif
    {
        cl_int err;
        void* p = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !p) return -1;
        memset(p, 0, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, p, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
        clFinish(queue);
    }
    return 0;
}

static size_t round_up_size(size_t v, size_t align) {
    if (align == 0) return v;
    return ((v + align - 1) / align) * align;
}

static size_t sanitize_local_size(cl_command_queue queue, size_t requested, size_t upper_blocks) {
    if (upper_blocks == 0) return 1;
    size_t l = requested;

    cl_device_id qdev = NULL;
    size_t max_wg = 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    }

    if (l == 0) {
        /* auto local size path */
        l = 8;
    }

    if (l > max_wg) l = max_wg;
    if (l > upper_blocks) l = upper_blocks;
    if (l == 0) l = 1;

    /* Keep local size as power-of-two for stable occupancy behavior. */
    size_t p2 = 1;
    while ((p2 << 1) <= l) p2 <<= 1;
    return p2;
}

static size_t parse_wi_per_cu_env(const char* primary_env, const char* fallback_env, size_t defv) {
    const char* env = NULL;
    char* end = NULL;
    unsigned long parsed;

    if (primary_env && *primary_env) env = getenv(primary_env);
    if ((!env || !*env) && fallback_env && *fallback_env) env = getenv(fallback_env);
    if (!env || !*env) return defv;

    parsed = strtoul(env, &end, 10);
    if (end != env && parsed > 0) return (size_t)parsed;
    return defv;
}

static size_t choose_comp_worker_count(cl_command_queue queue, size_t num_blocks, size_t local_size) {
    if (num_blocks == 0) return 1;

    cl_device_id qdev = NULL;
    cl_uint cu = 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    }
    if (cu == 0) cu = 1;

    size_t default_wi_per_cu = (num_blocks >= 4096) ? 16 : 24;
    size_t wi_per_cu = parse_wi_per_cu_env("LZ4_GPU_COMP_WI_PER_CU", "LZ4_GPU_WI_PER_CU", default_wi_per_cu);

    size_t target = (size_t)cu * wi_per_cu;
    if (target < local_size) target = local_size;
    if (target > num_blocks) target = num_blocks;
    if (target == 0) target = 1;
    return target;
}

static size_t choose_decomp_worker_count(cl_command_queue queue, size_t num_blocks, size_t local_size) {
    if (num_blocks == 0) return 1;

    cl_device_id qdev = NULL;
    cl_uint cu = 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    }
    if (cu == 0) cu = 1;

    size_t default_wi_per_cu = (num_blocks >= 4096) ? 48 : 96;
    size_t wi_per_cu = parse_wi_per_cu_env("LZ4_GPU_DECOMP_WI_PER_CU", "LZ4_GPU_WI_PER_CU", default_wi_per_cu);

    size_t target = (size_t)cu * wi_per_cu;
    if (target < local_size) target = local_size;
    if (target > num_blocks) target = num_blocks;
    if (target == 0) target = 1;
    return target;
}

static int lz4_debug_counters_enabled(void) {
    const char* env = getenv("LZ4_GPU_DEBUG_COUNTERS");
    if (!env || !*env) return 0;
    return strcmp(env, "0") != 0;
}

static void lz4_print_comp_debug_stats(const uint32_t* stats, int num_blocks) {
    unsigned long long search_iters = 0;
    unsigned long long fp_checks = 0;
    unsigned long long match_found = 0;
    unsigned long long literal_bytes = 0;
    unsigned long long match_bytes = 0;
    unsigned long long lastlit_bytes = 0;

    for (int i = 0; i < num_blocks; ++i) {
        const size_t base = (size_t)i * LZ4_DBG_COMP_N;
        search_iters += stats[base + LZ4_DBG_COMP_SEARCH_ITERS];
        fp_checks += stats[base + LZ4_DBG_COMP_FP_CHECKS];
        match_found += stats[base + LZ4_DBG_COMP_MATCH_FOUND];
        literal_bytes += stats[base + LZ4_DBG_COMP_LITERAL_BYTES];
        match_bytes += stats[base + LZ4_DBG_COMP_MATCH_BYTES];
        lastlit_bytes += stats[base + LZ4_DBG_COMP_LASTLIT_BYTES];
    }

    double avg_search = (num_blocks > 0) ? ((double)search_iters / (double)num_blocks) : 0.0;
    double hit_rate = (fp_checks > 0) ? ((double)match_found / (double)fp_checks) : 0.0;
    fprintf(stderr,
            "[LZ4-DBG][COMP] blocks=%d search_iters=%llu fp_checks=%llu match_found=%llu hit_rate=%.4f literals=%llu matches=%llu last_literals=%llu avg_search/block=%.2f\n",
            num_blocks,
            search_iters,
            fp_checks,
            match_found,
            hit_rate,
            literal_bytes,
            match_bytes,
            lastlit_bytes,
            avg_search);
}

static void lz4_print_dec_debug_stats(const uint32_t* stats, int num_blocks) {
    unsigned long long tokens = 0;
    unsigned long long literal_bytes = 0;
    unsigned long long match_bytes = 0;
    unsigned long long small_offsets = 0;
    unsigned long long output_errors = 0;

    for (int i = 0; i < num_blocks; ++i) {
        const size_t base = (size_t)i * LZ4_DBG_DEC_N;
        tokens += stats[base + LZ4_DBG_DEC_TOKENS];
        literal_bytes += stats[base + LZ4_DBG_DEC_LITERAL_BYTES];
        match_bytes += stats[base + LZ4_DBG_DEC_MATCH_BYTES];
        small_offsets += stats[base + LZ4_DBG_DEC_SMALL_OFFSETS];
        output_errors += stats[base + LZ4_DBG_DEC_OUTPUT_ERROR];
    }

    double avg_tokens = (num_blocks > 0) ? ((double)tokens / (double)num_blocks) : 0.0;
    double small_offset_ratio = (tokens > 0) ? ((double)small_offsets / (double)tokens) : 0.0;
    fprintf(stderr,
            "[LZ4-DBG][DECOMP] blocks=%d tokens=%llu literals=%llu matches=%llu small_offsets=%llu small_offset/token=%.4f output_errors=%llu avg_tokens/block=%.2f\n",
            num_blocks,
            tokens,
            literal_bytes,
            match_bytes,
            small_offsets,
            small_offset_ratio,
            output_errors,
            avg_tokens);
}

static int lz4_write_blocks_packed(FILE* fout,
                                   const uint8_t* base,
                                   const uint32_t* offsets,
                                   const uint32_t* sizes,
                                   size_t count) {
    size_t pack_kb = 1024;
    const char* env_pack_kb = getenv("LZ4_GPU_PACK_WRITE_KB");
    if (env_pack_kb && *env_pack_kb) {
        long v = strtol(env_pack_kb, NULL, 10);
        if (v > 64 && v <= 16384) pack_kb = (size_t)v;
    }

    size_t pack_cap = pack_kb * 1024;
    uint8_t* pack = (uint8_t*)malloc(pack_cap);
    size_t fill = 0;

    if (!fout || !base || !offsets || !sizes) return -1;

    if (!pack) {
        for (size_t i = 0; i < count; ++i) {
            size_t len = (size_t)sizes[i];
            if (len == 0) continue;
            if (fwrite(base + offsets[i], 1, len, fout) != len) return -1;
        }
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        size_t len = (size_t)sizes[i];
        const uint8_t* src;
        if (len == 0) continue;

        src = base + offsets[i];
        if (len > pack_cap) {
            if (fill > 0) {
                if (fwrite(pack, 1, fill, fout) != fill) {
                    free(pack);
                    return -1;
                }
                fill = 0;
            }
            if (fwrite(src, 1, len, fout) != len) {
                free(pack);
                return -1;
            }
            continue;
        }

        if (fill + len > pack_cap) {
            if (fwrite(pack, 1, fill, fout) != fill) {
                free(pack);
                return -1;
            }
            fill = 0;
        }

        memcpy(pack + fill, src, len);
        fill += len;
    }

    if (fill > 0) {
        if (fwrite(pack, 1, fill, fout) != fill) {
            free(pack);
            return -1;
        }
    }

    free(pack);
    return 0;
}

static int lz4_write_blocks_direct(FILE* fout,
                                   const uint8_t* base,
                                   const uint32_t* offsets,
                                   const uint32_t* sizes,
                                   size_t count) {
    if (!fout || !base || !offsets || !sizes) return -1;
    for (size_t i = 0; i < count; ++i) {
        size_t len = (size_t)sizes[i];
        if (len == 0) continue;
        if (fwrite(base + offsets[i], 1, len, fout) != len) return -1;
    }
    return 0;
}

static void lz4_set_stream_buffer(FILE* f) {
    char* vbuf;
    if (!f) return;
    vbuf = (char*)malloc(2U * 1024U * 1024U);
    if (!vbuf) return;
    if (setvbuf(f, vbuf, _IOFBF, 2U * 1024U * 1024U) != 0) {
        free(vbuf);
    }
}

int lz4_compress_core(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_kernel pack_kernel,
                    const char* input_path, const char* output_path,
                    size_t block_size, int acceleration, lz4_gpu_workspace_t* ws,
                    timing_t* t, int local_size,
                    int skip_input_upload) {
    const int hash_log = 14;
    cl_int err;
    uint64_t t1, t2;

    struct stat st_buf;
    if (!input_path || stat(input_path, &st_buf) != 0 || st_buf.st_size <= 0) return -1;
    size_t file_size = (size_t)st_buf.st_size;
    t->in_size = (unsigned long)file_size;

    t1 = get_us();
    int num_blocks = (file_size + block_size - 1) / block_size;
    int use_standard_copy = lz4_prefers_standard_copy(queue);
    size_t l_ws = sanitize_local_size(queue, (local_size > 0) ? (size_t)local_size : 1, (size_t)num_blocks);
    size_t worker_count = choose_comp_worker_count(queue, (size_t)num_blocks, l_ws);
    size_t g_ws = round_up_size(worker_count, l_ws);
    cl_uint kernel_num_args = 0;
    int kernel_has_dbg = 0;
    if (clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args), &kernel_num_args, NULL) == CL_SUCCESS) {
        kernel_has_dbg = (kernel_num_args >= 14U);
    }

    uint32_t* h_block_info = malloc(num_blocks * 2 * sizeof(uint32_t));
    uint32_t* h_out_offsets = malloc(num_blocks * sizeof(uint32_t));
    cl_mem dbg_comp_buf = NULL;
    int dbg_comp_requested = lz4_debug_counters_enabled();
    int dbg_comp_enabled = dbg_comp_requested && kernel_has_dbg;
    if (dbg_comp_requested && !kernel_has_dbg) {
        fprintf(stderr, "[LZ4-DBG][COMP] warning: kernel has no debug args, debug counters disabled\n");
    }
    uint32_t single_block_max_out = (uint32_t)(block_size * 1.1 + 64);
    int tableType = (block_size <= 65536) ? 0 : 1;
    for (int i = 0; i < num_blocks; i++) {
        h_block_info[i*2] = i * block_size;
        h_block_info[i*2+1] = (i == num_blocks - 1) ? (file_size - i * block_size) : (uint32_t)block_size;
        h_out_offsets[i] = i * single_block_max_out;
    }
    t->nblk = (unsigned long)num_blocks;
    t->blk_size_bytes = (unsigned long)block_size;

    t1 = get_us();
    ws->in_buf = ensure_buffer_ex(context, ws->in_buf, file_size, &ws->current_in_capacity, CL_MEM_READ_ONLY, !use_standard_copy, &err);
    ws->out_buf = ensure_buffer_ex(context, ws->out_buf, (size_t)num_blocks * single_block_max_out, &ws->current_out_capacity, CL_MEM_READ_WRITE, !use_standard_copy, &err);

    if (!skip_input_upload) {
        FILE* fin = fopen(input_path, "rb"); if (!fin) { free(h_block_info); free(h_out_offsets); return -1; }
        lz4_set_stream_buffer(fin);
        t2 = get_us();
        if (use_standard_copy) {
            uint8_t* h_in = (uint8_t*)malloc(file_size);
            if (!h_in) {
                fprintf(stderr, "[LZ4] malloc input staging buffer failed (%zu bytes)\n", file_size);
                fclose(fin);
                free(h_block_info);
                free(h_out_offsets);
                return -1;
            }
            t1 = get_us();
            if (fread(h_in, 1, file_size, fin) != file_size) {
                fprintf(stderr, "[LZ4] fread input failed for standard-copy path\n");
                free(h_in);
                fclose(fin);
                free(h_block_info);
                free(h_out_offsets);
                return -1;
            }
            t->file_read_us = (unsigned long)(get_us() - t1);
            t1 = get_us();
            if (write_buffer_auto(queue, ws->in_buf, h_in, file_size, 1) != 0) {
                fprintf(stderr, "[LZ4] upload input buffer failed for standard-copy path\n");
                free(h_in);
                fclose(fin);
                free(h_block_info);
                free(h_out_offsets);
                return -1;
            }
            t->data_upload_us = (unsigned long)(get_us() - t1);
            free(h_in);
        } else {
            void* mapped_in = clEnqueueMapBuffer(queue, ws->in_buf, CL_TRUE, CL_MAP_WRITE, 0, file_size, 0, NULL, NULL, &err);
            if (err == CL_SUCCESS && mapped_in) {
                t1 = get_us();
                (void)fread(mapped_in, 1, file_size, fin);
                t->file_read_us = (unsigned long)(get_us() - t1);
                clEnqueueUnmapMemObject(queue, ws->in_buf, mapped_in, 0, NULL, NULL);
            } else {
                fprintf(stderr, "[LZ4] map input buffer failed: %d\n", err);
                fclose(fin);
                free(h_block_info);
                free(h_out_offsets);
                return -1;
            }
            t->data_upload_us = 0;
        }
        fclose(fin);
        if (!use_standard_copy) {
            t->file_read_us = (unsigned long)(get_us() - t2);
        }
    } else {
        t->file_read_us = 0;
        t->data_upload_us = 0;
    }

    size_t out_offsets_cap = num_blocks * sizeof(uint32_t);
    size_t block_info_cap = num_blocks * 2 * sizeof(uint32_t);

    ws->out_offsets_buf = ensure_buffer_ex(context, ws->out_offsets_buf, out_offsets_cap, &ws->current_out_offsets_capacity, CL_MEM_READ_ONLY, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->out_offsets_buf) {
        fprintf(stderr, "[LZ4] allocate out_offsets buffer failed: %d\n", err);
        free(h_block_info);
        free(h_out_offsets);
        return -1;
    }

    ws->block_info_buf = ensure_buffer_ex(context, ws->block_info_buf, block_info_cap, &ws->current_blocks_capacity, CL_MEM_READ_ONLY, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->block_info_buf) {
        fprintf(stderr, "[LZ4] allocate block_info buffer failed: %d\n", err);
        free(h_block_info);
        free(h_out_offsets);
        return -1;
    }

    if (!skip_input_upload) {
        if (write_buffer_auto(queue, ws->out_offsets_buf, h_out_offsets, out_offsets_cap, use_standard_copy) != 0 ||
            write_buffer_auto(queue, ws->block_info_buf, h_block_info, block_info_cap, use_standard_copy) != 0) {
            fprintf(stderr, "[LZ4] upload block metadata buffers failed\n");
            free(h_block_info);
            free(h_out_offsets);
            return -1;
        }
    }

    ws->output_size_buf = ensure_buffer_ex(context, ws->output_size_buf, num_blocks * sizeof(uint32_t), &ws->current_osize_capacity, CL_MEM_READ_WRITE, !use_standard_copy, &err);
    if (dbg_comp_enabled) {
        size_t dbg_comp_bytes = (size_t)num_blocks * LZ4_DBG_COMP_N * sizeof(uint32_t);
        dbg_comp_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dbg_comp_bytes, NULL, &err);
        if (err != CL_SUCCESS || !dbg_comp_buf || zero_buffer(queue, dbg_comp_buf, dbg_comp_bytes) != 0) {
            if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
            dbg_comp_buf = NULL;
            dbg_comp_enabled = 0;
            fprintf(stderr, "[LZ4-DBG][COMP] warning: failed to enable debug counters, continuing without them\n");
        }
    }
    size_t dict_entries_per_worker = (size_t)1U << (tableType == 0 ? (hash_log + 1) : hash_log);
    size_t dict_size_per_worker = dict_entries_per_worker * sizeof(cl_uint);  /* 32-bit compact entries */
    size_t prev_dict_capacity = ws->current_dict_capacity;
    ws->dict_buf = ensure_buffer_ex(context, ws->dict_buf, g_ws * dict_size_per_worker, &ws->current_dict_capacity, CL_MEM_READ_WRITE, !use_standard_copy, &err);
    if (ws->current_dict_capacity != prev_dict_capacity) {
        (void)zero_buffer(queue, ws->dict_buf, ws->current_dict_capacity);
    }
    if (ws->comp_epoch_base == 0) ws->comp_epoch_base = 1;
    {
        uint32_t blocks_per_worker = (uint32_t)(((size_t)num_blocks + g_ws - 1) / g_ws);
        uint32_t epochs_needed = blocks_per_worker + 2U;
        if (epochs_needed >= UINT32_MAX - 1024U) epochs_needed = 1024U;
        /* 8-bit epoch in kernel: clear dict when low byte would wrap to avoid stale collisions */
        uint32_t cur_low = ws->comp_epoch_base & 0xFF;
        uint32_t end_low = (ws->comp_epoch_base + epochs_needed) & 0xFF;
        int wraps_8bit = (end_low <= cur_low) || (ws->comp_epoch_base > (uint32_t)(UINT32_MAX - epochs_needed));
        if (wraps_8bit) {
            (void)zero_buffer(queue, ws->dict_buf, ws->current_dict_capacity);
            ws->comp_epoch_base = 1;
        }
    }
    uint32_t epoch_base = ws->comp_epoch_base;
    ws->comp_epoch_base += (uint32_t)(((size_t)num_blocks + g_ws - 1) / g_ws) + 2U;
    t->buffer_alloc_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    int globalIndexBase = 0;
    int inputSize = (int)file_size;
    cl_mem dbg_comp_arg = dbg_comp_enabled ? dbg_comp_buf : ws->output_size_buf;
    uint32_t dbg_comp_flag = dbg_comp_enabled ? 1U : 0U;

    if (!skip_input_upload) {
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ws->in_buf);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &ws->out_buf);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &ws->output_size_buf);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &ws->block_info_buf);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &ws->out_offsets_buf);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &num_blocks);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &inputSize);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &tableType);
        err |= clSetKernelArg(kernel, 8, sizeof(int), &acceleration);
        err |= clSetKernelArg(kernel, 9, sizeof(int), &globalIndexBase);
        err |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &ws->dict_buf);
        if (kernel_has_dbg) {
            err |= clSetKernelArg(kernel, 12, sizeof(cl_mem), &dbg_comp_arg);
            err |= clSetKernelArg(kernel, 13, sizeof(uint32_t), &dbg_comp_flag);
        }
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[LZ4] set compress kernel args failed: %d\n", err);
            if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
            free(h_block_info);
            free(h_out_offsets);
            return -1;
        }
    }
    err = clSetKernelArg(kernel, 11, sizeof(uint32_t), &epoch_base);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[LZ4] set epoch_base kernel arg failed: %d\n", err);
        if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
        free(h_block_info);
        free(h_out_offsets);
        return -1;
    }

    t1 = get_us();
    t->global_size = (unsigned long)g_ws;
    t->local_size = (unsigned long)l_ws;
    cl_event ev;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &g_ws, &l_ws, 0, NULL, &ev);
    clWaitForEvents(1, &ev);
    t->kernel_exec_us = (unsigned long)(get_us() - t1);
    t->algo_config = hash_log;

    t1 = get_us();
    uint32_t* h_osizes = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    if (!h_osizes || read_buffer_auto(queue, ws->output_size_buf, h_osizes, num_blocks * sizeof(uint32_t), use_standard_copy) != 0) {
        fprintf(stderr, "[LZ4] read output_size buffer failed\n");
        if (h_osizes) free(h_osizes);
        if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
        free(h_block_info); free(h_out_offsets);
        return -1;
    }

    size_t total_compressed_size = 0;
    for (int i=0; i<num_blocks; i++) total_compressed_size += h_osizes[i];
    t->out_size = total_compressed_size;

    t->download_total_us = (unsigned long)(get_us() - t1);

    if (dbg_comp_enabled && dbg_comp_buf) {
        size_t dbg_comp_bytes = (size_t)num_blocks * LZ4_DBG_COMP_N * sizeof(uint32_t);
        uint32_t* dbg_comp_stats = (uint32_t*)malloc(dbg_comp_bytes);
        if (dbg_comp_stats) {
            if (clEnqueueReadBuffer(queue, dbg_comp_buf, CL_TRUE, 0, dbg_comp_bytes, dbg_comp_stats, 0, NULL, NULL) == CL_SUCCESS) {
                lz4_print_comp_debug_stats(dbg_comp_stats, num_blocks);
            }
            free(dbg_comp_stats);
        }
    }

    t1 = get_us();
    FILE* fout = fopen(output_path, "wb");
    if (fout) {
        lz4_set_stream_buffer(fout);
        uint32_t magic = 0x184D2204;
        fwrite(&magic, 1, 4, fout);
        fwrite(&num_blocks, 1, 4, fout);
        uint32_t bsize_u32 = (uint32_t)block_size;
        fwrite(&bsize_u32, 1, 4, fout);
        fwrite(h_osizes, 1, num_blocks * 4, fout);

        size_t total_out_read = (size_t)num_blocks * single_block_max_out;
        int use_compaction = lz4_should_use_device_compaction(total_compressed_size, total_out_read, (size_t)num_blocks, pack_kernel);
        if (use_compaction) {
            uint32_t* h_packed_offsets = (uint32_t*)malloc((size_t)num_blocks * sizeof(uint32_t));
            uint8_t* h_packed_out = NULL;
            if (!h_packed_offsets) {
                fclose(fout);
                if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                free(h_block_info); free(h_osizes); free(h_out_offsets);
                return -1;
            }

            {
                size_t packed_off = 0;
                for (int i = 0; i < num_blocks; ++i) {
                    h_packed_offsets[i] = (uint32_t)packed_off;
                    packed_off += (size_t)h_osizes[i];
                }
            }

            ws->packed_offsets_buf = ensure_buffer_ex(context, ws->packed_offsets_buf,
                                                      (size_t)num_blocks * sizeof(uint32_t),
                                                      &ws->current_packed_offsets_capacity,
                                                      CL_MEM_READ_ONLY, !use_standard_copy, &err);
            ws->packed_out_buf = ensure_buffer_ex(context, ws->packed_out_buf,
                                                  total_compressed_size ? total_compressed_size : 1,
                                                  &ws->current_packed_out_capacity,
                                                  CL_MEM_READ_WRITE, !use_standard_copy, &err);
            if (err != CL_SUCCESS || !ws->packed_offsets_buf || !ws->packed_out_buf ||
                write_buffer_auto(queue, ws->packed_offsets_buf, h_packed_offsets,
                                  (size_t)num_blocks * sizeof(uint32_t), use_standard_copy) != 0) {
                free(h_packed_offsets);
                fclose(fout);
                if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                free(h_block_info); free(h_osizes); free(h_out_offsets);
                return -1;
            }

            {
                size_t pack_lws = sanitize_local_size(queue, l_ws, (size_t)num_blocks);
                size_t pack_gws = round_up_size((size_t)num_blocks, pack_lws);
                cl_event pack_ev = NULL;
                uint32_t total_blocks_u32 = (uint32_t)num_blocks;

                err  = clSetKernelArg(pack_kernel, 0, sizeof(cl_mem), &ws->out_buf);
                err |= clSetKernelArg(pack_kernel, 1, sizeof(cl_mem), &ws->packed_out_buf);
                err |= clSetKernelArg(pack_kernel, 2, sizeof(cl_mem), &ws->out_offsets_buf);
                err |= clSetKernelArg(pack_kernel, 3, sizeof(cl_mem), &ws->packed_offsets_buf);
                err |= clSetKernelArg(pack_kernel, 4, sizeof(cl_mem), &ws->output_size_buf);
                err |= clSetKernelArg(pack_kernel, 5, sizeof(uint32_t), &total_blocks_u32);
                if (err != CL_SUCCESS) {
                    free(h_packed_offsets);
                    fclose(fout);
                    if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                    free(h_block_info); free(h_osizes); free(h_out_offsets);
                    return -1;
                }

                t1 = get_us();
                err = clEnqueueNDRangeKernel(queue, pack_kernel, 1, NULL, &pack_gws, &pack_lws, 0, NULL, &pack_ev);
                if (err != CL_SUCCESS || !pack_ev) {
                    free(h_packed_offsets);
                    fclose(fout);
                    if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                    free(h_block_info); free(h_osizes); free(h_out_offsets);
                    return -1;
                }
                clWaitForEvents(1, &pack_ev);
                clReleaseEvent(pack_ev);
                t->kernel_exec_us += (unsigned long)(get_us() - t1);
            }

            if (total_compressed_size > 0) {
                h_packed_out = (uint8_t*)malloc(total_compressed_size);
                if (!h_packed_out) {
                    free(h_packed_offsets);
                    fclose(fout);
                    if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                    free(h_block_info); free(h_osizes); free(h_out_offsets);
                    return -1;
                }
                {
                    uint64_t t_down0 = get_us();
                    if (read_buffer_auto(queue, ws->packed_out_buf, h_packed_out, total_compressed_size, use_standard_copy) != 0) {
                        fprintf(stderr, "[LZ4] read compacted payload buffer failed\n");
                        free(h_packed_out);
                        free(h_packed_offsets);
                        fclose(fout);
                        if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                        free(h_block_info); free(h_osizes); free(h_out_offsets);
                        return -1;
                    }
                    t->download_total_us = (unsigned long)(get_us() - t_down0);
                }
                if (fwrite(h_packed_out, 1, total_compressed_size, fout) != total_compressed_size) {
                    free(h_packed_out);
                    free(h_packed_offsets);
                    fclose(fout);
                    if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                    free(h_block_info); free(h_osizes); free(h_out_offsets);
                    return -1;
                }
                free(h_packed_out);
            }
            free(h_packed_offsets);
        } else {
            uint8_t* h_out = (uint8_t*)malloc(total_out_read);
            if (h_out) {
                uint64_t t_down0 = get_us();
                if (read_buffer_auto(queue, ws->out_buf, h_out, total_out_read, use_standard_copy) != 0) {
                    fprintf(stderr, "[LZ4] read compressed payload buffer failed\n");
                    free(h_out);
                    fclose(fout);
                    if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                    free(h_block_info); free(h_osizes); free(h_out_offsets);
                    return -1;
                }
                t->download_total_us = (unsigned long)(get_us() - t_down0);
                if (lz4_write_blocks_packed(fout, h_out, h_out_offsets, h_osizes, (size_t)num_blocks) != 0) {
                    free(h_out);
                    fclose(fout);
                    if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                    free(h_block_info); free(h_osizes); free(h_out_offsets);
                    return -1;
                }
                free(h_out);
            }
        }
        fclose(fout);
    }
    t->file_write_us = (unsigned long)(get_us() - t1);

    if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
    free(h_block_info); free(h_osizes); free(h_out_offsets);
    return 0;
}

int lz4_decompress_core(cl_context context, cl_command_queue queue, cl_kernel kernel,
                      const char* input_path, const char* output_path,
                      lz4_gpu_workspace_t* ws, timing_t* t, int local_size) {
    cl_int err;
    uint64_t t1;

    t1 = get_us();
    int use_standard_copy = lz4_prefers_standard_copy(queue);
    FILE* fin = fopen(input_path, "rb"); if (!fin) return -1;
    lz4_set_stream_buffer(fin);
    uint32_t magic, num_blocks, block_size_val;
    if (fread(&magic, 1, 4, fin) != 4 || magic != 0x184D2204) { fclose(fin); return -1; }
    if (fread(&num_blocks, 1, 4, fin) != 4) { fclose(fin); return -1; }
    if (fread(&block_size_val, 1, 4, fin) != 4) { fclose(fin); return -1; }

    uint32_t* h_comp_sizes = malloc(num_blocks * sizeof(uint32_t));
    fread(h_comp_sizes, 1, num_blocks * 4, fin);

    size_t header_size = 12 + num_blocks * 4;
    fseek(fin, 0, SEEK_END);
    size_t file_size = ftell(fin);
    size_t data_size = file_size - header_size;
    fseek(fin, header_size, SEEK_SET);
    t->file_read_us = 0;

    t1 = get_us();
    uint32_t* h_comp_offsets = malloc(num_blocks * sizeof(uint32_t));
    uint32_t* h_comp_sizes_32 = malloc(num_blocks * sizeof(uint32_t));
    uint32_t* h_out_offsets = malloc(num_blocks * sizeof(uint32_t));
    uint32_t* h_max_out_sizes = malloc(num_blocks * sizeof(uint32_t));

    uint64_t curr_in = 0;
    uint64_t curr_out = 0;
    uint64_t block_max = block_size_val;

    for (uint32_t i = 0; i < num_blocks; i++) {
        if (curr_in > UINT32_MAX || curr_out > UINT32_MAX || block_max > UINT32_MAX) {
            free(h_comp_sizes);
            free(h_comp_offsets);
            free(h_comp_sizes_32);
            free(h_out_offsets);
            free(h_max_out_sizes);
            return -1;
        }
        h_comp_offsets[i] = (uint32_t)curr_in;
        h_comp_sizes_32[i] = h_comp_sizes[i];
        h_out_offsets[i] = (uint32_t)curr_out;
        h_max_out_sizes[i] = (uint32_t)block_max;
        curr_in += h_comp_sizes[i];
        curr_out += block_max;
    }
    {
        size_t l_ws = sanitize_local_size(queue, (local_size > 0) ? (size_t)local_size : 1, (size_t)num_blocks);
        t->local_size = (unsigned long)l_ws;
    }
    t->nblk = (unsigned long)num_blocks;
    t->blk_size_bytes = (unsigned long)block_size_val;

    t1 = get_us();
    ws->in_buf = ensure_buffer_ex(context, ws->in_buf, data_size, &ws->current_in_capacity, CL_MEM_READ_ONLY, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->in_buf) {
        fclose(fin);
        free(h_comp_sizes);
        free(h_comp_offsets);
        free(h_comp_sizes_32);
        free(h_out_offsets);
        free(h_max_out_sizes);
        return -1;
    }
    ws->out_buf = ensure_buffer_ex(context, ws->out_buf, num_blocks * block_max, &ws->current_out_capacity, CL_MEM_READ_WRITE, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->out_buf) {
        fclose(fin);
        free(h_comp_sizes);
        free(h_comp_offsets);
        free(h_comp_sizes_32);
        free(h_out_offsets);
        free(h_max_out_sizes);
        return -1;
    }

    {
        uint64_t file_read_t0 = get_us();
    if (use_standard_copy) {
        uint8_t* h_in = (uint8_t*)malloc(data_size);
        if (!h_in) {
            fclose(fin);
            free(h_comp_sizes);
            free(h_comp_offsets);
            free(h_comp_sizes_32);
            free(h_out_offsets);
            free(h_max_out_sizes);
            return -1;
        }
        if (fread(h_in, 1, data_size, fin) != data_size ||
            write_buffer_auto(queue, ws->in_buf, h_in, data_size, 1) != 0) {
            fprintf(stderr, "[LZ4] read/upload compressed input failed for standard-copy decompress path\n");
            free(h_in);
            fclose(fin);
            free(h_comp_sizes);
            free(h_comp_offsets);
            free(h_comp_sizes_32);
            free(h_out_offsets);
            free(h_max_out_sizes);
            return -1;
        }
        free(h_in);
    } else {
        void* mapped_in = clEnqueueMapBuffer(queue, ws->in_buf, CL_TRUE, CL_MAP_WRITE, 0, data_size, 0, NULL, NULL, &err);
        if (err == CL_SUCCESS && mapped_in) {
            size_t nr = fread(mapped_in, 1, data_size, fin);
            if (nr != data_size) {
                fprintf(stderr, "[LZ4] fread compressed input failed for mapped decompress path\n");
                fclose(fin);
                clEnqueueUnmapMemObject(queue, ws->in_buf, mapped_in, 0, NULL, NULL);
                free(h_comp_sizes);
                free(h_comp_offsets);
                free(h_comp_sizes_32);
                free(h_out_offsets);
                free(h_max_out_sizes);
                return -1;
            }
            clEnqueueUnmapMemObject(queue, ws->in_buf, mapped_in, 0, NULL, NULL);
        } else {
            fprintf(stderr, "[LZ4] map compressed input buffer failed: %d\n", err);
            fclose(fin);
            free(h_comp_sizes);
            free(h_comp_offsets);
            free(h_comp_sizes_32);
            free(h_out_offsets);
            free(h_max_out_sizes);
            return -1;
        }
    }
    fclose(fin);
    t->file_read_us = (unsigned long)(get_us() - file_read_t0);
    }

    ws->decomp_comp_off_buf = ensure_buffer_ex(context, ws->decomp_comp_off_buf, num_blocks * 4,
                                            &ws->current_decomp_comp_off_capacity, CL_MEM_READ_ONLY, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->decomp_comp_off_buf) {
        free(h_comp_sizes);
        free(h_comp_offsets);
        free(h_comp_sizes_32);
        free(h_out_offsets);
        free(h_max_out_sizes);
        return -1;
    }
    ws->decomp_comp_size_buf = ensure_buffer_ex(context, ws->decomp_comp_size_buf, num_blocks * 4,
                                             &ws->current_decomp_comp_size_capacity, CL_MEM_READ_ONLY, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->decomp_comp_size_buf) {
        free(h_comp_sizes);
        free(h_comp_offsets);
        free(h_comp_sizes_32);
        free(h_out_offsets);
        free(h_max_out_sizes);
        return -1;
    }
    ws->decomp_out_off_buf = ensure_buffer_ex(context, ws->decomp_out_off_buf, num_blocks * 4,
                                           &ws->current_decomp_out_off_capacity, CL_MEM_READ_ONLY, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->decomp_out_off_buf) {
        free(h_comp_sizes);
        free(h_comp_offsets);
        free(h_comp_sizes_32);
        free(h_out_offsets);
        free(h_max_out_sizes);
        return -1;
    }
    ws->decomp_max_out_buf = ensure_buffer_ex(context, ws->decomp_max_out_buf, num_blocks * 4,
                                           &ws->current_decomp_max_out_capacity, CL_MEM_READ_ONLY, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->decomp_max_out_buf) {
        free(h_comp_sizes);
        free(h_comp_offsets);
        free(h_comp_sizes_32);
        free(h_out_offsets);
        free(h_max_out_sizes);
        return -1;
    }
    ws->decomp_sizes_out_buf = ensure_buffer_ex(context, ws->decomp_sizes_out_buf, num_blocks * 4,
                                             &ws->current_decomp_sizes_out_capacity, CL_MEM_READ_WRITE, !use_standard_copy, &err);
    if (err != CL_SUCCESS || !ws->decomp_sizes_out_buf) {
        free(h_comp_sizes);
        free(h_comp_offsets);
        free(h_comp_sizes_32);
        free(h_out_offsets);
        free(h_max_out_sizes);
        return -1;
    }

    if (write_buffer_auto(queue, ws->decomp_comp_off_buf, h_comp_offsets, num_blocks * 4, use_standard_copy) != 0 ||
        write_buffer_auto(queue, ws->decomp_comp_size_buf, h_comp_sizes_32, num_blocks * 4, use_standard_copy) != 0 ||
        write_buffer_auto(queue, ws->decomp_out_off_buf, h_out_offsets, num_blocks * 4, use_standard_copy) != 0 ||
        write_buffer_auto(queue, ws->decomp_max_out_buf, h_max_out_sizes, num_blocks * 4, use_standard_copy) != 0) {
        fprintf(stderr, "[LZ4] upload decompress metadata buffers failed\n");
        free(h_comp_sizes);
        free(h_comp_offsets);
        free(h_comp_sizes_32);
        free(h_out_offsets);
        free(h_max_out_sizes);
        return -1;
    }
    t->buffer_alloc_us = (unsigned long)(get_us() - t1);

    t->data_upload_us = use_standard_copy ? t->data_upload_us : 0;

    t1 = get_us();
    cl_mem dbg_dec_buf = NULL;
    cl_uint kernel_num_args = 0;
    int kernel_has_dbg = 0;
    if (clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args), &kernel_num_args, NULL) == CL_SUCCESS) {
        kernel_has_dbg = (kernel_num_args >= 10U);
    }
    int dbg_dec_requested = lz4_debug_counters_enabled();
    int dbg_dec_enabled = dbg_dec_requested && kernel_has_dbg;
    if (dbg_dec_requested && !kernel_has_dbg) {
        fprintf(stderr, "[LZ4-DBG][DECOMP] warning: kernel has no debug args, debug counters disabled\n");
    }
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &ws->in_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ws->out_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &ws->decomp_comp_off_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &ws->decomp_comp_size_buf);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &ws->decomp_out_off_buf);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &ws->decomp_max_out_buf);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &ws->decomp_sizes_out_buf);
    clSetKernelArg(kernel, 7, sizeof(uint32_t), &num_blocks);

    if (dbg_dec_enabled) {
        size_t dbg_dec_bytes = (size_t)num_blocks * LZ4_DBG_DEC_N * sizeof(uint32_t);
        dbg_dec_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dbg_dec_bytes, NULL, &err);
        if (err != CL_SUCCESS || !dbg_dec_buf || zero_buffer(queue, dbg_dec_buf, dbg_dec_bytes) != 0) {
            if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
            dbg_dec_buf = NULL;
            dbg_dec_enabled = 0;
            fprintf(stderr, "[LZ4-DBG][DECOMP] warning: failed to enable debug counters, continuing without them\n");
        }
    }

    if (kernel_has_dbg) {
        cl_mem dbg_dec_arg = dbg_dec_enabled ? dbg_dec_buf : ws->decomp_sizes_out_buf;
        uint32_t dbg_dec_flag = dbg_dec_enabled ? 1U : 0U;
        clSetKernelArg(kernel, 8, sizeof(cl_mem), &dbg_dec_arg);
        clSetKernelArg(kernel, 9, sizeof(uint32_t), &dbg_dec_flag);
    }

    size_t l_ws = (size_t)t->local_size;
    if (l_ws == 0) {
        l_ws = sanitize_local_size(queue, (local_size > 0) ? (size_t)local_size : 1, (size_t)num_blocks);
    }
    size_t worker_count = choose_decomp_worker_count(queue, (size_t)num_blocks, l_ws);
    size_t g_ws = round_up_size(worker_count, l_ws);
    t->global_size = (unsigned long)g_ws;
    t->local_size = (unsigned long)l_ws;
    cl_event ev;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &g_ws, &l_ws, 0, NULL, &ev);
    clWaitForEvents(1, &ev);
    t->kernel_exec_us = (unsigned long)(get_us() - t1);
    t->algo_config = 0; // Not applicable for LZ4 decompress

    t1 = get_us();
    uint32_t* h_final_sizes = (uint32_t*)malloc(num_blocks * 4);
    if (!h_final_sizes || read_buffer_auto(queue, ws->decomp_sizes_out_buf, h_final_sizes, num_blocks * 4, use_standard_copy) != 0) {
        fprintf(stderr, "[LZ4] read decompressed block sizes failed\n");
        if (h_final_sizes) free(h_final_sizes);
        if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
        free(h_comp_sizes); free(h_comp_offsets); free(h_comp_sizes_32);
        free(h_out_offsets); free(h_max_out_sizes);
        return -1;
    }

    size_t total_decomp_sz = 0;
    for (uint32_t i = 0; i < num_blocks; i++) total_decomp_sz += h_final_sizes[i];
    t->out_size = total_decomp_sz;

    t->download_total_us = (unsigned long)(get_us() - t1);

    if (dbg_dec_enabled && dbg_dec_buf) {
        size_t dbg_dec_bytes = (size_t)num_blocks * LZ4_DBG_DEC_N * sizeof(uint32_t);
        uint32_t* dbg_dec_stats = (uint32_t*)malloc(dbg_dec_bytes);
        if (dbg_dec_stats) {
            if (clEnqueueReadBuffer(queue, dbg_dec_buf, CL_TRUE, 0, dbg_dec_bytes, dbg_dec_stats, 0, NULL, NULL) == CL_SUCCESS) {
                lz4_print_dec_debug_stats(dbg_dec_stats, (int)num_blocks);
            }
            free(dbg_dec_stats);
        }
    }

    t1 = get_us();
    FILE* fout = fopen(output_path, "wb");
    if (fout) {
        lz4_set_stream_buffer(fout);
        uint8_t* h_out = (uint8_t*)malloc(total_decomp_sz);
        if (h_out) {
            uint64_t t_down0 = get_us();
            if (read_buffer_auto(queue, ws->out_buf, h_out, total_decomp_sz, use_standard_copy) != 0) {
                fprintf(stderr, "[LZ4] read decompressed payload failed\n");
                free(h_out);
                fclose(fout);
                if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
                free(h_comp_sizes); free(h_comp_offsets); free(h_comp_sizes_32);
                free(h_out_offsets); free(h_max_out_sizes); free(h_final_sizes);
                return -1;
            }
            t->download_total_us = (unsigned long)(get_us() - t_down0);
            if (fwrite(h_out, 1, total_decomp_sz, fout) != total_decomp_sz) {
                free(h_out);
                fclose(fout);
                if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
                free(h_comp_sizes); free(h_comp_offsets); free(h_comp_sizes_32);
                free(h_out_offsets); free(h_max_out_sizes); free(h_final_sizes);
                return -1;
            }
            free(h_out);
        }
        fclose(fout);
    }
    t->file_write_us = (unsigned long)(get_us() - t1);

    if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
    free(h_comp_sizes); free(h_comp_offsets); free(h_comp_sizes_32);
    free(h_out_offsets); free(h_max_out_sizes); free(h_final_sizes);
    return 0;
}

static char* read_file_bin_local(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = malloc(s); if (!buf) { fclose(f); return NULL; }
    if (fread(buf,1,s,f) != (size_t)s) { free(buf); fclose(f); return NULL; }
    if (out_len) *out_len = (size_t)s;
    fclose(f);
    return buf;
}

static int lz4_get_executable_path(char* out, size_t outlen) {
    if (!out || outlen == 0) return -1;
#if defined(_WIN32) || defined(_WIN64)
    {
        DWORD n = GetModuleFileNameA(NULL, out, (DWORD)outlen);
        if (n == 0 || n >= outlen) return -1;
        out[n] = '\0';
        return 0;
    }
#else
    {
        ssize_t n = readlink("/proc/self/exe", out, outlen - 1);
        if (n <= 0) return -1;
        out[n] = '\0';
        return 0;
    }
#endif
}

static int lz4_find_file_path(const char* name, char* out, size_t outlen) {
    char path[PATH_MAX];
    char base[PATH_MAX];
    const char* env = getenv("LZ4_GPU_DIR");

    if (!name || !out || outlen == 0) return -1;

    if (env && env[0]) {
        snprintf(path, sizeof(path), "%s/%s", env, name);
        if (access(path, R_OK) == 0) {
            strncpy(out, path, outlen - 1);
            out[outlen - 1] = '\0';
            return 0;
        }
    }

    {
        char exe_path[PATH_MAX] = {0};
        if (lz4_get_executable_path(exe_path, sizeof(exe_path)) == 0) {
            char* slash = strrchr(exe_path,
#if defined(_WIN32) || defined(_WIN64)
                                  '\\'
#else
                                  '/'
#endif
            );
            if (!slash) slash = strrchr(exe_path, '/');
            if (slash) {
                *slash = '\0';
                snprintf(path, sizeof(path), "%s/../lz4_gpu/%s", exe_path, name);
                if (access(path, R_OK) == 0) {
                    strncpy(out, path, outlen - 1);
                    out[outlen - 1] = '\0';
                    return 0;
                }
                snprintf(path, sizeof(path), "%s/%s", exe_path, name);
                if (access(path, R_OK) == 0) {
                    strncpy(out, path, outlen - 1);
                    out[outlen - 1] = '\0';
                    return 0;
                }
            }
        }
    }

    if (getcwd(base, sizeof(base)) != NULL) {
        snprintf(path, sizeof(path), "%s/%s", base, name);
        if (access(path, R_OK) == 0) {
            strncpy(out, path, outlen - 1);
            out[outlen - 1] = '\0';
            return 0;
        }
    }

    if (access(name, R_OK) == 0) {
        strncpy(out, name, outlen - 1);
        out[outlen - 1] = '\0';
        return 0;
    }

    return -1;
}

cl_program lz4_load_program(cl_context context, cl_device_id device) {
    const int hash_log = 14;
    int dbg_enabled = lz4_debug_counters_enabled();

    if (!dbg_enabled && !(getenv("LZ4_GPU_NO_CLBIN") && strcmp(getenv("LZ4_GPU_NO_CLBIN"), "1") == 0)) {
        char bin_name[128];
        char resolved_path[PATH_MAX];
        snprintf(bin_name, sizeof(bin_name), "lz4_gpu_%d.clbin", hash_log);
        if (lz4_find_file_path(bin_name, resolved_path, sizeof(resolved_path)) == 0) {
            size_t sz = 0;
            char* bin = read_file_bin_local(resolved_path, &sz);
            cl_int status, err;
            if (bin) {
                cl_program prog = clCreateProgramWithBinary(context, 1, &device, &sz, (const unsigned char**)&bin, &status, &err);
                free(bin);
                if (err == CL_SUCCESS && status == CL_SUCCESS) {
                    if (clBuildProgram(prog, 1, &device, NULL, NULL, NULL) == CL_SUCCESS) {
                        cl_int kernel_err = CL_SUCCESS;
                        cl_kernel kcomp = clCreateKernel(prog, "lz4_compress_block", &kernel_err);
                        cl_kernel kdec = NULL;
                        cl_kernel kpack = NULL;
                        if (kernel_err == CL_SUCCESS && kcomp) {
                            kdec = clCreateKernel(prog, "lz4_decompress_blocks", &kernel_err);
                        }
                        if (kernel_err == CL_SUCCESS && kdec) {
                            kpack = clCreateKernel(prog, "lz4_pack_blocks", &kernel_err);
                        }
                        if (kcomp) clReleaseKernel(kcomp);
                        if (kdec) clReleaseKernel(kdec);
                        if (kpack) clReleaseKernel(kpack);
                        if (kernel_err == CL_SUCCESS && kpack) {
                            return prog;
                        }
                    }
                    clReleaseProgram(prog);
                }
            }
        }
    }

    // Fallback: load from source
    {
        char resolved_src[PATH_MAX];
        const char* source_name = dbg_enabled ? "lz4_gpu_debug.cl" : "lz4_gpu.cl";
        if (lz4_find_file_path(source_name, resolved_src, sizeof(resolved_src)) != 0) {
            fprintf(stderr, "failed to locate OpenCL source: %s\n", source_name);
            return NULL;
        }
        FILE* f = fopen(resolved_src, "rb");
        if (!f) return NULL;
        fseek(f, 0, SEEK_END); size_t s_sz = ftell(f); fseek(f, 0, SEEK_SET);
        char* src = malloc(s_sz + 1); fread(src, 1, s_sz, f); src[s_sz] = 0; fclose(f);
        cl_int err;
        cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&src, &s_sz, &err);
        free(src);
        if (err != CL_SUCCESS || prog == NULL) {
            fprintf(stderr, "clCreateProgramWithSource failed: err=%d\n", err);
            return NULL;
        }
        char flags[192];
        snprintf(flags, sizeof(flags), "-I. -I./lz4_gpu -I../lz4_gpu -I.. -DLZ4_HASHLOG=%d -DLZ4_GPU_DEBUG_COUNTERS_RUNTIME=%d", hash_log, dbg_enabled ? 1 : 0);
        err = clBuildProgram(prog, 1, &device, flags, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0;
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            if (log_sz > 0) {
                char* log = (char*)malloc(log_sz + 1);
                if (log) {
                    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
                    log[log_sz] = '\0';
                    fprintf(stderr, "Build Error (err=%d, log_sz=%zu): %s\n", err, log_sz, log);
                    free(log);
                } else {
                    fprintf(stderr, "Build Error (err=%d, log_sz=%zu): <oom>\n", err, log_sz);
                }
            } else {
                fprintf(stderr, "Build Error (err=%d, log_sz=0)\n", err);
            }
            return NULL;
        }
        return prog;
    }
}
