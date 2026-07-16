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
#include "lz4_gpu_debug.h"
#include "lz4_gpu_utils.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int lz4_write_blocks_packed(FILE* fout,
                                   const uint8_t* base,
                                   const uint32_t* offsets,
                                   const uint32_t* sizes,
                                   size_t count);
static int lz4_env_flag_value(const char* name, int* is_set);

lz4_gpu_debug_config_t lz4_gpu_get_debug_config(void) {
    lz4_gpu_debug_config_t cfg;
    int debug_is_set = 0;
    int limit_is_set = 0;
    cfg.enabled = 0;
    cfg.block_limit = 0;

    cfg.enabled = lz4_env_flag_value("LZ4_GPU_DEBUG", &debug_is_set);
    if (!debug_is_set) cfg.enabled = 0;

    {
        const char* env = getenv("LZ4_GPU_DEBUG_BLOCK_LIMIT");
        if (env && *env) {
            char* end = NULL;
            unsigned long parsed = strtoul(env, &end, 10);
            if (end != env && *end == '\0') {
                cfg.block_limit = (int)parsed;
                limit_is_set = 1;
            }
        }
    }

    if (cfg.block_limit > 0 && limit_is_set && !debug_is_set) {
        cfg.enabled = 1;
    }
    return cfg;
}

void lz4_gpu_workspace_init(lz4_gpu_workspace_t* ws) {
    memset(ws, 0, sizeof(*ws));
    ws->comp_epoch_base = 1;
}

void lz4_gpu_workspace_free(lz4_gpu_workspace_t* ws) {
    if (ws->in_buf) clReleaseMemObject(ws->in_buf);
    if (ws->comp_in_buf) clReleaseMemObject(ws->comp_in_buf);
    if (ws->out_buf) clReleaseMemObject(ws->out_buf);
    if (ws->block_info_buf) clReleaseMemObject(ws->block_info_buf);
    if (ws->out_offsets_buf) clReleaseMemObject(ws->out_offsets_buf);
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

static int lz4_chunked_output_default_enabled(void) {
#if defined(_WIN32) || defined(_WIN64)
    return 1;
#else
    return 0;
#endif
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
        if (ferr == CL_SUCCESS) return 0;
    }
#endif
    {
        cl_int err;
        void* p = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !p) return -1;
        memset(p, 0, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, p, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
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

static size_t choose_comp_worker_count(cl_command_queue queue, size_t num_blocks, size_t local_size) {
    if (num_blocks == 0) return 1;

    cl_device_id qdev = NULL;
    cl_uint cu = 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    }
    if (cu == 0) cu = 1;

    /* Fixed compression launch ceiling kept from the current baseline: 24 lanes per CU. */
    size_t target = (size_t)cu * 24U;
    if (target < local_size) target = local_size;
    if (target > num_blocks) target = num_blocks;
    if (target == 0) target = 1;
    return target;
}

static int lz4_sanitize_hash_log(int hash_log) {
    if (hash_log < 11) return 11;
    if (hash_log > 15) return 15;
    return hash_log;
}

static void lz4_effective_dict_mode_for_block(size_t block_size, int* dict_clear, int* dict_entry_bits) {
    int clear = 0;
    int entry_bits = 32;

    if (block_size <= 64U * 1024U) {
        clear = 1;
        entry_bits = 16;
    }

    if (dict_clear) *dict_clear = clear;
    if (dict_entry_bits) *dict_entry_bits = entry_bits;
}

static size_t choose_comp_dict_pool_budget_bytes(cl_command_queue queue) {
    int env_set = 0;
    size_t env_mb = 0;
    const size_t MB = (size_t)1024U * 1024U;
    const size_t min_budget = 16U * MB;
    const size_t max_budget = 512U * MB;

    {
        const char* env = getenv("LZ4_GPU_COMP_DICT_POOL_MB");
        if (env && *env) {
            char* end = NULL;
            unsigned long parsed = strtoul(env, &end, 10);
            if (end != env && *end == '\0') {
                env_set = 1;
                env_mb = (size_t)parsed;
            }
        }
    }

    if (env_set && env_mb > 0) {
        size_t b = env_mb * MB;
        if (b < min_budget) b = min_budget;
        if (b > max_budget) b = max_budget;
        return b;
    }

    {
        cl_device_id qdev = NULL;
        cl_ulong global_mem = 0;
        if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev &&
            clGetDeviceInfo(qdev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL) == CL_SUCCESS &&
            global_mem > 0) {
            size_t b = (size_t)(global_mem / 32U);
            if (b < min_budget) b = min_budget;
            if (b > max_budget) b = max_budget;
            return b;
        }
    }

    return 64U * MB;
}

typedef struct {
    size_t raw_worker_count;
    size_t active_lane_count;
    size_t launched_wi_count;
    size_t dict_owner_count;
    size_t dict_entries_per_owner;
    size_t dict_bytes_per_owner;
    size_t dict_total_bytes;
    size_t dict_budget_bytes;
    size_t padding_wi_count;
    uint32_t blocks_per_owner;
} lz4_comp_plan_t;

static lz4_comp_plan_t lz4_build_comp_plan(cl_command_queue queue,
                                           size_t file_size,
                                           size_t block_size,
                                           size_t num_blocks,
                                           size_t local_size,
                                           int requested_hash_log) {
    const int hash_log = lz4_sanitize_hash_log(requested_hash_log);
    lz4_comp_plan_t plan;
    int dict_clear = 0;
    int entry_bits = 32;
    size_t dict_pool_budget_bytes;

    memset(&plan, 0, sizeof(plan));
    lz4_effective_dict_mode_for_block(block_size, &dict_clear, &entry_bits);
    (void)dict_clear;
    dict_pool_budget_bytes = choose_comp_dict_pool_budget_bytes(queue);
    plan.dict_entries_per_owner = (size_t)1U << hash_log;
    plan.dict_bytes_per_owner = plan.dict_entries_per_owner * ((entry_bits == 16) ? sizeof(cl_ushort) : sizeof(cl_uint));
    plan.raw_worker_count = choose_comp_worker_count(queue, num_blocks, local_size);
    /*
     * Keep launch parallelism independent from dictionary budgeting.
     * Dictionary footprint is controlled by D_BITS / entry width, while
     * occupancy is still driven by the raw worker count.
     */
    plan.dict_owner_count = plan.raw_worker_count;
    plan.active_lane_count = plan.raw_worker_count;
    if (plan.active_lane_count == 0) plan.active_lane_count = 1;
    plan.launched_wi_count = round_up_size(plan.active_lane_count, local_size);
    if (plan.launched_wi_count == 0) plan.launched_wi_count = local_size ? local_size : 1;
    if (plan.launched_wi_count < plan.active_lane_count) {
        plan.launched_wi_count = plan.active_lane_count;
    }
    plan.padding_wi_count = (plan.launched_wi_count > plan.active_lane_count)
        ? (plan.launched_wi_count - plan.active_lane_count)
        : 0;
    plan.dict_total_bytes = plan.dict_owner_count * plan.dict_bytes_per_owner;
    plan.dict_budget_bytes = dict_pool_budget_bytes;
    plan.blocks_per_owner = (uint32_t)((num_blocks + plan.active_lane_count - 1) / plan.active_lane_count);
    if (plan.blocks_per_owner == 0) plan.blocks_per_owner = 1;
    return plan;
}

static size_t choose_decomp_worker_count(cl_command_queue queue, size_t num_blocks, size_t local_size) {
    if (num_blocks == 0) return 1;

    cl_device_id qdev = NULL;
    cl_uint cu = 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    }
    if (cu == 0) cu = 1;

    /* Fixed decompression launch ceiling kept from the current baseline: 96 lanes per CU for smaller workloads,
     * 48 lanes per CU for very large workloads. */
    size_t target = (size_t)cu * ((num_blocks >= 4096) ? 48U : 96U);
    if (target < local_size) target = local_size;
    if (target > num_blocks) target = num_blocks;
    if (target == 0) target = 1;
    return target;
}

static int lz4_debug_counters_enabled(void) {
    return lz4_gpu_get_debug_config().enabled;
}

static void lz4_print_comp_debug_blocks(const uint32_t* stats, int num_blocks, int block_limit) {
    int shown = (block_limit < num_blocks) ? block_limit : num_blocks;
    for (int i = 0; i < shown; ++i) {
        const size_t base = (size_t)i * LZ4_DBG_COMP_N;
        fprintf(stderr,
                "[LZ4-DBG][COMP][BLOCK %d] search_iters=%u hash_tag_hits=%u distance_rejects=%u match_found=%u hash_inserts=%u literal_bytes=%u match_bytes=%u last_literals=%u\n",
                i,
                stats[base + LZ4_DBG_COMP_SEARCH_ITERS],
                stats[base + LZ4_DBG_COMP_HASH_TAG_HITS],
                stats[base + LZ4_DBG_COMP_HASH_DISTANCE_REJECTS],
                stats[base + LZ4_DBG_COMP_MATCH_FOUND],
                stats[base + LZ4_DBG_COMP_HASH_INSERTS],
                stats[base + LZ4_DBG_COMP_LITERAL_BYTES],
                stats[base + LZ4_DBG_COMP_MATCH_BYTES],
                stats[base + LZ4_DBG_COMP_LASTLIT_BYTES]);
    }
}

static void lz4_print_comp_debug_stats(const uint32_t* stats, int num_blocks, int block_limit) {
    unsigned long long search_iters = 0;
    unsigned long long hash_tag_hits = 0;
    unsigned long long distance_rejects = 0;
    unsigned long long match_found = 0;
    unsigned long long hash_inserts = 0;
    unsigned long long literal_bytes = 0;
    unsigned long long match_bytes = 0;
    unsigned long long lastlit_bytes = 0;

    for (int i = 0; i < num_blocks; ++i) {
        const size_t base = (size_t)i * LZ4_DBG_COMP_N;
        search_iters += stats[base + LZ4_DBG_COMP_SEARCH_ITERS];
        hash_tag_hits += stats[base + LZ4_DBG_COMP_HASH_TAG_HITS];
        distance_rejects += stats[base + LZ4_DBG_COMP_HASH_DISTANCE_REJECTS];
        match_found += stats[base + LZ4_DBG_COMP_MATCH_FOUND];
        hash_inserts += stats[base + LZ4_DBG_COMP_HASH_INSERTS];
        literal_bytes += stats[base + LZ4_DBG_COMP_LITERAL_BYTES];
        match_bytes += stats[base + LZ4_DBG_COMP_MATCH_BYTES];
        lastlit_bytes += stats[base + LZ4_DBG_COMP_LASTLIT_BYTES];
    }

    double avg_search = (num_blocks > 0) ? ((double)search_iters / (double)num_blocks) : 0.0;
    double tag_hit_rate = (search_iters > 0) ? ((double)hash_tag_hits / (double)search_iters) : 0.0;
    double match_rate = (search_iters > 0) ? ((double)match_found / (double)search_iters) : 0.0;
    fprintf(stderr,
            "[LZ4-DBG][COMP] blocks=%d search_iters=%llu hash_tag_hits=%llu distance_rejects=%llu match_found=%llu hash_inserts=%llu tag_hit/search=%.4f match/search=%.4f literals=%llu matches=%llu last_literals=%llu avg_search/block=%.2f\n",
            num_blocks,
            search_iters,
            hash_tag_hits,
            distance_rejects,
            match_found,
            hash_inserts,
            tag_hit_rate,
            match_rate,
            literal_bytes,
            match_bytes,
            lastlit_bytes,
            avg_search);

    if (block_limit > 0) lz4_print_comp_debug_blocks(stats, num_blocks, block_limit);
}

static void lz4_print_dec_debug_blocks(const uint32_t* stats, int num_blocks, int block_limit) {
    int shown = (block_limit < num_blocks) ? block_limit : num_blocks;
    for (int i = 0; i < shown; ++i) {
        const size_t base = (size_t)i * LZ4_DBG_DEC_N;
        fprintf(stderr,
                "[LZ4-DBG][DECOMP][BLOCK %d] tokens=%u literal_bytes=%u match_bytes=%u small_offsets=%u fast_literals=%u fast_matches=%u output_errors=%u\n",
                i,
                stats[base + LZ4_DBG_DEC_TOKENS],
                stats[base + LZ4_DBG_DEC_LITERAL_BYTES],
                stats[base + LZ4_DBG_DEC_MATCH_BYTES],
                stats[base + LZ4_DBG_DEC_SMALL_OFFSETS],
                stats[base + LZ4_DBG_DEC_FAST_LITERAL_PATHS],
                stats[base + LZ4_DBG_DEC_FAST_MATCH_PATHS],
                stats[base + LZ4_DBG_DEC_OUTPUT_ERROR]);
    }
}

static void lz4_print_dec_debug_stats(const uint32_t* stats, int num_blocks, int block_limit) {
    unsigned long long tokens = 0;
    unsigned long long literal_bytes = 0;
    unsigned long long match_bytes = 0;
    unsigned long long small_offsets = 0;
    unsigned long long fast_literals = 0;
    unsigned long long fast_matches = 0;
    unsigned long long output_errors = 0;

    for (int i = 0; i < num_blocks; ++i) {
        const size_t base = (size_t)i * LZ4_DBG_DEC_N;
        tokens += stats[base + LZ4_DBG_DEC_TOKENS];
        literal_bytes += stats[base + LZ4_DBG_DEC_LITERAL_BYTES];
        match_bytes += stats[base + LZ4_DBG_DEC_MATCH_BYTES];
        small_offsets += stats[base + LZ4_DBG_DEC_SMALL_OFFSETS];
        fast_literals += stats[base + LZ4_DBG_DEC_FAST_LITERAL_PATHS];
        fast_matches += stats[base + LZ4_DBG_DEC_FAST_MATCH_PATHS];
        output_errors += stats[base + LZ4_DBG_DEC_OUTPUT_ERROR];
    }

    double avg_tokens = (num_blocks > 0) ? ((double)tokens / (double)num_blocks) : 0.0;
    double small_offset_ratio = (tokens > 0) ? ((double)small_offsets / (double)tokens) : 0.0;
    fprintf(stderr,
            "[LZ4-DBG][DECOMP] blocks=%d tokens=%llu literals=%llu matches=%llu small_offsets=%llu fast_literals=%llu fast_matches=%llu small_offset/token=%.4f output_errors=%llu avg_tokens/block=%.2f\n",
            num_blocks,
            tokens,
            literal_bytes,
            match_bytes,
            small_offsets,
            fast_literals,
            fast_matches,
            small_offset_ratio,
            output_errors,
            avg_tokens);

    if (block_limit > 0) lz4_print_dec_debug_blocks(stats, num_blocks, block_limit);
}

static void lz4_print_comp_host_debug(const char* input_path,
                                      size_t file_size,
                                      int num_blocks,
                                      size_t block_size,
                                      size_t l_ws,
                      const lz4_comp_plan_t* plan,
                                      int dict_clear,
                                      int entry_bits,
                                      uint32_t epoch_base,
                                      int use_standard_copy,
                                      int kernel_has_dbg,
                                      int dbg_enabled) {
    double dict_per_input = (file_size > 0 && plan) ? ((double)plan->dict_total_bytes / (double)file_size) : 0.0;
    fprintf(stderr,
        "[LZ4-DBG][HOST][COMP] input=%s file_size=%zu blocks=%d block_size=%zu local=%zu raw_workers=%zu active_lanes=%zu launched=%zu pad=%zu dict_clear=%d entry_bits=%d dict_owners=%zu dict_entries/owner=%zu dict_bytes/owner=%zu dict_total=%zu dict_budget=%zu dict/input=%.3fx blocks/owner<=%u epoch_base=%u copy=%s kernel_debug_args=%s debug_enabled=%s\n",
            input_path ? input_path : "<null>",
            file_size,
            num_blocks,
            block_size,
            l_ws,
        plan ? plan->raw_worker_count : 0,
        plan ? plan->active_lane_count : 0,
        plan ? plan->launched_wi_count : 0,
        plan ? plan->padding_wi_count : 0,
            dict_clear,
            entry_bits,
        plan ? plan->dict_owner_count : 0,
        plan ? plan->dict_entries_per_owner : 0,
        plan ? plan->dict_bytes_per_owner : 0,
        plan ? plan->dict_total_bytes : 0,
        plan ? plan->dict_budget_bytes : 0,
            dict_per_input,
        plan ? plan->blocks_per_owner : 0,
            epoch_base,
            use_standard_copy ? "standard" : "mapped",
            kernel_has_dbg ? "yes" : "no",
            dbg_enabled ? "yes" : "no");
}

static void lz4_print_dec_host_debug(const char* input_path,
                                     uint32_t num_blocks,
                                     uint32_t block_size,
                                     size_t l_ws,
                                     size_t worker_count,
                                     size_t g_ws,
                                     size_t data_size,
                                     int use_standard_copy,
                                     int kernel_has_dbg,
                                     int dbg_enabled) {
    fprintf(stderr,
            "[LZ4-DBG][HOST][DECOMP] input=%s comp_bytes=%zu blocks=%u block_size=%u local=%zu workers=%zu global=%zu copy=%s kernel_debug_args=%s debug_enabled=%s\n",
            input_path ? input_path : "<null>",
            data_size,
            num_blocks,
            block_size,
            l_ws,
            worker_count,
            g_ws,
            use_standard_copy ? "standard" : "mapped",
            kernel_has_dbg ? "yes" : "no",
            dbg_enabled ? "yes" : "no");
}

static int lz4_write_blocks_packed(FILE* fout,
                                   const uint8_t* base,
                                   const uint32_t* offsets,
                                   const uint32_t* sizes,
                                   size_t count) {
    size_t pack_kb = 1024;
    uint8_t* pack = NULL;
    int temp_pack = 0;
    size_t fill = 0;

    size_t pack_cap = pack_kb * 1024;

    if (!fout || !base || !offsets || !sizes) return -1;

    pack = (uint8_t*)malloc(pack_cap);
    temp_pack = (pack != NULL);

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
            if (temp_pack) free(pack);
            return -1;
        }
    }

    if (temp_pack) free(pack);
    return 0;
}

static int lz4_write_blocks_from_mapped_buffer(cl_command_queue queue,
                                                cl_mem src_buf,
                                                size_t mapped_bytes,
                                                FILE* fout,
                                                const uint32_t* offsets,
                                                const uint32_t* sizes,
                                                size_t count,
                                                unsigned long* map_read_us) {
    cl_int err;
    void* mapped;
    uint64_t t0;

    if (!queue || !src_buf || !fout || !offsets || !sizes) return -1;

    t0 = get_us();
    mapped = clEnqueueMapBuffer(queue, src_buf, CL_TRUE, CL_MAP_READ, 0, mapped_bytes, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS || !mapped) {
        return -1;
    }
    if (map_read_us) *map_read_us = (unsigned long)(get_us() - t0);

    if (lz4_write_blocks_packed(fout, (const uint8_t*)mapped, offsets, sizes, count) != 0) {
        (void)clEnqueueUnmapMemObject(queue, src_buf, mapped, 0, NULL, NULL);
        return -1;
    }

    err = clEnqueueUnmapMemObject(queue, src_buf, mapped, 0, NULL, NULL);
    if (err != CL_SUCCESS) return -1;
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

static int lz4_write_contiguous_from_mapped_buffer(cl_command_queue queue,
                                                   cl_mem src_buf,
                                                   size_t mapped_bytes,
                                                   FILE* fout,
                                                   unsigned long* map_read_us) {
    cl_int err;
    void* mapped;
    uint64_t t0;

    if (!queue || !src_buf || !fout) return -1;
    if (mapped_bytes == 0) return 0;

    t0 = get_us();
    mapped = clEnqueueMapBuffer(queue, src_buf, CL_TRUE, CL_MAP_READ, 0, mapped_bytes, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS || !mapped) return -1;
    if (map_read_us) *map_read_us = (unsigned long)(get_us() - t0);

    if (fwrite(mapped, 1, mapped_bytes, fout) != mapped_bytes) {
        (void)clEnqueueUnmapMemObject(queue, src_buf, mapped, 0, NULL, NULL);
        return -1;
    }

    err = clEnqueueUnmapMemObject(queue, src_buf, mapped, 0, NULL, NULL);
    if (err != CL_SUCCESS) return -1;
    return 0;
}

static int lz4_readback_to_file_chunked(cl_command_queue queue,
                                        cl_mem src_buf,
                                        size_t total_bytes,
                                        FILE* fout,
                                        size_t chunk_bytes,
                                        unsigned long* download_us,
                                        unsigned long* write_us) {
    uint8_t* staging;
    size_t off = 0;

    if (!queue || !src_buf || !fout) return -1;
    if (total_bytes == 0) return 0;
    if (chunk_bytes < 256U * 1024U) chunk_bytes = 256U * 1024U;

    staging = (uint8_t*)malloc(chunk_bytes);
    if (!staging) return -1;

    while (off < total_bytes) {
        size_t step = total_bytes - off;
        cl_int err;
        uint64_t t0;
        if (step > chunk_bytes) step = chunk_bytes;

        t0 = get_us();
        err = clEnqueueReadBuffer(queue, src_buf, CL_TRUE, off, step, staging, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            free(staging);
            return -1;
        }
        if (download_us) *download_us += (unsigned long)(get_us() - t0);

        t0 = get_us();
        if (fwrite(staging, 1, step, fout) != step) {
            free(staging);
            return -1;
        }
        if (write_us) *write_us += (unsigned long)(get_us() - t0);
        off += step;
    }

    free(staging);
    return 0;
}
int lz4_compress_core(cl_context context, cl_command_queue queue, cl_kernel kernel,
                    const char* input_path, const char* output_path,
                    size_t block_size, int acceleration, int hash_log, lz4_gpu_workspace_t* ws,
                    timing_t* t, int local_size,
                    int skip_input_upload) {
    cl_int err;
    uint64_t t1, t2;

    struct stat st_buf;
    if (!input_path || stat(input_path, &st_buf) != 0 || st_buf.st_size <= 0) return -1;
    size_t file_size = (size_t)st_buf.st_size;
    t->in_size = (unsigned long)file_size;

    t1 = get_us();
    int num_blocks = (file_size + block_size - 1) / block_size;
    int use_standard_copy = lz4_prefers_standard_copy(queue);
    lz4_gpu_debug_config_t dbg_cfg = lz4_gpu_get_debug_config();

    size_t l_ws = sanitize_local_size(queue, (local_size > 0) ? (size_t)local_size : 1, (size_t)num_blocks);
    hash_log = lz4_sanitize_hash_log(hash_log);
    lz4_comp_plan_t comp_plan = lz4_build_comp_plan(queue,
                                                    file_size,
                                                    block_size,
                                                    (size_t)num_blocks,
                                                    l_ws,
                                                    hash_log);
    size_t g_ws = comp_plan.launched_wi_count;
    cl_uint kernel_num_args = 0;
    int kernel_has_dbg = 0;
    if (clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args), &kernel_num_args, NULL) == CL_SUCCESS) {
        kernel_has_dbg = (kernel_num_args >= 14U);
    }

    uint32_t* h_block_info = malloc(num_blocks * 2 * sizeof(uint32_t));
    uint32_t* h_out_offsets = malloc(num_blocks * sizeof(uint32_t));
    cl_mem dbg_comp_buf = NULL;
    int dbg_comp_requested = dbg_cfg.enabled;
    int dbg_comp_enabled = dbg_comp_requested && kernel_has_dbg;
    int dict_clear = 0;
    int dict_entry_bits = 32;
    if (dbg_comp_requested && !kernel_has_dbg) {
        fprintf(stderr, "[LZ4-DBG][COMP] warning: kernel has no debug args, debug counters disabled\n");
    }
    uint32_t single_block_max_out = (uint32_t)(block_size * 1.1 + 64);
    lz4_effective_dict_mode_for_block(block_size, &dict_clear, &dict_entry_bits);
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

    ws->comp_meta_cached_valid = 0;

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
    size_t prev_dict_capacity = ws->current_dict_capacity;
    ws->dict_buf = ensure_buffer_ex(context,
                                    ws->dict_buf,
                                    comp_plan.dict_total_bytes,
                                    &ws->current_dict_capacity,
                                    CL_MEM_READ_WRITE,
                                    !use_standard_copy,
                                    &err);
    if (ws->current_dict_capacity != prev_dict_capacity) {
        (void)zero_buffer(queue, ws->dict_buf, ws->current_dict_capacity);
    }
    if (ws->comp_epoch_base == 0) ws->comp_epoch_base = 1;
    {
        uint32_t epochs_needed = comp_plan.blocks_per_owner + 2U;
        if (epochs_needed >= UINT32_MAX - 1024U) epochs_needed = 1024U;
        /* 12-bit epoch in kernel: clear dict when low 12 bits would wrap to avoid stale collisions */
        uint32_t cur_low = ws->comp_epoch_base & 0xFFF;
        uint32_t end_low = (ws->comp_epoch_base + epochs_needed) & 0xFFF;
        int wraps_12bit = (end_low <= cur_low) || (ws->comp_epoch_base > (uint32_t)(UINT32_MAX - epochs_needed));
        if (wraps_12bit) {
            (void)zero_buffer(queue, ws->dict_buf, ws->current_dict_capacity);
            ws->comp_epoch_base = 1;
        }
    }
    uint32_t epoch_base = ws->comp_epoch_base;
    ws->comp_epoch_base += comp_plan.blocks_per_owner + 2U;
    if (dbg_comp_requested) {
        lz4_print_comp_host_debug(input_path,
                                  file_size,
                                  num_blocks,
                                  block_size,
                                  l_ws,
                                  &comp_plan,
                                  dict_clear,
                                  dict_entry_bits,
                                  epoch_base,
                                  use_standard_copy,
                                  kernel_has_dbg,
                                  dbg_comp_enabled);
    }
    t->buffer_alloc_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    int globalIndexBase = 0;
    int inputSize = (int)file_size;
    uint32_t active_lane_count = (uint32_t)comp_plan.active_lane_count;
    cl_mem dbg_comp_arg = dbg_comp_enabled ? dbg_comp_buf : ws->output_size_buf;
    uint32_t dbg_comp_flag = dbg_comp_enabled ? 1U : 0U;

    if (!skip_input_upload) {
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ws->in_buf);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &ws->out_buf);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &ws->output_size_buf);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &num_blocks);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &inputSize);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &block_size);
        err |= clSetKernelArg(kernel, 6, sizeof(uint32_t), &single_block_max_out);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &acceleration);
        err |= clSetKernelArg(kernel, 8, sizeof(int), &globalIndexBase);
        err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &ws->dict_buf);
        err |= clSetKernelArg(kernel, 10, sizeof(uint32_t), &active_lane_count);
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
    clReleaseEvent(ev);
    t->kernel_exec_us = (unsigned long)(get_us() - t1);
    t->algo_config = (unsigned long)hash_log;

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

    size_t total_out_read = (size_t)num_blocks * single_block_max_out;
    t->download_total_us = (unsigned long)(get_us() - t1);

    if (dbg_comp_enabled && dbg_comp_buf) {
        size_t dbg_comp_bytes = (size_t)num_blocks * LZ4_DBG_COMP_N * sizeof(uint32_t);
        uint32_t* dbg_comp_stats = (uint32_t*)malloc(dbg_comp_bytes);
        if (dbg_comp_stats) {
            if (clEnqueueReadBuffer(queue, dbg_comp_buf, CL_TRUE, 0, dbg_comp_bytes, dbg_comp_stats, 0, NULL, NULL) == CL_SUCCESS) {
                lz4_print_comp_debug_stats(dbg_comp_stats, num_blocks, dbg_cfg.block_limit);
            }
            free(dbg_comp_stats);
        }
    }

    t1 = get_us();
    FILE* fout = NULL;
    if (output_path && output_path[0] != '\0') {
        fout = fopen(output_path, "wb");
    }
    if (fout) {
        lz4_set_stream_buffer(fout);
        uint32_t magic = 0x184D2204;
        fwrite(&magic, 1, 4, fout);
        fwrite(&num_blocks, 1, 4, fout);
        uint32_t bsize_u32 = (uint32_t)block_size;
        fwrite(&bsize_u32, 1, 4, fout);
        fwrite(h_osizes, 1, num_blocks * 4, fout);
        if (!use_standard_copy) {
            if (lz4_write_blocks_from_mapped_buffer(queue,
                                                    ws->out_buf,
                                                    total_out_read,
                                                    fout,
                                                    h_out_offsets,
                                                    h_osizes,
                                                    (size_t)num_blocks,
                                                    &t->download_total_us) != 0) {
                fprintf(stderr, "[LZ4] mapped write for sparse payload failed\n");
                fclose(fout);
                if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                free(h_block_info); free(h_osizes); free(h_out_offsets);
                return -1;
            }
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
    t->file_write_us = (output_path && output_path[0] != '\0')
        ? (unsigned long)(get_us() - t1)
        : 0;

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
    lz4_gpu_debug_config_t dbg_cfg = lz4_gpu_get_debug_config();
    FILE* fin = fopen(input_path, "rb"); if (!fin) return -1;
    lz4_set_stream_buffer(fin);
    uint32_t magic, num_blocks, block_size_val;
    if (fread(&magic, 1, 4, fin) != 4 || magic != 0x184D2204) { fclose(fin); return -1; }
    if (fread(&num_blocks, 1, 4, fin) != 4) { fclose(fin); return -1; }
    if (fread(&block_size_val, 1, 4, fin) != 4) { fclose(fin); return -1; }

    /* Validate the header against the actual file before trusting num_blocks
     * for allocation or handing per-block (offset,length) to the kernel. A
     * corrupted/truncated frame (payloads cross the network in migration)
     * must be rejected here so a bogus length cannot defeat the kernel's own
     * output-bounds guard. */
    size_t header_size = 12 + (size_t)num_blocks * 4;
    fseek(fin, 0, SEEK_END);
    size_t file_size = ftell(fin);
    if (block_size_val == 0 || header_size > file_size) { fclose(fin); return -1; }
    size_t data_size = file_size - header_size;
    fseek(fin, 12, SEEK_SET);

    uint32_t* h_comp_sizes = malloc(num_blocks * sizeof(uint32_t));
    if (!h_comp_sizes) { fclose(fin); return -1; }
    if (fread(h_comp_sizes, 4, num_blocks, fin) != num_blocks) { free(h_comp_sizes); fclose(fin); return -1; }

    /* Per-block compressed lengths must exactly tile the payload region. */
    {
        uint64_t sum = 0;
        for (uint32_t i = 0; i < num_blocks; i++) sum += h_comp_sizes[i];
        if (sum != (uint64_t)data_size) { free(h_comp_sizes); fclose(fin); return -1; }
    }
    fseek(fin, header_size, SEEK_SET);
    t->in_size = (unsigned long)file_size;
    t->file_read_us = 0;
    t->data_upload_us = 0;

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
        unsigned long file_read_us_acc = 0;
        unsigned long upload_us_acc = 0;

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
            {
                uint64_t t_read0 = get_us();
                size_t nr = fread(h_in, 1, data_size, fin);
                file_read_us_acc += (unsigned long)(get_us() - t_read0);
                if (nr != data_size) {
                    fprintf(stderr, "[LZ4] read compressed input failed for standard-copy decompress path\n");
                    free(h_in);
                    fclose(fin);
                    free(h_comp_sizes);
                    free(h_comp_offsets);
                    free(h_comp_sizes_32);
                    free(h_out_offsets);
                    free(h_max_out_sizes);
                    return -1;
                }
            }
            {
                uint64_t t_up0 = get_us();
                if (write_buffer_auto(queue, ws->in_buf, h_in, data_size, 1) != 0) {
                    fprintf(stderr, "[LZ4] upload compressed input failed for standard-copy decompress path\n");
                    free(h_in);
                    fclose(fin);
                    free(h_comp_sizes);
                    free(h_comp_offsets);
                    free(h_comp_sizes_32);
                    free(h_out_offsets);
                    free(h_max_out_sizes);
                    return -1;
                }
                upload_us_acc += (unsigned long)(get_us() - t_up0);
            }
            free(h_in);
        } else {
            void* mapped_in = clEnqueueMapBuffer(queue, ws->in_buf, CL_TRUE, CL_MAP_WRITE, 0, data_size, 0, NULL, NULL, &err);
            if (err == CL_SUCCESS && mapped_in) {
                uint64_t t_read0 = get_us();
                size_t nr = fread(mapped_in, 1, data_size, fin);
                file_read_us_acc += (unsigned long)(get_us() - t_read0);
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
        t->file_read_us = file_read_us_acc;
        t->data_upload_us += upload_us_acc;
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

    if (use_standard_copy) {
        uint64_t t_meta_up0 = get_us();
        cl_event meta_ev[2] = {0};
        int ev_count = 0;
        err  = clEnqueueWriteBuffer(queue, ws->decomp_comp_off_buf, CL_FALSE, 0, num_blocks * 4, h_comp_offsets, 0, NULL, &meta_ev[ev_count]);
        if (err == CL_SUCCESS) ev_count++;
        if (err == CL_SUCCESS) {
            err = clEnqueueWriteBuffer(queue, ws->decomp_comp_size_buf, CL_FALSE, 0, num_blocks * 4, h_comp_sizes_32, 0, NULL, &meta_ev[ev_count]);
            if (err == CL_SUCCESS) ev_count++;
        }

        if (err != CL_SUCCESS || (ev_count > 0 && clWaitForEvents((cl_uint)ev_count, meta_ev) != CL_SUCCESS)) {
            for (int ei = 0; ei < ev_count; ++ei) if (meta_ev[ei]) clReleaseEvent(meta_ev[ei]);
            fprintf(stderr, "[LZ4] upload decompress metadata buffers failed\n");
            free(h_comp_sizes);
            free(h_comp_offsets);
            free(h_comp_sizes_32);
            free(h_out_offsets);
            free(h_max_out_sizes);
            return -1;
        }
        t->data_upload_us += (unsigned long)(get_us() - t_meta_up0);
        for (int ei = 0; ei < ev_count; ++ei) if (meta_ev[ei]) clReleaseEvent(meta_ev[ei]);
    } else {
        uint64_t t_meta_up0 = get_us();
        if (write_buffer_auto(queue, ws->decomp_comp_off_buf, h_comp_offsets, num_blocks * 4, 0) != 0 ||
            write_buffer_auto(queue, ws->decomp_comp_size_buf, h_comp_sizes_32, num_blocks * 4, 0) != 0) {
            fprintf(stderr, "[LZ4] upload decompress metadata buffers failed\n");
            free(h_comp_sizes);
            free(h_comp_offsets);
            free(h_comp_sizes_32);
            free(h_out_offsets);
            free(h_max_out_sizes);
            return -1;
        }
        t->data_upload_us += (unsigned long)(get_us() - t_meta_up0);
    }
    t->buffer_alloc_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    cl_mem dbg_dec_buf = NULL;
    cl_uint kernel_num_args = 0;
    int kernel_has_dbg = 0;
    if (clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args), &kernel_num_args, NULL) == CL_SUCCESS) {
        kernel_has_dbg = (kernel_num_args >= 9U);
    }
    int dbg_dec_requested = dbg_cfg.enabled;
    int dbg_dec_enabled = dbg_dec_requested && kernel_has_dbg;
    if (dbg_dec_requested && !kernel_has_dbg) {
        fprintf(stderr, "[LZ4-DBG][DECOMP] warning: kernel has no debug args, debug counters disabled\n");
    }
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &ws->in_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ws->out_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &ws->decomp_comp_off_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &ws->decomp_comp_size_buf);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &ws->decomp_sizes_out_buf);
    clSetKernelArg(kernel, 5, sizeof(uint32_t), &block_size_val);
    clSetKernelArg(kernel, 6, sizeof(uint32_t), &num_blocks);

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
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &dbg_dec_arg);
        clSetKernelArg(kernel, 8, sizeof(uint32_t), &dbg_dec_flag);
    }

    size_t l_ws = (size_t)t->local_size;
    if (l_ws == 0) {
        l_ws = sanitize_local_size(queue, (local_size > 0) ? (size_t)local_size : 1, (size_t)num_blocks);
    }
    size_t worker_count = choose_decomp_worker_count(queue, (size_t)num_blocks, l_ws);
    size_t g_ws = round_up_size(worker_count, l_ws);
    if (dbg_dec_requested) {
        lz4_print_dec_host_debug(input_path,
                                 num_blocks,
                                 block_size_val,
                                 l_ws,
                                 worker_count,
                                 g_ws,
                                 data_size,
                                 use_standard_copy,
                                 kernel_has_dbg,
                                 dbg_dec_enabled);
    }
    t->global_size = (unsigned long)g_ws;
    t->local_size = (unsigned long)l_ws;
    cl_event ev;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &g_ws, &l_ws, 0, NULL, &ev);
    clWaitForEvents(1, &ev);
    clReleaseEvent(ev);
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
    for (uint32_t i = 0; i < num_blocks; i++) {
        /* The kernel writes 0xFFFFFFFF into a block's size on a decode error
         * (e.g. the new corrupted-input guard). Reject instead of folding the
         * sentinel into the total, which would corrupt the readback size. */
        if (h_final_sizes[i] == 0xFFFFFFFFu) {
            fprintf(stderr, "[LZ4] decode error in block %u (corrupted input)\n", i);
            free(h_final_sizes);
            if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
            free(h_comp_sizes); free(h_comp_offsets); free(h_comp_sizes_32);
            free(h_out_offsets); free(h_max_out_sizes);
            return -1;
        }
        total_decomp_sz += h_final_sizes[i];
    }
    t->out_size = total_decomp_sz;

    t->download_total_us = (unsigned long)(get_us() - t1);

    if (dbg_dec_enabled && dbg_dec_buf) {
        size_t dbg_dec_bytes = (size_t)num_blocks * LZ4_DBG_DEC_N * sizeof(uint32_t);
        uint32_t* dbg_dec_stats = (uint32_t*)malloc(dbg_dec_bytes);
        if (dbg_dec_stats) {
            if (clEnqueueReadBuffer(queue, dbg_dec_buf, CL_TRUE, 0, dbg_dec_bytes, dbg_dec_stats, 0, NULL, NULL) == CL_SUCCESS) {
                lz4_print_dec_debug_stats(dbg_dec_stats, (int)num_blocks, dbg_cfg.block_limit);
            }
            free(dbg_dec_stats);
        }
    }

    t1 = get_us();
    FILE* fout = fopen(output_path, "wb");
    if (fout) {
        int disable_mapped_write = lz4_env_flag_value("LZ4_GPU_DISABLE_DECOMP_MAPPED_WRITE", NULL);
        size_t chunk_threshold_kb = (size_t)lz4_env_unsigned_value("LZ4_GPU_DECOMP_CHUNKED_THRESHOLD_KB", 16384U);
        size_t chunk_readback_kb = (size_t)lz4_env_unsigned_value("LZ4_GPU_DECOMP_READBACK_KB", 8192U);
        int force_chunked_set = 0;
        int force_chunked = lz4_env_flag_value("LZ4_GPU_DECOMP_FORCE_CHUNKED", &force_chunked_set);
        int disable_chunked = lz4_env_flag_value("LZ4_GPU_DECOMP_DISABLE_CHUNKED", NULL);
        int use_chunked_output;
        lz4_set_stream_buffer(fout);
        if (force_chunked_set) {
            use_chunked_output = force_chunked;
        } else {
            use_chunked_output = lz4_chunked_output_default_enabled() &&
                                 !disable_chunked &&
                                 total_decomp_sz >= chunk_threshold_kb * 1024ULL;
        }

        if (use_chunked_output) {
            unsigned long chunk_download_us = 0;
            unsigned long chunk_write_us = 0;
            if (lz4_readback_to_file_chunked(queue,
                                             ws->out_buf,
                                             total_decomp_sz,
                                             fout,
                                             chunk_readback_kb * 1024ULL,
                                             &chunk_download_us,
                                             &chunk_write_us) != 0) {
                fprintf(stderr, "[LZ4] chunked readback/write for decompressed payload failed\n");
                fclose(fout);
                if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
                free(h_comp_sizes); free(h_comp_offsets); free(h_comp_sizes_32);
                free(h_out_offsets); free(h_max_out_sizes); free(h_final_sizes);
                return -1;
            }
            t->download_total_us = chunk_download_us;
            t->file_write_us = chunk_write_us;
        } else if (!use_standard_copy && !disable_mapped_write) {
            if (lz4_write_contiguous_from_mapped_buffer(queue,
                                                        ws->out_buf,
                                                        total_decomp_sz,
                                                        fout,
                                                        &t->download_total_us) != 0) {
                fprintf(stderr, "[LZ4] mapped write for decompressed payload failed\n");
                fclose(fout);
                if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
                free(h_comp_sizes); free(h_comp_offsets); free(h_comp_sizes_32);
                free(h_out_offsets); free(h_max_out_sizes); free(h_final_sizes);
                return -1;
            }
        } else {
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
        }
        fclose(fout);
    }
    if (t->file_write_us == 0) {
        t->file_write_us = (unsigned long)(get_us() - t1);
    }

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

static int lz4_join_path2(char* out, size_t outlen, const char* a, const char* b) {
    size_t la;
    size_t lb;
    if (!out || outlen == 0 || !a || !b) return -1;
    la = strlen(a);
    lb = strlen(b);
    if (la + 1 + lb + 1 > outlen) return -1;
    memcpy(out, a, la);
    out[la] = '/';
    memcpy(out + la + 1, b, lb);
    out[la + 1 + lb] = '\0';
    return 0;
}

static int lz4_join_path3(char* out, size_t outlen, const char* a, const char* b, const char* c) {
    size_t la;
    size_t lb;
    size_t lc;
    if (!out || outlen == 0 || !a || !b || !c) return -1;
    la = strlen(a);
    lb = strlen(b);
    lc = strlen(c);
    if (la + 1 + lb + 1 + lc + 1 > outlen) return -1;
    memcpy(out, a, la);
    out[la] = '/';
    memcpy(out + la + 1, b, lb);
    out[la + 1 + lb] = '/';
    memcpy(out + la + 1 + lb + 1, c, lc);
    out[la + 1 + lb + 1 + lc] = '\0';
    return 0;
}

static int lz4_find_file_path(const char* name, char* out, size_t outlen) {
    char path[PATH_MAX];
    char base[PATH_MAX];

    if (!name || !out || outlen == 0) return -1;

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
                if (lz4_join_path3(path, sizeof(path), exe_path, "../lz4_gpu", name) == 0 &&
                    access(path, R_OK) == 0) {
                    strncpy(out, path, outlen - 1);
                    out[outlen - 1] = '\0';
                    return 0;
                }
                if (lz4_join_path2(path, sizeof(path), exe_path, name) == 0 &&
                    access(path, R_OK) == 0) {
                    strncpy(out, path, outlen - 1);
                    out[outlen - 1] = '\0';
                    return 0;
                }
            }
        }
    }

    if (getcwd(base, sizeof(base)) != NULL) {
        if (lz4_join_path2(path, sizeof(path), base, name) == 0 &&
            access(path, R_OK) == 0) {
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

static void lz4_dirname_from_path(const char* path, char* out, size_t outlen) {
    const char* slash;
    const char* backslash;
    size_t len;
    if (!out || outlen == 0) return;
    out[0] = '\0';
    if (!path || !*path) return;
    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    if (!slash || (backslash && backslash > slash)) slash = backslash;
    if (!slash) {
        strncpy(out, ".", outlen - 1);
        out[outlen - 1] = '\0';
        return;
    }
    len = (size_t)(slash - path);
    if (len >= outlen) len = outlen - 1;
    memcpy(out, path, len);
    out[len] = '\0';
    for (size_t i = 0; out[i]; ++i) {
        if (out[i] == '\\') out[i] = '/';
    }
}

cl_program lz4_load_program(cl_context context, cl_device_id device, int hash_log, size_t block_size) {
    int dbg_enabled = lz4_debug_counters_enabled();
    int use_clbin = lz4_env_flag_value("LZ4_GPU_USE_CLBIN", NULL);
    int dict_clear = 0;
    int dict_entry_bits = 32;

    lz4_effective_dict_mode_for_block(block_size, &dict_clear, &dict_entry_bits);
    hash_log = lz4_sanitize_hash_log(hash_log);

    if (use_clbin && !dbg_enabled && dict_clear == 0 && dict_entry_bits == 32) {
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
                        if (kernel_err == CL_SUCCESS && kcomp) {
                            kdec = clCreateKernel(prog, "lz4_decompress_blocks", &kernel_err);
                        }
                        if (kcomp) clReleaseKernel(kcomp);
                        if (kdec) clReleaseKernel(kdec);
                        if (kernel_err == CL_SUCCESS && kdec) {
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
        const char* source_name = "lz4_gpu.cl";
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
        char include_dir[PATH_MAX];
        char flags[PATH_MAX + 128];
        lz4_dirname_from_path(resolved_src, include_dir, sizeof(include_dir));
        if (dbg_enabled) {
            snprintf(flags, sizeof(flags), "-cl-std=CL1.2 -I%s -DLZ4_HASHLOG=%d -DLZ4_GPU_DEBUG_COUNTERS_RUNTIME=1 -DLZ4_GPU_DICT_CLEAR=%d -DLZ4_GPU_DICT_ENTRY_BITS=%d", include_dir, hash_log, dict_clear, dict_entry_bits);
        } else {
            snprintf(flags, sizeof(flags), "-cl-std=CL1.2 -I%s -DLZ4_HASHLOG=%d -DLZ4_GPU_DICT_CLEAR=%d -DLZ4_GPU_DICT_ENTRY_BITS=%d", include_dir, hash_log, dict_clear, dict_entry_bits);
        }
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
