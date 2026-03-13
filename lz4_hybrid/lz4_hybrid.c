#define _POSIX_C_SOURCE 200809L

#include <CL/cl.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>

#include "../lz4_gpu/lz4_gpu_core.h"
#include "../lz4_gpu/lz4_gpu_utils.h"
#include "../lib/lz4.h"

#define HYBRID_MAGIC 0x184D2205U
#define HYBRID_GPU_BLOCKS_STRIPED_FLAG 0x80000000U

typedef struct {
    size_t block_size;
    int acceleration;
    int hash_log;
    int local_size;
    int cpu_threads;
    double gpu_ratio;
    int adaptive_split;
    size_t adaptive_sample_blocks;
    int verbose;
} hybrid_cfg_t;

typedef struct {
    uint64_t cpu_kernel_us;
    uint64_t gpu_kernel_us;
    uint64_t parallel_us;
    uint64_t total_us;
} hybrid_metrics_t;

typedef struct {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev;
    cl_program prog;
    cl_kernel kcomp;
    cl_kernel kpack;
    cl_kernel kdec;
    lz4_gpu_workspace_t ws;
} ocl_env_t;

typedef struct {
    const unsigned char* src;
    size_t src_size;
    size_t block_size;
    const size_t* block_indices;
    size_t num_blocks;
    int num_threads;
    int acceleration;

    unsigned char* out_slots;
    size_t out_slot_size;
    uint32_t* out_sizes;

    uint64_t elapsed_us;
    int err;
} cpu_comp_job_t;

typedef struct {
    const unsigned char* comp_data;
    const uint32_t* comp_sizes;
    const uint32_t* comp_offsets;
    size_t block_size;
    const size_t* block_indices;
    size_t num_blocks;
    int num_threads;

    unsigned char* out_full;
    uint32_t* out_sizes;

    uint64_t elapsed_us;
    int err;
} cpu_decomp_job_t;

typedef struct {
    size_t thread_idx;
    size_t thread_count;
    cpu_comp_job_t* job;
} cpu_comp_worker_arg_t;

typedef struct {
    size_t thread_idx;
    size_t thread_count;
    cpu_decomp_job_t* job;
} cpu_decomp_worker_arg_t;

typedef struct {
    size_t sample_count;
    double mean_ratio_pct;
    size_t low_ratio_blocks;
    size_t high_ratio_blocks;
} lz4_sample_stats_t;

static cl_device_type preferred_opencl_device_type(void) {
    const char* pref = getenv("FORCE_OPENCL_DEVICE");
    if (!pref || !*pref) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "CPU") == 0) return CL_DEVICE_TYPE_CPU;
    if (strcasecmp(pref, "GPU") == 0) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "DEFAULT") == 0) return CL_DEVICE_TYPE_DEFAULT;
    if (strcasecmp(pref, "ALL") == 0) return CL_DEVICE_TYPE_ALL;
    return CL_DEVICE_TYPE_GPU;
}

static int gpu_compress_blocks(ocl_env_t* ocl,
                               const unsigned char* src,
                               size_t src_size,
                               size_t block_size,
                               int acceleration,
                               int local_size,
                               int hash_log,
                               uint32_t** out_sizes,
                               uint32_t** out_offsets,
                               unsigned char** out_slots,
                               size_t* out_slot_size,
                               uint64_t* kernel_us);

static int gpu_decompress_blocks(ocl_env_t* ocl,
                                 const unsigned char* comp_data,
                                 const uint32_t* comp_sizes,
                                 size_t num_blocks,
                                 size_t block_size,
                                 int local_size,
                                 unsigned char* out_full,
                                 uint32_t* out_sizes,
                                 uint64_t* kernel_us);

static size_t parse_size_bytes(const char* s) {
    char* endptr = NULL;
    size_t val = strtoul(s, &endptr, 10);
    if (endptr && (*endptr == 'k' || *endptr == 'K')) val *= 1024U;
    else if (endptr && (*endptr == 'm' || *endptr == 'M')) val *= 1024U * 1024U;
    return val;
}

static int cmp_double_asc(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
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
    {
        const double out = (n % 2 == 0) ? (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0 : tmp[n / 2];
        free(tmp);
        return out;
    }
}

static double elapsed_sec(const struct timespec* start, const struct timespec* end) {
    return (double)(end->tv_sec - start->tv_sec) + (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

static int is_number_string(const char* s) {
    char* endp = NULL;
    if (!s || !*s) return 0;
    (void)strtod(s, &endp);
    return endp && *endp == '\0';
}

static size_t block_input_size(size_t total_size, size_t block_size, size_t block_idx) {
    const size_t start = block_idx * block_size;
    if (start >= total_size) return 0;
    {
        const size_t rem = total_size - start;
        return rem < block_size ? rem : block_size;
    }
}

static size_t sampled_block_index(size_t sample_pos, size_t sample_count, size_t num_blocks) {
    if (num_blocks == 0) return 0;
    if (sample_count <= 1 || num_blocks <= 1) return 0;
    return (sample_pos * (num_blocks - 1)) / (sample_count - 1);
}

static void move_last_block_to_end(size_t* indices, size_t count, size_t num_blocks) {
    size_t last_block;
    if (!indices || count == 0 || num_blocks == 0) return;
    last_block = num_blocks - 1;
    if (indices[count - 1] == last_block) return;
    for (size_t i = 0; i < count; ++i) {
        if (indices[i] == last_block) {
            const size_t tmp = indices[count - 1];
            indices[count - 1] = indices[i];
            indices[i] = tmp;
            return;
        }
    }
}

static int partition_blocks_prefix(size_t num_blocks,
                                   size_t gpu_blocks,
                                   size_t** gpu_indices,
                                   size_t* gpu_count,
                                   size_t** cpu_indices,
                                   size_t* cpu_count) {
    size_t* gpu = NULL;
    size_t* cpu = NULL;
    size_t cpu_blocks;

    if (!gpu_indices || !gpu_count || !cpu_indices || !cpu_count) return -1;
    if (gpu_blocks > num_blocks) gpu_blocks = num_blocks;
    cpu_blocks = num_blocks - gpu_blocks;

    if (gpu_blocks > 0) {
        gpu = (size_t*)malloc(gpu_blocks * sizeof(size_t));
        if (!gpu) return -1;
        for (size_t i = 0; i < gpu_blocks; ++i) gpu[i] = i;
    }
    if (cpu_blocks > 0) {
        cpu = (size_t*)malloc(cpu_blocks * sizeof(size_t));
        if (!cpu) {
            free(gpu);
            return -1;
        }
        for (size_t i = 0; i < cpu_blocks; ++i) cpu[i] = gpu_blocks + i;
    }

    *gpu_indices = gpu;
    *gpu_count = gpu_blocks;
    *cpu_indices = cpu;
    *cpu_count = cpu_blocks;
    return 0;
}

static int partition_blocks_distributed(size_t num_blocks,
                                        size_t gpu_blocks,
                                        size_t** gpu_indices,
                                        size_t* gpu_count,
                                        size_t** cpu_indices,
                                        size_t* cpu_count) {
    size_t* gpu = NULL;
    size_t* cpu = NULL;
    size_t gpu_written = 0;
    size_t cpu_written = 0;

    if (!gpu_indices || !gpu_count || !cpu_indices || !cpu_count) return -1;
    if (gpu_blocks > num_blocks) gpu_blocks = num_blocks;

    if (gpu_blocks > 0) {
        gpu = (size_t*)malloc(gpu_blocks * sizeof(size_t));
        if (!gpu) return -1;
    }
    if (num_blocks > gpu_blocks) {
        cpu = (size_t*)malloc((num_blocks - gpu_blocks) * sizeof(size_t));
        if (!cpu) {
            free(gpu);
            return -1;
        }
    }

    for (size_t idx = 0; idx < num_blocks; ++idx) {
        if ((((idx + 1) * gpu_blocks) / num_blocks) != ((idx * gpu_blocks) / num_blocks)) {
            gpu[gpu_written++] = idx;
        } else {
            cpu[cpu_written++] = idx;
        }
    }

    move_last_block_to_end(gpu, gpu_written, num_blocks);
    move_last_block_to_end(cpu, cpu_written, num_blocks);

    *gpu_indices = gpu;
    *gpu_count = gpu_written;
    *cpu_indices = cpu;
    *cpu_count = cpu_written;
    return 0;
}

static int collect_lz4_sample_stats(const unsigned char* input,
                                    size_t input_size,
                                    size_t num_blocks,
                                    const hybrid_cfg_t* cfg,
                                    lz4_sample_stats_t* stats,
                                    double* sample_ratio_pct_out) {
    size_t sample_blocks;
    size_t sample_bytes = 0;
    size_t sample_comp_bytes = 0;
    char* tmp = NULL;
    size_t prev_block = SIZE_MAX;

    if (sample_ratio_pct_out) *sample_ratio_pct_out = 0.0;
    if (!input || !cfg || !stats || input_size == 0 || num_blocks == 0) return -1;
    memset(stats, 0, sizeof(*stats));

    sample_blocks = cfg->adaptive_sample_blocks ? cfg->adaptive_sample_blocks : 8;
    if (sample_blocks > num_blocks) sample_blocks = num_blocks;
    if (sample_blocks == 0) return -1;

    tmp = (char*)malloc((size_t)LZ4_compressBound((int)cfg->block_size));
    if (!tmp) return -1;

    for (size_t i = 0; i < sample_blocks; ++i) {
        const size_t blk_idx = sampled_block_index(i, sample_blocks, num_blocks);
        const size_t blk_sz = block_input_size(input_size, cfg->block_size, blk_idx);
        const char* src;
        int comp_sz;
        double blk_ratio_pct;

        if (blk_idx == prev_block || blk_sz == 0) continue;
        src = (const char*)(const void*)(input + blk_idx * cfg->block_size);
        comp_sz = LZ4_compress_fast(src,
                                    tmp,
                                    (int)blk_sz,
                                    LZ4_compressBound((int)cfg->block_size),
                                    cfg->acceleration > 0 ? cfg->acceleration : 1);
        if (comp_sz <= 0) continue;

        blk_ratio_pct = 100.0 * (double)comp_sz / (double)blk_sz;
        if (blk_ratio_pct < 35.0) stats->low_ratio_blocks++;
        else if (blk_ratio_pct > 70.0) stats->high_ratio_blocks++;

        sample_bytes += blk_sz;
        sample_comp_bytes += (size_t)comp_sz;
        stats->sample_count++;
        prev_block = blk_idx;
    }

    free(tmp);
    if (stats->sample_count == 0 || sample_bytes == 0) return -1;

    stats->mean_ratio_pct = 100.0 * (double)sample_comp_bytes / (double)sample_bytes;
    if (sample_ratio_pct_out) *sample_ratio_pct_out = stats->mean_ratio_pct;
    return 0;
}

static double choose_adaptive_gpu_ratio(const unsigned char* input,
                                        size_t input_size,
                                        size_t num_blocks,
                                        const hybrid_cfg_t* cfg,
                                        double* sample_ratio_pct_out) {
    double ratio;
    lz4_sample_stats_t stats;

    if (sample_ratio_pct_out) *sample_ratio_pct_out = 0.0;
    if (!input || input_size == 0 || num_blocks == 0 || !cfg) return cfg ? cfg->gpu_ratio : 1.0;
    if (cfg->cpu_threads <= 0) return 1.0;

    ratio = cfg->gpu_ratio;
    if (collect_lz4_sample_stats(input, input_size, num_blocks, cfg, &stats, sample_ratio_pct_out) == 0) {
        const double sample_ratio_pct = stats.mean_ratio_pct;
        if (sample_ratio_pct_out) *sample_ratio_pct_out = sample_ratio_pct;

        if (input_size < (8ULL * 1024ULL * 1024ULL)) {
            ratio *= 0.70;
        }
        if (stats.sample_count < 4) {
            ratio *= 0.80;
        }
        if (sample_ratio_pct < 20.0) {
            ratio *= 1.00;
        } else if (sample_ratio_pct < 35.0) {
            ratio *= 0.95;
        } else if (sample_ratio_pct > 70.0) {
            ratio = 0.90 + 0.10 * ratio;
        } else if (sample_ratio_pct > 55.0) {
            ratio = 0.75 + 0.25 * ratio;
        }
        if (stats.low_ratio_blocks * 2 >= stats.sample_count) {
            ratio *= 1.00;
        }
        if (stats.high_ratio_blocks * 2 >= stats.sample_count) {
            ratio = 0.88 + 0.12 * ratio;
        }
        if (cfg->cpu_threads >= 2 && input_size >= (32ULL * 1024ULL * 1024ULL) && sample_ratio_pct < 35.0) {
            ratio *= 0.90;
        }
    }

    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;
    return ratio;
}

static int read_entire_file(const char* path, unsigned char** out_buf, size_t* out_size) {
    FILE* f = NULL;
    unsigned char* buf = NULL;
    struct stat st;

    if (!path || !out_buf || !out_size) return -1;
    *out_buf = NULL;
    *out_size = 0;

    if (stat(path, &st) != 0 || st.st_size < 0) return -1;
    if (st.st_size == 0) return -1;

    buf = (unsigned char*)malloc((size_t)st.st_size);
    if (!buf) return -1;

    f = fopen(path, "rb");
    if (!f) {
        free(buf);
        return -1;
    }
    if (fread(buf, 1, (size_t)st.st_size, f) != (size_t)st.st_size) {
        fclose(f);
        free(buf);
        return -1;
    }
    fclose(f);
    *out_buf = buf;
    *out_size = (size_t)st.st_size;
    return 0;
}

static int write_entire_file(const char* path, const unsigned char* buf, size_t size) {
    FILE* f = NULL;
    if (!path || !buf || size == 0) return -1;
    f = fopen(path, "wb");
    if (!f) return -1;
    if (fwrite(buf, 1, size, f) != size) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static int create_temp_path(char* path_buf, size_t path_buf_size, const char* templ) {
    int fd;
    if (!path_buf || path_buf_size == 0 || !templ) return -1;
    if (strlen(templ) + 1 > path_buf_size) return -1;
    memcpy(path_buf, templ, strlen(templ) + 1);
    fd = mkstemp(path_buf);
    if (fd < 0) return -1;
    close(fd);
    remove(path_buf);
    return 0;
}

static int ocl_init(ocl_env_t* ocl, int hash_log) {
    cl_int err;
    cl_platform_id pf = NULL;
    cl_device_type pref_type = preferred_opencl_device_type();
    if (!ocl) return -1;
    memset(ocl, 0, sizeof(*ocl));
    lz4_gpu_workspace_init(&ocl->ws);

    err = clGetPlatformIDs(1, &pf, NULL);
    if (err != CL_SUCCESS || pf == NULL) {
        fprintf(stderr, "OpenCL init failed: clGetPlatformIDs err=%d\n", err);
        return -1;
    }

    err = clGetDeviceIDs(pf, pref_type, 1, &ocl->dev, NULL);
    if (err != CL_SUCCESS && pref_type != CL_DEVICE_TYPE_GPU) err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &ocl->dev, NULL);
    if (err != CL_SUCCESS && pref_type != CL_DEVICE_TYPE_DEFAULT) err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_DEFAULT, 1, &ocl->dev, NULL);
    if (err != CL_SUCCESS && pref_type != CL_DEVICE_TYPE_ALL) err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_ALL, 1, &ocl->dev, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL init failed: clGetDeviceIDs err=%d\n", err);
        return -1;
    }

    ocl->ctx = clCreateContext(NULL, 1, &ocl->dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || !ocl->ctx) {
        fprintf(stderr, "OpenCL init failed: clCreateContext err=%d\n", err);
        return -1;
    }

    ocl->queue = clCreateCommandQueue(ocl->ctx, ocl->dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || !ocl->queue) {
        fprintf(stderr, "OpenCL init failed: clCreateCommandQueue err=%d\n", err);
        return -1;
    }

    ocl->prog = lz4_load_program(ocl->ctx, ocl->dev, hash_log);
    if (!ocl->prog) {
        fprintf(stderr, "OpenCL init failed: lz4_load_program returned NULL\n");
        return -1;
    }

    ocl->kcomp = clCreateKernel(ocl->prog, "lz4_compress_block", &err);
    if (err != CL_SUCCESS || !ocl->kcomp) {
        fprintf(stderr, "OpenCL init failed: create kernel lz4_compress_block err=%d\n", err);
        return -1;
    }

    ocl->kpack = clCreateKernel(ocl->prog, "lz4_pack_blocks", &err);
    if (err != CL_SUCCESS || !ocl->kpack) {
        fprintf(stderr, "OpenCL init failed: create kernel lz4_pack_blocks err=%d\n", err);
        return -1;
    }

    ocl->kdec = clCreateKernel(ocl->prog, "lz4_decompress_blocks", &err);
    if (err != CL_SUCCESS || !ocl->kdec) {
        fprintf(stderr, "OpenCL init failed: create kernel lz4_decompress_blocks err=%d\n", err);
        return -1;
    }

    return 0;
}

static void ocl_free(ocl_env_t* ocl) {
    if (!ocl) return;
    lz4_gpu_workspace_free(&ocl->ws);
    if (ocl->kdec) clReleaseKernel(ocl->kdec);
    if (ocl->kpack) clReleaseKernel(ocl->kpack);
    if (ocl->kcomp) clReleaseKernel(ocl->kcomp);
    if (ocl->prog) clReleaseProgram(ocl->prog);
    if (ocl->queue) clReleaseCommandQueue(ocl->queue);
    if (ocl->ctx) clReleaseContext(ocl->ctx);
    memset(ocl, 0, sizeof(*ocl));
}

static size_t sanitize_local_size(cl_command_queue queue, size_t requested, size_t upper_blocks) {
    size_t l = requested;
    cl_device_id qdev = NULL;
    size_t max_wg = 1;
    if (upper_blocks == 0) return 1;
    if (l == 0) l = 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    }
    if (l > max_wg) l = max_wg;
    if (l > upper_blocks) l = upper_blocks;
    if (l == 0) l = 1;
    return l;
}

static size_t round_up_size(size_t v, size_t align) {
    if (align == 0) return v;
    return ((v + align - 1) / align) * align;
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
    if (!pack_kernel) return 0;
    if (force_set) return force_value;
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

static size_t choose_comp_worker_count(cl_command_queue queue, size_t num_blocks, size_t local_size) {
    cl_device_id qdev = NULL;
    cl_uint cu = 1;
    size_t default_wi_per_cu;
    size_t wi_per_cu;
    size_t target;

    if (num_blocks == 0) return 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    }
    if (cu == 0) cu = 1;

    default_wi_per_cu = (num_blocks >= 4096) ? 16 : 24;
    wi_per_cu = parse_wi_per_cu_env("LZ4_GPU_COMP_WI_PER_CU", "LZ4_GPU_WI_PER_CU", default_wi_per_cu);
    target = (size_t)cu * wi_per_cu;
    if (target < local_size) target = local_size;
    if (target > num_blocks) target = num_blocks;
    if (target == 0) target = 1;
    return target;
}

static size_t choose_decomp_worker_count(cl_command_queue queue, size_t num_blocks, size_t local_size) {
    cl_device_id qdev = NULL;
    cl_uint cu = 1;
    size_t default_wi_per_cu;
    size_t wi_per_cu;
    size_t target;

    if (num_blocks == 0) return 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    }
    if (cu == 0) cu = 1;

    default_wi_per_cu = (num_blocks >= 4096) ? 48 : 96;
    wi_per_cu = parse_wi_per_cu_env("LZ4_GPU_DECOMP_WI_PER_CU", "LZ4_GPU_WI_PER_CU", default_wi_per_cu);
    target = (size_t)cu * wi_per_cu;
    if (target < local_size) target = local_size;
    if (target > num_blocks) target = num_blocks;
    if (target == 0) target = 1;
    return target;
}

static int choose_lz4_table_type(size_t block_size) {
    const char* force = getenv("LZ4_FORCE_TABLETYPE");
    if (force && *force) {
        int forced = atoi(force);
        if (forced == 0 || forced == 1) return forced;
    }
    return (block_size <= 65536U) ? 0 : 1;
}

static int zero_cl_buffer(cl_command_queue queue, cl_mem buf, size_t bytes) {
    if (!buf || bytes == 0) return 0;
#if defined(CL_VERSION_1_2)
    {
        static const cl_uint z = 0;
        cl_int err = clEnqueueFillBuffer(queue, buf, &z, sizeof(z), 0, bytes, 0, NULL, NULL);
        if (err == CL_SUCCESS) {
            clFinish(queue);
            return 0;
        }
    }
#endif
    {
        cl_int err;
        void* mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !mapped) return -1;
        memset(mapped, 0, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
        clFinish(queue);
    }
    return 0;
}

static void* cpu_comp_worker(void* argp) {
    cpu_comp_worker_arg_t* arg = (cpu_comp_worker_arg_t*)argp;
    cpu_comp_job_t* job = arg->job;
    const size_t n = job->num_blocks;
    for (size_t i = arg->thread_idx; i < n; i += arg->thread_count) {
        const size_t g = job->block_indices ? job->block_indices[i] : i;
        const size_t in_sz = block_input_size(job->src_size, job->block_size, g);
        const unsigned char* src_ptr = job->src + g * job->block_size;
        unsigned char* dst_ptr = job->out_slots + i * job->out_slot_size;
        int csz;
        if (in_sz == 0 || in_sz > (size_t)INT32_MAX || job->out_slot_size > (size_t)INT32_MAX) {
            job->err = 1;
            return NULL;
        }
        csz = LZ4_compress_fast((const char*)src_ptr,
                                (char*)dst_ptr,
                                (int)in_sz,
                                (int)job->out_slot_size,
                                (job->acceleration > 0) ? job->acceleration : 1);
        if (csz <= 0) {
            job->err = 1;
            return NULL;
        }
        job->out_sizes[i] = (uint32_t)csz;
    }
    return NULL;
}

static void* cpu_comp_top(void* argp) {
    cpu_comp_job_t* job = (cpu_comp_job_t*)argp;
    uint64_t t0 = get_us();
    pthread_t* tids = NULL;
    cpu_comp_worker_arg_t* args = NULL;
    int tcount = job->num_threads;

    if (job->num_blocks == 0) {
        job->elapsed_us = 0;
        return NULL;
    }
    if (tcount <= 0) tcount = 1;
    if ((size_t)tcount > job->num_blocks) tcount = (int)job->num_blocks;

    if (tcount == 1) {
        cpu_comp_worker_arg_t arg = { .thread_idx = 0, .thread_count = 1, .job = job };
        cpu_comp_worker(&arg);
        job->elapsed_us = get_us() - t0;
        return NULL;
    }

    tids = (pthread_t*)malloc((size_t)tcount * sizeof(pthread_t));
    args = (cpu_comp_worker_arg_t*)malloc((size_t)tcount * sizeof(cpu_comp_worker_arg_t));
    if (!tids || !args) {
        job->err = 1;
        free(tids);
        free(args);
        return NULL;
    }

    for (int i = 0; i < tcount; ++i) {
        args[i].thread_idx = (size_t)i;
        args[i].thread_count = (size_t)tcount;
        args[i].job = job;
        if (pthread_create(&tids[i], NULL, cpu_comp_worker, &args[i]) != 0) {
            job->err = 1;
            tcount = i;
            break;
        }
    }
    for (int i = 0; i < tcount; ++i) pthread_join(tids[i], NULL);
    job->elapsed_us = get_us() - t0;
    free(tids);
    free(args);
    return NULL;
}

static void* cpu_decomp_worker(void* argp) {
    cpu_decomp_worker_arg_t* arg = (cpu_decomp_worker_arg_t*)argp;
    cpu_decomp_job_t* job = arg->job;

    for (size_t i = arg->thread_idx; i < job->num_blocks; i += arg->thread_count) {
        const size_t g = job->block_indices ? job->block_indices[i] : i;
        const int csz = (int)job->comp_sizes[i];
        unsigned char* dst_ptr = job->out_full + g * job->block_size;
        const unsigned char* src_ptr = job->comp_data + (size_t)job->comp_offsets[i];
        const int dsz = LZ4_decompress_safe((const char*)src_ptr, (char*)dst_ptr, csz, (int)job->block_size);
        if (dsz < 0) {
            job->err = 1;
            return NULL;
        }
        job->out_sizes[i] = (uint32_t)dsz;
    }
    return NULL;
}

static void* cpu_decomp_top(void* argp) {
    cpu_decomp_job_t* job = (cpu_decomp_job_t*)argp;
    uint64_t t0 = get_us();
    pthread_t* tids = NULL;
    cpu_decomp_worker_arg_t* args = NULL;
    int tcount = job->num_threads;

    if (job->num_blocks == 0) {
        job->elapsed_us = 0;
        return NULL;
    }
    if (tcount <= 0) tcount = 1;
    if ((size_t)tcount > job->num_blocks) tcount = (int)job->num_blocks;

    if (tcount == 1) {
        cpu_decomp_worker_arg_t arg = { .thread_idx = 0, .thread_count = 1, .job = job };
        cpu_decomp_worker(&arg);
        job->elapsed_us = get_us() - t0;
        return NULL;
    }

    tids = (pthread_t*)malloc((size_t)tcount * sizeof(pthread_t));
    args = (cpu_decomp_worker_arg_t*)malloc((size_t)tcount * sizeof(cpu_decomp_worker_arg_t));
    if (!tids || !args) {
        job->err = 1;
        free(tids);
        free(args);
        return NULL;
    }

    for (int i = 0; i < tcount; ++i) {
        args[i].thread_idx = (size_t)i;
        args[i].thread_count = (size_t)tcount;
        args[i].job = job;
        if (pthread_create(&tids[i], NULL, cpu_decomp_worker, &args[i]) != 0) {
            job->err = 1;
            tcount = i;
            break;
        }
    }
    for (int i = 0; i < tcount; ++i) pthread_join(tids[i], NULL);
    job->elapsed_us = get_us() - t0;
    free(tids);
    free(args);
    return NULL;
}

static int gpu_compress_blocks(ocl_env_t* ocl,
                               const unsigned char* src,
                               size_t src_size,
                               size_t block_size,
                               int acceleration,
                               int local_size,
                               int hash_log,
                               uint32_t** out_sizes,
                               uint32_t** out_offsets,
                               unsigned char** out_slots,
                               size_t* out_slot_size,
                               uint64_t* kernel_us) {
    cl_int err = CL_SUCCESS;
    uint32_t* h_block_info = NULL;
    uint32_t* h_out_offsets = NULL;
    uint32_t* h_packed_offsets = NULL;
    uint32_t* h_sizes = NULL;
    unsigned char* h_out = NULL;
    unsigned char* packed_out = NULL;
    size_t num_blocks;
    size_t single_block_max_out;
    size_t lsz;
    size_t gsz;
    int tableType;
    int inputSize;
    int totalBlocks;
    int globalIndexBase = 0;
    uint32_t epoch_base;
    size_t dict_entries_per_worker;
    size_t dict_bytes;
    size_t prev_dict_capacity;
    size_t packed_total = 0;
    size_t sparse_total = 0;
    int use_device_compaction = 0;

    if (!ocl || !src || src_size == 0 || block_size == 0 || !out_sizes || !out_offsets || !out_slots || !out_slot_size || !kernel_us) return -1;

    *out_sizes = NULL;
    *out_offsets = NULL;
    *out_slots = NULL;
    *out_slot_size = 0;
    *kernel_us = 0;

    num_blocks = (src_size + block_size - 1) / block_size;
    if (num_blocks == 0) return -1;
    if (num_blocks > (size_t)INT32_MAX) return -1;

    single_block_max_out = (size_t)((double)block_size * 1.1 + 64.0);
    if (single_block_max_out < (size_t)LZ4_compressBound((int)block_size)) {
        single_block_max_out = (size_t)LZ4_compressBound((int)block_size);
    }

    h_block_info = (uint32_t*)malloc(num_blocks * 2U * sizeof(uint32_t));
    h_out_offsets = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    h_packed_offsets = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    h_sizes = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    if (!h_block_info || !h_out_offsets || !h_packed_offsets || !h_sizes) goto fail;

    for (size_t i = 0; i < num_blocks; ++i) {
        const size_t start = i * block_size;
        const size_t bsz = (i + 1 == num_blocks) ? (src_size - start) : block_size;
        h_block_info[i * 2 + 0] = (uint32_t)start;
        h_block_info[i * 2 + 1] = (uint32_t)bsz;
        h_out_offsets[i] = (uint32_t)(i * single_block_max_out);
    }

    lsz = sanitize_local_size(ocl->queue, (size_t)((local_size > 0) ? local_size : 1), num_blocks);
    gsz = round_up_size(choose_comp_worker_count(ocl->queue, num_blocks, lsz), lsz);
    if (gsz == 0) gsz = 1;

    tableType = choose_lz4_table_type(block_size);

    ocl->ws.in_buf = ensure_buffer(ocl->ctx, ocl->ws.in_buf, src_size, &ocl->ws.current_in_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.in_buf) goto fail;
    ocl->ws.out_buf = ensure_buffer(ocl->ctx, ocl->ws.out_buf, num_blocks * single_block_max_out, &ocl->ws.current_out_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.out_buf) goto fail;
    ocl->ws.block_info_buf = ensure_buffer(ocl->ctx, ocl->ws.block_info_buf, num_blocks * 2U * sizeof(uint32_t), &ocl->ws.current_blocks_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.block_info_buf) goto fail;
    ocl->ws.out_offsets_buf = ensure_buffer(ocl->ctx, ocl->ws.out_offsets_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_out_offsets_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.out_offsets_buf) goto fail;
    ocl->ws.output_size_buf = ensure_buffer(ocl->ctx, ocl->ws.output_size_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_osize_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.output_size_buf) goto fail;

    dict_entries_per_worker = (size_t)1U << (tableType == 0 ? (hash_log + 1) : hash_log);
    dict_bytes = dict_entries_per_worker * sizeof(cl_uint) * gsz;
    prev_dict_capacity = ocl->ws.current_dict_capacity;
    ocl->ws.dict_buf = ensure_buffer(ocl->ctx, ocl->ws.dict_buf, dict_bytes, &ocl->ws.current_dict_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.dict_buf) goto fail;
    if (ocl->ws.current_dict_capacity != prev_dict_capacity) {
        if (zero_cl_buffer(ocl->queue, ocl->ws.dict_buf, ocl->ws.current_dict_capacity) != 0) goto fail;
    }
    if (ocl->ws.comp_epoch_base == 0) ocl->ws.comp_epoch_base = 1;
    {
        uint32_t blocks_per_worker = (uint32_t)(((size_t)num_blocks + gsz - 1) / gsz);
        uint32_t epochs_needed = blocks_per_worker + 2U;
        uint32_t cur_low = ocl->ws.comp_epoch_base & 0xFFU;
        uint32_t end_low = (ocl->ws.comp_epoch_base + epochs_needed) & 0xFFU;
        int wraps_8bit = (end_low <= cur_low) || (ocl->ws.comp_epoch_base > (uint32_t)(UINT32_MAX - epochs_needed));
        if (wraps_8bit) {
            if (zero_cl_buffer(ocl->queue, ocl->ws.dict_buf, ocl->ws.current_dict_capacity) != 0) goto fail;
            ocl->ws.comp_epoch_base = 1;
        }
    }
    epoch_base = ocl->ws.comp_epoch_base;
    ocl->ws.comp_epoch_base += (uint32_t)(((size_t)num_blocks + gsz - 1) / gsz) + 2U;

    if (write_buffer_mapped(ocl->queue, ocl->ws.in_buf, src, src_size) != 0 ||
        write_buffer_mapped(ocl->queue, ocl->ws.block_info_buf, h_block_info, num_blocks * 2U * sizeof(uint32_t)) != 0 ||
        write_buffer_mapped(ocl->queue, ocl->ws.out_offsets_buf, h_out_offsets, num_blocks * sizeof(uint32_t)) != 0) {
        goto fail;
    }

    inputSize = (int)src_size;
    totalBlocks = (int)num_blocks;

    err = CL_SUCCESS;
    err |= clSetKernelArg(ocl->kcomp, 0, sizeof(cl_mem), &ocl->ws.in_buf);
    err |= clSetKernelArg(ocl->kcomp, 1, sizeof(cl_mem), &ocl->ws.out_buf);
    err |= clSetKernelArg(ocl->kcomp, 2, sizeof(cl_mem), &ocl->ws.output_size_buf);
    err |= clSetKernelArg(ocl->kcomp, 3, sizeof(cl_mem), &ocl->ws.block_info_buf);
    err |= clSetKernelArg(ocl->kcomp, 4, sizeof(cl_mem), &ocl->ws.out_offsets_buf);
    err |= clSetKernelArg(ocl->kcomp, 5, sizeof(int), &totalBlocks);
    err |= clSetKernelArg(ocl->kcomp, 6, sizeof(int), &inputSize);
    err |= clSetKernelArg(ocl->kcomp, 7, sizeof(int), &tableType);
    err |= clSetKernelArg(ocl->kcomp, 8, sizeof(int), &acceleration);
    err |= clSetKernelArg(ocl->kcomp, 9, sizeof(int), &globalIndexBase);
    err |= clSetKernelArg(ocl->kcomp, 10, sizeof(cl_mem), &ocl->ws.dict_buf);
    err |= clSetKernelArg(ocl->kcomp, 11, sizeof(uint32_t), &epoch_base);
    if (err != CL_SUCCESS) goto fail;

    {
        const uint64_t t0 = get_us();
    sparse_total = num_blocks * single_block_max_out;

        err = clEnqueueNDRangeKernel(ocl->queue, ocl->kcomp, 1, NULL, &gsz, &lsz, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto fail;
        clFinish(ocl->queue);
        *kernel_us = get_us() - t0;
    }

    {
        void* mapped_sizes = clEnqueueMapBuffer(ocl->queue, ocl->ws.output_size_buf, CL_TRUE, CL_MAP_READ,
                                                0, num_blocks * sizeof(uint32_t), 0, NULL, NULL, &err);
        if (err == CL_SUCCESS && mapped_sizes) {
            memcpy(h_sizes, mapped_sizes, num_blocks * sizeof(uint32_t));
            clEnqueueUnmapMemObject(ocl->queue, ocl->ws.output_size_buf, mapped_sizes, 0, NULL, NULL);
            clFinish(ocl->queue);
        } else {
            err = clEnqueueReadBuffer(ocl->queue, ocl->ws.output_size_buf, CL_TRUE, 0, num_blocks * sizeof(uint32_t), h_sizes, 0, NULL, NULL);
            if (err != CL_SUCCESS) goto fail;
        }
    }

    for (size_t i = 0; i < num_blocks; ++i) {
        if (h_sizes[i] == 0xFFFFFFFFU) goto fail;
        h_packed_offsets[i] = (uint32_t)packed_total;
        packed_total += (size_t)h_sizes[i];
    }

    use_device_compaction = lz4_should_use_device_compaction(packed_total, sparse_total, num_blocks, ocl->kpack);

    if (packed_total > 0) {
        packed_out = (unsigned char*)malloc(packed_total);
        if (!packed_out) goto fail;
    }

    if (use_device_compaction) {
        size_t pack_global;
        uint32_t totalBlocks32 = (uint32_t)num_blocks;
        ocl->ws.packed_offsets_buf = ensure_buffer(ocl->ctx,
                                                   ocl->ws.packed_offsets_buf,
                                                   num_blocks * sizeof(uint32_t),
                                                   &ocl->ws.current_packed_offsets_capacity,
                                                   &err);
        if (err != CL_SUCCESS || !ocl->ws.packed_offsets_buf) goto fail;
        ocl->ws.packed_out_buf = ensure_buffer(ocl->ctx,
                                               ocl->ws.packed_out_buf,
                                               packed_total,
                                               &ocl->ws.current_packed_out_capacity,
                                               &err);
        if (packed_total > 0 && (err != CL_SUCCESS || !ocl->ws.packed_out_buf)) goto fail;
        if (write_buffer_mapped(ocl->queue,
                                ocl->ws.packed_offsets_buf,
                                h_packed_offsets,
                                num_blocks * sizeof(uint32_t)) != 0) {
            goto fail;
        }

        err = CL_SUCCESS;
        err |= clSetKernelArg(ocl->kpack, 0, sizeof(cl_mem), &ocl->ws.out_buf);
        err |= clSetKernelArg(ocl->kpack, 1, sizeof(cl_mem), &ocl->ws.packed_out_buf);
        err |= clSetKernelArg(ocl->kpack, 2, sizeof(cl_mem), &ocl->ws.out_offsets_buf);
        err |= clSetKernelArg(ocl->kpack, 3, sizeof(cl_mem), &ocl->ws.packed_offsets_buf);
        err |= clSetKernelArg(ocl->kpack, 4, sizeof(cl_mem), &ocl->ws.output_size_buf);
        err |= clSetKernelArg(ocl->kpack, 5, sizeof(uint32_t), &totalBlocks32);
        if (err != CL_SUCCESS) goto fail;

        pack_global = round_up_size(num_blocks, lsz);
        if (pack_global == 0) pack_global = lsz;
        {
            const uint64_t t0 = get_us();
            err = clEnqueueNDRangeKernel(ocl->queue, ocl->kpack, 1, NULL, &pack_global, &lsz, 0, NULL, NULL);
            if (err != CL_SUCCESS) goto fail;
            clFinish(ocl->queue);
            *kernel_us += get_us() - t0;
        }

        if (packed_total > 0) {
            void* mapped_packed = clEnqueueMapBuffer(ocl->queue, ocl->ws.packed_out_buf, CL_TRUE, CL_MAP_READ,
                                                     0, packed_total, 0, NULL, NULL, &err);
            if (err == CL_SUCCESS && mapped_packed) {
                memcpy(packed_out, mapped_packed, packed_total);
                clEnqueueUnmapMemObject(ocl->queue, ocl->ws.packed_out_buf, mapped_packed, 0, NULL, NULL);
                clFinish(ocl->queue);
            } else {
                err = clEnqueueReadBuffer(ocl->queue, ocl->ws.packed_out_buf, CL_TRUE, 0, packed_total, packed_out, 0, NULL, NULL);
                if (err != CL_SUCCESS) goto fail;
            }
        }
    } else {
        if (sparse_total > 0) {
            h_out = (unsigned char*)malloc(sparse_total);
            if (!h_out) goto fail;
            {
                void* mapped_out = clEnqueueMapBuffer(ocl->queue, ocl->ws.out_buf, CL_TRUE, CL_MAP_READ,
                                                      0, sparse_total, 0, NULL, NULL, &err);
                if (err == CL_SUCCESS && mapped_out) {
                    memcpy(h_out, mapped_out, sparse_total);
                    clEnqueueUnmapMemObject(ocl->queue, ocl->ws.out_buf, mapped_out, 0, NULL, NULL);
                    clFinish(ocl->queue);
                } else {
                    err = clEnqueueReadBuffer(ocl->queue, ocl->ws.out_buf, CL_TRUE, 0, sparse_total, h_out, 0, NULL, NULL);
                    if (err != CL_SUCCESS) goto fail;
                }
            }
        }
        for (size_t i = 0; i < num_blocks; ++i) {
            memcpy(packed_out + h_packed_offsets[i], h_out + i * single_block_max_out, (size_t)h_sizes[i]);
        }
    }

    *out_sizes = h_sizes;
    *out_offsets = h_packed_offsets;
    *out_slots = packed_out;
    *out_slot_size = packed_total;

    h_sizes = NULL;
    h_packed_offsets = NULL;
    packed_out = NULL;

    free(h_block_info);
    free(h_out_offsets);
    free(h_packed_offsets);
    free(h_sizes);
    free(h_out);
    free(packed_out);
    return 0;

fail:
    free(h_block_info);
    free(h_out_offsets);
    free(h_packed_offsets);
    free(h_sizes);
    free(h_out);
    free(packed_out);
    return -1;
}

static int gpu_decompress_blocks(ocl_env_t* ocl,
                                 const unsigned char* comp_data,
                                 const uint32_t* comp_sizes,
                                 size_t num_blocks,
                                 size_t block_size,
                                 int local_size,
                                 unsigned char* out_full,
                                 uint32_t* out_sizes,
                                 uint64_t* kernel_us) {
    cl_int err = CL_SUCCESS;
    uint32_t* h_comp_off = NULL;
    uint32_t* h_out_off = NULL;
    uint32_t* h_max_out = NULL;
    size_t comp_total = 0;
    size_t lsz;
    size_t gsz;
    uint32_t totalBlocks;

    if (!ocl || !comp_data || !comp_sizes || !out_full || !out_sizes || !kernel_us) return -1;
    if (num_blocks == 0) {
        *kernel_us = 0;
        return 0;
    }
    if (num_blocks > (size_t)UINT32_MAX) return -1;

    h_comp_off = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    h_out_off = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    h_max_out = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    if (!h_comp_off || !h_out_off || !h_max_out) goto fail;

    for (size_t i = 0; i < num_blocks; ++i) {
        h_comp_off[i] = (uint32_t)comp_total;
        h_out_off[i] = (uint32_t)(i * block_size);
        h_max_out[i] = (uint32_t)block_size;
        comp_total += (size_t)comp_sizes[i];
    }

    ocl->ws.in_buf = ensure_buffer(ocl->ctx, ocl->ws.in_buf, comp_total, &ocl->ws.current_in_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.in_buf) goto fail;
    ocl->ws.out_buf = ensure_buffer(ocl->ctx, ocl->ws.out_buf, num_blocks * block_size, &ocl->ws.current_out_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.out_buf) goto fail;
    ocl->ws.decomp_comp_off_buf = ensure_buffer(ocl->ctx, ocl->ws.decomp_comp_off_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_decomp_comp_off_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.decomp_comp_off_buf) goto fail;
    ocl->ws.decomp_comp_size_buf = ensure_buffer(ocl->ctx, ocl->ws.decomp_comp_size_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_decomp_comp_size_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.decomp_comp_size_buf) goto fail;
    ocl->ws.decomp_out_off_buf = ensure_buffer(ocl->ctx, ocl->ws.decomp_out_off_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_decomp_out_off_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.decomp_out_off_buf) goto fail;
    ocl->ws.decomp_max_out_buf = ensure_buffer(ocl->ctx, ocl->ws.decomp_max_out_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_decomp_max_out_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.decomp_max_out_buf) goto fail;
    ocl->ws.decomp_sizes_out_buf = ensure_buffer(ocl->ctx, ocl->ws.decomp_sizes_out_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_decomp_sizes_out_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.decomp_sizes_out_buf) goto fail;

    if (write_buffer_mapped(ocl->queue, ocl->ws.in_buf, comp_data, comp_total) != 0 ||
        write_buffer_mapped(ocl->queue, ocl->ws.decomp_comp_off_buf, h_comp_off, num_blocks * sizeof(uint32_t)) != 0 ||
        write_buffer_mapped(ocl->queue, ocl->ws.decomp_comp_size_buf, comp_sizes, num_blocks * sizeof(uint32_t)) != 0 ||
        write_buffer_mapped(ocl->queue, ocl->ws.decomp_out_off_buf, h_out_off, num_blocks * sizeof(uint32_t)) != 0 ||
        write_buffer_mapped(ocl->queue, ocl->ws.decomp_max_out_buf, h_max_out, num_blocks * sizeof(uint32_t)) != 0) {
        goto fail;
    }

    totalBlocks = (uint32_t)num_blocks;
    err = CL_SUCCESS;
    err |= clSetKernelArg(ocl->kdec, 0, sizeof(cl_mem), &ocl->ws.in_buf);
    err |= clSetKernelArg(ocl->kdec, 1, sizeof(cl_mem), &ocl->ws.out_buf);
    err |= clSetKernelArg(ocl->kdec, 2, sizeof(cl_mem), &ocl->ws.decomp_comp_off_buf);
    err |= clSetKernelArg(ocl->kdec, 3, sizeof(cl_mem), &ocl->ws.decomp_comp_size_buf);
    err |= clSetKernelArg(ocl->kdec, 4, sizeof(cl_mem), &ocl->ws.decomp_out_off_buf);
    err |= clSetKernelArg(ocl->kdec, 5, sizeof(cl_mem), &ocl->ws.decomp_max_out_buf);
    err |= clSetKernelArg(ocl->kdec, 6, sizeof(cl_mem), &ocl->ws.decomp_sizes_out_buf);
    err |= clSetKernelArg(ocl->kdec, 7, sizeof(uint32_t), &totalBlocks);
    if (err != CL_SUCCESS) goto fail;

    lsz = sanitize_local_size(ocl->queue, (size_t)((local_size > 0) ? local_size : 1), num_blocks);
    gsz = round_up_size(choose_decomp_worker_count(ocl->queue, num_blocks, lsz), lsz);
    if (gsz == 0) gsz = 1;

    {
        const uint64_t t0 = get_us();
        err = clEnqueueNDRangeKernel(ocl->queue, ocl->kdec, 1, NULL, &gsz, &lsz, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto fail;
        clFinish(ocl->queue);
        *kernel_us = get_us() - t0;
    }

    {
        void* mapped_sizes = clEnqueueMapBuffer(ocl->queue, ocl->ws.decomp_sizes_out_buf, CL_TRUE, CL_MAP_READ,
                                                0, num_blocks * sizeof(uint32_t), 0, NULL, NULL, &err);
        if (err == CL_SUCCESS && mapped_sizes) {
            memcpy(out_sizes, mapped_sizes, num_blocks * sizeof(uint32_t));
            clEnqueueUnmapMemObject(ocl->queue, ocl->ws.decomp_sizes_out_buf, mapped_sizes, 0, NULL, NULL);
            clFinish(ocl->queue);
        } else {
            err = clEnqueueReadBuffer(ocl->queue, ocl->ws.decomp_sizes_out_buf, CL_TRUE, 0, num_blocks * sizeof(uint32_t), out_sizes, 0, NULL, NULL);
            if (err != CL_SUCCESS) goto fail;
        }
    }

    for (size_t i = 0; i < num_blocks; ++i) {
        if (out_sizes[i] == 0xFFFFFFFFU) goto fail;
    }

    {
        void* mapped_out = clEnqueueMapBuffer(ocl->queue, ocl->ws.out_buf, CL_TRUE, CL_MAP_READ,
                                              0, num_blocks * block_size, 0, NULL, NULL, &err);
        if (err == CL_SUCCESS && mapped_out) {
            const unsigned char* out_ptr = (const unsigned char*)mapped_out;
            for (size_t i = 0; i < num_blocks; ++i) {
                memcpy(out_full + i * block_size, out_ptr + i * block_size, out_sizes[i]);
            }
            clEnqueueUnmapMemObject(ocl->queue, ocl->ws.out_buf, mapped_out, 0, NULL, NULL);
            clFinish(ocl->queue);
        } else {
            unsigned char* tmp_out = (unsigned char*)malloc(num_blocks * block_size);
            if (!tmp_out) goto fail;
            err = clEnqueueReadBuffer(ocl->queue, ocl->ws.out_buf, CL_TRUE, 0, num_blocks * block_size, tmp_out, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                free(tmp_out);
                goto fail;
            }
            for (size_t i = 0; i < num_blocks; ++i) {
                memcpy(out_full + i * block_size, tmp_out + i * block_size, out_sizes[i]);
            }
            free(tmp_out);
        }
    }
    free(h_comp_off);
    free(h_out_off);
    free(h_max_out);
    return 0;

fail:
    free(h_comp_off);
    free(h_out_off);
    free(h_max_out);
    return -1;
}

static int hybrid_compress_memory(ocl_env_t* ocl,
                                  const unsigned char* input,
                                  size_t input_size,
                                  const hybrid_cfg_t* cfg,
                                  unsigned char** out_buf,
                                  size_t* out_size,
                                  hybrid_metrics_t* m) {
    size_t num_blocks;
    size_t gpu_blocks;
    size_t cpu_blocks;
    size_t cpu_slot_size = 0;
    double effective_gpu_ratio;
    double sample_ratio_pct = 0.0;
    uint32_t* all_sizes = NULL;
    unsigned char* final_out = NULL;
    size_t final_sz = 0;
    size_t* gpu_block_indices = NULL;
    size_t* cpu_block_indices = NULL;
    size_t* gpu_pos_by_block = NULL;
    size_t* cpu_pos_by_block = NULL;
    unsigned char* gpu_input = NULL;
    size_t gpu_input_size = 0;

    cpu_comp_job_t cpu_job;
    pthread_t cpu_thread;
    int cpu_thread_started = 0;

    uint32_t* gpu_sizes = NULL;
    uint32_t* gpu_offsets = NULL;
    unsigned char* gpu_slots = NULL;
    size_t gpu_slot_size = 0;

    memset(&cpu_job, 0, sizeof(cpu_job));
    if (m) memset(m, 0, sizeof(*m));

    if (!ocl || !input || input_size == 0 || !cfg || !out_buf || !out_size) return -1;
    *out_buf = NULL;
    *out_size = 0;

    num_blocks = (input_size + cfg->block_size - 1) / cfg->block_size;
    effective_gpu_ratio = cfg->gpu_ratio;
    if (cfg->cpu_threads <= 0) {
        effective_gpu_ratio = 1.0;
    } else if (cfg->adaptive_split) {
        effective_gpu_ratio = choose_adaptive_gpu_ratio(input, input_size, num_blocks, cfg, &sample_ratio_pct);
    }
    gpu_blocks = (size_t)((double)num_blocks * effective_gpu_ratio + 0.5);
    if (gpu_blocks > num_blocks) gpu_blocks = num_blocks;
    if (partition_blocks_distributed(num_blocks,
                                     gpu_blocks,
                                     &gpu_block_indices,
                                     &gpu_blocks,
                                     &cpu_block_indices,
                                     &cpu_blocks) != 0) {
        goto fail;
    }

    if (cfg->verbose && cfg->adaptive_split) {
        fprintf(stderr,
                "Adaptive split: sample_ratio=%.2f%% requested_gpu_ratio=%.2f effective_gpu_ratio=%.2f gpu_blocks=%zu cpu_blocks=%zu\n",
                sample_ratio_pct,
                cfg->gpu_ratio,
                effective_gpu_ratio,
                gpu_blocks,
                cpu_blocks);
    }

    all_sizes = (uint32_t*)calloc(num_blocks, sizeof(uint32_t));
    if (!all_sizes) goto fail;

    if (cpu_blocks > 0) {
        cpu_slot_size = (size_t)LZ4_compressBound((int)cfg->block_size);
        {
            size_t gpu_style_slot = (size_t)((double)cfg->block_size * 1.1 + 64.0);
            if (gpu_style_slot > cpu_slot_size) cpu_slot_size = gpu_style_slot;
        }
        cpu_job.src = input;
        cpu_job.src_size = input_size;
        cpu_job.block_size = cfg->block_size;
        cpu_job.block_indices = cpu_block_indices;
        cpu_job.num_blocks = cpu_blocks;
        cpu_job.num_threads = cfg->cpu_threads;
        cpu_job.acceleration = cfg->acceleration;
        cpu_job.out_slot_size = cpu_slot_size;
        cpu_job.out_slots = (unsigned char*)malloc(cpu_blocks * cpu_slot_size);
        cpu_job.out_sizes = (uint32_t*)calloc(cpu_blocks, sizeof(uint32_t));
        if (!cpu_job.out_slots || !cpu_job.out_sizes) goto fail;

    }

    {
        const uint64_t parallel_t0 = get_us();
        if (cpu_blocks > 0) {
            if (pthread_create(&cpu_thread, NULL, cpu_comp_top, &cpu_job) != 0) goto fail;
            cpu_thread_started = 1;
        }

        if (gpu_blocks > 0) {
            gpu_input = (unsigned char*)calloc(gpu_blocks, cfg->block_size);
            if (!gpu_input) goto fail;
            gpu_input_size = gpu_blocks * cfg->block_size;
            for (size_t i = 0; i < gpu_blocks; ++i) {
                const size_t g = gpu_block_indices[i];
                const size_t blk_sz = block_input_size(input_size, cfg->block_size, g);
                memcpy(gpu_input + i * cfg->block_size, input + g * cfg->block_size, blk_sz);
            }
            if (gpu_block_indices[gpu_blocks - 1] == num_blocks - 1) {
                gpu_input_size -= cfg->block_size - block_input_size(input_size, cfg->block_size, num_blocks - 1);
            }
            if (gpu_compress_blocks(ocl,
                                    gpu_input,
                                    gpu_input_size,
                                    cfg->block_size,
                                    cfg->acceleration,
                                    cfg->local_size,
                                    cfg->hash_log,
                                    &gpu_sizes,
                                    &gpu_offsets,
                                    &gpu_slots,
                                    &gpu_slot_size,
                                    m ? &m->gpu_kernel_us : &(uint64_t){0}) != 0) {
                goto fail;
            }
        }

        if (cpu_thread_started) {
            pthread_join(cpu_thread, NULL);
            cpu_thread_started = 0;
            if (cpu_job.err) goto fail;
            if (m) m->cpu_kernel_us = cpu_job.elapsed_us;
        }
        if (m) {
            m->parallel_us = get_us() - parallel_t0;
            if (m->parallel_us < m->cpu_kernel_us) m->parallel_us = m->cpu_kernel_us;
            if (m->parallel_us < m->gpu_kernel_us) m->parallel_us = m->gpu_kernel_us;
        }
    }

    for (size_t i = 0; i < gpu_blocks; ++i) all_sizes[gpu_block_indices[i]] = gpu_sizes[i];
    for (size_t i = 0; i < cpu_blocks; ++i) all_sizes[cpu_block_indices[i]] = cpu_job.out_sizes[i];

    gpu_pos_by_block = (size_t*)malloc(num_blocks * sizeof(size_t));
    cpu_pos_by_block = (size_t*)malloc(num_blocks * sizeof(size_t));
    if (!gpu_pos_by_block || !cpu_pos_by_block) goto fail;
    for (size_t i = 0; i < num_blocks; ++i) {
        gpu_pos_by_block[i] = SIZE_MAX;
        cpu_pos_by_block[i] = SIZE_MAX;
    }
    for (size_t i = 0; i < gpu_blocks; ++i) gpu_pos_by_block[gpu_block_indices[i]] = i;
    for (size_t i = 0; i < cpu_blocks; ++i) cpu_pos_by_block[cpu_block_indices[i]] = i;

    final_sz = 16U + num_blocks * 4U;
    for (size_t i = 0; i < num_blocks; ++i) final_sz += (size_t)all_sizes[i];
    final_out = (unsigned char*)malloc(final_sz);
    if (!final_out) goto fail;

    {
        unsigned char* p = final_out;
        uint32_t magic = HYBRID_MAGIC;
        uint32_t nblk = (uint32_t)num_blocks;
        uint32_t bsz = (uint32_t)cfg->block_size;
        uint32_t gblk = (uint32_t)gpu_blocks | HYBRID_GPU_BLOCKS_STRIPED_FLAG;
        memcpy(p, &magic, 4); p += 4;
        memcpy(p, &nblk, 4); p += 4;
        memcpy(p, &bsz, 4); p += 4;
        memcpy(p, &gblk, 4); p += 4;
        memcpy(p, all_sizes, num_blocks * 4U); p += num_blocks * 4U;

        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            if (gpu_pos_by_block[block_idx] != SIZE_MAX) {
                const size_t pos = gpu_pos_by_block[block_idx];
                memcpy(p, gpu_slots + (size_t)gpu_offsets[pos], (size_t)gpu_sizes[pos]);
                p += gpu_sizes[pos];
            } else {
                const size_t pos = cpu_pos_by_block[block_idx];
                memcpy(p, cpu_job.out_slots + pos * cpu_slot_size, (size_t)cpu_job.out_sizes[pos]);
                p += cpu_job.out_sizes[pos];
            }
        }
    }

    *out_buf = final_out;
    *out_size = final_sz;
    free(all_sizes);
    free(gpu_pos_by_block);
    free(cpu_pos_by_block);
    free(gpu_block_indices);
    free(cpu_block_indices);
    free(gpu_input);
    free(cpu_job.out_slots);
    free(cpu_job.out_sizes);
    free(gpu_sizes);
    free(gpu_offsets);
    free(gpu_slots);
    return 0;

fail:
    if (cpu_thread_started) pthread_join(cpu_thread, NULL);
    free(all_sizes);
    free(final_out);
    free(gpu_pos_by_block);
    free(cpu_pos_by_block);
    free(gpu_block_indices);
    free(cpu_block_indices);
    free(gpu_input);
    free(cpu_job.out_slots);
    free(cpu_job.out_sizes);
    free(gpu_sizes);
    free(gpu_offsets);
    free(gpu_slots);
    return -1;
}

static int hybrid_decompress_memory(ocl_env_t* ocl,
                                    const unsigned char* comp,
                                    size_t comp_size,
                                    const hybrid_cfg_t* cfg,
                                    unsigned char** out_buf,
                                    size_t* out_size,
                                    hybrid_metrics_t* m) {
    uint32_t magic, num_blocks_u32, block_size_u32, gpu_blocks_u32;
    size_t num_blocks, block_size, gpu_blocks, cpu_blocks;
    int striped_gpu_layout = 0;
    const uint32_t* sizes = NULL;
    const unsigned char* data_ptr;
    size_t header_size;
    size_t payload_size = 0;
    uint32_t* global_offsets = NULL;
    size_t* gpu_block_indices = NULL;
    size_t* cpu_block_indices = NULL;

    unsigned char* out_full = NULL;
    unsigned char* gpu_comp_data = NULL;
    unsigned char* gpu_out_full = NULL;
    uint32_t* gpu_out_sizes = NULL;
    uint32_t* gpu_comp_sizes = NULL;
    uint32_t* cpu_comp_sizes = NULL;
    uint32_t* cpu_comp_offsets = NULL;
    uint32_t* cpu_out_sizes = NULL;

    cpu_decomp_job_t cpu_job;
    pthread_t cpu_thread;
    int cpu_thread_started = 0;

    memset(&cpu_job, 0, sizeof(cpu_job));
    if (m) memset(m, 0, sizeof(*m));
    if (!ocl || !comp || comp_size < 16 || !cfg || !out_buf || !out_size) return -1;
    *out_buf = NULL;
    *out_size = 0;

    memcpy(&magic, comp + 0, 4);
    memcpy(&num_blocks_u32, comp + 4, 4);
    memcpy(&block_size_u32, comp + 8, 4);
    memcpy(&gpu_blocks_u32, comp + 12, 4);
    if (magic != HYBRID_MAGIC) return -1;

    num_blocks = (size_t)num_blocks_u32;
    block_size = (size_t)block_size_u32;
    striped_gpu_layout = (gpu_blocks_u32 & HYBRID_GPU_BLOCKS_STRIPED_FLAG) != 0;
    gpu_blocks = (size_t)(gpu_blocks_u32 & ~HYBRID_GPU_BLOCKS_STRIPED_FLAG);
    if (num_blocks == 0 || block_size == 0 || gpu_blocks > num_blocks) return -1;

    header_size = 16U + num_blocks * 4U;
    if (header_size > comp_size) return -1;
    sizes = (const uint32_t*)(const void*)(comp + 16);
    data_ptr = comp + header_size;

    for (size_t i = 0; i < num_blocks; ++i) payload_size += (size_t)sizes[i];
    if (header_size + payload_size != comp_size) return -1;

    if (striped_gpu_layout) {
        if (partition_blocks_distributed(num_blocks,
                                         gpu_blocks,
                                         &gpu_block_indices,
                                         &gpu_blocks,
                                         &cpu_block_indices,
                                         &cpu_blocks) != 0) {
            goto fail;
        }
    } else {
        if (partition_blocks_prefix(num_blocks,
                                    gpu_blocks,
                                    &gpu_block_indices,
                                    &gpu_blocks,
                                    &cpu_block_indices,
                                    &cpu_blocks) != 0) {
            goto fail;
        }
    }

    global_offsets = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    if (!global_offsets) goto fail;
    {
        size_t off = 0;
        for (size_t i = 0; i < num_blocks; ++i) {
            global_offsets[i] = (uint32_t)off;
            off += (size_t)sizes[i];
        }
    }

    out_full = (unsigned char*)malloc(num_blocks * block_size);
    if (!out_full) goto fail;

    if (gpu_blocks > 0) {
        size_t gpu_comp_total = 0;
        gpu_comp_sizes = (uint32_t*)malloc(gpu_blocks * sizeof(uint32_t));
        gpu_out_sizes = (uint32_t*)calloc(gpu_blocks, sizeof(uint32_t));
        gpu_out_full = (unsigned char*)malloc(gpu_blocks * block_size);
        if (!gpu_comp_sizes || !gpu_out_sizes || !gpu_out_full) goto fail;
        for (size_t i = 0; i < gpu_blocks; ++i) gpu_comp_total += (size_t)sizes[gpu_block_indices[i]];
        if (gpu_comp_total > 0) {
            size_t off = 0;
            gpu_comp_data = (unsigned char*)malloc(gpu_comp_total);
            if (!gpu_comp_data) goto fail;
            for (size_t i = 0; i < gpu_blocks; ++i) {
                const size_t blk_idx = gpu_block_indices[i];
                const size_t blk_sz = (size_t)sizes[blk_idx];
                gpu_comp_sizes[i] = sizes[blk_idx];
                memcpy(gpu_comp_data + off, data_ptr + (size_t)global_offsets[blk_idx], blk_sz);
                off += blk_sz;
            }
        }
    }
    if (cpu_blocks > 0) {
        cpu_comp_sizes = (uint32_t*)malloc(cpu_blocks * sizeof(uint32_t));
        cpu_comp_offsets = (uint32_t*)malloc(cpu_blocks * sizeof(uint32_t));
        cpu_out_sizes = (uint32_t*)calloc(cpu_blocks, sizeof(uint32_t));
        if (!cpu_comp_sizes || !cpu_comp_offsets || !cpu_out_sizes) goto fail;
        for (size_t i = 0; i < cpu_blocks; ++i) {
            const size_t blk_idx = cpu_block_indices[i];
            cpu_comp_sizes[i] = sizes[blk_idx];
            cpu_comp_offsets[i] = global_offsets[blk_idx];
        }
    }

    {
        const uint64_t parallel_t0 = get_us();

        if (cpu_blocks > 0) {
            cpu_job.comp_data = data_ptr;
            cpu_job.comp_sizes = cpu_comp_sizes;
            cpu_job.comp_offsets = cpu_comp_offsets;
            cpu_job.block_size = block_size;
            cpu_job.block_indices = cpu_block_indices;
            cpu_job.num_blocks = cpu_blocks;
            cpu_job.num_threads = cfg->cpu_threads;
            cpu_job.out_full = out_full;
            cpu_job.out_sizes = cpu_out_sizes;

            if (pthread_create(&cpu_thread, NULL, cpu_decomp_top, &cpu_job) != 0) goto fail;
            cpu_thread_started = 1;
        }

        if (gpu_blocks > 0) {
            if (gpu_decompress_blocks(ocl,
                                      gpu_comp_data,
                                      gpu_comp_sizes,
                                      gpu_blocks,
                                      block_size,
                                      cfg->local_size,
                                      gpu_out_full,
                                      gpu_out_sizes,
                                      m ? &m->gpu_kernel_us : &(uint64_t){0}) != 0) {
                goto fail;
            }
        }

        if (cpu_thread_started) {
            pthread_join(cpu_thread, NULL);
            cpu_thread_started = 0;
            if (cpu_job.err) goto fail;
            if (m) m->cpu_kernel_us = cpu_job.elapsed_us;
        }
        if (m) {
            m->parallel_us = get_us() - parallel_t0;
            if (m->parallel_us < m->cpu_kernel_us) m->parallel_us = m->cpu_kernel_us;
            if (m->parallel_us < m->gpu_kernel_us) m->parallel_us = m->gpu_kernel_us;
        }
    }

    for (size_t i = 0; i < gpu_blocks; ++i) {
        const size_t blk_idx = gpu_block_indices[i];
        memcpy(out_full + blk_idx * block_size, gpu_out_full + i * block_size, (size_t)gpu_out_sizes[i]);
    }

    {
        size_t total_out = 0;
        for (size_t i = 0; i < gpu_blocks; ++i) total_out += (size_t)gpu_out_sizes[i];
        for (size_t i = 0; i < cpu_blocks; ++i) total_out += (size_t)cpu_out_sizes[i];
        *out_buf = out_full;
        *out_size = total_out;
    }

    free(global_offsets);
    free(gpu_block_indices);
    free(cpu_block_indices);
    free(gpu_comp_data);
    free(gpu_out_full);
    free(gpu_comp_sizes);
    free(gpu_out_sizes);
    free(cpu_comp_sizes);
    free(cpu_comp_offsets);
    free(cpu_out_sizes);
    return 0;

fail:
    if (cpu_thread_started) pthread_join(cpu_thread, NULL);
    free(global_offsets);
    free(gpu_block_indices);
    free(cpu_block_indices);
    free(out_full);
    free(gpu_comp_data);
    free(gpu_out_full);
    free(gpu_comp_sizes);
    free(gpu_out_sizes);
    free(cpu_comp_sizes);
    free(cpu_comp_offsets);
    free(cpu_out_sizes);
    return -1;
}

static void show_help(const char* prog) {
    fprintf(stderr, "LZ4 Hybrid CPU+GPU Tool\n");
    fprintf(stderr, "Usage: %s [options] <input_file>\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -c                       Compress mode (default)\n");
    fprintf(stderr, "  -d, --decompress         Decompress mode\n");
    fprintf(stderr, "  -o, --output FILE        Output file\n");
    fprintf(stderr, "  -b, --block-size N       Block size in bytes (default: 16K)\n");
    fprintf(stderr, "  -a, --acceleration N     GPU acceleration (default: 1)\n");
    fprintf(stderr, "  -H, --hash N             GPU hash log bits (default: 14)\n");
    fprintf(stderr, "  -l, --local N            GPU local work-group size (default: 1)\n");
    fprintf(stderr, "  -T, --cpu-threads N      CPU thread count for CPU portion (default: 1)\n");
    fprintf(stderr, "  --adaptive               Enable adaptive per-file CPU/GPU split\n");
    fprintf(stderr, "  --sample-blocks N        Adaptive sample block count (default: 8)\n");
    fprintf(stderr, "  --gpu-ratio F            Fraction of blocks assigned to GPU (default: 0.7)\n");
    fprintf(stderr, "  --bench [SECONDS]        Benchmark mode\n");
    fprintf(stderr, "  --bench-io               Include file write/read in bench total throughput\n");
    fprintf(stderr, "  --bench-seconds N        Benchmark duration in seconds (default: 3)\n");
    fprintf(stderr, "  -v, --verbose            Verbose output\n");
}

static int run_bench(const char* input_path, hybrid_cfg_t* cfg, double bench_seconds, int include_file_io) {
    unsigned char* input = NULL;
    size_t input_size = 0;
    ocl_env_t ocl;

    struct timespec ts0, ts1;
    size_t cap = 16, n = 0;
    double* comp_k = NULL;
    double* comp_t = NULL;
    double* dec_k = NULL;
    double* dec_t = NULL;
    double* ratio = NULL;
    int verify_ok = 1;
    char tmp_comp_path[PATH_MAX] = {0};
    char tmp_dec_path[PATH_MAX] = {0};

    if (bench_seconds <= 0.0) bench_seconds = 3.0;
    if (read_entire_file(input_path, &input, &input_size) != 0) {
        fprintf(stderr, "bench error: failed to read input\n");
        return 1;
    }

    if (ocl_init(&ocl, cfg->hash_log) != 0) {
        fprintf(stderr, "bench error: OpenCL init failed\n");
        free(input);
        return 1;
    }

    if (include_file_io) {
        if (create_temp_path(tmp_comp_path, sizeof(tmp_comp_path), "/tmp/lz4_hybrid_bench_io_comp_XXXXXX") != 0 ||
            create_temp_path(tmp_dec_path, sizeof(tmp_dec_path), "/tmp/lz4_hybrid_bench_io_dec_XXXXXX") != 0) {
            fprintf(stderr, "bench error: failed to create temp paths\n");
            free(input);
            ocl_free(&ocl);
            return 1;
        }
    }

    comp_k = (double*)malloc(cap * sizeof(double));
    comp_t = (double*)malloc(cap * sizeof(double));
    dec_k = (double*)malloc(cap * sizeof(double));
    dec_t = (double*)malloc(cap * sizeof(double));
    ratio = (double*)malloc(cap * sizeof(double));
    if (!comp_k || !comp_t || !dec_k || !dec_t || !ratio) verify_ok = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts0);

    while (verify_ok) {
        unsigned char* comp_buf = NULL;
        size_t comp_size = 0;
        unsigned char* dec_buf = NULL;
        size_t dec_size = 0;
        hybrid_metrics_t cm, dm;
        double in_mb;
        uint64_t comp_write_us = 0;
        uint64_t comp_read_us = 0;
        uint64_t dec_write_us = 0;

        const uint64_t ctot0 = get_us();
        if (hybrid_compress_memory(&ocl, input, input_size, cfg, &comp_buf, &comp_size, &cm) != 0) {
            verify_ok = 0;
            break;
        }

        if (include_file_io) {
            const uint64_t t_write0 = get_us();
            if (write_entire_file(tmp_comp_path, comp_buf, comp_size) != 0) {
                free(comp_buf);
                verify_ok = 0;
                break;
            }
            comp_write_us = get_us() - t_write0;
        }
        cm.total_us = get_us() - ctot0;
        if (include_file_io && cm.total_us < cm.parallel_us + comp_write_us) {
            cm.total_us = cm.parallel_us + comp_write_us;
        }

        const uint64_t dtot0 = get_us();
        if (include_file_io) {
            unsigned char* comp_file_buf = NULL;
            size_t comp_file_sz = 0;
            const uint64_t t_read0 = get_us();
            if (read_entire_file(tmp_comp_path, &comp_file_buf, &comp_file_sz) != 0) {
                free(comp_buf);
                verify_ok = 0;
                break;
            }
            comp_read_us = get_us() - t_read0;
            free(comp_buf);
            comp_buf = comp_file_buf;
            comp_size = comp_file_sz;
            remove(tmp_comp_path);
        }
        if (hybrid_decompress_memory(&ocl, comp_buf, comp_size, cfg, &dec_buf, &dec_size, &dm) != 0) {
            free(comp_buf);
            verify_ok = 0;
            break;
        }

        if (include_file_io) {
            const uint64_t t_write0 = get_us();
            if (write_entire_file(tmp_dec_path, dec_buf, dec_size) != 0) {
                free(comp_buf);
                free(dec_buf);
                verify_ok = 0;
                break;
            }
            dec_write_us = get_us() - t_write0;
            remove(tmp_dec_path);
        }
        dm.total_us = get_us() - dtot0;
        if (include_file_io && dm.total_us < dm.parallel_us + comp_read_us + dec_write_us) {
            dm.total_us = dm.parallel_us + comp_read_us + dec_write_us;
        }

        if (dec_size != input_size || memcmp(dec_buf, input, input_size) != 0) {
            free(comp_buf);
            free(dec_buf);
            verify_ok = 0;
            break;
        }

        if (n == cap) {
            size_t ncap = cap * 2;
            double* p;
            p = (double*)realloc(comp_k, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); free(dec_buf); break; } comp_k = p;
            p = (double*)realloc(comp_t, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); free(dec_buf); break; } comp_t = p;
            p = (double*)realloc(dec_k, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); free(dec_buf); break; } dec_k = p;
            p = (double*)realloc(dec_t, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); free(dec_buf); break; } dec_t = p;
            p = (double*)realloc(ratio, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); free(dec_buf); break; } ratio = p;
            cap = ncap;
        }

        in_mb = (double)input_size / (1024.0 * 1024.0);
        {
            uint64_t comp_kernel_us = cm.gpu_kernel_us;
            if (cm.cpu_kernel_us > comp_kernel_us) comp_kernel_us = cm.cpu_kernel_us;
            comp_k[n] = (comp_kernel_us > 0) ? (in_mb * 1000000.0 / (double)comp_kernel_us) : 0.0;
        }
        comp_t[n] = (cm.total_us > 0) ? (in_mb * 1000000.0 / (double)cm.total_us) : 0.0;
        {
            uint64_t dec_kernel_us = dm.gpu_kernel_us;
            if (dm.cpu_kernel_us > dec_kernel_us) dec_kernel_us = dm.cpu_kernel_us;
            dec_k[n] = (dec_kernel_us > 0) ? (in_mb * 1000000.0 / (double)dec_kernel_us) : 0.0;
        }
        dec_t[n] = (dm.total_us > 0) ? (in_mb * 1000000.0 / (double)dm.total_us) : 0.0;
        ratio[n] = (input_size > 0) ? (100.0 * (double)comp_size / (double)input_size) : 0.0;
        ++n;

        free(comp_buf);
        free(dec_buf);

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (elapsed_sec(&ts0, &ts1) >= bench_seconds && n > 0) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    if (n > 0) {
        printf("Bench Compress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s ratio=%.2f%%\n",
               median_double(comp_k, n), median_double(comp_t, n), median_double(ratio, n));
        printf("Bench Decompress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s verify=%s\n",
               median_double(dec_k, n), median_double(dec_t, n), verify_ok ? "OK" : "FAIL");
        printf("Bench Summary : iterations=%zu seconds=%.2f\n", n, elapsed_sec(&ts0, &ts1));
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_k);
    free(comp_t);
    free(dec_k);
    free(dec_t);
    free(ratio);
    free(input);
    if (include_file_io) {
        remove(tmp_comp_path);
        remove(tmp_dec_path);
    }
    ocl_free(&ocl);
    return verify_ok ? 0 : 1;
}

int main(int argc, char** argv) {
    int mode_decompress = 0;
    int bench_mode = 0;
    int bench_include_io = 0;
    double bench_seconds = 3.0;
    const char* input_path = NULL;
    char output_path[1024] = {0};
    int output_explicit = 0;

    hybrid_cfg_t cfg;
    cfg.block_size = 16 * 1024;
    cfg.acceleration = 1;
    cfg.hash_log = 14;
    cfg.local_size = 1;
    cfg.cpu_threads = 1;
    cfg.gpu_ratio = 0.7;
    cfg.adaptive_split = 0;
    cfg.adaptive_sample_blocks = 8;
    cfg.verbose = 0;

    if (argc < 2) {
        show_help(argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-c") == 0) {
            mode_decompress = 0;
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) {
            mode_decompress = 1;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                strncpy(output_path, argv[++i], sizeof(output_path) - 1);
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: -o requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--block-size") == 0) {
            if (i + 1 < argc) cfg.block_size = parse_size_bytes(argv[++i]);
            else { fprintf(stderr, "Error: -b requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--acceleration") == 0) {
            if (i + 1 < argc) cfg.acceleration = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -a requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--hash") == 0) {
            if (i + 1 < argc) cfg.hash_log = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -H requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--local") == 0) {
            if (i + 1 < argc) cfg.local_size = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -l requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--cpu-threads") == 0) {
            if (i + 1 < argc) cfg.cpu_threads = atoi(argv[++i]);
            else { fprintf(stderr, "Error: -T requires an argument\n"); return 1; }
        } else if (strcmp(argv[i], "--adaptive") == 0) {
            cfg.adaptive_split = 1;
        } else if (strcmp(argv[i], "--sample-blocks") == 0) {
            if (i + 1 < argc) cfg.adaptive_sample_blocks = (size_t)strtoull(argv[++i], NULL, 10);
            else { fprintf(stderr, "Error: --sample-blocks requires an argument\n"); return 1; }
        } else if (strncmp(argv[i], "--sample-blocks=", 16) == 0) {
            cfg.adaptive_sample_blocks = (size_t)strtoull(argv[i] + 16, NULL, 10);
        } else if (strcmp(argv[i], "--gpu-ratio") == 0) {
            if (i + 1 < argc) cfg.gpu_ratio = atof(argv[++i]);
            else { fprintf(stderr, "Error: --gpu-ratio requires an argument\n"); return 1; }
        } else if (strncmp(argv[i], "--gpu-ratio=", 12) == 0) {
            cfg.gpu_ratio = atof(argv[i] + 12);
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-' && is_number_string(argv[i + 1])) {
                bench_seconds = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--bench-io") == 0) {
            bench_include_io = 1;
        } else if (strcmp(argv[i], "--bench-seconds") == 0) {
            if (i + 1 < argc) bench_seconds = atof(argv[++i]);
            else { fprintf(stderr, "Error: --bench-seconds requires an argument\n"); return 1; }
        } else if (strncmp(argv[i], "--bench-seconds=", 16) == 0) {
            bench_seconds = atof(argv[i] + 16);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            cfg.verbose = 1;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
            show_help(argv[0]);
            return 1;
        } else {
            if (!input_path) input_path = argv[i];
            else if (!output_explicit) {
                strncpy(output_path, argv[i], sizeof(output_path) - 1);
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: Too many positional arguments\n");
                show_help(argv[0]);
                return 1;
            }
        }
    }

    if (!input_path) {
        show_help(argv[0]);
        return 0;
    }
    if (cfg.block_size == 0) {
        fprintf(stderr, "Error: block size must be > 0\n");
        return 1;
    }
    if (cfg.cpu_threads <= 0) cfg.cpu_threads = 1;
    if (cfg.local_size <= 0) cfg.local_size = 1;
    if (cfg.hash_log <= 0) cfg.hash_log = 14;
    if (cfg.hash_log == 13) cfg.hash_log = 14;
    if (cfg.gpu_ratio < 0.0) cfg.gpu_ratio = 0.0;
    if (cfg.gpu_ratio > 1.0) cfg.gpu_ratio = 1.0;

    if (bench_mode) {
        if (mode_decompress) {
            fprintf(stderr, "Error: --bench requires compress mode input\n");
            return 1;
        }
        return run_bench(input_path, &cfg, bench_seconds, bench_include_io);
    }

    if (!output_explicit) {
        if (!mode_decompress) snprintf(output_path, sizeof(output_path), "%s.lz4", input_path);
        else snprintf(output_path, sizeof(output_path), "%s.dec", input_path);
    }

    {
        int rc = 1;
        ocl_env_t ocl;
        unsigned char* in_buf = NULL;
        size_t in_sz = 0;
        unsigned char* out_buf = NULL;
        size_t out_sz = 0;
        hybrid_metrics_t met;
        uint64_t total0 = get_us();

        if (read_entire_file(input_path, &in_buf, &in_sz) != 0) {
            fprintf(stderr, "Error: failed to read input file %s (%s)\n", input_path, strerror(errno));
            return 1;
        }

        if (ocl_init(&ocl, cfg.hash_log) != 0) {
            fprintf(stderr, "Error: OpenCL init failed\n");
            free(in_buf);
            return 1;
        }

        if (!mode_decompress) {
            const uint64_t t0 = get_us();
            if (hybrid_compress_memory(&ocl, in_buf, in_sz, &cfg, &out_buf, &out_sz, &met) != 0) {
                fprintf(stderr, "Error: compression failed\n");
                goto done;
            }
            met.total_us = get_us() - t0;
            if (write_entire_file(output_path, out_buf, out_sz) != 0) {
                fprintf(stderr, "Error: failed writing output file %s\n", output_path);
                goto done;
            }
            if (cfg.verbose) {
                double in_mb = (double)in_sz / (1024.0 * 1024.0);
                double k_tp = (met.parallel_us > 0) ? (in_mb * 1000000.0 / (double)met.parallel_us) : 0.0;
                double t_tp = (met.total_us > 0) ? (in_mb * 1000000.0 / (double)met.total_us) : 0.0;
                printf("Hybrid Compress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s ratio=%.2f%%\n",
                       k_tp, t_tp, (in_sz > 0) ? (100.0 * (double)out_sz / (double)in_sz) : 0.0);
            } else {
                printf("%s : %zu -> %zu in %.2f ms\n", input_path, in_sz, out_sz, (get_us() - total0) / 1000.0);
            }
        } else {
            const uint64_t t0 = get_us();
            if (hybrid_decompress_memory(&ocl, in_buf, in_sz, &cfg, &out_buf, &out_sz, &met) != 0) {
                fprintf(stderr, "Error: decompression failed\n");
                goto done;
            }
            met.total_us = get_us() - t0;
            if (write_entire_file(output_path, out_buf, out_sz) != 0) {
                fprintf(stderr, "Error: failed writing output file %s\n", output_path);
                goto done;
            }
            if (cfg.verbose) {
                double out_mb = (double)out_sz / (1024.0 * 1024.0);
                double k_tp = (met.parallel_us > 0) ? (out_mb * 1000000.0 / (double)met.parallel_us) : 0.0;
                double t_tp = (met.total_us > 0) ? (out_mb * 1000000.0 / (double)met.total_us) : 0.0;
                printf("Hybrid Decompress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s\n", k_tp, t_tp);
            } else {
                printf("%s : %zu -> %zu in %.2f ms\n", input_path, in_sz, out_sz, (get_us() - total0) / 1000.0);
            }
        }

        rc = 0;
done:
        free(in_buf);
        free(out_buf);
        ocl_free(&ocl);
        return rc;
    }
}
