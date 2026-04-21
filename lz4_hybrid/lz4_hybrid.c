#define _POSIX_C_SOURCE 200809L

#include <CL/cl.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "../lz4_gpu/lz4_gpu_core.h"
#include "../lz4_gpu/lz4_gpu_utils.h"
#include "../lib/lz4.h"

#define HYBRID_MAGIC 0x184D2205U

#ifndef LZ4_HYBRID_COMP_WI_PER_CU_DEFAULT
#define LZ4_HYBRID_COMP_WI_PER_CU_DEFAULT 24
#endif

typedef struct {
    size_t block_size;
    int acceleration;
    int local_size;
    int cpu_threads;
    double gpu_ratio;
    int adaptive_split;
    size_t adaptive_sample_blocks;
    int verbose;
} hybrid_cfg_t;

typedef struct {
    uint64_t file_read_us;
    uint64_t ocl_setup_us;
    uint64_t data_upload_us;
    uint64_t cpu_kernel_us;
    uint64_t gpu_kernel_us;
    uint64_t parallel_us;
    uint64_t data_download_us;
    uint64_t file_write_us;
    uint64_t total_us;
} hybrid_metrics_t;

typedef struct {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev;
    cl_program prog;
    cl_kernel kcomp;
    cl_kernel kcomp_mapped;
    cl_kernel kdec;
    cl_kernel kdec_mapped;
    lz4_gpu_workspace_t ws;
    cl_mem cached_kcomp_arg0;
    cl_mem cached_kcomp_arg1;
    cl_mem cached_kcomp_arg2;
    cl_mem cached_kcomp_arg3;
    cl_mem cached_kcomp_arg4;
    cl_mem cached_kcomp_arg10;
    cl_mem cached_kdec_arg0;
    cl_mem cached_kdec_arg1;
    cl_mem cached_kdec_arg2;
    cl_mem cached_kdec_arg3;
    cl_mem cached_kdec_arg4;
    cl_mem cached_kdec_arg5;
    cl_mem cached_kdec_arg6;
    uint64_t cached_decomp_meta_hash;
    size_t cached_decomp_meta_count;
    int cached_decomp_meta_valid;
    uint64_t cached_block_map_hash;
    size_t cached_block_map_count;
    int cached_block_map_valid;
    int adaptive_ratio_cache_valid;
    size_t adaptive_ratio_cache_input_size;
    size_t adaptive_ratio_cache_num_blocks;
    size_t adaptive_ratio_cache_block_size;
    int adaptive_ratio_cache_acceleration;
    int adaptive_ratio_cache_cpu_threads;
    size_t adaptive_ratio_cache_sample_blocks;
    double adaptive_ratio_cache_value;
    double adaptive_ratio_cache_sample_ratio_pct;
} ocl_env_t;

static int set_kernel_mem_arg_if_changed(cl_kernel kernel, cl_uint index, cl_mem* cache, cl_mem value) {
    if (*cache == value) return CL_SUCCESS;
    *cache = value;
    return clSetKernelArg(kernel, index, sizeof(cl_mem), &value);
}

typedef struct {
    const unsigned char* src;
    size_t src_size;
    size_t block_size;
    const size_t* block_indices;
    size_t block_index_base;
    size_t num_blocks;
    int num_threads;
    int acceleration;

    unsigned char* out_slots;
    size_t out_slot_size;
    uint32_t* out_sizes;

    uint64_t elapsed_us;
    _Atomic size_t next_block;
    int err;
} cpu_comp_job_t;

typedef struct {
    const unsigned char* comp_data;
    const uint32_t* comp_sizes;
    const uint32_t* comp_offsets;
    size_t block_size;
    const size_t* block_indices;
    size_t block_index_base;
    size_t num_blocks;
    int num_threads;

    unsigned char* out_full;
    uint32_t* out_sizes;

    uint64_t elapsed_us;
    _Atomic size_t next_block;
    int err;
} cpu_decomp_job_t;

typedef struct {
    size_t thread_idx;
    size_t thread_count;
    cpu_comp_job_t* job;
    uint64_t elapsed_us;
} cpu_comp_worker_arg_t;

typedef struct {
    size_t thread_idx;
    size_t thread_count;
    cpu_decomp_job_t* job;
    uint64_t elapsed_us;
} cpu_decomp_worker_arg_t;

typedef struct {
    size_t sample_count;
    double mean_ratio_pct;
    size_t low_ratio_blocks;
    size_t high_ratio_blocks;
    double sample_cpu_throughput;
} lz4_sample_stats_t;

typedef struct {
    double cpu_throughput;
    double gpu_throughput;
    double gpu_overhead_s;
    double cpu_energy_per_byte;
    double gpu_energy_per_byte;
    int    is_unified_memory;
    int    valid;
} device_profile_t;

static device_profile_t g_dev_profile = {0};

static double read_sysfs_double(const char* path) {
    FILE* f = fopen(path, "r");
    double val = -1.0;
    if (f) { if (fscanf(f, "%lf", &val) != 1) val = -1.0; fclose(f); }
    return val;
}

static uint64_t read_rapl_energy_uj(const char* domain_path) {
    FILE* f = fopen(domain_path, "r");
    uint64_t val = 0;
    if (f) { if (fscanf(f, "%" SCNu64, &val) != 1) val = 0; fclose(f); }
    return val;
}

static double read_env_double_or_neg(const char* name) {
    const char* v;
    char* endp = NULL;
    double out;
    if (!name) return -1.0;
    v = getenv(name);
    if (!v || !*v) return -1.0;
    out = strtod(v, &endp);
    if (!endp || endp == v) return -1.0;
    return out;
}

static double read_cpu_availability(void) {
    static uint64_t prev_total = 0, prev_idle = 0;
    FILE* f = fopen("/proc/stat", "r");
    char buf[256];
    uint64_t user, nice, sys, idle, iowait, irq, softirq, steal;
    uint64_t total, diff_total, diff_idle;
    double avail;
    if (!f) return 1.0;
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 1.0; }
    fclose(f);
    if (sscanf(buf,
               "cpu %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64
               " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64,
               &user, &nice, &sys, &idle, &iowait, &irq, &softirq, &steal) < 4)
        return 1.0;
    total = user + nice + sys + idle + iowait + irq + softirq + steal;
    diff_total = total - prev_total;
    diff_idle = idle - prev_idle;
    prev_total = total;
    prev_idle = idle;
    if (diff_total == 0) return 1.0;
    avail = (double)diff_idle / (double)diff_total;
    if (avail < 0.05) avail = 0.05;
    if (avail > 1.0) avail = 1.0;
    return avail;
}

static double read_gpu_availability(void) {
    double busy = read_sysfs_double("/sys/class/drm/card0/device/gpu_busy_percent");
    if (busy < 0.0) busy = read_sysfs_double("/sys/class/drm/card1/device/gpu_busy_percent");
    if (busy < 0.0) return 1.0;
    if (busy > 100.0) busy = 100.0;
    return 1.0 - busy / 100.0;
}

static double read_cpu_freq_scale(void) {
    double max_f = read_sysfs_double("/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq");
    double cur = read_sysfs_double("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq");
    double target_mhz = -1.0;
    double target_pct = -1.0;
    double scale;

    target_mhz = read_env_double_or_neg("LZ4_HYBRID_CPU_FREQ_TARGET_MHZ");
    if (target_mhz <= 0.0) target_mhz = read_env_double_or_neg("HYBRID_CPU_FREQ_TARGET_MHZ");
    target_pct = read_env_double_or_neg("LZ4_HYBRID_CPU_FREQ_TARGET_PCT");
    if (target_pct <= 0.0) target_pct = read_env_double_or_neg("HYBRID_CPU_FREQ_TARGET_PCT");

    if (target_mhz > 0.0 && max_f > 0.0) {
        scale = (target_mhz * 1000.0) / max_f;
    } else if (target_pct > 0.0) {
        scale = target_pct / 100.0;
    } else if (cur > 0.0 && max_f > 0.0) {
        scale = cur / max_f;
    } else {
        scale = 1.0;
    }

    if (scale < 0.30) scale = 0.30;
    if (scale > 1.20) scale = 1.20;
    return scale;
}

static double read_gpu_freq_scale(void) {
    double cur = read_sysfs_double("/sys/class/drm/card0/gt_cur_freq_mhz");
    double max_f = read_sysfs_double("/sys/class/drm/card0/gt_max_freq_mhz");
    double target_mhz = -1.0;
    double target_pct = -1.0;
    double scale;

    if (cur <= 0.0 || max_f <= 0.0) {
        cur = read_sysfs_double("/sys/class/drm/card1/gt_cur_freq_mhz");
        max_f = read_sysfs_double("/sys/class/drm/card1/gt_max_freq_mhz");
    }

    target_mhz = read_env_double_or_neg("LZ4_HYBRID_GPU_FREQ_TARGET_MHZ");
    if (target_mhz <= 0.0) target_mhz = read_env_double_or_neg("HYBRID_GPU_FREQ_TARGET_MHZ");
    target_pct = read_env_double_or_neg("LZ4_HYBRID_GPU_FREQ_TARGET_PCT");
    if (target_pct <= 0.0) target_pct = read_env_double_or_neg("HYBRID_GPU_FREQ_TARGET_PCT");

    if (target_mhz > 0.0 && max_f > 0.0) {
        scale = target_mhz / max_f;
    } else if (target_pct > 0.0) {
        scale = target_pct / 100.0;
    } else if (cur > 0.0 && max_f > 0.0) {
        scale = cur / max_f;
    } else {
        scale = 1.0;
    }

    if (scale < 0.30) scale = 0.30;
    if (scale > 1.20) scale = 1.20;
    return scale;
}

static long get_online_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (si.dwNumberOfProcessors > 0) ? (long)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? n : 1;
#endif
}

static int gpu_compress_blocks(ocl_env_t* ocl,
                               const unsigned char* src,
                               size_t src_size,
                               size_t block_size,
                               int acceleration,
                               int local_size,
                               const uint32_t* mapped_block_indices,
                               size_t mapped_block_count,
                               uint32_t** out_sizes,
                               uint32_t** out_offsets,
                               unsigned char** out_slots,
                               size_t* out_slot_size,
                               uint64_t* kernel_us,
                               uint64_t* upload_us,
                               uint64_t* download_us,
                               int skip_input_upload);

static void calibrate_device_profile(ocl_env_t* ocl, const hybrid_cfg_t* cfg) {
    if (g_dev_profile.valid) return;

    cl_bool unified = CL_FALSE;
    clGetDeviceInfo(ocl->dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL);
    g_dev_profile.is_unified_memory = (unified == CL_TRUE) ? 1 : 0;

    {
        size_t cal_size = 2 * 1024 * 1024;
        unsigned char* cal_buf = (unsigned char*)malloc(cal_size);
        char* cal_tmp = NULL;
        int cal_bound;
        uint64_t t0, t1;
        int comp_sz;

        if (!cal_buf) { g_dev_profile.valid = 1; g_dev_profile.cpu_throughput = 500e6; g_dev_profile.gpu_throughput = 2000e6; g_dev_profile.gpu_overhead_s = 0.0005; return; }
        memset(cal_buf, 0xAB, cal_size);
        for (size_t i = 0; i < cal_size; i += 97) cal_buf[i] = (unsigned char)(i & 0xFF);

        cal_bound = LZ4_compressBound((int)cal_size);
        cal_tmp = (char*)malloc((size_t)cal_bound);
        if (!cal_tmp) { free(cal_buf); g_dev_profile.valid = 1; g_dev_profile.cpu_throughput = 500e6; g_dev_profile.gpu_throughput = 2000e6; g_dev_profile.gpu_overhead_s = 0.0005; return; }

        {
            uint64_t e0 = read_rapl_energy_uj("/sys/class/powercap/intel-rapl:0:0/energy_uj");
            t0 = get_us();
            for (int rep = 0; rep < 3; rep++) {
                comp_sz = LZ4_compress_fast((const char*)cal_buf, cal_tmp, (int)cal_size, cal_bound,
                                            cfg->acceleration > 0 ? cfg->acceleration : 1);
            }
            t1 = get_us();
            (void)comp_sz;
            g_dev_profile.cpu_throughput = (3.0 * (double)cal_size) / ((double)(t1 - t0) * 1e-6);

            {
                uint64_t e1 = read_rapl_energy_uj("/sys/class/powercap/intel-rapl:0:0/energy_uj");
                if (e0 > 0 && e1 > e0) {
                    double energy_j = (double)(e1 - e0) * 1e-6;
                    g_dev_profile.cpu_energy_per_byte = energy_j / (3.0 * (double)cal_size);
                }
            }
        }
        free(cal_tmp);

        {
            uint32_t* gp_sizes = NULL;
            uint32_t* gp_offsets = NULL;
            unsigned char* gp_slots = NULL;
            size_t gp_slot_size = 0;
            uint64_t gp_kernel_us = 0;

            uint64_t ge0 = read_rapl_energy_uj("/sys/class/powercap/intel-rapl:0:1/energy_uj");
            t0 = get_us();
            if (gpu_compress_blocks(ocl, cal_buf, cal_size,
                                    cfg->block_size > 0 ? cfg->block_size : 32768,
                                    cfg->acceleration > 0 ? cfg->acceleration : 1,
                                    cfg->local_size,
                                    NULL,
                                    0,
                                    &gp_sizes, &gp_offsets, &gp_slots, &gp_slot_size,
                                    &gp_kernel_us, NULL, NULL, 0) == 0) {
                t1 = get_us();
                {
                    double total_s = (double)(t1 - t0) * 1e-6;
                    double kernel_s = (double)gp_kernel_us * 1e-6;
                    if (kernel_s <= 0.0) kernel_s = total_s;
                    if (kernel_s <= 0.0) kernel_s = 1e-6;
                    g_dev_profile.gpu_throughput = (double)cal_size / kernel_s;
                    g_dev_profile.gpu_overhead_s = total_s - kernel_s;
                }
                if (g_dev_profile.gpu_overhead_s < 0.0) g_dev_profile.gpu_overhead_s = 0.0;

                {
                    uint64_t ge1 = read_rapl_energy_uj("/sys/class/powercap/intel-rapl:0:1/energy_uj");
                    if (ge0 > 0 && ge1 > ge0) {
                        double energy_j = (double)(ge1 - ge0) * 1e-6;
                        g_dev_profile.gpu_energy_per_byte = energy_j / (double)cal_size;
                    }
                }

                free(gp_sizes); free(gp_offsets); free(gp_slots);
            } else {
                g_dev_profile.gpu_throughput = 2000e6;
                g_dev_profile.gpu_overhead_s = 0.0005;
            }
        }
        free(cal_buf);
    }

    if (g_dev_profile.cpu_throughput <= 0.0) g_dev_profile.cpu_throughput = 500e6;
    if (g_dev_profile.gpu_throughput <= 0.0) g_dev_profile.gpu_throughput = 2000e6;
    g_dev_profile.valid = 1;
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

static int gpu_decompress_blocks(ocl_env_t* ocl,
                                 const unsigned char* comp_data,
                                 const uint32_t* comp_sizes,
                                 size_t num_blocks,
                                 size_t block_size,
                                 int local_size,
                                 unsigned char* out_full,
                                 uint32_t* out_sizes,
                                 uint64_t* kernel_us,
                                 uint64_t* upload_us,
                                 uint64_t* download_us);

static uint64_t lz4_hash_u32_array(const uint32_t* data, size_t count);
static void lz4_adaptive_choose_objective_weights(size_t input_size,
                                                  size_t num_blocks,
                                                  double sample_ratio_pct,
                                                  int is_unified_memory,
                                                  double cpu_avail,
                                                  double gpu_avail,
                                                  long thread_count,
                                                  long total_cores,
                                                  double cpu_freq_scale,
                                                  double gpu_freq_scale,
                                                  double* perf_weight_pct,
                                                  double* energy_weight_pct,
                                                  double* ratio_weight_pct,
                                                  double* target_ratio_pct,
                                                  double* dec_host_penalty_pct);
static double lz4_cpu_thread_scale(long thread_count, int is_unified_memory);
static double lz4_refine_ratio_candidate(size_t total_input_sz,
                                         size_t num_blocks,
                                         double Pc_eff,
                                         double Pg_eff,
                                         double t0,
                                         double cpu_freq_scale,
                                         double gpu_freq_scale,
                                         double thread_util,
                                         double sample_ratio_pct,
                                         double seed_ratio,
                                         double min_ratio,
                                         double max_ratio);

static size_t parse_size_bytes(const char* s) {
    char* endptr = NULL;
    size_t val = strtoul(s, &endptr, 10);
    if (endptr && (*endptr == 'k' || *endptr == 'K')) val *= 1024U;
    else if (endptr && (*endptr == 'm' || *endptr == 'M')) val *= 1024U * 1024U;
    return val;
}

static size_t adaptive_skip_ocl_threshold_bytes(void) {
    const char* pref = getenv("FORCE_OPENCL_DEVICE");
    int force_cpu = (pref && *pref && strcasecmp(pref, "CPU") == 0);



    return force_cpu ? (4U * 1024U * 1024U) : (8U * 1024U * 1024U);
}

static int adaptive_should_skip_ocl(const hybrid_cfg_t* cfg, size_t input_size) {
    size_t threshold;
    if (!cfg || !cfg->adaptive_split) return 0;
    if (cfg->cpu_threads <= 0) return 0;
    threshold = adaptive_skip_ocl_threshold_bytes();
    return threshold > 0 && input_size < threshold;
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

static size_t bench_default_dec_repeat(size_t input_size) {
    (void)input_size;
    return 1;
}

static size_t bench_dec_repeat_from_env(size_t input_size) {
    return bench_default_dec_repeat(input_size);
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

static double lz4_entropy_bits_per_byte(const unsigned char* src, size_t len) {
    unsigned counts[256];
    double inv_len;
    double h = 0.0;

    if (!src || len == 0) return 0.0;
    memset(counts, 0, sizeof(counts));

    for (size_t i = 0; i < len; ++i) {
        counts[src[i]]++;
    }

    inv_len = 1.0 / (double)len;
    for (size_t i = 0; i < 256; ++i) {
        if (counts[i] == 0) continue;
        {
            double p = (double)counts[i] * inv_len;
            h -= p * (log(p) / log(2.0));
        }
    }

    if (h < 0.0) h = 0.0;
    if (h > 8.0) h = 8.0;
    return h;
}

static double lz4_clamp_double(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static double lz4_norm_from_bytes(double bytes, double ref_mb, double span) {
    double ref_bytes;
    double v;

    if (bytes <= 0.0) return 0.0;
    if (ref_mb <= 0.0) ref_mb = 1.0;
    if (span <= 0.0) span = 1.0;

    ref_bytes = ref_mb * 1024.0 * 1024.0;
    if (ref_bytes <= 0.0) ref_bytes = 1024.0 * 1024.0;

    v = log2(bytes / ref_bytes + 1.0) / span;
    return lz4_clamp_double(v, 0.0, 1.0);
}

static double lz4_cpu_thread_scale(long thread_count, int is_unified_memory) {
    double gain;
    double scale;
    double cap;

    if (thread_count <= 1) return 1.0;

    gain = is_unified_memory ? 0.52 : 0.60;
    scale = 1.0 + (double)(thread_count - 1) * gain;
    cap = 1.0 + (is_unified_memory ? 1.55 : 1.75) * log2((double)thread_count + 1.0);
    if (scale > cap) scale = cap;
    if (scale < 1.0) scale = 1.0;
    return scale;
}

static double lz4_refine_ratio_candidate(size_t total_input_sz,
                                         size_t num_blocks,
                                         double Pc_eff,
                                         double Pg_eff,
                                         double t0,
                                         double cpu_freq_scale,
                                         double gpu_freq_scale,
                                         double thread_util,
                                         double sample_ratio_pct,
                                         double seed_ratio,
                                         double min_ratio,
                                         double max_ratio) {
    static const double cands[] = {0.0, 0.04, 0.08, 0.12, 0.16, 0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.76, 0.82, 0.88, 0.94, 1.0};
    const size_t nc = sizeof(cands) / sizeof(cands[0]);
    double best_r = seed_ratio;
    double best_obj = 1e300;
    double B = (double)total_input_sz;
    double pc = (Pc_eff > 1.0) ? Pc_eff : 1.0;
    double pg = (Pg_eff > 1.0) ? Pg_eff : 1.0;
    double block_bytes;
    double size_norm;
    double block_norm;
    double ratio_norm;
    double compressibility;
    double device_adv;

    if (B <= 0.0 || num_blocks == 0) return seed_ratio;
    seed_ratio = lz4_clamp_double(seed_ratio, 0.0, 1.0);
    min_ratio = lz4_clamp_double(min_ratio, 0.0, 1.0);
    max_ratio = lz4_clamp_double(max_ratio, 0.0, 1.0);
    if (max_ratio < min_ratio) max_ratio = min_ratio;

    block_bytes = B / (double)num_blocks;
    if (block_bytes < 4096.0) block_bytes = 4096.0;

    size_norm = lz4_norm_from_bytes(B, 1.0, 8.0);
    block_norm = lz4_norm_from_bytes(block_bytes, 0.0625, 2.5);
    ratio_norm = lz4_clamp_double(sample_ratio_pct / 100.0, 0.0, 1.0);
    compressibility = 1.0 - ratio_norm;
    device_adv = (pg - pc) / (pg + pc);

    for (size_t i = 0; i < nc; ++i) {
        size_t gpu_blocks;
        size_t cpu_blocks;
        double r;
        double bytes_gpu;
        double bytes_cpu;
        double split_mix;
        double gpu_overhead_scale;
        double gpu_comp_overhead_s = 0.0;
        double gpu_dec_overhead_s = 0.0;
        double mix_host_s = 0.0;
        double comp_cpu_s;
        double comp_gpu_s;
        double comp_s;
        double dec_cpu_tp;
        double dec_gpu_tp;
        double dec_cpu_s;
        double dec_gpu_s;
        double dec_s;
        double load_risk_s = 0.0;
        double ratio_pen_s = 0.0;
        double smooth_pen;
        double direction_bonus;
        double obj;

        gpu_blocks = (size_t)((double)num_blocks * cands[i] + 0.5);
        if (gpu_blocks > num_blocks) gpu_blocks = num_blocks;
        cpu_blocks = num_blocks - gpu_blocks;
        r = (double)gpu_blocks / (double)num_blocks;
        if (r < min_ratio || r > max_ratio) {
            continue;
        }

        bytes_gpu = B * r;
        bytes_cpu = B - bytes_gpu;
        split_mix = 4.0 * r * (1.0 - r);

        gpu_overhead_scale = 0.85 + (1.0 - size_norm) * (0.95 - 0.35 * block_norm);
        gpu_overhead_scale = lz4_clamp_double(gpu_overhead_scale, 0.45, 1.90);

        if (gpu_blocks > 0) {
            gpu_comp_overhead_s = t0 * gpu_overhead_scale;
            gpu_dec_overhead_s = 0.52 * t0 * gpu_overhead_scale;
        }

        comp_cpu_s = bytes_cpu / pc;
        comp_gpu_s = (gpu_blocks > 0) ? (gpu_comp_overhead_s + bytes_gpu / pg) : 0.0;
        comp_s = (gpu_blocks > 0 && cpu_blocks > 0)
            ? ((comp_cpu_s > comp_gpu_s) ? comp_cpu_s : comp_gpu_s)
            : (comp_cpu_s + comp_gpu_s);

        dec_cpu_tp = pc * (1.28 + 0.10 * (1.0 - size_norm));
        dec_gpu_tp = pg * (0.96 + 0.06 * gpu_freq_scale);
        if (dec_cpu_tp < 1.0) dec_cpu_tp = 1.0;
        if (dec_gpu_tp < 1.0) dec_gpu_tp = 1.0;
        dec_cpu_s = bytes_cpu / dec_cpu_tp;
        dec_gpu_s = (gpu_blocks > 0) ? (gpu_dec_overhead_s + bytes_gpu / dec_gpu_tp) : 0.0;
        dec_s = (gpu_blocks > 0 && cpu_blocks > 0)
            ? ((dec_cpu_s > dec_gpu_s) ? dec_cpu_s : dec_gpu_s)
            : (dec_cpu_s + dec_gpu_s);

        mix_host_s = split_mix * (0.00004 + (1.0 - size_norm) * 0.00012 + (1.0 - block_norm) * 0.00007);

        if (thread_util < 0.45) {
            load_risk_s += split_mix * (0.45 - thread_util) * 0.00018;
        }
        if (cpu_freq_scale < 0.75 && cpu_blocks > 0) {
            load_risk_s += (0.75 - cpu_freq_scale) * (1.0 - r) * 0.00016;
        }
        if (gpu_freq_scale < 0.72 && gpu_blocks > 0) {
            load_risk_s += (0.72 - gpu_freq_scale) * r * 0.00012;
        }

        ratio_pen_s = split_mix * (0.00011 * compressibility * r + 0.00004 * ratio_norm * (1.0 - r));

        smooth_pen = fabs(r - seed_ratio) * 0.0012;
        direction_bonus = 0.00011 * device_adv * (r - 0.5);
        obj = 0.58 * comp_s + 0.42 * dec_s + mix_host_s + load_risk_s + ratio_pen_s + smooth_pen - direction_bonus;

        if (obj < best_obj) {
            best_obj = obj;
            best_r = r;
        }
    }

    if (best_obj >= 1e299) {
        best_r = seed_ratio;
    }

    if (best_r < min_ratio) best_r = min_ratio;
    if (best_r > max_ratio) best_r = max_ratio;

    return best_r;
}

static int collect_lz4_sample_stats(const unsigned char* input,
                                    size_t input_size,
                                    size_t num_blocks,
                                    const hybrid_cfg_t* cfg,
                                    lz4_sample_stats_t* stats,
                                    double* sample_ratio_pct_out) {
    size_t sample_blocks;
    double ratio_sum = 0.0;
    size_t sample_bytes = 0;
    size_t prev_block = SIZE_MAX;
    char* tmp = NULL;
    int tmp_cap = 0;
    uint64_t t0 = 0;

    if (sample_ratio_pct_out) *sample_ratio_pct_out = 0.0;
    if (!input || !cfg || !stats || input_size == 0 || num_blocks == 0) return -1;
    memset(stats, 0, sizeof(*stats));

    sample_blocks = cfg->adaptive_sample_blocks ? cfg->adaptive_sample_blocks : 8;
    if (sample_blocks > 6) sample_blocks = 6;
    if (sample_blocks > num_blocks) sample_blocks = num_blocks;
    if (sample_blocks == 0) return -1;

    tmp_cap = LZ4_compressBound((int)cfg->block_size);
    if (tmp_cap > 0) {
        tmp = (char*)malloc((size_t)tmp_cap);
        if (tmp) t0 = get_us();
    }

    for (size_t i = 0; i < sample_blocks; ++i) {
        size_t blk_idx = sampled_block_index(i, sample_blocks, num_blocks);
        size_t blk_sz = block_input_size(input_size, cfg->block_size, blk_idx);
        size_t scan_sz;
        double ent;
        double est_ratio_pct;

        if (blk_idx == prev_block || blk_sz == 0) continue;

        scan_sz = blk_sz;
        if (scan_sz > 16384U) scan_sz = 16384U;
        ent = lz4_entropy_bits_per_byte(input + blk_idx * cfg->block_size, scan_sz);

        if (tmp) {
            int comp_sz = LZ4_compress_fast((const char*)(const void*)(input + blk_idx * cfg->block_size),
                                            tmp,
                                            (int)blk_sz,
                                            tmp_cap,
                                            cfg->acceleration > 0 ? cfg->acceleration : 1);
            if (comp_sz > 0) {
                est_ratio_pct = 100.0 * (double)comp_sz / (double)blk_sz;
                sample_bytes += blk_sz;
            } else {
                est_ratio_pct = 18.0 + ent * 8.8;
            }
        } else {
            est_ratio_pct = 18.0 + ent * 8.8;
        }

        est_ratio_pct = lz4_clamp_double(est_ratio_pct, 16.0, 95.0);

        if (est_ratio_pct < 35.0) stats->low_ratio_blocks++;
        else if (est_ratio_pct > 70.0) stats->high_ratio_blocks++;

        ratio_sum += est_ratio_pct;
        stats->sample_count++;
        prev_block = blk_idx;
    }

    if (tmp) {
        uint64_t t1 = get_us();
        if (sample_bytes > 0 && t1 > t0) {
            stats->sample_cpu_throughput = (double)sample_bytes / ((double)(t1 - t0) * 1e-6);
        }
        free(tmp);
    }

    if (stats->sample_count == 0) return -1;

    stats->mean_ratio_pct = ratio_sum / (double)stats->sample_count;
    if (sample_ratio_pct_out) *sample_ratio_pct_out = stats->mean_ratio_pct;
    return 0;
}

static double choose_adaptive_gpu_ratio(ocl_env_t* ocl,
                                        const unsigned char* input,
                                        size_t input_size,
                                        size_t num_blocks,
                                        const hybrid_cfg_t* cfg,
                                        double* sample_ratio_pct_out) {
    /*
     * 端到端时延模型（压缩+解压）驱动的异构比例决策：
     *   obj(r) = 0.56*max(Tc_cpu, Tc_gpu) + 0.44*max(Td_cpu, Td_gpu) + HostMix(r)
     * 其中 Tc/Td 由设备吞吐、频率、负载、输入规模、块大小共同决定。
     *
     * 设计目标：
     *   1) 轻量建模（避免每轮 sample 压缩探测）
     *   2) 对 T=1 场景更激进地使用 GPU
     *   3) 显式计入混合调度主机端开销，避免“理论比率高、实测时延差”
     */
    lz4_sample_stats_t stats;
    double Pc0, Pg0, t0;
    double gC, gG;
    double sC, sG;
    double Pc_eff, Pg_eff;
    double r_star;
    double r_perf = 0.0;
    double r_energy = 0.0;
    double cpu_freq_scale = 1.0;
    double gpu_freq_scale = 1.0;
    double thread_util = 1.0;
    double cpu_thread_scale = 1.0;
    double cpu_thread_penalty = 1.0;
    double cpu_power_proxy = 0.0;
    double gpu_power_proxy = 0.0;
    double eC_eff = 0.0;
    double eG_eff = 0.0;
    double perf_weight_pct = 82.0;
    double energy_weight_pct = 18.0;
    double ratio_weight_pct = 0.0;
    double target_ratio_pct = 45.0;
    double dec_host_penalty_pct = 0.0;
    double min_ratio = 0.0;
    double max_ratio = 1.0;
    double block_bytes;
    double size_norm;
    double block_norm;
    double parallel_norm;
    double pg_parallel_scale;
    double ratio_norm = 0.5;
    double compressibility = 0.5;
    double gpu_overhead_scale;
    double t0_eff;
    double device_adv;
    double center;
    double width;
    long thread_count;
    long total_cores;
    double B = (double)input_size;

    if (sample_ratio_pct_out) *sample_ratio_pct_out = 0.0;
    if (!input || input_size == 0 || num_blocks == 0 || !cfg || !ocl)
        return 0.5;

    if (ocl->adaptive_ratio_cache_valid &&
        ocl->adaptive_ratio_cache_input_size == input_size &&
        ocl->adaptive_ratio_cache_num_blocks == num_blocks &&
        ocl->adaptive_ratio_cache_block_size == cfg->block_size &&
        ocl->adaptive_ratio_cache_acceleration == cfg->acceleration &&
        ocl->adaptive_ratio_cache_cpu_threads == cfg->cpu_threads &&
        ocl->adaptive_ratio_cache_sample_blocks == cfg->adaptive_sample_blocks) {
        if (sample_ratio_pct_out) {
            *sample_ratio_pct_out = ocl->adaptive_ratio_cache_sample_ratio_pct;
        }
        return ocl->adaptive_ratio_cache_value;
    }

    memset(&stats, 0, sizeof(stats));

    /* --- 1. Device capability profile (cached) --- */
    calibrate_device_profile(ocl, cfg);
    Pc0 = g_dev_profile.cpu_throughput;
    Pg0 = g_dev_profile.gpu_throughput;
    t0  = g_dev_profile.gpu_overhead_s;

    if (Pc0 > 0.0 && Pg0 > 0.0) {
        double rel = Pc0 / Pg0;
        if (rel > 3.0) Pc0 = Pg0 * 3.0;
        if (rel < 0.25) Pc0 = Pg0 * 0.25;
    }

    /* --- Determine thread count --- */
    total_cores = get_online_cpu_count();
    if (total_cores <= 0) total_cores = 4;
    if (cfg->cpu_threads > 0)
        thread_count = cfg->cpu_threads;
    else
        thread_count = total_cores;
    cpu_thread_scale = lz4_cpu_thread_scale(thread_count, g_dev_profile.is_unified_memory);
    thread_util = (double)thread_count / (double)total_cores;
    if (thread_util < 0.10) thread_util = 0.10;
    if (thread_util > 1.0) thread_util = 1.0;

    /* --- 2. Data characteristics from sampling --- */
    gC = 1.0;
    gG = 1.0;
    if (collect_lz4_sample_stats(input, input_size, num_blocks, cfg, &stats, sample_ratio_pct_out) == 0) {
        ratio_norm = lz4_clamp_double(stats.mean_ratio_pct / 100.0, 0.0, 1.0);
        compressibility = 1.0 - ratio_norm;
        if (stats.sample_cpu_throughput > 0.0 && Pc0 > 0.0) {
            gC = lz4_clamp_double(stats.sample_cpu_throughput / Pc0, 0.30, 1.25);
        } else {
            gC = lz4_clamp_double(1.0 + 0.24 * (compressibility - 0.5), 0.72, 1.32);
        }
        gC *= (0.92 + 0.16 * compressibility);
        gC = lz4_clamp_double(gC, 0.30, 1.32);
        gG = lz4_clamp_double(1.0 + 0.18 * (0.5 - compressibility), 0.70, 1.28);
    }

    /* --- 3. Runtime state --- */
    sC = read_cpu_availability();
    sG = read_gpu_availability();
    cpu_freq_scale = read_cpu_freq_scale();
    gpu_freq_scale = read_gpu_freq_scale();

    lz4_adaptive_choose_objective_weights((size_t)B,
                                          num_blocks,
                                          stats.sample_count > 0 ? stats.mean_ratio_pct : 0.0,
                                          g_dev_profile.is_unified_memory,
                                          sC,
                                          sG,
                                          thread_count,
                                          total_cores,
                                          cpu_freq_scale,
                                          gpu_freq_scale,
                                          &perf_weight_pct,
                                          &energy_weight_pct,
                                          &ratio_weight_pct,
                                          &target_ratio_pct,
                                          &dec_host_penalty_pct);

    if (cfg->cpu_threads > 0 && cfg->cpu_threads < total_cores) {
        double scale = (double)total_cores / (double)cfg->cpu_threads;
        sC = sC * scale;
        if (sC > 1.0) sC = 1.0;
    }

    block_bytes = (cfg->block_size > 0)
        ? (double)cfg->block_size
        : ((num_blocks > 0) ? (B / (double)num_blocks) : 65536.0);
    if (block_bytes < 4096.0) block_bytes = 4096.0;

    size_norm = lz4_norm_from_bytes(B, 1.0, 8.0);
    block_norm = lz4_norm_from_bytes(block_bytes, 0.0625, 2.5);
    parallel_norm = 1.0 - exp(-(double)num_blocks / 192.0);
    pg_parallel_scale = 0.55 + 2.45 * parallel_norm;

    if (thread_count <= 2) {
        cpu_thread_penalty = 1.0 - (0.20 + 0.08 * (1.0 - size_norm)) * (1.0 - thread_util);
    }
    cpu_thread_penalty = lz4_clamp_double(cpu_thread_penalty, 0.58, 1.08);

    gpu_overhead_scale = 0.85 + (1.0 - size_norm) * (0.95 - 0.35 * block_norm);
    gpu_overhead_scale = lz4_clamp_double(gpu_overhead_scale, 0.45, 1.90);

    Pc_eff = Pc0 * gC * sC * cpu_thread_scale * cpu_freq_scale * cpu_thread_penalty;
    Pg_eff = Pg0 * gG * sG * gpu_freq_scale * (0.76 + 0.24 * size_norm + 0.14 * block_norm) * pg_parallel_scale;

    if (Pc_eff < 1.0) Pc_eff = 1.0;
    if (Pg_eff < 1.0) Pg_eff = 1.0;

    t0_eff = t0 * gpu_overhead_scale;

    if (Pc_eff + Pg_eff <= 0.0) return 0.5;
    r_star = Pg_eff / (Pc_eff + Pg_eff);
    if (B > 0.0 && t0_eff > 0.0) {
        r_star -= (t0_eff * Pc_eff * Pg_eff) / (B * (Pc_eff + Pg_eff));
    }
    r_perf = r_star;

    if (gpu_freq_scale > cpu_freq_scale + 0.05) {
        r_star += 0.05 * (gpu_freq_scale - cpu_freq_scale);
    } else if (gpu_freq_scale + 0.10 < cpu_freq_scale) {
        r_star -= 0.05 * (cpu_freq_scale - gpu_freq_scale);
    }

    if (stats.sample_count > 0 && ratio_weight_pct > 0.0) {
        double ratio_shift = (ratio_weight_pct / 100.0) * (0.14 * compressibility - 0.06 * ratio_norm);
        r_star -= ratio_shift * (0.35 + 0.65 * size_norm);
    }

    /* 轻量能耗项只做弱正则，主目标仍是时延最短 */
    {
        cpu_power_proxy = (0.35 + 0.65 * cpu_freq_scale * cpu_freq_scale) * (0.45 + 0.55 * thread_util);
        gpu_power_proxy = (0.45 + 0.55 * gpu_freq_scale * gpu_freq_scale);

        eC_eff = g_dev_profile.cpu_energy_per_byte;
        eG_eff = g_dev_profile.gpu_energy_per_byte;
        if (eC_eff <= 0.0 && Pc_eff > 0.0) eC_eff = cpu_power_proxy / Pc_eff;
        if (eG_eff <= 0.0 && Pg_eff > 0.0) eG_eff = gpu_power_proxy / Pg_eff;

        if (eC_eff > 0.0 && eG_eff > 0.0) {
            r_energy = (Pg_eff * eC_eff) / (Pc_eff * eG_eff + Pg_eff * eC_eff);
            double sum_w = perf_weight_pct + energy_weight_pct;
            if (sum_w <= 0.0) {
                perf_weight_pct = 82.0;
                energy_weight_pct = 18.0;
                sum_w = 100.0;
            }
            r_star = (perf_weight_pct * r_perf + energy_weight_pct * r_energy) / sum_w;
        }
    }

    if (dec_host_penalty_pct > 0.0) {
        double host_pen = (dec_host_penalty_pct / 100.0) * (0.55 + 0.45 * (1.0 - size_norm));
        host_pen = lz4_clamp_double(host_pen, 0.0, 0.20);
        {
            double factor = 1.0 - host_pen;
            if (factor < 0.80) factor = 0.80;
            r_star *= factor;
        }
    }

    device_adv = (Pg_eff - Pc_eff) / (Pg_eff + Pc_eff);
    center = 0.50 + 0.44 * device_adv;
    center += 0.05 * (gpu_freq_scale - cpu_freq_scale);
    center += 0.04 * (0.5 - ratio_norm);
    center = lz4_clamp_double(center, 0.02, 0.98);

    width = 0.16 + 0.24 * size_norm + 0.10 * (1.0 - fabs(device_adv));
    width -= 0.06 * (1.0 - block_norm);
    width = lz4_clamp_double(width, 0.12, 0.42);

    min_ratio = lz4_clamp_double(center - width, 0.0, 0.94);
    max_ratio = lz4_clamp_double(center + width, 0.06, 1.0);
    if (max_ratio < min_ratio) max_ratio = min_ratio;

    if (r_star < min_ratio) r_star = min_ratio;
    if (r_star > max_ratio) r_star = max_ratio;

    r_star = lz4_refine_ratio_candidate((size_t)B,
                                        num_blocks,
                                        Pc_eff,
                                        Pg_eff,
                                        t0_eff,
                                        cpu_freq_scale,
                                        gpu_freq_scale,
                                        thread_util,
                                        stats.sample_count > 0 ? stats.mean_ratio_pct : 0.0,
                                        r_star,
                                        min_ratio,
                                        max_ratio);

    if (r_star < min_ratio) r_star = min_ratio;
    if (r_star > max_ratio) r_star = max_ratio;
    r_star = lz4_clamp_double(r_star, 0.0, 1.0);

    if (cfg->verbose) {
        fprintf(stderr,
                "Adaptive(E2E): Pc0=%.0f gC=%.2f sC=%.2f threads=%ld cpuScale=%.2f cpuPen=%.2f Pc_eff=%.0f | "
                "Pg0=%.0f gG=%.2f sG=%.2f Pg_eff=%.0f | "
                "freqC=%.2f freqG=%.2f util=%.2f sizeN=%.2f blockN=%.2f parN=%.2f pScale=%.2f ratioN=%.2f pC=%.3f pG=%.3f | "
                "eC=%.2e eG=%.2e rPerf=%.4f rEnergy=%.4f perfW=%.1f energyW=%.1f ratioW=%.1f targetR=%.1f decHostPen=%.1f | "
                "t0=%.6f t0eff=%.6f B=%.0f center=%.4f width=%.4f min=%.4f max=%.4f r*=%.4f\n",
                Pc0, gC, sC, thread_count, cpu_thread_scale, cpu_thread_penalty, Pc_eff,
                Pg0, gG, sG, Pg_eff,
                cpu_freq_scale,
                gpu_freq_scale,
                thread_util,
                size_norm,
                block_norm,
                parallel_norm,
                pg_parallel_scale,
                ratio_norm,
                cpu_power_proxy,
                gpu_power_proxy,
                eC_eff,
                eG_eff,
                r_perf,
                r_energy,
                perf_weight_pct,
                energy_weight_pct,
                ratio_weight_pct,
                target_ratio_pct,
                dec_host_penalty_pct,
                t0,
                t0_eff,
                B,
                center,
                width,
                min_ratio,
                max_ratio,
                r_star);
    }

    ocl->adaptive_ratio_cache_valid = 1;
    ocl->adaptive_ratio_cache_input_size = input_size;
    ocl->adaptive_ratio_cache_num_blocks = num_blocks;
    ocl->adaptive_ratio_cache_block_size = cfg->block_size;
    ocl->adaptive_ratio_cache_acceleration = cfg->acceleration;
    ocl->adaptive_ratio_cache_cpu_threads = cfg->cpu_threads;
    ocl->adaptive_ratio_cache_sample_blocks = cfg->adaptive_sample_blocks;
    ocl->adaptive_ratio_cache_value = r_star;
    ocl->adaptive_ratio_cache_sample_ratio_pct = stats.mean_ratio_pct;

    return r_star;
}

static int read_entire_file(const char* path, unsigned char** out_buf, size_t* out_size) {
    FILE* f = NULL;
    unsigned char* buf = NULL;
    struct stat st;

    if (!path || !out_buf || !out_size) return -1;
    *out_buf = NULL;
    *out_size = 0;

    if (strcmp(path, "-") == 0) {
        size_t cap = 1U << 20;
        size_t used = 0;
        size_t nread;
        buf = (unsigned char*)malloc(cap);
        if (!buf) return -1;

        while ((nread = fread(buf + used, 1, cap - used, stdin)) > 0) {
            used += nread;
            if (used == cap) {
                size_t new_cap = cap * 2;
                unsigned char* nb = (unsigned char*)realloc(buf, new_cap);
                if (!nb) {
                    free(buf);
                    return -1;
                }
                buf = nb;
                cap = new_cap;
            }
        }
        if (ferror(stdin) || used == 0) {
            free(buf);
            return -1;
        }

        *out_buf = buf;
        *out_size = used;
        return 0;
    }

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

static int path_is_dash(const char* path) {
    return path && strcmp(path, "-") == 0;
}

static int write_entire_file(const char* path, const unsigned char* buf, size_t size) {
    FILE* f = NULL;
    if (!path || !buf || size == 0) return -1;

    if (strcmp(path, "-") == 0) {
        if (fwrite(buf, 1, size, stdout) != size) return -1;
        if (fflush(stdout) != 0) return -1;
        return 0;
    }

    f = fopen(path, "wb");
    if (!f) return -1;
    if (fwrite(buf, 1, size, f) != size) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static void ocl_release_kernels_only(ocl_env_t* ocl) {
    if (!ocl) return;
    if (ocl->kdec_mapped) { clReleaseKernel(ocl->kdec_mapped); ocl->kdec_mapped = NULL; }
    if (ocl->kdec) { clReleaseKernel(ocl->kdec); ocl->kdec = NULL; }
    if (ocl->kcomp_mapped) { clReleaseKernel(ocl->kcomp_mapped); ocl->kcomp_mapped = NULL; }
    if (ocl->kcomp) { clReleaseKernel(ocl->kcomp); ocl->kcomp = NULL; }
}

static int ocl_create_required_kernels(ocl_env_t* ocl) {
    cl_int err;
    if (!ocl || !ocl->prog) return -1;

    ocl->kcomp = clCreateKernel(ocl->prog, "lz4_compress_block", &err);
    if (err != CL_SUCCESS || !ocl->kcomp) return -1;

    ocl->kcomp_mapped = clCreateKernel(ocl->prog, "lz4_compress_blocks_mapped", &err);
    if (err != CL_SUCCESS || !ocl->kcomp_mapped) return -1;

    ocl->kdec = clCreateKernel(ocl->prog, "lz4_decompress_blocks", &err);
    if (err != CL_SUCCESS || !ocl->kdec) return -1;

    ocl->kdec_mapped = clCreateKernel(ocl->prog, "lz4_decompress_blocks_mapped", &err);
    if (err != CL_SUCCESS || !ocl->kdec_mapped) return -1;

    return 0;
}

static int ocl_init(ocl_env_t* ocl) {
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

#if defined(CL_VERSION_2_0)
    {
        const cl_queue_properties props[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
        };
        ocl->queue = clCreateCommandQueueWithProperties(ocl->ctx, ocl->dev, props, &err);
    }
#else
    ocl->queue = clCreateCommandQueue(ocl->ctx, ocl->dev, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (err != CL_SUCCESS || !ocl->queue) {
        fprintf(stderr, "OpenCL init failed: clCreateCommandQueue err=%d\n", err);
        return -1;
    }

    ocl->prog = lz4_load_program(ocl->ctx, ocl->dev);
    if (!ocl->prog) {
        fprintf(stderr, "OpenCL init failed: lz4_load_program returned NULL\n");
        return -1;
    }

    if (ocl_create_required_kernels(ocl) != 0) {
        fprintf(stderr, "OpenCL init failed: missing required kernels in loaded program (please regenerate matching .clbin with Makefile precompile)\n");
        return -1;
    }

    return 0;
}

static void ocl_free(ocl_env_t* ocl) {
    if (!ocl) return;
    lz4_gpu_workspace_free(&ocl->ws);
    ocl_release_kernels_only(ocl);
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

static size_t lz4_adaptive_adjust_gpu_blocks(size_t num_blocks, double gpu_ratio, int* collapsed_out) {
    size_t gpu_blocks;
    unsigned min_mixed;
    unsigned quantum;
    int collapse_set = 0;
    int collapse_small;

    if (collapsed_out) *collapsed_out = 0;
    if (num_blocks == 0) return 0;
    if (gpu_ratio <= 0.0) return 0;
    if (gpu_ratio >= 1.0) return num_blocks;

    gpu_blocks = (size_t)((double)num_blocks * gpu_ratio + 0.5);
    if (gpu_blocks > num_blocks) gpu_blocks = num_blocks;

    min_mixed = 8U;
    quantum = 4U;
    collapse_small = 1;
    (void)collapse_set;

    if (min_mixed > 0) {
        if (num_blocks <= (size_t)min_mixed * 2U) {
            if (collapse_small && gpu_blocks > 0 && gpu_blocks < num_blocks) {
                gpu_blocks = (gpu_blocks * 2 >= num_blocks) ? num_blocks : 0;
                if (collapsed_out) *collapsed_out = 1;
            }
        } else {
            if (gpu_blocks > 0 && gpu_blocks < (size_t)min_mixed) gpu_blocks = (size_t)min_mixed;
            if (gpu_blocks < num_blocks && (num_blocks - gpu_blocks) < (size_t)min_mixed) {
                gpu_blocks = num_blocks - (size_t)min_mixed;
            }
        }
    }

    if (quantum > 1 && gpu_blocks > 0 && gpu_blocks < num_blocks) {
        size_t rounded = ((gpu_blocks + (size_t)quantum / 2U) / (size_t)quantum) * (size_t)quantum;
        if (min_mixed > 0 && rounded < (size_t)min_mixed) rounded = (size_t)min_mixed;
        if (min_mixed > 0 && rounded > num_blocks - (size_t)min_mixed) rounded = num_blocks - (size_t)min_mixed;
        if (rounded == 0) rounded = 1;
        if (rounded >= num_blocks) rounded = num_blocks - 1;
        gpu_blocks = rounded;
    }

    return gpu_blocks;
}

static uint64_t lz4_hash_u32_array(const uint32_t* data, size_t count) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = (const unsigned char*)(const void*)data;
    size_t nbytes = count * sizeof(uint32_t);
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void lz4_adaptive_choose_objective_weights(size_t input_size,
                                                  size_t num_blocks,
                                                  double sample_ratio_pct,
                                                  int is_unified_memory,
                                                  double cpu_avail,
                                                  double gpu_avail,
                                                  long thread_count,
                                                  long total_cores,
                                                  double cpu_freq_scale,
                                                  double gpu_freq_scale,
                                                  double* perf_weight_pct,
                                                  double* energy_weight_pct,
                                                  double* ratio_weight_pct,
                                                  double* target_ratio_pct,
                                                  double* dec_host_penalty_pct) {
    double perf = 68.0;
    double energy = 22.0;
    double ratio = 10.0;
    double target_ratio = 45.0;
    double dec_penalty = 2.5;
    double thread_util = 1.0;
    const size_t small_file = 4U * 1024U * 1024U;
    const size_t large_file = 64U * 1024U * 1024U;

    (void)num_blocks;

    if (total_cores > 0) {
        thread_util = (double)thread_count / (double)total_cores;
        if (thread_util < 0.10) thread_util = 0.10;
        if (thread_util > 1.0) thread_util = 1.0;
    }

    if (input_size <= small_file) {
        perf = 84.0;
        energy = 12.0;
        ratio = 4.0;
        dec_penalty = 4.0;
    } else if (input_size >= large_file) {
        perf = 64.0;
        energy = 24.0;
        ratio = 12.0;
        dec_penalty = 5.0;
    }

    if (sample_ratio_pct > 65.0) {
        perf += 4.0;
        ratio -= 4.0;
        target_ratio = 52.0;
    } else if (sample_ratio_pct < 45.0 && sample_ratio_pct > 0.0) {
        perf -= 2.0;
        ratio += 5.0;
        target_ratio = 42.0;
    }

    if (!is_unified_memory) {
        energy += 5.0;
        perf -= 3.0;
    }

    if (thread_util >= 0.75) {
        energy += 2.0;
        perf -= 1.0;
    } else if (thread_util <= 0.35) {
        perf += 4.0;
        energy -= 2.0;
    }

    if (cpu_freq_scale < 0.80 && gpu_freq_scale > cpu_freq_scale) {
        perf += 4.0;
        energy += 3.0;
        ratio -= 2.0;
        dec_penalty -= 2.0;
    }
    if (gpu_freq_scale < 0.75 && cpu_freq_scale >= gpu_freq_scale) {
        perf -= 3.0;
        energy += 2.0;
        ratio += 2.0;
        dec_penalty += 2.0;
    }

    if (cpu_avail < 0.35) dec_penalty += 2.5;
    if (gpu_avail < 0.35) dec_penalty -= 3.0;

    if (ratio < 2.0) ratio = 2.0;
    if (energy < 10.0) energy = 10.0;
    if (perf < 25.0) perf = 25.0;
    if (dec_penalty < 0.0) dec_penalty = 0.0;
    if (dec_penalty > 8.0) dec_penalty = 8.0;

    {
        double sum = perf + energy + ratio;
        if (sum <= 0.0) {
            perf = 66.0;
            energy = 22.0;
            ratio = 12.0;
            sum = 100.0;
        }
        perf = perf * 100.0 / sum;
        energy = energy * 100.0 / sum;
        ratio = ratio * 100.0 / sum;
    }

    if (perf_weight_pct) *perf_weight_pct = perf;
    if (energy_weight_pct) *energy_weight_pct = energy;
    if (ratio_weight_pct) *ratio_weight_pct = ratio;
    if (target_ratio_pct) *target_ratio_pct = target_ratio;
    if (dec_host_penalty_pct) *dec_host_penalty_pct = dec_penalty;
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

    default_wi_per_cu = (num_blocks >= 4096) ? 16 : LZ4_HYBRID_COMP_WI_PER_CU_DEFAULT;
    wi_per_cu = default_wi_per_cu;
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
    wi_per_cu = default_wi_per_cu;
    target = (size_t)cu * wi_per_cu;
    if (target < local_size) target = local_size;
    if (target > num_blocks) target = num_blocks;
    if (target == 0) target = 1;
    return target;
}

static int choose_lz4_table_type(size_t block_size) {
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

static int hybrid_device_host_unified_memory(cl_command_queue queue) {
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

static int hybrid_prefers_standard_copy(cl_command_queue queue) {
    const char* env = getenv("LZ4_STANDARD_COPY");
    if (env && *env) {
        if (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0 || strcasecmp(env, "yes") == 0) return 1;
        if (strcmp(env, "0") == 0 || strcasecmp(env, "false") == 0 || strcasecmp(env, "no") == 0) return 0;
    }
    return hybrid_device_host_unified_memory(queue) ? 0 : 1;
}

static int hybrid_write_buffer_auto(cl_command_queue queue, cl_mem buf, const void* src, size_t bytes, int standard_copy) {
    cl_int err;
    if (!buf || !src || bytes == 0) return 0;

    if (standard_copy) {
        err = clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, src, 0, NULL, NULL);
        return (err == CL_SUCCESS) ? 0 : -1;
    }

    {
        void* mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
        if (err == CL_SUCCESS && mapped) {
            memcpy(mapped, src, bytes);
            err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
            if (err != CL_SUCCESS) return -1;
            return 0;
        }
    }

    err = clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, src, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

static int hybrid_read_buffer_auto(cl_command_queue queue, cl_mem buf, void* dst, size_t bytes, int standard_copy) {
    cl_int err;
    if (!buf || !dst || bytes == 0) return 0;

    if (standard_copy) {
        err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
        return (err == CL_SUCCESS) ? 0 : -1;
    }

    {
        void* mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &err);
        if (err == CL_SUCCESS && mapped) {
            memcpy(dst, mapped, bytes);
            err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
            if (err != CL_SUCCESS) return -1;
            return 0;
        }
    }

    err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

static void* cpu_comp_worker(void* argp) {
    cpu_comp_worker_arg_t* arg = (cpu_comp_worker_arg_t*)argp;
    cpu_comp_job_t* job = arg->job;
    const uint64_t t0 = get_us();
    const int accel = (job->acceleration > 0) ? job->acceleration : 1;
    const size_t base = job->block_index_base;
    int out_cap;

    (void)arg->thread_idx;
    (void)arg->thread_count;

    if (job->out_slot_size > (size_t)INT32_MAX) {
        job->err = 1;
        return NULL;
    }
    out_cap = (int)job->out_slot_size;

    if (job->block_indices) {
        for (;;) {
            const size_t i = atomic_fetch_add(&job->next_block, 1);
            size_t g;
            size_t in_sz;
            const unsigned char* src_ptr;
            unsigned char* dst_ptr;
            int csz;
            if (i >= job->num_blocks) break;
            g = job->block_indices[i];
            in_sz = block_input_size(job->src_size, job->block_size, g);
            src_ptr = job->src + g * job->block_size;
            dst_ptr = job->out_slots + i * job->out_slot_size;
            if (in_sz == 0 || in_sz > (size_t)INT32_MAX) {
                job->err = 1;
                return NULL;
            }
            csz = LZ4_compress_fast((const char*)src_ptr,
                                    (char*)dst_ptr,
                                    (int)in_sz,
                                    out_cap,
                                    accel);
            if (csz <= 0) {
                job->err = 1;
                return NULL;
            }
            job->out_sizes[i] = (uint32_t)csz;
        }
    } else {
        for (;;) {
            const size_t i = atomic_fetch_add(&job->next_block, 1);
            size_t g;
            size_t in_sz;
            const unsigned char* src_ptr;
            unsigned char* dst_ptr;
            int csz;
            if (i >= job->num_blocks) break;
            g = base + i;
            in_sz = block_input_size(job->src_size, job->block_size, g);
            src_ptr = job->src + g * job->block_size;
            dst_ptr = job->out_slots + i * job->out_slot_size;
            if (in_sz == 0 || in_sz > (size_t)INT32_MAX) {
                job->err = 1;
                return NULL;
            }
            csz = LZ4_compress_fast((const char*)src_ptr,
                                    (char*)dst_ptr,
                                    (int)in_sz,
                                    out_cap,
                                    accel);
            if (csz <= 0) {
                job->err = 1;
                return NULL;
            }
            job->out_sizes[i] = (uint32_t)csz;
        }
    }
    arg->elapsed_us = get_us() - t0;
    return NULL;
}

static void* cpu_comp_top(void* argp) {
    cpu_comp_job_t* job = (cpu_comp_job_t*)argp;
    pthread_t* tids = NULL;
    cpu_comp_worker_arg_t* args = NULL;
    pthread_t tids_stack[64];
    cpu_comp_worker_arg_t args_stack[64];
    int use_heap = 0;
    uint64_t kernel_us = 0;
    int tcount = job->num_threads;
    int created = 0;

    if (job->num_blocks == 0) {
        job->elapsed_us = 0;
        return NULL;
    }
    atomic_store(&job->next_block, 0);
    if (tcount <= 0) tcount = 1;
    if ((size_t)tcount > job->num_blocks) tcount = (int)job->num_blocks;

    if (tcount == 1) {
        cpu_comp_worker_arg_t arg = { .thread_idx = 0, .thread_count = 1, .job = job, .elapsed_us = 0 };
        cpu_comp_worker(&arg);
        job->elapsed_us = arg.elapsed_us;
        return NULL;
    }

    if (tcount <= (int)(sizeof(tids_stack) / sizeof(tids_stack[0]))) {
        tids = tids_stack;
        args = args_stack;
    } else {
        use_heap = 1;
        tids = (pthread_t*)malloc((size_t)tcount * sizeof(pthread_t));
        args = (cpu_comp_worker_arg_t*)malloc((size_t)tcount * sizeof(cpu_comp_worker_arg_t));
        if (!tids || !args) {
            job->err = 1;
            free(tids);
            free(args);
            return NULL;
        }
    }

    for (int i = 0; i < tcount; ++i) {
        args[i].thread_idx = (size_t)i;
        args[i].thread_count = (size_t)tcount;
        args[i].job = job;
        args[i].elapsed_us = 0;
        if (pthread_create(&tids[i], NULL, cpu_comp_worker, &args[i]) != 0) {
            job->err = 1;
            created = i;
            break;
        }
        created = i + 1;
    }
    for (int i = 0; i < created; ++i) pthread_join(tids[i], NULL);
    for (int i = 0; i < created; ++i) {
        if (args[i].elapsed_us > kernel_us) kernel_us = args[i].elapsed_us;
    }
    job->elapsed_us = kernel_us;
    if (use_heap) {
        free(tids);
        free(args);
    }
    return NULL;
}

static void* cpu_decomp_worker(void* argp) {
    cpu_decomp_worker_arg_t* arg = (cpu_decomp_worker_arg_t*)argp;
    cpu_decomp_job_t* job = arg->job;
    const uint64_t t0 = get_us();
    const size_t base = job->block_index_base;

    (void)arg->thread_idx;
    (void)arg->thread_count;

    if (job->block_indices) {
        for (;;) {
            const size_t i = atomic_fetch_add(&job->next_block, 1);
            size_t g;
            int csz;
            unsigned char* dst_ptr;
            const unsigned char* src_ptr;
            int dsz;
            if (i >= job->num_blocks) break;
            g = job->block_indices[i];
            csz = (int)job->comp_sizes[i];
            dst_ptr = job->out_full + g * job->block_size;
            src_ptr = job->comp_data + (size_t)job->comp_offsets[i];

            dsz = LZ4_decompress_safe((const char*)src_ptr, (char*)dst_ptr, csz, (int)job->block_size);
            if (dsz < 0) {
                job->err = 1;
                return NULL;
            }
            job->out_sizes[i] = (uint32_t)dsz;
        }
    } else {
        for (;;) {
            const size_t i = atomic_fetch_add(&job->next_block, 1);
            unsigned char* dst_ptr;
            int dsz;
            int csz;
            const unsigned char* src_ptr;
            if (i >= job->num_blocks) break;

            csz = (int)job->comp_sizes[i];
            src_ptr = job->comp_data + (size_t)job->comp_offsets[i];
            dst_ptr = job->out_full + (base + i) * job->block_size;

            dsz = LZ4_decompress_safe((const char*)src_ptr, (char*)dst_ptr, csz, (int)job->block_size);
            if (dsz < 0) {
                job->err = 1;
                return NULL;
            }
            job->out_sizes[i] = (uint32_t)dsz;
        }
    }
    arg->elapsed_us = get_us() - t0;
    return NULL;
}

static void* cpu_decomp_top(void* argp) {
    cpu_decomp_job_t* job = (cpu_decomp_job_t*)argp;
    pthread_t* tids = NULL;
    cpu_decomp_worker_arg_t* args = NULL;
    pthread_t tids_stack[64];
    cpu_decomp_worker_arg_t args_stack[64];
    int use_heap = 0;
    uint64_t kernel_us = 0;
    int tcount = job->num_threads;
    int created = 0;

    if (job->num_blocks == 0) {
        job->elapsed_us = 0;
        return NULL;
    }
    atomic_store(&job->next_block, 0);
    if (tcount <= 0) tcount = 1;
    if ((size_t)tcount > job->num_blocks) tcount = (int)job->num_blocks;

    if (tcount == 1) {
        cpu_decomp_worker_arg_t arg = { .thread_idx = 0, .thread_count = 1, .job = job, .elapsed_us = 0 };
        cpu_decomp_worker(&arg);
        job->elapsed_us = arg.elapsed_us;
        return NULL;
    }

    if (tcount <= (int)(sizeof(tids_stack) / sizeof(tids_stack[0]))) {
        tids = tids_stack;
        args = args_stack;
    } else {
        use_heap = 1;
        tids = (pthread_t*)malloc((size_t)tcount * sizeof(pthread_t));
        args = (cpu_decomp_worker_arg_t*)malloc((size_t)tcount * sizeof(cpu_decomp_worker_arg_t));
        if (!tids || !args) {
            job->err = 1;
            free(tids);
            free(args);
            return NULL;
        }
    }

    for (int i = 0; i < tcount; ++i) {
        args[i].thread_idx = (size_t)i;
        args[i].thread_count = (size_t)tcount;
        args[i].job = job;
        args[i].elapsed_us = 0;
        if (pthread_create(&tids[i], NULL, cpu_decomp_worker, &args[i]) != 0) {
            job->err = 1;
            created = i;
            break;
        }
        created = i + 1;
    }
    for (int i = 0; i < created; ++i) pthread_join(tids[i], NULL);
    for (int i = 0; i < created; ++i) {
        if (args[i].elapsed_us > kernel_us) kernel_us = args[i].elapsed_us;
    }
    job->elapsed_us = kernel_us;
    if (use_heap) {
        free(tids);
        free(args);
    }
    return NULL;
}

static int gpu_compress_blocks(ocl_env_t* ocl,
                               const unsigned char* src,
                               size_t src_size,
                               size_t block_size,
                               int acceleration,
                               int local_size,
                               const uint32_t* mapped_block_indices,
                               size_t mapped_block_count,
                               uint32_t** out_sizes,
                               uint32_t** out_offsets,
                               unsigned char** out_slots,
                               size_t* out_slot_size,
                               uint64_t* kernel_us,
                               uint64_t* upload_us,
                               uint64_t* download_us,
                               int skip_input_upload) {
    const int hash_log = 14;
    cl_int err = CL_SUCCESS;
    uint32_t* h_packed_offsets = NULL;
    uint32_t* h_sizes = NULL;
    unsigned char* h_out = NULL;
    unsigned char* packed_out = NULL;
    size_t num_blocks;
    size_t single_block_max_out;
    uint32_t single_block_max_out_u32;
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
    int use_standard_copy;
    int use_mapped_indices;
    const int map_cache_enabled = 1;
    uint64_t mapped_hash = 0;
    int mapped_upload_needed = 1;
    cl_mem prev_block_info_buf = NULL;
    uint64_t upload_acc = 0;
    uint64_t download_acc = 0;

    if (!ocl || !src || src_size == 0 || block_size == 0 || !out_sizes || !out_offsets || !out_slots || !out_slot_size || !kernel_us) return -1;

    use_mapped_indices = (mapped_block_indices != NULL && mapped_block_count > 0);

    *out_sizes = NULL;
    *out_offsets = NULL;
    *out_slots = NULL;
    *out_slot_size = 0;
    *kernel_us = 0;
    if (upload_us) *upload_us = 0;
    if (download_us) *download_us = 0;
    use_standard_copy = hybrid_prefers_standard_copy(ocl->queue);

    if (use_mapped_indices) {
        num_blocks = mapped_block_count;
    } else {
        num_blocks = (src_size + block_size - 1) / block_size;
    }
    if (num_blocks == 0) return -1;
    if (num_blocks > (size_t)INT32_MAX) return -1;

    single_block_max_out = (size_t)((double)block_size * 1.1 + 64.0);
    if (single_block_max_out < (size_t)LZ4_compressBound((int)block_size)) {
        single_block_max_out = (size_t)LZ4_compressBound((int)block_size);
    }
    if (single_block_max_out > UINT32_MAX) return -1;
    single_block_max_out_u32 = (uint32_t)single_block_max_out;

    h_packed_offsets = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    h_sizes = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    if (!h_packed_offsets || !h_sizes) goto fail;

    lsz = sanitize_local_size(ocl->queue, (size_t)((local_size > 0) ? local_size : 1), num_blocks);
    gsz = round_up_size(choose_comp_worker_count(ocl->queue, num_blocks, lsz), lsz);
    if (gsz == 0) gsz = 1;

    tableType = choose_lz4_table_type(block_size);

    ocl->ws.comp_in_buf = ensure_buffer(ocl->ctx, ocl->ws.comp_in_buf, src_size, &ocl->ws.current_comp_in_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.comp_in_buf) goto fail;
    ocl->ws.out_buf = ensure_buffer(ocl->ctx, ocl->ws.out_buf, num_blocks * single_block_max_out, &ocl->ws.current_out_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.out_buf) goto fail;
    ocl->ws.output_size_buf = ensure_buffer(ocl->ctx, ocl->ws.output_size_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_osize_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.output_size_buf) goto fail;
    if (use_mapped_indices) {
        prev_block_info_buf = ocl->ws.block_info_buf;
        ocl->ws.block_info_buf = ensure_buffer(ocl->ctx,
                                               ocl->ws.block_info_buf,
                                               num_blocks * sizeof(uint32_t),
                                               &ocl->ws.current_blocks_capacity,
                                               &err);
        if (err != CL_SUCCESS || !ocl->ws.block_info_buf) goto fail;
        if (ocl->ws.block_info_buf != prev_block_info_buf) {
            ocl->cached_block_map_valid = 0;
        }
        mapped_hash = lz4_hash_u32_array(mapped_block_indices, num_blocks);
        if (map_cache_enabled && ocl->cached_block_map_valid &&
            ocl->cached_block_map_count == num_blocks &&
            ocl->cached_block_map_hash == mapped_hash) {
            mapped_upload_needed = 0;
        }
    }

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

    if (!skip_input_upload) {
        const uint64_t t_up0 = get_us();
        if (hybrid_write_buffer_auto(ocl->queue, ocl->ws.comp_in_buf, src, src_size, use_standard_copy) != 0) {
            goto fail;
        }
        upload_acc += (get_us() - t_up0);
    }
    if (use_mapped_indices && mapped_upload_needed) {
        const uint64_t t_up0 = get_us();
        if (hybrid_write_buffer_auto(ocl->queue,
                                     ocl->ws.block_info_buf,
                                     mapped_block_indices,
                                     num_blocks * sizeof(uint32_t),
                                     use_standard_copy) != 0) {
            goto fail;
        }
        upload_acc += (get_us() - t_up0);
        if (map_cache_enabled) {
            ocl->cached_block_map_hash = mapped_hash;
            ocl->cached_block_map_count = num_blocks;
            ocl->cached_block_map_valid = 1;
        }
    }

    inputSize = (int)src_size;
    totalBlocks = (int)num_blocks;

    if (!skip_input_upload || use_mapped_indices) {
        err = CL_SUCCESS;
        if (use_mapped_indices) {
            err |= clSetKernelArg(ocl->kcomp_mapped, 0, sizeof(cl_mem), &ocl->ws.comp_in_buf);
            err |= clSetKernelArg(ocl->kcomp_mapped, 1, sizeof(cl_mem), &ocl->ws.out_buf);
            err |= clSetKernelArg(ocl->kcomp_mapped, 2, sizeof(cl_mem), &ocl->ws.output_size_buf);
            err |= clSetKernelArg(ocl->kcomp_mapped, 3, sizeof(cl_mem), &ocl->ws.block_info_buf);
            err |= clSetKernelArg(ocl->kcomp_mapped, 4, sizeof(int), &totalBlocks);
            err |= clSetKernelArg(ocl->kcomp_mapped, 5, sizeof(int), &inputSize);
            err |= clSetKernelArg(ocl->kcomp_mapped, 6, sizeof(int), &block_size);
            err |= clSetKernelArg(ocl->kcomp_mapped, 7, sizeof(uint32_t), &single_block_max_out_u32);
            err |= clSetKernelArg(ocl->kcomp_mapped, 8, sizeof(int), &tableType);
            err |= clSetKernelArg(ocl->kcomp_mapped, 9, sizeof(int), &acceleration);
            err |= clSetKernelArg(ocl->kcomp_mapped, 10, sizeof(int), &globalIndexBase);
            err |= clSetKernelArg(ocl->kcomp_mapped, 11, sizeof(cl_mem), &ocl->ws.dict_buf);
        } else {
            err |= set_kernel_mem_arg_if_changed(ocl->kcomp, 0, &ocl->cached_kcomp_arg0, ocl->ws.comp_in_buf);
            err |= set_kernel_mem_arg_if_changed(ocl->kcomp, 1, &ocl->cached_kcomp_arg1, ocl->ws.out_buf);
            err |= set_kernel_mem_arg_if_changed(ocl->kcomp, 2, &ocl->cached_kcomp_arg2, ocl->ws.output_size_buf);
            err |= clSetKernelArg(ocl->kcomp, 3, sizeof(int), &totalBlocks);
            err |= clSetKernelArg(ocl->kcomp, 4, sizeof(int), &inputSize);
            err |= clSetKernelArg(ocl->kcomp, 5, sizeof(int), &block_size);
            err |= clSetKernelArg(ocl->kcomp, 6, sizeof(uint32_t), &single_block_max_out_u32);
            err |= clSetKernelArg(ocl->kcomp, 7, sizeof(int), &tableType);
            err |= clSetKernelArg(ocl->kcomp, 8, sizeof(int), &acceleration);
            err |= clSetKernelArg(ocl->kcomp, 9, sizeof(int), &globalIndexBase);
            err |= set_kernel_mem_arg_if_changed(ocl->kcomp, 10, &ocl->cached_kcomp_arg10, ocl->ws.dict_buf);
        }
        if (err != CL_SUCCESS) goto fail;
    }
    if (use_mapped_indices) {
        err = clSetKernelArg(ocl->kcomp_mapped, 12, sizeof(uint32_t), &epoch_base);
    } else {
        err = clSetKernelArg(ocl->kcomp, 11, sizeof(uint32_t), &epoch_base);
    }
    if (err != CL_SUCCESS) goto fail;

    {
        const uint64_t t0 = get_us();
    sparse_total = num_blocks * single_block_max_out;

        err = clEnqueueNDRangeKernel(ocl->queue,
                         use_mapped_indices ? ocl->kcomp_mapped : ocl->kcomp,
                         1,
                         NULL,
                         &gsz,
                         &lsz,
                         0,
                         NULL,
                         NULL);
        if (err != CL_SUCCESS) goto fail;
        clFinish(ocl->queue);
        *kernel_us = get_us() - t0;
    }

    {
        const uint64_t t_down0 = get_us();
        if (hybrid_read_buffer_auto(ocl->queue, ocl->ws.output_size_buf, h_sizes, num_blocks * sizeof(uint32_t), use_standard_copy) != 0) goto fail;
        download_acc += (get_us() - t_down0);
    }

    for (size_t i = 0; i < num_blocks; ++i) {
        if (h_sizes[i] == 0xFFFFFFFFU) goto fail;
        h_packed_offsets[i] = (uint32_t)packed_total;
        packed_total += (size_t)h_sizes[i];
    }


    if (packed_total > 0) {
        packed_out = (unsigned char*)malloc(packed_total);
        if (!packed_out) goto fail;
    }

    if (sparse_total > 0) {
        h_out = (unsigned char*)malloc(sparse_total);
        if (!h_out) goto fail;
        {
            const uint64_t t_down0 = get_us();
        if (hybrid_read_buffer_auto(ocl->queue, ocl->ws.out_buf, h_out, sparse_total, use_standard_copy) != 0) goto fail;
            download_acc += (get_us() - t_down0);
        }
    }
    for (size_t i = 0; i < num_blocks; ++i) {
        memcpy(packed_out + h_packed_offsets[i], h_out + i * single_block_max_out, (size_t)h_sizes[i]);
    }

    *out_sizes = h_sizes;
    *out_offsets = h_packed_offsets;
    *out_slots = packed_out;
    *out_slot_size = packed_total;
    if (upload_us) *upload_us = upload_acc;
    if (download_us) *download_us = download_acc;

    h_sizes = NULL;
    h_packed_offsets = NULL;
    packed_out = NULL;

    free(h_packed_offsets);
    free(h_sizes);
    free(h_out);
    free(packed_out);
    return 0;

fail:
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
                                 uint64_t* kernel_us,
                                 uint64_t* upload_us,
                                 uint64_t* download_us) {
    cl_int err = CL_SUCCESS;
    uint32_t* h_comp_off = NULL;
    size_t comp_total = 0;
    size_t lsz;
    size_t gsz;
    uint32_t block_size_u32;
    uint32_t totalBlocks;
    uint64_t decomp_meta_hash;
    int need_upload_meta = 1;
    int use_standard_copy;
    uint64_t upload_acc = 0;
    uint64_t download_acc = 0;

    if (!ocl || !comp_data || !comp_sizes || !out_full || !out_sizes || !kernel_us) return -1;
    if (num_blocks == 0) {
        *kernel_us = 0;
        if (upload_us) *upload_us = 0;
        if (download_us) *download_us = 0;
        return 0;
    }
    if (num_blocks > (size_t)UINT32_MAX) return -1;
    if (block_size > (size_t)UINT32_MAX) return -1;
    block_size_u32 = (uint32_t)block_size;
    use_standard_copy = hybrid_prefers_standard_copy(ocl->queue);
    if (upload_us) *upload_us = 0;
    if (download_us) *download_us = 0;

    h_comp_off = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
    if (!h_comp_off) goto fail;

    for (size_t i = 0; i < num_blocks; ++i) {
        h_comp_off[i] = (uint32_t)comp_total;
        comp_total += (size_t)comp_sizes[i];
    }

    size_t prev_comp_off_cap = ocl->ws.current_decomp_comp_off_capacity;
    size_t prev_comp_size_cap = ocl->ws.current_decomp_comp_size_capacity;
    ocl->ws.in_buf = ensure_buffer(ocl->ctx, ocl->ws.in_buf, comp_total, &ocl->ws.current_in_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.in_buf) goto fail;
    ocl->ws.out_buf = ensure_buffer(ocl->ctx, ocl->ws.out_buf, num_blocks * block_size, &ocl->ws.current_out_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.out_buf) goto fail;
    ocl->ws.decomp_comp_off_buf = ensure_buffer(ocl->ctx, ocl->ws.decomp_comp_off_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_decomp_comp_off_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.decomp_comp_off_buf) goto fail;
    ocl->ws.decomp_comp_size_buf = ensure_buffer(ocl->ctx, ocl->ws.decomp_comp_size_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_decomp_comp_size_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.decomp_comp_size_buf) goto fail;
    ocl->ws.decomp_sizes_out_buf = ensure_buffer(ocl->ctx, ocl->ws.decomp_sizes_out_buf, num_blocks * sizeof(uint32_t), &ocl->ws.current_decomp_sizes_out_capacity, &err);
    if (err != CL_SUCCESS || !ocl->ws.decomp_sizes_out_buf) goto fail;

    if (ocl->ws.current_decomp_comp_off_capacity != prev_comp_off_cap ||
        ocl->ws.current_decomp_comp_size_capacity != prev_comp_size_cap) {
        ocl->cached_decomp_meta_valid = 0;
    }

    decomp_meta_hash = lz4_hash_u32_array(h_comp_off, num_blocks);
    decomp_meta_hash = decomp_meta_hash ^
        (lz4_hash_u32_array(comp_sizes, num_blocks) + 0x9e3779b97f4a7c15ULL +
         (decomp_meta_hash << 6) + (decomp_meta_hash >> 2));
    if (ocl->cached_decomp_meta_valid &&
        ocl->cached_decomp_meta_count == num_blocks &&
        ocl->cached_decomp_meta_hash == decomp_meta_hash) {
        need_upload_meta = 0;
    }

    {
        const uint64_t t_up0 = get_us();
        if (hybrid_write_buffer_auto(ocl->queue, ocl->ws.in_buf, comp_data, comp_total, use_standard_copy) != 0) {
            goto fail;
        }
        upload_acc += (get_us() - t_up0);
    }
    if (need_upload_meta) {
        const uint64_t t_up0 = get_us();
        if (hybrid_write_buffer_auto(ocl->queue, ocl->ws.decomp_comp_off_buf, h_comp_off, num_blocks * sizeof(uint32_t), use_standard_copy) != 0 ||
            hybrid_write_buffer_auto(ocl->queue, ocl->ws.decomp_comp_size_buf, comp_sizes, num_blocks * sizeof(uint32_t), use_standard_copy) != 0) {
            goto fail;
        }
        upload_acc += (get_us() - t_up0);
    }
    if (need_upload_meta) {
        ocl->cached_decomp_meta_hash = decomp_meta_hash;
        ocl->cached_decomp_meta_count = num_blocks;
        ocl->cached_decomp_meta_valid = 1;
    }

    totalBlocks = (uint32_t)num_blocks;
    err = CL_SUCCESS;
    err |= set_kernel_mem_arg_if_changed(ocl->kdec, 0, &ocl->cached_kdec_arg0, ocl->ws.in_buf);
    err |= set_kernel_mem_arg_if_changed(ocl->kdec, 1, &ocl->cached_kdec_arg1, ocl->ws.out_buf);
    err |= set_kernel_mem_arg_if_changed(ocl->kdec, 2, &ocl->cached_kdec_arg2, ocl->ws.decomp_comp_off_buf);
    err |= set_kernel_mem_arg_if_changed(ocl->kdec, 3, &ocl->cached_kdec_arg3, ocl->ws.decomp_comp_size_buf);
    err |= set_kernel_mem_arg_if_changed(ocl->kdec, 4, &ocl->cached_kdec_arg4, ocl->ws.decomp_sizes_out_buf);
    err |= clSetKernelArg(ocl->kdec, 5, sizeof(uint32_t), &block_size_u32);
    err |= clSetKernelArg(ocl->kdec, 6, sizeof(uint32_t), &totalBlocks);
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
        const uint64_t t_down0 = get_us();
        if (hybrid_read_buffer_auto(ocl->queue, ocl->ws.decomp_sizes_out_buf, out_sizes, num_blocks * sizeof(uint32_t), use_standard_copy) != 0) goto fail;
        download_acc += (get_us() - t_down0);
    }

    for (size_t i = 0; i < num_blocks; ++i) {
        if (out_sizes[i] == 0xFFFFFFFFU) goto fail;
    }

    {
        const uint64_t t_down0 = get_us();
        if (hybrid_read_buffer_auto(ocl->queue, ocl->ws.out_buf, out_full, num_blocks * block_size, use_standard_copy) != 0) goto fail;
        download_acc += (get_us() - t_down0);
    }
    if (upload_us) *upload_us = upload_acc;
    if (download_us) *download_us = download_acc;
    free(h_comp_off);
    return 0;

fail:
    free(h_comp_off);
    return -1;
}

static int hybrid_compress_memory(ocl_env_t* ocl,
                                  const unsigned char* input,
                                  size_t input_size,
                                  const hybrid_cfg_t* cfg,
                                  unsigned char** out_buf,
                                  size_t* out_size,
                                  hybrid_metrics_t* m,
                                  int skip_input_upload) {
    size_t num_blocks;
    size_t gpu_blocks;
    size_t cpu_blocks;
    int adaptive_collapsed_split = 0;
    size_t cpu_slot_size = 0;
    double effective_gpu_ratio;
    double sample_ratio_pct = 0.0;
    uint32_t* all_sizes = NULL;
    unsigned char* final_out = NULL;
    size_t final_sz = 0;
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

    if (!input || input_size == 0 || !cfg || !out_buf || !out_size) return -1;
    *out_buf = NULL;
    *out_size = 0;

    num_blocks = (input_size + cfg->block_size - 1) / cfg->block_size;
    effective_gpu_ratio = cfg->gpu_ratio;
    if (cfg->cpu_threads <= 0) {
        effective_gpu_ratio = 1.0;
    } else if (cfg->adaptive_split) {
        if (!ocl) return -1;
        effective_gpu_ratio = choose_adaptive_gpu_ratio(ocl, input, input_size, num_blocks, cfg, &sample_ratio_pct);
    }
    if (cfg->adaptive_split) {
        gpu_blocks = lz4_adaptive_adjust_gpu_blocks(num_blocks,
                                                    effective_gpu_ratio,
                                                    &adaptive_collapsed_split);
    } else {
        gpu_blocks = (size_t)((double)num_blocks * effective_gpu_ratio + 0.5);
    }
    if (gpu_blocks > num_blocks) gpu_blocks = num_blocks;
    if (gpu_blocks > 0 && !ocl) goto fail;
    cpu_blocks = num_blocks - gpu_blocks;

    if (cfg->verbose && cfg->adaptive_split) {
        fprintf(stderr,
                "Adaptive split: sample_ratio=%.2f%% requested_gpu_ratio=%.2f effective_gpu_ratio=%.2f gpu_blocks=%zu cpu_blocks=%zu\n",
                sample_ratio_pct,
                cfg->gpu_ratio,
                effective_gpu_ratio,
                gpu_blocks,
                cpu_blocks);
            if (adaptive_collapsed_split) {
                fprintf(stderr,
                    "Adaptive split: mixed block count too small, collapsed to %s-only\n",
                    gpu_blocks == 0 ? "CPU" : "GPU");
            }
    }

    all_sizes = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
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
        cpu_job.block_indices = NULL;
        cpu_job.block_index_base = gpu_blocks;
        cpu_job.num_blocks = cpu_blocks;
        cpu_job.num_threads = cfg->cpu_threads;
        cpu_job.acceleration = cfg->acceleration;
        cpu_job.out_slot_size = cpu_slot_size;
        cpu_job.out_slots = (unsigned char*)malloc(cpu_blocks * cpu_slot_size);
        cpu_job.out_sizes = (uint32_t*)malloc(cpu_blocks * sizeof(uint32_t));
        if (!cpu_job.out_slots || !cpu_job.out_sizes) goto fail;

    }

    {
        const uint64_t parallel_t0 = get_us();
        if (cpu_blocks > 0) {
            if (pthread_create(&cpu_thread, NULL, cpu_comp_top, &cpu_job) != 0) goto fail;
            cpu_thread_started = 1;
        }

        if (gpu_blocks > 0) {
            uint64_t gpu_upload_us = 0;
            uint64_t gpu_download_us = 0;
            if (cpu_blocks == 0) {
                gpu_input = (unsigned char*)(uintptr_t)input;
                gpu_input_size = input_size;
            } else {
                gpu_input = (unsigned char*)(uintptr_t)input;
                gpu_input_size = gpu_blocks * cfg->block_size;
            }
            if (gpu_compress_blocks(ocl,
                                    gpu_input,
                                    gpu_input_size,
                                    cfg->block_size,
                                    cfg->acceleration,
                                    cfg->local_size,
                                    NULL,
                                    0,
                                    &gpu_sizes,
                                    &gpu_offsets,
                                    &gpu_slots,
                                    &gpu_slot_size,
                                    m ? &m->gpu_kernel_us : &(uint64_t){0},
                                    &gpu_upload_us,
                                    &gpu_download_us,
                                    skip_input_upload) != 0) {
                goto fail;
            }
            if (m) {
                m->data_upload_us += gpu_upload_us;
                m->data_download_us += gpu_download_us;
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
        all_sizes[i] = gpu_sizes[i];
    }
    for (size_t i = 0; i < cpu_blocks; ++i) {
        all_sizes[gpu_blocks + i] = cpu_job.out_sizes[i];
    }

    final_sz = 16U + num_blocks * 4U;
    for (size_t i = 0; i < num_blocks; ++i) final_sz += (size_t)all_sizes[i];
    final_out = (unsigned char*)malloc(final_sz);
    if (!final_out) goto fail;

    {
        unsigned char* p = final_out;
        uint32_t magic = HYBRID_MAGIC;
        uint32_t nblk = (uint32_t)num_blocks;
        uint32_t bsz = (uint32_t)cfg->block_size;
        uint32_t gblk = (uint32_t)gpu_blocks;
        memcpy(p, &magic, 4); p += 4;
        memcpy(p, &nblk, 4); p += 4;
        memcpy(p, &bsz, 4); p += 4;
        memcpy(p, &gblk, 4); p += 4;
        memcpy(p, all_sizes, num_blocks * 4U); p += num_blocks * 4U;

        if (gpu_blocks > 0) {
            memcpy(p, gpu_slots, gpu_slot_size);
            p += gpu_slot_size;
        }
        for (size_t i = 0; i < cpu_blocks; ++i) {
            const size_t sz = (size_t)cpu_job.out_sizes[i];
            memcpy(p, cpu_job.out_slots + i * cpu_slot_size, sz);
            p += sz;
        }
    }

    *out_buf = final_out;
    *out_size = final_sz;
    free(all_sizes);
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
    const uint32_t* sizes = NULL;
    const unsigned char* data_ptr;
    size_t header_size;
    size_t payload_size = 0;

    unsigned char* out_full = NULL;
    const unsigned char* gpu_comp_data = NULL;
    uint32_t* gpu_out_sizes = NULL;
    const uint32_t* cpu_comp_sizes = NULL;
    uint32_t* cpu_comp_offsets = NULL;
    uint32_t* cpu_out_sizes = NULL;

    cpu_decomp_job_t cpu_job;
    pthread_t cpu_thread;
    int cpu_thread_started = 0;

    memset(&cpu_job, 0, sizeof(cpu_job));
    if (m) memset(m, 0, sizeof(*m));
    if (!comp || comp_size < 16 || !cfg || !out_buf || !out_size) return -1;
    *out_buf = NULL;
    *out_size = 0;

    memcpy(&magic, comp + 0, 4);
    memcpy(&num_blocks_u32, comp + 4, 4);
    memcpy(&block_size_u32, comp + 8, 4);
    memcpy(&gpu_blocks_u32, comp + 12, 4);
    if (magic != HYBRID_MAGIC) return -1;

    num_blocks = (size_t)num_blocks_u32;
    block_size = (size_t)block_size_u32;
    gpu_blocks = (size_t)gpu_blocks_u32;
    if (num_blocks == 0 || block_size == 0 || gpu_blocks > num_blocks) return -1;
    if (gpu_blocks > 0 && !ocl) return -1;

    header_size = 16U + num_blocks * 4U;
    if (header_size > comp_size) return -1;
    sizes = (const uint32_t*)(const void*)(comp + 16);
    data_ptr = comp + header_size;

    for (size_t i = 0; i < num_blocks; ++i) payload_size += (size_t)sizes[i];
    if (header_size + payload_size != comp_size) return -1;

    cpu_blocks = num_blocks - gpu_blocks;

    out_full = (unsigned char*)malloc(num_blocks * block_size);
    if (!out_full) goto fail;

    if (gpu_blocks > 0) {
        gpu_out_sizes = (uint32_t*)calloc(gpu_blocks, sizeof(uint32_t));
        if (!gpu_out_sizes) goto fail;
        gpu_comp_data = data_ptr;
    }
    if (cpu_blocks > 0) {
        size_t cpu_off = 0;
        cpu_out_sizes = (uint32_t*)calloc(cpu_blocks, sizeof(uint32_t));
        if (!cpu_out_sizes) goto fail;
        cpu_comp_sizes = sizes + gpu_blocks;
        cpu_comp_offsets = (uint32_t*)malloc(cpu_blocks * sizeof(uint32_t));
        if (!cpu_comp_offsets) goto fail;

        for (size_t i = 0; i < gpu_blocks; ++i) cpu_off += (size_t)sizes[i];
        for (size_t i = 0; i < cpu_blocks; ++i) {
            cpu_comp_offsets[i] = (uint32_t)cpu_off;
            cpu_off += (size_t)cpu_comp_sizes[i];
        }
    }

    {
        const uint64_t parallel_t0 = get_us();

        if (cpu_blocks > 0) {
            cpu_job.comp_data = data_ptr;
            cpu_job.comp_sizes = cpu_comp_sizes;
            cpu_job.comp_offsets = cpu_comp_offsets;
            cpu_job.block_size = block_size;
            cpu_job.block_indices = NULL;
            cpu_job.block_index_base = gpu_blocks;
            cpu_job.num_blocks = cpu_blocks;
            cpu_job.num_threads = cfg->cpu_threads;
            cpu_job.out_full = out_full;
            cpu_job.out_sizes = cpu_out_sizes;

            if (pthread_create(&cpu_thread, NULL, cpu_decomp_top, &cpu_job) != 0) goto fail;
            cpu_thread_started = 1;
        }

        if (gpu_blocks > 0) {
            uint64_t gpu_upload_us = 0;
            uint64_t gpu_download_us = 0;
            if (gpu_decompress_blocks(ocl,
                                      gpu_comp_data,
                                      sizes,
                                      gpu_blocks,
                                      block_size,
                                      cfg->local_size,
                                      out_full,
                                      gpu_out_sizes,
                                      m ? &m->gpu_kernel_us : &(uint64_t){0},
                                      &gpu_upload_us,
                                      &gpu_download_us) != 0) {
                goto fail;
            }
            if (m) {
                m->data_upload_us += gpu_upload_us;
                m->data_download_us += gpu_download_us;
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

    {
        size_t total_out = 0;
        for (size_t i = 0; i < gpu_blocks; ++i) total_out += (size_t)gpu_out_sizes[i];
        for (size_t i = 0; i < cpu_blocks; ++i) total_out += (size_t)cpu_out_sizes[i];
        *out_buf = out_full;
        *out_size = total_out;
    }

    free(gpu_out_sizes);
    free(cpu_comp_offsets);
    free(cpu_out_sizes);
    return 0;

fail:
    if (cpu_thread_started) pthread_join(cpu_thread, NULL);
    free(out_full);
    free(gpu_out_sizes);
    free(cpu_comp_offsets);
    free(cpu_out_sizes);
    return -1;
}

static void show_help(const char* prog) {
    fprintf(stderr, "LZ4 Hybrid CPU+GPU Tool\n");
    fprintf(stderr, "Usage: %s [options] <input_file|->\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -c                       Compress mode (default)\n");
    fprintf(stderr, "  -d, --decompress         Decompress mode\n");
    fprintf(stderr, "  -o, --output FILE        Output file (use '-' for stdout)\n");
    fprintf(stderr, "  -b, --block-size N       Block size in bytes (default: 64K)\n");
    fprintf(stderr, "  -a, --acceleration N     GPU acceleration (default: 1)\n");
    fprintf(stderr, "  -l, --local N            GPU local work-group size (default: 1)\n");
    fprintf(stderr, "  -T, --cpu-threads N      CPU thread count for CPU portion (default: auto = all cores)\n");
    fprintf(stderr, "  --adaptive               Enable adaptive per-file CPU/GPU split\n");
    fprintf(stderr, "  --sample-blocks N        Adaptive sample block count (default: 8)\n");
    fprintf(stderr, "  --gpu-ratio F            Fraction of blocks assigned to GPU (default: 0.5)\n");
    fprintf(stderr, "  --bench [N]              Benchmark mode with optional N seconds (default: 3)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Streaming:\n");
    fprintf(stderr, "  input '-'                Read input from stdin\n");
    fprintf(stderr, "  output '-'               Write output to stdout\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Environment knobs:\n");
    fprintf(stderr, "  FORCE_OPENCL_DEVICE=GPU|CPU|DEFAULT|ALL   OpenCL device selection\n");
    fprintf(stderr, "  LZ4_STANDARD_COPY=0|1                      Host-memory copy mode (0=map/zero-copy, 1=standard copy)\n");
    fprintf(stderr, "  -v, --verbose            Verbose output\n");
}

static void print_hybrid_time_row(FILE* msg, const char* label, uint64_t us, uint64_t total_us) {
    double ms;
    double pct;
    if (!msg || !label) return;
    ms = (double)us / 1000.0;
    pct = (total_us > 0) ? (100.0 * (double)us / (double)total_us) : 0.0;
    fprintf(msg, "  %-18s : %10.3f ms (%6.2f%%)\n", label, ms, pct);
}

static void print_hybrid_verbose_timing(FILE* msg, const hybrid_metrics_t* met) {
    uint64_t total_us;
    uint64_t parallel_us;
    uint64_t host_overhead_us;

    if (!msg || !met) return;

    parallel_us = met->parallel_us;
    if (parallel_us < met->cpu_kernel_us) parallel_us = met->cpu_kernel_us;
    if (parallel_us < met->gpu_kernel_us) parallel_us = met->gpu_kernel_us;

    total_us = met->total_us;
    if (total_us < parallel_us) total_us = parallel_us;
    host_overhead_us = (total_us > parallel_us) ? (total_us - parallel_us) : 0;

    print_hybrid_time_row(msg, "File Read", met->file_read_us, total_us);
    print_hybrid_time_row(msg, "OCI Setup", met->ocl_setup_us, total_us);
    print_hybrid_time_row(msg, "Data Upload", met->data_upload_us, total_us);
    print_hybrid_time_row(msg, "GPU Kernel", met->gpu_kernel_us, total_us);
    print_hybrid_time_row(msg, "CPU Kernel", met->cpu_kernel_us, total_us);
    print_hybrid_time_row(msg, "Parallel Span", parallel_us, total_us);
    print_hybrid_time_row(msg, "Data Download", met->data_download_us, total_us);
    print_hybrid_time_row(msg, "File Write", met->file_write_us, total_us);
    print_hybrid_time_row(msg, "Host Overhead", host_overhead_us, total_us);
    fprintf(msg, "  %-18s : %10.3f ms\n", "TOTAL INCLUSIVE", (double)total_us / 1000.0);
}

static int run_bench(const char* input_path, hybrid_cfg_t* cfg, double bench_seconds) {
    unsigned char* input = NULL;
    size_t input_size = 0;
    hybrid_cfg_t run_cfg;
    hybrid_cfg_t* active_cfg = cfg;
    ocl_env_t ocl;
    int ocl_ready = 0;
    int cpu_only_mode = 0;
    int skip_ocl_adaptive = 0;

    struct timespec ts0, ts1;
    const size_t bench_drop_iterations = 1;
    size_t total_successful_iterations = 0;
    size_t cap = 16, n = 0;
    double* comp_k = NULL;
    double* dec_k = NULL;
    double* ratio = NULL;
    double* comp_cpu_us_arr = NULL;
    double* comp_gpu_us_arr = NULL;
    double* dec_cpu_us_arr = NULL;
    double* dec_gpu_us_arr = NULL;
    int verify_ok = 1;
    size_t dec_repeat = 1;
    double adaptive_ratio_sum = 0.0;
    size_t adaptive_ratio_count = 0;
    double adaptive_ratio_min = 1.0;
    double adaptive_ratio_max = 0.0;

    if (bench_seconds <= 0.0) bench_seconds = 3.0;
    if (read_entire_file(input_path, &input, &input_size) != 0) {
        fprintf(stderr, "bench error: failed to read input\n");
        return 1;
    }

    dec_repeat = bench_dec_repeat_from_env(input_size);

    run_cfg = *cfg;
    skip_ocl_adaptive = adaptive_should_skip_ocl(cfg, input_size);
    if (skip_ocl_adaptive) {
        run_cfg.adaptive_split = 0;
        run_cfg.gpu_ratio = 0.0;
        active_cfg = &run_cfg;
        if (cfg->verbose) {
            fprintf(stderr,
                    "Adaptive: input_size=%zu < skip_ocl_threshold=%zu, forcing CPU-only path\n",
                    input_size,
                    adaptive_skip_ocl_threshold_bytes());
        }
    }

    cpu_only_mode = (!active_cfg->adaptive_split && active_cfg->gpu_ratio <= 0.0);
    if (!cpu_only_mode) {
        if (ocl_init(&ocl) != 0) {
            fprintf(stderr, "bench error: OpenCL init failed\n");
            free(input);
            return 1;
        }
        ocl_ready = 1;
    }

    comp_k = (double*)malloc(cap * sizeof(double));
    dec_k = (double*)malloc(cap * sizeof(double));
    ratio = (double*)malloc(cap * sizeof(double));
    comp_cpu_us_arr = (double*)malloc(cap * sizeof(double));
    comp_gpu_us_arr = (double*)malloc(cap * sizeof(double));
    dec_cpu_us_arr = (double*)malloc(cap * sizeof(double));
    dec_gpu_us_arr = (double*)malloc(cap * sizeof(double));
    if (!comp_k || !dec_k || !ratio || !comp_cpu_us_arr || !comp_gpu_us_arr || !dec_cpu_us_arr || !dec_gpu_us_arr) verify_ok = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts0);

    while (verify_ok) {
        unsigned char* comp_buf = NULL;
        size_t comp_size = 0;
        hybrid_metrics_t cm, dm;
        double in_mb;
        uint64_t best_dec_kernel_us = 0;
        uint64_t best_dec_cpu_us = 0;
        uint64_t best_dec_gpu_us = 0;
        int have_dec_sample = 0;
        {
            int skip_upload = (active_cfg->gpu_ratio >= 1.0 && total_successful_iterations > 0) ? 1 : 0;
            if (hybrid_compress_memory(ocl_ready ? &ocl : NULL, input, input_size, active_cfg, &comp_buf, &comp_size, &cm, skip_upload) != 0) {
                verify_ok = 0;
                break;
            }
        }

        if (active_cfg->adaptive_split && comp_buf && comp_size >= 16U) {
            uint32_t magic = 0;
            uint32_t nblk = 0;
            uint32_t gblk = 0;
            memcpy(&magic, comp_buf + 0, sizeof(uint32_t));
            memcpy(&nblk, comp_buf + 4, sizeof(uint32_t));
            memcpy(&gblk, comp_buf + 12, sizeof(uint32_t));
            if (magic == HYBRID_MAGIC && nblk > 0 && gblk <= nblk) {
                double gr = (double)gblk / (double)nblk;
                adaptive_ratio_sum += gr;
                adaptive_ratio_count += 1;
                if (gr < adaptive_ratio_min) adaptive_ratio_min = gr;
                if (gr > adaptive_ratio_max) adaptive_ratio_max = gr;
            }
        }

        for (size_t rep = 0; rep < dec_repeat; ++rep) {
            unsigned char* dec_buf = NULL;
            size_t dec_size = 0;
            uint64_t dec_kernel_us;

            if (hybrid_decompress_memory(ocl_ready ? &ocl : NULL, comp_buf, comp_size, active_cfg, &dec_buf, &dec_size, &dm) != 0) {
                free(dec_buf);
                verify_ok = 0;
                break;
            }

            if (rep == 0 && (dec_size != input_size || memcmp(dec_buf, input, input_size) != 0)) {
                free(dec_buf);
                verify_ok = 0;
                break;
            }

            dec_kernel_us = dm.gpu_kernel_us;
            if (dm.cpu_kernel_us > dec_kernel_us) dec_kernel_us = dm.cpu_kernel_us;
            if (!have_dec_sample || dec_kernel_us > best_dec_kernel_us) {
                best_dec_kernel_us = dec_kernel_us;
                best_dec_cpu_us = dm.cpu_kernel_us;
                best_dec_gpu_us = dm.gpu_kernel_us;
            }
            have_dec_sample = 1;

            free(dec_buf);
        }

        if (!verify_ok || !have_dec_sample) {
            free(comp_buf);
            verify_ok = 0;
            break;
        }

        total_successful_iterations += 1;
        if (total_successful_iterations <= bench_drop_iterations) {
            free(comp_buf);
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            continue;
        }

        if (n == cap) {
            size_t ncap = cap * 2;
            double* p;
            p = (double*)realloc(comp_k, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); break; } comp_k = p;
            p = (double*)realloc(dec_k, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); break; } dec_k = p;
            p = (double*)realloc(ratio, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); break; } ratio = p;
            p = (double*)realloc(comp_cpu_us_arr, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); break; } comp_cpu_us_arr = p;
            p = (double*)realloc(comp_gpu_us_arr, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); break; } comp_gpu_us_arr = p;
            p = (double*)realloc(dec_cpu_us_arr, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); break; } dec_cpu_us_arr = p;
            p = (double*)realloc(dec_gpu_us_arr, ncap * sizeof(double)); if (!p) { verify_ok = 0; free(comp_buf); break; } dec_gpu_us_arr = p;
            cap = ncap;
        }

        in_mb = (double)input_size / (1024.0 * 1024.0);
        {
            uint64_t comp_kernel_us = cm.gpu_kernel_us;
            if (cm.cpu_kernel_us > comp_kernel_us) comp_kernel_us = cm.cpu_kernel_us;
            comp_k[n] = (comp_kernel_us > 0) ? (in_mb * 1000000.0 / (double)comp_kernel_us) : 0.0;
        }
        dec_k[n] = (best_dec_kernel_us > 0) ? (in_mb * 1000000.0 / (double)best_dec_kernel_us) : 0.0;
        ratio[n] = (input_size > 0) ? (100.0 * (double)comp_size / (double)input_size) : 0.0;
        comp_cpu_us_arr[n] = (double)cm.cpu_kernel_us;
        comp_gpu_us_arr[n] = (double)cm.gpu_kernel_us;
        dec_cpu_us_arr[n] = (double)best_dec_cpu_us;
        dec_gpu_us_arr[n] = (double)best_dec_gpu_us;
        ++n;

        free(comp_buf);

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (elapsed_sec(&ts0, &ts1) >= bench_seconds && n > 0) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    if (n > 0) {
           printf("Bench Compress : kernel_tp=%.2f MB/s ratio=%.2f%%\n",
               median_double(comp_k, n), median_double(ratio, n));
           printf("Bench Decompress : kernel_tp=%.2f MB/s verify=%s\n",
               median_double(dec_k, n), verify_ok ? "OK" : "FAIL");
           printf("Bench Detail : comp_cpu_us=%.0f comp_gpu_us=%.0f dec_cpu_us=%.0f dec_gpu_us=%.0f\n",
               median_double(comp_cpu_us_arr, n),
               median_double(comp_gpu_us_arr, n),
               median_double(dec_cpu_us_arr, n),
               median_double(dec_gpu_us_arr, n));
        if (active_cfg->adaptive_split && adaptive_ratio_count > 0) {
            double adaptive_ratio_mean = adaptive_ratio_sum / (double)adaptive_ratio_count;
            printf("Bench Adaptive : gpu_ratio_mean=%.4f min=%.4f max=%.4f samples=%zu\n",
                   adaptive_ratio_mean,
                   adaptive_ratio_min,
                   adaptive_ratio_max,
                   adaptive_ratio_count);
            printf("Bench Adaptive : gpu_ratio=%.4f objective=perf_energy_ratio min=%.4f max=%.4f samples=%zu\n",
                   adaptive_ratio_mean,
                   adaptive_ratio_min,
                   adaptive_ratio_max,
                   adaptive_ratio_count);
        }
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_k);
    free(dec_k);
    free(ratio);
    free(comp_cpu_us_arr);
    free(comp_gpu_us_arr);
    free(dec_cpu_us_arr);
    free(dec_gpu_us_arr);
    free(input);
    if (ocl_ready) ocl_free(&ocl);
    return verify_ok ? 0 : 1;
}

int main(int argc, char** argv) {
    int mode_decompress = 0;
    int bench_mode = 0;
    double bench_seconds = 3.0;
    const char* input_path = NULL;
    char output_path[1024] = {0};
    int output_explicit = 0;
    int output_to_stdout = 0;

    hybrid_cfg_t cfg;
    cfg.block_size = 64 * 1024;
    cfg.acceleration = 1;
    cfg.local_size = 1;
    cfg.cpu_threads = 0;  /* 0 = auto-detect at runtime */
    cfg.gpu_ratio = 0.5;
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
        } else if (strcmp(argv[i], "--split-prefix") == 0) {
            /* Deprecated: accepted for backward compatibility. */
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-' && is_number_string(argv[i + 1])) {
                bench_seconds = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            cfg.verbose = 1;
        } else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0) {
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
    if (cfg.cpu_threads <= 0) {
        long ncpu = get_online_cpu_count();
        cfg.cpu_threads = (ncpu > 0) ? (int)ncpu : 4;
    }
    if (cfg.local_size <= 0) cfg.local_size = 1;
    if (cfg.gpu_ratio < 0.0) cfg.gpu_ratio = 0.0;
    if (cfg.gpu_ratio > 1.0) cfg.gpu_ratio = 1.0;

    if (bench_mode) {
        if (mode_decompress) {
            fprintf(stderr, "Error: --bench requires compress mode input\n");
            return 1;
        }
        if (path_is_dash(input_path)) {
            fprintf(stderr, "Error: --bench does not support stdin input ('-')\n");
            return 1;
        }
        return run_bench(input_path, &cfg, bench_seconds);
    }

    if (!output_explicit) {
        if (path_is_dash(input_path)) {
            strncpy(output_path, "-", sizeof(output_path) - 1);
            output_explicit = 1;
        } else if (!mode_decompress) {
            snprintf(output_path, sizeof(output_path), "%s.lz4", input_path);
        } else {
            snprintf(output_path, sizeof(output_path), "%s.dec", input_path);
        }
    }
    output_to_stdout = path_is_dash(output_path);

    {
        int rc = 1;
        ocl_env_t ocl;
        hybrid_cfg_t run_cfg;
        hybrid_cfg_t* active_cfg = &cfg;
        int ocl_ready = 0;
        int skip_ocl_for_compress = 0;
        int skip_ocl_for_decompress = 0;
        int skip_ocl_adaptive = 0;
        unsigned char* in_buf = NULL;
        size_t in_sz = 0;
        unsigned char* out_buf = NULL;
        size_t out_sz = 0;
        hybrid_metrics_t met;
        uint64_t total_start_us = 0;
        uint64_t file_read_us = 0;
        uint64_t ocl_setup_us = 0;
        uint64_t file_write_us = 0;
        uint64_t total_end_us = 0;
        uint64_t total0 = 0;

        {
            uint64_t t_read0 = get_us();
            if (read_entire_file(input_path, &in_buf, &in_sz) != 0) {
                fprintf(stderr, "Error: failed to read input file %s (%s)\n", input_path, strerror(errno));
                return 1;
            }
            file_read_us = get_us() - t_read0;
            total_start_us = t_read0;
        }
        total0 = total_start_us;

        run_cfg = cfg;
        skip_ocl_adaptive = (!mode_decompress && adaptive_should_skip_ocl(&cfg, in_sz));
        if (skip_ocl_adaptive) {
            run_cfg.adaptive_split = 0;
            run_cfg.gpu_ratio = 0.0;
            active_cfg = &run_cfg;
            if (cfg.verbose) {
                fprintf(stderr,
                        "Adaptive: input_size=%zu < skip_ocl_threshold=%zu, forcing CPU-only path\n",
                        in_sz,
                        adaptive_skip_ocl_threshold_bytes());
            }
        }

        skip_ocl_for_compress = (!mode_decompress && !active_cfg->adaptive_split && active_cfg->gpu_ratio <= 0.0);
        if (mode_decompress && in_sz >= 16U) {
            uint32_t magic = 0;
            uint32_t gpu_blocks_u32 = 0;
            memcpy(&magic, in_buf + 0, 4);
            memcpy(&gpu_blocks_u32, in_buf + 12, 4);
            if (magic == HYBRID_MAGIC && gpu_blocks_u32 == 0) {
                skip_ocl_for_decompress = 1;
            }
        }

        if (!(skip_ocl_for_compress || skip_ocl_for_decompress)) {
            const uint64_t t_ocl0 = get_us();
            if (ocl_init(&ocl) != 0) {
                fprintf(stderr, "Error: OpenCL init failed\n");
                free(in_buf);
                return 1;
            }
            ocl_setup_us = get_us() - t_ocl0;
            ocl_ready = 1;
        }

        if (!mode_decompress) {
            const uint64_t t0 = get_us();
            if (hybrid_compress_memory(ocl_ready ? &ocl : NULL, in_buf, in_sz, active_cfg, &out_buf, &out_sz, &met, 0) != 0) {
                fprintf(stderr, "Error: compression failed\n");
                goto done;
            }
            met.total_us = get_us() - t0;
            {
                const uint64_t t_w0 = get_us();
            if (write_entire_file(output_path, out_buf, out_sz) != 0) {
                fprintf(stderr, "Error: failed writing output file %s\n", output_path);
                goto done;
            }
                file_write_us = get_us() - t_w0;
            }
            met.file_read_us = file_read_us;
            met.ocl_setup_us = ocl_setup_us;
            met.file_write_us = file_write_us;
            total_end_us = get_us();
            if (total_end_us > total_start_us) {
                met.total_us = total_end_us - total_start_us;
            }
            if (cfg.verbose) {
                FILE* msg = output_to_stdout ? stderr : stdout;
                double in_mb = (double)in_sz / (1024.0 * 1024.0);
                double k_tp = (met.parallel_us > 0) ? (in_mb * 1000000.0 / (double)met.parallel_us) : 0.0;
                fprintf(msg,
                        "Hybrid Compress : kernel_tp=%.2f MB/s ratio=%.2f%%\n",
                        k_tp,
                        (in_sz > 0) ? (100.0 * (double)out_sz / (double)in_sz) : 0.0);
                print_hybrid_verbose_timing(msg, &met);
            } else {
                FILE* msg = output_to_stdout ? stderr : stdout;
                fprintf(msg, "%s : %zu -> %zu in %.2f ms\n", input_path, in_sz, out_sz, (get_us() - total0) / 1000.0);
            }
        } else {
            const uint64_t t0 = get_us();
            if (hybrid_decompress_memory(ocl_ready ? &ocl : NULL, in_buf, in_sz, active_cfg, &out_buf, &out_sz, &met) != 0) {
                fprintf(stderr, "Error: decompression failed\n");
                goto done;
            }
            met.total_us = get_us() - t0;
            {
                const uint64_t t_w0 = get_us();
            if (write_entire_file(output_path, out_buf, out_sz) != 0) {
                fprintf(stderr, "Error: failed writing output file %s\n", output_path);
                goto done;
            }
                file_write_us = get_us() - t_w0;
            }
            met.file_read_us = file_read_us;
            met.ocl_setup_us = ocl_setup_us;
            met.file_write_us = file_write_us;
            total_end_us = get_us();
            if (total_end_us > total_start_us) {
                met.total_us = total_end_us - total_start_us;
            }
            if (cfg.verbose) {
                FILE* msg = output_to_stdout ? stderr : stdout;
                double out_mb = (double)out_sz / (1024.0 * 1024.0);
                double k_tp = (met.parallel_us > 0) ? (out_mb * 1000000.0 / (double)met.parallel_us) : 0.0;
                fprintf(msg, "Hybrid Decompress : kernel_tp=%.2f MB/s\n", k_tp);
                print_hybrid_verbose_timing(msg, &met);
            } else {
                FILE* msg = output_to_stdout ? stderr : stdout;
                fprintf(msg, "%s : %zu -> %zu in %.2f ms\n", input_path, in_sz, out_sz, (get_us() - total0) / 1000.0);
            }
        }

        rc = 0;
done:
        free(in_buf);
        free(out_buf);
    if (ocl_ready) ocl_free(&ocl);
        return rc;
    }
}
