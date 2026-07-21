#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(_WIN32)
#include <process.h>
#include <windows.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "lz4.h"
#include "threadpool.h"
#include "timefn.h"

#define LZ4TP_MAGIC "LZ4TP1\0\0"
#define DEFAULT_BLOCK_SIZE (64U * 1024U)
#define DEFAULT_HASH_LOG 14
#define DEFAULT_WORKERS 4
#define MAX_BLOCK_SIZE (4U * 1024U * 1024U)
#define TARGET_BATCH_BYTES (16U * 1024U * 1024U)

typedef enum {
    MODE_NONE = 0,
    MODE_COMPRESS,
    MODE_DECOMPRESS
} operation_mode;

typedef struct {
    const unsigned char* input;
    unsigned char* output;
    int input_size;
    int output_capacity;
    int result;
} block_job;

typedef struct {
    unsigned workers;
    unsigned active_workers;
    uint32_t block_size;
    uint32_t hash_log;
    size_t chunk_blocks;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t kernel_us;
    uint64_t no_ocl_us;
    uint64_t total_us;
    uint64_t runtime_setup_us;
} operation_metric;

static int read_exact(FILE* file, void* data, size_t size) {
    return size == 0 || (file && fread(data, 1, size, file) == size);
}

static int write_exact(FILE* file, const void* data, size_t size) {
    return size == 0 || (file && fwrite(data, 1, size, file) == size);
}

static uint64_t elapsed_us(TIME_t start, TIME_t end) {
    Duration_ns elapsed = TIME_span_ns(start, end);
    return (uint64_t)((elapsed + 999U) / 1000U);
}

static int size_mul(size_t left, size_t right, size_t* result) {
    if (!result || (left != 0 && right > SIZE_MAX / left)) return -1;
    *result = left * right;
    return 0;
}

static int size_add(size_t left, size_t right, size_t* result) {
    if (!result || right > SIZE_MAX - left) return -1;
    *result = left + right;
    return 0;
}

static int parse_unsigned(const char* text, unsigned* value) {
    char* end = NULL;
    unsigned long parsed;
    if (!text || !*text || !value) return -1;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed > UINT_MAX) return -1;
    *value = (unsigned)parsed;
    return 0;
}

static int worker_count_valid(unsigned workers) {
    return workers == 1 || workers == 2 || workers == 4 || workers == 8;
}

static int make_temp_path(const char* output_path, char* temp_path, size_t capacity) {
    int length;
    if (!output_path || !*output_path || !temp_path || capacity == 0) return -1;
    length = snprintf(temp_path, capacity, "%s.tmp.%lu", output_path,
                      (unsigned long)getpid());
    return length >= 0 && (size_t)length < capacity ? 0 : -1;
}

static int replace_path(const char* temp_path, const char* output_path) {
#if defined(_WIN32)
    return MoveFileExA(temp_path, output_path,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(temp_path, output_path);
#endif
}

static int finalize_output(FILE** file, const char* temp_path, const char* output_path) {
    if (!file || !*file || fflush(*file) != 0 || fclose(*file) != 0) {
        if (file) *file = NULL;
        return -1;
    }
    *file = NULL;
    return replace_path(temp_path, output_path);
}

static int resolve_target_path(const char* path, char* resolved, size_t capacity) {
    if (!path || !*path || !resolved || capacity == 0) return -1;
#if defined(_WIN32)
    return _fullpath(resolved, path, capacity) ? 0 : -1;
#else
    {
        char copy[PATH_MAX];
        char parent[PATH_MAX];
        char resolved_parent[PATH_MAX];
        char* separator;
        const char* name;
        int length;
        if (strlen(path) >= sizeof(copy)) return -1;
        memcpy(copy, path, strlen(path) + 1);
        separator = strrchr(copy, '/');
        if (!separator) {
            memcpy(parent, ".", 2);
            name = copy;
        } else {
            name = separator + 1;
            if (separator == copy) {
                memcpy(parent, "/", 2);
            } else {
                *separator = '\0';
                if (strlen(copy) >= sizeof(parent)) return -1;
                memcpy(parent, copy, strlen(copy) + 1);
            }
        }
        if (!*name || !realpath(parent, resolved_parent)) return -1;
        length = snprintf(resolved, capacity, "%s/%s", resolved_parent, name);
        return length >= 0 && (size_t)length < capacity ? 0 : -1;
    }
#endif
}

static int paths_identify_same_file(const char* left, const char* right) {
#if !defined(_WIN32)
    struct stat left_stat, right_stat;
#endif
    char left_full[PATH_MAX], right_full[PATH_MAX];
    if (!left || !right) return 0;
    if (strcmp(left, right) == 0) return 1;
    if (resolve_target_path(left, left_full, sizeof(left_full)) == 0 &&
        resolve_target_path(right, right_full, sizeof(right_full)) == 0) {
#if defined(_WIN32)
        if (_stricmp(left_full, right_full) == 0) return 1;
#else
        if (strcmp(left_full, right_full) == 0) return 1;
#endif
    }
#if defined(_WIN32)
    {
        HANDLE left_handle = CreateFileA(
            left, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        HANDLE right_handle = CreateFileA(
            right, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        BY_HANDLE_FILE_INFORMATION left_info, right_info;
        int same = 0;
        if (left_handle != INVALID_HANDLE_VALUE && right_handle != INVALID_HANDLE_VALUE &&
            GetFileInformationByHandle(left_handle, &left_info) &&
            GetFileInformationByHandle(right_handle, &right_info)) {
            same = left_info.dwVolumeSerialNumber == right_info.dwVolumeSerialNumber &&
                   left_info.nFileIndexHigh == right_info.nFileIndexHigh &&
                   left_info.nFileIndexLow == right_info.nFileIndexLow;
        }
        if (left_handle != INVALID_HANDLE_VALUE) CloseHandle(left_handle);
        if (right_handle != INVALID_HANDLE_VALUE) CloseHandle(right_handle);
        return same;
    }
#else
    if (stat(left, &left_stat) != 0 || stat(right, &right_stat) != 0) return 0;
    return left_stat.st_dev == right_stat.st_dev && left_stat.st_ino == right_stat.st_ino;
#endif
}

static int path_exists(const char* path) {
    struct stat path_stat;
    return path && stat(path, &path_stat) == 0;
}

static void compress_block(void* opaque) {
    block_job* job = (block_job*)opaque;
    job->result = LZ4_compress_fast((const char*)job->input, (char*)job->output,
                                    job->input_size, job->output_capacity, 1);
}

static void decompress_block(void* opaque) {
    block_job* job = (block_job*)opaque;
    job->result = LZ4_decompress_safe((const char*)job->input, (char*)job->output,
                                      job->input_size, job->output_capacity);
}

static size_t choose_batch_blocks(unsigned workers, uint32_t block_size) {
    size_t desired = (size_t)workers * 16U;
    size_t memory_limited = TARGET_BATCH_BYTES / block_size;
    if (memory_limited < workers) memory_limited = workers;
    return desired < memory_limited ? desired : memory_limited;
}

static int submit_jobs(TPool* pool, block_job* jobs, size_t count,
                       void (*function)(void*), uint64_t* elapsed) {
    size_t index;
    TIME_t start = TIME_getTime();
    for (index = 0; index < count; ++index) {
        jobs[index].result = 0;
        TPool_submitJob(pool, function, &jobs[index]);
    }
    TPool_jobsCompleted(pool);
    *elapsed += elapsed_us(start, TIME_getTime());
    for (index = 0; index < count; ++index) {
        if (jobs[index].result <= 0) return -1;
    }
    return 0;
}

static int write_metric(const char* path, const char* operation,
                        const operation_metric* metric) {
    char temp_path[PATH_MAX];
    FILE* file = NULL;
    double ratio_pct;
    int ok;
    if (!path) return 0;
    if (make_temp_path(path, temp_path, sizeof(temp_path)) != 0) return -1;
    remove(temp_path);
    file = fopen(temp_path, "wb");
    if (!file) return -1;
    ratio_pct = strcmp(operation, "compress") == 0
              ? (metric->input_bytes ? 100.0 * (double)metric->output_bytes /
                                       (double)metric->input_bytes : 0.0)
              : (metric->output_bytes ? 100.0 * (double)metric->input_bytes /
                                        (double)metric->output_bytes : 0.0);
    ok = fprintf(
        file,
        "{\n"
        "  \"schema\": \"heterolz.operation-metric.v1\",\n"
        "  \"engine\": \"native_lz4\",\n"
        "  \"operation\": \"%s\",\n"
        "  \"input_bytes\": %llu,\n"
        "  \"output_bytes\": %llu,\n"
        "  \"n\": 1,\n"
        "  \"block_size\": %u,\n"
        "  \"hash_log\": %u,\n"
        "  \"hash_log_role\": \"container_metadata\",\n"
        "  \"chunk_blocks\": %zu,\n"
        "  \"workers\": %u,\n"
        "  \"active_workers\": %u,\n"
        "  \"acceleration\": 1,\n"
        "  \"kernel_us\": %llu,\n"
        "  \"no_ocl_us\": %llu,\n"
        "  \"total_us\": %llu,\n"
        "  \"ocl_setup_us\": 0,\n"
        "  \"runtime_setup_us\": %llu,\n"
        "  \"ratio_pct\": %.9f\n"
        "}\n",
        operation,
        (unsigned long long)metric->input_bytes,
        (unsigned long long)metric->output_bytes,
        metric->block_size, metric->hash_log, metric->chunk_blocks,
        metric->workers, metric->active_workers,
        (unsigned long long)metric->kernel_us,
        (unsigned long long)metric->no_ocl_us,
        (unsigned long long)metric->total_us,
        (unsigned long long)metric->runtime_setup_us,
        ratio_pct) >= 0;
    if (!ok || finalize_output(&file, temp_path, path) != 0) {
        if (file) fclose(file);
        remove(temp_path);
        return -1;
    }
    return 0;
}

static int compress_file(const char* input_path, const char* output_path,
                         const char* metrics_path, unsigned workers,
                         uint32_t block_size, uint32_t hash_log) {
    TIME_t total_start = TIME_getTime();
    TIME_t setup_start, setup_end, total_end;
    struct stat input_stat;
    FILE* input_file = NULL;
    FILE* output_file = NULL;
    TPool* pool = NULL;
    uint32_t* compressed_sizes = NULL;
    unsigned char* input = NULL;
    unsigned char* compressed = NULL;
    block_job* jobs = NULL;
    char temp_path[PATH_MAX] = {0};
    int temp_active = 0;
    int result = 1;
    size_t block_count, batch_blocks, input_capacity, output_capacity;
    size_t compressed_bound, payload_total = 0, base;
    operation_metric metric = {0};

    if (stat(input_path, &input_stat) != 0 || input_stat.st_size < 0 ||
        (uint64_t)input_stat.st_size > (uint64_t)SIZE_MAX ||
        make_temp_path(output_path, temp_path, sizeof(temp_path)) != 0) {
        fprintf(stderr, "native compress: invalid input or output path\n");
        return 1;
    }
    block_count = (size_t)input_stat.st_size / block_size +
                  ((size_t)input_stat.st_size % block_size != 0);
    if (block_count > UINT32_MAX) {
        fprintf(stderr, "native compress: too many blocks\n");
        return 1;
    }
    compressed_bound = (size_t)LZ4_compressBound((int)block_size);
    batch_blocks = choose_batch_blocks(workers, block_size);
    if (compressed_bound == 0 ||
        size_mul(batch_blocks, block_size, &input_capacity) != 0 ||
        size_mul(batch_blocks, compressed_bound, &output_capacity) != 0 ||
        block_count > SIZE_MAX / sizeof(*compressed_sizes)) {
        fprintf(stderr, "native compress: dimensions overflow\n");
        return 1;
    }

    setup_start = TIME_getTime();
    pool = TPool_create((int)workers, (int)batch_blocks);
    setup_end = TIME_getTime();
    compressed_sizes = (uint32_t*)calloc(block_count ? block_count : 1,
                                         sizeof(*compressed_sizes));
    input = (unsigned char*)malloc(input_capacity ? input_capacity : 1);
    compressed = (unsigned char*)malloc(output_capacity ? output_capacity : 1);
    jobs = (block_job*)calloc(batch_blocks, sizeof(*jobs));
    input_file = fopen(input_path, "rb");
    remove(temp_path);
    output_file = fopen(temp_path, "w+b");
    temp_active = output_file != NULL;
    if (!pool || !compressed_sizes || !input || !compressed || !jobs ||
        !input_file || !output_file) {
        fprintf(stderr, "native compress: setup failed\n");
        goto done;
    }

    {
        uint32_t n = 1, block_count_u32 = (uint32_t)block_count;
        uint64_t original_size = (uint64_t)input_stat.st_size;
        if (!write_exact(output_file, LZ4TP_MAGIC, 8) ||
            !write_exact(output_file, &n, 4) ||
            !write_exact(output_file, &hash_log, 4) ||
            !write_exact(output_file, &block_size, 4) ||
            !write_exact(output_file, &block_count_u32, 4) ||
            !write_exact(output_file, &original_size, 8) ||
            !write_exact(output_file, compressed_sizes,
                         block_count * sizeof(*compressed_sizes))) {
            fprintf(stderr, "native compress: header write failed\n");
            goto done;
        }
    }

    for (base = 0; base < block_count; base += batch_blocks) {
        size_t count = block_count - base;
        size_t consumed = base * (size_t)block_size;
        size_t remaining = (size_t)input_stat.st_size - consumed;
        size_t input_bytes;
        size_t index;
        if (count > batch_blocks) count = batch_blocks;
        input_bytes = remaining < count * (size_t)block_size
                    ? remaining : count * (size_t)block_size;
        if (!read_exact(input_file, input, input_bytes)) {
            fprintf(stderr, "native compress: input read failed\n");
            goto done;
        }
        for (index = 0; index < count; ++index) {
            size_t offset = index * (size_t)block_size;
            size_t length = input_bytes - offset;
            if (length > block_size) length = block_size;
            jobs[index].input = input + offset;
            jobs[index].output = compressed + index * compressed_bound;
            jobs[index].input_size = (int)length;
            jobs[index].output_capacity = (int)compressed_bound;
        }
        if (submit_jobs(pool, jobs, count, compress_block, &metric.kernel_us) != 0) {
            fprintf(stderr, "native compress: block compression failed\n");
            goto done;
        }
        for (index = 0; index < count; ++index) {
            size_t compressed_size = (size_t)jobs[index].result;
            if (compressed_size > UINT32_MAX ||
                !write_exact(output_file, jobs[index].output, compressed_size) ||
                size_add(payload_total, compressed_size, &payload_total) != 0) {
                fprintf(stderr, "native compress: payload write failed\n");
                goto done;
            }
            compressed_sizes[base + index] = (uint32_t)compressed_size;
        }
    }
    if (fgetc(input_file) != EOF || ferror(input_file) ||
        fseek(output_file, 32L, SEEK_SET) != 0 ||
        !write_exact(output_file, compressed_sizes,
                     block_count * sizeof(*compressed_sizes)) ||
        finalize_output(&output_file, temp_path, output_path) != 0) {
        fprintf(stderr, "native compress: input changed or finalization failed\n");
        goto done;
    }
    temp_active = 0;
    total_end = TIME_getTime();
    metric.workers = workers;
    metric.active_workers = block_count < workers ? (unsigned)block_count : workers;
    metric.block_size = block_size;
    metric.hash_log = hash_log;
    metric.chunk_blocks = block_count < batch_blocks ? block_count : batch_blocks;
    metric.input_bytes = (uint64_t)input_stat.st_size;
    metric.output_bytes = 32U + (uint64_t)block_count * sizeof(*compressed_sizes) + payload_total;
    metric.runtime_setup_us = elapsed_us(setup_start, setup_end);
    metric.total_us = elapsed_us(total_start, total_end);
    metric.no_ocl_us = metric.total_us > metric.runtime_setup_us
                     ? metric.total_us - metric.runtime_setup_us : metric.total_us;
    if (metric.no_ocl_us < metric.kernel_us) metric.no_ocl_us = metric.kernel_us;
    if (metric.total_us < metric.no_ocl_us) metric.total_us = metric.no_ocl_us;
    if (write_metric(metrics_path, "compress", &metric) != 0) {
        fprintf(stderr, "native compress: metrics write failed\n");
        remove(output_path);
        goto done;
    }
    fprintf(stderr,
            "[native-lz4-compress] %s -> %s : %llu -> %llu, blocks=%zu workers=%u\n",
            input_path, output_path, (unsigned long long)metric.input_bytes,
            (unsigned long long)metric.output_bytes, block_count, workers);
    result = 0;

done:
    if (input_file) fclose(input_file);
    if (output_file) fclose(output_file);
    if (temp_active) remove(temp_path);
    if (pool) TPool_free(pool);
    free(compressed_sizes);
    free(input);
    free(compressed);
    free(jobs);
    return result;
}

static int decompress_file(const char* input_path, const char* output_path,
                           const char* metrics_path, unsigned workers,
                           int requested_block_size, int requested_hash_log) {
    TIME_t total_start = TIME_getTime();
    TIME_t setup_start, setup_end, total_end;
    struct stat frame_stat;
    FILE* input_file = NULL;
    FILE* output_file = NULL;
    TPool* pool = NULL;
    uint32_t* compressed_sizes = NULL;
    unsigned char* compressed = NULL;
    unsigned char* output = NULL;
    block_job* jobs = NULL;
    char temp_path[PATH_MAX] = {0};
    char magic[8];
    uint32_t n, hash_log, block_size, block_count_u32;
    uint64_t original_size;
    size_t block_count, compressed_bound, batch_blocks;
    size_t compressed_capacity, output_capacity, payload_total = 0, base;
    int temp_active = 0;
    int result = 1;
    operation_metric metric = {0};

    if (stat(input_path, &frame_stat) != 0 || frame_stat.st_size < 32 ||
        (uint64_t)frame_stat.st_size > (uint64_t)SIZE_MAX ||
        make_temp_path(output_path, temp_path, sizeof(temp_path)) != 0) {
        fprintf(stderr, "native decompress: invalid input or output path\n");
        return 1;
    }
    input_file = fopen(input_path, "rb");
    if (!input_file || !read_exact(input_file, magic, 8) ||
        memcmp(magic, LZ4TP_MAGIC, 8) != 0 ||
        !read_exact(input_file, &n, 4) || !read_exact(input_file, &hash_log, 4) ||
        !read_exact(input_file, &block_size, 4) ||
        !read_exact(input_file, &block_count_u32, 4) ||
        !read_exact(input_file, &original_size, 8)) {
        fprintf(stderr, "native decompress: invalid LZ4TP1 header\n");
        goto done;
    }
    block_count = block_count_u32;
    if (n != 1 || hash_log < 11 || hash_log > 15 || block_size == 0 ||
        block_size > MAX_BLOCK_SIZE || original_size > SIZE_MAX ||
        block_count != (size_t)(original_size / block_size +
                                (original_size % block_size != 0)) ||
        (requested_block_size > 0 && (uint32_t)requested_block_size != block_size) ||
        (requested_hash_log > 0 && (uint32_t)requested_hash_log != hash_log) ||
        block_count > SIZE_MAX / sizeof(*compressed_sizes)) {
        fprintf(stderr, "native decompress: unsupported or inconsistent frame\n");
        goto done;
    }
    compressed_sizes = (uint32_t*)malloc((block_count ? block_count : 1) *
                                         sizeof(*compressed_sizes));
    if (!compressed_sizes ||
        !read_exact(input_file, compressed_sizes,
                    block_count * sizeof(*compressed_sizes))) {
        fprintf(stderr, "native decompress: metadata read failed\n");
        goto done;
    }
    compressed_bound = (size_t)LZ4_compressBound((int)block_size);
    for (base = 0; base < block_count; ++base) {
        size_t expected = (size_t)original_size - base * (size_t)block_size;
        if (expected > block_size) expected = block_size;
        if (compressed_sizes[base] == 0 || compressed_sizes[base] > compressed_bound ||
            size_add(payload_total, compressed_sizes[base], &payload_total) != 0 ||
            expected == 0) {
            fprintf(stderr, "native decompress: invalid block metadata\n");
            goto done;
        }
    }
    if (32U + block_count * sizeof(*compressed_sizes) > (size_t)frame_stat.st_size ||
        payload_total != (size_t)frame_stat.st_size - 32U -
                         block_count * sizeof(*compressed_sizes)) {
        fprintf(stderr, "native decompress: frame length mismatch\n");
        goto done;
    }

    batch_blocks = choose_batch_blocks(workers, block_size);
    if (size_mul(batch_blocks, compressed_bound, &compressed_capacity) != 0 ||
        size_mul(batch_blocks, block_size, &output_capacity) != 0) {
        fprintf(stderr, "native decompress: dimensions overflow\n");
        goto done;
    }
    setup_start = TIME_getTime();
    pool = TPool_create((int)workers, (int)batch_blocks);
    setup_end = TIME_getTime();
    compressed = (unsigned char*)malloc(compressed_capacity ? compressed_capacity : 1);
    output = (unsigned char*)malloc(output_capacity ? output_capacity : 1);
    jobs = (block_job*)calloc(batch_blocks, sizeof(*jobs));
    remove(temp_path);
    output_file = fopen(temp_path, "wb");
    temp_active = output_file != NULL;
    if (!pool || !compressed || !output || !jobs || !output_file) {
        fprintf(stderr, "native decompress: setup failed\n");
        goto done;
    }

    for (base = 0; base < block_count; base += batch_blocks) {
        size_t count = block_count - base;
        size_t index;
        if (count > batch_blocks) count = batch_blocks;
        for (index = 0; index < count; ++index) {
            size_t global = base + index;
            size_t expected = (size_t)original_size - global * (size_t)block_size;
            if (expected > block_size) expected = block_size;
            jobs[index].input = compressed + index * compressed_bound;
            jobs[index].output = output + index * (size_t)block_size;
            jobs[index].input_size = (int)compressed_sizes[global];
            jobs[index].output_capacity = (int)expected;
            if (!read_exact(input_file, (void*)jobs[index].input,
                            compressed_sizes[global])) {
                fprintf(stderr, "native decompress: payload read failed\n");
                goto done;
            }
        }
        if (submit_jobs(pool, jobs, count, decompress_block, &metric.kernel_us) != 0) {
            fprintf(stderr, "native decompress: block decompression failed\n");
            goto done;
        }
        for (index = 0; index < count; ++index) {
            size_t global = base + index;
            size_t expected = (size_t)original_size - global * (size_t)block_size;
            if (expected > block_size) expected = block_size;
            if ((size_t)jobs[index].result != expected ||
                !write_exact(output_file, jobs[index].output, expected)) {
                fprintf(stderr, "native decompress: decoded length or write failed\n");
                goto done;
            }
        }
    }
    if (fgetc(input_file) != EOF || ferror(input_file) ||
        finalize_output(&output_file, temp_path, output_path) != 0) {
        fprintf(stderr, "native decompress: trailing input or finalization failed\n");
        goto done;
    }
    temp_active = 0;
    total_end = TIME_getTime();
    metric.workers = workers;
    metric.active_workers = block_count < workers ? (unsigned)block_count : workers;
    metric.block_size = block_size;
    metric.hash_log = hash_log;
    metric.chunk_blocks = block_count < batch_blocks ? block_count : batch_blocks;
    metric.input_bytes = (uint64_t)frame_stat.st_size;
    metric.output_bytes = original_size;
    metric.runtime_setup_us = elapsed_us(setup_start, setup_end);
    metric.total_us = elapsed_us(total_start, total_end);
    metric.no_ocl_us = metric.total_us > metric.runtime_setup_us
                     ? metric.total_us - metric.runtime_setup_us : metric.total_us;
    if (metric.no_ocl_us < metric.kernel_us) metric.no_ocl_us = metric.kernel_us;
    if (metric.total_us < metric.no_ocl_us) metric.total_us = metric.no_ocl_us;
    if (write_metric(metrics_path, "decompress", &metric) != 0) {
        fprintf(stderr, "native decompress: metrics write failed\n");
        remove(output_path);
        goto done;
    }
    fprintf(stderr,
            "[native-lz4-decompress] %s -> %s : %llu bytes, blocks=%zu workers=%u\n",
            input_path, output_path, (unsigned long long)original_size,
            block_count, workers);
    result = 0;

done:
    if (input_file) fclose(input_file);
    if (output_file) fclose(output_file);
    if (temp_active) remove(temp_path);
    if (pool) TPool_free(pool);
    free(compressed_sizes);
    free(compressed);
    free(output);
    free(jobs);
    return result;
}

static void show_help(const char* program) {
    fprintf(stderr,
            "usage: %s (-c|-d) [--workers 1|2|4|8] [-B bytes] [--d-bits 11..15] "
            "[--metrics-json path] <input> -o <output>\n",
            program);
}

int main(int argc, char** argv) {
    operation_mode mode = MODE_NONE;
    unsigned workers = DEFAULT_WORKERS;
    unsigned block_size = DEFAULT_BLOCK_SIZE;
    unsigned hash_log = DEFAULT_HASH_LOG;
    int block_size_set = 0, hash_log_set = 0;
    const char* input_path = NULL;
    const char* output_path = NULL;
    const char* metrics_path = NULL;
    int index;
    const char* env_workers = getenv("HETEROLZ_CPU_THREADS");
    if (env_workers && parse_unsigned(env_workers, &workers) != 0) {
        fprintf(stderr, "invalid HETEROLZ_CPU_THREADS\n");
        return 2;
    }
    for (index = 1; index < argc; ++index) {
        const char* argument = argv[index];
        if (strcmp(argument, "-c") == 0 || strcmp(argument, "--compress") == 0) {
            if (mode != MODE_NONE) { show_help(argv[0]); return 2; }
            mode = MODE_COMPRESS;
        } else if (strcmp(argument, "-d") == 0 || strcmp(argument, "--decompress") == 0) {
            if (mode != MODE_NONE) { show_help(argv[0]); return 2; }
            mode = MODE_DECOMPRESS;
        } else if (strcmp(argument, "--workers") == 0) {
            if (++index >= argc || parse_unsigned(argv[index], &workers) != 0) {
                show_help(argv[0]); return 2;
            }
        } else if (strcmp(argument, "-B") == 0 || strcmp(argument, "--block-size") == 0) {
            if (++index >= argc || parse_unsigned(argv[index], &block_size) != 0) {
                show_help(argv[0]); return 2;
            }
            block_size_set = 1;
        } else if (strcmp(argument, "--d-bits") == 0) {
            if (++index >= argc || parse_unsigned(argv[index], &hash_log) != 0) {
                show_help(argv[0]); return 2;
            }
            hash_log_set = 1;
        } else if (strcmp(argument, "--metrics-json") == 0) {
            if (++index >= argc) { show_help(argv[0]); return 2; }
            metrics_path = argv[index];
        } else if (strcmp(argument, "-o") == 0) {
            if (++index >= argc) { show_help(argv[0]); return 2; }
            output_path = argv[index];
        } else if (strcmp(argument, "-h") == 0 || strcmp(argument, "--help") == 0) {
            show_help(argv[0]);
            return 0;
        } else if (argument[0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argument);
            return 2;
        } else if (!input_path) {
            input_path = argument;
        } else {
            show_help(argv[0]);
            return 2;
        }
    }
    if (mode == MODE_NONE || !input_path || !output_path || !worker_count_valid(workers) ||
        block_size == 0 || block_size > MAX_BLOCK_SIZE || hash_log < 11 || hash_log > 15 ||
        paths_identify_same_file(input_path, output_path) ||
        (metrics_path && (paths_identify_same_file(metrics_path, input_path) ||
                          paths_identify_same_file(metrics_path, output_path)))) {
        show_help(argv[0]);
        return 2;
    }
    if (path_exists(output_path) || (metrics_path && path_exists(metrics_path))) {
        fprintf(stderr, "output and metrics paths must be new\n");
        return 2;
    }
    if (mode == MODE_COMPRESS) {
        return compress_file(input_path, output_path, metrics_path, workers,
                             (uint32_t)block_size, (uint32_t)hash_log);
    }
    return decompress_file(input_path, output_path, metrics_path, workers,
                           block_size_set ? (int)block_size : -1,
                           hash_log_set ? (int)hash_log : -1);
}
