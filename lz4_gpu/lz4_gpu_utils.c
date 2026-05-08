#include "lz4_gpu_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

uint64_t g_ocl_init_us = 0;
uint64_t g_kernel_load_us = 0;

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

static cl_device_type lz4_preferred_opencl_device_type(void) {
    const char* pref = getenv("FORCE_OPENCL_DEVICE");
    if (!pref || !*pref) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "CPU") == 0) return CL_DEVICE_TYPE_CPU;
    if (strcasecmp(pref, "GPU") == 0) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "DEFAULT") == 0) return CL_DEVICE_TYPE_DEFAULT;
    if (strcasecmp(pref, "ALL") == 0) return CL_DEVICE_TYPE_ALL;
    return CL_DEVICE_TYPE_GPU;
}

cl_int lz4_select_opencl_platform_device(cl_platform_id* out_pf, cl_device_id* out_dev) {
    cl_uint num_platforms = 0;
    cl_platform_id* platforms = NULL;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_int r = CL_DEVICE_NOT_FOUND;
    cl_device_type pref_type = lz4_preferred_opencl_device_type();

    if (!out_pf || !out_dev) return CL_INVALID_VALUE;
    *out_pf = NULL;
    *out_dev = NULL;

    if (err != CL_SUCCESS || num_platforms == 0) return CL_DEVICE_NOT_FOUND;

    platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) return CL_OUT_OF_HOST_MEMORY;

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        return err;
    }

    if (pref_type == CL_DEVICE_TYPE_GPU) {
        r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, out_dev, out_pf);
    } else if (pref_type == CL_DEVICE_TYPE_CPU) {
        r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, out_dev, out_pf);
    } else if (pref_type == CL_DEVICE_TYPE_DEFAULT) {
        r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, out_dev, out_pf);
    } else {
        r = lz4_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, out_dev, out_pf);
    }

    free(platforms);
    return r;
}

uint64_t get_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

int lz4_read_file_to_buf(const char* path, void* buf, size_t sz, unsigned long* time_us) {
    uint64_t t1 = get_us();
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    size_t nr = fread(buf, 1, sz, f);
    fclose(f);
    uint64_t t2 = get_us();
    if (time_us) *time_us = (unsigned long)(t2 - t1);
    return (nr == sz) ? 0 : -1;
}

const char* format_size(size_t size) {
    static char buf[32];
    if (size < 1024) snprintf(buf, sizeof(buf), "%zu B", size);
    else if (size < 1024 * 1024) snprintf(buf, sizeof(buf), "%.2f KB", size / 1024.0);
    else if (size < 1024 * 1024 * 1024) snprintf(buf, sizeof(buf), "%.2f MB", size / (1024.0 * 1024.0));
    else snprintf(buf, sizeof(buf), "%.2f GB", size / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

static void print_time_row(FILE *f, const char *label, unsigned long us, unsigned long total_us) {
    double ms = us / 1000.0;
    double pct = total_us > 0 ? (100.0 * us / total_us) : 0.0;
    fprintf(f, "%-22s : %10.3f ms (%6.2f%%)\n", label, ms, pct);
}

void print_response_stats(const response_t* resp, const char* input_path, int mode) {
    const timing_t* t = &resp->timing;
    size_t in_size = t->in_size;
    if (in_size == 0) {
        struct stat st;
        if (stat(input_path, &st) == 0) in_size = st.st_size;
    }

    const char* mode_str = (mode == 0) ? "Compress" : "Decompress";

    printf("\n==============================================================================\n");
    printf("  LZ4 GPU PERFORMANCE REPORT (%s)\n", mode_str);
    printf("==============================================================================\n");
    printf("%-22s : %s\n", "Input File", input_path);
    printf("%-22s : %zu bytes (%s)\n", "Input Size", in_size, format_size(in_size));

    if (mode == 0) {
        double ratio_pct = in_size > 0 ? (100.0 * (double)t->out_size / in_size) : 0;
        printf("%-22s : %zu bytes (%s) (%.2f%% ratio)\n", "Output Size", (size_t)t->out_size, format_size(t->out_size), ratio_pct);
    } else {
        printf("%-22s : %zu bytes (%s)\n", "Output Size", (size_t)t->out_size, format_size(t->out_size));
    }

    printf("%-22s : LZ4 GPU (HashLog: %d)\n", "Algorithm", t->algo_config);
    printf("%-22s : %lu blocks (BlockSize: %s)\n", "Workload", t->nblk, format_size(t->blk_size_bytes));
    printf("%-22s : Global: %lu, Local: %lu\n", "Grid Size", t->global_size, t->local_size);

    printf("------------------------------------------------------------------------------\n");
    printf("  Detailed Timing Breakdown\n");
    printf("------------------------------------------------------------------------------\n");

    unsigned long total_us = resp->time_us;
    print_time_row(stdout, "File Read", t->file_read_us, total_us);
    print_time_row(stdout, "OCI Setup", t->ocl_setup_us, total_us);
    print_time_row(stdout, "Buffer Alloc", t->buffer_alloc_us, total_us);
    print_time_row(stdout, "Data Upload/Map", t->data_upload_us, total_us);
    print_time_row(stdout, "Kernel Execution", t->kernel_exec_us, total_us);
    print_time_row(stdout, "Data Download/Unmap", t->download_total_us, total_us);
    print_time_row(stdout, "File Write", t->file_write_us, total_us);

    printf("------------------------------------------------------------------------------\n");
    printf("%-22s : %10.3f ms\n", "TOTAL INCLUSIVE", total_us / 1000.0);
    printf("------------------------------------------------------------------------------\n");

    size_t throughput_bytes = (mode == 0) ? in_size : (size_t)t->out_size;
    double in_mb = (double)throughput_bytes / (1024.0 * 1024.0);
    if (total_us > 0) {
        double mb_s = in_mb / ((double)total_us / 1000000.0);
        printf("%-22s : %10.2f MB/s\n", "Inclusive Throughput", mb_s);
    }
    if (t->kernel_exec_us > 0) {
        double mb_s = in_mb / ((double)t->kernel_exec_us / 1000000.0);
        printf("%-22s : %10.2f MB/s\n", "Kernel Throughput", mb_s);
    }
    printf("==============================================================================\n\n");
}
