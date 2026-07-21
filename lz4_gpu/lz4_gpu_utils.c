#if !defined(_WIN32)
#define _GNU_SOURCE
#endif
#include "lz4_gpu_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <limits.h>
#if !defined(_WIN32)
#include <sched.h>
#endif
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

uint64_t g_ocl_init_us = 0;
uint64_t g_kernel_load_us = 0;
static cl_device_id lz4_owned_subdevice = NULL;
static cl_uint lz4_owned_parent_compute_units = 0;

#if !defined(_WIN32)
static int lz4_parse_cpu_set(const char* value, cpu_set_t* set) {
    const char* cursor = value;
    char* end = NULL;
    if (!value || !*value || !set) return -1;
    CPU_ZERO(set);
    while (*cursor) {
        errno = 0;
        unsigned long cpu = strtoul(cursor, &end, 10);
        if (errno == ERANGE || end == cursor || cpu >= CPU_SETSIZE) return -1;
        CPU_SET((int)cpu, set);
        if (*end == '\0') return 0;
        if (*end != ',') return -1;
        cursor = end + 1;
        if (!*cursor) return -1;
    }
    return -1;
}

static int lz4_set_affinity_from_env(const char* name) {
    cpu_set_t set;
    const char* value = getenv(name);
    if (!value || !*value) return 0;
    if (lz4_parse_cpu_set(value, &set) != 0) return -1;
    return sched_setaffinity(0, sizeof(set), &set);
}

static int lz4_get_affinity_from_env(const char* name, cpu_set_t* set) {
    const char* value = getenv(name);
    if (!value || !*value) return 0;
    return lz4_parse_cpu_set(value, set) == 0 ? 1 : -1;
}
#endif

static cl_int lz4_try_get_device(cl_platform_id* platforms,
                                 cl_uint num_platforms,
                                 cl_device_type dtype,
                                 cl_device_id* out_dev,
                                 cl_platform_id* out_pf) {
    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        cl_uint count = 0;
        cl_int r = clGetDeviceIDs(platforms[pi], dtype, 0, NULL, &count);
        if (r != CL_SUCCESS || count == 0) continue;
        cl_device_id* devices = (cl_device_id*)calloc(count, sizeof(*devices));
        if (!devices) return CL_OUT_OF_HOST_MEMORY;
        r = clGetDeviceIDs(platforms[pi], dtype, count, devices, NULL);
        if (r == CL_SUCCESS) {
            for (cl_uint di = 0; di < count; ++di) {
                cl_device_type actual_type = 0;
                if (clGetDeviceInfo(devices[di], CL_DEVICE_TYPE,
                                    sizeof(actual_type), &actual_type, NULL) != CL_SUCCESS) {
                    continue;
                }
                if (dtype == CL_DEVICE_TYPE_ALL || (actual_type & dtype)) {
                    *out_dev = devices[di];
                    *out_pf = platforms[pi];
                    free(devices);
                    return CL_SUCCESS;
                }
            }
        }
        free(devices);
    }
    return CL_DEVICE_NOT_FOUND;
}

static cl_device_type lz4_preferred_opencl_device_type(int* strict) {
    const char* pref = getenv("FORCE_OPENCL_DEVICE");
    *strict = pref && *pref;
    if (!*strict) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "CPU") == 0) return CL_DEVICE_TYPE_CPU;
    if (strcasecmp(pref, "GPU") == 0) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "DEFAULT") == 0) return CL_DEVICE_TYPE_DEFAULT;
    if (strcasecmp(pref, "ALL") == 0) return CL_DEVICE_TYPE_ALL;
    return 0;
}

static int lz4_cpu_thread_limit(cl_uint* limit) {
    const char* value = getenv("HETEROLZ_CPU_THREADS");
    char* end = NULL;
    unsigned long parsed;
    if (!value || !*value) return 0;
    errno = 0;
    parsed = strtoul(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed < 1 || parsed > UINT_MAX) {
        return -1;
    }
    *limit = (cl_uint)parsed;
    return 1;
}

static cl_int lz4_partition_cpu_device(cl_device_id root, cl_uint limit,
                                       cl_device_id* out_device) {
    cl_uint compute_units = 0;
    cl_uint count = 0;
    cl_device_id* devices = NULL;
    const cl_device_partition_property properties[] = {
        CL_DEVICE_PARTITION_EQUALLY,
        (cl_device_partition_property)limit,
        0,
    };
    cl_int err = clGetDeviceInfo(root, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(compute_units), &compute_units, NULL);
    if (err != CL_SUCCESS) return err;
    if (compute_units < limit) return CL_INVALID_VALUE;
    err = clCreateSubDevices(root, properties, 0, NULL, &count);
    if (err != CL_SUCCESS || count == 0) return err != CL_SUCCESS ? err : CL_DEVICE_PARTITION_FAILED;
    devices = (cl_device_id*)calloc(count, sizeof(*devices));
    if (!devices) return CL_OUT_OF_HOST_MEMORY;
    err = clCreateSubDevices(root, properties, count, devices, NULL);
    if (err == CL_SUCCESS) {
        *out_device = devices[0];
        for (cl_uint index = 1; index < count; ++index) clReleaseDevice(devices[index]);
    }
    free(devices);
    return err;
}

cl_int lz4_select_opencl_platform_device(cl_platform_id* out_pf, cl_device_id* out_dev) {
    cl_uint num_platforms = 0;
    cl_platform_id* platforms = NULL;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_int r = CL_DEVICE_NOT_FOUND;
    int strict = 0;
    cl_device_type pref_type = lz4_preferred_opencl_device_type(&strict);
#if !defined(_WIN32)
    cpu_set_t original_affinity;
    cpu_set_t discovery_affinity;
    int affinity_expanded = 0;
#endif

    if (!out_pf || !out_dev) return CL_INVALID_VALUE;
    *out_pf = NULL;
    *out_dev = NULL;

    if (pref_type == 0) return CL_INVALID_VALUE;
    if (err != CL_SUCCESS || num_platforms == 0) return CL_DEVICE_NOT_FOUND;

#if !defined(_WIN32)
    if (pref_type == CL_DEVICE_TYPE_CPU) {
        int discovery_status = lz4_get_affinity_from_env(
            "HETEROLZ_CPU_DISCOVERY_SET", &discovery_affinity);
        if (discovery_status < 0) return CL_INVALID_VALUE;
        if (discovery_status > 0) {
            if (sched_getaffinity(0, sizeof(original_affinity), &original_affinity) != 0 ||
                    sched_setaffinity(0, sizeof(discovery_affinity), &discovery_affinity) != 0) {
                return CL_INVALID_OPERATION;
            }
            affinity_expanded = 1;
        }
    }
#endif

    platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
#if !defined(_WIN32)
        if (affinity_expanded) (void)sched_setaffinity(0, sizeof(original_affinity), &original_affinity);
#endif
        return CL_OUT_OF_HOST_MEMORY;
    }

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
#if !defined(_WIN32)
        if (affinity_expanded) (void)sched_setaffinity(0, sizeof(original_affinity), &original_affinity);
#endif
        return err;
    }

    if (strict) {
        r = lz4_try_get_device(platforms, num_platforms, pref_type, out_dev, out_pf);
    } else if (pref_type == CL_DEVICE_TYPE_GPU) {
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

    if (r == CL_SUCCESS && *out_dev != NULL) {
        cl_device_type selected_type = 0;
        cl_uint cpu_limit = 0;
        cl_uint parent_compute_units = 0;
        int limit_status = lz4_cpu_thread_limit(&cpu_limit);
        if (limit_status < 0) {
            *out_dev = NULL;
            *out_pf = NULL;
            r = CL_INVALID_VALUE;
        } else if (limit_status > 0 &&
                   clGetDeviceInfo(*out_dev, CL_DEVICE_TYPE, sizeof(selected_type),
                                   &selected_type, NULL) == CL_SUCCESS &&
                   (selected_type & CL_DEVICE_TYPE_CPU)) {
            cl_device_id partitioned = NULL;
            (void)clGetDeviceInfo(*out_dev, CL_DEVICE_MAX_COMPUTE_UNITS,
                                  sizeof(parent_compute_units), &parent_compute_units, NULL);
            r = lz4_partition_cpu_device(*out_dev, cpu_limit, &partitioned);
            if (r == CL_SUCCESS) {
                *out_dev = partitioned;
                lz4_owned_subdevice = partitioned;
                lz4_owned_parent_compute_units = parent_compute_units;
            }
            else {
                *out_dev = NULL;
                *out_pf = NULL;
            }
        }
    }

#if !defined(_WIN32)
    if (r == CL_SUCCESS && *out_dev != NULL &&
            lz4_set_affinity_from_env("HETEROLZ_CPU_SET") != 0) {
        lz4_release_opencl_device(out_dev);
        *out_pf = NULL;
        r = CL_INVALID_OPERATION;
    }
#endif

    free(platforms);
    return r;
}

void lz4_release_opencl_device(cl_device_id* device) {
    if (!device || !*device) return;
    if (*device == lz4_owned_subdevice) {
        clReleaseDevice(*device);
        lz4_owned_subdevice = NULL;
        lz4_owned_parent_compute_units = 0;
    }
    *device = NULL;
}

int lz4_opencl_device_partition_info(cl_device_id device, cl_uint* parent_compute_units) {
    if (!device || device != lz4_owned_subdevice) return 0;
    if (parent_compute_units) *parent_compute_units = lz4_owned_parent_compute_units;
    return 1;
}

void lz4_device_profile_key(cl_device_id device, char* buffer, size_t buffer_size) {
    char name[256] = {0};
    cl_uint compute_units = 0;
    if (!buffer || buffer_size == 0) return;
    buffer[0] = '\0';
    if (!device || clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL) != CL_SUCCESS) return;
    (void)clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(compute_units), &compute_units, NULL);
    if (lz4_opencl_device_partition_info(device, NULL))
        snprintf(buffer, buffer_size, "%s|cpu-cu=%u", name, compute_units);
    else snprintf(buffer, buffer_size, "%s", name);
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
