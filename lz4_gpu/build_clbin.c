#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

static unsigned char* read_file_bin(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    unsigned char* buf;
    long sz;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    buf = (unsigned char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[sz] = 0;
    if (out_size) *out_size = (size_t)sz;
    return buf;
}

static void print_build_log(cl_program prog, cl_device_id device, cl_int err) {
    size_t log_sz = 0;
    if (clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz) != CL_SUCCESS || log_sz == 0) {
        fprintf(stderr, "OpenCL build failed: err=%d\n", err);
        return;
    }
    {
        char* log = (char*)malloc(log_sz + 1);
        if (!log) {
            fprintf(stderr, "OpenCL build failed: err=%d (log alloc failed)\n", err);
            return;
        }
        if (clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL) == CL_SUCCESS) {
            log[log_sz] = '\0';
            fprintf(stderr, "Build Error:\n%s\n", log);
        } else {
            fprintf(stderr, "OpenCL build failed: err=%d (could not read build log)\n", err);
        }
        free(log);
    }
}

static int pick_device(cl_platform_id* out_platform, cl_device_id* out_device) {
    cl_uint num_platforms = 0;
    cl_platform_id* platforms = NULL;
    cl_int err;
    cl_device_type candidates[] = {
        CL_DEVICE_TYPE_GPU,
        CL_DEVICE_TYPE_DEFAULT,
        CL_DEVICE_TYPE_ALL,
    };

    if (!out_platform || !out_device) return -1;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "clGetPlatformIDs failed: err=%d platforms=%u\n", err, num_platforms);
        return -1;
    }
    platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    if (!platforms) return -1;
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs(list) failed: err=%d\n", err);
        free(platforms);
        return -1;
    }

    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        for (size_t ci = 0; ci < sizeof(candidates) / sizeof(candidates[0]); ++ci) {
            cl_device_id device = NULL;
            err = clGetDeviceIDs(platforms[pi], candidates[ci], 1, &device, NULL);
            if (err == CL_SUCCESS && device) {
                *out_platform = platforms[pi];
                *out_device = device;
                free(platforms);
                return 0;
            }
        }
    }

    fprintf(stderr, "No usable OpenCL device found for build_clbin\n");
    free(platforms);
    return -1;
}

static int save_bin(const char* name, cl_program prog) {
    size_t sz = 0;
    unsigned char* storage = NULL;
    unsigned char* bins[1] = { NULL };
    FILE* f;

    if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(sz), &sz, NULL) != CL_SUCCESS || sz == 0) {
        fprintf(stderr, "Failed to query program binary size\n");
        return -1;
    }
    storage = (unsigned char*)malloc(sz);
    if (!storage) return -1;
    bins[0] = storage;
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bins), bins, NULL) != CL_SUCCESS) {
        fprintf(stderr, "Failed to query program binary\n");
        free(storage);
        return -1;
    }
    f = fopen(name, "wb");
    if (!f) {
        perror("fopen output");
        free(storage);
        return -1;
    }
    if (fwrite(storage, 1, sz, f) != sz) {
        perror("fwrite output");
        fclose(f);
        free(storage);
        return -1;
    }
    fclose(f);
    free(storage);
    printf("Saved %s (%zu bytes)\n", name, sz);
    return 0;
}

int main(int argc, char** argv) {
    const char* output_name = NULL;
    const char* flags = NULL;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context ctx = NULL;
    cl_program prog = NULL;
    unsigned char* src = NULL;
    size_t src_sz = 0;
    cl_int err;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s -o <output.clbin> \"<compiler flags>\"\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_name = argv[++i];
        } else if (i == argc - 1) {
            flags = argv[i];
        }
    }

    if (!output_name || !flags) {
        fprintf(stderr, "Missing output name or compiler flags\n");
        return 1;
    }

    if (pick_device(&platform, &device) != 0) {
        return 1;
    }

    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !ctx) {
        fprintf(stderr, "clCreateContext failed: err=%d\n", err);
        return 1;
    }

    src = read_file_bin("lz4_gpu.cl", &src_sz);
    if (!src) {
        fprintf(stderr, "Could not read lz4_gpu.cl\n");
        clReleaseContext(ctx);
        return 1;
    }

    prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_sz, &err);
    if (err != CL_SUCCESS || !prog) {
        fprintf(stderr, "clCreateProgramWithSource failed: err=%d\n", err);
        free(src);
        clReleaseContext(ctx);
        return 1;
    }

    err = clBuildProgram(prog, 1, &device, flags, NULL, NULL);
    if (err != CL_SUCCESS) {
        print_build_log(prog, device, err);
        clReleaseProgram(prog);
        free(src);
        clReleaseContext(ctx);
        return 1;
    }

    if (save_bin(output_name, prog) != 0) {
        clReleaseProgram(prog);
        free(src);
        clReleaseContext(ctx);
        return 1;
    }

    clReleaseProgram(prog);
    free(src);
    clReleaseContext(ctx);
    (void)platform;
    return 0;
}
