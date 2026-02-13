#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

void save_bin(const char* name, cl_program prog) {
    size_t sz;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &sz, NULL);
    unsigned char* bin = malloc(sz);
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &bin, NULL);
    FILE* f = fopen(name, "wb");
    if (f) {
        fwrite(bin, 1, sz, f);
        fclose(f);
        printf("Saved %s (%zu bytes)\n", name, sz);
    }
    free(bin);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s -o <output.clbin> \"<compiler flags>\"\n", argv[0]);
        return 1;
    }

    const char* output_name = NULL;
    const char* flags = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_name = argv[++i];
        } else if (i == argc - 1) {
            flags = argv[i];
        }
    }

    if (!output_name || !flags) return 1;

    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    FILE* f = fopen("lz4_gpu.cl", "r");
    if (!f) return 1;
    fseek(f, 0, SEEK_END); size_t s_sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* src = malloc(s_sz + 1);
    fread(src, 1, s_sz, f);
    src[s_sz] = 0;
    fclose(f);

    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &s_sz, &err);
    if (clBuildProgram(prog, 1, &device, flags, NULL, NULL) != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char* log = malloc(log_sz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        fprintf(stderr, "Build Error:\n%s\n", log);
        free(log);
        return 1;
    }

    save_bin(output_name, prog);
    clReleaseProgram(prog);
    clReleaseContext(ctx);
    free(src);
    return 0;
}
