/*
 * Small helper: compile lz4_gpu.cl into an OpenCL program binary using provided build flags.
 * Usage: build_clbin -o <out_clbin> "<build_flags>"
 * Example: ./build_clbin -o lz4_gpu_vec.clbin "-DLZ4_GPU_VECTOR_IO=1"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

static char* read_file_text(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = malloc(s+1); if (!buf) { fclose(f); return NULL; }
    if (fread(buf,1,s,f) != (size_t)s) { free(buf); fclose(f); return NULL; }
    buf[s] = '\0'; if (out_len) *out_len = (size_t)s; fclose(f); return buf;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("Usage: %s -o <out_clbin> \"<build_flags>\"\n", argv[0]); return 1; }
    const char* out = NULL; const char* flags = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-o") && i+1 < argc) { out = argv[++i]; }
        else flags = argv[i];
    }
    if (!out) { printf("Missing -o <out_clbin>\n"); return 2; }

    cl_int err; cl_uint numPlatforms = 0;
    if (clGetPlatformIDs(0,NULL,&numPlatforms) != CL_SUCCESS || numPlatforms==0) { fprintf(stderr,"No OpenCL platforms\n"); return 3; }
    cl_platform_id platform; clGetPlatformIDs(1,&platform,NULL);
    cl_uint numDevices = 0; clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    if (numDevices==0) { fprintf(stderr,"No OpenCL devices\n"); return 4; }
    cl_device_id device; clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL,1,&device,NULL,NULL,&err); if (err!=CL_SUCCESS) { fprintf(stderr,"clCreateContext failed\n"); return 5; }

    size_t src_len = 0; char* src = read_file_text("lz4_gpu.cl", &src_len); if (!src) { fprintf(stderr,"Cannot read lz4_gpu.cl\n"); return 6; }
    const char* s = src; cl_program program = clCreateProgramWithSource(ctx,1,&s,&src_len,&err); if (err!=CL_SUCCESS) { fprintf(stderr,"clCreateProgramWithSource failed\n"); return 7; }
    // build with provided flags (may be NULL)
    err = clBuildProgram(program, 0, NULL, flags, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logsz = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
        char* log = malloc(logsz+1); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL); log[logsz]=0;
        fprintf(stderr, "Build log:\n%s\n", log); free(log); return 8;
    }
    size_t num_devices = 1; clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_devices, NULL);
    size_t* bin_sizes = malloc(sizeof(size_t)*num_devices); clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*num_devices, bin_sizes, NULL);
    unsigned char** bins = malloc(sizeof(unsigned char*)*num_devices); for (size_t i=0;i<num_devices;i++) bins[i]=malloc(bin_sizes[i]);
    clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*)*num_devices, bins, NULL);
    FILE* f = fopen(out, "wb"); if (!f) { fprintf(stderr,"Cannot open %s for write\n", out); return 9; }
    fwrite(bins[0], 1, bin_sizes[0], f); fclose(f);
    printf("Wrote %s (%zu bytes)\n", out, bin_sizes[0]);
    for (size_t i=0;i<num_devices;i++) { free(bins[i]); }
    free(bins); free(bin_sizes); free(src);
    clReleaseProgram(program); clReleaseContext(ctx);
    return 0;
}
