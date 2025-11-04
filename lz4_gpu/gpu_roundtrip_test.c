/*
 * Minimal OpenCL host test harness for backup_vector lz4_gpu kernels.
 * - Builds from `lz4_gpu.cl` or uses precompiled `lz4_gpu.clbin` if present
 * - Splits input into 64KB blocks, runs lz4_compress_block_accelerated per block
 * - Runs lz4_decompress_blocks to reconstruct original and validates
 * - Reports kernel execution times via OpenCL event profiling
 *
 * Build (Linux):
 *   gcc -O2 gpu_roundtrip_test.c -o gpu_roundtrip_test -lOpenCL
 * Run:
 *   ./gpu_roundtrip_test <input-file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <CL/cl.h>

#define BLOCK_SIZE (64*1024)
#define EST_DST_CAPACITY(src) ((src) + (src)/255 + 64)

static char* read_file_text(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, size, f) != (size_t)size) { free(buf); fclose(f); return NULL; }
    buf[size] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)size;
    return buf;
}

static void write_binary_file(const char* path, const unsigned char* data, size_t len) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(data, 1, len, f);
    fclose(f);
}

static void die(const char* msg) { perror(msg); exit(1); }

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <input-file> [clbin_path] [acceleration]\n", argv[0]);
        printf("  clbin_path : optional path to precompiled OpenCL binary (default: lz4_gpu.clbin)\n");
        printf("  acceleration: optional integer passed to compression kernel (default: 1)\n");
        return 1;
    }

    const char* src_path = argv[1];
    const char* bin_path = "lz4_gpu.clbin"; /* default precompiled binary */
    int acceleration = 1; /* default acceleration */
    if (argc >= 3) {
        /* If argv[2] looks like a number, treat it as acceleration; otherwise treat as bin path */
        char* endptr = NULL;
        long v = strtol(argv[2], &endptr, 10);
        if (endptr != argv[2] && *endptr == '\0' && v > 0) {
            acceleration = (int)v;
        } else {
            bin_path = argv[2];
        }
    }
    if (argc >= 4) {
        char* endptr = NULL;
        long v = strtol(argv[3], &endptr, 10);
        if (endptr != argv[3] && *endptr == '\0' && v > 0) {
            acceleration = (int)v;
        }
    }
    size_t src_len = 0;
    FILE* f = fopen(src_path, "rb");
    if (!f) die("fopen");
    fseek(f, 0, SEEK_END);
    src_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* src = malloc(src_len);
    if (!src) die("malloc src");
    if (fread(src, 1, src_len, f) != src_len) die("fread");
    fclose(f);

    // Partition into blocks
    size_t totalBlocks = (src_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Input %zu bytes => %zu blocks\n", src_len, totalBlocks);

    // Prepare blockOffsets (pairs) and outputOffsets
    uint32_t* blockOffsets = calloc(totalBlocks * 2, sizeof(uint32_t));
    uint32_t* outputOffsets = calloc(totalBlocks, sizeof(uint32_t));
    uint32_t* initialOutputOffsets = calloc(totalBlocks, sizeof(uint32_t));
    size_t max_compressed_buffer = 0;
    for (size_t i = 0; i < totalBlocks; ++i) {
        uint32_t start = (uint32_t)(i * BLOCK_SIZE);
        uint32_t sz = (uint32_t)((i == totalBlocks - 1) ? (src_len - i*BLOCK_SIZE) : BLOCK_SIZE);
        blockOffsets[i*2 + 0] = start;
        blockOffsets[i*2 + 1] = sz;
        uint32_t dstCap = (uint32_t)EST_DST_CAPACITY(sz);
        initialOutputOffsets[i] = (uint32_t)(i * ((BLOCK_SIZE/2) + 256 + 1)); // conservative stride
        outputOffsets[i] = initialOutputOffsets[i];
        max_compressed_buffer += dstCap;
    }

    printf("Allocating device buffers: compressed buffer approx %zu bytes\n", max_compressed_buffer);

    // Setup OpenCL
    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) die("clGetPlatformIDs");
    cl_platform_id* platforms = malloc(sizeof(cl_platform_id) * numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms, NULL);
    cl_platform_id platform = platforms[0];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        // fallback to CPU device
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, NULL, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) die("clGetDeviceIDs");
    }
    cl_device_id* devices = malloc(sizeof(cl_device_id) * numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) die("clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) die("clCreateCommandQueue");

    // Try to load precompiled binary first
    if (argc >= 3) {
        /* bin_path may have been set above when argv[2] was non-numeric */
    }
    FILE* fb = fopen(bin_path, "rb");
    cl_program program = NULL;
    if (fb) {
        fseek(fb, 0, SEEK_END);
        long bsize = ftell(fb);
        fseek(fb, 0, SEEK_SET);
        unsigned char* bin = malloc(bsize);
        if (fread(bin, 1, bsize, fb) != (size_t)bsize) die("read bin");
        fclose(fb);
        cl_int bin_status;
        program = clCreateProgramWithBinary(context, 1, &device, (const size_t*)&bsize, (const unsigned char**)&bin, &bin_status, &err);
        free(bin);
        if (err != CL_SUCCESS) die("clCreateProgramWithBinary");
        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            // print build log
            size_t logsz = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
            char* log = malloc(logsz+1); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL); log[logsz]=0;
            printf("Build log:\n%s\n", log); free(log);
            die("clBuildProgram (binary)");
        }
        printf("Loaded program from binary %s\n", bin_path);
    } else {
        // read source
        size_t srccl_len = 0;
        char* srccl = read_file_text("lz4_gpu.cl", &srccl_len);
        if (!srccl) die("read lz4_gpu.cl");
        const char* s = srccl;
        program = clCreateProgramWithSource(context, 1, &s, &srccl_len, &err);
        if (err != CL_SUCCESS) die("clCreateProgramWithSource");
        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t logsz = 0; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
            char* log = malloc(logsz+1); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL); log[logsz]=0;
            printf("Build log:\n%s\n", log); free(log);
            die("clBuildProgram");
        }
        printf("Built program from source\n");

        // Export binary for future runs
        size_t num_devices = 0; clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_devices, NULL);
        size_t binary_sizes_count = num_devices;
        unsigned char** binaries = malloc(sizeof(unsigned char*) * num_devices);
        size_t* binary_sizes = malloc(sizeof(size_t) * num_devices);
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*num_devices, binary_sizes, NULL);
        for (size_t i = 0; i < num_devices; ++i) binaries[i] = malloc(binary_sizes[i]);
        err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*)*num_devices, binaries, NULL);
        if (err == CL_SUCCESS) {
            // write first binary
            write_binary_file(bin_path, binaries[0], binary_sizes[0]);
            printf("Wrote precompiled binary %s (%zu bytes)\n", bin_path, binary_sizes[0]);
        }
        for (size_t i = 0; i < num_devices; ++i) free(binaries[i]); free(binaries); free(binary_sizes);
        free(srccl);
    }

    // Create kernels
    cl_kernel k_compress = clCreateKernel(program, "lz4_compress_block_accelerated", &err);
    if (err != CL_SUCCESS) die("clCreateKernel compress");
    cl_kernel k_decompress = clCreateKernel(program, "lz4_decompress_blocks", &err);
    if (err != CL_SUCCESS) die("clCreateKernel decompress");

    // Create buffers
    // Create input buffer and explicitly enqueue write to measure host->device time
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, src_len, NULL, &err); if (err!=CL_SUCCESS) die("clCreateBuffer input");
    cl_event ev_write_input = NULL;
    err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, src_len, src, 0, NULL, &ev_write_input); if (err!=CL_SUCCESS) die("clEnqueueWriteBuffer input");
    // read profiling for host->device
    cl_ulong write_start=0, write_end=0;
    if (ev_write_input) {
        clGetEventProfilingInfo(ev_write_input, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_start, NULL);
        clGetEventProfilingInfo(ev_write_input, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write_end, NULL);
    }
    size_t compressed_buf_sz = max_compressed_buffer + 1024;
    cl_mem d_compressed = clCreateBuffer(context, CL_MEM_READ_WRITE, compressed_buf_sz, NULL, &err); if (err!=CL_SUCCESS) die("clCreateBuffer compressed");

    cl_mem d_blockOffsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*totalBlocks*2, blockOffsets, &err); if (err!=CL_SUCCESS) die("clCreateBuffer blockOffsets");
    cl_mem d_outputOffsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint32_t)*totalBlocks, outputOffsets, &err); if (err!=CL_SUCCESS) die("clCreateBuffer outputOffsets");
    cl_mem d_blockSizes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t)*totalBlocks, NULL, &err); if (err!=CL_SUCCESS) die("clCreateBuffer blockSizes");

    // set args for compress kernel
    err  = clSetKernelArg(k_compress, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(k_compress, 1, sizeof(cl_mem), &d_compressed);
    err |= clSetKernelArg(k_compress, 2, sizeof(cl_mem), &d_blockSizes);
    err |= clSetKernelArg(k_compress, 3, sizeof(cl_mem), &d_blockOffsets);
    err |= clSetKernelArg(k_compress, 4, sizeof(cl_mem), &d_outputOffsets);
    int totalBlocks_i = (int)totalBlocks;
    err |= clSetKernelArg(k_compress, 5, sizeof(int), &totalBlocks_i);
    int inputSize_i = (int)src_len;
    err |= clSetKernelArg(k_compress, 6, sizeof(int), &inputSize_i);
    int tableType = 1;
    err |= clSetKernelArg(k_compress, 7, sizeof(int), &tableType);
    /* pass configured acceleration into kernel */
    err |= clSetKernelArg(k_compress, 8, sizeof(int), &acceleration);
    if (err != CL_SUCCESS) die("clSetKernelArg compress");

    size_t global = totalBlocks;
    size_t local = 1; /* force single work-item per work-group to avoid __local hash table races */
    cl_event ev_comp;
    err = clEnqueueNDRangeKernel(queue, k_compress, 1, NULL, &global, &local, 0, NULL, &ev_comp); if (err!=CL_SUCCESS) die("clEnqueueNDRangeKernel compress");
    clFinish(queue);
    // profile
    cl_ulong startt=0, endt=0; clGetEventProfilingInfo(ev_comp, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startt, NULL); clGetEventProfilingInfo(ev_comp, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endt, NULL);
    double comp_ms = (endt - startt) * 1e-6;
    double host_to_device_ms = 0.0;
    if (ev_write_input) host_to_device_ms = (write_end - write_start) * 1e-6;
    printf("Host->Device upload time: %.3f ms\n", host_to_device_ms);
    printf("Compress kernel time: %.3f ms\n", comp_ms);

    // read blockSizes
    uint32_t* blockSizes = malloc(sizeof(uint32_t)*totalBlocks);
    // Read back blockSizes with profiling
    cl_event ev_read_blockSizes = NULL;
    err = clEnqueueReadBuffer(queue, d_blockSizes, CL_FALSE, 0, sizeof(uint32_t)*totalBlocks, blockSizes, 0, NULL, &ev_read_blockSizes); if (err!=CL_SUCCESS) die("clEnqueueReadBuffer blockSizes");
    clFinish(queue);
    cl_ulong read_bs_start=0, read_bs_end=0;
    if (ev_read_blockSizes) {
        clGetEventProfilingInfo(ev_read_blockSizes, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_bs_start, NULL);
        clGetEventProfilingInfo(ev_read_blockSizes, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_bs_end, NULL);
    }
    double device_to_host_blocksizes_ms = (read_bs_end - read_bs_start) * 1e-6;
    printf("Device->Host read blockSizes time: %.3f ms\n", device_to_host_blocksizes_ms);

    // Create arrays for comp offsets/sizes as 64-bit for decompress kernel
    uint64_t* comp_offsets = malloc(sizeof(uint64_t)*totalBlocks);
    uint64_t* comp_sizes = malloc(sizeof(uint64_t)*totalBlocks);
    uint64_t* out_offsets = malloc(sizeof(uint64_t)*totalBlocks);
    uint64_t* max_out_sizes = malloc(sizeof(uint64_t)*totalBlocks);
    uint32_t* sizes_out = calloc(totalBlocks, sizeof(uint32_t));

    for (size_t i = 0; i < totalBlocks; ++i) {
        comp_offsets[i] = (uint64_t)initialOutputOffsets[i];
        comp_sizes[i] = blockSizes[i];
        out_offsets[i] = (uint64_t)(i * BLOCK_SIZE);
        max_out_sizes[i] = (uint64_t)BLOCK_SIZE;
    }

    cl_mem d_comp_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t)*totalBlocks, comp_offsets, &err); if (err!=CL_SUCCESS) die("d_comp_offsets");
    cl_mem d_comp_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t)*totalBlocks, comp_sizes, &err); if (err!=CL_SUCCESS) die("d_comp_sizes");
    cl_mem d_out_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t)*totalBlocks, out_offsets, &err); if (err!=CL_SUCCESS) die("d_out_offsets");
    cl_mem d_max_out_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t)*totalBlocks, max_out_sizes, &err); if (err!=CL_SUCCESS) die("d_max_out_sizes");
    cl_mem d_sizes_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t)*totalBlocks, NULL, &err); if (err!=CL_SUCCESS) die("d_sizes_out");

    // Allocate decompressed output buffer
    unsigned char* decompressed = malloc(totalBlocks * BLOCK_SIZE);
    cl_mem d_decompressed = clCreateBuffer(context, CL_MEM_READ_WRITE, totalBlocks * BLOCK_SIZE, NULL, &err); if (err!=CL_SUCCESS) die("d_decompressed");

    // set args for decompress
    err  = clSetKernelArg(k_decompress, 0, sizeof(cl_mem), &d_compressed);
    err |= clSetKernelArg(k_decompress, 1, sizeof(cl_mem), &d_decompressed);
    err |= clSetKernelArg(k_decompress, 2, sizeof(cl_mem), &d_comp_offsets);
    err |= clSetKernelArg(k_decompress, 3, sizeof(cl_mem), &d_comp_sizes);
    err |= clSetKernelArg(k_decompress, 4, sizeof(cl_mem), &d_out_offsets);
    err |= clSetKernelArg(k_decompress, 5, sizeof(cl_mem), &d_max_out_sizes);
    err |= clSetKernelArg(k_decompress, 6, sizeof(cl_mem), &d_sizes_out);
    err |= clSetKernelArg(k_decompress, 7, sizeof(uint32_t), &totalBlocks_i);
    if (err != CL_SUCCESS) die("clSetKernelArg decompress");

    cl_event ev_decomp;
    err = clEnqueueNDRangeKernel(queue, k_decompress, 1, NULL, &global, &local, 0, NULL, &ev_decomp); if (err!=CL_SUCCESS) die("clEnqueueNDRangeKernel decompress");
    clFinish(queue);
    clGetEventProfilingInfo(ev_decomp, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startt, NULL);
    clGetEventProfilingInfo(ev_decomp, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endt, NULL);
    double decomp_ms = (endt - startt) * 1e-6;
    printf("Decompress kernel time: %.3f ms\n", decomp_ms);

    // read decompressed output
    cl_event ev_read_decomp = NULL;
    err = clEnqueueReadBuffer(queue, d_decompressed, CL_FALSE, 0, totalBlocks * BLOCK_SIZE, decompressed, 0, NULL, &ev_read_decomp); if (err!=CL_SUCCESS) die("read decompressed");
    clFinish(queue);
    cl_ulong read_decomp_start=0, read_decomp_end=0;
    if (ev_read_decomp) {
        clGetEventProfilingInfo(ev_read_decomp, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_decomp_start, NULL);
        clGetEventProfilingInfo(ev_read_decomp, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_decomp_end, NULL);
    }
    double device_to_host_decomp_ms = (read_decomp_end - read_decomp_start) * 1e-6;
    printf("Device->Host read decompressed time: %.3f ms\n", device_to_host_decomp_ms);

    // Validate
    int ok = 1;
    for (size_t i = 0; i < src_len; ++i) {
        if (src[i] != decompressed[i]) {
            ok = 0;
            printf("Mismatch at %zu: src=%02x got=%02x\n", i, src[i], decompressed[i]);
            // create ab_results dir if missing and dump failing block for offline analysis
            const char* outdir = "ab_results";
            #ifdef _WIN32
            _mkdir(outdir);
            #else
            mkdir(outdir, 0755);
            #endif
            size_t block_idx = i / BLOCK_SIZE;
            uint32_t orig_size = blockOffsets[block_idx*2 + 1];
            uint32_t comp_size = blockSizes[block_idx];
            uint32_t comp_offset = initialOutputOffsets[block_idx];
            // read compressed bytes from device
            if (comp_size > 0) {
                unsigned char* compbuf = malloc(comp_size);
                if (compbuf) {
                    // blocking read
                    cl_int r = clEnqueueReadBuffer(queue, d_compressed, CL_TRUE, (size_t)comp_offset, comp_size, compbuf, 0, NULL, NULL);
                    if (r == CL_SUCCESS) {
                        char path_comp[512];
                        snprintf(path_comp, sizeof(path_comp), "%s/fail_block_%zu_comp.bin", outdir, block_idx);
                        write_binary_file(path_comp, compbuf, comp_size);
                        printf("Wrote compressed failing block to %s\n", path_comp);
                    } else {
                        printf("Failed to read compressed block from device (err %d)\n", r);
                    }
                    free(compbuf);
                }
            }
            // dump expected and actual decompressed block
            size_t block_src_off = blockOffsets[block_idx*2 + 0];
            char path_exp[512]; char path_got[512];
            snprintf(path_exp, sizeof(path_exp), "%s/fail_block_%zu_exp.bin", outdir, block_idx);
            snprintf(path_got, sizeof(path_got), "%s/fail_block_%zu_got.bin", outdir, block_idx);
            write_binary_file(path_exp, src + block_src_off, orig_size);
            write_binary_file(path_got, decompressed + block_idx * BLOCK_SIZE, orig_size);
            printf("Wrote expected -> %s and actual -> %s\n", path_exp, path_got);
            break;
        }
    }
    printf("Round-trip %s\n", ok ? "OK" : "FAILED");
    printf("Total kernel time: %.3f ms (compress + decompress)\n", comp_ms + decomp_ms);

    // Write per-block CSV stats
    FILE* csv = fopen("gpu_roundtrip_stats.csv", "w");
    if (csv) {
        fprintf(csv, "block,orig_size,comp_size,ratio,comp_offset\n");
        size_t total_orig = 0, total_comp = 0;
        for (size_t i = 0; i < totalBlocks; ++i) {
            uint32_t orig = blockOffsets[i*2 + 1];
            uint32_t comp = blockSizes[i];
            double ratio = comp ? ((double)orig / (double)comp) : 0.0;
            uint32_t comp_offset = initialOutputOffsets[i];
            fprintf(csv, "%zu,%u,%u,%.3f,%u\n", i, orig, comp, ratio, comp_offset);
            total_orig += orig; total_comp += comp;
        }
        fprintf(csv, "TOTAL,%zu,%zu,%.3f,\n", total_orig, total_comp, total_comp ? ((double)total_orig/total_comp) : 0.0);
        fclose(csv);
        printf("Wrote per-block stats to gpu_roundtrip_stats.csv\n");
    }

    // cleanup
    clReleaseMemObject(d_input); clReleaseMemObject(d_compressed); clReleaseMemObject(d_blockOffsets);
    clReleaseMemObject(d_outputOffsets); clReleaseMemObject(d_blockSizes);
    clReleaseMemObject(d_comp_offsets); clReleaseMemObject(d_comp_sizes); clReleaseMemObject(d_out_offsets);
    clReleaseMemObject(d_max_out_sizes); clReleaseMemObject(d_decompressed); clReleaseMemObject(d_sizes_out);
    clReleaseKernel(k_compress); clReleaseKernel(k_decompress); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);

    free(src); free(blockOffsets); free(outputOffsets); free(initialOutputOffsets); free(blockSizes);
    free(comp_offsets); free(comp_sizes); free(out_offsets); free(max_out_sizes); free(decompressed);
    return ok ? 0 : 2;
}
