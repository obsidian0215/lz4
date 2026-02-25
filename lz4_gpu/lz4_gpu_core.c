#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "lz4_gpu_core.h"
#include "lz4_gpu_utils.h"

void lz4_gpu_workspace_init(lz4_gpu_workspace_t* ws) {
    memset(ws, 0, sizeof(*ws));
}

void lz4_gpu_workspace_free(lz4_gpu_workspace_t* ws) {
    if (ws->in_buf) clReleaseMemObject(ws->in_buf);
    if (ws->out_buf) clReleaseMemObject(ws->out_buf);
    if (ws->block_info_buf) clReleaseMemObject(ws->block_info_buf);
    if (ws->output_size_buf) clReleaseMemObject(ws->output_size_buf);
    if (ws->dict_buf) clReleaseMemObject(ws->dict_buf);
    memset(ws, 0, sizeof(*ws));
}

static cl_mem ensure_buffer(cl_context context, cl_mem buf, size_t size, size_t* current_capacity, cl_int* err) {
    if (buf && *current_capacity >= size) return buf;
    if (buf) clReleaseMemObject(buf);
    *current_capacity = size;
    /* Use ALLOC_HOST_PTR for zero-copy style mapping */
    return clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, err);
}

static size_t round_up_size(size_t v, size_t align) {
    if (align == 0) return v;
    return ((v + align - 1) / align) * align;
}

static size_t sanitize_local_size(cl_command_queue queue, size_t requested, size_t upper_blocks) {
    if (upper_blocks == 0) return 1;
    size_t l = (requested == 0) ? 1 : requested;

    cl_device_id qdev = NULL;
    size_t max_wg = 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    }

    if (l > max_wg) l = max_wg;
    if (l > upper_blocks) l = upper_blocks;
    if (l == 0) l = 1;

    /* Keep local size as power-of-two for stable occupancy behavior. */
    size_t p2 = 1;
    while ((p2 << 1) <= l) p2 <<= 1;
    return p2;
}

static size_t choose_worker_count(cl_command_queue queue, size_t num_blocks, size_t local_size) {
    if (num_blocks == 0) return 1;

    cl_device_id qdev = NULL;
    cl_uint cu = 1;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(qdev), &qdev, NULL) == CL_SUCCESS && qdev) {
        clGetDeviceInfo(qdev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    }
    if (cu == 0) cu = 1;

    size_t wi_per_cu = 12;
    const char* env = getenv("LZ4_GPU_WI_PER_CU");
    if (env && *env) {
        char* end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed > 0) wi_per_cu = (size_t)parsed;
    }

    size_t target = (size_t)cu * wi_per_cu;
    if (target < local_size) target = local_size;
    if (target > num_blocks) target = num_blocks;
    if (target == 0) target = 1;
    return target;
}

int lz4_compress_core(cl_context context, cl_command_queue queue, cl_kernel kernel,
                    const char* input_path, const char* output_path,
                    size_t block_size, int acceleration, lz4_gpu_workspace_t* ws,
                    timing_t* t, int local_size, int hash_log) {
    cl_int err;
    uint64_t t1, t2;

    t1 = get_us();
    FILE* fin = fopen(input_path, "rb"); if (!fin) return -1;
    fseek(fin, 0, SEEK_END); size_t file_size = ftell(fin); fseek(fin, 0, SEEK_SET);
    if (file_size == 0) { fclose(fin); return -1; }
    t->in_size = (unsigned long)file_size;

    t2 = get_us();
    t->file_read_us = (unsigned long)(t2 - t1);

    t1 = get_us();
    int num_blocks = (file_size + block_size - 1) / block_size;
    size_t l_ws = sanitize_local_size(queue, (local_size > 0) ? (size_t)local_size : 1, (size_t)num_blocks);
    size_t worker_count = choose_worker_count(queue, (size_t)num_blocks, l_ws);
    size_t g_ws = round_up_size(worker_count, l_ws);

    uint32_t* h_block_info = malloc(num_blocks * 2 * sizeof(uint32_t));
    uint32_t* h_out_offsets = malloc(num_blocks * sizeof(uint32_t));
    uint32_t single_block_max_out = (uint32_t)(block_size * 1.1 + 64);
    for (int i = 0; i < num_blocks; i++) {
        h_block_info[i*2] = i * block_size;
        h_block_info[i*2+1] = (i == num_blocks - 1) ? (file_size - i * block_size) : (uint32_t)block_size;
        h_out_offsets[i] = i * single_block_max_out;
    }
    t->nblk = (unsigned long)num_blocks;
    t->blk_size_bytes = (unsigned long)block_size;

    t1 = get_us();
    ws->in_buf = ensure_buffer(context, ws->in_buf, file_size, &ws->current_in_capacity, &err);
    ws->out_buf = ensure_buffer(context, ws->out_buf, (size_t)num_blocks * single_block_max_out, &ws->current_out_capacity, &err);

    /* Zero-copy file read: map in_buf and read directly */
    void* mapped_in = clEnqueueMapBuffer(queue, ws->in_buf, CL_TRUE, CL_MAP_WRITE, 0, file_size, 0, NULL, NULL, &err);
    if (err == CL_SUCCESS) {
        fread(mapped_in, 1, file_size, fin);
        clEnqueueUnmapMemObject(queue, ws->in_buf, mapped_in, 0, NULL, NULL);
    }
    fclose(fin);

    size_t out_offsets_cap = num_blocks * sizeof(uint32_t);
    cl_mem out_offsets_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, out_offsets_cap, h_out_offsets, &err);

    ws->output_size_buf = ensure_buffer(context, ws->output_size_buf, num_blocks * sizeof(uint32_t), &ws->current_osize_capacity, &err);
    size_t dict_size_per_worker = (1ULL << hash_log) * (sizeof(int));
    ws->dict_buf = ensure_buffer(context, ws->dict_buf, g_ws * dict_size_per_worker, &ws->current_dict_capacity, &err);
    t->buffer_alloc_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    cl_mem b_info_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_blocks * 2 * sizeof(uint32_t), h_block_info, &err);
    /* Zero-copy: data_upload_us is now minimal or part of mapping */
    t->data_upload_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    int tableType = (block_size <= 65536) ? 0 : 1;
    int globalIndexBase = 0;
    int inputSize = (int)file_size;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &ws->in_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ws->out_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &ws->output_size_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &b_info_buf);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_offsets_buf);
    clSetKernelArg(kernel, 5, sizeof(int), &num_blocks);
    clSetKernelArg(kernel, 6, sizeof(int), &inputSize);
    clSetKernelArg(kernel, 7, sizeof(int), &tableType);
    clSetKernelArg(kernel, 8, sizeof(int), &acceleration);
    clSetKernelArg(kernel, 9, sizeof(int), &globalIndexBase);
    clSetKernelArg(kernel, 10, sizeof(cl_mem), &ws->dict_buf);

    t1 = get_us();
    t->global_size = (unsigned long)g_ws;
    t->local_size = (unsigned long)l_ws;
    cl_event ev;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &g_ws, &l_ws, 0, NULL, &ev);
    clWaitForEvents(1, &ev);
    t->kernel_exec_us = (unsigned long)(get_us() - t1);
    t->algo_config = hash_log;

    t1 = get_us();
    uint32_t* h_osizes = malloc(num_blocks * sizeof(uint32_t));
    clEnqueueReadBuffer(queue, ws->output_size_buf, CL_TRUE, 0, num_blocks * sizeof(uint32_t), h_osizes, 0, NULL, NULL);

    size_t total_compressed_size = 0;
    for (int i=0; i<num_blocks; i++) total_compressed_size += h_osizes[i];
    t->out_size = total_compressed_size;

    t->download_total_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    FILE* fout = fopen(output_path, "wb");
    if (fout) {
        uint32_t magic = 0x184D2204;
        fwrite(&magic, 1, 4, fout);
        fwrite(&num_blocks, 1, 4, fout);
        uint32_t bsize_u32 = (uint32_t)block_size;
        fwrite(&bsize_u32, 1, 4, fout);
        fwrite(h_osizes, 1, num_blocks * 4, fout);

        size_t total_out_read = (size_t)num_blocks * single_block_max_out;
        uint8_t* h_out = malloc(total_out_read);
        clEnqueueReadBuffer(queue, ws->out_buf, CL_TRUE, 0, total_out_read, h_out, 0, NULL, NULL);

        for (int i = 0; i < num_blocks; i++) {
            size_t offset = h_out_offsets[i];
            fwrite(h_out + offset, 1, h_osizes[i], fout);
        }
        fclose(fout);
        free(h_out);
    }
    t->file_write_us = (unsigned long)(get_us() - t1);

    clReleaseMemObject(out_offsets_buf);
    clReleaseMemObject(b_info_buf);
    free(h_block_info); free(h_osizes); free(h_out_offsets);
    return 0;
}

int lz4_decompress_core(cl_context context, cl_command_queue queue, cl_kernel kernel,
                      const char* input_path, const char* output_path,
                      lz4_gpu_workspace_t* ws, timing_t* t, int local_size) {
    cl_int err;
    uint64_t t1;

    t1 = get_us();
    FILE* fin = fopen(input_path, "rb"); if (!fin) return -1;
    uint32_t magic, num_blocks, block_size_val;
    if (fread(&magic, 1, 4, fin) != 4 || magic != 0x184D2204) { fclose(fin); return -1; }
    if (fread(&num_blocks, 1, 4, fin) != 4) { fclose(fin); return -1; }
    if (fread(&block_size_val, 1, 4, fin) != 4) { fclose(fin); return -1; }

    uint32_t* h_comp_sizes = malloc(num_blocks * sizeof(uint32_t));
    fread(h_comp_sizes, 1, num_blocks * 4, fin);

    size_t header_size = 12 + num_blocks * 4;
    fseek(fin, 0, SEEK_END);
    size_t file_size = ftell(fin);
    size_t data_size = file_size - header_size;
    fclose(fin);
    t->file_read_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    uint64_t* h_comp_offsets = malloc(num_blocks * sizeof(uint64_t));
    uint64_t* h_comp_sizes_64 = malloc(num_blocks * sizeof(uint64_t));
    uint64_t* h_out_offsets = malloc(num_blocks * sizeof(uint64_t));
    uint64_t* h_max_out_sizes = malloc(num_blocks * sizeof(uint64_t));

    uint64_t curr_in = 0;
    uint64_t curr_out = 0;
    uint64_t block_max = block_size_val;

    for (uint32_t i = 0; i < num_blocks; i++) {
        h_comp_offsets[i] = curr_in;
        h_comp_sizes_64[i] = h_comp_sizes[i];
        h_out_offsets[i] = curr_out;
        h_max_out_sizes[i] = block_max;
        curr_in += h_comp_sizes[i];
        curr_out += block_max;
    }
    t->nblk = (unsigned long)num_blocks;
    t->blk_size_bytes = (unsigned long)block_size_val;

    t1 = get_us();
    ws->in_buf = ensure_buffer(context, ws->in_buf, data_size, &ws->current_in_capacity, &err);
    ws->out_buf = ensure_buffer(context, ws->out_buf, num_blocks * block_max, &ws->current_out_capacity, &err);

    /* Zero-copy style: map and read directly */
    void* mapped_in = clEnqueueMapBuffer(queue, ws->in_buf, CL_TRUE, CL_MAP_WRITE, 0, data_size, 0, NULL, NULL, &err);
    if (err == CL_SUCCESS) {
        FILE* fin_data = fopen(input_path, "rb");
        if (fin_data) {
            fseek(fin_data, header_size, SEEK_SET);
            fread(mapped_in, 1, data_size, fin_data);
            fclose(fin_data);
        }
        clEnqueueUnmapMemObject(queue, ws->in_buf, mapped_in, 0, NULL, NULL);
    }

    cl_mem d_comp_off = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_blocks * 8, h_comp_offsets, &err);
    cl_mem d_comp_size = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_blocks * 8, h_comp_sizes_64, &err);
    cl_mem d_out_off = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_blocks * 8, h_out_offsets, &err);
    cl_mem d_max_out = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_blocks * 8, h_max_out_sizes, &err);
    cl_mem d_sizes_out = clCreateBuffer(context, CL_MEM_READ_WRITE, num_blocks * 4, NULL, &err);
    t->buffer_alloc_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    /* Zero-copy: data_upload_us is now minimal */
    t->data_upload_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &ws->in_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &ws->out_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_comp_off);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_comp_size);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_out_off);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_max_out);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_sizes_out);
    clSetKernelArg(kernel, 7, sizeof(uint32_t), &num_blocks);

    size_t l_ws = sanitize_local_size(queue, (local_size > 0) ? (size_t)local_size : 1, (size_t)num_blocks);
    size_t g_ws = round_up_size((size_t)num_blocks, l_ws);
    t->global_size = (unsigned long)g_ws;
    t->local_size = (unsigned long)l_ws;
    cl_event ev;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &g_ws, &l_ws, 0, NULL, &ev);
    clWaitForEvents(1, &ev);
    t->kernel_exec_us = (unsigned long)(get_us() - t1);
    t->algo_config = 0; // Not applicable for LZ4 decompress

    t1 = get_us();
    uint32_t* h_final_sizes = malloc(num_blocks * 4);
    clEnqueueReadBuffer(queue, d_sizes_out, CL_TRUE, 0, num_blocks * 4, h_final_sizes, 0, NULL, NULL);

    size_t total_decomp_sz = 0;
    for (uint32_t i = 0; i < num_blocks; i++) total_decomp_sz += h_final_sizes[i];
    t->out_size = total_decomp_sz;

    t->download_total_us = (unsigned long)(get_us() - t1);

    t1 = get_us();
    uint8_t* h_out = malloc(num_blocks * block_max);
    clEnqueueReadBuffer(queue, ws->out_buf, CL_TRUE, 0, num_blocks * block_max, h_out, 0, NULL, NULL);

    FILE* fout = fopen(output_path, "wb");
    if (fout) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            fwrite(h_out + h_out_offsets[i], 1, h_final_sizes[i], fout);
        }
        fclose(fout);
    }
    t->file_write_us = (unsigned long)(get_us() - t1);

    clReleaseMemObject(d_comp_off); clReleaseMemObject(d_comp_size);
    clReleaseMemObject(d_out_off); clReleaseMemObject(d_max_out);
    clReleaseMemObject(d_sizes_out);
    free(h_comp_sizes); free(h_comp_offsets); free(h_comp_sizes_64);
    free(h_out_offsets); free(h_max_out_sizes); free(h_final_sizes); free(h_out);
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

cl_program lz4_load_program(cl_context context, cl_device_id device, int hash_log) {
    char bin_name[128];
    snprintf(bin_name, sizeof(bin_name), "/root/lz4/lz4_gpu/lz4_gpu_v3_%d.clbin", hash_log);
    size_t sz = 0;
    char* bin = read_file_bin_local(bin_name, &sz);
    if (bin) {
        cl_int status, err;
        cl_program prog = clCreateProgramWithBinary(context, 1, &device, &sz, (const unsigned char**)&bin, &status, &err);
        free(bin);
        if (err == CL_SUCCESS) {
            clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
            return prog;
        }
    }

    // Fallback: load from source
    const char* filename = "/root/lz4/lz4_gpu/lz4_gpu.cl";
    FILE* f = fopen(filename, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); size_t s_sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* src = malloc(s_sz + 1); fread(src, 1, s_sz, f); src[s_sz] = 0; fclose(f);
    cl_int err;
    cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&src, &s_sz, &err);
    free(src);
    char flags[128];
    snprintf(flags, sizeof(flags), "-I. -DLZ4_HASHLOG=%d", hash_log);
    err = clBuildProgram(prog, 1, &device, flags, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char* log = malloc(log_sz); clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        fprintf(stderr, "Build Error: %s\n", log); free(log);
        return NULL;
    }
    return prog;
}
