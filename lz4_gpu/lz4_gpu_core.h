#ifndef LZ4_GPU_CORE_H
#define LZ4_GPU_CORE_H

#include <CL/cl.h>
#include <stdint.h>
#include "lz4_gpu_protocol.h"

typedef struct {
    cl_mem in_buf;
    cl_mem comp_in_buf;
    cl_mem out_buf;
    cl_mem packed_out_buf;
    cl_mem block_info_buf;
    cl_mem out_offsets_buf;
    cl_mem packed_offsets_buf;
    cl_mem output_size_buf;
    cl_mem dict_buf;
    cl_mem decomp_comp_off_buf;
    cl_mem decomp_comp_size_buf;
    cl_mem decomp_out_off_buf;
    cl_mem decomp_max_out_buf;
    cl_mem decomp_sizes_out_buf;
    uint32_t comp_epoch_base;
    size_t current_in_capacity;
    size_t current_comp_in_capacity;
    size_t current_out_capacity;
    size_t current_packed_out_capacity;
    size_t current_blocks_capacity;
    size_t current_out_offsets_capacity;
    size_t current_packed_offsets_capacity;
    size_t current_osize_capacity;
    size_t current_dict_capacity;
    size_t current_decomp_comp_off_capacity;
    size_t current_decomp_comp_size_capacity;
    size_t current_decomp_out_off_capacity;
    size_t current_decomp_max_out_capacity;
    size_t current_decomp_sizes_out_capacity;
    uint32_t comp_meta_cached_blocks;
    uint32_t comp_meta_cached_block_size;
    uint32_t comp_meta_cached_single_block_max_out;
    int comp_meta_cached_valid;
} lz4_gpu_workspace_t;

void lz4_gpu_workspace_init(lz4_gpu_workspace_t* ws);
void lz4_gpu_workspace_free(lz4_gpu_workspace_t* ws);

cl_program lz4_load_program(cl_context context, cl_device_id device, int hash_log, size_t block_size);

cl_mem ensure_buffer_ex(cl_context context, cl_mem buf, size_t size, size_t* current_capacity, cl_mem_flags flags, int alloc_host_ptr, cl_int* err);
cl_mem ensure_buffer(cl_context context, cl_mem buf, size_t size, size_t* current_capacity, cl_int* err);
int write_buffer_auto(cl_command_queue queue, cl_mem buf, const void* src, size_t bytes, int standard_copy);
int write_buffer_mapped(cl_command_queue queue, cl_mem buf, const void* src, size_t bytes);
int read_buffer_auto(cl_command_queue queue, cl_mem buf, void* dst, size_t bytes, int standard_copy);

int lz4_compress_core(cl_context context, cl_command_queue queue, cl_kernel kernel,
                    const char* input_path, const char* output_path,
                    size_t block_size, int acceleration, lz4_gpu_workspace_t* ws,
                    timing_t* t, int local_size,
                    int skip_input_upload);

int lz4_decompress_core(cl_context context, cl_command_queue queue, cl_kernel kernel,
                      const char* input_path, const char* output_path,
                      lz4_gpu_workspace_t* ws, timing_t* timing, int local_size);

#endif
