#ifndef LZ4_GPU_CORE_H
#define LZ4_GPU_CORE_H

#include <CL/cl.h>
#include <stdint.h>
#include "lz4_gpu_protocol.h"

typedef struct {
    cl_mem in_buf;
    cl_mem out_buf;
    cl_mem block_info_buf;
    cl_mem output_size_buf;
    cl_mem dict_buf;
    size_t current_in_capacity;
    size_t current_out_capacity;
    size_t current_blocks_capacity;
    size_t current_osize_capacity;
    size_t current_dict_capacity;
} lz4_gpu_workspace_t;

void lz4_gpu_workspace_init(lz4_gpu_workspace_t* ws);
void lz4_gpu_workspace_free(lz4_gpu_workspace_t* ws);

cl_program lz4_load_program(cl_context context, cl_device_id device, int hash_log);

int lz4_compress_core(cl_context context, cl_command_queue queue, cl_kernel kernel,
                    const char* input_path, const char* output_path,
                    size_t block_size, int acceleration, lz4_gpu_workspace_t* ws,
                    timing_t* t, int local_size, int hash_log);

int lz4_decompress_core(cl_context context, cl_command_queue queue, cl_kernel kernel,
                      const char* input_path, const char* output_path,
                      lz4_gpu_workspace_t* ws, timing_t* timing, int local_size);

#endif
