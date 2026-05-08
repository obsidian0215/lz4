#ifndef LZ4_GPU_UTILS_H
#define LZ4_GPU_UTILS_H

#include <stdint.h>
#include <CL/cl.h>
#include "lz4_gpu_protocol.h"

uint64_t get_us(void);
extern uint64_t g_ocl_init_us;
extern uint64_t g_kernel_load_us;
int lz4_read_file_to_buf(const char* path, void* buf, size_t sz, unsigned long* time_us);
void print_response_stats(const response_t* resp, const char* input_path, int mode);
cl_int lz4_select_opencl_platform_device(cl_platform_id* out_pf, cl_device_id* out_dev);

#endif
