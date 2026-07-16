#ifndef LZ4_GPU_PROTOCOL_H
#define LZ4_GPU_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>
#include "timing.h"

#define SOCKET_PATH "/tmp/lz4_gpu_daemon.sock"
#define LZ4_DAEMON_REQUEST_MAGIC 0x4c5a3447u
#define LZ4_DAEMON_REQUEST_VERSION 3u
#define LZ4_DAEMON_FLAG_RAW_BUFFER 0x1u

extern uint64_t g_ocl_init_us;
extern uint64_t g_kernel_load_us;

enum {
    mode_compress = 0,
    mode_decompress = 1
};

typedef struct {
    uint32_t magic;
    uint32_t version;
    int mode;
    int acceleration;
    int block_size;
    int local_size;
    int hash_log;
    uint32_t flags;
    size_t input_size;
    char input_path[1024];
    char output_path[1024];
} request_t;

typedef struct {
    int status;
    char message[256];
    unsigned long time_us;
    size_t out_size;
    timing_t timing;
} response_t;

#endif
