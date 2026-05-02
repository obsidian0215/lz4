#ifndef LZ4_GPU_TIMING_H
#define LZ4_GPU_TIMING_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    unsigned long file_read_us;
    unsigned long ocl_setup_us;
    unsigned long buffer_alloc_us;
    unsigned long data_upload_us;
    unsigned long kernel_exec_us;
    unsigned long download_total_us;
    unsigned long file_write_us;

    unsigned long in_size;
    unsigned long out_size;
    unsigned long blk_size_bytes;
    unsigned long nblk;
    unsigned long global_size;
    unsigned long local_size;
    int algo_config;
} timing_t;

static inline void print_us_tag(FILE *f, const char *tag, unsigned long us) {
    if (!f) return;
    if (us == 0) fprintf(f, "%-22s : %8.3f ms\n", tag, 0.0);
    else if (us >= 1000ul) fprintf(f, "%-22s : %8.3f ms\n", tag, us / 1000.0);
    else fprintf(f, "%-22s : %8lu us\n", tag, us);
}

#endif
