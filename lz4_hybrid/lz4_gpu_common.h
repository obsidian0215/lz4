#ifndef LZ4_GPU_COMMON_H
#define LZ4_GPU_COMMON_H

#include <stdint.h>
#include <string.h>
#include "xxhash.h"

static inline uint32_t LZ4_read32(const void* ptr) {
    uint32_t val;
    memcpy(&val, ptr, 4);
    return val;
}

static inline void LZ4_write32(void* ptr, uint32_t val) {
    memcpy(ptr, &val, 4);
}

static inline unsigned char LZ4F_headerChecksum(const void* header, size_t length) {
    uint32_t xxh = XXH32(header, length, 0);
    return (unsigned char)(xxh >> 8);
}

#endif /* LZ4_GPU_COMMON_H */
