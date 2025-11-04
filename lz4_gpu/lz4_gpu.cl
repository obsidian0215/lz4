/**
 * LZ4 GPU Parallel Compression Implementation
 * Direct port from LZ4 v1.10.0 compression algorithm
 * Copyright (c) Yann Collet. BSD 2-Clause License
 */

// Core LZ4 Constants - EXACT port from original LZ4 v1.10.0
#define MINMATCH 4
#define WILDCOPYLENGTH 8
#define LASTLITERALS 5
#define MFLIMIT 12
#define MATCH_SAFEGUARD_DISTANCE ((2*WILDCOPYLENGTH) - MINMATCH)
#define FASTLOOP_SAFE_DISTANCE 64
#define LZ4_DISTANCE_MAX 65535
#define LZ4_DISTANCE_ABSOLUTE_MAX 65535
#define LZ4_HASHLOG 14
#define LZ4_MEMORY_USAGE 14
#define LZ4_64Klimit ((64 * 1024) + (MFLIMIT-1))
#define LZ4_minLength (MFLIMIT+1)
#define ML_BITS 4
#define ML_MASK ((1U<<ML_BITS)-1)
#define RUN_BITS (8-ML_BITS)
#define RUN_MASK ((1U<<RUN_BITS)-1)

// Memory access functions - Direct port from LZ4 v1.10.0
uint LZ4_read32(const __global unsigned char* ptr) {
    // Little-endian read - OpenCL is little-endian on most platforms
    return (uint)ptr[0] | ((uint)ptr[1] << 8) | ((uint)ptr[2] << 16) | ((uint)ptr[3] << 24);
}

ulong LZ4_read64(const __global unsigned char* ptr) {
    // Compose 64-bit little-endian from two 32-bit reads to avoid unaligned 64-bit loads
    uint lo = LZ4_read32(ptr);
    uint hi = LZ4_read32(ptr + 4);
    return ((ulong)hi << 32) | (ulong)lo;
}

/* Kernel debug macro controls. Define LZ4_GPU_KERNEL_DEBUG when building the
 * program to enable verbose kernel printf(). By default these macros compile
 * out to nothing to avoid excessive kernel-side stdout.
 */
#ifdef LZ4_GPU_KERNEL_DEBUG
#define KDEBUG(...) printf(__VA_ARGS__)
#else
#define KDEBUG(...) do {} while(0)
#endif

/* Errors in kernel are still useful to print */
#define KERROR(...) printf(__VA_ARGS__)

uint LZ4_readLE32(const __global unsigned char* ptr) {
    return LZ4_read32(ptr); // OpenCL is little-endian
}

ushort LZ4_read16(const __global unsigned char* ptr) {
    return (ushort)ptr[0] | ((ushort)ptr[1] << 8);
}

ushort LZ4_readLE16(const __global unsigned char* ptr) {
    return LZ4_read16(ptr);
}

void LZ4_write16(__global unsigned char* ptr, ushort value) {
    ptr[0] = (unsigned char)value;
    ptr[1] = (unsigned char)(value >> 8);
}

void LZ4_write32(__global unsigned char* ptr, uint value) {
    ptr[0] = (unsigned char)value;
    ptr[1] = (unsigned char)(value >> 8);
    ptr[2] = (unsigned char)(value >> 16);
    ptr[3] = (unsigned char)(value >> 24);
}

void LZ4_writeLE32(__global unsigned char* ptr, uint value) {
    LZ4_write32(ptr, value); // OpenCL is little-endian
}

void LZ4_writeLE64(__global unsigned char* ptr, ulong value) {
    ptr[0] = (unsigned char)value;
    ptr[1] = (unsigned char)(value >> 8);
    ptr[2] = (unsigned char)(value >> 16);
    ptr[3] = (unsigned char)(value >> 24);
    ptr[4] = (unsigned char)(value >> 32);
    ptr[5] = (unsigned char)(value >> 40);
    ptr[6] = (unsigned char)(value >> 48);
    ptr[7] = (unsigned char)(value >> 56);
}

// Hash functions - Direct port from LZ4 v1.10.0
uint LZ4_hash4(uint sequence, int tableType) {
    if (tableType == 0) // byU16
        return ((sequence * 2654435761U) >> ((MINMATCH*8)-(LZ4_HASHLOG+1)));
    else // byU32
        return ((sequence * 2654435761U) >> ((MINMATCH*8)-LZ4_HASHLOG));
}

uint LZ4_hash5(ulong sequence, int tableType) {
    const uint hashLog = (tableType == 0) ? LZ4_HASHLOG+1 : LZ4_HASHLOG;
    const ulong prime5bytes = 889523592379ULL;
    return (uint)(((sequence << 24) * prime5bytes) >> (64 - hashLog));
}

uint LZ4_hashPosition(const __global unsigned char* p, int tableType) {
    // Direct port from LZ4 v1.10.0 LZ4_hashPosition function
    // Use safe reads: avoid unaligned 64-bit pointer dereference which can be
    // undefined on many OpenCL devices. Compose 64-bit value from two 32-bit reads.
    if ((sizeof(size_t) == 8) && (tableType != 0)) { // not byU16 - use 64-bit hash
        ulong sequence = LZ4_read64(p);
        return LZ4_hash5(sequence, tableType);
    } else {
        uint sequence = LZ4_read32(p);
        return LZ4_hash4(sequence, tableType);
    }
}

// Hash table operations - Direct port from LZ4 v1.10.0
void LZ4_putIndexOnHash(uint idx, uint h, __local uint* tableBase, int tableType, uint hashTableSize) {
    switch (tableType) {
    default:
    case 0: // byU16
        {
            __local unsigned short* hashTable = (__local unsigned short*)tableBase;
            uint hashIndex = h & (hashTableSize - 1);
            hashTable[hashIndex] = (unsigned short)idx;
            return;
        }
    case 1: // byU32
        {
            uint hashIndex = h & (hashTableSize - 1);
            tableBase[hashIndex] = idx;
            return;
        }
    }
}

uint LZ4_getIndexOnHash(uint h, __local uint* tableBase, int tableType, uint hashTableSize) {
    switch (tableType) {
    default:
    case 0: // byU16
        {
            __local unsigned short* hashTable = (__local unsigned short*)tableBase;
            uint hashIndex = h & (hashTableSize - 1);
            return (uint)hashTable[hashIndex];
        }
    case 1: // byU32
        {
            uint hashIndex = h & (hashTableSize - 1);
            return tableBase[hashIndex];
        }
    }
    return 0;
}

void LZ4_clearHash(uint h, __local uint* tableBase, int tableType, uint hashTableSize) {
    switch (tableType) {
    default:
    case 0: // byU16
        {
            __local unsigned short* hashTable = (__local unsigned short*)tableBase;
            uint hashIndex = h & (hashTableSize - 1);
            hashTable[hashIndex] = 0;
            return;
        }
    case 1: // byU32
        {
            uint hashIndex = h & (hashTableSize - 1);
            tableBase[hashIndex] = 0;
            return;
        }
    }
}

// Common bytes calculation - Direct port from LZ4 v1.10.0
uint LZ4_NbCommonBytes(uint val) {
    // Count number of matching bytes from least-significant byte upward
    // given diff = x ^ y. Returns 0..3 (number of identical low-order bytes).
    if (val == 0) return 4;
    uint n = 0;
    if ((val & 0xFF) == 0) { n++; val >>= 8; }
    if ((val & 0xFF) == 0) { n++; val >>= 8; }
    if ((val & 0xFF) == 0) { n++; val >>= 8; }
    return n;
}

uint LZ4_count(const __global unsigned char* p1, const __global unsigned char* p2, const __global unsigned char* pLimit) {
    const __global unsigned char* pStart = p1;

    if (p1 < pLimit - 3) {
        uint diff = LZ4_read32(p1) ^ LZ4_read32(p2);
        if (diff != 0) {
            return LZ4_NbCommonBytes(diff) + (uint)(p1 - pStart);
        }
        p1 += 4;
        p2 += 4;
    }

    // Fast path: compare 8 bytes at a time using safe 64-bit composed reads
    while (p1 < pLimit - 7) {
        ulong diff64 = LZ4_read64(p1) ^ LZ4_read64(p2);
        if (diff64 != 0) {
            uint lo = (uint)diff64;
            if (lo != 0) {
                return LZ4_NbCommonBytes(lo) + (uint)(p1 - pStart);
            } else {
                uint hi = (uint)(diff64 >> 32);
                return (4 + LZ4_NbCommonBytes(hi) + (uint)(p1 - pStart));
            }
        }
        p1 += 8;
        p2 += 8;
    }

    // Fallback to 4-byte comparisons
    while (p1 < pLimit - 3) {
        uint diff = LZ4_read32(p1) ^ LZ4_read32(p2);
        if (diff != 0) {
            return LZ4_NbCommonBytes(diff) + (uint)(p1 - pStart);
        }
        p1 += 4;
        p2 += 4;
    }

    if (p1 < pLimit - 1 && LZ4_read16(p1) == LZ4_read16(p2)) {
        p1 += 2;
        p2 += 2;
    }

    if (p1 < pLimit && *p1 == *p2) {
        p1++;
    }

    return (uint)(p1 - pStart);
}

// Memory copy functions - EXACT port from LZ4 v1.10.0
void LZ4_memcpy(__global unsigned char* dstPtr, const __global unsigned char* srcPtr, size_t length) {
#ifdef LZ4_GPU_VECTOR_IO
    // Use vector loads/stores where possible; vload/vstore do not require alignment
    while (length >= 16) {
        uchar16 v = vload16(0, (const __global uchar*)srcPtr);
        vstore16(v, 0, (__global uchar*)dstPtr);
        srcPtr += 16; dstPtr += 16; length -= 16;
    }
    if (length >= 8) {
        uchar8 v8 = vload8(0, (const __global uchar*)srcPtr);
        vstore8(v8, 0, (__global uchar*)dstPtr);
        srcPtr += 8; dstPtr += 8; length -= 8;
    }
    for (size_t i = 0; i < length; ++i) dstPtr[i] = srcPtr[i];
#else
    for (size_t i = 0; i < length; ++i) dstPtr[i] = srcPtr[i];
#endif
}

/* Safe wild copy that avoids potentially misaligned 64-bit loads/stores */
void LZ4_wildCopy8(__global unsigned char* dstPtr, const __global unsigned char* srcPtr, __global unsigned char* dstEnd) {
    #ifdef LZ4_GPU_VECTOR_IO
    do {
        uchar8 v8 = vload8(0, (const __global uchar*)srcPtr);
        vstore8(v8, 0, (__global uchar*)dstPtr);
        dstPtr += 8; srcPtr += 8;
    } while (dstPtr < dstEnd);
    #else
    do {
        for (int i = 0; i < 8; ++i) dstPtr[i] = srcPtr[i];
        dstPtr += 8; srcPtr += 8;
    } while (dstPtr < dstEnd);
    #endif
}

void LZ4_wildCopy32(__global unsigned char* dstPtr, const __global unsigned char* srcPtr, __global unsigned char* dstEnd) {
    #ifdef LZ4_GPU_VECTOR_IO
    do {
        // copy 32 bytes per loop with two 16-byte vector ops
        uchar16 v0 = vload16(0, (const __global uchar*)srcPtr);
        uchar16 v1 = vload16(0, (const __global uchar*)(srcPtr + 16));
        vstore16(v0, 0, (__global uchar*)dstPtr);
        vstore16(v1, 0, (__global uchar*)(dstPtr + 16));
        dstPtr += 32; srcPtr += 32;
    } while (dstPtr < dstEnd);
    #else
    do {
        for (int i = 0; i < 32; ++i) dstPtr[i] = srcPtr[i];
        dstPtr += 32; srcPtr += 32;
    } while (dstPtr < dstEnd);
    #endif
}

void LZ4_memmove(__global unsigned char* dstPtr, const __global unsigned char* srcPtr, size_t length) {
    if (dstPtr <= srcPtr) {
        for (size_t i = 0; i < length; ++i) {
            dstPtr[i] = srcPtr[i];
        }
    } else {
        for (size_t i = length; i != 0; --i) {
            dstPtr[i - 1] = srcPtr[i - 1];
        }
    }
}

// Memory copy using offset - EXACT port from LZ4 v1.10.0
//void LZ4_memcpy_using_offset(__global unsigned char* dstPtr, const __global unsigned char* srcPtr, __global unsigned char* dstEnd, uint offset) {
//    const unsigned inc32table[8] = {0, 1, 2, 1, 0, 4, 4, 4};
//    const int      dec64table[8] = {0, 0, 0, -1, -4, 1, 2, 3};

//    if (offset < 8) {
        // Copy first 4 bytes individually
//        dstPtr[0] = srcPtr[0];
//        dstPtr[1] = srcPtr[1];
//        dstPtr[2] = srcPtr[2];
//        dstPtr[3] = srcPtr[3];

        // Handle next 4 bytes with table lookup
//        srcPtr += inc32table[offset];
//        *(__global uint*)(dstPtr + 4) = *(__global uint*)srcPtr;
//        srcPtr -= dec64table[offset];
//        dstPtr += 8;
//        srcPtr += 8;

        // Use wild copy for remaining bytes
//        LZ4_wildCopy8(dstPtr, srcPtr, dstEnd);
//    } else {
        // Large offset - direct copy
//        *(__global ulong*)dstPtr = *(__global ulong*)srcPtr;
//        dstPtr += 8;
//        srcPtr += 8;
//        LZ4_wildCopy8(dstPtr, srcPtr, dstEnd);
//    }
//}

// Forward declaration for accelerated core to avoid implicit declaration errors
int lz4_compress_core_accelerated(
    __global const unsigned char* src,
    __global unsigned char* dst,
    int srcSize,
    int dstCapacity,
    int tableType,
    __local unsigned int* hashTable,
    int acceleration
);

// Core compression function - Direct port from LZ4 v1.10.0
int lz4_compress_core(
    __global const unsigned char* src,
    __global unsigned char* dst,
    int srcSize,
    int dstCapacity,
    int tableType,
    __local unsigned int* hashTable
) {
    // 调用带acceleration参数的函数，acceleration设为默认值1
    return lz4_compress_core_accelerated(src, dst, srcSize, dstCapacity, tableType, hashTable, 1);
}

// Core compression function with acceleration support - Enhanced LZ4 GPU implementation
int lz4_compress_core_accelerated(
    __global const unsigned char* src,
    __global unsigned char* dst,
    int srcSize,
    int dstCapacity,
    int tableType,
    __local unsigned int* hashTable,
    int acceleration
) {
    const int hashTableSize = 1 << LZ4_HASHLOG;
    const __global unsigned char* ip = src;
    __global unsigned char* op = dst;
    __global unsigned char* const oend = op + dstCapacity;

    const __global unsigned char* const iend = ip + srcSize;
    const __global unsigned char* anchor = ip;
    const __global unsigned char* const mflimitPlusOne = iend - MFLIMIT + 1;
    const __global unsigned char* const matchlimit = iend - LASTLITERALS;


    // Early exit for small inputs: encode as a literal-only sequence (valid LZ4 block)
    if (srcSize < LZ4_minLength) {
        uint litLength = (uint)(iend - anchor);
        // token + possible ext + literals
        if (op + 1 + litLength + (litLength/255) + 1 > oend) {
            return 0; // output overflow
        }
        if (litLength >= RUN_MASK) {
            uint len = litLength - RUN_MASK;
            *op++ = (RUN_MASK << ML_BITS);
            for (; len >= 255; len -= 255) {
                *op++ = 255;
            }
            *op++ = (unsigned char)len;
        } else {
            *op++ = (unsigned char)(litLength << ML_BITS);
        }
        for (uint i = 0; i < litLength; ++i) {
            *op++ = anchor[i];
        }
        return (int)(op - dst);
    }

    // Initialize hash table
    uint startIndex = 0;
    const __global unsigned char* base = src - startIndex;

    // 清空本地hash table
    for (int i = 0; i < hashTableSize; i++) {
        hashTable[i] = 0;
    }

    // First byte
    if (ip < iend) {
        uint h = LZ4_hashPosition(ip, tableType);
        LZ4_putIndexOnHash(startIndex, h, hashTable, tableType, hashTableSize);
        ip++;
    }

    uint forwardH = (ip < iend) ? LZ4_hashPosition(ip, tableType) : 0;

    // Main compression loop
    for (;;) {
        const __global unsigned char* match;
        __global unsigned char* token;
        const __global unsigned char* filledIp;
        uint matchLength = 0; // Declare matchLength at the beginning

        // Find a match with acceleration support
        if (ip + MINMATCH <= iend - LASTLITERALS) {
            uint h = forwardH;
            uint current = (uint)(ip - base);

            // Acceleration search logic - ported from CPU version
            if (acceleration <= 1) {
                // Standard search (acceleration = 1)
                uint matchIndex = LZ4_getIndexOnHash(h, hashTable, tableType, hashTableSize);

                if (matchIndex < current && (current - matchIndex) <= LZ4_DISTANCE_MAX) {
                    match = base + matchIndex;

                    // Check if match is valid
                    if (match >= base && match + MINMATCH <= iend) {
                        uint matchVal = LZ4_read32(match);
                        uint ipVal = LZ4_read32(ip);

                        if (matchVal == ipVal) {
                            // Find match length
                            if (ip + MINMATCH <= matchlimit && match + MINMATCH <= matchlimit) {
                                uint matchCode = LZ4_count(ip + MINMATCH, match + MINMATCH, matchlimit);
                                matchLength = matchCode + MINMATCH;

                                // Found a match!
                                goto _encode_match;
                            }
                        }
                    }
                }
            } else {
                // Accelerated search - skip positions for faster compression
                const int LZ4_skipTrigger = 6;  // Same as CPU version
                int searchMatchNb = acceleration << LZ4_skipTrigger;
                int step = 1;
                int foundMatch = 0;

                // Accelerated search loop
                for (int searchStep = 0; searchStep < searchMatchNb && !foundMatch; searchStep += step) {
                    uint matchIndex = LZ4_getIndexOnHash(h, hashTable, tableType, hashTableSize);

                    if (matchIndex < current && (current - matchIndex) <= LZ4_DISTANCE_MAX) {
                        match = base + matchIndex;

                        // Check if match is valid
                        if (match >= base && match + MINMATCH <= iend) {
                            uint matchVal = LZ4_read32(match);
                            uint ipVal = LZ4_read32(ip);

                            if (matchVal == ipVal) {
                                // Find match length
                                if (ip + MINMATCH <= matchlimit && match + MINMATCH <= matchlimit) {
                                    uint matchCode = LZ4_count(ip + MINMATCH, match + MINMATCH, matchlimit);
                                    uint candidateLength = matchCode + MINMATCH;

                                    // Accept this match if it's long enough or we're in aggressive mode
                                    if (candidateLength >= MINMATCH + (acceleration >> 2)) {
                                        matchLength = candidateLength;
                                        foundMatch = 1;
                                        goto _encode_match;
                                    }
                                }
                            }
                        }
                    }

                    // Update step size (same logic as CPU version)
                    step = (searchMatchNb >> LZ4_skipTrigger);
                    h = LZ4_hashPosition(ip + step, tableType);
                    ip += step;
                }

                // If we did accelerated search but found nothing, fall back to position 0
                if (!foundMatch) {
                    ip = src + current;  // Reset position
                    h = LZ4_hashPosition(ip, tableType);
                }
            }
        }

        // No match found, advance
        if (ip >= mflimitPlusOne) break;
        LZ4_putIndexOnHash((uint)(ip - base), forwardH, hashTable, tableType, hashTableSize);
        forwardH = LZ4_hashPosition(ip + 1, tableType);
        ip++;
        continue;

    _encode_match:
        // Catch up
        filledIp = ip;
        if ((match > base) && (ip[-1] == match[-1])) {
            do { ip--; match--; } while (((ip > anchor) & (match > base)) && (ip[-1] == match[-1]));
        }

        // Encode literals
        uint litLength = (uint)(ip - anchor);
        token = op++;

        if (op + litLength + 2 > oend) {
            return 0; // output buffer overflow
        }

        // Fix: Properly encode literal length according to LZ4 specification
        if (litLength >= RUN_MASK) {
            uint len = litLength - RUN_MASK;
            *token = (RUN_MASK << ML_BITS);
            for(; len >= 255; len -= 255) {
                if (op >= oend) return 0;
                *op++ = 255;
            }
            if (op >= oend) return 0;
            *op++ = (unsigned char)len;
        } else {
            *token = (unsigned char)(litLength << ML_BITS);
        }

        // Copy literals
        for (uint i = 0; i < litLength; i++) {
            if (op >= oend) return 0;
            *op++ = anchor[i];
        }

        // Encode offset
        uint offset = (uint)(ip - match);
        if (op + 2 > oend) return 0;
        LZ4_write16(op, (ushort)offset);
        op += 2;

        // Encode match length
        uint matchCode = matchLength - MINMATCH;
        if (matchCode >= ML_MASK) {
            *token += ML_MASK;
            matchCode -= ML_MASK;
            while (matchCode >= 255) {
                if (op >= oend) return 0;
                *op++ = 255;
                matchCode -= 255;
            }
            if (op >= oend) return 0;
            *op++ = (unsigned char)matchCode;
        } else {
            *token += (unsigned char)matchCode;
        }
        ip += matchLength;
        anchor = ip;

        // Test end of chunk
        if (ip >= mflimitPlusOne) break;

        // Fill table
        uint h = LZ4_hashPosition(ip - 2, tableType);
        LZ4_putIndexOnHash((uint)(ip - 2 - base), h, hashTable, tableType, hashTableSize);

        // Optional fast path for immediate next match (disabled by default for correctness parity)
        // Define LZ4_GPU_ENABLE_SECOND_MATCH to enable.
#ifdef LZ4_GPU_ENABLE_SECOND_MATCH
        {
            uint h2 = LZ4_hashPosition(ip, tableType);
            uint current2 = (uint)(ip - base);
            uint matchIndex2 = LZ4_getIndexOnHash(h2, hashTable, tableType, hashTableSize);
            if (matchIndex2 < current2 && (current2 - matchIndex2) <= LZ4_DISTANCE_MAX) {
                const __global unsigned char* match2 = base + matchIndex2;
                if (match2 >= base && match2 + MINMATCH <= iend && LZ4_read32(match2) == LZ4_read32(ip)) {
                    // Found another match
                    token = op++;
                    *token = 0;
                    // Encode zero literals and match
                    uint offset2 = (uint)(ip - match2);
                    if (op + 2 > oend) return 0;
                    LZ4_write16(op, (ushort)offset2);
                    op += 2;

                    // Calculate match length
                    uint matchCode2 = LZ4_count(ip + MINMATCH, match2 + MINMATCH, matchlimit);
                    uint matchLength2 = matchCode2 + MINMATCH;

                    if (matchCode2 >= ML_MASK) {
                        *token += ML_MASK;
                        matchCode2 -= ML_MASK;
                        while (matchCode2 >= 255) {
                            if (op >= oend) return 0;
                            *op++ = 255;
                            matchCode2 -= 255;
                        }
                        if (op >= oend) return 0;
                        *op++ = (unsigned char)matchCode2;
                    } else {
                        *token += (unsigned char)matchCode2;
                    }

                    ip += matchLength2;
                    anchor = ip;
                    goto _next_match_end;
                }
            }
        }
#endif

    // Prepare next loop: resume at current ip
    forwardH = LZ4_hashPosition(ip, tableType);
    continue;

    _next_match_end:
        // Prepare next loop: resume at current ip
        forwardH = LZ4_hashPosition(ip, tableType);
    }

    // _last_literals:
    // Encode last literals
    uint lastRun = (uint)(iend - anchor);
    if (op + lastRun + 1 > oend) {
        return 0; // output buffer overflow
    }

    if (lastRun >= RUN_MASK) {
        uint accumulator = lastRun - RUN_MASK;
        *op++ = RUN_MASK << ML_BITS;
        for(; accumulator >= 255; accumulator -= 255) {
            if (op >= oend) return 0;
            *op++ = 255;
        }
        if (op >= oend) return 0;
        *op++ = (unsigned char)accumulator;
    } else {
        *op++ = (unsigned char)(lastRun << ML_BITS);
    }

    for (uint i = 0; i < lastRun; i++) {
        if (op >= oend) return 0;
        *op++ = anchor[i];
    }

    int finalSize = (int)(op - dst);
    // printf("DEBUG: Compression complete, final size=%d, dst=%p, op=%p\n", finalSize, dst, op);

    // Debug: Print the actual compressed data
    // printf("DEBUG: Compressed data:");
    // int debug_count = finalSize < 16 ? finalSize : 16;
    // for (int i = 0; i < debug_count; i++) {
        // printf(" %02x", dst[i]);
    // }
    // printf("\n");

    return finalSize;
}

// Main OpenCL kernel - Parallel LZ4 compression with per-work-item block parameters
__kernel void lz4_compress_block(
    __global const unsigned char* input,
    __global unsigned char* output,
    __global unsigned int* blockSizes,
    __global const unsigned int* blockOffsets,  // [offset, size] pairs for each block
    __global const unsigned int* outputOffsets,  // Output offset for each block
    int totalBlocks,
    int inputSize,
    int tableType
) {
    int gid = get_global_id(0);

    if (gid >= totalBlocks) {
        return;
    }

    // 从blockOffsets获取块的起始位置和大小
    int blockOffsetIndex = gid * 2;
    int start = blockOffsets[blockOffsetIndex];      // 块的起始偏移量
    int blockSize = blockOffsets[blockOffsetIndex + 1];  // 块的大小

    // 确保块范围有效
    if (start >= inputSize || blockSize <= 0) {
        blockSizes[gid] = 0;
        return;
    }

    int end = start + blockSize;
    if (end > inputSize) {
        end = inputSize;
    }

    // LZ4压缩实现
    int srcSize = end - start;
    // 增加dstCapacity的估计，防止缓冲区溢出
    int dstCapacity = srcSize + (srcSize/255) + 64; // 增加额外空间
    int outputOffset = outputOffsets[gid];  // 从outputOffsets获取输出偏移量
    __global unsigned char* dst = output + outputOffset;

    // 调试：打印块信息
    if (gid == 0) {
        KDEBUG("DEBUG: First block - start=%d, blockSize=%d, srcSize=%d, dstCapacity=%d\n",
               start, blockSize, srcSize, dstCapacity);
    }

    // 压缩块 - 每个工作项分配本地hash table
    const int hashTableSize = 1 << LZ4_HASHLOG;
    __local unsigned int localHashTable[1 << LZ4_HASHLOG];

    // 同步确保所有工作项都分配了本地内存
    barrier(CLK_LOCAL_MEM_FENCE);

    // 调试：检查输入数据
    if (gid < 3) { // 只检查前3个块
        uint first4 = LZ4_read32(input + start);
        KDEBUG("DEBUG: Block %d: start=%d, srcSize=%d, first4=%08x\n",
               gid, start, srcSize, first4);
    }

    int compressedSize = lz4_compress_core_accelerated(
        input + start,
        dst,
        srcSize,
        dstCapacity,
        tableType,
        localHashTable,
        1  // Default acceleration value for now - will be made configurable
    );

    // 调试：检查压缩结果
    if (gid < 3 && compressedSize > 0) { // 只检查前3个块
        uint first4_compressed = LZ4_read32(dst);
        KDEBUG("DEBUG: Block %d: compressedSize=%d, first4_compressed=%08x\n",
               gid, compressedSize, first4_compressed);
    }

    // 记录块大小
    blockSizes[gid] = compressedSize;
}

// Enhanced OpenCL kernel with acceleration support
__kernel void lz4_compress_block_accelerated(
    __global const unsigned char* input,
    __global unsigned char* output,
    __global unsigned int* blockSizes,
    __global const unsigned int* blockOffsets,
    __global const unsigned int* outputOffsets,
    int totalBlocks,
    int inputSize,
    int tableType,
    int acceleration  // New acceleration parameter
) {
    int gid = get_global_id(0);

    if (gid >= totalBlocks) {
        return;
    }

    // 从blockOffsets获取块的起始位置和大小
    int blockOffsetIndex = gid * 2;
    int start = blockOffsets[blockOffsetIndex];      // 块的起始偏移量
    int blockSize = blockOffsets[blockOffsetIndex + 1];  // 块的大小

    // 确保块范围有效
    if (start >= inputSize || blockSize <= 0) {
        blockSizes[gid] = 0;
        return;
    }

    int end = start + blockSize;
    if (end > inputSize) {
        end = inputSize;
    }

    // LZ4压缩实现
    int srcSize = end - start;
    // 增加dstCapacity的估计，防止缓冲区溢出
    int dstCapacity = srcSize + (srcSize/255) + 64; // 增加额外空间
    int outputOffset = outputOffsets[gid];  // 从outputOffsets获取输出偏移量
    __global unsigned char* dst = output + outputOffset;

    // 调试：打印块信息
    if (gid == 0) {
        KDEBUG("DEBUG: First block - start=%d, blockSize=%d, srcSize=%d, dstCapacity=%d, acceleration=%d\n",
               start, blockSize, srcSize, dstCapacity, acceleration);
    }

    // 压缩块 - 每个工作项分配本地hash table
    const int hashTableSize = 1 << LZ4_HASHLOG;
    __local unsigned int localHashTable[1 << LZ4_HASHLOG];

    // 同步确保所有工作项都分配了本地内存
    barrier(CLK_LOCAL_MEM_FENCE);

    // 调试：检查输入数据
    if (gid < 3) { // 只检查前3个块
        uint first4 = LZ4_read32(input + start);
        KDEBUG("DEBUG: Block %d: start=%d, srcSize=%d, first4=%08x, acceleration=%d\n",
               gid, start, srcSize, first4, acceleration);
    }

    int compressedSize = lz4_compress_core_accelerated(
        input + start,
        dst,
        srcSize,
        dstCapacity,
        tableType,
        localHashTable,
        acceleration  // 使用传入的acceleration参数
    );

    // 调试：检查压缩结果
    if (gid < 3 && compressedSize > 0) { // 只检查前3个块
        uint first4_compressed = LZ4_read32(dst);
        KDEBUG("DEBUG: Block %d: compressedSize=%d, first4_compressed=%08x, acceleration=%d\n",
               gid, compressedSize, first4_compressed, acceleration);
    }

    // 记录块大小
    blockSizes[gid] = compressedSize;
}

// Helper functions
#define LZ4_COMPRESSBOUND(isize) ((unsigned int)(isize) > (unsigned int)LZ4_MAX_INPUT_SIZE ? 0 : (isize) + ((isize)/255) + 16)
#define LZ4_MAX_INPUT_SIZE 0x7E000000   /* 2 113 929 216 bytes */

// Test kernel for debugging
__kernel void lz4_test_kernel(
    __global const unsigned char* input,
    __global unsigned char* output,
    int inputSize
) {
    int gid = get_global_id(0);
    if (gid != 0) return;

    // Simple test: copy first 10 bytes
    for (int i = 0; i < 10 && i < inputSize; i++) {
        output[i] = input[i];
    }

    // Test memory access
    if (inputSize >= 4) {
        uint testValue = LZ4_read32(input);
        LZ4_write32(output + 10, testValue);
    }
}

// LZ4 Frame Format Constants
#define LZ4F_MAGICNUMBER 0x184D2204U
#define LZ4F_MAGIC_SKIPPABLE_START 0x184D2A50U
#define LZ4F_BLOCKUNCOMPRESSED_FLAG 0x80000000U
#define LZ4F_BLOCKSIZEID_DEFAULT 4  // max64KB

// Frame Header Sizes
#define LZ4F_HEADER_SIZE_MIN 7
#define LZ4F_HEADER_SIZE_MAX 19
#define LZ4F_BLOCK_HEADER_SIZE 4
#define LZ4F_BLOCK_CHECKSUM_SIZE 4
#define LZ4F_CONTENT_CHECKSUM_SIZE 4
#define LZ4F_ENDMARK_SIZE 4

// Bit field constants for frame header
#define _1BIT  0x01
#define _2BITS 0x03
#define _3BITS 0x07
#define _4BITS 0x0F
#define _8BITS 0xFF

// Frame header structure
typedef struct LZ4F_frameHeader_s {
    uint FLG;                    // Frame header flags
    uint BD;                     // Block descriptor
    uint contentSize;           // Content size (optional)
    uint dictID;                // Dictionary ID (optional)
} LZ4F_frameHeader_t;


// Parallel LZ4 Block Decompression
__kernel void lz4_decompress_block(const __global uchar* input, __global uchar* output,
        unsigned long compressed_offset, unsigned long compressed_size,
        unsigned long output_offset, unsigned long max_output_size,
        __global uint* output_sizes, uint block_index)
{
    const __global uchar* ip = input + compressed_offset;
    const __global uchar* const iend = ip + compressed_size;
    __global uchar* op = output + output_offset;
    __global uchar* const oend = op + max_output_size;
    __global uchar* const base_output = op;

    const uint inc32table[8] = {0, 1, 2, 1, 0, 4, 4, 4};
    const int  dec64table[8] = {0, 0, 0, -1, -4, 1, 2, 3};

    if ((compressed_size == 0) || (max_output_size == 0)) {
        output_sizes[block_index] = 0;
        return;
    }

    while (ip < iend) {
        unsigned token = *ip++;
        size_t literalLength = token >> ML_BITS;
        size_t matchLength;

        if (literalLength == RUN_MASK) {
            unsigned s;
            do {
                if (ip >= iend) {
                    KERROR("LZ4_GPU: literal length overflow (block %u)\n", block_index);
                    goto _output_error;
                }
                s = *ip++;
                literalLength += s;
            } while (s == 255);
        }

        __global uchar* cpy = op + literalLength;

        if (cpy > oend) {
            KERROR("LZ4_GPU: literal copy exceeds output (block %u)\n", block_index);
            goto _output_error;
        }
        if (ip + literalLength > iend) {
            KERROR("LZ4_GPU: literal copy exceeds input (block %u)\n", block_index);
            goto _output_error;
        }

        if ((cpy > oend - MFLIMIT) || (ip + literalLength > iend - (2 + 1 + LASTLITERALS))) {
            if ((ip + literalLength) != iend) {
                KERROR("LZ4_GPU: last literals must consume input (block %u)\n", block_index);
                goto _output_error;
            }
            if (cpy > oend) {
                KERROR("LZ4_GPU: last literals exceed output (block %u)\n", block_index);
                goto _output_error;
            }
            LZ4_memmove(op, ip, literalLength);
            ip += literalLength;
            op += literalLength;
            output_sizes[block_index] = (uint)(op - base_output);
            return;
        }

        LZ4_wildCopy8(op, ip, cpy);
        ip += literalLength;
        op = cpy;

        if (ip >= iend) {
            output_sizes[block_index] = (uint)(op - base_output);
            return;
        }

        if (ip + 2 > iend) {
            KERROR("LZ4_GPU: insufficient bytes for offset (block %u)\n", block_index);
            goto _output_error;
        }
        size_t offset = (size_t)LZ4_readLE16(ip);
        ip += 2;

        if (offset == 0) {
            KERROR("LZ4_GPU: zero offset detected (block %u)\n", block_index);
            goto _output_error;
        }

        const __global uchar* match = op - offset;
        if (match < base_output) {
            KERROR("LZ4_GPU: match outside of output window (block %u)\n", block_index);
            goto _output_error;
        }

        matchLength = token & ML_MASK;
        if (matchLength == ML_MASK) {
            unsigned s;
            do {
                if (ip > iend - LASTLITERALS) {
                    KERROR("LZ4_GPU: match length overflow (block %u)\n", block_index);
                    goto _output_error;
                }
                s = *ip++;
                matchLength += s;
            } while (s == 255);
        }
        matchLength += MINMATCH;

        cpy = op + matchLength;
        if (cpy > oend) {
            KERROR("LZ4_GPU: match copy exceeds output (block %u)\n", block_index);
            goto _output_error;
        }

        if (offset < 8) {
            // Copy first 4 bytes individually
            op[0] = match[0];
            op[1] = match[1];
            op[2] = match[2];
            op[3] = match[3];
            // Next 4 bytes according to tables, but avoid unaligned 32-bit loads
            match += inc32table[offset];
            op[4] = match[0];
            op[5] = match[1];
            op[6] = match[2];
            op[7] = match[3];
            match -= dec64table[offset];
        } else {
            // Copy 8 bytes safely. Use vector loads/stores when available for better throughput.
#ifdef LZ4_GPU_VECTOR_IO
            {
                uchar8 v8 = vload8(0, (const __global uchar*)match);
                vstore8(v8, 0, (__global uchar*)op);
                match += 8;
            }
#else
            for (int i = 0; i < 8; ++i) op[i] = match[i];
            match += 8;
#endif
        }
        op += 8;

        if (cpy > oend - MATCH_SAFEGUARD_DISTANCE) {
            __global uchar* const oCopyLimit = oend - (WILDCOPYLENGTH - 1);
            if (cpy > oend - LASTLITERALS) {
                KERROR("LZ4_GPU: match copy violates LASTLITERALS (block %u)\n", block_index);
                goto _output_error;
            }
            if (op < oCopyLimit) {
                LZ4_wildCopy8(op, match, oCopyLimit);
                match += (size_t)(oCopyLimit - op);
                op = oCopyLimit;
            }
            while (op < cpy) {
                *op++ = *match++;
            }
            op = cpy;
        } else {
            // Copy at least 8 bytes, safely. Use vector ops when available.
#ifdef LZ4_GPU_VECTOR_IO
            {
                uchar8 v8 = vload8(0, (const __global uchar*)match);
                vstore8(v8, 0, (__global uchar*)op);
            }
            if (matchLength > 16) {
                LZ4_wildCopy8(op + 8, match + 8, cpy);
            }
            op = cpy;
#else
            for (int i = 0; i < 8; ++i) op[i] = match[i];
            if (matchLength > 16) {
                LZ4_wildCopy8(op + 8, match + 8, cpy);
            }
            op = cpy;
#endif
        }
    }

    output_sizes[block_index] = (uint)(op - base_output);
    return;

_output_error:
    output_sizes[block_index] = 0xFFFFFFFFu;
    return;
}

// Size-only parser: compute decompressed size for each block in parallel without writing output
__kernel void lz4_decompress_blocks_sizeonly(
        const __global uchar* input,
        __global const ulong* comp_offsets,
        __global const ulong* comp_sizes,
        ulong max_block_out_size,
        __global uint* sizes_out,
        uint totalBlocks)
{
    uint gid = get_global_id(0);
    if (gid >= totalBlocks) return;

    const __global uchar* ip = input + comp_offsets[gid];
    const __global uchar* const iend = ip + comp_sizes[gid];

    // Track only the produced size (no actual writes)
    ulong produced = 0;

    if (comp_sizes[gid] == 0) { sizes_out[gid] = 0; return; }

    while (ip < iend) {
        unsigned token = *ip++;
        ulong literalLength = (ulong)(token >> ML_BITS);

        if (literalLength == RUN_MASK) {
            unsigned s;
            do {
                if (ip >= iend) { sizes_out[gid] = 0xFFFFFFFFu; return; }
                s = *ip++;
                literalLength += s;
            } while (s == 255);
        }

        if (ip + literalLength > iend) { sizes_out[gid] = 0xFFFFFFFFu; return; }
        // produce literals
        if (produced + literalLength > max_block_out_size) { sizes_out[gid] = 0xFFFFFFFFu; return; }
        produced += literalLength;
        ip += literalLength;

        if (ip >= iend) { sizes_out[gid] = (uint)produced; return; }

        if (ip + 2 > iend) { sizes_out[gid] = 0xFFFFFFFFu; return; }
        ulong offset = (ulong)LZ4_readLE16(ip);
        ip += 2;
        if (offset == 0) { sizes_out[gid] = 0xFFFFFFFFu; return; }
        // match must reference already produced data
        if (offset > produced) { sizes_out[gid] = 0xFFFFFFFFu; return; }

        ulong matchLength = (ulong)(token & ML_MASK);
        if (matchLength == ML_MASK) {
            unsigned s;
            do {
                if (ip > iend - LASTLITERALS) { sizes_out[gid] = 0xFFFFFFFFu; return; }
                s = *ip++;
                matchLength += s;
            } while (s == 255);
        }
        matchLength += MINMATCH;

        if (produced + matchLength > max_block_out_size) { sizes_out[gid] = 0xFFFFFFFFu; return; }
        produced += matchLength;
    }

    sizes_out[gid] = (uint)produced;
}

// Parallel multi-block decompression: each work-item decompresses one block to its output offset
__kernel void lz4_decompress_blocks(
        const __global uchar* input,
        __global uchar* output,
        __global const ulong* comp_offsets,
        __global const ulong* comp_sizes,
        __global const ulong* out_offsets,
        __global const ulong* max_out_sizes,
        __global uint* sizes_out,
        uint totalBlocks)
{
    uint gid = get_global_id(0);
    if (gid >= totalBlocks) return;

    const __global uchar* ip = input + comp_offsets[gid];
    const __global uchar* const iend = ip + comp_sizes[gid];
    __global uchar* op = output + out_offsets[gid];
    __global uchar* const base_output = op;
    __global uchar* const oend = op + max_out_sizes[gid];

    const uint inc32table[8] = {0, 1, 2, 1, 0, 4, 4, 4};
    const int  dec64table[8] = {0, 0, 0, -1, -4, 1, 2, 3};

    if ((comp_sizes[gid] == 0) || (max_out_sizes[gid] == 0)) { sizes_out[gid] = 0; return; }

    while (ip < iend) {
        unsigned token = *ip++;
        size_t literalLength = token >> ML_BITS;
        size_t matchLength;

        if (literalLength == RUN_MASK) {
            unsigned s;
            do {
                if (ip >= iend) { KERROR("LZ4_GPU: literal length overflow (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }
                s = *ip++;
                literalLength += s;
            } while (s == 255);
        }

        __global uchar* cpy = op + literalLength;
        if (cpy > oend) { KERROR("LZ4_GPU: literal copy exceeds output (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }
        if (ip + literalLength > iend) { KERROR("LZ4_GPU: literal copy exceeds input (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }

        if ((cpy > oend - MFLIMIT) || (ip + literalLength > iend - (2 + 1 + LASTLITERALS))) {
            if ((ip + literalLength) != iend) { KERROR("LZ4_GPU: last literals must consume input (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }
            if (cpy > oend) { KERROR("LZ4_GPU: last literals exceed output (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }
            LZ4_memmove(op, ip, literalLength);
            ip += literalLength;
            op += literalLength;
            sizes_out[gid] = (uint)(op - base_output);
            return;
        }

        LZ4_wildCopy8(op, ip, cpy);
        ip += literalLength;
        op = cpy;

        if (ip >= iend) { sizes_out[gid] = (uint)(op - base_output); return; }

        if (ip + 2 > iend) { KERROR("LZ4_GPU: insufficient bytes for offset (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }
        size_t offset = (size_t)LZ4_readLE16(ip);
        ip += 2;
        if (offset == 0) { KERROR("LZ4_GPU: zero offset detected (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }

        const __global uchar* match = op - offset;
        if (match < base_output) { KERROR("LZ4_GPU: match outside of output window (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }

        matchLength = token & ML_MASK;
        if (matchLength == ML_MASK) {
            unsigned s;
            do {
                if (ip > iend - LASTLITERALS) { KERROR("LZ4_GPU: match length overflow (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }
                s = *ip++;
                matchLength += s;
            } while (s == 255);
        }
        matchLength += MINMATCH;

        cpy = op + matchLength;
        if (cpy > oend) { KERROR("LZ4_GPU: match copy exceeds output (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }

        if (offset < 8) {
            op[0] = match[0]; op[1] = match[1]; op[2] = match[2]; op[3] = match[3];
            match += inc32table[offset];
            op[4] = match[0]; op[5] = match[1]; op[6] = match[2]; op[7] = match[3];
            match -= dec64table[offset];
        } else {
#ifdef LZ4_GPU_VECTOR_IO
            {
                uchar8 v8 = vload8(0, (const __global uchar*)match);
                vstore8(v8, 0, (__global uchar*)op);
                match += 8;
            }
#else
            for (int i = 0; i < 8; ++i) op[i] = match[i];
            match += 8;
#endif
        }
        op += 8;

        if (cpy > oend - MATCH_SAFEGUARD_DISTANCE) {
            __global uchar* const oCopyLimit = oend - (WILDCOPYLENGTH - 1);
            if (cpy > oend - LASTLITERALS) { KERROR("LZ4_GPU: match copy violates LASTLITERALS (blk %u)\n", gid); sizes_out[gid]=0xFFFFFFFFu; return; }
            if (op < oCopyLimit) {
                LZ4_wildCopy8(op, match, oCopyLimit);
                match += (size_t)(oCopyLimit - op);
                op = oCopyLimit;
            }
            while (op < cpy) { *op++ = *match++; }
            op = cpy;
        } else {
#ifdef LZ4_GPU_VECTOR_IO
            {
                uchar8 v8 = vload8(0, (const __global uchar*)match);
                vstore8(v8, 0, (__global uchar*)op);
            }
            if (matchLength > 16) { LZ4_wildCopy8(op + 8, match + 8, cpy); }
            op = cpy;
#else
            for (int i = 0; i < 8; ++i) op[i] = match[i];
            if (matchLength > 16) { LZ4_wildCopy8(op + 8, match + 8, cpy); }
            op = cpy;
#endif
        }
    }

    sizes_out[gid] = (uint)(op - base_output);
}
