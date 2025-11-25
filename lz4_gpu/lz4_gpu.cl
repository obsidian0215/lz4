/*
 * LZ4 GPU Parallel Implementation
 * Based on LZ4 v1.10.0 source code
 * Copyright (c) Yann Collet. BSD 2-Clause License
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// --- Constants ---
#define MINMATCH 4
#define WILDCOPYLENGTH 8
#define LASTLITERALS 5
#define MFLIMIT 12
#define MATCH_SAFEGUARD_DISTANCE ((2*WILDCOPYLENGTH) - MINMATCH)
#define FASTLOOP_SAFE_DISTANCE 64
#define LZ4_DISTANCE_MAX 65535

#define ML_BITS 4
#define ML_MASK ((1U<<ML_BITS)-1)
#define RUN_BITS (8-ML_BITS)
#define RUN_MASK ((1U<<RUN_BITS)-1)

#define LZ4_HASHLOG 14
#define LZ4_MEMORY_USAGE LZ4_HASHLOG

// --- Types ---
typedef unsigned char BYTE;
typedef unsigned short U16;
typedef unsigned int U32;
typedef unsigned long U64;

// --- Memory Access ---
// Use unaligned access where possible, but fallback to byte-wise if needed
inline U16 LZ4_read16(const __global BYTE* ptr) {
    return (U16)ptr[0] | ((U16)ptr[1] << 8);
}

inline U32 LZ4_read32(const __global BYTE* ptr) {
    return (U32)ptr[0] | ((U32)ptr[1] << 8) | ((U32)ptr[2] << 16) | ((U32)ptr[3] << 24);
}

inline U64 LZ4_read64(const __global BYTE* ptr) {
    return (U64)LZ4_read32(ptr) | ((U64)LZ4_read32(ptr + 4) << 32);
}

inline void LZ4_write16(__global BYTE* ptr, U16 value) {
    ptr[0] = (BYTE)value;
    ptr[1] = (BYTE)(value >> 8);
}

inline void LZ4_write32(__global BYTE* ptr, U32 value) {
    ptr[0] = (BYTE)value;
    ptr[1] = (BYTE)(value >> 8);
    ptr[2] = (BYTE)(value >> 16);
    ptr[3] = (BYTE)(value >> 24);
}

inline void LZ4_write64(__global BYTE* ptr, U64 value) {
    LZ4_write32(ptr, (U32)value);
    LZ4_write32(ptr + 4, (U32)(value >> 32));
}

inline U16 LZ4_readLE16(const __global BYTE* ptr) { return LZ4_read16(ptr); }

// --- Memory Copy ---
inline void LZ4_memcpy(__global BYTE* dst, const __global BYTE* src, int size) {
#ifdef LZ4_GPU_VECTOR_IO
    // Vectorized copy for larger sizes
    while (size >= 16) {
        vstore16(vload16(0, src), 0, dst);
        dst += 16; src += 16; size -= 16;
    }
    if (size >= 8) {
        vstore8(vload8(0, src), 0, dst);
        dst += 8; src += 8; size -= 8;
    }
#endif
    for (int i = 0; i < size; ++i) dst[i] = src[i];
}

inline void LZ4_memmove(__global BYTE* dst, const __global BYTE* src, int size) {
    if (dst < src) {
        LZ4_memcpy(dst, src, size);
    } else {
        for (int i = size - 1; i >= 0; --i) dst[i] = src[i];
    }
}

inline void LZ4_wildCopy8(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    do {
#ifdef LZ4_GPU_VECTOR_IO
        vstore8(vload8(0, src), 0, dst);
#else
        LZ4_memcpy(dst, src, 8);
#endif
        dst += 8; src += 8;
    } while (dst < dstEnd);
}

inline void LZ4_wildCopy32(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    do {
#ifdef LZ4_GPU_VECTOR_IO
        vstore16(vload16(0, src), 0, dst);
        vstore16(vload16(0, src+16), 0, dst+16);
#else
        LZ4_memcpy(dst, src, 32);
#endif
        dst += 32; src += 32;
    } while (dst < dstEnd);
}

// --- Hashing ---
inline U32 LZ4_hash4(U32 sequence, int tableType) {
    if (tableType == 0) // byU16
        return ((sequence * 2654435761U) >> ((MINMATCH*8)-(LZ4_HASHLOG+1)));
    else // byU32
        return ((sequence * 2654435761U) >> ((MINMATCH*8)-LZ4_HASHLOG));
}

inline U32 LZ4_hash5(U64 sequence, int tableType) {
    const U32 hashLog = (tableType == 0) ? LZ4_HASHLOG+1 : LZ4_HASHLOG;
    const U64 prime5bytes = 889523592379ULL;
    return (U32)(((sequence << 24) * prime5bytes) >> (64 - hashLog));
}

inline U32 LZ4_hashPosition(const __global BYTE* p, int tableType) {
    if ((sizeof(size_t) == 8) && (tableType != 0)) {
        return LZ4_hash5(LZ4_read64(p), tableType);
    }
    return LZ4_hash4(LZ4_read32(p), tableType);
}

inline void LZ4_putIndexOnHash(U32 idx, U32 h, __local U32* tableBase, int tableType, U32 hashTableSize) {
    if (tableType == 0) { // byU16
        __local U16* hashTable = (__local U16*)tableBase;
        hashTable[h & (hashTableSize - 1)] = (U16)idx;
    } else { // byU32
        tableBase[h & (hashTableSize - 1)] = idx;
    }
}

inline U32 LZ4_getIndexOnHash(U32 h, __local U32* tableBase, int tableType, U32 hashTableSize) {
    if (tableType == 0) { // byU16
        __local U16* hashTable = (__local U16*)tableBase;
        return (U32)hashTable[h & (hashTableSize - 1)];
    } else { // byU32
        return tableBase[h & (hashTableSize - 1)];
    }
}

// --- Matching ---
inline U32 LZ4_NbCommonBytes(U32 val) {
    if (val == 0) return 4;
    U32 n = 0;
    if ((val & 0xFF) == 0) { n++; val >>= 8; }
    if ((val & 0xFF) == 0) { n++; val >>= 8; }
    if ((val & 0xFF) == 0) { n++; val >>= 8; }
    return n;
}

inline U32 LZ4_count(const __global BYTE* pIn, const __global BYTE* pMatch, const __global BYTE* pInLimit) {
    const __global BYTE* const pStart = pIn;

    if (pIn < pInLimit - 3) {
        U32 diff = LZ4_read32(pMatch) ^ LZ4_read32(pIn);
        if (diff != 0) return LZ4_NbCommonBytes(diff);
        pIn += 4; pMatch += 4;
    }

    while (pIn < pInLimit - 3) {
        U32 diff = LZ4_read32(pMatch) ^ LZ4_read32(pIn);
        if (diff != 0) return (U32)(pIn - pStart) + LZ4_NbCommonBytes(diff);
        pIn += 4; pMatch += 4;
    }

    if ((pIn < pInLimit - 1) && (LZ4_read16(pMatch) == LZ4_read16(pIn))) {
        pIn += 2; pMatch += 2;
    }
    if ((pIn < pInLimit) && (*pMatch == *pIn)) pIn++;
    return (U32)(pIn - pStart);
}

// --- Compression Core ---
int lz4_compress_core_accelerated(
    __global const BYTE* src,
    __global BYTE* dst,
    int srcSize,
    int dstCapacity,
    int tableType,
    __local U32* hashTable,
    int acceleration
) {
    const int hashTableSize = 1 << LZ4_HASHLOG;
    const __global BYTE* ip = src;
    __global BYTE* op = dst;
    __global BYTE* const oend = op + dstCapacity;
    const __global BYTE* const iend = ip + srcSize;
    const __global BYTE* anchor = ip;
    const __global BYTE* const mflimitPlusOne = iend - MFLIMIT + 1;
    const __global BYTE* const matchlimit = iend - LASTLITERALS;

    // Init hash table
    for (int i = 0; i < hashTableSize; i++) hashTable[i] = 0;

    if (srcSize < (MFLIMIT+1)) goto _last_literals;

    // First Byte
    LZ4_putIndexOnHash(0, LZ4_hashPosition(ip, tableType), hashTable, tableType, hashTableSize);
    ip++;
    U32 forwardH = LZ4_hashPosition(ip, tableType);

    for (;;) {
        const __global BYTE* match;
        __global BYTE* token;

        // Find a match
        {
            const __global BYTE* forwardIp = ip;
            int step = 1;
            int searchMatchNb = acceleration << 6; // LZ4_skipTrigger=6

            do {
                U32 h = forwardH;
                U32 current = (U32)(forwardIp - src);
                U32 matchIndex = LZ4_getIndexOnHash(h, hashTable, tableType, hashTableSize);

                ip = forwardIp;
                forwardIp += step;
                step = (searchMatchNb++ >> 6);

                if (forwardIp > mflimitPlusOne) goto _last_literals;

                match = src + matchIndex;
                forwardH = LZ4_hashPosition(forwardIp, tableType);
                LZ4_putIndexOnHash(current, h, hashTable, tableType, hashTableSize);

                if ((matchIndex < current) && (current - matchIndex < LZ4_DISTANCE_MAX) && (LZ4_read32(match) == LZ4_read32(ip))) {
                    break; // Match found
                }
            } while(1);
        }

        // Catch up
        while ((ip > anchor) && (match > src) && (ip[-1] == match[-1])) { ip--; match--; }

        // Encode Literals
        {
            unsigned litLength = (unsigned)(ip - anchor);
            token = op++;
            if (op + litLength + (2 + 1 + LASTLITERALS) + (litLength/255) > oend) return 0;

            if (litLength >= RUN_MASK) {
                unsigned len = litLength - RUN_MASK;
                *token = (RUN_MASK << ML_BITS);
                for(; len >= 255; len -= 255) *op++ = 255;
                *op++ = (BYTE)len;
            } else {
                *token = (BYTE)(litLength << ML_BITS);
            }

            LZ4_wildCopy8(op, anchor, op + litLength);
            op += litLength;
        }

_next_match:
        // Encode Offset
        LZ4_write16(op, (U16)(ip - match)); op += 2;

        // Encode MatchLength
        {
            unsigned matchCode = LZ4_count(ip + MINMATCH, match + MINMATCH, matchlimit);
            ip += matchCode + MINMATCH;

            if (matchCode >= ML_MASK) {
                *token += ML_MASK;
                matchCode -= ML_MASK;
                LZ4_write32(op, 0xFFFFFFFF);
                while (matchCode >= 4*255) {
                    op += 4;
                    LZ4_write32(op, 0xFFFFFFFF);
                    matchCode -= 4*255;
                }
                op += matchCode / 255;
                *op++ = (BYTE)(matchCode % 255);
            } else {
                *token += (BYTE)matchCode;
            }
        }

        anchor = ip;

        if (ip >= mflimitPlusOne) break;

        // Fill table
        LZ4_putIndexOnHash((U32)(ip - 2 - src), LZ4_hashPosition(ip - 2, tableType), hashTable, tableType, hashTableSize);

        // Test next position
        U32 h = LZ4_hashPosition(ip, tableType);
        U32 current = (U32)(ip - src);
        U32 matchIndex = LZ4_getIndexOnHash(h, hashTable, tableType, hashTableSize);
        LZ4_putIndexOnHash(current, h, hashTable, tableType, hashTableSize);

        if ((matchIndex < current) && (current - matchIndex < LZ4_DISTANCE_MAX) && (LZ4_read32(src + matchIndex) == LZ4_read32(ip))) {
            token = op++;
            *token = 0;
            match = src + matchIndex;
            goto _next_match;
        }

        forwardH = LZ4_hashPosition(++ip, tableType);
    }

_last_literals:
    {
        size_t lastRun = (size_t)(iend - anchor);
        if (op + lastRun + 1 + ((lastRun + 255 - RUN_MASK) / 255) > oend) return 0;

        if (lastRun >= RUN_MASK) {
            size_t accumulator = lastRun - RUN_MASK;
            *op++ = RUN_MASK << ML_BITS;
            for(; accumulator >= 255; accumulator -= 255) *op++ = 255;
            *op++ = (BYTE)accumulator;
        } else {
            *op++ = (BYTE)(lastRun << ML_BITS);
        }
        LZ4_memcpy(op, anchor, lastRun);
        op += lastRun;
    }

    return (int)(op - dst);
}

// --- Decompression Core ---
// Implements LZ4_decompress_safe_generic logic
void lz4_decompress_generic(
    const __global BYTE* src,
    __global BYTE* dst,
    int srcSize,
    int outputSize,
    __global U32* outputSizePtr
) {
    const __global BYTE* ip = src;
    const __global BYTE* const iend = ip + srcSize;
    __global BYTE* op = dst;
    __global BYTE* const oend = op + outputSize;
    __global BYTE* cpy;

    const __global BYTE* const shortiend = iend - 14 - 2;
    const __global BYTE* const shortoend = oend - 14 - 18;

    const unsigned inc32table[8] = {0, 1, 2, 1, 0, 4, 4, 4};
    const int      dec64table[8] = {0, 0, 0, -1, -4, 1, 2, 3};

    if (srcSize == 0) { *outputSizePtr = 0; return; }

    for (;;) {
        unsigned token = *ip++;
        size_t length = token >> ML_BITS;
        size_t offset;
        __global BYTE* match;

        // Fast path
        if ((length != RUN_MASK) && (ip < shortiend) && (op <= shortoend)) {
            LZ4_memcpy(op, ip, 16);
            op += length; ip += length;

            length = token & ML_MASK;
            offset = LZ4_readLE16(ip); ip += 2;
            match = op - offset;

            if ((length != ML_MASK) && (offset >= 8) && (match >= dst)) {
                LZ4_memcpy(op, match, 8);
                LZ4_memcpy(op + 8, match + 8, 8);
                LZ4_memcpy(op + 16, match + 16, 2);
                op += length + MINMATCH;
                continue;
            }
            goto _copy_match;
        }

        // Decode literal length
        if (length == RUN_MASK) {
            unsigned s;
            do {
                if (ip >= iend) goto _output_error;
                s = *ip++;
                length += s;
            } while (s == 255);
        }

        // Copy literals
        cpy = op + length;
        if ((cpy > oend - MFLIMIT) || (ip + length > iend - (2 + 1 + LASTLITERALS))) {
            if (ip + length != iend) goto _output_error;
            if (cpy > oend) goto _output_error;
            LZ4_memcpy(op, ip, length);
            op += length;
            break; // End of block
        }
        LZ4_wildCopy8(op, ip, cpy);
        ip += length; op = cpy;

        // Get offset
        offset = LZ4_readLE16(ip); ip += 2;
        match = op - offset;
        length = token & ML_MASK;

_copy_match:
        if (length == ML_MASK) {
            unsigned s;
            do {
                if (ip >= iend - LASTLITERALS) goto _output_error;
                s = *ip++;
                length += s;
            } while (s == 255);
        }
        length += MINMATCH;

        if (match < dst) goto _output_error;

        cpy = op + length;
        if (cpy > oend - MATCH_SAFEGUARD_DISTANCE) {
            if (cpy > oend - LASTLITERALS) goto _output_error;
            if (op < oend - (WILDCOPYLENGTH - 1)) {
                LZ4_wildCopy8(op, match, oend - (WILDCOPYLENGTH - 1));
                match += (oend - (WILDCOPYLENGTH - 1)) - op;
                op = oend - (WILDCOPYLENGTH - 1);
            }
            while (op < cpy) *op++ = *match++;
        } else {
            if (offset < 8) {
                op[0] = match[0]; op[1] = match[1]; op[2] = match[2]; op[3] = match[3];
                match += inc32table[offset];
                LZ4_memcpy(op + 4, match, 4);
                match -= dec64table[offset];
            } else {
                LZ4_memcpy(op, match, 8);
                match += 8;
            }
            op += 8;
            LZ4_wildCopy8(op, match, cpy);
        }
        op = cpy;
    }

    *outputSizePtr = (U32)(op - dst);
    return;

_output_error:
    *outputSizePtr = 0xFFFFFFFF;
}

// --- Kernels ---

__kernel void lz4_compress_block_accelerated(
    __global const BYTE* input,
    __global BYTE* output,
    __global U32* blockSizes,
    __global const U32* blockOffsets,
    __global const U32* outputOffsets,
    __global const U32* maxOutputSizes,
    int totalBlocks,
    int inputSize,
    int tableType,
    int acceleration,
    __local U32* localHashTable
) {
    int gid = get_global_id(0);
    if (gid >= totalBlocks) return;

    int start = blockOffsets[gid * 2];
    int blockSize = blockOffsets[gid * 2 + 1];
    if (start >= inputSize || blockSize <= 0) { blockSizes[gid] = 0; return; }

    int dstCapacity = (int)maxOutputSizes[gid];
    __global BYTE* dst = output + outputOffsets[gid];

    blockSizes[gid] = lz4_compress_core_accelerated(
        input + start, dst, blockSize, dstCapacity, tableType, localHashTable, acceleration
    );
}

__kernel void lz4_decompress_block(
    const __global BYTE* input,
    __global BYTE* output,
    U64 compressed_offset,
    U64 compressed_size,
    U64 output_offset,
    U64 max_output_size,
    __global U32* output_sizes,
    U32 block_index
) {
    lz4_decompress_generic(
        input + compressed_offset,
        output + output_offset,
        (int)compressed_size,
        (int)max_output_size,
        &output_sizes[block_index]
    );
}

__kernel void lz4_decompress_blocks(
    const __global BYTE* input,
    __global BYTE* output,
    __global const U64* comp_offsets,
    __global const U64* comp_sizes,
    __global const U64* out_offsets,
    __global const U64* max_out_sizes,
    __global U32* sizes_out,
    U32 totalBlocks
) {
    int gid = get_global_id(0);
    if (gid >= totalBlocks) return;

    lz4_decompress_generic(
        input + comp_offsets[gid],
        output + out_offsets[gid],
        (int)comp_sizes[gid],
        (int)max_out_sizes[gid],
        &sizes_out[gid]
    );
}

__kernel void lz4_compress_block(
    __global const BYTE* input,
    __global BYTE* output,
    __global U32* blockSizes,
    __global const U32* blockOffsets,
    __global const U32* outputOffsets,
    int totalBlocks,
    int inputSize,
    int tableType,
    __local U32* localHashTable
) {
    int gid = get_global_id(0);
    if (gid >= totalBlocks) return;

    int start = blockOffsets[gid * 2];
    int blockSize = blockOffsets[gid * 2 + 1];
    if (start >= inputSize || blockSize <= 0) { blockSizes[gid] = 0; return; }

    // Calculate conservative capacity
    int dstCapacity = blockSize + (blockSize/255) + 256;
    __global BYTE* dst = output + outputOffsets[gid];

    blockSizes[gid] = lz4_compress_core_accelerated(
        input + start, dst, blockSize, dstCapacity, tableType, localHashTable, 1
    );
}

__kernel void lz4_decompress_blocks_sizeonly(
        const __global BYTE* input,
        __global const U64* comp_offsets,
        __global const U64* comp_sizes,
        U64 max_block_out_size,
        __global U32* sizes_out,
        U32 totalBlocks)
{
    int gid = get_global_id(0);
    if (gid >= totalBlocks) return;

    const __global BYTE* ip = input + comp_offsets[gid];
    const __global BYTE* const iend = ip + comp_sizes[gid];
    U64 produced = 0;

    if (comp_sizes[gid] == 0) { sizes_out[gid] = 0; return; }

    for (;;) {
        unsigned token = *ip++;
        size_t length = token >> ML_BITS;

        if (length == RUN_MASK) {
            unsigned s;
            do {
                if (ip >= iend) { sizes_out[gid] = 0xFFFFFFFF; return; }
                s = *ip++;
                length += s;
            } while (s == 255);
        }

        if (produced + length > max_block_out_size) { sizes_out[gid] = 0xFFFFFFFF; return; }
        produced += length;
        ip += length;

        if (ip >= iend) { sizes_out[gid] = (U32)produced; return; }

        if (ip + 2 > iend) { sizes_out[gid] = 0xFFFFFFFF; return; }
        size_t offset = LZ4_readLE16(ip); ip += 2;
        if (offset == 0 || offset > produced) { sizes_out[gid] = 0xFFFFFFFF; return; }

        length = token & ML_MASK;
        if (length == ML_MASK) {
            unsigned s;
            do {
                if (ip >= iend - LASTLITERALS) { sizes_out[gid] = 0xFFFFFFFF; return; }
                s = *ip++;
                length += s;
            } while (s == 255);
        }
        length += MINMATCH;

        if (produced + length > max_block_out_size) { sizes_out[gid] = 0xFFFFFFFF; return; }
        produced += length;
    }
}
