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
typedef unsigned char U8;
/* Fingerprint value type (we support up to 16 bits for compact storage) */
#if LZ4_GPU_FINGERPRINT_BITS <= 8
typedef U8 fp_t;
#else
typedef U16 fp_t;
#endif

// --- Memory Access ---
// Use unaligned access where possible, but fallback to byte-wise if needed
inline U16 LZ4_read16(const __global BYTE* ptr) __attribute__((always_inline)) {
    return (U16)ptr[0] | ((U16)ptr[1] << 8);
}

inline U32 LZ4_read32(const __global BYTE* ptr) __attribute__((always_inline)) {
    return as_uint(vload4(0, (const __global uchar*)ptr));
}

inline U64 LZ4_read64(const __global BYTE* ptr) __attribute__((always_inline)) {
    return as_ulong(vload8(0, (const __global uchar*)ptr));
}

inline void LZ4_write16(__global BYTE* ptr, U16 value) __attribute__((always_inline)) {
    ptr[0] = (BYTE)value;
    ptr[1] = (BYTE)(value >> 8);
}

inline void LZ4_write32(__global BYTE* ptr, U32 value) __attribute__((always_inline)) {
    vstore4(as_uchar4(value), 0, (__global uchar*)ptr);
}

inline void LZ4_write64(__global BYTE* ptr, U64 value) __attribute__((always_inline)) {
    vstore8(as_uchar8(value), 0, (__global uchar*)ptr);
}

inline U16 LZ4_readLE16(const __global BYTE* ptr) __attribute__((always_inline)) { return LZ4_read16(ptr); }

// --- Memory Copy (Vectorized) ---
inline void LZ4_UA_COPYN(__global BYTE* d, const __global BYTE* s, uint nn)
{
    if (nn >= 32) {
        while (nn >= 32) {
            vstore16(vload16(0, (const __global uchar*)s), 0, (__global uchar*)d);
            vstore16(vload16(1, (const __global uchar*)s), 0, (__global uchar*)(d + 16));
            d += 32; s += 32; nn -= 32;
        }
    }
    if (nn >= 16) {
        vstore16(vload16(0, (const __global uchar*)s), 0, (__global uchar*)d);
        d += 16; s += 16; nn -= 16;
    }
    if (nn >= 8) {
        vstore8(vload8(0, (const __global uchar*)s), 0, (__global uchar*)d);
        d += 8; s += 8; nn -= 8;
    }
    if (nn >= 4) {
        vstore4(vload4(0, (const __global uchar*)s), 0, (__global uchar*)d);
        d += 4; s += 4; nn -= 4;
    }
    while (nn > 0) { *d++ = *s++; nn--; }
}

inline void LZ4_COPY_MATCH(__global BYTE* op, const __global BYTE* m_pos, uint len)
{
    uint offset = op - m_pos;
    if (offset >= len) {
        LZ4_UA_COPYN(op, m_pos, len);
        return;
    }
    if (offset <= 4) {
        if (offset == 1) {
            BYTE c = *m_pos; uchar16 v16 = (uchar16)c;
            while (len >= 64) {
                vstore16(v16, 0, (__global uchar*)op);
                vstore16(v16, 1, (__global uchar*)op);
                vstore16(v16, 2, (__global uchar*)op);
                vstore16(v16, 3, (__global uchar*)op);
                op += 64; len -= 64;
            }
            while (len >= 16) {
                vstore16(v16, 0, (__global uchar*)op);
                op += 16; len -= 16;
            }
            if (len >= 8) {
                vstore8(v16.lo, 0, (__global uchar*)op);
                op += 8; len -= 8;
            }
            if (len >= 4) {
                vstore4(v16.s0123, 0, (__global uchar*)op);
                op += 4; len -= 4;
            }
            while (len > 0) { *op++ = c; len--; }
            return;
        }
        if (offset == 2) {
            BYTE p0 = m_pos[0], p1 = m_pos[1];
            uchar16 v16 = (uchar16)(p0, p1, p0, p1, p0, p1, p0, p1, p0, p1, p0, p1, p0, p1, p0, p1);
            while (len >= 16) {
                vstore16(v16, 0, (__global uchar*)op);
                op += 16; len -= 16;
            }
            if (len >= 8) {
                vstore8(v16.lo, 0, (__global uchar*)op);
                op += 8; len -= 8;
            }
            if (len >= 4) {
                vstore4(v16.s0123, 0, (__global uchar*)op);
                op += 4; len -= 4;
            }
            while (len >= 2) { *op++ = p0; *op++ = p1; len -= 2; }
            if (len) *op++ = p0;
            return;
        }
        if (offset == 3) {
            BYTE p0 = m_pos[0], p1 = m_pos[1], p2 = m_pos[2];
            while (len >= 3) { *op++ = p0; *op++ = p1; *op++ = p2; len -= 3; }
            if (len == 2) { *op++ = p0; *op++ = p1; }
            else if (len == 1) { *op++ = p0; }
            return;
        }
        if (offset == 4) {
            BYTE p0 = m_pos[0], p1 = m_pos[1], p2 = m_pos[2], p3 = m_pos[3];
            uchar16 v16 = (uchar16)(p0, p1, p2, p3, p0, p1, p2, p3, p0, p1, p2, p3, p0, p1, p2, p3);
            while (len >= 16) {
                vstore16(v16, 0, (__global uchar*)op);
                op += 16; len -= 16;
            }
            if (len >= 8) {
                vstore8(v16.lo, 0, (__global uchar*)op);
                op += 8; len -= 8;
            }
            if (len >= 4) {
                vstore4(v16.s0123, 0, (__global uchar*)op);
                op += 4; len -= 4;
            }
            while (len--) { *op = *m_pos; op++; m_pos++; }
            return;
        }
    }
    if (offset >= 64) {
        while (len >= 64) {
            vstore16(vload16(0, (const __global uchar*)m_pos), 0, (__global uchar*)op);
            vstore16(vload16(1, (const __global uchar*)m_pos), 0, (__global uchar*)(op + 16));
            vstore16(vload16(2, (const __global uchar*)m_pos), 0, (__global uchar*)(op + 32));
            vstore16(vload16(3, (const __global uchar*)m_pos), 0, (__global uchar*)(op + 48));
            op += 64; m_pos += 64; len -= 64;
        }
    }
    if (offset >= 32) {
        while (len >= 32) {
            vstore16(vload16(0, (const __global uchar*)m_pos), 0, (__global uchar*)op);
            vstore16(vload16(1, (const __global uchar*)m_pos), 0, (__global uchar*)(op + 16));
            op += 32; m_pos += 32; len -= 32;
        }
    }
    if (offset >= 16) {
        while (len >= 16) {
            vstore16(vload16(0, (const __global uchar*)m_pos), 0, (__global uchar*)op);
            op += 16; m_pos += 16; len -= 16;
        }
    }
    if (offset >= 8) {
        while (len >= 8) {
            vstore8(vload8(0, (const __global uchar*)m_pos), 0, (__global uchar*)op);
            op += 8; m_pos += 8; len -= 8;
        }
    }
    if (offset >= 4) {
        if (len >= 4) {
            vstore4(vload4(0, (const __global uchar*)m_pos), 0, (__global uchar*)op);
            op += 4; m_pos += 4; len -= 4;
        }
    }
    while (len > 0) { *op++ = *m_pos++; len--; }
}

inline void LZ4_memcpy(__global BYTE* dst, const __global BYTE* src, int size) {
    /* Use optimized vectorized copy instead of slow scalar loop */
    LZ4_UA_COPYN(dst, src, (uint)size);
}


#ifndef LZ4_GPU_DISABLE_VEC_COPY
// Use LZ4_UA_COPYN for literal-copy helpers (input -> output)
inline void LZ4_lit_memcpy(__global BYTE* dst, const __global BYTE* src, int size) {
    LZ4_UA_COPYN(dst, src, (uint)size);
}

inline void LZ4_match_memcpy(__global BYTE* dst, const __global BYTE* src, int size) {
    LZ4_UA_COPYN(dst, src, (uint)size);
}

inline void LZ4_match_wildCopy8(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    LZ4_UA_COPYN(dst, src, (uint)(dstEnd - dst));
}

inline void LZ4_wildCopy8(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    /* Forward-copy semantics for possible overlaps */
    while (dst < dstEnd) {
        *dst++ = *src++;
    }
}

inline void LZ4_lit_wildCopy8(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    if (dstEnd - dst <= 8) {
        LZ4_write64(dst, LZ4_read64(src));
    } else if (dstEnd - dst <= 16) {
        LZ4_write64(dst, LZ4_read64(src));
        LZ4_write64(dst + 8, LZ4_read64(src + 8));
    } else {
        LZ4_UA_COPYN(dst, src, (uint)(dstEnd - dst));
    }
}

inline void LZ4_wildCopy32(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    LZ4_UA_COPYN(dst, src, (uint)(dstEnd - dst));
}
#else
// --- Scalar literal-copy helpers (input -> output) ---
// These are safe to use for literal copying because source (ip) is the input
// buffer and does not overlap with destination; use 8-byte chunks for speed.
inline void LZ4_lit_memcpy(__global BYTE* dst, const __global BYTE* src, int size) {
    int i = 0;
    int end64 = size & ~7;
    for (; i < end64; i += 8) {
        U32 lo = LZ4_read32(src + i);
        U32 hi = LZ4_read32(src + i + 4);
        LZ4_write32(dst + i, lo);
        LZ4_write32(dst + i + 4, hi);
    }
    for (; i < size; ++i) dst[i] = src[i];
}

// --- Scalar match-copy helpers (用于 match -> dst 的拷贝)
// Forward-copy semantics are required for match-copy (src < dst)
inline void LZ4_match_memcpy(__global BYTE* dst, const __global BYTE* src, int size) {
    int i = 0;
    int end64 = size & ~7;
    for (; i < end64; i += 8) {
        U32 lo = LZ4_read32(src + i);
        U32 hi = LZ4_read32(src + i + 4);
        LZ4_write32(dst + i, lo);
        LZ4_write32(dst + i + 4, hi);
    }
    for (; i < size; ++i) dst[i] = src[i];
}

inline void LZ4_match_wildCopy8(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    while (dst < dstEnd) {
        U32 a = LZ4_read32(src);
        U32 b = LZ4_read32(src + 4);
        LZ4_write32(dst, a);
        LZ4_write32(dst + 4, b);
        dst += 8; src += 8;
    }
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
        LZ4_memcpy(dst, src, 8);
        dst += 8; src += 8;
    } while (dst < dstEnd);
}

// lit-specific wild copy (safe for input -> output because src is input zone)
inline void LZ4_lit_wildCopy8(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    while (dst < dstEnd) {
        LZ4_memcpy(dst, src, 8);
        dst += 8; src += 8;
    }
}

inline void LZ4_wildCopy32(__global BYTE* dst, const __global BYTE* src, __global BYTE* dstEnd) {
    do {
        LZ4_memcpy(dst, src, 32);
        dst += 32; src += 32;
    } while (dst < dstEnd);
}
#endif

// --- Hashing ---
inline U32 LZ4_hash4(U32 sequence, int tableType) __attribute__((always_inline)) {
    if (tableType == 0) // byU16
        return ((sequence * 2654435761U) >> ((MINMATCH*8)-(LZ4_HASHLOG+1)));
    else // byU32
        return ((sequence * 2654435761U) >> ((MINMATCH*8)-LZ4_HASHLOG));
}

inline U32 LZ4_hash5(U64 sequence, int tableType) __attribute__((always_inline)) {
    const U32 hashLog = (tableType == 0) ? LZ4_HASHLOG+1 : LZ4_HASHLOG;
    const U64 prime5bytes = 889523592379ULL;
    return (U32)(((sequence << 24) * prime5bytes) >> (64 - hashLog));
}

inline U32 LZ4_hashPosition(const __global BYTE* p, int tableType, U32* sequence) __attribute__((always_inline)) {
    U32 s = LZ4_read32(p);
    if (sequence) *sequence = s;
    return LZ4_hash4(s, tableType);
}

inline void LZ4_putIndexOnHash(U32 idx, U32 h, __global U64* tableBase, int tableType, U32 sequence, U32 epoch) __attribute__((always_inline)) {
    U32 const mask = (1 << LZ4_HASHLOG) - 1;
    U32 entry;
    if (tableType == 0) { // byU16 with 16-bit fingerprint
        entry = (idx & 0xFFFF) | ((sequence >> 16) << 16);
    } else { // byU32
        entry = idx;
    }
    tableBase[h & mask] = (((U64)epoch) << 32) | (U64)entry;
}

inline U32 LZ4_getIndexOnHash(U32 h, __global U64* tableBase, int tableType, U32 sequence, int* fp_match, U32 epoch) __attribute__((always_inline)) {
    U32 const mask = (1 << LZ4_HASHLOG) - 1;
    U64 const packed = tableBase[h & mask];
    U32 const tag = (U32)(packed >> 32);
    if (tag != epoch) {
        *fp_match = 0;
        return 0;
    }
    U32 const entry = (U32)packed;
    if (tableType == 0) { // byU16
        *fp_match = ((entry >> 16) == (sequence >> 16));
        return entry & 0xFFFF;
    } else { // byU32
        *fp_match = 1; // No fingerprint for byU32
        return entry;
    }
}

inline void LZ4_putIndexOnHashLocal(U32 idx, U32 h, __local U16* table) __attribute__((always_inline)) {
    table[h & ((1 << (LZ4_HASHLOG + 1)) - 1)] = (U16)idx;
}

inline U32 LZ4_getIndexOnHashLocal(U32 h, __local U16* table) __attribute__((always_inline)) {
    return (U32)table[h & ((1 << (LZ4_HASHLOG + 1)) - 1)];
}

// --- Matching ---
inline U32 LZ4_NbCommonBytes(U32 val) {
    if (val == 0) return 4;
    // Use clz/ctz hardware acceleration for counting matching bytes (trailing zeros)
    return (31 - clz(val & -val)) >> 3;
}

/* Count common bytes in a 64-bit word starting from least-significant byte */
inline U32 LZ4_NbCommonBytes64(U64 val) {
    if (val == 0ULL) return 8;
    // Hardware-accelerated count of matching bytes
    return (unsigned)((63 - clz(val & -(long)val)) >> 3);
}


inline U32 LZ4_count(const __global BYTE* pIn, const __global BYTE* pMatch, const __global BYTE* pInLimit) {
    const __global BYTE* const pStart = pIn;

    /* Prefer 64-bit compares where available to reduce loop iterations */
    if (pIn < pInLimit - 7) {
        U64 diff = LZ4_read64(pMatch) ^ LZ4_read64(pIn);
        if (diff != 0ULL) return LZ4_NbCommonBytes64(diff);
        pIn += 8;
        pMatch += 8;
    }

    while (pIn < pInLimit - 7) {
        U64 diff = LZ4_read64(pMatch) ^ LZ4_read64(pIn);
        if (diff != 0ULL) return (U32)(pIn - pStart) + LZ4_NbCommonBytes64(diff);
        pIn += 8; pMatch += 8;
    }

    /* Fallback to 32-bit comparisons for remaining bytes */
    if (pIn < pInLimit - 3) {
        U32 diff = LZ4_read32(pMatch) ^ LZ4_read32(pIn);
        if (diff != 0) return LZ4_NbCommonBytes(diff) + (U32)(pIn - pStart);
        pIn += 4; pMatch += 4;
    }

    while (pIn < pInLimit - 3) {
        U32 diff = LZ4_read32(pMatch) ^ LZ4_read32(pIn);
        if (diff != 0) return (U32)(pIn - pStart) + LZ4_NbCommonBytes(diff);
        pIn += 4; pMatch += 4;
    }

    if (pIn < pInLimit - 1 && LZ4_read16(pMatch) == LZ4_read16(pIn)) {
        pIn += 2;
        pMatch += 2;
    }

    if (pIn < pInLimit && *pMatch == *pIn) pIn++;
    return (U32)(pIn - pStart);
}

// --- Compression Core (Global Table) ---
int lz4_compress_core_accelerated(
    __global const BYTE* restrict src,
    __global BYTE* restrict dst,
    int srcSize,
    int dstCapacity,
    int tableType,
    __global U64* restrict hashTable,
    int acceleration,
    U32 epoch
) {
    const __global BYTE* ip = src;
    __global BYTE* op = dst;
    __global BYTE* const oend = op + dstCapacity;
    const __global BYTE* const iend = ip + srcSize;
    const __global BYTE* anchor = ip;
    const __global BYTE* const mflimitPlusOne = iend - MFLIMIT + 1;
    const __global BYTE* const matchlimit = iend - LASTLITERALS;

    if (srcSize < (MFLIMIT+1)) goto _last_literals_g;

    U32 ipValue, forwardIpValue;
    U32 h_init = LZ4_hashPosition(ip, tableType, &ipValue);
    LZ4_putIndexOnHash(0, h_init, hashTable, tableType, ipValue, epoch);
    ip++;
    U32 forwardH = LZ4_hashPosition(ip, tableType, &forwardIpValue);

    for (;;) {
        const __global BYTE* match;
        __global BYTE* token;
        {
            const __global BYTE* forwardIp = ip;
            int step = 1;
            int searchMatchNb = acceleration << 6;
            do {
                U32 h_iter = forwardH;
                ipValue = forwardIpValue;
                U32 current = (U32)(forwardIp - src);
                int fp_match;
                U32 matchIndex = LZ4_getIndexOnHash(h_iter, hashTable, tableType, ipValue, &fp_match, epoch);
                ip = forwardIp;
                forwardIp += step;
                step = (searchMatchNb++ >> 6);
                if (forwardIp > mflimitPlusOne) goto _last_literals_g;
                forwardH = LZ4_hashPosition(forwardIp, tableType, &forwardIpValue);
                LZ4_putIndexOnHash(current, h_iter, hashTable, tableType, ipValue, epoch);
                if (fp_match && (matchIndex < current) && (current - matchIndex < LZ4_DISTANCE_MAX)) {
                    match = src + matchIndex;
                    if (LZ4_read32(match) == ipValue) break;
                }
            } while(1);
        }
        while ((ip > anchor) && (match > src) && (ip[-1] == match[-1])) { ip--; match--; }
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
            LZ4_lit_wildCopy8(op, anchor, op + litLength);
            op += litLength;
        }
_next_match_g:
        LZ4_write16(op, (U16)(ip - match)); op += 2;
        {
            unsigned matchCode = LZ4_count(ip + MINMATCH, match + MINMATCH, matchlimit);
            ip += matchCode + MINMATCH;
            if (matchCode >= ML_MASK) {
                *token += ML_MASK;
                matchCode -= ML_MASK;
                while (matchCode >= 255) { *op++ = 255; matchCode -= 255; }
                *op++ = (BYTE)matchCode;
            } else {
                *token += (BYTE)matchCode;
            }
        }
        anchor = ip;
        if (ip >= mflimitPlusOne) break;
        U32 seq2, seq_ip, seq_f;
        U32 h2 = LZ4_hashPosition(ip - 2, tableType, &seq2);
        LZ4_putIndexOnHash((U32)(ip - 2 - src), h2, hashTable, tableType, seq2, epoch);
        U32 h_ip = LZ4_hashPosition(ip, tableType, &seq_ip);
        U32 current = (U32)(ip - src);
        int fp_match;
        U32 matchIndex = LZ4_getIndexOnHash(h_ip, hashTable, tableType, seq_ip, &fp_match, epoch);
        LZ4_putIndexOnHash(current, h_ip, hashTable, tableType, seq_ip, epoch);
        if (fp_match && (matchIndex < current) && (current - matchIndex < LZ4_DISTANCE_MAX)) {
            if (LZ4_read32(src + matchIndex) == seq_ip) {
                token = op++; *token = 0; match = src + matchIndex; goto _next_match_g;
            }
        }
        forwardH = LZ4_hashPosition(++ip, tableType, &forwardIpValue);
    }
_last_literals_g:
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

#ifdef LZ4_GPU_ENABLE_STATS
    /* Write collected stats for this block */
    block_stats[stats_index * ST_N_STATS + ST_MATCH_SEARCH] = stat_match_search;
    block_stats[stats_index * ST_N_STATS + ST_MATCH_FOUND] = stat_match_found;
    block_stats[stats_index * ST_N_STATS + ST_LIT_EMITS] = stat_lit_emits;
    block_stats[stats_index * ST_N_STATS + ST_MATCH_EMITS] = stat_match_emits;
    block_stats[stats_index * ST_N_STATS + ST_MEMCPY_CALLS] = stat_memcpy_calls;
    block_stats[stats_index * ST_N_STATS + ST_MATCH_COMPARES] = stat_match_compares;
    block_stats[stats_index * ST_N_STATS + ST_MATCH_COMPARES_FAILED] = stat_match_compares_failed;
    block_stats[stats_index * ST_N_STATS + ST_SEARCH_ITERS] = stat_search_iters;
    block_stats[stats_index * ST_N_STATS + ST_FP_FILTERS] = stat_fp_filters;
    block_stats[stats_index * ST_N_STATS + ST_TABLE_INIT_WRITES] = stat_table_init_writes;
#endif

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

            {
                uint mlen_fast = (uint)(length + MINMATCH);
                if ((length != ML_MASK) && (match >= dst) && (offset >= mlen_fast)) {
                    LZ4_UA_COPYN(op, match, mlen_fast);
                    op += mlen_fast;
                    continue;
                }
            }

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
        LZ4_lit_wildCopy8(op, ip, cpy);
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
        LZ4_COPY_MATCH(op, match, (uint)length);
        op = cpy;
    }

    *outputSizePtr = (U32)(op - dst);
    return;

_output_error:
    *outputSizePtr = 0xFFFFFFFF;
}

// --- Kernels ---
#ifndef LZ4_GPU_NO_ENTRY_KERNELS

__kernel void lz4_compress_block(
    __global const BYTE* input,
    __global BYTE* output,
    __global U32* blockSizes,
    __global const U32* blockOffsets,
    __global const U32* outputOffsets,
    int totalBlocks,
    int inputSize,
    int tableType,
    int acceleration,
    int globalIndexBase,
    __global U64* globalHashTablePool,
    U32 epoch_base
) {
    const uint wi = get_global_id(0);
    const uint total_wi = get_global_size(0);
    const uint dict_entries = (1U << LZ4_HASHLOG);

    __global U64* dict = globalHashTablePool + (size_t)wi * dict_entries;
    U32 epoch = epoch_base + 1U;

    for (uint b = wi; b < (uint)totalBlocks; b += total_wi, ++epoch) {
        int start = blockOffsets[b * 2];
        int blockSize = blockOffsets[b * 2 + 1];
        if (start < inputSize && blockSize > 0) {
            int dstCapacity = blockSize + (blockSize / 255) + 256;
            __global BYTE* dst = output + outputOffsets[b];
            blockSizes[globalIndexBase + b] = lz4_compress_core_accelerated(
                input + start, dst, blockSize, dstCapacity, tableType, dict, acceleration, epoch
            );
        } else {
            blockSizes[globalIndexBase + b] = 0;
        }
    }
}

__kernel void lz4_decompress_block(
    const __global BYTE* input,
    __global BYTE* output,
    U32 compressed_offset,
    U32 compressed_size,
    U32 output_offset,
    U32 max_output_size,
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
    __global const U32* comp_offsets,
    __global const U32* comp_sizes,
    __global const U32* out_offsets,
    __global const U32* max_out_sizes,
    __global U32* sizes_out,
    U32 totalBlocks
) {
    int gid = get_global_id(0);
    int gsz = get_global_size(0);

    for (int idx = gid; idx < (int)totalBlocks; idx += gsz) {
        lz4_decompress_generic(
            input + comp_offsets[idx],
            output + out_offsets[idx],
            (int)comp_sizes[idx],
            (int)max_out_sizes[idx],
            &sizes_out[idx]
        );
    }
}

#endif /* LZ4_GPU_NO_ENTRY_KERNELS */

