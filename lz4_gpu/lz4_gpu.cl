/*
 * LZ4 GPU Parallel Implementation
 * Based on LZ4 v1.10.0 source code
 * Copyright (c) Yann Collet. BSD 2-Clause License
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

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

#ifndef LZ4_HASHLOG
#define LZ4_HASHLOG 14
#endif
#define LZ4_MEMORY_USAGE LZ4_HASHLOG

#ifndef LZ4_GPU_DICT_CLEAR
#define LZ4_GPU_DICT_CLEAR 0
#endif

#ifndef LZ4_GPU_DICT_ENTRY_BITS
#define LZ4_GPU_DICT_ENTRY_BITS 32
#endif

#if LZ4_GPU_DICT_ENTRY_BITS == 16
typedef unsigned short LZ4_DICT_ENTRY;
#else
typedef unsigned int LZ4_DICT_ENTRY;
#endif

// --- Types ---
typedef unsigned char BYTE;
typedef unsigned short U16;
typedef unsigned int U32;
typedef unsigned long U64;
typedef unsigned char U8;

#ifndef LZ4_GPU_DEBUG_COUNTERS_RUNTIME
#define LZ4_GPU_DEBUG_COUNTERS_RUNTIME 0
#endif

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
#include "lz4_gpu_debug.h"
#define LZ4_GPU_COMP_DEBUG_ARGS_DECL , __global U32* dbg_stats, U32 dbg_index, U32 dbg_enabled
#define LZ4_GPU_COMP_DEBUG_ARGS_PASS , dbg_stats, b, dbg_enabled
#define LZ4_GPU_DEC_DEBUG_ARGS_DECL , __global U32* dbg_stats, U32 dbg_index, U32 dbg_enabled
#define LZ4_GPU_DEC_DEBUG_ARGS_PASS , dbg_stats, idx, dbg_enabled

inline void lz4_gpu_debug_write_comp_stats(
    __global U32* dbg_stats,
    U32 dbg_index,
    U32 dbg_enabled,
    U32 search_iters,
    U32 hash_tag_hits,
    U32 distance_rejects,
    U32 match_found,
    U32 hash_inserts,
    U32 literal_bytes,
    U32 match_bytes,
    U32 lastlit_bytes
) {
    if (!dbg_enabled) return;
    {
        __global U32* block_stats = dbg_stats + (size_t)dbg_index * LZ4_DBG_COMP_N;
        block_stats[LZ4_DBG_COMP_SEARCH_ITERS] = search_iters;
        block_stats[LZ4_DBG_COMP_HASH_TAG_HITS] = hash_tag_hits;
        block_stats[LZ4_DBG_COMP_HASH_DISTANCE_REJECTS] = distance_rejects;
        block_stats[LZ4_DBG_COMP_MATCH_FOUND] = match_found;
        block_stats[LZ4_DBG_COMP_HASH_INSERTS] = hash_inserts;
        block_stats[LZ4_DBG_COMP_LITERAL_BYTES] = literal_bytes;
        block_stats[LZ4_DBG_COMP_MATCH_BYTES] = match_bytes;
        block_stats[LZ4_DBG_COMP_LASTLIT_BYTES] = lastlit_bytes;
    }
}

inline void lz4_gpu_debug_write_dec_stats(
    __global U32* dbg_stats,
    U32 dbg_index,
    U32 dbg_enabled,
    U32 tokens,
    U32 literal_bytes,
    U32 match_bytes,
    U32 small_offsets,
    U32 fast_literals,
    U32 fast_matches,
    U32 output_errors
) {
    if (!dbg_enabled) return;
    {
        __global U32* block_stats = dbg_stats + (size_t)dbg_index * LZ4_DBG_DEC_N;
        block_stats[LZ4_DBG_DEC_TOKENS] = tokens;
        block_stats[LZ4_DBG_DEC_LITERAL_BYTES] = literal_bytes;
        block_stats[LZ4_DBG_DEC_MATCH_BYTES] = match_bytes;
        block_stats[LZ4_DBG_DEC_SMALL_OFFSETS] = small_offsets;
        block_stats[LZ4_DBG_DEC_FAST_LITERAL_PATHS] = fast_literals;
        block_stats[LZ4_DBG_DEC_FAST_MATCH_PATHS] = fast_matches;
        block_stats[LZ4_DBG_DEC_OUTPUT_ERROR] = output_errors;
    }
}
#else
#define LZ4_GPU_COMP_DEBUG_ARGS_DECL
#define LZ4_GPU_COMP_DEBUG_ARGS_PASS
#define LZ4_GPU_DEC_DEBUG_ARGS_DECL
#define LZ4_GPU_DEC_DEBUG_ARGS_PASS
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

// --- Fast Direct Match Copy Helper (M2 backport from v2-r11) ---
inline void lz4_v1_fast_direct_match_copy_18(__global BYTE* op,
                                             const __global BYTE* match,
                                             uint mlen_fast) {
    if (mlen_fast <= 8U) {
        LZ4_write64(op, LZ4_read64(match));
        return;
    }
    LZ4_write64(op, LZ4_read64(match));
    LZ4_write64(op + 8, LZ4_read64(match + 8));
    if (mlen_fast > 16U) {
        op[16] = match[16];
        if (mlen_fast > 17U) op[17] = match[17];
    }
}

// --- Hashing ---
inline U32 LZ4_hash4(U32 sequence) __attribute__((always_inline)) {
    return ((sequence * 2654435761U) >> ((MINMATCH*8)-LZ4_HASHLOG));
}

inline U32 LZ4_hashPosition(const __global BYTE* p, U32* sequence) __attribute__((always_inline)) {
    U32 s = LZ4_read32(p);
    *sequence = s;
    return LZ4_hash4(s);
}

inline U32 LZ4_hashMask(void) __attribute__((always_inline)) {
    return (1U << LZ4_HASHLOG) - 1U;
}

inline void LZ4_putIndexOnHashMasked(U32 idx, U32 h, __global LZ4_DICT_ENTRY* tableBase, U32 mask, U32 epoch) __attribute__((always_inline)) {
#if LZ4_GPU_DICT_CLEAR
    (void)epoch;
    tableBase[h & mask] = (LZ4_DICT_ENTRY)(idx + 1U);
#else
    /* Compact 32-bit entry: [12-bit epoch | 20-bit low position]. */
    U32 packed = ((epoch & 0xFFF) << 20) | (idx & 0xFFFFF);
    tableBase[h & mask] = packed;
#endif
}

inline U32 LZ4_getIndexOnHashMasked(U32 h, __global LZ4_DICT_ENTRY* tableBase, U32 mask, int* entry_valid, U32 epoch, U32 current) __attribute__((always_inline)) {
    U32 const packed = tableBase[h & mask];
#if LZ4_GPU_DICT_CLEAR
    (void)epoch;
    (void)current;
    if (packed == 0) {
        *entry_valid = 0;
        return 0;
    }
    *entry_valid = 1;
    return packed - 1U;
#else
    U32 const tag = (packed >> 20);
    if (tag != (epoch & 0xFFF)) {
        *entry_valid = 0;
        return 0;
    }
    *entry_valid = 1;
    {
        U32 pos20 = packed & 0xFFFFF;
        U32 matchIndex = (current & 0xFFF00000U) | pos20;
        if (matchIndex > current) matchIndex -= 0x100000U;
        return matchIndex;
    }
#endif
}

inline void LZ4_clearDictEntries(__global LZ4_DICT_ENTRY* dict, uint dict_entries) __attribute__((always_inline)) {
#if LZ4_GPU_DICT_CLEAR
#if LZ4_GPU_DICT_ENTRY_BITS == 16
    __global uint4* dict4 = (__global uint4*)dict;
    uint vec_count = dict_entries >> 3;
    uint tail = dict_entries & 7U;
    for (uint i = 0; i < vec_count; ++i) dict4[i] = (uint4)(0U, 0U, 0U, 0U);
    if (tail != 0) {
        __global ushort* dict16 = (__global ushort*)dict;
        uint base = vec_count << 3;
        for (uint i = 0; i < tail; ++i) dict16[base + i] = (ushort)0;
    }
#else
    __global uint4* dict4 = (__global uint4*)dict;
    uint vec_count = dict_entries >> 2;
    uint tail = dict_entries & 3U;
    for (uint i = 0; i < vec_count; ++i) dict4[i] = (uint4)(0U, 0U, 0U, 0U);
    if (tail != 0) {
        __global uint* dict32 = (__global uint*)dict;
        uint base = vec_count << 2;
        for (uint i = 0; i < tail; ++i) dict32[base + i] = 0U;
    }
#endif
#else
    (void)dict;
    (void)dict_entries;
#endif
}

inline void LZ4_putIndexOnHash(U32 idx, U32 h, __global LZ4_DICT_ENTRY* tableBase, U32 epoch) __attribute__((always_inline)) {
    U32 const mask = LZ4_hashMask();
    LZ4_putIndexOnHashMasked(idx, h, tableBase, mask, epoch);
}

inline U32 LZ4_getIndexOnHash(U32 h, __global LZ4_DICT_ENTRY* tableBase, int* entry_valid, U32 epoch, U32 current) __attribute__((always_inline)) {
    U32 const mask = LZ4_hashMask();
    return LZ4_getIndexOnHashMasked(h, tableBase, mask, entry_valid, epoch, current);
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

    /* Batch 16-byte compares first to reduce loop/branch overhead on long matches,
     * while keeping overhead low on short matches. */
    while (pIn < pInLimit - 15) {
        U64 diff0 = LZ4_read64(pMatch) ^ LZ4_read64(pIn);
        if (diff0 != 0ULL) return (U32)(pIn - pStart) + LZ4_NbCommonBytes64(diff0);

        U64 diff1 = LZ4_read64(pMatch + 8) ^ LZ4_read64(pIn + 8);
        if (diff1 != 0ULL) return (U32)(pIn - pStart) + 8 + LZ4_NbCommonBytes64(diff1);

        pIn += 16;
        pMatch += 16;
    }

    while (pIn < pInLimit - 7) {
        U64 diff = LZ4_read64(pMatch) ^ LZ4_read64(pIn);
        if (diff != 0ULL) return (U32)(pIn - pStart) + LZ4_NbCommonBytes64(diff);
        pIn += 8; pMatch += 8;
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
    __global LZ4_DICT_ENTRY* restrict hashTable,
    int acceleration,
    U32 epoch
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    , __global U32* dbg_stats,
    U32 dbg_index,
    U32 dbg_enabled
#endif
) {
    const __global BYTE* ip = src;
    __global BYTE* op = dst;
    __global BYTE* const oend = op + dstCapacity;
    const __global BYTE* const iend = ip + srcSize;
    const __global BYTE* anchor = ip;
    const __global BYTE* const mflimitPlusOne = iend - MFLIMIT + 1;
    const __global BYTE* const matchlimit = iend - LASTLITERALS;
    const U32 hashMask = LZ4_hashMask();

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    U32 stat_search_iters = 0;
    U32 stat_hash_tag_hits = 0;
    U32 stat_distance_rejects = 0;
    U32 stat_match_found = 0;
    U32 stat_hash_inserts = 0;
    U32 stat_literal_bytes = 0;
    U32 stat_match_bytes = 0;
    U32 stat_lastlit_bytes = 0;
#endif

    if (srcSize < (MFLIMIT+1)) goto _last_literals_g;

    U32 ipValue, forwardIpValue;
    U32 h_init = LZ4_hashPosition(ip, &ipValue);
    LZ4_putIndexOnHashMasked(0, h_init, hashTable, hashMask, epoch);
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    if (dbg_enabled) stat_hash_inserts++;
#endif
    ip++;
    U32 forwardH = LZ4_hashPosition(ip, &forwardIpValue);

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
                int entry_valid;
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                if (dbg_enabled) stat_search_iters++;
#endif
                U32 matchIndex = LZ4_getIndexOnHashMasked(h_iter, hashTable, hashMask, &entry_valid, epoch, current);
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                if (dbg_enabled && entry_valid) stat_hash_tag_hits++;
#endif
                ip = forwardIp;
                forwardIp += step;
                step = (searchMatchNb++ >> 6);
                if (forwardIp > mflimitPlusOne) goto _last_literals_g;
                forwardH = LZ4_hashPosition(forwardIp, &forwardIpValue);
                LZ4_putIndexOnHashMasked(current, h_iter, hashTable, hashMask, epoch);
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                if (dbg_enabled) stat_hash_inserts++;
#endif
                if (entry_valid && (matchIndex < current) && (current - matchIndex < LZ4_DISTANCE_MAX)) {
                    match = src + matchIndex;
                    if (LZ4_read32(match) == ipValue) {
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                        if (dbg_enabled) stat_match_found++;
#endif
                        break;
                    }
                } else if (entry_valid) {
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                    if (dbg_enabled) stat_distance_rejects++;
#endif
                }
            } while(1);
        }
        while ((ip > anchor) && (match > src) && (ip[-1] == match[-1])) { ip--; match--; }
        {
            unsigned litLength = (unsigned)(ip - anchor);
            token = op++;
            if (op + litLength + (2 + 1 + LASTLITERALS) + (litLength/255) > oend) {
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                lz4_gpu_debug_write_comp_stats(dbg_stats, dbg_index, dbg_enabled,
                                               stat_search_iters, stat_hash_tag_hits, stat_distance_rejects,
                                               stat_match_found, stat_hash_inserts, stat_literal_bytes,
                                               stat_match_bytes, stat_lastlit_bytes);
#endif
                return 0;
            }
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            if (dbg_enabled) stat_literal_bytes += litLength;
#endif
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
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            if (dbg_enabled) stat_match_bytes += (matchCode + MINMATCH);
#endif
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
        U32 seq2, seq_ip;
        U32 h2 = LZ4_hashPosition(ip - 2, &seq2);
        LZ4_putIndexOnHashMasked((U32)(ip - 2 - src), h2, hashTable, hashMask, epoch);
    #if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
        if (dbg_enabled) stat_hash_inserts++;
    #endif
        U32 h_ip = LZ4_hashPosition(ip, &seq_ip);
        U32 current = (U32)(ip - src);
        int entry_valid;
    #if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
        if (dbg_enabled) stat_search_iters++;
    #endif
        U32 matchIndex = LZ4_getIndexOnHashMasked(h_ip, hashTable, hashMask, &entry_valid, epoch, current);
    #if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
        if (dbg_enabled && entry_valid) stat_hash_tag_hits++;
    #endif
        LZ4_putIndexOnHashMasked(current, h_ip, hashTable, hashMask, epoch);
    #if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
        if (dbg_enabled) stat_hash_inserts++;
    #endif
        if (entry_valid && (matchIndex < current) && (current - matchIndex < LZ4_DISTANCE_MAX)) {
            if (LZ4_read32(src + matchIndex) == seq_ip) {
    #if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            if (dbg_enabled) stat_match_found++;
    #endif
                token = op++; *token = 0; match = src + matchIndex; goto _next_match_g;
            }
        } else if (entry_valid) {
    #if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            if (dbg_enabled) stat_distance_rejects++;
    #endif
        }
        forwardH = LZ4_hashPosition(++ip, &forwardIpValue);
    }
_last_literals_g:
    {
        size_t lastRun = (size_t)(iend - anchor);
        if (op + lastRun + 1 + ((lastRun + 255 - RUN_MASK) / 255) > oend) {
    #if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            lz4_gpu_debug_write_comp_stats(dbg_stats, dbg_index, dbg_enabled,
                           stat_search_iters, stat_hash_tag_hits, stat_distance_rejects,
                           stat_match_found, stat_hash_inserts, stat_literal_bytes,
                           stat_match_bytes, stat_lastlit_bytes);
    #endif
            return 0;
        }
    #if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
        if (dbg_enabled) stat_lastlit_bytes += (U32)lastRun;
    #endif
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
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    lz4_gpu_debug_write_comp_stats(dbg_stats, dbg_index, dbg_enabled,
                                   stat_search_iters, stat_hash_tag_hits, stat_distance_rejects,
                                   stat_match_found, stat_hash_inserts, stat_literal_bytes,
                                   stat_match_bytes, stat_lastlit_bytes);
#endif
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
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    , __global U32* dbg_stats,
    U32 dbg_index,
    U32 dbg_enabled
#endif
) {
    const __global BYTE* ip = src;
    const __global BYTE* const iend = ip + srcSize;
    const U32 output_end = (U32)outputSize;
    const U32 fast_output_limit = (outputSize >= 32) ? (output_end - 32U) : 0U;
    const uint fast_output_ok = (outputSize >= 32) ? 1U : 0U;
    __global BYTE* const oend = dst + outputSize;
    U32 op_rel = 0U;

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    U32 stat_tokens = 0;
    U32 stat_literal_bytes = 0;
    U32 stat_match_bytes = 0;
    U32 stat_small_offsets = 0;
    U32 stat_fast_literals = 0;
    U32 stat_fast_matches = 0;
    U32 stat_output_errors = 0;
#endif

    const __global BYTE* const shortiend = iend - 14 - 2;
    const __global BYTE* const shortoend = oend - 14 - 18;

    if (srcSize == 0) {
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
        lz4_gpu_debug_write_dec_stats(dbg_stats, dbg_index, dbg_enabled,
                                      stat_tokens, stat_literal_bytes, stat_match_bytes,
                                      stat_small_offsets, stat_fast_literals, stat_fast_matches,
                                      stat_output_errors);
#endif
        *outputSizePtr = 0;
        return;
    }

    for (;;) {
        unsigned token = *ip++;
        U32 length = token >> ML_BITS;
        U32 offset = 0U;
        U32 match_rel = 0U;
        __global BYTE* op;
        __global BYTE* match;
        __global BYTE* cpy;

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
        if (dbg_enabled) stat_tokens++;
#endif

        // Fast path
        if ((length != RUN_MASK) && (ip < shortiend) && fast_output_ok && (op_rel <= fast_output_limit)) {
            op = dst + (size_t)op_rel;
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            if (dbg_enabled) {
                stat_fast_literals++;
                stat_literal_bytes += length;
            }
#endif
            if (length <= 8U) {
                LZ4_write64(op, LZ4_read64(ip));
            } else {
                LZ4_memcpy(op, ip, 16);
            }
            op_rel += length;
            ip += length;

            length = (U32)(token & ML_MASK);
            offset = LZ4_readLE16(ip);
            ip += 2;
            if (offset > op_rel) goto _output_error;
            match_rel = op_rel - offset;
            op = dst + (size_t)op_rel;
            match = dst + (size_t)match_rel;

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            if (dbg_enabled && offset < 8) stat_small_offsets++;
#endif

            {
                U32 mlen_fast = length + MINMATCH;
                if ((length != ML_MASK) && (offset >= mlen_fast)) {
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                    if (dbg_enabled) {
                        stat_fast_matches++;
                        stat_match_bytes += mlen_fast;
                    }
#endif
                    lz4_v1_fast_direct_match_copy_18(op, match, mlen_fast);
                    op_rel += mlen_fast;
                    continue;
                }
            }

            if ((length != ML_MASK) && (offset >= 8)) {
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                if (dbg_enabled) {
                    stat_fast_matches++;
                    stat_match_bytes += (length + MINMATCH);
                }
#endif
                if (length <= 4U) {
                    LZ4_write64(op, LZ4_read64(match));
                } else {
                    LZ4_write64(op, LZ4_read64(match));
                    LZ4_write64(op + 8, LZ4_read64(match + 8));
                    if (length > 12U) {
                        op[16] = match[16];
                        if (length > 13U) op[17] = match[17];
                    }
                }
                op_rel += length + MINMATCH;
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

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    if (dbg_enabled) stat_literal_bytes += length;
#endif

        // Copy literals
    op = dst + (size_t)op_rel;
        cpy = op + length;
        if ((cpy > oend - MFLIMIT) || (ip + length > iend - (2 + 1 + LASTLITERALS))) {
            if (ip + length != iend) goto _output_error;
            if (cpy > oend) goto _output_error;
            LZ4_memcpy(op, ip, length);
        op_rel += length;
            break; // End of block
        }
        LZ4_lit_wildCopy8(op, ip, cpy);
    ip += length;
    op_rel += length;

        // Get offset
    offset = LZ4_readLE16(ip);
    ip += 2;
    if (offset > op_rel) goto _output_error;
    match_rel = op_rel - offset;
    length = (U32)(token & ML_MASK);

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    if (dbg_enabled && offset < 8) stat_small_offsets++;
#endif

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

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    if (dbg_enabled) stat_match_bytes += length;
#endif

    op = dst + (size_t)op_rel;
    match = dst + (size_t)match_rel;

        cpy = op + length;
        LZ4_COPY_MATCH(op, match, (uint)length);
    op_rel += length;
    }

#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    lz4_gpu_debug_write_dec_stats(dbg_stats, dbg_index, dbg_enabled,
                                  stat_tokens, stat_literal_bytes, stat_match_bytes,
                                  stat_small_offsets, stat_fast_literals, stat_fast_matches,
                                  stat_output_errors);
#endif
    *outputSizePtr = op_rel;
    return;

_output_error:
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    if (dbg_enabled) stat_output_errors++;
    lz4_gpu_debug_write_dec_stats(dbg_stats, dbg_index, dbg_enabled,
                                  stat_tokens, stat_literal_bytes, stat_match_bytes,
                                  stat_small_offsets, stat_fast_literals, stat_fast_matches,
                                  stat_output_errors);
#endif
    *outputSizePtr = 0xFFFFFFFF;
}


// --- Two-phase (intra-block parallel) core ---------------------------------
// Compress base[lo..hi) as a self-contained LZ4 substream. Matches may reference
// [0,current) within the block (cross-segment). ownTable = private evolving hash
// (epoch-tagged, no clear). dictTable(+hasDict) = read-only dense prefix cross
// dict of [0,lo). Same greedy logic as lz4_compress_core_accelerated, plus a dict
// fallback in the two candidate-search sites. Returns compressed size, 0 on overflow.
int lz4_compress_core_seg(
    __global const BYTE* restrict base,
    int lo, int hi,
    __global BYTE* restrict dst,
    int dstCapacity,
    __global LZ4_DICT_ENTRY* restrict ownTable,
    __global LZ4_DICT_ENTRY* restrict dictTable,
    int hasDict,
    U32 epoch)
{
    const __global BYTE* ip = base + lo;
    __global BYTE* op = dst;
    __global BYTE* const oend = op + dstCapacity;
    const __global BYTE* const iend = base + hi;
    const __global BYTE* anchor = ip;
    const __global BYTE* const mflimitPlusOne = iend - MFLIMIT + 1;
    const __global BYTE* const matchlimit = iend - LASTLITERALS;
    const U32 hashMask = LZ4_hashMask();

    if ((hi - lo) < (MFLIMIT+1)) goto _last_literals_seg;

    U32 ipValue, forwardIpValue;
    U32 h_init = LZ4_hashPosition(ip, &ipValue);
    LZ4_putIndexOnHashMasked((U32)(ip - base), h_init, ownTable, hashMask, epoch);
    ip++;
    U32 forwardH = LZ4_hashPosition(ip, &forwardIpValue);

    for (;;) {
        const __global BYTE* match;
        __global BYTE* token;
        {
            const __global BYTE* forwardIp = ip;
            int step = 1;
            int searchMatchNb = 1 << 6;
            do {
                U32 h_iter = forwardH;
                ipValue = forwardIpValue;
                U32 current = (U32)(forwardIp - base);
                int entry_valid;
                U32 matchIndex = LZ4_getIndexOnHashMasked(h_iter, ownTable, hashMask, &entry_valid, epoch, current);
                ip = forwardIp;
                forwardIp += step;
                step = (searchMatchNb++ >> 6);
                if (forwardIp > mflimitPlusOne) goto _last_literals_seg;
                forwardH = LZ4_hashPosition(forwardIp, &forwardIpValue);
                LZ4_putIndexOnHashMasked(current, h_iter, ownTable, hashMask, epoch);
                const __global BYTE* cand = 0;
                if (entry_valid && (matchIndex < current) && (current - matchIndex < LZ4_DISTANCE_MAX)
                    && LZ4_read32(base + matchIndex) == ipValue) {
                    cand = base + matchIndex;
                }
                if (!cand && hasDict) {
                    int dvalid;
                    U32 dIdx = LZ4_getIndexOnHashMasked(h_iter, dictTable, hashMask, &dvalid, epoch, current);
                    if (dvalid && (dIdx < current) && (current - dIdx < LZ4_DISTANCE_MAX)
                        && LZ4_read32(base + dIdx) == ipValue) {
                        cand = base + dIdx;
                    }
                }
                if (cand) { match = cand; break; }
            } while (1);
        }
        while ((ip > anchor) && (match > base) && (ip[-1] == match[-1])) { ip--; match--; }
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
_next_match_seg:
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
        {
            U32 seq2;
            U32 h2 = LZ4_hashPosition(ip - 2, &seq2);
            LZ4_putIndexOnHashMasked((U32)(ip - 2 - base), h2, ownTable, hashMask, epoch);
        }
        {
            U32 seq_ip;
            U32 h_ip = LZ4_hashPosition(ip, &seq_ip);
            U32 current = (U32)(ip - base);
            int entry_valid;
            U32 matchIndex = LZ4_getIndexOnHashMasked(h_ip, ownTable, hashMask, &entry_valid, epoch, current);
            LZ4_putIndexOnHashMasked(current, h_ip, ownTable, hashMask, epoch);
            const __global BYTE* cand2 = 0;
            if (entry_valid && (matchIndex < current) && (current - matchIndex < LZ4_DISTANCE_MAX)
                && LZ4_read32(base + matchIndex) == seq_ip) {
                cand2 = base + matchIndex;
            }
            if (!cand2 && hasDict) {
                int dvalid;
                U32 dIdx = LZ4_getIndexOnHashMasked(h_ip, dictTable, hashMask, &dvalid, epoch, current);
                if (dvalid && (dIdx < current) && (current - dIdx < LZ4_DISTANCE_MAX)
                    && LZ4_read32(base + dIdx) == seq_ip) {
                    cand2 = base + dIdx;
                }
            }
            if (cand2) { token = op++; *token = 0; match = cand2; goto _next_match_seg; }
        }
        forwardH = LZ4_hashPosition(++ip, &forwardIpValue);
    }
_last_literals_seg:
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

// Decode one substream into shared block buffer `out`, op starting at opStart
// (absolute within the block). Matches resolve as out[op-offset], reaching back
// into earlier segments already decoded contiguously. Returns new op, or -1.
int lz4_decompress_seg(const __global BYTE* src, int srcSize,
                       __global BYTE* out, int opStart, int outCap)
{
    const __global BYTE* ip = src;
    const __global BYTE* const iend = src + srcSize;
    int op = opStart;
    if (srcSize == 0) return op;
    for (;;) {
        unsigned token = *ip++;
        U32 ll = token >> ML_BITS;
        if (ll == RUN_MASK) { unsigned s; do { if (ip >= iend) return -1; s = *ip++; ll += s; } while (s == 255); }
        if (op + (int)ll > outCap) return -1;
        if (ip + ll > iend) return -1;
        LZ4_UA_COPYN(out + op, ip, ll); op += (int)ll; ip += ll;
        if (ip == iend) break;
        if (ip + 2 > iend) return -1;
        U32 off = LZ4_readLE16(ip); ip += 2;
        if (off == 0 || off > (U32)op) return -1;
        int mpos = op - (int)off;
        U32 ml = token & ML_MASK;
        if (ml == ML_MASK) { unsigned s; do { if (ip >= iend) return -1; s = *ip++; ml += s; } while (s == 255); }
        ml += MINMATCH;
        if (op + (int)ml > outCap) return -1;
        LZ4_COPY_MATCH(out + op, out + mpos, (uint)ml);
        op += (int)ml;
    }
    return op;
}

// --- Kernels ---
#ifndef LZ4_GPU_NO_ENTRY_KERNELS

__kernel void lz4_compress_block(
    __global const BYTE* input,
    __global BYTE* output,
    __global U32* blockSizes,
    int totalBlocks,
    int inputSize,
    int blockSize,
    int singleBlockMaxOut,
    int acceleration,
    int globalIndexBase,
    __global LZ4_DICT_ENTRY* globalHashTablePool,
    U32 active_lanes,
    U32 epoch_base
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    , __global U32* dbg_stats,
    U32 dbg_enabled
#endif
) {
    const uint wi = get_global_id(0);
    const uint total_wi = active_lanes;
    const uint dict_entries = (1U << LZ4_HASHLOG);

    if (wi >= total_wi) return;

    __global LZ4_DICT_ENTRY* dict = globalHashTablePool + (size_t)wi * dict_entries;
    U32 epoch = epoch_base + 1U;

    for (uint b = wi; b < (uint)totalBlocks; b += total_wi, ++epoch) {
        int start = (int)b * blockSize;
        int remain = inputSize - start;
        int thisBlockSize = (remain > blockSize) ? blockSize : remain;
        if (start < inputSize && thisBlockSize > 0) {
#if LZ4_GPU_DICT_CLEAR
            LZ4_clearDictEntries(dict, dict_entries);
#endif
            int dstCapacity = singleBlockMaxOut;
            __global BYTE* dst = output + (size_t)b * (size_t)singleBlockMaxOut;
            blockSizes[globalIndexBase + b] = lz4_compress_core_accelerated(
                input + start, dst, thisBlockSize, dstCapacity, dict, acceleration, epoch
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                , dbg_stats, b, dbg_enabled
#endif
            );
        } else {
            blockSizes[globalIndexBase + b] = 0;
        }
    }
}

__kernel void lz4_compress_blocks_mapped(
    __global const BYTE* input,
    __global BYTE* output,
    __global U32* blockSizes,
    __global const U32* blockIndices,
    int totalBlocks,
    int inputSize,
    int blockSize,
    int singleBlockMaxOut,
    int acceleration,
    int globalIndexBase,
    __global LZ4_DICT_ENTRY* globalHashTablePool,
    U32 active_lanes,
    U32 epoch_base
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    , __global U32* dbg_stats,
    U32 dbg_enabled
#endif
) {
    const uint wi = get_global_id(0);
    const uint total_wi = active_lanes;
    const uint dict_entries = (1U << LZ4_HASHLOG);

    if (wi >= total_wi) return;

    __global LZ4_DICT_ENTRY* dict = globalHashTablePool + (size_t)wi * dict_entries;
    U32 epoch = epoch_base + 1U;

    for (uint b = wi; b < (uint)totalBlocks; b += total_wi, ++epoch) {
        int srcBlock = (int)blockIndices[b];
        int start = srcBlock * blockSize;
        int remain = inputSize - start;
        int thisBlockSize = (remain > blockSize) ? blockSize : remain;
        if (start < inputSize && thisBlockSize > 0) {
#if LZ4_GPU_DICT_CLEAR
            LZ4_clearDictEntries(dict, dict_entries);
#endif
            int dstCapacity = singleBlockMaxOut;
            __global BYTE* dst = output + (size_t)b * (size_t)singleBlockMaxOut;
            blockSizes[globalIndexBase + b] = lz4_compress_core_accelerated(
                input + start, dst, thisBlockSize, dstCapacity, dict, acceleration, epoch
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
                , dbg_stats, b, dbg_enabled
#endif
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
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    , __global U32* dbg_stats,
    U32 dbg_enabled
#endif
) {
    lz4_decompress_generic(
        input + compressed_offset,
        output + output_offset,
        (int)compressed_size,
        (int)max_output_size,
        &output_sizes[block_index]
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
        , dbg_stats, block_index, dbg_enabled
#endif
    );
}

__kernel void lz4_decompress_blocks(
    const __global BYTE* input,
    __global BYTE* output,
    __global const U32* comp_offsets,
    __global const U32* comp_sizes,
    __global U32* sizes_out,
    U32 block_size,
    U32 totalBlocks
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    , __global U32* dbg_stats,
    U32 dbg_enabled
#endif
) {
    int gid = get_global_id(0);
    int gsz = get_global_size(0);

    for (int idx = gid; idx < (int)totalBlocks; idx += gsz) {
        lz4_decompress_generic(
            input + comp_offsets[idx],
            output + (size_t)idx * (size_t)block_size,
            (int)comp_sizes[idx],
            (int)block_size,
            &sizes_out[idx]
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            , dbg_stats, (U32)idx, dbg_enabled
#endif
        );
    }
}

__kernel void lz4_decompress_blocks_mapped(
    const __global BYTE* input,
    __global BYTE* output,
    __global const U32* comp_offsets,
    __global const U32* comp_sizes,
    __global const U32* block_indices,
    __global U32* sizes_out,
    U32 block_size,
    U32 totalMappedBlocks
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
    , __global U32* dbg_stats,
    U32 dbg_enabled
#endif
) {
    int gid = get_global_id(0);
    int gsz = get_global_size(0);

    for (int idx = gid; idx < (int)totalMappedBlocks; idx += gsz) {
        U32 src_idx = block_indices[idx];
        lz4_decompress_generic(
            input + comp_offsets[src_idx],
            output + (size_t)idx * (size_t)block_size,
            (int)comp_sizes[src_idx],
            (int)block_size,
            &sizes_out[idx]
#if LZ4_GPU_DEBUG_COUNTERS_RUNTIME
            , dbg_stats, (U32)idx, dbg_enabled
#endif
        );
    }
}

#ifndef LZ4_TP_BUILD_CHUNKS
#define LZ4_TP_BUILD_CHUNKS 4
#endif

// Build N-1 dense cumulative prefix dicts per block: dict k (k=1..N-1) hashes
// [0,k*segLen) so segment s reads dict s to reference all earlier segments.
// Layout: prefixPool[(b*(N-1)+(k-1)) * dict_entries]. Epoch-tagged (no clear).
__kernel void lz4_tp_build_prefix(
    __global const BYTE* input,
    __global LZ4_DICT_ENTRY* prefixPool,
    int nblk, int inputSize, int blockSize, int N, U32 epoch)
{
    uint g = get_global_id(0);
    int nk = N - 1;
    if (nk < 1) return;
    const int CH = LZ4_TP_BUILD_CHUNKS;
    if (g >= (uint)(nblk * nk * CH)) return;
    const uint dict_entries = (1U << LZ4_HASHLOG);
    const U32 hashMask = LZ4_hashMask();
    int c = (int)g % CH;
    int idx = (int)g / CH;
    int b = idx / nk;
    int k = idx % nk + 1;
    int start = b * blockSize;
    int L = inputSize - start; if (L > blockSize) L = blockSize;
    if (L <= 0) return;
    int segLen = (L + N - 1) / N;
    int prefixEnd = k * segLen; if (prefixEnd > L) prefixEnd = L;
    int chunk = (prefixEnd + CH - 1) / CH;
    int clo = c * chunk, chi = clo + chunk; if (chi > prefixEnd) chi = prefixEnd;
    __global const BYTE* base = input + start;
    __global LZ4_DICT_ENTRY* dict = prefixPool + (size_t)(b * nk + (k - 1)) * dict_entries;
    for (int i = clo; i + MINMATCH <= chi; i++) {
        U32 seq;
        U32 h = LZ4_hashPosition(base + i, &seq);
        /* I4-determinism fix: atomic_max keeps the highest idx (= nearest match,
           standard LZ4's own preference) regardless of chunk scheduling, so the
           shared prefix table is byte-identical across devices and runs. */
#if LZ4_GPU_DICT_CLEAR
        atomic_max((volatile __global uint*)&dict[h & hashMask], (U32)i + 1U);
#else
        atomic_max((volatile __global uint*)&dict[h & hashMask],
                   ((epoch & 0xFFF) << 20) | ((U32)i & 0xFFFFF));
#endif
    }
}

// Scan: N work-items per block, one per segment; segment s>0 reads prefix dict s.
__kernel void lz4_tp_scan_seg(
    __global const BYTE* input,
    __global BYTE* out,
    __global U32* sizes,
    __global LZ4_DICT_ENTRY* ownPool,
    __global LZ4_DICT_ENTRY* prefixPool,
    int nblk, int inputSize, int blockSize, int segMaxOut, int N, U32 epoch)
{
    uint g = get_global_id(0);
    if (g >= (uint)(nblk * N)) return;
    const uint dict_entries = (1U << LZ4_HASHLOG);
    int b = (int)g / N, s = (int)g % N;
    int nk = N - 1;
    int start = b * blockSize;
    int L = inputSize - start; if (L > blockSize) L = blockSize;
    __global const BYTE* base = input + start;
    int segLen = (L + N - 1) / N;
    int lo = s * segLen, hi = lo + segLen; if (hi > L) hi = L;
    if (L <= 0 || lo >= L) {
        /* Canonical empty LZ4 block: one zero token, decoding to zero bytes. */
        out[(size_t)g * segMaxOut] = 0;
        sizes[g] = 1;
        return;
    }
    __global LZ4_DICT_ENTRY* ownTable = ownPool + (size_t)g * dict_entries;
    __global LZ4_DICT_ENTRY* dict = (s > 0) ? prefixPool + (size_t)(b * nk + (s - 1)) * dict_entries : 0;
    sizes[g] = (U32)lz4_compress_core_seg(base, lo, hi, out + (size_t)g * segMaxOut, segMaxOut,
                                          ownTable, dict, s > 0 ? 1 : 0, epoch);
}

// Decode: one work-item per block decodes its N segments in order into output.
__kernel void lz4_tp_decompress_segmented(
    __global const BYTE* input,
    __global BYTE* output,
    __global const U32* comp_offsets,   // per (block,segment)
    __global const U32* comp_sizes,     // per (block,segment)
    __global U32* sizes_out,            // per block
    U32 block_size, int N, U32 totalBlocks)
{
    int gid = get_global_id(0), gsz = get_global_size(0);
    for (int idx = gid; idx < (int)totalBlocks; idx += gsz) {
        int op = 0, ok = 1;
        for (int s = 0; s < N; s++) {
            int gg = idx * N + s;
            op = lz4_decompress_seg(input + comp_offsets[gg], (int)comp_sizes[gg],
                                    output + (size_t)idx * block_size, op, (int)block_size);
            if (op < 0) { ok = 0; break; }
        }
        sizes_out[idx] = ok ? (U32)op : 0xFFFFFFFF;
    }
}

#endif /* LZ4_GPU_NO_ENTRY_KERNELS */
