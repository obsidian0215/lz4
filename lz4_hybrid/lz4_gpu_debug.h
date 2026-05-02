#ifndef LZ4_GPU_DEBUG_H
#define LZ4_GPU_DEBUG_H

#define LZ4_DBG_COMP_SEARCH_ITERS 0
#define LZ4_DBG_COMP_HASH_TAG_HITS 1
#define LZ4_DBG_COMP_HASH_DISTANCE_REJECTS 2
#define LZ4_DBG_COMP_MATCH_FOUND 3
#define LZ4_DBG_COMP_HASH_INSERTS 4
#define LZ4_DBG_COMP_LITERAL_BYTES 5
#define LZ4_DBG_COMP_MATCH_BYTES 6
#define LZ4_DBG_COMP_LASTLIT_BYTES 7
#define LZ4_DBG_COMP_N 8

#define LZ4_DBG_DEC_TOKENS 0
#define LZ4_DBG_DEC_LITERAL_BYTES 1
#define LZ4_DBG_DEC_MATCH_BYTES 2
#define LZ4_DBG_DEC_SMALL_OFFSETS 3
#define LZ4_DBG_DEC_FAST_LITERAL_PATHS 4
#define LZ4_DBG_DEC_FAST_MATCH_PATHS 5
#define LZ4_DBG_DEC_OUTPUT_ERROR 6
#define LZ4_DBG_DEC_N 7

typedef struct {
    int enabled;
    int block_limit;
} lz4_gpu_debug_config_t;

lz4_gpu_debug_config_t lz4_gpu_get_debug_config(void);

#endif
