/*
 * LZ4 GPU Compressor Host Interface (C version)
 * Pure C interface for LZ4 frame compression on GPU via OpenCL
 */

#ifndef LZ4_GPU_HOST_H
#define LZ4_GPU_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants
#define LZ4_GPU_MAX_BLOCK_SIZE (64 * 1024)  // 64KB
#define LZ4_GPU_MIN_BLOCK_SIZE (4 * 1024)   // 4KB - used as conservative min when parsing frames
#define LZ4_GPU_HASH_TABLE_SIZE (1 << 14)         // 16384 entries (must match LZ4_HASHLOG in kernel)
#define LZ4F_HEADER_SIZE_MAX 19
#define LZ4F_ENDMARK_SIZE 4
#define LZ4F_CONTENT_CHECKSUM_SIZE 4

// Error codes
typedef enum {
    LZ4_GPU_SUCCESS = 0,
    LZ4_GPU_PLATFORM_ERROR,
    LZ4_GPU_DEVICE_ERROR,
    LZ4_GPU_CONTEXT_ERROR,
    LZ4_GPU_QUEUE_ERROR,
    LZ4_GPU_PROGRAM_ERROR,
    LZ4_GPU_BUILD_ERROR,
    LZ4_GPU_KERNEL_ERROR,
    LZ4_GPU_BUFFER_ERROR,
    LZ4_GPU_UPLOAD_ERROR,
    LZ4_GPU_DOWNLOAD_ERROR,
    LZ4_GPU_KERNEL_ARGS_ERROR,
    LZ4_GPU_KERNEL_LAUNCH_ERROR,
    LZ4_GPU_COMPRESS_ERROR,
    LZ4_GPU_DECOMPRESS_ERROR,
    LZ4_GPU_INVALID_PARAMS,
    LZ4_GPU_BUFFER_TOO_SMALL,
    LZ4_GPU_NOT_INITIALIZED
} LZ4GPUErrorCode;

// Performance timing structure
typedef struct {
    double total_ms;        // Total operation time
    double alloc_ms;        // Buffer allocation time
    double h2d_ms;          // Host to Device transfer time
    double kernel_ms;       // Kernel execution time (host-observed wall time)
    double kernel_ms_device; // Kernel execution time reported by device profiling event (if available)
    double d2h_ms;          // Device to Host transfer time
    double frame_ms;        // Frame assembly time (CPU)
    double setup_ms;        // Kernel argument setup / enqueue time
    double event_profile_ms; // Sum of profiled event durations (if profiling enabled)
    size_t input_bytes;     // Input size in bytes
    size_t output_bytes;    // Output size in bytes
    int num_blocks;         // Number of blocks processed
} LZ4GPUTiming;

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

// Define min function if not available
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif

// BYTE type definition (from original LZ4)
#ifndef BYTE
typedef unsigned char BYTE;
#endif

#include <CL/cl.h>
#include "xxhash.h"

// LZ4 Frame Format Constants (for host code)
#define LZ4F_MAGICNUMBER 0x184D2204U
#define LZ4F_BLOCKSIZEID_DEFAULT 4
#define LZ4F_BLOCK_HEADER_SIZE 4
#define LZ4F_HEADER_SIZE_MAX 19
#define LZ4F_HEADER_SIZE_MIN 7
#define LZ4F_ENDMARK_SIZE 4
#define LZ4F_CONTENT_CHECKSUM_SIZE 4
#define LZ4F_BLOCK_CHECKSUM_SIZE 4
#define LZ4F_BLOCKUNCOMPRESSED_FLAG 0x80000000U

// Bit manipulation constants from original LZ4
#define _1BIT  0x01
#define _2BITS 0x03
#define _3BITS 0x07
#define _4BITS 0x0F

// LZ4 GPU Compression Constants
#define LZ4_GPU_HASH_LOG 14  // Hash table log size (14 = 16KB hash table)

struct LZ4GPUCompressor {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    // Kernels
    cl_kernel compress_kernel;
    cl_kernel decompress_kernel;  // Block decompression
    cl_kernel frame_decompress_kernel;  // Frame decompression

    // Memory buffers
    cl_mem input_buffer;
    cl_mem output_buffer;

    // Buffer sizes (for dynamic allocation)
    size_t input_buffer_size;
    size_t output_buffer_size;

    // Device capabilities (queried at initialization)
    cl_uint device_compute_units;
    size_t device_max_work_group_size;

    // Dynamic block size chosen per-device / per-input
    size_t dynamic_block_size;

    /* Build/runtime options controlled by host API or CLI. These flags
     * are used by lz4_gpu_build_program_with_options() to select precompiled
     * binaries or to pass -D options to the OpenCL compiler when building
     * from source. They must be part of the public struct so callers that
     * allocate/inspect the compressor can set them prior to initialize(). */
    int enable_vector_io;      /* if 1, build/load vectorized kernel variant */
    int enable_kernel_debug;   /* if 1, compile kernel with LZ4_GPU_KERNEL_DEBUG */
    int prefer_precompiled;    /* if 1, prefer loading a precompiled .clbin when available */
    char precompiled_path[256];/* optional path to a precompiled clbin to load */
    int enable_profiling;      /* if 1, create command queue with profiling enabled and collect timings */

    /* Runtime kernel hashlog chosen at build time based on device local memory.
     * This is set by lz4_gpu_build_program_with_options() so host can allocate
     * the correct per-work-group local memory size when launching kernels. */
    int kernel_hashlog;

    // Performance timing (last operation)
    LZ4GPUTiming last_timing;

    // Error state
    LZ4GPUErrorCode last_error;
    char error_message[256];
};

// Opaque compressor structure
typedef struct LZ4GPUCompressor LZ4GPUCompressor;

// API Functions

// Create a new GPU compressor instance
LZ4GPUCompressor* lz4_gpu_create_compressor(void);

// Destroy compressor and free resources
void lz4_gpu_destroy_compressor(LZ4GPUCompressor* compressor);

// Initialize OpenCL context and GPU resources
int lz4_gpu_initialize(LZ4GPUCompressor* compressor);

// Compress data to LZ4 frame format using GPU
size_t lz4_gpu_compress_frame(LZ4GPUCompressor* compressor,
                             const void* input, size_t input_size,
                             void* output, size_t output_capacity);

// Compress data to LZ4 frame format using GPU with acceleration control
size_t lz4_gpu_compress_frame_accelerated(LZ4GPUCompressor* compressor,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        int acceleration);

// Compress single block using GPU (fallback function)
// Note: This function is not implemented yet - use lz4_gpu_compress_frame instead
// size_t lz4_gpu_compress_frame_single(LZ4GPUCompressor* compressor,
//                                     const void* input, size_t input_size,
//                                     void* output, size_t output_capacity);

// Decompress LZ4 frame using GPU - Host-side frame parsing with parallel GPU block decompression
size_t lz4_gpu_decompress_frame(LZ4GPUCompressor* compressor,
                                 const void* input, size_t input_size,
                                 void* output, size_t output_capacity);

// Estimate the decompressed stream size without performing full decompression.
// Returns the required output size on success, or 0 on failure (and sets compressor error).
size_t lz4_gpu_estimate_decompressed_size(LZ4GPUCompressor* compressor,
                                          const void* input, size_t input_size);

// Decompress single LZ4 block using GPU
size_t lz4_gpu_decompress_block(LZ4GPUCompressor* compressor,
                                const void* compressed_block, size_t compressed_size,
                                void* output, size_t max_output_size);

// Runtime/build option setters
void lz4_gpu_set_vector_io(LZ4GPUCompressor* compressor, int enabled);
void lz4_gpu_set_kernel_debug(LZ4GPUCompressor* compressor, int enabled);
/* Enable or disable host-side debug printing at runtime (overrides env var/cached value)
 * Call before lz4_gpu_initialize() if you want debug prints during init.
 */
void lz4_gpu_set_host_debug(LZ4GPUCompressor* compressor, int enabled);
int lz4_gpu_rebuild_program(LZ4GPUCompressor* compressor);
void lz4_gpu_use_precompiled(LZ4GPUCompressor* compressor, int enabled);
void lz4_gpu_set_precompiled_binary(LZ4GPUCompressor* compressor, const char* path);

// Query device capabilities for the initialized compressor.
// If the compressor is initialized, fills out_compute_units and out_max_work_group_size when non-NULL and returns 1 on success.
// Returns 0 on failure (e.g. compressor NULL or not initialized).
int lz4_gpu_query_device_capabilities(LZ4GPUCompressor* compressor, cl_uint* out_compute_units, size_t* out_max_work_group_size);

// Get timing information from the last operation
// Returns 1 on success, 0 if no timing data available
int lz4_gpu_get_last_timing(LZ4GPUCompressor* compressor, LZ4GPUTiming* timing);

// Print timing information in human-readable format
void lz4_gpu_print_timing(const LZ4GPUTiming* timing);

// Error handling
LZ4GPUErrorCode lz4_gpu_get_last_error(LZ4GPUCompressor* compressor);
const char* lz4_gpu_get_error_message(LZ4GPUCompressor* compressor);

#ifdef __cplusplus
}
#endif

#endif // LZ4_GPU_HOST_H