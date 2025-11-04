/*
 * LZ4 GPU Compressor Example (C version)
 * Demonstrates GPU-accelerated LZ4 frame compression
 */

#include "lz4_gpu_host.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>


#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define O_RDONLY _O_RDONLY
#define O_BINARY _O_BINARY
#else
#include <unistd.h>
#include <fcntl.h>
#endif

// Helper function for reading little-endian 32-bit integers
static uint32_t readLE32(const unsigned char* ptr) {
    return (uint32_t)ptr[0] | ((uint32_t)ptr[1] << 8) | ((uint32_t)ptr[2] << 16) | ((uint32_t)ptr[3] << 24);
}

// Simple file I/O functions
static size_t read_file(const char* filename, void** buffer) {
    int fd = open(filename, O_RDONLY
#ifdef O_BINARY
                  | O_BINARY
#endif
                  );
    if (fd == -1) {
        perror("Failed to open input file");
        return 0;
    }

    off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size == -1) {
        perror("Failed to get file size");
        close(fd);
        return 0;
    }

    lseek(fd, 0, SEEK_SET);

    *buffer = malloc(file_size);
    if (!*buffer) {
        fprintf(stderr, "Failed to allocate memory for file\n");
        close(fd);
        return 0;
    }

    ssize_t bytes_read = read(fd, *buffer, file_size);
    close(fd);

    if (bytes_read != file_size) {
        perror("Failed to read file");
        free(*buffer);
        return 0;
    }

    return (size_t)file_size;
}

static int write_file(const char* filename, const void* buffer, size_t size) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open output file");
        return 0;
    }

    size_t written = fwrite(buffer, 1, size, file);
    fclose(file);

    if (written != size) {
        fprintf(stderr, "Failed to write complete file\n");
        return 0;
    }

    return 1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [output_file]\n", argv[0]);
        fprintf(stderr, "Compresses input file using GPU LZ4 and saves to output file\n");
        fprintf(stderr, "If output_file is not specified, it defaults to 'input_file.lz4'\n");
        return 1;
    }

    const char* input_file = argv[1];
    char default_output[1024];
    const char* output_file;

    if (argc >= 3) {
        output_file = argv[2];
    } else {
        // Generate default output filename: input_file.lz4
        size_t input_len = strlen(input_file);
        if (input_len + 5 > sizeof(default_output)) {
            fprintf(stderr, "Input filename too long for default output\n");
            return 1;
        }
        strcpy(default_output, input_file);
        strcat(default_output, ".lz4");
        output_file = default_output;
    }

    // Read input file
    void* input_data = NULL;
    size_t input_size = read_file(input_file, &input_data);
    if (input_size == 0) {
        return 1;
    }

    printf("Input file size: %zu bytes\n", input_size);

    // Create and initialize GPU compressor
    LZ4GPUCompressor* compressor = lz4_gpu_create_compressor();
    if (!compressor) {
        fprintf(stderr, "Failed to create compressor\n");
        free(input_data);
        return 1;
    }

    if (!lz4_gpu_initialize(compressor)) {
        fprintf(stderr, "GPU initialization failed: %s\n",
                lz4_gpu_get_error_message(compressor));
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        return 1;
    }

    printf("GPU compressor initialized successfully\n");

    // Print device capabilities for debugging
    {
        cl_uint compute_units = 0;
        size_t max_wg = 0;
        if (lz4_gpu_query_device_capabilities(compressor, &compute_units, &max_wg)) {
            printf("Device compute units: %u, max work-group size: %zu\n", (unsigned int)compute_units, max_wg);
        } else {
            printf("Warning: failed to query device capabilities\n");
        }
    }

    // Prepare output buffer - use conservative estimate for worst case compression
    // LZ4 can expand up to ~1.1x in worst case, plus frame overhead
    size_t output_capacity = input_size + (input_size / 10) + 1024 * 1024; // More generous estimate
    void* output_data = malloc(output_capacity);
    if (!output_data) {
        fprintf(stderr, "Failed to allocate output buffer\n");
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        return 1;
    }

    // Test different acceleration levels
    printf("=== Testing Different Acceleration Levels ===\n");

    // Test standard acceleration levels
    int acceleration_levels[] = {1, 4, 8, 16};
    size_t* compressed_sizes = malloc(sizeof(size_t) * 4);
    clock_t* compression_times = malloc(sizeof(clock_t) * 4);

    if (!compressed_sizes || !compression_times) {
        fprintf(stderr, "Failed to allocate test arrays\n");
        free(output_data);
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        return 1;
    }

    for (int i = 0; i < 4; i++) {
        int accel = acceleration_levels[i];
        printf("\n--- Testing acceleration level %d ---\n", accel);

        clock_t accel_start = clock();
        compressed_sizes[i] = lz4_gpu_compress_frame_accelerated(
            compressor, input_data, input_size,
            output_data, output_capacity, accel
        );
        clock_t accel_end = clock();

        if (compressed_sizes[i] == 0) {
            fprintf(stderr, "Compression failed for acceleration %d: %s\n",
                    accel, lz4_gpu_get_error_message(compressor));
            compressed_sizes[i] = 0;
            compression_times[i] = 0;
        } else {
            compression_times[i] = accel_end - accel_start;
            double time_sec = (double)compression_times[i] / CLOCKS_PER_SEC;
            printf("Acceleration %d: %zu bytes, %.3f seconds, %.2f MB/s\n",
                   accel, compressed_sizes[i], time_sec,
                   (input_size / 1024.0 / 1024.0) / time_sec);
        }
    }

    // Find best acceleration level (balance of speed and compression ratio)
    int best_idx = 0;
    double best_score = 0;
    for (int i = 0; i < 4; i++) {
        if (compressed_sizes[i] > 0) {
            double time_sec = (double)compression_times[i] / CLOCKS_PER_SEC;
            double throughput = (input_size / 1024.0 / 1024.0) / time_sec;
            double ratio = (double)compressed_sizes[i] / input_size;

            // Score = throughput / ratio (higher is better)
            double score = throughput / ratio;

            printf("Acceleration %d score: %.2f (throughput=%.2f MB/s, ratio=%.3f)\n",
                   acceleration_levels[i], score, throughput, ratio);

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
    }

    printf("\n=== Best acceleration level: %d (score: %.2f) ===\n",
           acceleration_levels[best_idx], best_score);

    // Re-run compression at the best acceleration level to ensure the output buffer contains
    // the compressed data we will write to disk, and to measure the final time correctly.
    int best_accel = acceleration_levels[best_idx];
    printf("\n=== Final Results (Re-running with best acceleration level %d) ===\n", best_accel);

    clock_t t0 = clock();
    size_t final_compressed_size = lz4_gpu_compress_frame_accelerated(
        compressor, input_data, input_size,
        output_data, output_capacity, best_accel
    );
    clock_t t1 = clock();
    double compression_time = (final_compressed_size > 0) ? (double)(t1 - t0) / CLOCKS_PER_SEC : 0.0;

    if (final_compressed_size == 0) {
        LZ4GPUErrorCode err_code = lz4_gpu_get_last_error(compressor);
        fprintf(stderr, "Compression failed (error code %d): %s\n",
                err_code, lz4_gpu_get_error_message(compressor));
        free(output_data);
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        free(compressed_sizes);
        free(compression_times);
        return 1;
    }

    printf("Compressed size: %zu bytes\n", final_compressed_size);
    printf("Compression ratio: %.3f\n", (double)final_compressed_size / input_size);
    printf("Compression time: %.3f seconds\n", compression_time);
    printf("Compression throughput: %.2f MB/s\n",
           (input_size / 1024.0 / 1024.0) / compression_time);

    // Free test arrays
    free(compressed_sizes);
    free(compression_times);

    // Write compressed data to output file
    if (!write_file(output_file, output_data, final_compressed_size)) {
        free(output_data);
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        return 1;
    }

    printf("Compressed data written to: %s\n", output_file);

    // Now test decompression by decompressing the compressed data in memory
    printf("\n--- Testing Decompression ---\n");

    // Prepare decompression buffer (with extra space for worst case)
    size_t decompress_capacity = input_size + 1024; // Should be enough
    void* decompressed_data = malloc(decompress_capacity);
    if (!decompressed_data) {
        fprintf(stderr, "Failed to allocate decompression buffer\n");
        free(output_data);
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        return 1;
    }


    // Note: test_simple.dat should be created manually with simple pattern data

    // Now test with the original large file
    printf("\n=== Testing with large file ===\n");

    // Decompress the compressed data
    clock_t decompress_start_time = clock();

    size_t decompressed_size = lz4_gpu_decompress_frame(
        compressor, output_data, final_compressed_size,
        decompressed_data, decompress_capacity
    );

    clock_t decompress_end_time = clock();
    double decompression_time = (double)(decompress_end_time - decompress_start_time) / CLOCKS_PER_SEC;

    if (decompressed_size == 0) {
        fprintf(stderr, "Decompression failed: %s\n",
                lz4_gpu_get_error_message(compressor));

        // Let's try a simpler test with the legacy single-block decompressor
        printf("\nTrying legacy single-block decompression for debugging...\n");

        // Find the first block in the frame (skip header)
        unsigned char* compressed_ptr = (unsigned char*)output_data;

        // Skip LZ4 frame header (should be 15 bytes for our frame format with content size)
        if (final_compressed_size >= 15 && readLE32(compressed_ptr) == 0x184D2204) {
            compressed_ptr += 15; // Skip header (magic + FLG + BD + content size + checksum)

            // Try to find and decompress first block
            if (compressed_ptr + 4 < (unsigned char*)output_data + final_compressed_size) {
                uint32_t block_header = readLE32(compressed_ptr);
                compressed_ptr += 4;

                uint32_t block_size = block_header & 0x7FFFFFFF;
                uint32_t is_compressed = !(block_header & 0x80000000);

                printf("First block header: 0x%08x, size: %u, compressed: %d\n",
                       block_header, block_size, is_compressed);

                if (block_size > 0 && compressed_ptr + block_size <= (unsigned char*)output_data + final_compressed_size) {
                    // Use legacy decompressor for this block
                    size_t legacy_decompressed = lz4_gpu_decompress_block(
                        compressor, compressed_ptr, block_size,
                        decompressed_data, decompress_capacity
                    );

                    printf("Legacy decompression result: %zu bytes\n", legacy_decompressed);

                    if (legacy_decompressed > 0) {
                        printf("Legacy decompressed first 32 bytes:\n");
                        for (size_t i = 0; i < 32 && i < legacy_decompressed; i++) {
                            printf("%02x ", ((unsigned char*)decompressed_data)[i]);
                        }
                        printf("\n");
                    }
                }
            }
        }

        free(decompressed_data);
        free(output_data);
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        return 1;
    }

    printf("Decompressed size: %zu bytes\n", decompressed_size);
    printf("Decompression time: %.3f seconds\n", decompression_time);
    printf("Decompression throughput: %.2f MB/s\n",
           (decompressed_size / 1024.0 / 1024.0) / decompression_time);

    // Verify the decompressed data matches the original
    if (decompressed_size != input_size) {
        fprintf(stderr, "ERROR: Decompressed size (%zu) doesn't match original size (%zu)\n",
                decompressed_size, input_size);
        free(decompressed_data);
        free(output_data);
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        return 1;
    }

    if (memcmp(input_data, decompressed_data, input_size) != 0) {
        fprintf(stderr, "ERROR: Decompressed data doesn't match original data!\n");

        // Show first 32 bytes for debugging
        printf("Original:   ");
        for (size_t i = 0; i < 32 && i < input_size; i++) {
            printf("%02x ", ((unsigned char*)input_data)[i]);
        }
        printf("\n");

        printf("Decompressed: ");
        for (size_t i = 0; i < 32 && i < decompressed_size; i++) {
            printf("%02x ", ((unsigned char*)decompressed_data)[i]);
        }
        printf("\n");

        free(decompressed_data);
        free(output_data);
        lz4_gpu_destroy_compressor(compressor);
        free(input_data);
        return 1;
    }

    printf("✓ Data verification successful - decompressed data matches original!\n");

    // Cleanup
    free(decompressed_data);
    free(output_data);
    lz4_gpu_destroy_compressor(compressor);
    free(input_data);

    printf("Compression and decompression test completed successfully!\n");
    return 0;
}

// Example function for in-memory checkpoint compression
size_t compress_memory_snapshot(LZ4GPUCompressor* compressor,
                              const void* snapshot_data, size_t snapshot_size,
                              void* compressed_buffer, size_t buffer_capacity) {
    if (!lz4_gpu_initialize(compressor)) {
        fprintf(stderr, "Failed to initialize GPU compressor\n");
        return 0;
    }

    clock_t start_time = clock();

    size_t compressed_size = lz4_gpu_compress_frame(
        compressor, snapshot_data, snapshot_size,
        compressed_buffer, buffer_capacity
    );

    clock_t end_time = clock();
    double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    if (compressed_size > 0) {
        printf("Memory snapshot compressed: %zu -> %zu bytes\n",
               snapshot_size, compressed_size);
        printf("Compression time: %.6f seconds\n", duration);
        printf("Throughput: %.2f MB/s\n",
               (snapshot_size / 1024.0 / 1024.0) / duration);
    } else {
        fprintf(stderr, "Compression failed: %s\n",
                lz4_gpu_get_error_message(compressor));
    }

    return compressed_size;
}

// Example usage for checkpoint compression
void example_checkpoint_compression() {
    const size_t CHECKPOINT_SIZE = 256 * 1024 * 1024; // 256MB checkpoint
    unsigned char* checkpoint_data = (unsigned char*)malloc(CHECKPOINT_SIZE);

    if (!checkpoint_data) {
        fprintf(stderr, "Failed to allocate checkpoint memory\n");
        return;
    }

    // Fill with some test data (in real scenario, this would be actual memory state)
    for (size_t i = 0; i < CHECKPOINT_SIZE; ++i) {
        checkpoint_data[i] = (unsigned char)(i % 256);
    }

    LZ4GPUCompressor* compressor = lz4_gpu_create_compressor();
    if (!compressor) {
        free(checkpoint_data);
        return;
    }

    unsigned char* compressed_buffer = (unsigned char*)malloc(CHECKPOINT_SIZE / 2); // Assume 2:1 compression
    if (!compressed_buffer) {
        lz4_gpu_destroy_compressor(compressor);
        free(checkpoint_data);
        return;
    }

    size_t compressed_size = compress_memory_snapshot(
        compressor, checkpoint_data, CHECKPOINT_SIZE,
        compressed_buffer, CHECKPOINT_SIZE / 2
    );

    if (compressed_size > 0) {
        printf("Checkpoint compression successful!\n");
        printf("Space saved: %zu bytes\n", CHECKPOINT_SIZE - compressed_size);
    } else {
        printf("Checkpoint compression failed!\n");
    }

    free(compressed_buffer);
    lz4_gpu_destroy_compressor(compressor);
    free(checkpoint_data);
}
