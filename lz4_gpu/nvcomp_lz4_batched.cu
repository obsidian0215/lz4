// Formal nvCOMP LZ4 baseline using the low-level C++ batched API.

#include <cuda_runtime.h>
#include <nvcomp/lz4.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string input;
  std::size_t chunk_size = 65536;
  int warmup = 3;
  int iterations = 1;
  int gpu_index = 0;
  int decompress_backend = 0;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void nvcomp_check(nvcompStatus_t status, const char *operation) {
  if (status != nvcompSuccess) {
    fail(std::string(operation) + " returned status " +
         std::to_string(static_cast<int>(status)));
  }
}

std::size_t parse_size(const std::string &value, const char *name) {
  std::size_t consumed = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &consumed, 10);
  } catch (const std::exception &) {
    fail(std::string("invalid ") + name);
  }
  if (consumed != value.size() || parsed == 0 ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    fail(std::string("invalid ") + name);
  }
  return static_cast<std::size_t>(parsed);
}

int parse_nonnegative_int(const std::string &value, const char *name) {
  std::size_t consumed = 0;
  long parsed = 0;
  try {
    parsed = std::stol(value, &consumed, 10);
  } catch (const std::exception &) {
    fail(std::string("invalid ") + name);
  }
  if (consumed != value.size() || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    fail(std::string("invalid ") + name);
  }
  return static_cast<int>(parsed);
}

Options parse_args(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (index + 1 >= argc) {
      fail("missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--input") {
      options.input = value;
    } else if (argument == "--chunk-size") {
      options.chunk_size = parse_size(value, "chunk size");
    } else if (argument == "--warmup") {
      options.warmup = parse_nonnegative_int(value, "warmup");
    } else if (argument == "--iterations") {
      options.iterations = parse_nonnegative_int(value, "iterations");
    } else if (argument == "--gpu-index") {
      options.gpu_index = parse_nonnegative_int(value, "GPU index");
    } else if (argument == "--decompress-backend") {
      options.decompress_backend = parse_nonnegative_int(value, "decompress backend");
    } else {
      fail("unknown argument: " + argument);
    }
  }
  if (options.input.empty() || options.iterations <= 0 ||
      options.decompress_backend > 2) {
    fail("input, positive iterations, and decompression backend 0/1/2 are required");
  }
  return options;
}

std::vector<std::uint8_t> read_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    fail("cannot open input file");
  }
  const std::streamsize length = input.tellg();
  if (length <= 0) {
    fail("nvCOMP formal baseline requires a non-empty input");
  }
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(length));
  if (!input.read(reinterpret_cast<char *>(data.data()), length)) {
    fail("cannot read complete input file");
  }
  return data;
}

std::size_t round_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    fail("nvCOMP returned zero alignment");
  }
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - remainder)) {
    fail("aligned allocation size overflow");
  }
  return value + alignment - remainder;
}

template <typename T> T *device_alloc(std::size_t count) {
  T *pointer = nullptr;
  const std::size_t bytes = std::max<std::size_t>(1, count * sizeof(T));
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&pointer), bytes), "cudaMalloc");
  return pointer;
}

double median(std::vector<float> values) {
  if (values.empty()) {
    fail("cannot calculate an empty median");
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[middle];
  }
  return (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0;
}

void verify_statuses(const std::vector<nvcompStatus_t> &statuses, const char *phase) {
  for (std::size_t index = 0; index < statuses.size(); ++index) {
    if (statuses[index] != nvcompSuccess) {
      fail(std::string(phase) + " failed for chunk " + std::to_string(index));
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_args(argc, argv);
    const std::vector<std::uint8_t> input = read_file(options.input);
    cuda_check(cudaSetDevice(options.gpu_index), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    const std::size_t batch_size =
        (input.size() - 1) / options.chunk_size + 1;
    std::vector<std::size_t> chunk_sizes(batch_size);
    std::vector<std::size_t> chunk_offsets(batch_size);
    for (std::size_t index = 0; index < batch_size; ++index) {
      chunk_offsets[index] = index * options.chunk_size;
      chunk_sizes[index] =
          std::min(options.chunk_size, input.size() - chunk_offsets[index]);
    }

    nvcompBatchedLZ4CompressOpts_t comp_opts = nvcompBatchedLZ4CompressDefaultOpts;
    nvcompBatchedLZ4DecompressOpts_t decomp_opts = nvcompBatchedLZ4DecompressDefaultOpts;
    comp_opts.data_type = NVCOMP_TYPE_CHAR;
    decomp_opts.data_type = NVCOMP_TYPE_CHAR;
    comp_opts.bitshuffle_mode = static_cast<nvcompBitshuffleMode_t>(0);
    decomp_opts.bitshuffle_mode = static_cast<nvcompBitshuffleMode_t>(0);
    decomp_opts.backend =
        static_cast<nvcompDecompressBackend_t>(options.decompress_backend);

    nvcompAlignmentRequirements_t comp_alignment{};
    nvcompAlignmentRequirements_t decomp_alignment{};
    nvcomp_check(nvcompBatchedLZ4CompressGetRequiredAlignments(
                     comp_opts, &comp_alignment),
                 "nvcompBatchedLZ4CompressGetRequiredAlignments");
    nvcomp_check(nvcompBatchedLZ4DecompressGetRequiredAlignments(
                     decomp_opts, &decomp_alignment),
                 "nvcompBatchedLZ4DecompressGetRequiredAlignments");

    const std::size_t input_stride =
        round_up(options.chunk_size, std::max<std::size_t>(16, comp_alignment.input));
    std::uint8_t *d_input = device_alloc<std::uint8_t>(input_stride * batch_size);
    std::vector<void *> h_input_ptrs(batch_size);
    for (std::size_t index = 0; index < batch_size; ++index) {
      h_input_ptrs[index] = d_input + input_stride * index;
      cuda_check(cudaMemcpyAsync(h_input_ptrs[index], input.data() + chunk_offsets[index],
                                 chunk_sizes[index], cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(input)");
    }
    void **d_input_ptrs = device_alloc<void *>(batch_size);
    std::size_t *d_input_sizes = device_alloc<std::size_t>(batch_size);
    cuda_check(cudaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(),
                               batch_size * sizeof(void *), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(input pointers)");
    cuda_check(cudaMemcpyAsync(d_input_sizes, chunk_sizes.data(),
                               batch_size * sizeof(std::size_t), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(input sizes)");

    std::size_t max_compressed_chunk = 0;
    nvcomp_check(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
                     options.chunk_size, comp_opts, &max_compressed_chunk),
                 "nvcompBatchedLZ4CompressGetMaxOutputChunkSize");
    const std::size_t compressed_stride =
        round_up(max_compressed_chunk,
                 std::max(comp_alignment.output, decomp_alignment.input));
    std::uint8_t *d_compressed =
        device_alloc<std::uint8_t>(compressed_stride * batch_size);
    std::vector<void *> h_compressed_ptrs(batch_size);
    for (std::size_t index = 0; index < batch_size; ++index) {
      h_compressed_ptrs[index] = d_compressed + compressed_stride * index;
    }
    void **d_compressed_ptrs = device_alloc<void *>(batch_size);
    std::size_t *d_compressed_sizes = device_alloc<std::size_t>(batch_size);
    nvcompStatus_t *d_comp_statuses = device_alloc<nvcompStatus_t>(batch_size);
    cuda_check(cudaMemcpyAsync(d_compressed_ptrs, h_compressed_ptrs.data(),
                               batch_size * sizeof(void *), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(compressed pointers)");

    std::size_t comp_temp_bytes = 0;
    nvcomp_check(nvcompBatchedLZ4CompressGetTempSizeSync(
                     d_input_ptrs, d_input_sizes, batch_size, options.chunk_size,
                     comp_opts, &comp_temp_bytes, input.size(), stream),
                 "nvcompBatchedLZ4CompressGetTempSizeSync");
    void *d_comp_temp = device_alloc<std::uint8_t>(comp_temp_bytes);

    auto compress = [&]() {
      nvcomp_check(nvcompBatchedLZ4CompressAsync(
                       d_input_ptrs, d_input_sizes, options.chunk_size, batch_size,
                       d_comp_temp, comp_temp_bytes, d_compressed_ptrs,
                       d_compressed_sizes, comp_opts, d_comp_statuses, stream),
                   "nvcompBatchedLZ4CompressAsync");
    };
    compress();
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(initial compression)");

    std::vector<std::size_t> compressed_sizes(batch_size);
    cuda_check(cudaMemcpy(compressed_sizes.data(), d_compressed_sizes,
                          batch_size * sizeof(std::size_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy(compressed sizes)");
    std::vector<nvcompStatus_t> comp_statuses(batch_size);
    cuda_check(cudaMemcpy(comp_statuses.data(), d_comp_statuses,
                          batch_size * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy(compression statuses)");
    verify_statuses(comp_statuses, "compression");

    std::size_t *d_expected_sizes = device_alloc<std::size_t>(batch_size);
    nvcomp_check(nvcompBatchedLZ4GetDecompressSizeAsync(
                     d_compressed_ptrs, d_compressed_sizes, d_expected_sizes,
                     batch_size, stream),
                 "nvcompBatchedLZ4GetDecompressSizeAsync");
    std::vector<std::size_t> encoded_sizes(batch_size);
    cuda_check(cudaMemcpyAsync(encoded_sizes.data(), d_expected_sizes,
                               batch_size * sizeof(std::size_t), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(encoded sizes)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(encoded sizes)");
    if (encoded_sizes != chunk_sizes) {
      fail("nvCOMP encoded decompressed sizes do not match the input chunks");
    }

    const std::size_t output_stride =
        round_up(options.chunk_size, decomp_alignment.output);
    std::uint8_t *d_output = device_alloc<std::uint8_t>(output_stride * batch_size);
    std::vector<void *> h_output_ptrs(batch_size);
    for (std::size_t index = 0; index < batch_size; ++index) {
      h_output_ptrs[index] = d_output + output_stride * index;
    }
    void **d_output_ptrs = device_alloc<void *>(batch_size);
    std::size_t *d_output_sizes = device_alloc<std::size_t>(batch_size);
    nvcompStatus_t *d_decomp_statuses = device_alloc<nvcompStatus_t>(batch_size);
    cuda_check(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(),
                               batch_size * sizeof(void *), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(output pointers)");

    std::size_t decomp_temp_bytes = 0;
    nvcomp_check(nvcompBatchedLZ4DecompressGetTempSizeSync(
                     d_compressed_ptrs, d_compressed_sizes, batch_size,
                     options.chunk_size, &decomp_temp_bytes, input.size(), decomp_opts,
                     d_decomp_statuses, stream),
                 "nvcompBatchedLZ4DecompressGetTempSizeSync");
    void *d_decomp_temp = device_alloc<std::uint8_t>(decomp_temp_bytes);
    auto decompress = [&]() {
      nvcomp_check(nvcompBatchedLZ4DecompressAsync(
                       d_compressed_ptrs, d_compressed_sizes, d_expected_sizes,
                       d_output_sizes, batch_size, d_decomp_temp, decomp_temp_bytes,
                       d_output_ptrs, decomp_opts, d_decomp_statuses, stream),
                   "nvcompBatchedLZ4DecompressAsync");
    };

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      compress();
      decompress();
    }
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");

    std::vector<float> compression_ms;
    std::vector<float> decompression_ms;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      cuda_check(cudaEventRecord(start, stream), "cudaEventRecord(compression start)");
      compress();
      cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord(compression stop)");
      cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(compression)");
      float elapsed = 0.0F;
      cuda_check(cudaEventElapsedTime(&elapsed, start, stop),
                 "cudaEventElapsedTime(compression)");
      compression_ms.push_back(elapsed);
    }
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      cuda_check(cudaEventRecord(start, stream), "cudaEventRecord(decompression start)");
      decompress();
      cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord(decompression stop)");
      cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(decompression)");
      float elapsed = 0.0F;
      cuda_check(cudaEventElapsedTime(&elapsed, start, stop),
                 "cudaEventElapsedTime(decompression)");
      decompression_ms.push_back(elapsed);
    }

    std::vector<std::uint8_t> compressed_host(compressed_stride * batch_size);
    const auto compression_warm_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < batch_size; ++index) {
      cuda_check(cudaMemcpyAsync(h_input_ptrs[index], input.data() + chunk_offsets[index],
                                 chunk_sizes[index], cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(warm compression input)");
    }
    compress();
    cuda_check(cudaMemcpyAsync(compressed_sizes.data(), d_compressed_sizes,
                               batch_size * sizeof(std::size_t), cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(warm compressed sizes)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warm compressed sizes)");
    for (std::size_t index = 0; index < batch_size; ++index) {
      if (compressed_sizes[index] == 0 || compressed_sizes[index] > compressed_stride) {
        fail("nvCOMP returned an invalid compressed chunk size");
      }
      cuda_check(cudaMemcpyAsync(compressed_host.data() + compressed_stride * index,
                                 h_compressed_ptrs[index], compressed_sizes[index],
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync(warm compressed output)");
    }
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warm compression)");
    const auto compression_warm_stop = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> recovered(input.size());
    const auto decompression_warm_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < batch_size; ++index) {
      cuda_check(cudaMemcpyAsync(h_compressed_ptrs[index],
                                 compressed_host.data() + compressed_stride * index,
                                 compressed_sizes[index], cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(warm compressed input)");
    }
    cuda_check(cudaMemcpyAsync(d_compressed_sizes, compressed_sizes.data(),
                               batch_size * sizeof(std::size_t), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync(warm compressed sizes H2D)");
    decompress();
    for (std::size_t index = 0; index < batch_size; ++index) {
      cuda_check(cudaMemcpyAsync(recovered.data() + chunk_offsets[index],
                                 h_output_ptrs[index], chunk_sizes[index],
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync(warm decompressed output)");
    }
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warm decompression)");
    const auto decompression_warm_stop = std::chrono::steady_clock::now();

    std::vector<std::size_t> output_sizes(batch_size);
    std::vector<nvcompStatus_t> decomp_statuses(batch_size);
    cuda_check(cudaMemcpy(output_sizes.data(), d_output_sizes,
                          batch_size * sizeof(std::size_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy(output sizes)");
    cuda_check(cudaMemcpy(decomp_statuses.data(), d_decomp_statuses,
                          batch_size * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy(decompression statuses)");
    verify_statuses(decomp_statuses, "decompression");
    if (output_sizes != chunk_sizes || recovered != input) {
      fail("nvCOMP roundtrip verification failed");
    }

    std::size_t compressed_bytes = 0;
    for (const std::size_t size : compressed_sizes) {
      compressed_bytes += size;
    }
    const auto comp_warm_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        compression_warm_stop - compression_warm_start).count();
    const auto decomp_warm_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        decompression_warm_stop - decompression_warm_start).count();

    std::cout << "schema=heterolz.nvcomp-lz4-batched.v1\n"
              << "api=low-level-batched-cpp\n"
              << "algorithm=LZ4\n"
              << "data_type=char\n"
              << "bitshuffle_mode=0\n"
              << "decompress_backend=" << options.decompress_backend << "\n"
              << "input_bytes=" << input.size() << "\n"
              << "compressed_bytes=" << compressed_bytes << "\n"
              << "chunk_size=" << options.chunk_size << "\n"
              << "batch_size=" << batch_size << "\n"
              << "warmup=" << options.warmup << "\n"
              << "iterations=" << options.iterations << "\n"
              << "stream=explicit-created-default-flags\n"
              << "synchronization=cuda-event-kernel-and-stream-sync-phase-boundaries\n"
              << "warm_e2e_scope=h2d-operation-synchronization-exact-d2h\n"
              << "compression_kernel_ms=" << median(compression_ms) << "\n"
              << "decompression_kernel_ms=" << median(decompression_ms) << "\n"
              << "compression_warm_e2e_ns=" << comp_warm_ns << "\n"
              << "decompression_warm_e2e_ns=" << decomp_warm_ns << "\n"
              << "roundtrip_ok=true\n";

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(d_decomp_temp);
    cudaFree(d_decomp_statuses);
    cudaFree(d_output_sizes);
    cudaFree(d_output_ptrs);
    cudaFree(d_output);
    cudaFree(d_expected_sizes);
    cudaFree(d_comp_temp);
    cudaFree(d_comp_statuses);
    cudaFree(d_compressed_sizes);
    cudaFree(d_compressed_ptrs);
    cudaFree(d_compressed);
    cudaFree(d_input_sizes);
    cudaFree(d_input_ptrs);
    cudaFree(d_input);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "nvCOMP LZ4 batched failed: " << error.what() << '\n';
    return 1;
  }
}
