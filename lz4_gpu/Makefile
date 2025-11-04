# LZ4 GPU Compressor Makefile for Linux Testing
# Supports OpenCL GPU acceleration for LZ4 frame compression

# Compiler settings
CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra -pedantic
LDFLAGS =

# OpenCL detection and settings
OPENCL_FOUND := $(shell pkg-config --exists OpenCL && echo yes || echo no)
ifeq ($(OPENCL_FOUND),yes)
	CFLAGS += $(shell pkg-config --cflags OpenCL)
	LDFLAGS += $(shell pkg-config --libs OpenCL)
else
	# Fallback: try common OpenCL library locations
	CFLAGS += -I/usr/include/CL -I/opt/AMDAPP/include
	LDFLAGS += -lOpenCL
endif


# Build targets
TARGET = lz4_gpu
DEBUG_TARGET = lz4_gpu_debug
SOURCES = lz4_gpu_host.c lz4_gpu.c xxhash.c
HEADERS = lz4_gpu_host.h
KERNEL = lz4_gpu.cl

# Default target
all: check_dependencies $(TARGET) $(DEBUG_TARGET) precompile_clbins

# Build only release binary (no debug, no precompile)
release: check_dependencies $(TARGET)


# Check for required dependencies
check_dependencies:
	@echo "Checking dependencies..."
	@command -v $(CC) >/dev/null 2>&1 || { echo "Error: $(CC) compiler not found"; exit 1; }
	@if [ "$(OPENCL_FOUND)" = "yes" ]; then \
		echo "OpenCL found via pkg-config"; \
	else \
		echo "Warning: OpenCL not found via pkg-config, trying direct linking"; \
	fi
	@test -f $(KERNEL) || { echo "Error: $(KERNEL) not found"; exit 1; }
	@echo "Dependencies OK"

# Main executable (release)
$(TARGET): CFLAGS += -O2
$(TARGET): $(SOURCES) $(HEADERS)
	@echo "Compiling LZ4 GPU compressor (release)..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
	@echo "Build successful: $(TARGET)"

# Debug executable
$(DEBUG_TARGET): CFLAGS += -g -DDEBUG
$(DEBUG_TARGET): $(SOURCES) $(HEADERS)
	@echo "Compiling LZ4 GPU compressor (debug)..."
	$(CC) $(CFLAGS) -o $(DEBUG_TARGET) $(SOURCES) $(LDFLAGS)
	@echo "Build successful: $(DEBUG_TARGET)"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	rm -f *.o
	rm -f *.log

# Test build and basic functionality
test: $(TARGET)
	@echo "Testing LZ4 GPU compressor..."
	@if [ -f $(TARGET) ]; then \
		echo "Executable exists: $(TARGET)"; \
		if command -v file >/dev/null 2>&1; then \
			file $(TARGET); \
		fi; \
		if command -v ldd >/dev/null 2>&1; then \
			echo "Linked libraries:"; \
			ldd $(TARGET) 2>/dev/null || echo "ldd failed"; \
		fi; \
	else \
		echo "Error: $(TARGET) not found"; \
		exit 1; \
	fi

# Create test data for benchmarking
create_test_data:
	@echo "Creating test data..."
	@if command -v python3 >/dev/null 2>&1; then \
		python3 -c "import sys; data = bytearray(); [data.append((i % 256) ^ ((i >> 8) % 256)) for i in range(64 * 1024 * 1024)]; open('test_data_64MB.dat', 'wb').write(data)"; \
		echo 'Test data created with Python: test_data_64MB.dat'; \
	else \
		dd if=/dev/zero of=test_data_64MB.dat bs=1M count=64 2>/dev/null && \
		echo 'Test data created with dd: test_data_64MB.dat (may not compress well)'; \
	fi
	@ls -lh test_data_64MB.dat 2>/dev/null || echo "File created successfully"

# Benchmark test (requires test data)
benchmark: $(TARGET) create_test_data
	@echo "Running benchmark test..."
	@if [ -f $(TARGET) ] && [ -f test_data_64MB.dat ]; then \
		time ./$(TARGET) test_data_64MB.dat test_output.lz4 2>&1; \
		if [ -f test_output.lz4 ]; then \
			ls -lh test_data_64MB.dat test_output.lz4 2>/dev/null || ls -l test_data_64MB.dat test_output.lz4; \
			echo "Benchmark completed successfully"; \
		else \
			echo "Error: Output file not created"; \
			exit 1; \
		fi; \
	else \
		echo "Error: Test data or executable not found"; \
		exit 1; \
	fi


# GPU roundtrip test: build and run a small round-trip test program
gpu_roundtrip_test: $(TARGET)
	@echo "Building and running gpu_roundtrip_test..."
	$(CC) $(CFLAGS) -o gpu_roundtrip_test gpu_roundtrip_test.c $(LDFLAGS)
	@if [ -x ./gpu_roundtrip_test ]; then \
		./gpu_roundtrip_test || echo "gpu_roundtrip_test exited with non-zero status"; \
	else \
		echo "Error: gpu_roundtrip_test not built"; exit 1; \
	fi

# Run the AB experiment harness script (run_ab.sh)
run_ab: $(TARGET)
	@echo "Running run_ab.sh (AB harness)..."
	@if [ -f run_ab.sh ]; then \
		chmod +x run_ab.sh 2>/dev/null || true; \
		./run_ab.sh || echo "run_ab.sh exited with non-zero status"; \
	else \
		echo "Error: run_ab.sh not found"; exit 1; \
	fi

# Full test suite
test_all: clean all test benchmark

# Installation check
check_opencl:
	@echo "Checking OpenCL installation..."
	@if pkg-config --exists OpenCL; then \
		echo "OpenCL found via pkg-config:"; \
		pkg-config --modversion OpenCL; \
		pkg-config --cflags OpenCL; \
		pkg-config --libs OpenCL; \
	else \
		echo "OpenCL not found via pkg-config"; \
	fi
	@if [ -f /usr/include/CL/cl.h ]; then \
		echo "OpenCL headers found at /usr/include/CL/"; \
	elif [ -f /opt/AMDAPP/include/CL/cl.h ]; then \
		echo "OpenCL headers found at /opt/AMDAPP/include/"; \
	else \
		echo "OpenCL headers not found in common locations"; \
		echo "Please install OpenCL development headers:"; \
		echo "  Ubuntu/Debian: sudo apt-get install opencl-headers ocl-icd-opencl-dev"; \
		echo "  CentOS/RHEL: sudo yum install opencl-headers ocl-icd-devel"; \
	fi
	@if ldconfig -p | grep -q libOpenCL; then \
		echo "OpenCL library found in ldconfig"; \
	else \
		echo "OpenCL library not found in ldconfig"; \
	fi

# Precompile OpenCL kernels into .clbin (release and debug)
precompile_clbins:
	@echo "Prebuilding OpenCL binaries (clbin)..."
	@# Run detection/compile and generation in a single shell so BUILD_BIN persists
	@BUILD_BIN=""; \
	if [ -x ./build_clbin ]; then \
		BUILD_BIN=./build_clbin; \
	elif [ -f ./build_clbin.c ]; then \
		$(CC) $(CFLAGS) build_clbin.c -o build_clbin $(LDFLAGS) || true; \
		chmod +x build_clbin 2>/dev/null || true; \
		BUILD_BIN=./build_clbin; \
	elif [ -f backup_vector/build_clbin.c ]; then \
		$(CC) $(CFLAGS) backup_vector/build_clbin.c -o build_clbin $(LDFLAGS) || true; \
		chmod +x build_clbin 2>/dev/null || true; \
		BUILD_BIN=./build_clbin; \
	fi; \
	if [ -n "$$BUILD_BIN" ] && [ -x "$$BUILD_BIN" ]; then \
		echo "Using $$BUILD_BIN to generate clbins"; \
		$$BUILD_BIN -o lz4_gpu.clbin "" || true; \
		$$BUILD_BIN -o lz4_gpu_debug.clbin "-DLZ4_GPU_KERNEL_DEBUG" || true; \
		$$BUILD_BIN -o lz4_gpu_vec.clbin "-DLZ4_GPU_VECTOR_IO=1" || true; \
		$$BUILD_BIN -o lz4_gpu_debug_vec.clbin "-DLZ4_GPU_VECTOR_IO=1 -DLZ4_GPU_KERNEL_DEBUG" || true; \
	else \
		echo "No build_clbin available; skipping clbin prebuild"; \
	fi

# Debug target kept for compatibility
debug: $(DEBUG_TARGET)

# Help target
help:
	@echo "LZ4 GPU Compressor Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build the main executable (default)"
	@echo "  clean            - Remove build artifacts"
	@echo "  test             - Test if build was successful"
	@echo "  benchmark        - Run performance benchmark"
	@echo "  test_all         - Run full test suite"
	@echo "  check_opencl     - Check OpenCL installation"
	@echo "  check_dependencies - Verify build requirements"
	@echo "  create_test_data - Generate test data for benchmarking"
	@echo "  debug            - Build with debug symbols"
	@echo "  help             - Show this help message"
	@echo "  gpu_roundtrip_test - Build and run a small GPU roundtrip test"
	@echo "  run_ab           - Run the AB experiment harness (run_ab.sh)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build the compressor"
	@echo "  make test_all          # Build and run full tests"
	@echo "  make benchmark         # Run performance benchmark"
	@echo "  make check_opencl      # Verify OpenCL setup"

# Phony targets
.PHONY: all clean test benchmark test_all check_opencl check_dependencies gpu_roundtrip_test run_ab
.PHONY: create_test_data debug help

# Default to help if no target specified
.DEFAULT_GOAL := help