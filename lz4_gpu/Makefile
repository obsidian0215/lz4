ifeq ($(OS),Windows_NT)
MSYS2_USR_BIN ?= C:/msys64/usr/bin
ifneq ($(wildcard $(MSYS2_USR_BIN)/sh.exe),)
SHELL := $(MSYS2_USR_BIN)/sh.exe
export PATH := $(MSYS2_USR_BIN);$(PATH)
endif
endif

CC ?= cc
HOSTCC ?= cc
NVCC ?= nvcc
NVCOMP_ROOT ?=
NVCOMP_CUDA_ARCH ?= sm_89

ifeq ($(OS),Windows_NT)
EXEEXT ?= .exe
HOSTEXEEXT ?=
PTHREAD_LIB ?= -lpthread
OPENCL_INCLUDE_DIR ?=
OPENCL_LIB_DIR ?=
OPENCL_LIB_NAME ?= OpenCL
OPENCL_CFLAGS := $(if $(OPENCL_INCLUDE_DIR),-I$(OPENCL_INCLUDE_DIR),)
OPENCL_LDFLAGS := $(if $(OPENCL_LIB_DIR),-L$(OPENCL_LIB_DIR),) -l$(OPENCL_LIB_NAME)
CFLAGS ?= -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -DWIN32_LEAN_AND_MEAN -I. -I../lib -I.. $(OPENCL_CFLAGS)
LDFLAGS ?= $(OPENCL_LDFLAGS) -lm $(PTHREAD_LIB) -lws2_32
HOST_OPENCL_CFLAGS ?=
HOST_OPENCL_LDFLAGS ?=-lOpenCL -lm -lpthread
PRECOMPILE_HELPER := build_clbin_host$(HOSTEXEEXT)
else
EXEEXT ?=
HOSTEXEEXT ?=$(EXEEXT)
PTHREAD_CFLAGS ?= -pthread
PTHREAD_LIB ?= -pthread
OPENCL_CFLAGS ?=
OPENCL_LDFLAGS ?=-lOpenCL
CFLAGS ?= -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lib -I.. $(OPENCL_CFLAGS) $(PTHREAD_CFLAGS)
LDFLAGS ?= $(OPENCL_LDFLAGS) -lm $(PTHREAD_LIB)
HOST_OPENCL_CFLAGS ?=$(OPENCL_CFLAGS)
HOST_OPENCL_LDFLAGS ?=$(LDFLAGS)
PRECOMPILE_HELPER := build_clbin$(EXEEXT)
endif

COMMON_SRCS = lz4_gpu.c lz4_gpu_utils.c lz4_gpu_core.c lz4_tp_profile.c
ifeq ($(OS),Windows_NT)
DAEMON_SRCS =
else
DAEMON_SRCS = lz4_gpu_daemon.c lz4_gpu_client.c
endif
SRCS = $(COMMON_SRCS) $(DAEMON_SRCS)
OBJS = $(SRCS:.c=.o)

all: lz4_gpu native_lz4_blocks

lz4_gpu: $(SRCS)
	$(CC) $(CFLAGS) -o $@$(EXEEXT) $^ $(LDFLAGS)

native_lz4_blocks: native_lz4_blocks.c ../programs/threadpool.c ../programs/timefn.c ../lib/lz4.c
	$(CC) $(CFLAGS) -DLZ4IO_MULTITHREAD=1 -I../programs -I../lib -o $@$(EXEEXT) $^ $(PTHREAD_LIB)

nvcomp-baseline: nvcomp_lz4_batched.cu
	test -n "$(NVCOMP_ROOT)"
	$(NVCC) -std=c++17 -O3 -arch=$(NVCOMP_CUDA_ARCH) -I$(NVCOMP_ROOT)/include \
		$< -L$(NVCOMP_ROOT)/lib -Wl,-rpath,$(NVCOMP_ROOT)/lib -lnvcomp \
		-o nvcomp_lz4_batched

build_clbin: build_clbin.c
	$(CC) $(CFLAGS) -o $@$(EXEEXT) $^ $(LDFLAGS)

build_clbin_host: build_clbin.c
	$(HOSTCC) -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lib -I.. $(HOST_OPENCL_CFLAGS) -o $@$(HOSTEXEEXT) $^ $(HOST_OPENCL_LDFLAGS)

tp_ref_decode: tp_ref_decode.c
	$(CC) $(CFLAGS) -I../lib -o $@$(EXEEXT) $^ ../lib/lz4.c

tp_to_lz4: tp_to_lz4.c ../lib/xxhash.c
	$(CC) $(CFLAGS) -I../lib -o $@$(EXEEXT) $^

interop-tools: tp_ref_decode tp_to_lz4

verify-twophase: lz4_gpu interop-tools
	bash ./verify_twophase.sh ./lz4_gpu

verify-daemon-twophase: lz4_gpu interop-tools
	bash ./verify_daemon_twophase.sh ./lz4_gpu

verify-linux-early:
	bash ./verify_linux_early.sh

formal-run-help:
	python3 ./run_formal_acceptance.py --help

test-formal-tools:
	python3 -m unittest ./test_formal_run_controls.py ./test_formal_resource_probe.py ./test_promote_formal_result.py ./test_gpulz_external_adapter.py ./test_baseline_evidence.py ./test_nvcomp_external_adapter.py

precompile: $(PRECOMPILE_HELPER)
	@echo "Precompiling LZ4 GPU kernels for various hash sizes..."
	@for bits in 14; do \
		echo "Building lz4_gpu_$${bits}.clbin (LZ4_HASHLOG=$$bits)..."; \
		./$(PRECOMPILE_HELPER) -o lz4_gpu_$${bits}.clbin "-I. -DLZ4_HASHLOG=$$bits" || exit 1; \
	done

clean:
	rm -f lz4_gpu lz4_gpu.exe native_lz4_blocks native_lz4_blocks.exe nvcomp_lz4_batched nvcomp_lz4_batched.exe build_clbin build_clbin.exe build_clbin_host tp_ref_decode tp_ref_decode.exe tp_to_lz4 tp_to_lz4.exe *.o *.clbin

.PHONY: all clean nvcomp-baseline precompile interop-tools verify-twophase verify-daemon-twophase verify-linux-early formal-run-help test-formal-tools
