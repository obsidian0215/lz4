CC ?= cc
HOSTCC ?= cc

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
LDFLAGS ?= $(OPENCL_LDFLAGS) -lm $(PTHREAD_LIB)
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

COMMON_SRCS = lz4_gpu.c lz4_gpu_utils.c lz4_gpu_core.c
ifeq ($(OS),Windows_NT)
DAEMON_SRCS =
else
DAEMON_SRCS = lz4_gpu_daemon.c lz4_gpu_client.c
endif
SRCS = $(COMMON_SRCS) $(DAEMON_SRCS)
OBJS = $(SRCS:.c=.o)

all: lz4_gpu build_clbin precompile

lz4_gpu: $(SRCS)
	$(CC) $(CFLAGS) -o $@$(EXEEXT) $^ $(LDFLAGS)

build_clbin: build_clbin.c
	$(CC) $(CFLAGS) -o $@$(EXEEXT) $^ $(LDFLAGS)

build_clbin_host: build_clbin.c
	$(HOSTCC) -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lib -I.. $(HOST_OPENCL_CFLAGS) -o $@$(HOSTEXEEXT) $^ $(HOST_OPENCL_LDFLAGS)

precompile: $(PRECOMPILE_HELPER)
	@echo "Precompiling LZ4 GPU kernels for various hash sizes..."
	@for bits in 14 15; do \
		echo "Building lz4_gpu_$$bits.clbin (LZ4_HASHLOG=$$bits)..."; \
		./$(PRECOMPILE_HELPER) -o lz4_gpu_$$bits.clbin "-I. -DLZ4_HASHLOG=$$bits" || exit 1; \
	done

clean:
	rm -f lz4_gpu lz4_gpu.exe build_clbin build_clbin.exe build_clbin_host *.o *.clbin

.PHONY: all clean precompile
