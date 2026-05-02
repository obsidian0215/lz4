ifeq ($(OS),Windows_NT)
MSYS2_USR_BIN ?= C:/msys64/usr/bin
ifneq ($(wildcard $(MSYS2_USR_BIN)/sh.exe),)
SHELL := $(MSYS2_USR_BIN)/sh.exe
export PATH := $(MSYS2_USR_BIN);$(PATH)
endif
endif

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

COMMON_SRCS = lz4_hybrid.c lz4_gpu_utils.c lz4_gpu_core.c
ifeq ($(OS),Windows_NT)
DAEMON_SRCS =
else
DAEMON_SRCS = lz4_gpu_daemon.c lz4_gpu_client.c
endif
SRCS = $(COMMON_SRCS) $(DAEMON_SRCS)

all: lz4_hybrid build_clbin

lz4_hybrid: $(SRCS)
	$(CC) $(CFLAGS) -o $@$(EXEEXT) $^ $(LDFLAGS)

build_clbin: build_clbin.c
	$(CC) $(CFLAGS) -o $@$(EXEEXT) $^ $(LDFLAGS)

build_clbin_host: build_clbin.c
	$(HOSTCC) -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lib -I.. $(HOST_OPENCL_CFLAGS) -o $@$(HOSTEXEEXT) $^ $(HOST_OPENCL_LDFLAGS)

precompile: $(PRECOMPILE_HELPER)
	@echo "Precompiling LZ4 hybrid OpenCL kernels..."
	@for bits in 14; do \
		echo "Building lz4_gpu_$${bits}.clbin (LZ4_HASHLOG=$$bits)..."; \
		./$(PRECOMPILE_HELPER) -o lz4_gpu_$${bits}.clbin "-I. -DLZ4_HASHLOG=$$bits" || exit 1; \
	done

clean:
	rm -f lz4_hybrid lz4_hybrid.exe build_clbin build_clbin.exe build_clbin_host *.o

clean-clbin:
	rm -f *.clbin

distclean: clean clean-clbin

.PHONY: all clean clean-clbin distclean precompile
