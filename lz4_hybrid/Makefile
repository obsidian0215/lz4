CC ?= cc

ifeq ($(OS),Windows_NT)
EXEEXT ?= .exe
PTHREAD_LIB ?= -lpthread
OPENCL_INCLUDE_DIR ?=
OPENCL_LIB_DIR ?=
OPENCL_LIB_NAME ?= OpenCL
OPENCL_CFLAGS := $(if $(OPENCL_INCLUDE_DIR),-I$(OPENCL_INCLUDE_DIR),)
OPENCL_LDFLAGS := $(if $(OPENCL_LIB_DIR),-L$(OPENCL_LIB_DIR),) -l$(OPENCL_LIB_NAME)
CFLAGS ?= -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -DWIN32_LEAN_AND_MEAN -I. -I../lz4_gpu -I../lib $(OPENCL_CFLAGS)
LDFLAGS ?= $(OPENCL_LDFLAGS) -lm $(PTHREAD_LIB)
else
EXEEXT ?=
OPENCL_CFLAGS ?=
OPENCL_LDFLAGS ?=-lOpenCL
CFLAGS ?= -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lz4_gpu -I../lib $(OPENCL_CFLAGS)
LDFLAGS ?= $(OPENCL_LDFLAGS) -lm -lpthread
endif

all: lz4_hybrid

lz4_hybrid: lz4_hybrid.c ../lz4_gpu/lz4_gpu_core.c ../lz4_gpu/lz4_gpu_utils.c ../lib/lz4.c
	$(CC) $(CFLAGS) -o $@$(EXEEXT) $^ $(LDFLAGS)

clean:
	rm -f lz4_hybrid lz4_hybrid.exe

.PHONY: all clean
