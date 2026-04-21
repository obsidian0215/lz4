ifeq ($(OS),Windows_NT)
MSYS2_USR_BIN ?= C:/msys64/usr/bin
ifneq ($(wildcard $(MSYS2_USR_BIN)/sh.exe),)
SHELL := $(MSYS2_USR_BIN)/sh.exe
export PATH := $(MSYS2_USR_BIN);$(PATH)
endif
endif

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
PTHREAD_CFLAGS ?= -pthread
PTHREAD_LIB ?= -pthread
OPENCL_CFLAGS ?=
OPENCL_LDFLAGS ?=-lOpenCL
CFLAGS ?= -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lz4_gpu -I../lib $(OPENCL_CFLAGS) $(PTHREAD_CFLAGS)
LDFLAGS ?= $(OPENCL_LDFLAGS) -lm $(PTHREAD_LIB)
endif

GPU_DIR = ../lz4_gpu

all: sync-kernels lz4_hybrid

lz4_hybrid: lz4_hybrid.c ../lz4_gpu/lz4_gpu_core.c ../lz4_gpu/lz4_gpu_utils.c ../lib/lz4.c
	$(CC) $(CFLAGS) -o $@$(EXEEXT) $^ $(LDFLAGS)

sync-kernels:
	@for f in $(GPU_DIR)/*.cl $(GPU_DIR)/*.clbin; do \
		[ -e "$$f" ] || continue; \
		bn=$$(basename "$$f"); \
		needs_sync=0; \
		if [ ! -e "$$bn" ]; then \
			needs_sync=1; \
		elif [ -f "$$bn" ]; then \
			linktxt=$$(tr -d '\r\n' < "$$bn" 2>/dev/null || true); \
			case "$$linktxt" in \
				../lz4_gpu/*) needs_sync=1 ;; \
			esac; \
		fi; \
		if [ $$needs_sync -eq 1 ]; then ln -sf "$$f" "$$bn" 2>/dev/null || cp -f "$$f" "$$bn"; fi; \
	done

clean:
	rm -f lz4_hybrid lz4_hybrid.exe
	rm -f *.cl *.clbin

.PHONY: all clean sync-kernels
