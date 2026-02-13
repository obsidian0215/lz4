CC = cc
CFLAGS = -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lib -I..
LDFLAGS = -lOpenCL -lm -lpthread

SRCS = lz4_gpu.c lz4_gpu_daemon.c lz4_gpu_client.c lz4_gpu_utils.c lz4_gpu_core.c
OBJS = $(SRCS:.c=.o)

all: lz4_gpu build_clbin precompile

lz4_gpu: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

build_clbin: build_clbin.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

precompile: build_clbin
	@echo "Precompiling LZ4 GPU kernels for various hash sizes..."
	@for bits in 12 13 14 15 16; do \
		echo "Building lz4_gpu_$$bits.clbin (LZ4_HASHLOG=$$bits)..."; \
		./build_clbin -o lz4_gpu_$$bits.clbin "-I. -DLZ4_HASHLOG=$$bits" || exit 1; \
	done

clean:
	rm -f lz4_gpu build_clbin *.o *.clbin

.PHONY: all clean precompile
