CC = cc
CFLAGS = -O2 -Wall -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lz4_gpu -I../lib
LDFLAGS = -lOpenCL -lm -lpthread

all: lz4_hybrid

lz4_hybrid: lz4_hybrid.c ../lz4_gpu/lz4_gpu_core.c ../lz4_gpu/lz4_gpu_utils.c ../lib/lz4.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f lz4_hybrid

.PHONY: all clean
