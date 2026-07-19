/* I4b: reconstruct a LZ4TP1 frame using ONLY the reference liblz4 block API.
 *
 * No HeteroLZ kernel, no OpenCL, no custom block decoder. If this reproduces
 * the original byte-for-byte, our compressed payload IS standard LZ4 block
 * format: seg 0 of each 64K block is a standalone LZ4 block (LZ4_decompress_safe);
 * seg s>0 references the cumulative prefix, decoded by the stock library's
 * LZ4_decompress_safe_usingDict in contiguous prefix mode (dictStart+dictSize==dst).
 *
 * Build:  cc -O2 tp_ref_decode.c -llz4 -o tp_ref_decode
 * Run:    ./tp_ref_decode <frame.lz4tp> <original_file>
 * Exit:   0 = byte-identical reconstruction, 1 = mismatch/decode error, 2 = I/O.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <sys/stat.h>
#include <lz4.h>

static int read_exact(FILE* f, void* data, size_t size) {
    return size == 0 || (f && fread(data, 1, size, f) == size);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <frame.lz4tp> <original>\n", argv[0]); return 2; }

    int rc = 2;
    FILE* f = NULL;
    FILE* original_file = NULL;
    uint32_t* csz = NULL;
    unsigned char* compressed = NULL;
    unsigned char* block = NULL;
    unsigned char* reference = NULL;
    struct stat frame_st, original_st;
    if (stat(argv[1], &frame_st) != 0 || frame_st.st_size < 32 ||
        stat(argv[2], &original_st) != 0 || original_st.st_size < 0) {
        fprintf(stderr, "invalid input files\n");
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f) { perror("open frame"); return 2; }
    char magic[8] = {0};
    if (!read_exact(f, magic, 8) || memcmp(magic, "LZ4TP1\0\0", 8) != 0) {
        fprintf(stderr, "not a LZ4TP1 frame\n"); goto done;
    }
    uint32_t N = 0, hl = 0, bs = 0, nblk = 0; uint64_t orig = 0;
    if (!read_exact(f, &N, 4) || !read_exact(f, &hl, 4) || !read_exact(f, &bs, 4) ||
        !read_exact(f, &nblk, 4) || !read_exact(f, &orig, 8)) {
        fprintf(stderr,"hdr read\n"); goto done;
    }
    uint64_t expected_nblk = bs ? orig / bs + (orig % bs != 0) : 0;
    if ((N != 1 && N != 2 && N != 4 && N != 8) || hl < 11 || hl > 15 ||
        bs == 0 || bs > INT_MAX || orig > SIZE_MAX || nblk != expected_nblk ||
        (uint64_t)original_st.st_size != orig) {
        fprintf(stderr, "bad header or original size\n"); goto done;
    }

    if (orig == 0) {
        if (frame_st.st_size != 32 || fgetc(f) != EOF) {
            fprintf(stderr, "trailing data in empty frame\n"); goto done;
        }
        printf("REF-OK  N=%u nblk=0  0 bytes reconstructed by reference liblz4 [empty frame] (v%s)\n",
               N, LZ4_versionString());
        rc = 0;
        goto done;
    }

    if ((size_t)nblk > SIZE_MAX / (size_t)N) { fprintf(stderr, "metadata overflow\n"); goto done; }
    size_t meta = (size_t)nblk * (size_t)N;
    if (meta > SIZE_MAX / sizeof(uint32_t)) { fprintf(stderr, "metadata overflow\n"); goto done; }
    csz = (uint32_t*)malloc(meta * sizeof(uint32_t));
    if (!csz || !read_exact(f, csz, meta * sizeof(uint32_t))) {
        fprintf(stderr, "read sizes\n"); goto done;
    }
    size_t payload_total = 0, max_csz = 0;
    for (size_t g = 0; g < meta; g++) {
        if (csz[g] == 0 || csz[g] > INT_MAX || csz[g] > SIZE_MAX - payload_total) {
            fprintf(stderr, "invalid compressed size\n"); goto done;
        }
        payload_total += csz[g];
        if (csz[g] > max_csz) max_csz = csz[g];
    }
    size_t meta_bytes = meta * sizeof(uint32_t);
    if (meta_bytes > SIZE_MAX - 32 || payload_total > SIZE_MAX - 32 - meta_bytes ||
        (uint64_t)(32 + meta_bytes + payload_total) != (uint64_t)frame_st.st_size) {
        fprintf(stderr, "frame length mismatch\n"); goto done;
    }
    compressed = (unsigned char*)malloc(max_csz);
    block = (unsigned char*)malloc(bs);
    reference = (unsigned char*)malloc(bs);
    original_file = fopen(argv[2], "rb");
    if (!compressed || !block || !reference || !original_file) {
        fprintf(stderr, "host allocation or original open failed\n"); goto done;
    }

    size_t outpos = 0;
    int used_dict = 0;

    for (uint32_t b = 0; b < nblk; b++) {
        int op = 0;
        for (uint32_t s = 0; s < N; s++) {
            size_t g = (size_t)b * N + s;
            if (!read_exact(f, compressed, csz[g])) {
                fprintf(stderr, "read payload\n"); goto done;
            }
            const char* src = (const char*)compressed;
            int ssz = (int)csz[g];
            int n;
            if (s == 0) {
                n = LZ4_decompress_safe(src, (char*)block, ssz, (int)bs);
            } else {
                if (op < 0 || op > (int)bs) { fprintf(stderr, "block overflow\n"); rc = 1; goto done; }
                /* dictStart=block, dictSize=op, dst=block+op -> contiguous prefix */
                n = LZ4_decompress_safe_usingDict(src, (char*)(block + op), ssz,
                                                  (int)bs - op, (const char*)block, op);
                used_dict = 1;
            }
            if (n < 0 || n > (int)bs - op) {
                fprintf(stderr, "REF DECODE FAIL blk=%u seg=%u ret=%d\n", b, s, n);
                rc = 1; goto done;
            }
            op += n;
        }
        size_t expected = (size_t)orig - outpos;
        if (expected > bs) expected = bs;
        if ((size_t)op != expected || !read_exact(original_file, reference, expected)) {
            fprintf(stderr, "LENGTH MISMATCH block=%u got=%d expect=%zu\n", b, op, expected);
            rc = 1; goto done;
        }
        if (memcmp(block, reference, expected) != 0) {
            size_t i;
            for (i = 0; i < expected; i++) if (block[i] != reference[i]) break;
            fprintf(stderr, "BYTE MISMATCH at offset %zu (got %02x want %02x)\n",
                    outpos + i, block[i], reference[i]);
            rc = 1; goto done;
        }
        outpos += expected;
    }

    if (outpos != (size_t)orig || fgetc(f) != EOF || fgetc(original_file) != EOF) {
        fprintf(stderr, "trailing data or final length mismatch\n");
        rc = 1; goto done;
    }

    printf("REF-OK  N=%u nblk=%u  %llu bytes reconstructed by reference liblz4 %s (v%s)\n",
           N, nblk, (unsigned long long)orig,
           used_dict ? "[LZ4_decompress_safe + LZ4_decompress_safe_usingDict prefix]"
                     : "[LZ4_decompress_safe only]",
           LZ4_versionString());
    rc = 0;

done:
    if (f) fclose(f);
    if (original_file) fclose(original_file);
    free(csz);
    free(compressed);
    free(block);
    free(reference);
    return rc;
}
