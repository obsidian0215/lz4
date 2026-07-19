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
#include <lz4.h>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <frame.lz4tp> <original>\n", argv[0]); return 2; }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open frame"); return 2; }
    char magic[8] = {0};
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "LZ4TP1\0\0", 8) != 0) {
        fprintf(stderr, "not a LZ4TP1 frame\n"); return 2;
    }
    uint32_t N = 0, hl = 0, bs = 0, nblk = 0; uint64_t orig = 0;
    if (fread(&N,4,1,f)!=1 || fread(&hl,4,1,f)!=1 || fread(&bs,4,1,f)!=1 ||
        fread(&nblk,4,1,f)!=1 || fread(&orig,8,1,f)!=1) { fprintf(stderr,"hdr read\n"); return 2; }
    if (N < 1 || bs == 0 || nblk == 0) { fprintf(stderr, "bad header\n"); return 2; }

    size_t meta = (size_t)nblk * N;
    uint32_t* csz = (uint32_t*)malloc(meta * 4);
    if (fread(csz, 4, meta, f) != meta) { fprintf(stderr, "read sizes\n"); return 2; }
    size_t ptot = 0; for (size_t g = 0; g < meta; g++) ptot += csz[g];
    unsigned char* payload = (unsigned char*)malloc(ptot ? ptot : 1);
    if (fread(payload, 1, ptot, f) != ptot) { fprintf(stderr, "read payload\n"); return 2; }
    fclose(f);

    size_t* coff = (size_t*)malloc(meta * sizeof(size_t));
    { size_t acc = 0; for (size_t g = 0; g < meta; g++) { coff[g] = acc; acc += csz[g]; } }

    unsigned char* out = (unsigned char*)malloc(orig ? orig : 1);
    unsigned char* blk = (unsigned char*)malloc(bs);
    size_t outpos = 0;
    int used_dict = 0;

    for (uint32_t b = 0; b < nblk; b++) {
        int op = 0;
        for (uint32_t s = 0; s < N; s++) {
            size_t g = (size_t)b * N + s;
            const char* src = (const char*)(payload + coff[g]);
            int ssz = (int)csz[g];
            int n;
            if (s == 0) {
                n = LZ4_decompress_safe(src, (char*)blk, ssz, (int)bs);
            } else {
                /* dictStart=blk, dictSize=op, dst=blk+op -> contiguous prefix */
                n = LZ4_decompress_safe_usingDict(src, (char*)(blk + op), ssz,
                                                  (int)bs - op, (const char*)blk, op);
                used_dict = 1;
            }
            if (n < 0) { fprintf(stderr, "REF DECODE FAIL blk=%u seg=%u ret=%d\n", b, s, n); return 1; }
            op += n;
        }
        if (outpos + (size_t)op > orig) { fprintf(stderr, "overflow blk=%u\n", b); return 1; }
        memcpy(out + outpos, blk, op); outpos += op;
    }

    if (outpos != orig) {
        fprintf(stderr, "LENGTH MISMATCH got=%zu expect=%llu\n", outpos, (unsigned long long)orig);
        return 1;
    }

    FILE* g = fopen(argv[2], "rb");
    if (!g) { perror("open orig"); return 2; }
    unsigned char* ref = (unsigned char*)malloc(orig ? orig : 1);
    if (fread(ref, 1, orig, g) != orig) { fprintf(stderr, "read orig\n"); return 2; }
    fclose(g);

    if (memcmp(out, ref, orig) != 0) {
        size_t i; for (i = 0; i < orig; i++) if (out[i] != ref[i]) break;
        fprintf(stderr, "BYTE MISMATCH at offset %zu (got %02x want %02x)\n", i, out[i], ref[i]);
        return 1;
    }

    printf("REF-OK  N=%u nblk=%u  %llu bytes reconstructed by reference liblz4 %s (v%s)\n",
           N, nblk, (unsigned long long)orig,
           used_dict ? "[LZ4_decompress_safe + LZ4_decompress_safe_usingDict prefix]"
                     : "[LZ4_decompress_safe only]",
           LZ4_versionString());
    return 0;
}
