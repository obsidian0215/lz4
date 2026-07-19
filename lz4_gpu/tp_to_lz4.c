/* I4c: transcode an N=1 LZ4TP1 frame into a standard .lz4 frame that the stock
 * `lz4` CLI decodes. NO recompression: we copy our exact compressed block bytes.
 * Each N=1 block is already a standalone standard LZ4 block, so we wrap them in a
 * block-independent LZ4 frame (magic + FLG/BD + header-checksum + [blocksize+data]* + endmark).
 * Proves a lossless bridge from HeteroLZ output to the standard LZ4 ecosystem.
 *
 * Build: cc -O2 tp_to_lz4.c /root/lz4/lib/xxhash.c -I/root/lz4/lib -o tp_to_lz4
 * Run:   ./tp_to_lz4 <frame_N1.lz4tp> <out.lz4>   (frame must be N=1)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define XXH_STATIC_LINKING_ONLY
#include "xxhash.h"

static void put_u32le(FILE* f, uint32_t v) {
    unsigned char b[4] = { (unsigned char)v, (unsigned char)(v>>8), (unsigned char)(v>>16), (unsigned char)(v>>24) };
    fwrite(b, 1, 4, f);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <frame_N1.lz4tp> <out.lz4>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open frame"); return 2; }
    char magic[8] = {0};
    if (fread(magic,1,8,f)!=8 || memcmp(magic,"LZ4TP1\0\0",8)!=0) { fprintf(stderr,"not LZ4TP1\n"); return 2; }
    uint32_t N=0,hl=0,bs=0,nblk=0; uint64_t orig=0;
    fread(&N,4,1,f); fread(&hl,4,1,f); fread(&bs,4,1,f); fread(&nblk,4,1,f); fread(&orig,8,1,f);
    if (N != 1) { fprintf(stderr, "this bridge handles N=1 only (got N=%u)\n", N); return 2; }
    size_t meta = (size_t)nblk;                 /* N=1 -> one segment per block */
    uint32_t* csz = (uint32_t*)malloc(meta*4);
    if (fread(csz,4,meta,f)!=meta) { fprintf(stderr,"read sizes\n"); return 2; }
    size_t ptot=0; for (size_t g=0; g<meta; g++) ptot += csz[g];
    unsigned char* payload = (unsigned char*)malloc(ptot);
    if (fread(payload,1,ptot,f)!=ptot) { fprintf(stderr,"read payload\n"); return 2; }
    fclose(f);

    FILE* o = fopen(argv[2], "wb");
    if (!o) { perror("open out"); return 2; }
    /* LZ4 frame header */
    put_u32le(o, 0x184D2204u);                  /* frame magic */
    unsigned char FLG = 0x60;                    /* v01, block-independent, no checksums, no content-size */
    unsigned char BD  = 0x70;                    /* block max = 4 MB (our 64K blocks fit) */
    unsigned char desc[2] = { FLG, BD };
    unsigned char HC = (unsigned char)((XXH32(desc, 2, 0) >> 8) & 0xFF);
    fwrite(desc, 1, 2, o);
    fwrite(&HC, 1, 1, o);
    /* one data block per LZ4TP block: 4-byte size (high bit 0 = compressed) + bytes */
    size_t off = 0;
    for (size_t g = 0; g < meta; g++) {
        uint32_t sz = csz[g];                    /* high bit 0 -> LZ4-compressed block */
        put_u32le(o, sz);
        fwrite(payload + off, 1, sz, o);
        off += sz;
    }
    put_u32le(o, 0x00000000u);                   /* end mark */
    fclose(o);
    printf("bridged N=1 frame -> %s (%zu standard LZ4 blocks, block-independent .lz4)\n", argv[2], meta);
    return 0;
}
