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
#include <limits.h>
#include <errno.h>
#include <sys/stat.h>
#if defined(_WIN32)
#include <process.h>
#include <windows.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
#define XXH_STATIC_LINKING_ONLY
#include "xxhash.h"

static int write_exact(FILE* f, const void* data, size_t size) {
    return size == 0 || (f && fwrite(data, 1, size, f) == size);
}

static int read_exact(FILE* f, void* data, size_t size) {
    return size == 0 || (f && fread(data, 1, size, f) == size);
}

static int put_u32le(FILE* f, uint32_t v) {
    unsigned char b[4] = { (unsigned char)v, (unsigned char)(v>>8), (unsigned char)(v>>16), (unsigned char)(v>>24) };
    return write_exact(f, b, 4);
}

static int paths_identify_same_file(const char* left, const char* right) {
    if (!left || !right) return 0;
    if (strcmp(left, right) == 0) return 1;
#if defined(_WIN32)
    char* left_full = _fullpath(NULL, left, 0);
    char* right_full = _fullpath(NULL, right, 0);
    int same = left_full && right_full && _stricmp(left_full, right_full) == 0;
    free(left_full);
    free(right_full);
    if (same) return 1;
    HANDLE left_handle = CreateFileA(left, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                     NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    HANDLE right_handle = CreateFileA(right, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (left_handle == INVALID_HANDLE_VALUE || right_handle == INVALID_HANDLE_VALUE) {
        if (left_handle != INVALID_HANDLE_VALUE) CloseHandle(left_handle);
        if (right_handle != INVALID_HANDLE_VALUE) CloseHandle(right_handle);
        return 0;
    }
    BY_HANDLE_FILE_INFORMATION left_info, right_info;
    same = GetFileInformationByHandle(left_handle, &left_info) &&
           GetFileInformationByHandle(right_handle, &right_info) &&
           left_info.dwVolumeSerialNumber == right_info.dwVolumeSerialNumber &&
           left_info.nFileIndexHigh == right_info.nFileIndexHigh &&
           left_info.nFileIndexLow == right_info.nFileIndexLow;
    CloseHandle(left_handle);
    CloseHandle(right_handle);
    return same;
#else
    struct stat left_stat, right_stat;
    if (stat(left, &left_stat) != 0 || stat(right, &right_stat) != 0) return 0;
    return left_stat.st_dev == right_stat.st_dev && left_stat.st_ino == right_stat.st_ino;
#endif
}

static int replace_path(const char* temp_path, const char* output_path) {
#if defined(_WIN32)
    return MoveFileExA(temp_path, output_path,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(temp_path, output_path);
#endif
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <frame_N1.lz4tp> <out.lz4>\n", argv[0]); return 2; }
    if (paths_identify_same_file(argv[1], argv[2])) {
        fprintf(stderr, "input and output must differ\n"); return 2;
    }
    int rc = 2, temp_active = 0;
    FILE* f = NULL;
    FILE* o = NULL;
    uint32_t* csz = NULL;
    unsigned char* compressed = NULL;
    struct stat frame_st;
    char temp_path[4096];
    if (stat(argv[1], &frame_st) != 0 || frame_st.st_size < 32 ||
        snprintf(temp_path, sizeof(temp_path), "%s.tmp.%lu", argv[2],
                 (unsigned long)getpid()) >= (int)sizeof(temp_path)) {
        fprintf(stderr, "invalid input or output path\n");
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f) { perror("open frame"); return 2; }
    char magic[8] = {0};
    if (!read_exact(f, magic, 8) || memcmp(magic,"LZ4TP1\0\0",8)!=0) {
        fprintf(stderr,"not LZ4TP1\n"); goto done;
    }
    uint32_t N=0,hl=0,bs=0,nblk=0; uint64_t orig=0;
    if (!read_exact(f, &N, 4) || !read_exact(f, &hl, 4) || !read_exact(f, &bs, 4) ||
        !read_exact(f, &nblk, 4) || !read_exact(f, &orig, 8)) {
        fprintf(stderr, "header read failed\n"); goto done;
    }
    uint64_t expected_nblk = bs ? orig / bs + (orig % bs != 0) : 0;
    if (N != 1 || hl < 11 || hl > 15 || bs == 0 || bs > 4U * 1024U * 1024U ||
        nblk != expected_nblk) {
        fprintf(stderr, "unsupported or inconsistent LZ4TP1 header\n"); goto done;
    }
    size_t meta = (size_t)nblk;                 /* N=1 -> one segment per block */
    if (meta > SIZE_MAX / sizeof(uint32_t)) { fprintf(stderr, "metadata overflow\n"); goto done; }
    csz = (uint32_t*)malloc(meta ? meta * sizeof(uint32_t) : sizeof(uint32_t));
    if (!csz || !read_exact(f, csz, meta * sizeof(uint32_t))) {
        fprintf(stderr,"read sizes\n"); goto done;
    }
    size_t payload_total = 0, max_csz = 0;
    size_t max_block_out = (size_t)bs + (size_t)bs / 255 + 64;
    for (size_t g = 0; g < meta; g++) {
        if (csz[g] == 0 || csz[g] >= 0x80000000U || csz[g] > max_block_out ||
            csz[g] > SIZE_MAX - payload_total) {
            fprintf(stderr, "invalid compressed block size\n"); goto done;
        }
        payload_total += csz[g];
        if (csz[g] > max_csz) max_csz = csz[g];
    }
    size_t meta_bytes = meta * sizeof(uint32_t);
    if (meta_bytes > SIZE_MAX - 32 || payload_total > SIZE_MAX - 32 - meta_bytes ||
        (uint64_t)(32 + meta_bytes + payload_total) != (uint64_t)frame_st.st_size) {
        fprintf(stderr, "frame length mismatch\n"); goto done;
    }
    compressed = (unsigned char*)malloc(max_csz ? max_csz : 1);
    if (!compressed) { fprintf(stderr, "host allocation failed\n"); goto done; }

    remove(temp_path);
    o = fopen(temp_path, "wb");
    if (!o) { perror("open out"); goto done; }
    temp_active = 1;
    /* LZ4 frame header */
    if (!put_u32le(o, 0x184D2204u)) goto write_fail; /* frame magic */
    unsigned char FLG = 0x60;                    /* v01, block-independent, no checksums, no content-size */
    unsigned char BD  = 0x70;                    /* block max = 4 MB (our 64K blocks fit) */
    unsigned char desc[2] = { FLG, BD };
    unsigned char HC = (unsigned char)((XXH32(desc, 2, 0) >> 8) & 0xFF);
    if (!write_exact(o, desc, 2) || !write_exact(o, &HC, 1)) goto write_fail;
    /* one data block per LZ4TP block: 4-byte size (high bit 0 = compressed) + bytes */
    for (size_t g = 0; g < meta; g++) {
        if (!read_exact(f, compressed, csz[g]) || !put_u32le(o, csz[g]) ||
            !write_exact(o, compressed, csz[g])) goto write_fail;
    }
    if (fgetc(f) != EOF || ferror(f) || !put_u32le(o, 0x00000000u) ||
        fflush(o) != 0 || fclose(o) != 0) {
        o = NULL;
        goto write_fail;
    }
    o = NULL;
    if (replace_path(temp_path, argv[2]) != 0) goto write_fail;
    temp_active = 0;
    printf("bridged N=1 frame -> %s (%zu standard LZ4 blocks, block-independent .lz4)\n", argv[2], meta);
    rc = 0;
    goto done;

write_fail:
    fprintf(stderr, "bridge read or write failed\n");

done:
    if (f) fclose(f);
    if (o) fclose(o);
    if (temp_active) remove(temp_path);
    free(csz);
    free(compressed);
    return rc;
}
