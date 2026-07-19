#include "lz4_tp_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <errno.h>
#if defined(_WIN32)
#include <windows.h>
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#define LZ4_TP_PROFILE_TAG "LZ4TP_PROFILE_V1"

typedef struct {
    int block_size;
    int hash_log;
    int w_sat;
    char device_name[256];
} lz4_tp_profile_record_t;

static int parse_record(char* line, lz4_tp_profile_record_t* record) {
    char* save = NULL;
    char* tag = strtok_r(line, "\t", &save);
    char* block = strtok_r(NULL, "\t", &save);
    char* hash = strtok_r(NULL, "\t", &save);
    char* w_sat = strtok_r(NULL, "\t", &save);
    char* device = strtok_r(NULL, "\r\n", &save);
    if (!tag || !block || !hash || !w_sat || !device || strcmp(tag, LZ4_TP_PROFILE_TAG) != 0) return -1;

    char* end = NULL;
    errno = 0;
    long block_value = strtol(block, &end, 10);
    if (errno == ERANGE || !end || *end != 0 || block_value <= 0 || block_value > INT_MAX) return -1;
    errno = 0;
    long hash_value = strtol(hash, &end, 10);
    if (errno == ERANGE || !end || *end != 0 || hash_value < 11 || hash_value > 15) return -1;
    errno = 0;
    long w_value = strtol(w_sat, &end, 10);
    if (errno == ERANGE || !end || *end != 0 || w_value <= 0 || w_value > INT_MAX) return -1;

    record->block_size = (int)block_value;
    record->hash_log = (int)hash_value;
    record->w_sat = (int)w_value;
    size_t n = strlen(device);
    if (n >= sizeof(record->device_name)) return -1;
    memcpy(record->device_name, device, n);
    record->device_name[n] = 0;
    return record->device_name[0] ? 0 : -1;
}

static int same_key(const lz4_tp_profile_record_t* record, const char* device_name,
                    int block_size, int hash_log) {
    return record->block_size == block_size && record->hash_log == hash_log &&
           strcmp(record->device_name, device_name) == 0;
}

int lz4_tp_profile_lookup(const char* path, const char* device_name,
                          int block_size, int hash_log) {
    if (!path || !device_name || !*device_name || block_size <= 0 ||
        hash_log < 11 || hash_log > 15) return 0;
    FILE* f = fopen(path, "r");
    if (!f) return 0;

    char line[768];
    int result = 0;
    while (fgets(line, sizeof(line), f)) {
        lz4_tp_profile_record_t record;
        if (parse_record(line, &record) == 0 && same_key(&record, device_name, block_size, hash_log)) {
            result = record.w_sat;
            break;
        }
    }
    fclose(f);
    return result;
}

int lz4_tp_profile_store(const char* path, const char* device_name,
                         int block_size, int hash_log, int w_sat) {
    if (!path || !device_name || !*device_name || block_size <= 0 ||
        hash_log < 11 || hash_log > 15 || w_sat <= 0 ||
        strlen(device_name) >= sizeof(((lz4_tp_profile_record_t*)0)->device_name) ||
        strpbrk(device_name, "\t\r\n") != NULL) return -1;

    lz4_tp_profile_record_t* records = NULL;
    size_t count = 0, capacity = 0;
    FILE* f = fopen(path, "r");
    if (f) {
        char line[768];
        while (fgets(line, sizeof(line), f)) {
            lz4_tp_profile_record_t record;
            if (parse_record(line, &record) != 0) continue;
            if (count == capacity) {
                if (capacity > SIZE_MAX / 2) { fclose(f); free(records); return -1; }
                size_t next_capacity = capacity ? capacity * 2 : 8;
                if (next_capacity > SIZE_MAX / sizeof(*records)) {
                    fclose(f); free(records); return -1;
                }
                lz4_tp_profile_record_t* next =
                    (lz4_tp_profile_record_t*)realloc(records, next_capacity * sizeof(*records));
                if (!next) { fclose(f); free(records); return -1; }
                records = next;
                capacity = next_capacity;
            }
            records[count++] = record;
        }
        fclose(f);
    }

    int found = 0;
    for (size_t i = 0; i < count; i++) {
        if (same_key(&records[i], device_name, block_size, hash_log)) {
            records[i].w_sat = w_sat;
            found = 1;
            break;
        }
    }
    if (!found) {
        if (count == capacity) {
            if (capacity > SIZE_MAX / 2) { free(records); return -1; }
            size_t next_capacity = capacity ? capacity * 2 : 8;
            if (next_capacity > SIZE_MAX / sizeof(*records)) { free(records); return -1; }
            lz4_tp_profile_record_t* next =
                (lz4_tp_profile_record_t*)realloc(records, next_capacity * sizeof(*records));
            if (!next) { free(records); return -1; }
            records = next;
            capacity = next_capacity;
        }
        records[count].block_size = block_size;
        records[count].hash_log = hash_log;
        records[count].w_sat = w_sat;
        size_t n = strlen(device_name);
        memcpy(records[count].device_name, device_name, n);
        records[count].device_name[n] = 0;
        count++;
    }

    if (strlen(path) > SIZE_MAX - 32) { free(records); return -1; }
    size_t temp_len = strlen(path) + 32;
    char* temp_path = (char*)malloc(temp_len);
    if (!temp_path) { free(records); return -1; }
    snprintf(temp_path, temp_len, "%s.tmp.%lu", path, (unsigned long)getpid());
    f = fopen(temp_path, "w");
    if (!f) { free(temp_path); free(records); return -1; }
    int status = 0;
    for (size_t i = 0; i < count; i++) {
        if (fprintf(f, "%s\t%d\t%d\t%d\t%s\n", LZ4_TP_PROFILE_TAG,
                    records[i].block_size, records[i].hash_log, records[i].w_sat,
                    records[i].device_name) < 0) {
            status = -1;
            break;
        }
    }
    if (fclose(f) != 0) status = -1;
    if (status == 0) {
#if defined(_WIN32)
        if (!MoveFileExA(temp_path, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) status = -1;
#else
        if (rename(temp_path, path) != 0) status = -1;
#endif
    }
    if (status != 0) remove(temp_path);
    free(temp_path);
    free(records);
    return status;
}
