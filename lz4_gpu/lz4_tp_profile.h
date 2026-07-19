#ifndef LZ4_TP_PROFILE_H
#define LZ4_TP_PROFILE_H

int lz4_tp_profile_lookup(const char* path, const char* device_name,
                          int block_size, int hash_log);
int lz4_tp_profile_store(const char* path, const char* device_name,
                         int block_size, int hash_log, int w_sat);

#endif
