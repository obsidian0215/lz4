#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_utils.h"

extern int g_verbose;

int is_daemon_running(void) {
    return access(SOCKET_PATH, F_OK) == 0;
}

static int client_read_full(int fd, void* buf, size_t len);
static int client_write_full(int fd, const void* buf, size_t len);

static int do_daemon_op(int mode, const char* input, const char* output, int block_size, int acceleration, int local_size, int hash_log, int twophase) {
    int sock;
    struct sockaddr_un addr;
    request_t req;
    response_t resp;
    struct stat st;

    if (!input || !output || strlen(input) >= sizeof(req.input_path) ||
        strlen(output) >= sizeof(req.output_path)) {
        fprintf(stderr, "Error: daemon input or output path is too long\n");
        return -1;
    }

    if (!is_daemon_running()) {
        fprintf(stderr, "Error: Daemon is not running. Start it with --daemon\n");
        return -1;
    }

    if (stat(input, &st) != 0) {
        perror("stat input");
        return -1;
    }

    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect daemon");
        close(sock);
        return -1;
    }

    memset(&req, 0, sizeof(req));
    req.magic = LZ4_DAEMON_REQUEST_MAGIC;
    req.version = LZ4_DAEMON_REQUEST_VERSION;
    req.mode = mode;
    memcpy(req.input_path, input, strlen(input) + 1);
    memcpy(req.output_path, output, strlen(output) + 1);
    req.block_size = block_size;
    req.acceleration = acceleration;
    req.local_size = local_size;
    req.hash_log = hash_log;
    if (twophase && mode == mode_compress) req.flags |= LZ4_DAEMON_FLAG_TWOPHASE;

    if (client_write_full(sock, &req, sizeof(req)) != 0) {
        perror("send request");
        close(sock);
        return -1;
    }

    if (client_read_full(sock, &resp, sizeof(resp)) != 0) {
        perror("recv response");
        close(sock);
        return -1;
    }

    close(sock);

    if (resp.status == 0) {
        if (g_verbose) {
            print_response_stats(&resp, input, mode);
        } else {
            printf("%s : %zu -> %zu bytes, %.2f ms\n", input, (size_t)st.st_size, (size_t)resp.out_size, resp.time_us / 1000.0);
        }
        return 0;
    } else {
        fprintf(stderr, "Daemon operation failed: %d%s%s\n", resp.status,
                resp.message[0] ? " - " : "", resp.message);
        return -1;
    }
}

static int client_read_full(int fd, void* buf, size_t len)
{
    unsigned char* p = (unsigned char*)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int client_write_full(int fd, const void* buf, size_t len)
{
    const unsigned char* p = (const unsigned char*)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int read_stdin_all(unsigned char** out, size_t* out_len)
{
    size_t cap = 1 << 20;
    size_t len = 0;
    unsigned char* buf = (unsigned char*)malloc(cap);
    if (!buf) return -1;
    for (;;) {
        size_t got;
        if (len == cap) {
            if (cap > SIZE_MAX / 2) {
                free(buf);
                return -1;
            }
            size_t next = cap * 2;
            unsigned char* nb = (unsigned char*)realloc(buf, next);
            if (!nb) {
                free(buf);
                return -1;
            }
            buf = nb;
            cap = next;
        }
        got = fread(buf + len, 1, cap - len, stdin);
        len += got;
        if (got == 0) {
            if (ferror(stdin)) {
                free(buf);
                return -1;
            }
            break;
        }
    }
    *out = buf;
    *out_len = len;
    return 0;
}

static int write_stdout_all(const void* data, size_t len)
{
    return fwrite(data, 1, len, stdout) == len ? 0 : -1;
}

static int do_daemon_raw_op(int mode, int block_size, int acceleration, int local_size, int hash_log)
{
    int sock;
    struct sockaddr_un addr;
    request_t req;
    response_t resp;
    unsigned char* input = NULL;
    size_t input_len = 0;
    uint64_t out_len = 0;
    unsigned char* out = NULL;
    int rc = -1;

    if (!is_daemon_running()) {
        fprintf(stderr, "Error: Daemon is not running. Start it with --daemon\n");
        return -1;
    }
    if (read_stdin_all(&input, &input_len) != 0 || input_len == 0) {
        fprintf(stderr, "Error: failed to read stdin raw payload\n");
        free(input);
        return -1;
    }

    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        free(input);
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect daemon");
        close(sock);
        free(input);
        return -1;
    }

    memset(&req, 0, sizeof(req));
    req.magic = LZ4_DAEMON_REQUEST_MAGIC;
    req.version = LZ4_DAEMON_REQUEST_VERSION;
    req.mode = mode;
    req.block_size = block_size;
    req.acceleration = acceleration;
    req.local_size = local_size;
    req.hash_log = hash_log;
    req.flags = LZ4_DAEMON_FLAG_RAW_BUFFER;
    req.input_size = input_len;

    if (client_write_full(sock, &req, sizeof(req)) != 0 ||
        client_write_full(sock, input, input_len) != 0 ||
        client_read_full(sock, &resp, sizeof(resp)) != 0) {
        perror("daemon raw exchange failed");
        goto out;
    }
    if (resp.status != 0) {
        fprintf(stderr, "Daemon raw operation failed: %d\n", resp.status);
        goto out;
    }
    if (client_read_full(sock, &out_len, sizeof(out_len)) != 0) goto out;
    if (out_len > 0) {
        if (out_len > (uint64_t)SIZE_MAX) goto out;
        out = (unsigned char*)malloc((size_t)out_len);
        if (!out) goto out;
        if (client_read_full(sock, out, (size_t)out_len) != 0) goto out;
        if (write_stdout_all(out, (size_t)out_len) != 0) goto out;
    }
    rc = 0;

out:
    free(out);
    free(input);
    close(sock);
    return rc;
}

int run_lz4_client(int mode, const char* input_path, const char* output_path, int block_size, int acceleration, int local_size, int hash_log, int raw_buffer, int twophase) {
    if (raw_buffer) {
        return do_daemon_raw_op(mode, block_size, acceleration, local_size, hash_log);
    }
    return do_daemon_op(mode, input_path, output_path, block_size, acceleration, local_size, hash_log, twophase);
}
