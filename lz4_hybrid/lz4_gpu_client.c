#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_utils.h"

extern int g_verbose;

int is_daemon_running(void) {
    return access(SOCKET_PATH, F_OK) == 0;
}

static int do_daemon_op(int mode, const char* input, const char* output, int block_size, int acceleration, int local_size,
                        uint32_t cpu_share_pct, uint32_t cpu_threads, uint32_t adaptive) {
    int sock;
    struct sockaddr_un addr;
    request_t req;
    response_t resp;
    struct stat st;

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
    req.mode = mode;
    strncpy(req.input_path, input, sizeof(req.input_path) - 1);
    strncpy(req.output_path, output, sizeof(req.output_path) - 1);
    req.block_size = block_size;
    req.acceleration = acceleration;
    req.local_size = local_size;
    req.cpu_share_pct = cpu_share_pct;
    req.cpu_threads = cpu_threads;
    req.adaptive = adaptive;

    if (send(sock, &req, sizeof(req), 0) != sizeof(req)) {
        perror("send request");
        close(sock);
        return -1;
    }

    if (recv(sock, &resp, sizeof(resp), 0) != sizeof(resp)) {
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
        fprintf(stderr, "Daemon operation failed: %d\n", resp.status);
        return -1;
    }
}

int run_lz4_client(int mode, const char* input_path, const char* output_path, int block_size, int acceleration, int local_size,
                   uint32_t cpu_share_pct, uint32_t cpu_threads, uint32_t adaptive) {
    return do_daemon_op(mode, input_path, output_path, block_size, acceleration, local_size,
                        cpu_share_pct, cpu_threads, adaptive);
}
