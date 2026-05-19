#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <limits.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <CL/cl.h>
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_core.h"
#include "lz4_gpu_utils.h"

extern long syscall(long number, ...);

#define MAX_WORKERS_CAP 16
#define LZ4_DAEMON_KERNEL_MODES 2

enum {
    LZ4_DAEMON_MODE_CLEAR16 = 0,
    LZ4_DAEMON_MODE_EPOCH32 = 1
};

typedef struct {
    int id;
    cl_command_queue queue;
    cl_kernel kernel_comp[LZ4_DAEMON_KERNEL_MODES];
    cl_kernel kernel_decomp[LZ4_DAEMON_KERNEL_MODES];
    lz4_gpu_workspace_t ws;
    pthread_t thread;
    int client_fd;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int has_work;
} worker_res_t;

struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program_comp[LZ4_DAEMON_KERNEL_MODES];
    pthread_mutex_t compile_lock;
    worker_res_t workers[MAX_WORKERS_CAP];
    int active_workers;
    int server_sock;
    volatile int running;
    int pid_fd;
} g_state;

static int daemon_kernel_mode_for_block(uint32_t block_size) {
    return (block_size <= 64U * 1024U) ? LZ4_DAEMON_MODE_CLEAR16 : LZ4_DAEMON_MODE_EPOCH32;
}

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int daemon_read_full(int fd, void* buf, size_t len) {
    unsigned char* p = (unsigned char*)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int daemon_write_full(int fd, const void* buf, size_t len) {
    const unsigned char* p = (const unsigned char*)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int daemon_fd_path(int fd, char* out, size_t out_len) {
    if (!out || out_len == 0 || fd < 0) return -1;
    if (snprintf(out, out_len, "/proc/self/fd/%d", fd) >= (int)out_len) return -1;
    return 0;
}

static int daemon_memfd_create(const char* name) {
#ifdef SYS_memfd_create
    return (int)syscall(SYS_memfd_create, name ? name : "lz4_gpu_daemon", 0);
#else
    (void)name;
    return -1;
#endif
}

static int daemon_write_fd_full(int fd, const void* buf, size_t len) {
    const unsigned char* p = (const unsigned char*)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int daemon_send_fd_payload(int sock, int fd) {
    struct stat st;
    char* buf = NULL;
    uint64_t len = 0;
    int rc = -1;
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_size < 0) {
        len = 0;
        (void)daemon_write_full(sock, &len, sizeof(len));
        return -1;
    }
    len = (uint64_t)st.st_size;
    if (daemon_write_full(sock, &len, sizeof(len)) != 0) return -1;
    if (len == 0) return 0;
    buf = (char*)malloc(1 << 20);
    if (!buf) return -1;
    if (lseek(fd, 0, SEEK_SET) < 0) {
        free(buf);
        return -1;
    }
    while (len > 0) {
        size_t want = (len > (uint64_t)(1 << 20)) ? (size_t)(1 << 20) : (size_t)len;
        size_t got = read(fd, buf, want);
        if (got == 0) goto out;
        if (daemon_write_full(sock, buf, got) != 0) goto out;
        len -= got;
    }
    rc = 0;
out:
    free(buf);
    return rc;
}

static int daemon_request_valid(const request_t* req, response_t* res) {
    if (!req || !res) return 0;
    if (req->magic != LZ4_DAEMON_REQUEST_MAGIC ||
        req->version != LZ4_DAEMON_REQUEST_VERSION) {
        res->status = -1;
        snprintf(res->message, sizeof(res->message),
                 "daemon protocol mismatch: restart daemon/client");
        return 0;
    }
    return 1;
}

static int parse_env_int(const char* name, int* out_value) {
    const char* env = getenv(name);
    char* end = NULL;
    long v;
    if (!env || !*env) return 0;
    v = strtol(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    *out_value = (int)v;
    return 1;
}

static int choose_daemon_worker_count(cl_device_id device) {
    int env_workers = 0;
    if (parse_env_int("LZ4_DAEMON_WORKERS", &env_workers)) {
        return clamp_int(env_workers, 1, MAX_WORKERS_CAP);
    }

    cl_uint cu = 0;
    long cpu_online = sysconf(_SC_NPROCESSORS_ONLN);
    int cpu_budget = (cpu_online > 0) ? (int)cpu_online / 2 : 1;
    int cu_budget = 1;

    if (cpu_budget < 1) cpu_budget = 1;
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL) == CL_SUCCESS && cu > 0) {
        cu_budget = (int)((cu + 15) / 16);
        if (cu_budget < 1) cu_budget = 1;
    }

    return clamp_int((cpu_budget < cu_budget) ? cpu_budget : cu_budget, 1, MAX_WORKERS_CAP);
}

static int create_pidfile(void) {
    int fd = open("/tmp/lz4_gpu_daemon.pid", O_RDWR | O_CREAT, 0644);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX | LOCK_NB) < 0) {
        close(fd);
        return -1;
    }
    char buf[16];
    int n = snprintf(buf, sizeof(buf), "%d\n", getpid());
    ftruncate(fd, 0);
    write(fd, buf, n);
    g_state.pid_fd = fd;
    return 0;
}

static void remove_pidfile(void) {
    if (g_state.pid_fd >= 0) {
        flock(g_state.pid_fd, LOCK_UN);
        close(g_state.pid_fd);
        unlink("/tmp/lz4_gpu_daemon.pid");
    }
}

static void signal_handler(int sig) {
    (void)sig;
    g_state.running = 0;
}

static void cleanup_resources(void) {
    g_state.running = 0;
    if (g_state.server_sock >= 0) {
        close(g_state.server_sock);
        g_state.server_sock = -1;
    }
    unlink(SOCKET_PATH);

    for (int i = 0; i < g_state.active_workers; i++) {
        pthread_mutex_lock(&g_state.workers[i].lock);
        pthread_cond_signal(&g_state.workers[i].cond);
        pthread_mutex_unlock(&g_state.workers[i].lock);
    }
    for (int i = 0; i < g_state.active_workers; i++) {
        pthread_join(g_state.workers[i].thread, NULL);
        pthread_mutex_lock(&g_state.workers[i].lock);
        for (int m = 0; m < LZ4_DAEMON_KERNEL_MODES; m++) {
            if (g_state.workers[i].kernel_comp[m]) {
                clReleaseKernel(g_state.workers[i].kernel_comp[m]);
                g_state.workers[i].kernel_comp[m] = NULL;
            }
            if (g_state.workers[i].kernel_decomp[m]) {
                clReleaseKernel(g_state.workers[i].kernel_decomp[m]);
                g_state.workers[i].kernel_decomp[m] = NULL;
            }
        }
        if (g_state.workers[i].client_fd >= 0) {
            close(g_state.workers[i].client_fd);
            g_state.workers[i].client_fd = -1;
        }
        if (g_state.workers[i].queue) {
            clReleaseCommandQueue(g_state.workers[i].queue);
            g_state.workers[i].queue = NULL;
        }
        lz4_gpu_workspace_free(&g_state.workers[i].ws);
        pthread_mutex_unlock(&g_state.workers[i].lock);
        pthread_mutex_destroy(&g_state.workers[i].lock);
        pthread_cond_destroy(&g_state.workers[i].cond);
    }

    pthread_mutex_lock(&g_state.compile_lock);
    for (int m = 0; m < LZ4_DAEMON_KERNEL_MODES; m++) {
        if (g_state.program_comp[m]) {
            clReleaseProgram(g_state.program_comp[m]);
            g_state.program_comp[m] = NULL;
        }
    }
    pthread_mutex_unlock(&g_state.compile_lock);
    pthread_mutex_destroy(&g_state.compile_lock);

    if (g_state.context) {
        clReleaseContext(g_state.context);
        g_state.context = NULL;
    }
}

int init_resources(void) {
    uint64_t t1 = get_us();
    cl_int err;
    err = lz4_select_opencl_platform_device(&g_state.platform, &g_state.device);
    if (err != CL_SUCCESS || g_state.device == NULL || g_state.platform == NULL) return -1;

    {
        char pfname[256] = {0};
        char devname[256] = {0};
        cl_device_type devtype = 0;
        clGetPlatformInfo(g_state.platform, CL_PLATFORM_NAME, sizeof(pfname), pfname, NULL);
        clGetDeviceInfo(g_state.device, CL_DEVICE_NAME, sizeof(devname), devname, NULL);
        clGetDeviceInfo(g_state.device, CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
        fprintf(stderr, "[DAEMON OpenCL] Selected platform=%s, device=%s (type=%s)\n",
                pfname,
                devname,
                (devtype & CL_DEVICE_TYPE_GPU) ? "GPU" :
                (devtype & CL_DEVICE_TYPE_CPU) ? "CPU" :
                (devtype & CL_DEVICE_TYPE_DEFAULT) ? "DEFAULT" : "UNKNOWN");
    }

    g_state.context = clCreateContext(NULL, 1, &g_state.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return -1;
    g_ocl_init_us = get_us() - t1;

    pthread_mutex_init(&g_state.compile_lock, NULL);
    g_state.active_workers = choose_daemon_worker_count(g_state.device);
    g_state.server_sock = -1;

    for (int i = 0; i < g_state.active_workers; i++) {
        g_state.workers[i].id = i;
        g_state.workers[i].client_fd = -1;
        {
            cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
            g_state.workers[i].queue = clCreateCommandQueueWithProperties(g_state.context, g_state.device, props, &err);
        }
        lz4_gpu_workspace_init(&g_state.workers[i].ws);
        pthread_mutex_init(&g_state.workers[i].lock, NULL);
        pthread_cond_init(&g_state.workers[i].cond, NULL);
    }
    return 0;
}

void process_request(worker_res_t* w, request_t* req, response_t* res) {
    uint64_t t1 = get_us();
    timing_t t; memset(&t, 0, sizeof(t));
    int ret = -1;

    /* Per user request: in daemon mode, ocl_setup_us is always 0 for all requests */
    t.ocl_setup_us = 0;

    if (req->mode == mode_compress) {
        const int kernel_mode = daemon_kernel_mode_for_block((uint32_t)req->block_size);

        pthread_mutex_lock(&g_state.compile_lock);
        if (!g_state.program_comp[kernel_mode]) g_state.program_comp[kernel_mode] = lz4_load_program(g_state.context, g_state.device, 14, req->block_size);
        cl_program prog = g_state.program_comp[kernel_mode];
        if (!w->kernel_comp[kernel_mode] && prog) {
            cl_int err;
            w->kernel_comp[kernel_mode] = clCreateKernel(prog, "lz4_compress_block", &err);
        }
        cl_kernel kernel = w->kernel_comp[kernel_mode];
        pthread_mutex_unlock(&g_state.compile_lock);

        if (kernel) {
            ret = lz4_compress_core(g_state.context, w->queue, kernel, req->input_path, req->output_path,
                                  req->block_size, req->acceleration, &w->ws, &t, req->local_size, 0);
        }
    } else {
        const int kernel_mode = daemon_kernel_mode_for_block((uint32_t)req->block_size);
        pthread_mutex_lock(&g_state.compile_lock);
        if (!g_state.program_comp[kernel_mode]) g_state.program_comp[kernel_mode] = lz4_load_program(g_state.context, g_state.device, 14, req->block_size);
        cl_program prog = g_state.program_comp[kernel_mode];
        if (!w->kernel_decomp[kernel_mode] && prog) {
            cl_int err;
            w->kernel_decomp[kernel_mode] = clCreateKernel(prog, "lz4_decompress_blocks", &err);
        }
        cl_kernel kernel = w->kernel_decomp[kernel_mode];
        pthread_mutex_unlock(&g_state.compile_lock);

        if (kernel) {
            ret = lz4_decompress_core(g_state.context, w->queue, kernel, req->input_path, req->output_path, &w->ws, &t, req->local_size);
        }
    }

    uint64_t t2 = get_us();
    res->status = (ret == 0) ? 0 : -1;
    res->time_us = (unsigned long)(t2 - t1);
    res->timing = t;
    struct stat st;
    if (stat(req->output_path, &st) == 0) res->out_size = st.st_size;
}

void* worker_thread(void* arg) {
    worker_res_t* w = (worker_res_t*)arg;
    while (g_state.running) {
        pthread_mutex_lock(&w->lock);
        while (!w->has_work && g_state.running) {
            pthread_cond_wait(&w->cond, &w->lock);
        }
        if (!g_state.running) { pthread_mutex_unlock(&w->lock); break; }
        request_t req;
        int raw_in_fd = -1;
        int raw_out_fd = -1;
        char raw_in_path[64] = {0};
        char raw_out_path[64] = {0};
        if (daemon_read_full(w->client_fd, &req, sizeof(req)) == 0) {
            response_t res; memset(&res, 0, sizeof(res));
            if (!daemon_request_valid(&req, &res)) {
                (void)daemon_write_full(w->client_fd, &res, sizeof(res));
                close(w->client_fd); w->client_fd = -1; w->has_work = 0;
                pthread_mutex_unlock(&w->lock);
                continue;
            }
            if (req.flags & LZ4_DAEMON_FLAG_RAW_BUFFER) {
                uint64_t raw_len = (uint64_t)req.input_size;
                char* raw_buf = NULL;
                if (raw_len == 0 || raw_len > (uint64_t)SIZE_MAX) {
                    response_t res; memset(&res, 0, sizeof(res));
                    res.status = -1;
                    snprintf(res.message, sizeof(res.message), "invalid raw input size");
                    (void)daemon_write_full(w->client_fd, &res, sizeof(res));
                    close(w->client_fd); w->client_fd = -1; w->has_work = 0;
                    pthread_mutex_unlock(&w->lock);
                    continue;
                }
                raw_buf = (char*)malloc((size_t)raw_len);
                if (!raw_buf || daemon_read_full(w->client_fd, raw_buf, (size_t)raw_len) != 0) {
                    free(raw_buf);
                    response_t res; memset(&res, 0, sizeof(res));
                    res.status = -1;
                    snprintf(res.message, sizeof(res.message), "failed to receive raw payload");
                    (void)daemon_write_full(w->client_fd, &res, sizeof(res));
                    close(w->client_fd); w->client_fd = -1; w->has_work = 0;
                    pthread_mutex_unlock(&w->lock);
                    continue;
                }
                raw_in_fd = daemon_memfd_create("lz4_gpu_raw_in");
                raw_out_fd = daemon_memfd_create("lz4_gpu_raw_out");
                if (raw_in_fd < 0 || raw_out_fd < 0 ||
                    daemon_write_fd_full(raw_in_fd, raw_buf, (size_t)raw_len) != 0 ||
                    lseek(raw_in_fd, 0, SEEK_SET) < 0 ||
                    daemon_fd_path(raw_in_fd, raw_in_path, sizeof(raw_in_path)) != 0 ||
                    daemon_fd_path(raw_out_fd, raw_out_path, sizeof(raw_out_path)) != 0) {
                    free(raw_buf);
                    if (raw_in_fd >= 0) close(raw_in_fd);
                    if (raw_out_fd >= 0) close(raw_out_fd);
                    response_t res; memset(&res, 0, sizeof(res));
                    res.status = -1;
                    snprintf(res.message, sizeof(res.message), "failed to materialize raw payload");
                    (void)daemon_write_full(w->client_fd, &res, sizeof(res));
                    close(w->client_fd); w->client_fd = -1; w->has_work = 0;
                    pthread_mutex_unlock(&w->lock);
                    continue;
                }
                free(raw_buf);
                memset(req.input_path, 0, sizeof(req.input_path));
                memset(req.output_path, 0, sizeof(req.output_path));
                strncpy(req.input_path, raw_in_path, sizeof(req.input_path) - 1);
                strncpy(req.output_path, raw_out_path, sizeof(req.output_path) - 1);
            }
            process_request(w, &req, &res);
            if (daemon_write_full(w->client_fd, &res, sizeof(res)) != 0) {
                if (raw_in_fd >= 0) close(raw_in_fd);
                if (raw_out_fd >= 0) close(raw_out_fd);
                close(w->client_fd); w->client_fd = -1; w->has_work = 0;
                pthread_mutex_unlock(&w->lock);
                continue;
            }
            if ((req.flags & LZ4_DAEMON_FLAG_RAW_BUFFER) && res.status == 0) {
                if (daemon_send_fd_payload(w->client_fd, raw_out_fd) != 0) {
                    fprintf(stderr, "failed to send raw daemon payload\n");
                }
            }
            if (raw_in_fd >= 0) close(raw_in_fd);
            if (raw_out_fd >= 0) close(raw_out_fd);
        }
        close(w->client_fd); w->client_fd = -1; w->has_work = 0;
        pthread_mutex_unlock(&w->lock);
    }
    return NULL;
}

int run_daemon() {
    if (create_pidfile() < 0) { fprintf(stderr, "Daemon already running\n"); return 1; }
    signal(SIGTERM, signal_handler); signal(SIGINT, signal_handler);
    if (init_resources() < 0) return 1;
    g_state.server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr; memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX; strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    unlink(SOCKET_PATH);
    if (bind(g_state.server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) return 1;
    listen(g_state.server_sock, 5);
    g_state.running = 1;
    for (int i = 0; i < g_state.active_workers; i++) {
        pthread_create(&g_state.workers[i].thread, NULL, worker_thread, &g_state.workers[i]);
    }
    printf("LZ4 GPU Daemon started. workers=%d, listening on %s\n", g_state.active_workers, SOCKET_PATH); fflush(stdout);
    while (g_state.running) {
        int client = accept(g_state.server_sock, NULL, NULL);
        if (client < 0) { if (errno == EINTR) continue; break; }
        int assigned = 0;
        while (g_state.running && !assigned) {
            for (int i = 0; i < g_state.active_workers; i++) {
                if (pthread_mutex_trylock(&g_state.workers[i].lock) == 0) {
                    if (!g_state.workers[i].has_work) {
                        g_state.workers[i].client_fd = client;
                        g_state.workers[i].has_work = 1;
                        pthread_cond_signal(&g_state.workers[i].cond);
                        assigned = 1;
                    }
                    pthread_mutex_unlock(&g_state.workers[i].lock);
                    if (assigned) break;
                }
            }
            if (!assigned) {
                struct timespec ts = {0, 1000000};
                nanosleep(&ts, NULL);
            }
        }
        if (!assigned) close(client);
    }
    cleanup_resources();
    remove_pidfile();
    return 0;
}
