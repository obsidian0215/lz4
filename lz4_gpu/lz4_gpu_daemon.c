#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <CL/cl.h>
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_core.h"
#include "lz4_gpu_utils.h"

#define MAX_WORKERS 4
#define MIN_HASH_LOG 10
#define MAX_HASH_LOG 16

typedef struct {
    int id;
    cl_command_queue queue;
    cl_kernel kernel_comp[MAX_HASH_LOG + 1];
    cl_kernel kernel_decomp;
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
    cl_program program_comp[MAX_HASH_LOG + 1];
    pthread_mutex_t compile_lock;
    worker_res_t workers[MAX_WORKERS];
    int server_sock;
    volatile int running;
    int pid_fd;
} g_state;

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
    g_state.running = 0;
}

static char* read_file_bin(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = malloc(s); if (!buf) { fclose(f); return NULL; }
    if (fread(buf,1,s,f) != (size_t)s) { free(buf); fclose(f); return NULL; }
    if (out_len) *out_len = (size_t)s;
    fclose(f);
    return buf;
}

cl_program load_program_with_hash(cl_context context, cl_device_id device, int hash_log) {
    uint64_t t1 = get_us();
    char bin_name[128];
    snprintf(bin_name, sizeof(bin_name), "/root/lz4/lz4_gpu/lz4_gpu_%d.clbin", hash_log);
    size_t sz = 0;
    char* bin = read_file_bin(bin_name, &sz);
    if (bin) {
        cl_int status, err;
        cl_program prog = clCreateProgramWithBinary(context, 1, &device, &sz, (const unsigned char**)&bin, &status, &err);
        free(bin);
        if (err == CL_SUCCESS) {
            clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
            return prog;
        }
    }
    // Fallback: compile from source
    char flags[128];
    snprintf(flags, sizeof(flags), "-I. -DLZ4_HASHLOG=%d", hash_log);
    FILE* f = fopen("/root/lz4/lz4_gpu/lz4_gpu.cl", "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); size_t s_sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* src = malloc(s_sz + 1); fread(src, 1, s_sz, f); src[s_sz] = 0; fclose(f);
    cl_int err;
    cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&src, &s_sz, &err);
    free(src);
    clBuildProgram(prog, 1, &device, flags, NULL, NULL);
    g_kernel_load_us += (get_us() - t1);
    return prog;
}

int init_resources(void) {
    uint64_t t1 = get_us();
    cl_int err;
    err = clGetPlatformIDs(1, &g_state.platform, NULL);
    if (err != CL_SUCCESS) return -1;
    err = clGetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_GPU, 1, &g_state.device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_DEFAULT, 1, &g_state.device, NULL);
    }
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_ALL, 1, &g_state.device, NULL);
    }
    if (err != CL_SUCCESS) return -1;
    g_state.context = clCreateContext(NULL, 1, &g_state.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return -1;
    g_ocl_init_us = get_us() - t1;

    pthread_mutex_init(&g_state.compile_lock, NULL);

    for (int i = 0; i < MAX_WORKERS; i++) {
        g_state.workers[i].id = i;
        g_state.workers[i].queue = clCreateCommandQueue(g_state.context, g_state.device, 0, &err);
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
        int h_log = req->hash_log;
        if (h_log < MIN_HASH_LOG) h_log = 14;
        if (h_log > MAX_HASH_LOG) h_log = MAX_HASH_LOG;

        pthread_mutex_lock(&g_state.compile_lock);
        if (!g_state.program_comp[h_log]) g_state.program_comp[h_log] = load_program_with_hash(g_state.context, g_state.device, h_log);
        cl_program prog = g_state.program_comp[h_log];
        if (!w->kernel_comp[h_log] && prog) {
            cl_int err;
            w->kernel_comp[h_log] = clCreateKernel(prog, "lz4_compress_block", &err);
        }
        cl_kernel kernel = w->kernel_comp[h_log];
        pthread_mutex_unlock(&g_state.compile_lock);

        if (kernel) {
            ret = lz4_compress_core(g_state.context, w->queue, kernel, req->input_path, req->output_path,
                                  req->block_size, req->acceleration, &w->ws, &t, req->local_size, h_log);
        }
    } else {
        int h_log = 14;
        pthread_mutex_lock(&g_state.compile_lock);
        if (!g_state.program_comp[h_log]) g_state.program_comp[h_log] = load_program_with_hash(g_state.context, g_state.device, h_log);
        cl_program prog = g_state.program_comp[h_log];
        if (!w->kernel_decomp && prog) {
            cl_int err;
            w->kernel_decomp = clCreateKernel(prog, "lz4_decompress_blocks", &err);
        }
        cl_kernel kernel = w->kernel_decomp;
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
        if (read(w->client_fd, &req, sizeof(req)) == sizeof(req)) {
            response_t res; memset(&res, 0, sizeof(res));
            process_request(w, &req, &res);
            write(w->client_fd, &res, sizeof(res));
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
    for (int i = 0; i < MAX_WORKERS; i++) pthread_create(&g_state.workers[i].thread, NULL, worker_thread, &g_state.workers[i]);
    printf("LZ4 GPU Daemon started. Listening on %s\n", SOCKET_PATH); fflush(stdout);
    while (g_state.running) {
        int client = accept(g_state.server_sock, NULL, NULL);
        if (client < 0) { if (errno == EINTR) continue; break; }
        int found = 0;
        for (int i = 0; i < MAX_WORKERS; i++) {
            if (pthread_mutex_trylock(&g_state.workers[i].lock) == 0) {
                if (!g_state.workers[i].has_work) {
                    g_state.workers[i].client_fd = client;
                    g_state.workers[i].has_work = 1;
                    pthread_cond_signal(&g_state.workers[i].cond);
                    found = 1;
                }
                pthread_mutex_unlock(&g_state.workers[i].lock);
                if (found) break;
            }
        }
        if (!found) close(client);
    }
    remove_pidfile();
    return 0;
}
