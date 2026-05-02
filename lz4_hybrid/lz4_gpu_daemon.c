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
#include <time.h>
#include <limits.h>
#include <CL/cl.h>
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_core.h"
#include "lz4_gpu_utils.h"

int lz4_daemon_split_file_request(int mode,
                                  const char* input_path,
                                  const char* output_path,
                                  int block_size,
                                  int acceleration,
                                  int local_size,
                                  uint32_t cpu_share_pct,
                                  uint32_t cpu_threads,
                                  uint32_t adaptive,
                                  unsigned long* elapsed_us);

#define MAX_WORKERS_CAP 16
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
    worker_res_t workers[MAX_WORKERS_CAP];
    int active_workers;
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

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
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
        cu_budget = (int)((cu + 7) / 8); /* about 1 daemon worker per 8 CUs */
        if (cu_budget < 1) cu_budget = 1;
    }

    return clamp_int((cpu_budget < cu_budget) ? cpu_budget : cu_budget, 1, MAX_WORKERS_CAP);
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

static int path_join2(char* out, size_t out_len, const char* a, const char* b) {
    size_t la, lb;
    if (!out || !a || !b || out_len == 0) return -1;
    la = strlen(a); lb = strlen(b);
    if (la + 1 + lb + 1 > out_len) return -1;
    memcpy(out, a, la);
    out[la] = '/';
    memcpy(out + la + 1, b, lb);
    out[la + 1 + lb] = '\0';
    return 0;
}

static int path_join3(char* out, size_t out_len, const char* a, const char* b, const char* c) {
    char tmp[PATH_MAX];
    if (path_join2(tmp, sizeof(tmp), a, b) != 0) return -1;
    return path_join2(out, out_len, tmp, c);
}

static int get_exe_dir(char* out, size_t out_len) {
    char exe[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) return -1;
    exe[n] = '\0';
    char* slash = strrchr(exe, '/');
    if (!slash) return -1;
    *slash = '\0';
    if (strlen(exe) + 1 > out_len) return -1;
    strcpy(out, exe);
    return 0;
}

static int dirname_of_path(const char* path, char* out, size_t out_len) {
    const char* slash;
    size_t len;
    if (!path || !out || out_len == 0) return -1;
    slash = strrchr(path, '/');
    if (!slash) return -1;
    len = (size_t)(slash - path);
    if (len + 1 > out_len) return -1;
    memcpy(out, path, len);
    out[len] = '\0';
    return 0;
}

static int resolve_daemon_file(const char* env_name, const char* filename, char* out, size_t out_len) {
    char cwd[PATH_MAX];
    char exe_dir[PATH_MAX];
    const char* env = getenv(env_name);
    if (env && *env && access(env, R_OK) == 0) {
        if (strlen(env) + 1 > out_len) return -1;
        strcpy(out, env);
        return 0;
    }

    if (getcwd(cwd, sizeof(cwd)) && path_join2(out, out_len, cwd, filename) == 0 && access(out, R_OK) == 0) {
        return 0;
    }

    if (get_exe_dir(exe_dir, sizeof(exe_dir)) == 0) {
        if (path_join2(out, out_len, exe_dir, filename) == 0 && access(out, R_OK) == 0) {
            return 0;
        }
        if (path_join3(out, out_len, exe_dir, "../lz4_gpu", filename) == 0 && access(out, R_OK) == 0) {
            return 0;
        }
    }

    if (access(filename, R_OK) == 0) {
        if (strlen(filename) + 1 > out_len) return -1;
        strcpy(out, filename);
        return 0;
    }

    return -1;
}

cl_program load_program(cl_context context, cl_device_id device) {
    const int hash_log = 14;
    uint64_t t1 = get_us();
    char bin_name[128];
    char bin_path[PATH_MAX];

    snprintf(bin_name, sizeof(bin_name), "lz4_gpu_%d.clbin", hash_log);
    if (resolve_daemon_file("LZ4_GPU_DAEMON_CLBIN", bin_name, bin_path, sizeof(bin_path)) == 0) {
        size_t sz = 0;
        char* bin = read_file_bin(bin_path, &sz);
        if (bin) {
            cl_int status, err;
            cl_program prog = clCreateProgramWithBinary(context, 1, &device, &sz, (const unsigned char**)&bin, &status, &err);
            free(bin);
            if (err == CL_SUCCESS && status == CL_SUCCESS) {
                if (clBuildProgram(prog, 1, &device, NULL, NULL, NULL) == CL_SUCCESS) {
                    return prog;
                }
                clReleaseProgram(prog);
            }
        }
    }

    {
        char src_path[PATH_MAX];
        char include_dir[PATH_MAX];
        char flags[PATH_MAX + 64];
        FILE* f;
        size_t s_sz;
        char* src;
        cl_int err;

        if (resolve_daemon_file("LZ4_GPU_DAEMON_CL", "lz4_gpu.cl", src_path, sizeof(src_path)) != 0) {
            return NULL;
        }

        if (dirname_of_path(src_path, include_dir, sizeof(include_dir)) != 0) {
            strcpy(include_dir, ".");
        }

        snprintf(flags, sizeof(flags), "-I. -I%s -DLZ4_HASHLOG=%d", include_dir, hash_log);
        f = fopen(src_path, "r");
        if (!f) return NULL;
        fseek(f, 0, SEEK_END); s_sz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
        src = malloc(s_sz + 1);
        if (!src) { fclose(f); return NULL; }
        fread(src, 1, s_sz, f); src[s_sz] = 0; fclose(f);

        cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&src, &s_sz, &err);
        free(src);
        if (err != CL_SUCCESS || !prog) return NULL;
        if (clBuildProgram(prog, 1, &device, flags, NULL, NULL) != CL_SUCCESS) {
            clReleaseProgram(prog);
            return NULL;
        }
        g_kernel_load_us += (get_us() - t1);
        return prog;
    }
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
    g_state.active_workers = choose_daemon_worker_count(g_state.device);

    for (int i = 0; i < g_state.active_workers; i++) {
        g_state.workers[i].id = i;
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
    struct stat in_st;
    size_t input_size = 0;
    if (stat(req->input_path, &in_st) == 0 && in_st.st_size > 0) input_size = (size_t)in_st.st_size;

    /* Per user request: in daemon mode, ocl_setup_us is always 0 for all requests */
    t.ocl_setup_us = 0;

    if (req->cpu_share_pct > 0 || req->adaptive) {
        unsigned long elapsed_us = 0;
        uint32_t cpu_share = req->adaptive ? 50U : req->cpu_share_pct;
        if (cpu_share > 100U) cpu_share = 100U;
        printf("[LZ4-DAEMON][SPLIT] mode=%s cpu_share=%u%% cpu_threads=%u adaptive=%u\n",
               req->mode == mode_decompress ? "decompress" : "compress",
               cpu_share, req->cpu_threads, req->adaptive);
        ret = lz4_daemon_split_file_request(req->mode,
                                            req->input_path,
                                            req->output_path,
                                            req->block_size,
                                            req->acceleration,
                                            req->local_size,
                                            cpu_share,
                                            req->cpu_threads,
                                            req->adaptive,
                                            &elapsed_us);
        res->status = (ret == 0) ? 0 : -1;
        res->time_us = elapsed_us;
        res->timing.ocl_setup_us = 0;
        res->timing.in_size = input_size;
        res->timing.algo_config = req->acceleration;
        res->timing.blk_size_bytes = (size_t)req->block_size;
        if (stat(req->output_path, &in_st) == 0 && in_st.st_size > 0) {
            res->out_size = (size_t)in_st.st_size;
            res->timing.out_size = res->out_size;
        }
        snprintf(res->message, sizeof(res->message), ret == 0 ? "Split success (daemon request)" : "Split request failed");
        return;
    }

    if (req->mode == mode_compress) {
        const int h_log = 14;

        pthread_mutex_lock(&g_state.compile_lock);
        if (!g_state.program_comp[h_log]) g_state.program_comp[h_log] = load_program(g_state.context, g_state.device);
        cl_program prog = g_state.program_comp[h_log];
        if (!w->kernel_comp[h_log] && prog) {
            cl_int err;
            w->kernel_comp[h_log] = clCreateKernel(prog, "lz4_compress_block", &err);
        }
        cl_kernel kernel = w->kernel_comp[h_log];
        pthread_mutex_unlock(&g_state.compile_lock);

        if (kernel) {
            ret = lz4_compress_core(g_state.context, w->queue, kernel, req->input_path, req->output_path,
                                  req->block_size, req->acceleration, &w->ws, &t, req->local_size, 0);
        }
    } else {
        int h_log = 14;
        pthread_mutex_lock(&g_state.compile_lock);
        if (!g_state.program_comp[h_log]) g_state.program_comp[h_log] = load_program(g_state.context, g_state.device);
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
    remove_pidfile();
    return 0;
}
