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
#include <math.h>
#include <sys/syscall.h>
#include <CL/cl.h>
#include "lz4_gpu_protocol.h"
#include "lz4_gpu_core.h"
#include "lz4_gpu_utils.h"

extern long syscall(long number, ...);

#define MAX_WORKERS_CAP 16
#define LZ4_DAEMON_KERNEL_MODES 2
#define LZ4_HASHLOG_MIN 11
#define LZ4_HASHLOG_MAX 15
#define LZ4_HASHLOG_COUNT (LZ4_HASHLOG_MAX - LZ4_HASHLOG_MIN + 1)
#define LZ4_DAEMON_PROGRAM_SLOTS (LZ4_DAEMON_KERNEL_MODES * LZ4_HASHLOG_COUNT)

enum {
    LZ4_DAEMON_MODE_CLEAR16 = 0,
    LZ4_DAEMON_MODE_EPOCH32 = 1
};

typedef struct {
    int id;
    cl_command_queue queue;
    cl_kernel kernel_comp[LZ4_DAEMON_PROGRAM_SLOTS];
    cl_kernel kernel_decomp[LZ4_DAEMON_PROGRAM_SLOTS];
    cl_kernel kernel_tp_build[LZ4_HASHLOG_COUNT]; /* I3: two-phase, keyed by hash_log slot */
    cl_kernel kernel_tp_scan[LZ4_HASHLOG_COUNT];
    cl_command_queue tp_queue;   /* I3: profiling-enabled queue for pure-kernel tp timing */
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
    cl_program program_comp[LZ4_DAEMON_PROGRAM_SLOTS];
    pthread_mutex_t compile_lock;
    worker_res_t workers[MAX_WORKERS_CAP];
    int active_workers;
    int server_sock;
    volatile int running;
    int pid_fd;
} g_state;

/* ============================================================================
 * I3: two-phase compression served by the daemon + online W_sat learning.
 *   - A dedicated EPOCH32 tp program (DICT_CLEAR=0, ENTRY_BITS=32) cached
 *     global-under-lock, keyed by hash_log; kernels are per-worker.
 *   - N is chosen either from the on-disk profile (I3a, LZ4_TP_LEARN=0) or by
 *     an online explore/exploit learner that converges W_sat from live traffic
 *     (I3b, default). Both live in one binary; the learner warm-starts from the
 *     profile at init and persists W_sat_est back to it whenever it changes.
 * ==========================================================================*/
static cl_program g_tp_program[LZ4_HASHLOG_COUNT];   /* guarded by g_state.compile_lock */

/* Per-(nblk,N) throughput table. best_thr[occ] (occ=nblk*N) is a projection of this;
 * we key by nblk so the refit can compare N at a FIXED occupancy-of-data (nblk), which
 * is what actually decides the best N. A pure per-occ table conflates file-size with
 * segmentation (e.g. on a CPU, more segments never help, but bigger files raise
 * throughput at N=1, so a per-occ 0.8*peak crossing would wrongly infer a large W_sat
 * and pick N>1 for small inputs). The N index maps 0/1/2/3 -> N=1/2/4/8. */
static const int TP_NVAL[4] = {1, 2, 4, 8};
static int tp_nidx(int N) { return N == 1 ? 0 : N == 2 ? 1 : N == 4 ? 2 : 3; }

typedef struct { uint32_t nblk; double thr[4]; int seen[4]; } tp_row_t;
#define LZ4_TP_LEARN_CAP 256
static struct {
    pthread_mutex_t lock;
    tp_row_t row[LZ4_TP_LEARN_CAP];  /* one row per distinct nblk */
    int      n_row;
    int      n_cell;      /* distinct (nblk,N) cells sampled (>= "distinct occupancies") */
    double   peak_thr;    /* best throughput ever observed (MB/s) */
    int      W_sat_est;   /* 0 = unknown */
    int      n_obs;       /* total two-phase observations */
    char     devname[256];
} g_tp_learn;

static int daemon_kernel_mode_for_block(uint32_t block_size) {
    return (block_size <= 64U * 1024U) ? LZ4_DAEMON_MODE_CLEAR16 : LZ4_DAEMON_MODE_EPOCH32;
}

static int daemon_hash_log_from_request(const request_t* req) {
    int hash_log = req && req->hash_log ? req->hash_log : 14;
    if (hash_log < LZ4_HASHLOG_MIN) hash_log = LZ4_HASHLOG_MIN;
    if (hash_log > LZ4_HASHLOG_MAX) hash_log = LZ4_HASHLOG_MAX;
    return hash_log;
}

static int daemon_program_slot(uint32_t block_size, int hash_log) {
    int mode = daemon_kernel_mode_for_block(block_size);
    if (hash_log < LZ4_HASHLOG_MIN) hash_log = LZ4_HASHLOG_MIN;
    if (hash_log > LZ4_HASHLOG_MAX) hash_log = LZ4_HASHLOG_MAX;
    return (hash_log - LZ4_HASHLOG_MIN) * LZ4_DAEMON_KERNEL_MODES + mode;
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

/* ---- I3: two-phase codec + online W_sat learning implementation ---- */
#define LZ4TP_MAGIC "LZ4TP1\0\0"
static int g_tp_learn_enabled = 1;   /* LZ4_TP_LEARN=0 falls back to profile-based N (I3a) */

/* profile I/O (device-keyed W_sat), mirrors lz4_gpu.c */
static const char* tp_profile_path(void) {
    const char* p = getenv("LZ4TP_PROFILE");
    return (p && *p) ? p : "lz4tp.profile";
}

static void tp_device_name(char* buf, size_t n) {
    buf[0] = 0;
    if (g_state.device) clGetDeviceInfo(g_state.device, CL_DEVICE_NAME, n, buf, NULL);
}

static int tp_profile_lookup(const char* devname) {
    FILE* f = fopen(tp_profile_path(), "r");
    if (!f) return 0;
    char line[512]; int w = 0;
    while (fgets(line, sizeof(line), f)) {
        char* tab = strrchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        if (strcmp(line, devname) == 0) { w = atoi(tab + 1); break; }
    }
    fclose(f);
    return w;
}

static void tp_profile_store(const char* devname, int W_sat) {
    /* rewrite file: keep other devices, replace/append this one */
    const char* path = tp_profile_path();
    char (*names)[256] = NULL; int* ws = NULL; int cnt = 0, cap = 0, found = 0;
    FILE* f = fopen(path, "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            char* nl = strchr(line, '\n'); if (nl) *nl = 0;
            char* tab = strrchr(line, '\t'); if (!tab) continue; *tab = 0;
            if (cnt == cap) { cap = cap ? cap*2 : 8;
                names = realloc(names, cap*sizeof(*names)); ws = realloc(ws, cap*sizeof(int)); }
            strncpy(names[cnt], line, 255); names[cnt][255]=0; ws[cnt] = atoi(tab+1); cnt++;
        }
        fclose(f);
    }
    f = fopen(path, "w");
    if (!f) { free(names); free(ws); return; }
    for (int i = 0; i < cnt; i++) {
        if (strcmp(names[i], devname) == 0) { fprintf(f, "%s\t%d\n", devname, W_sat); found = 1; }
        else fprintf(f, "%s\t%d\n", names[i], ws[i]);
    }
    if (!found) fprintf(f, "%s\t%d\n", devname, W_sat);
    fclose(f); free(names); free(ws);
}

/* EXPLOIT rule: largest N in {1,2,4,8} with nblk*N <= 0.75*W_sat (== lz4_tp_pick_N) */
static int tp_pick_N(size_t nblk, int W_sat) {
    int cand[4] = {1,2,4,8}; int best = 1;
    double thresh = 0.75 * (double)W_sat;
    for (int i = 0; i < 4; i++) if ((double)nblk * cand[i] <= thresh) best = cand[i];
    return best;
}

/* learning-table helpers -- caller holds g_tp_learn.lock */
static tp_row_t* tp_learn_row(uint32_t nblk) {
    for (int i = 0; i < g_tp_learn.n_row; i++)
        if (g_tp_learn.row[i].nblk == nblk) return &g_tp_learn.row[i];
    if (g_tp_learn.n_row >= LZ4_TP_LEARN_CAP) return NULL;
    tp_row_t* r = &g_tp_learn.row[g_tp_learn.n_row++];
    r->nblk = nblk;
    for (int k = 0; k < 4; k++) { r->thr[k] = 0.0; r->seen[k] = 0; }
    return r;
}

/* EXPLORE: for THIS file (nblk), sample the least-sampled N (tie -> larger N). This
 * spreads observations across N at each occupancy-of-data, which is what the refit
 * needs, and does not depend on the request-arrival order. */
static int tp_learn_pick_explore_N(uint32_t nblk) {
    tp_row_t* r = tp_learn_row(nblk);
    if (!r) return 1;
    int bestN = 1, bestCount = INT_MAX;
    for (int k = 0; k < 4; k++)
        if (r->seen[k] <= bestCount) { bestCount = r->seen[k]; bestN = TP_NVAL[k]; }
    return bestN;
}

/* whether the table has enough signal to fit W_sat: >= 3 distinct nblk, and at least
 * one nblk probed with both N=1 and N=8 (so the full N-scaling is visible), and enough
 * total cells that we've actually "seen the ramp, not just the plateau". */
static int tp_learn_ready(void) {
    if (g_tp_learn.n_row < 3 || g_tp_learn.n_cell < 6) return 0;
    for (int i = 0; i < g_tp_learn.n_row; i++)
        if (g_tp_learn.row[i].seen[0] && g_tp_learn.row[i].seen[3]) return 1;
    return 0;
}

/* refit W_sat by DECISION QUALITY (same criterion as lz4_calibrate_device): pick the W
 * whose pick_N(nblk,W) maximizes geomean of achieved/oracle throughput over sampled
 * rows. Recovers "N=1 always" (small W) on CPUs and the saturation occupancy on GPUs. */
static int tp_learn_fit_W(void) {
    int bestW = 1; double bestScore = -1e18; int wlo = 1, whi = 1;
    for (int W = 1; W <= 8192; W++) {
        double s = 0.0; int used = 0;
        for (int i = 0; i < g_tp_learn.n_row; i++) {
            tp_row_t* r = &g_tp_learn.row[i];
            if (!r->seen[0]) continue;                     /* need an N=1 baseline for this row */
            double orc = 0.0;
            for (int k = 0; k < 4; k++) if (r->seen[k] && r->thr[k] > orc) orc = r->thr[k];
            if (orc <= 0.0) continue;
            int pick = 1;   /* fit threshold is W directly (no 0.75) -- pick_N applies the
                             * 0.75 margin at deployment, exactly like lz4_calibrate_device */
            for (int k = 0; k < 4; k++)
                if ((double)r->nblk * TP_NVAL[k] <= (double)W && r->seen[k]) pick = TP_NVAL[k];
            int idx = tp_nidx(pick);
            double got = (r->seen[idx] && r->thr[idx] > 0.0) ? r->thr[idx] : r->thr[0];
            if (got <= 0.0) continue;
            s += log(got / orc); used++;
        }
        if (used == 0) continue;
        s /= used;
        if (s > bestScore + 1e-9) { bestScore = s; wlo = whi = W; }
        else if (s > bestScore - 1e-9) { whi = W; }
    }
    bestW = (wlo + whi) / 2;   /* midpoint of the flat optimum plateau */
    return bestW;
}

/* record one observation, refit W_sat_est, persist on change */
static void tp_learn_record(size_t nblk, int N, double thr, int* W_sat_est_out) {
    pthread_mutex_lock(&g_tp_learn.lock);
    tp_row_t* r = tp_learn_row((uint32_t)nblk);
    if (r) {
        int k = tp_nidx(N);
        if (r->seen[k] == 0) g_tp_learn.n_cell++;
        if (thr > r->thr[k]) r->thr[k] = thr;
        r->seen[k]++;
    }
    if (thr > g_tp_learn.peak_thr) g_tp_learn.peak_thr = thr;
    g_tp_learn.n_obs++;

    if (tp_learn_ready()) {
        int W = tp_learn_fit_W();
        if (W > 0 && W != g_tp_learn.W_sat_est) {
            g_tp_learn.W_sat_est = W;
            tp_profile_store(g_tp_learn.devname, g_tp_learn.W_sat_est);
        }
    }
    if (W_sat_est_out) *W_sat_est_out = g_tp_learn.W_sat_est;
    pthread_mutex_unlock(&g_tp_learn.lock);
}

/* choose N for this request (occ_out = nblk*N) */
static int tp_select_N(size_t nblk, uint32_t* occ_out) {
    int N = 1;
    if (!g_tp_learn_enabled) {
        int W = tp_profile_lookup(g_tp_learn.devname);   /* I3a: straight from profile */
        N = (W > 0) ? tp_pick_N(nblk, W) : 1;
    } else {
        pthread_mutex_lock(&g_tp_learn.lock);
        int W = g_tp_learn.W_sat_est;
        int n_obs = g_tp_learn.n_obs;
        /* explore while not yet confident, else ~1 in 7 (7 is coprime with typical
         * request-sequence periods, so exploration is not aliased to one file size) */
        int explore = (W <= 0) || !tp_learn_ready() || (n_obs % 7 == 0);
        if (!explore && W > 0) N = tp_pick_N(nblk, W);
        else N = tp_learn_pick_explore_N((uint32_t)nblk);
        pthread_mutex_unlock(&g_tp_learn.lock);
    }
    if (N < 1) N = 1; if (N > 8) N = 8;
    if (occ_out) *occ_out = (uint32_t)(nblk * (size_t)N);
    return N;
}

/* pure GPU kernel time (us) from a profiling event, matches bench's event_elapsed_us */
static double tp_event_us(cl_event ev) {
    cl_ulong st = 0, en = 0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) != CL_SUCCESS) return 0.0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, NULL) != CL_SUCCESS) return 0.0;
    return (en >= st) ? (double)(en - st) / 1000.0 : 0.0;
}

/* two-phase codec: a copy of lz4_tp_compress_to_file that uses the passed-in warm
 * ctx/queue and cached epoch32 kbuild/kscan; per-request cl_mem, no ctx/queue release.
 * queue must be profiling-enabled: kernel time is measured via CL events (pure GPU
 * time, matching lz4_tp_measure_one) so throughput reflects occupancy, not dispatch. */
static int lz4_tp_compress_daemon(cl_context ctx, cl_command_queue queue, cl_device_id dev,
                                  cl_kernel kbuild, cl_kernel kscan,
                                  const char* input_path, const char* output_path,
                                  int N, int hash_log, int block_size, double* kernel_us_out) {
    (void)dev;
    if (kernel_us_out) *kernel_us_out = 0.0;
    struct stat st;
    if (!input_path || stat(input_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "tp-daemon: invalid input\n"); return 1;
    }
    if (block_size <= 0) block_size = 64 * 1024;
    if (N < 1) N = 1; if (N > 64) N = 64;
    if (hash_log < LZ4_HASHLOG_MIN) hash_log = LZ4_HASHLOG_MIN;
    if (hash_log > LZ4_HASHLOG_MAX) hash_log = LZ4_HASHLOG_MAX;

    size_t orig_size = (size_t)st.st_size;
    unsigned char* input_ref = (unsigned char*)malloc(orig_size);
    unsigned long rd_us = 0;
    if (!input_ref || lz4_read_file_to_buf(input_path, input_ref, orig_size, &rd_us) != 0) {
        free(input_ref); fprintf(stderr, "tp-daemon: read failed\n"); return 1;
    }

    size_t dict_entries = (size_t)1u << hash_log;
    size_t nblk = (orig_size + (size_t)block_size - 1) / (size_t)block_size;
    int nk = N - 1;
    int segLenMax = (block_size + N - 1) / N;
    int segMaxOut = segLenMax + segLenMax / 255 + 64;
    size_t meta_cnt = nblk * (size_t)N;
    size_t prefix_bytes = (size_t)nblk * (size_t)nk * dict_entries * sizeof(cl_uint);
    size_t own_bytes    = (size_t)nblk * (size_t)N  * dict_entries * sizeof(cl_uint);
    size_t out_bytes    = (size_t)nblk * (size_t)N  * (size_t)segMaxOut;

    cl_int err = CL_SUCCESS;
    cl_mem d_input=NULL,d_prefix=NULL,d_own=NULL,d_out=NULL,d_sizes=NULL;
    cl_uint* sizes = NULL; unsigned char* padded = NULL; unsigned char* payload = NULL;
    size_t payload_total = 0;
    int status = 1;

    d_input  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, orig_size, NULL, &err);
    d_prefix = clCreateBuffer(ctx, CL_MEM_READ_WRITE, prefix_bytes ? prefix_bytes : 4, NULL, &err);
    d_own    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, own_bytes, NULL, &err);
    d_out    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    d_sizes  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, meta_cnt * sizeof(cl_uint), NULL, &err);
    if (!d_input || !d_prefix || !d_own || !d_out || !d_sizes) {
        fprintf(stderr, "tp-daemon: buffer alloc failed (%d)\n", err); goto done;
    }
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, orig_size, input_ref, 0, NULL, NULL);
    { cl_uint zero = 0;
      if (prefix_bytes) clEnqueueFillBuffer(queue, d_prefix, &zero, sizeof(zero), 0, prefix_bytes, 0, NULL, NULL);
      clEnqueueFillBuffer(queue, d_own, &zero, sizeof(zero), 0, own_bytes, 0, NULL, NULL);
      clFinish(queue);
    }

    {
    cl_int i_nblk = (cl_int)nblk, i_insize = (cl_int)orig_size, i_blk = block_size,
           i_segmax = segMaxOut, i_N = N;
    cl_uint epoch = 1;
    size_t lws1 = 1;
    clSetKernelArg(kbuild, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kbuild, 1, sizeof(cl_mem), &d_prefix);
    clSetKernelArg(kbuild, 2, sizeof(cl_int), &i_nblk);
    clSetKernelArg(kbuild, 3, sizeof(cl_int), &i_insize);
    clSetKernelArg(kbuild, 4, sizeof(cl_int), &i_blk);
    clSetKernelArg(kbuild, 5, sizeof(cl_int), &i_N);
    clSetKernelArg(kbuild, 6, sizeof(cl_uint), &epoch);
    clSetKernelArg(kscan, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kscan, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kscan, 2, sizeof(cl_mem), &d_sizes);
    clSetKernelArg(kscan, 3, sizeof(cl_mem), &d_own);
    clSetKernelArg(kscan, 4, sizeof(cl_mem), &d_prefix);
    clSetKernelArg(kscan, 5, sizeof(cl_int), &i_nblk);
    clSetKernelArg(kscan, 6, sizeof(cl_int), &i_insize);
    clSetKernelArg(kscan, 7, sizeof(cl_int), &i_blk);
    clSetKernelArg(kscan, 8, sizeof(cl_int), &i_segmax);
    clSetKernelArg(kscan, 9, sizeof(cl_int), &i_N);
    clSetKernelArg(kscan, 10, sizeof(cl_uint), &epoch);

    double kus = 0.0;
    if (nk >= 1) {
        size_t g_build = (size_t)nblk * (size_t)nk * 4; /* LZ4_TP_BUILD_CHUNKS=4 */
        cl_event e_build;
        err = clEnqueueNDRangeKernel(queue, kbuild, 1, NULL, &g_build, &lws1, 0, NULL, &e_build);
        if (err != CL_SUCCESS) { fprintf(stderr, "tp-daemon: build enqueue %d\n", err); goto done; }
        clWaitForEvents(1, &e_build); kus += tp_event_us(e_build); clReleaseEvent(e_build);
    }
    { size_t g_scan = (size_t)nblk * (size_t)N;
      cl_event e_scan;
      err = clEnqueueNDRangeKernel(queue, kscan, 1, NULL, &g_scan, &lws1, 0, NULL, &e_scan);
      if (err != CL_SUCCESS) { fprintf(stderr, "tp-daemon: scan enqueue %d\n", err); goto done; }
      clWaitForEvents(1, &e_scan); kus += tp_event_us(e_scan); clReleaseEvent(e_scan); }
    clFinish(queue);
    if (kernel_us_out) *kernel_us_out = kus;
    }

    sizes = (cl_uint*)malloc(meta_cnt * sizeof(cl_uint));
    padded = (unsigned char*)malloc(out_bytes);
    if (!sizes || !padded) { fprintf(stderr, "tp-daemon: host oom\n"); goto done; }
    clEnqueueReadBuffer(queue, d_sizes, CL_TRUE, 0, meta_cnt * sizeof(cl_uint), sizes, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out,   CL_TRUE, 0, out_bytes, padded, 0, NULL, NULL);

    for (size_t g = 0; g < meta_cnt; g++) {
        if (sizes[g] == 0 || sizes[g] > (cl_uint)segMaxOut) {
            fprintf(stderr, "tp-daemon: bad segment size seg=%zu sz=%u (overflow?)\n", g, sizes[g]); goto done;
        }
        payload_total += sizes[g];
    }
    payload = (unsigned char*)malloc(payload_total ? payload_total : 1);
    if (!payload) { fprintf(stderr, "tp-daemon: payload oom\n"); goto done; }
    { size_t off = 0;
      for (size_t g = 0; g < meta_cnt; g++) { memcpy(payload + off, padded + g * (size_t)segMaxOut, sizes[g]); off += sizes[g]; } }

    {
        FILE* f = fopen(output_path, "wb");
        if (!f) { fprintf(stderr, "tp-daemon: cannot open %s\n", output_path); goto done; }
        cl_uint u_N = (cl_uint)N, u_hl = (cl_uint)hash_log, u_bs = (cl_uint)block_size, u_nblk = (cl_uint)nblk;
        unsigned long long u_orig = (unsigned long long)orig_size;
        fwrite(LZ4TP_MAGIC, 1, 8, f);
        fwrite(&u_N, 4, 1, f); fwrite(&u_hl, 4, 1, f); fwrite(&u_bs, 4, 1, f); fwrite(&u_nblk, 4, 1, f);
        fwrite(&u_orig, 8, 1, f);
        fwrite(sizes, sizeof(cl_uint), meta_cnt, f);
        fwrite(payload, 1, payload_total, f);
        fclose(f);
    }
    status = 0;

done:
    free(sizes); free(padded); free(payload); free(input_ref);
    if (d_input) clReleaseMemObject(d_input);
    if (d_prefix) clReleaseMemObject(d_prefix);
    if (d_own) clReleaseMemObject(d_own);
    if (d_out) clReleaseMemObject(d_out);
    if (d_sizes) clReleaseMemObject(d_sizes);
    return status;
}

/* orchestrate a daemon two-phase compress: ensure program+kernels, select N, run, learn */
static int daemon_twophase_compress(worker_res_t* w, request_t* req) {
    int hash_log = daemon_hash_log_from_request(req);
    int block_size = req->block_size > 0 ? req->block_size : (64 * 1024);
    int slot = hash_log - LZ4_HASHLOG_MIN;
    if (slot < 0) slot = 0;
    if (slot >= LZ4_HASHLOG_COUNT) slot = LZ4_HASHLOG_COUNT - 1;

    struct stat st;
    if (stat(req->input_path, &st) != 0 || st.st_size <= 0) { fprintf(stderr, "[tp] invalid input\n"); return -1; }
    size_t nblk = ((size_t)st.st_size + (size_t)block_size - 1) / (size_t)block_size;

    /* dedicated EPOCH32 tp program (forces -DLZ4_GPU_DICT_CLEAR=0 -DLZ4_GPU_DICT_ENTRY_BITS=32
     * via LZ4_GPU_EPOCH32, kept under compile_lock so no base compile observes the env). */
    pthread_mutex_lock(&g_state.compile_lock);
    if (!g_tp_program[slot]) {
        setenv("LZ4_GPU_EPOCH32", "1", 1);
        g_tp_program[slot] = lz4_load_program(g_state.context, g_state.device, hash_log, (size_t)block_size);
        unsetenv("LZ4_GPU_EPOCH32");
    }
    cl_program tprog = g_tp_program[slot];
    cl_kernel kbuild = NULL, kscan = NULL;
    if (tprog) {
        cl_int e;
        if (!w->kernel_tp_build[slot]) w->kernel_tp_build[slot] = clCreateKernel(tprog, "lz4_tp_build_prefix", &e);
        if (!w->kernel_tp_scan[slot])  w->kernel_tp_scan[slot]  = clCreateKernel(tprog, "lz4_tp_scan_seg", &e);
        kbuild = w->kernel_tp_build[slot]; kscan = w->kernel_tp_scan[slot];
    }
    pthread_mutex_unlock(&g_state.compile_lock);
    if (!kbuild || !kscan) { fprintf(stderr, "[tp] program/kernel load failed\n"); return -1; }

    /* dedicated profiling-enabled queue for pure-kernel tp timing (isolated from base) */
    if (!w->tp_queue) {
        cl_int qe;
        cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        w->tp_queue = clCreateCommandQueueWithProperties(g_state.context, g_state.device, props, &qe);
    }
    cl_command_queue q = w->tp_queue ? w->tp_queue : w->queue;

    uint32_t occ = 0;
    int N = tp_select_N(nblk, &occ);

    double kernel_us = 0.0;
    int rc = lz4_tp_compress_daemon(g_state.context, q, g_state.device,
                                    kbuild, kscan, req->input_path, req->output_path,
                                    N, hash_log, block_size, &kernel_us);
    if (rc != 0) return -1;

    double thr = (kernel_us > 0.0) ? ((double)st.st_size / kernel_us) : 0.0; /* bytes/us == MB/s */
    int W_sat_est = g_tp_learn.W_sat_est;
    if (g_tp_learn_enabled && thr > 0.0) tp_learn_record(nblk, N, thr, &W_sat_est);

    fprintf(stderr, "[learn] dev=%s nblk=%zu N=%d occ=%u thr=%.1fMB/s W_sat_est=%d\n",
            g_tp_learn.devname[0] ? g_tp_learn.devname : "?", nblk, N, occ, thr, W_sat_est);
    fflush(stderr);
    return 0;
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
        for (int m = 0; m < LZ4_DAEMON_PROGRAM_SLOTS; m++) {
            if (g_state.workers[i].kernel_comp[m]) {
                clReleaseKernel(g_state.workers[i].kernel_comp[m]);
                g_state.workers[i].kernel_comp[m] = NULL;
            }
            if (g_state.workers[i].kernel_decomp[m]) {
                clReleaseKernel(g_state.workers[i].kernel_decomp[m]);
                g_state.workers[i].kernel_decomp[m] = NULL;
            }
        }
        for (int m = 0; m < LZ4_HASHLOG_COUNT; m++) {
            if (g_state.workers[i].kernel_tp_build[m]) {
                clReleaseKernel(g_state.workers[i].kernel_tp_build[m]);
                g_state.workers[i].kernel_tp_build[m] = NULL;
            }
            if (g_state.workers[i].kernel_tp_scan[m]) {
                clReleaseKernel(g_state.workers[i].kernel_tp_scan[m]);
                g_state.workers[i].kernel_tp_scan[m] = NULL;
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
        if (g_state.workers[i].tp_queue) {
            clReleaseCommandQueue(g_state.workers[i].tp_queue);
            g_state.workers[i].tp_queue = NULL;
        }
        lz4_gpu_workspace_free(&g_state.workers[i].ws);
        pthread_mutex_unlock(&g_state.workers[i].lock);
        pthread_mutex_destroy(&g_state.workers[i].lock);
        pthread_cond_destroy(&g_state.workers[i].cond);
    }

    pthread_mutex_lock(&g_state.compile_lock);
    for (int m = 0; m < LZ4_DAEMON_PROGRAM_SLOTS; m++) {
        if (g_state.program_comp[m]) {
            clReleaseProgram(g_state.program_comp[m]);
            g_state.program_comp[m] = NULL;
        }
    }
    for (int m = 0; m < LZ4_HASHLOG_COUNT; m++) {
        if (g_tp_program[m]) {
            clReleaseProgram(g_tp_program[m]);
            g_tp_program[m] = NULL;
        }
    }
    pthread_mutex_unlock(&g_state.compile_lock);
    pthread_mutex_destroy(&g_state.compile_lock);
    pthread_mutex_destroy(&g_tp_learn.lock);

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

    /* I3: two-phase learning state (warm-start W_sat from the on-disk profile) */
    pthread_mutex_init(&g_tp_learn.lock, NULL);
    { int v; if (parse_env_int("LZ4_TP_LEARN", &v)) g_tp_learn_enabled = (v != 0); }
    tp_device_name(g_tp_learn.devname, sizeof(g_tp_learn.devname));
    g_tp_learn.W_sat_est = tp_profile_lookup(g_tp_learn.devname);
    fprintf(stderr, "[learn] init dev=%s learn=%d warm-start W_sat_est=%d\n",
            g_tp_learn.devname, g_tp_learn_enabled, g_tp_learn.W_sat_est);
    return 0;
}

void process_request(worker_res_t* w, request_t* req, response_t* res) {
    uint64_t t1 = get_us();
    timing_t t; memset(&t, 0, sizeof(t));
    int ret = -1;

    /* Per user request: in daemon mode, ocl_setup_us is always 0 for all requests */
    t.ocl_setup_us = 0;

    if (req->mode == mode_compress && (req->flags & LZ4_DAEMON_FLAG_TWOPHASE)) {
        ret = daemon_twophase_compress(w, req);
    } else if (req->mode == mode_compress) {
        const int hash_log = daemon_hash_log_from_request(req);
        const int kernel_mode = daemon_program_slot((uint32_t)req->block_size, hash_log);

        pthread_mutex_lock(&g_state.compile_lock);
        if (!g_state.program_comp[kernel_mode]) g_state.program_comp[kernel_mode] = lz4_load_program(g_state.context, g_state.device, hash_log, req->block_size);
        cl_program prog = g_state.program_comp[kernel_mode];
        cl_kernel kernel = NULL;
        if (prog) {
            if (!w->kernel_comp[kernel_mode]) { cl_int err; w->kernel_comp[kernel_mode] = clCreateKernel(prog, "lz4_compress_block", &err); }
            kernel = w->kernel_comp[kernel_mode];
        }
        pthread_mutex_unlock(&g_state.compile_lock);

        if (kernel) {
            ret = lz4_compress_core(g_state.context, w->queue, kernel, req->input_path, req->output_path,
                                  req->block_size, req->acceleration, hash_log, &w->ws, &t, req->local_size, 0);
        }
    } else {
        const int hash_log = daemon_hash_log_from_request(req);
        const int kernel_mode = daemon_program_slot((uint32_t)req->block_size, hash_log);
        pthread_mutex_lock(&g_state.compile_lock);
        if (!g_state.program_comp[kernel_mode]) g_state.program_comp[kernel_mode] = lz4_load_program(g_state.context, g_state.device, hash_log, req->block_size);
        cl_program prog = g_state.program_comp[kernel_mode];
        cl_kernel kernel = NULL;
        if (prog) {
            if (!w->kernel_decomp[kernel_mode]) { cl_int err; w->kernel_decomp[kernel_mode] = clCreateKernel(prog, "lz4_decompress_blocks", &err); }
            kernel = w->kernel_decomp[kernel_mode];
        }
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
