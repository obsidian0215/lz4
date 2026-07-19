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
#include "lz4_tp_profile.h"

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
    int worker_sync_count;
    int worker_thread_count;
    int compile_lock_initialized;
    int tp_learn_lock_initialized;
    int server_sock;
    volatile int running;
    int pid_fd;
} g_state = {.server_sock = -1, .pid_fd = -1};

/* ============================================================================
 * I3: two-phase compression served by the daemon.
 *   - A dedicated EPOCH32 tp program (DICT_CLEAR=0, ENTRY_BITS=32) cached
 *     global-under-lock, keyed by hash_log; kernels are per-worker.
 *   - Production uses the on-disk calibration profile. The live-traffic learner
 *     is retained only as an explicit experimental path because its estimate is
 *     sensitive to the observed file-size distribution.
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
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
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
        ssize_t got = read(fd, buf, want);
        if (got <= 0) goto out;
        if (daemon_write_full(sock, buf, (size_t)got) != 0) goto out;
        len -= (size_t)got;
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
    uint32_t allowed_flags = LZ4_DAEMON_FLAG_RAW_BUFFER | LZ4_DAEMON_FLAG_TWOPHASE;
    int raw = (req->flags & LZ4_DAEMON_FLAG_RAW_BUFFER) != 0;
    int twophase = (req->flags & LZ4_DAEMON_FLAG_TWOPHASE) != 0;
    if ((req->mode != mode_compress && req->mode != mode_decompress) ||
        req->acceleration < 1 || req->block_size < 1 || req->local_size < 1 ||
        (req->hash_log != 0 && (req->hash_log < LZ4_HASHLOG_MIN || req->hash_log > LZ4_HASHLOG_MAX)) ||
        (req->flags & ~allowed_flags) != 0 ||
        (twophase && (raw || req->mode != mode_compress || req->block_size != 64 * 1024)) ||
        (raw && req->input_size == 0) ||
        (!raw && (memchr(req->input_path, '\0', sizeof(req->input_path)) == NULL ||
                  memchr(req->output_path, '\0', sizeof(req->output_path)) == NULL ||
                  req->input_path[0] == '\0' || req->output_path[0] == '\0'))) {
        res->status = -1;
        snprintf(res->message, sizeof(res->message), "invalid daemon request fields");
        return 0;
    }
    return 1;
}

static int parse_env_int(const char* name, int* out_value) {
    const char* env = getenv(name);
    char* end = NULL;
    long v;
    if (!env || !*env) return 0;
    errno = 0;
    v = strtol(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' || v < INT_MIN || v > INT_MAX) return 0;
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
    if (n <= 0 || n >= (int)sizeof(buf) || ftruncate(fd, 0) != 0 ||
        daemon_write_fd_full(fd, buf, (size_t)n) != 0) {
        flock(fd, LOCK_UN);
        close(fd);
        unlink("/tmp/lz4_gpu_daemon.pid");
        return -1;
    }
    g_state.pid_fd = fd;
    return 0;
}

static void remove_pidfile(void) {
    if (g_state.pid_fd >= 0) {
        flock(g_state.pid_fd, LOCK_UN);
        close(g_state.pid_fd);
        unlink("/tmp/lz4_gpu_daemon.pid");
        g_state.pid_fd = -1;
    }
}

static void signal_handler(int sig) {
    (void)sig;
    g_state.running = 0;
}

/* ---- I3: two-phase codec + online W_sat learning implementation ---- */
#define LZ4TP_MAGIC "LZ4TP1\0\0"
static int g_tp_learn_enabled = 0;   /* set LZ4_TP_LEARN=1 only for experimental diagnosis */

/* profile I/O (device-keyed W_sat), mirrors lz4_gpu.c */
static const char* tp_profile_path(void) {
    const char* p = getenv("LZ4TP_PROFILE");
    return (p && *p) ? p : "lz4tp.profile";
}

static int tp_size_mul(size_t a, size_t b, size_t* out) {
    if (!out || (a != 0 && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int tp_size_add(size_t a, size_t b, size_t* out) {
    if (!out || b > SIZE_MAX - a) return -1;
    *out = a + b;
    return 0;
}

static int tp_write_exact(FILE* f, const void* data, size_t size) {
    return size == 0 || (f && fwrite(data, 1, size, f) == size);
}

static int tp_read_exact(FILE* f, void* data, size_t size) {
    return size == 0 || (f && fread(data, 1, size, f) == size);
}

static int tp_open_temp_output(const char* output_path, char* temp_path,
                               size_t capacity, FILE** output_file) {
    if (!output_path || !*output_path || !temp_path || !output_file ||
        snprintf(temp_path, capacity, "%s.tmp.XXXXXX", output_path) >= (int)capacity) {
        return -1;
    }
    int fd = mkstemp(temp_path);
    if (fd < 0) return -1;
    *output_file = fdopen(fd, "w+b");
    if (!*output_file) {
        close(fd);
        unlink(temp_path);
        return -1;
    }
    return 0;
}

static int tp_finalize_temp_output(FILE** output_file, const char* temp_path,
                                   const char* output_path) {
    if (!output_file || !*output_file) return -1;
    if (fflush(*output_file) != 0 || fclose(*output_file) != 0) {
        *output_file = NULL;
        unlink(temp_path);
        return -1;
    }
    *output_file = NULL;
    if (rename(temp_path, output_path) != 0) {
        unlink(temp_path);
        return -1;
    }
    return 0;
}

static int tp_paths_identify_same_file(const char* left, const char* right) {
    if (!left || !right) return 0;
    if (strcmp(left, right) == 0) return 1;
    struct stat left_stat, right_stat;
    if (stat(left, &left_stat) != 0 || stat(right, &right_stat) != 0) return 0;
    return left_stat.st_dev == right_stat.st_dev && left_stat.st_ino == right_stat.st_ino;
}

static size_t tp_apply_test_chunk_override(size_t safe_limit) {
    const char* value = getenv("LZ4TP_TEST_CHUNK_BLOCKS");
    if (!value || !*value || safe_limit == 0) return safe_limit;
    char* end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed == 0 || parsed > (unsigned long long)SIZE_MAX)
        return safe_limit;
    return (size_t)parsed < safe_limit ? (size_t)parsed : safe_limit;
}

static size_t tp_choose_chunk_blocks(cl_device_id device, size_t total_blocks,
                                     int N, int block_size, int hash_log) {
    if (!device || total_blocks == 0 || N < 1 || block_size < 1 ||
        hash_log < LZ4_HASHLOG_MIN || hash_log > LZ4_HASHLOG_MAX) return 0;
    const size_t hard_cap = 512;
    size_t dict_entries = (size_t)1u << hash_log;
    size_t seg_len_max = ((size_t)block_size + (size_t)N - 1) / (size_t)N;
    size_t seg_max_out = 0, per_prefix = 0, per_own = 0, per_output = 0;
    size_t per_total = 0, tmp = 0;
    if (tp_size_add(seg_len_max, seg_len_max / 255, &seg_max_out) != 0 ||
        tp_size_add(seg_max_out, 64, &seg_max_out) != 0 ||
        tp_size_mul((size_t)(N - 1), dict_entries, &tmp) != 0 ||
        tp_size_mul(tmp, sizeof(cl_uint), &per_prefix) != 0 ||
        tp_size_mul((size_t)N, dict_entries, &tmp) != 0 ||
        tp_size_mul(tmp, sizeof(cl_uint), &per_own) != 0 ||
        tp_size_mul((size_t)N, seg_max_out, &per_output) != 0 ||
        tp_size_add((size_t)block_size, per_prefix, &per_total) != 0 ||
        tp_size_add(per_total, per_own, &per_total) != 0 ||
        tp_size_add(per_total, per_output, &per_total) != 0 ||
        tp_size_mul((size_t)N, sizeof(cl_uint), &tmp) != 0 ||
        tp_size_add(per_total, tmp, &per_total) != 0 || per_output == 0) {
        return 0;
    }

    size_t limit = total_blocks < hard_cap ? total_blocks : hard_cap;
    size_t by_offset = (size_t)UINT_MAX / per_output;
    if (by_offset < limit) limit = by_offset;
    cl_ulong max_alloc = 0, global_mem = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    cl_ulong alloc_u64 = max_alloc ? (max_alloc / 4) * 3 : (cl_ulong)256 * 1024 * 1024;
    size_t alloc_budget = alloc_u64 > (cl_ulong)SIZE_MAX ? SIZE_MAX : (size_t)alloc_u64;
    size_t components[] = {(size_t)block_size, per_prefix, per_own, per_output};
    for (size_t i = 0; i < sizeof(components) / sizeof(components[0]); i++) {
        if (components[i] == 0) continue;
        size_t by_alloc = alloc_budget / components[i];
        if (by_alloc < limit) limit = by_alloc;
    }
    cl_ulong global_u64 = global_mem ? global_mem / 4 : (cl_ulong)512 * 1024 * 1024;
    size_t global_cap = (size_t)512 * 1024 * 1024;
    size_t global_budget = global_u64 > (cl_ulong)global_cap ? global_cap : (size_t)global_u64;
    size_t by_global = global_budget / per_total;
    if (by_global < limit) limit = by_global;
    return tp_apply_test_chunk_override(limit);
}

static int tp_write_empty_frame(const char* output_path, int block_size, int hash_log) {
    char temp_path[PATH_MAX];
    FILE* f = NULL;
    if (tp_open_temp_output(output_path, temp_path, sizeof(temp_path), &f) != 0) return -1;
    uint32_t n = 1, hash = (uint32_t)hash_log, block = (uint32_t)block_size, nblk = 0;
    uint64_t orig = 0;
    int ok = tp_write_exact(f, LZ4TP_MAGIC, 8) &&
             tp_write_exact(f, &n, 4) && tp_write_exact(f, &hash, 4) &&
             tp_write_exact(f, &block, 4) && tp_write_exact(f, &nblk, 4) &&
             tp_write_exact(f, &orig, 8);
    if (!ok) {
        fclose(f);
        unlink(temp_path);
        return -1;
    }
    return tp_finalize_temp_output(&f, temp_path, output_path);
}

static void tp_device_name(char* buf, size_t n) {
    buf[0] = 0;
    if (g_state.device) clGetDeviceInfo(g_state.device, CL_DEVICE_NAME, n, buf, NULL);
}

/* EXPLOIT rule: apply the calibrated occupancy threshold to the chunk that the
 * device will actually execute, rather than to the entire file. */
static int tp_pick_N(size_t nblk, int W_sat, int block_size, int hash_log,
                     size_t* chunk_blocks_out) {
    int cand[4] = {1,2,4,8}; int best = 1;
    size_t best_chunk = 0;
    double thresh = 0.75 * (double)W_sat;
    for (int i = 0; i < 4; i++) {
        size_t chunk = tp_choose_chunk_blocks(g_state.device, nblk, cand[i], block_size, hash_log);
        if (chunk == 0) continue;
        size_t effective_blocks = nblk < chunk ? nblk : chunk;
        if ((double)effective_blocks * (double)cand[i] <= thresh) {
            best = cand[i];
            best_chunk = chunk;
        } else if (cand[i] == 1 && best_chunk == 0) {
            best_chunk = chunk;
        }
    }
    if (chunk_blocks_out) *chunk_blocks_out = best_chunk;
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
            if (lz4_tp_profile_store(tp_profile_path(), g_tp_learn.devname,
                                     64 * 1024, 14, g_tp_learn.W_sat_est) != 0) {
                fprintf(stderr, "[learn] cannot update profile %s\n", tp_profile_path());
            }
        }
    }
    if (W_sat_est_out) *W_sat_est_out = g_tp_learn.W_sat_est;
    pthread_mutex_unlock(&g_tp_learn.lock);
}

/* Choose N for this request and report the execution-chunk occupancy. */
static int tp_select_N(size_t nblk, int block_size, int hash_log,
                       uint32_t* occ_out, size_t* chunk_blocks_out) {
    int N = 1;
    size_t chunk_blocks = 0;
    if (!g_tp_learn_enabled) {
        int W = lz4_tp_profile_lookup(tp_profile_path(), g_tp_learn.devname,
                                      block_size, hash_log);
        if (W > 0) N = tp_pick_N(nblk, W, block_size, hash_log, &chunk_blocks);
    } else {
        pthread_mutex_lock(&g_tp_learn.lock);
        int W = g_tp_learn.W_sat_est;
        int n_obs = g_tp_learn.n_obs;
        /* explore while not yet confident, else ~1 in 7 (7 is coprime with typical
         * request-sequence periods, so exploration is not aliased to one file size) */
        int explore = (W <= 0) || !tp_learn_ready() || (n_obs % 7 == 0);
        if (!explore && W > 0)
            N = tp_pick_N(nblk, W, block_size, hash_log, &chunk_blocks);
        else N = tp_learn_pick_explore_N((uint32_t)nblk);
        pthread_mutex_unlock(&g_tp_learn.lock);
    }
    if (N < 1) N = 1;
    if (N > 8) N = 8;
    if (chunk_blocks == 0)
        chunk_blocks = tp_choose_chunk_blocks(g_state.device, nblk, N, block_size, hash_log);
    size_t effective_blocks = nblk < chunk_blocks ? nblk : chunk_blocks;
    size_t occupancy = effective_blocks > SIZE_MAX / (size_t)N ? SIZE_MAX : effective_blocks * (size_t)N;
    if (occ_out) *occ_out = occupancy > UINT32_MAX ? UINT32_MAX : (uint32_t)occupancy;
    if (chunk_blocks_out) *chunk_blocks_out = chunk_blocks;
    return N;
}

/* pure GPU kernel time (us) from a profiling event, matches bench's event_elapsed_us */
static double tp_event_us(cl_event ev) {
    cl_ulong st = 0, en = 0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) != CL_SUCCESS) return 0.0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, NULL) != CL_SUCCESS) return 0.0;
    return (en >= st) ? (double)(en - st) / 1000.0 : 0.0;
}

static int tp_compress_chunk(cl_context ctx, cl_command_queue queue,
                             cl_kernel kbuild, cl_kernel kscan,
                             const unsigned char* input, size_t input_size,
                             size_t chunk_nblk, int N, int hash_log, int block_size,
                             int seg_max_out, cl_uint* sizes_out,
                             unsigned char* payload, size_t payload_capacity,
                             size_t* payload_used, double* kernel_us) {
    int status = 1;
    int nk = N - 1;
    size_t dict_entries = (size_t)1u << hash_log;
    size_t meta_cnt = 0, prefix_bytes = 0, own_bytes = 0, out_bytes = 0, tmp = 0;
    cl_int err = CL_SUCCESS;
    cl_mem d_input = NULL, d_prefix = NULL, d_own = NULL, d_out = NULL, d_sizes = NULL;
    unsigned char* padded = NULL;
    cl_event build_event = NULL, scan_event = NULL;

    if (!ctx || !queue || !kbuild || !kscan || !input || input_size == 0 ||
        input_size > (size_t)INT_MAX || chunk_nblk == 0 || chunk_nblk > (size_t)INT_MAX ||
        N < 1 || block_size < 1 || seg_max_out < 1 || !sizes_out || !payload ||
        !payload_used || !kernel_us ||
        tp_size_mul(chunk_nblk, (size_t)N, &meta_cnt) != 0 ||
        tp_size_mul(chunk_nblk, (size_t)nk, &tmp) != 0 ||
        tp_size_mul(tmp, dict_entries, &tmp) != 0 ||
        tp_size_mul(tmp, sizeof(cl_uint), &prefix_bytes) != 0 ||
        tp_size_mul(meta_cnt, dict_entries, &tmp) != 0 ||
        tp_size_mul(tmp, sizeof(cl_uint), &own_bytes) != 0 ||
        tp_size_mul(meta_cnt, (size_t)seg_max_out, &out_bytes) != 0 ||
        out_bytes > payload_capacity) {
        fprintf(stderr, "tp-daemon: invalid chunk dimensions\n");
        goto done;
    }

    d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY, input_size, NULL, &err);
    d_prefix = clCreateBuffer(ctx, CL_MEM_READ_WRITE, prefix_bytes ? prefix_bytes : 4, NULL, &err);
    d_own = clCreateBuffer(ctx, CL_MEM_READ_WRITE, own_bytes, NULL, &err);
    d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    d_sizes = clCreateBuffer(ctx, CL_MEM_READ_WRITE, meta_cnt * sizeof(cl_uint), NULL, &err);
    if (!d_input || !d_prefix || !d_own || !d_out || !d_sizes) {
        fprintf(stderr, "tp-daemon: chunk buffer alloc failed (%d)\n", err);
        goto done;
    }
    if (clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL) != CL_SUCCESS)
        goto done;
    {
        cl_uint zero = 0;
        if (prefix_bytes && clEnqueueFillBuffer(queue, d_prefix, &zero, sizeof(zero),
                                                0, prefix_bytes, 0, NULL, NULL) != CL_SUCCESS)
            goto done;
        if (clEnqueueFillBuffer(queue, d_own, &zero, sizeof(zero),
                                0, own_bytes, 0, NULL, NULL) != CL_SUCCESS ||
            clFinish(queue) != CL_SUCCESS)
            goto done;
    }

    cl_int i_nblk = (cl_int)chunk_nblk, i_insize = (cl_int)input_size;
    cl_int i_blk = block_size, i_segmax = seg_max_out, i_N = N;
    cl_uint epoch = 1;
    cl_int arg_err = CL_SUCCESS;
    arg_err |= clSetKernelArg(kbuild, 0, sizeof(cl_mem), &d_input);
    arg_err |= clSetKernelArg(kbuild, 1, sizeof(cl_mem), &d_prefix);
    arg_err |= clSetKernelArg(kbuild, 2, sizeof(cl_int), &i_nblk);
    arg_err |= clSetKernelArg(kbuild, 3, sizeof(cl_int), &i_insize);
    arg_err |= clSetKernelArg(kbuild, 4, sizeof(cl_int), &i_blk);
    arg_err |= clSetKernelArg(kbuild, 5, sizeof(cl_int), &i_N);
    arg_err |= clSetKernelArg(kbuild, 6, sizeof(cl_uint), &epoch);
    arg_err |= clSetKernelArg(kscan, 0, sizeof(cl_mem), &d_input);
    arg_err |= clSetKernelArg(kscan, 1, sizeof(cl_mem), &d_out);
    arg_err |= clSetKernelArg(kscan, 2, sizeof(cl_mem), &d_sizes);
    arg_err |= clSetKernelArg(kscan, 3, sizeof(cl_mem), &d_own);
    arg_err |= clSetKernelArg(kscan, 4, sizeof(cl_mem), &d_prefix);
    arg_err |= clSetKernelArg(kscan, 5, sizeof(cl_int), &i_nblk);
    arg_err |= clSetKernelArg(kscan, 6, sizeof(cl_int), &i_insize);
    arg_err |= clSetKernelArg(kscan, 7, sizeof(cl_int), &i_blk);
    arg_err |= clSetKernelArg(kscan, 8, sizeof(cl_int), &i_segmax);
    arg_err |= clSetKernelArg(kscan, 9, sizeof(cl_int), &i_N);
    arg_err |= clSetKernelArg(kscan, 10, sizeof(cl_uint), &epoch);
    if (arg_err != CL_SUCCESS) goto done;

    size_t lws = 1;
    if (nk >= 1) {
        size_t g_build = chunk_nblk * (size_t)nk * 4;
        err = clEnqueueNDRangeKernel(queue, kbuild, 1, NULL, &g_build, &lws,
                                     0, NULL, &build_event);
        if (err != CL_SUCCESS || clWaitForEvents(1, &build_event) != CL_SUCCESS) goto done;
        *kernel_us += tp_event_us(build_event);
    }
    size_t g_scan = chunk_nblk * (size_t)N;
    err = clEnqueueNDRangeKernel(queue, kscan, 1, NULL, &g_scan, &lws,
                                 0, NULL, &scan_event);
    if (err != CL_SUCCESS || clWaitForEvents(1, &scan_event) != CL_SUCCESS ||
        clFinish(queue) != CL_SUCCESS) goto done;
    *kernel_us += tp_event_us(scan_event);

    padded = (unsigned char*)malloc(out_bytes);
    if (!padded ||
        clEnqueueReadBuffer(queue, d_sizes, CL_TRUE, 0, meta_cnt * sizeof(cl_uint),
                            sizes_out, 0, NULL, NULL) != CL_SUCCESS ||
        clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, out_bytes,
                            padded, 0, NULL, NULL) != CL_SUCCESS) {
        goto done;
    }
    for (size_t g = 0; g < meta_cnt; g++) {
        size_t size = sizes_out[g];
        if (size == 0 || size > (size_t)seg_max_out || size > payload_capacity ||
            *payload_used > payload_capacity - size) {
            fprintf(stderr, "tp-daemon: invalid chunk segment size\n");
            goto done;
        }
        memcpy(payload + *payload_used, padded + g * (size_t)seg_max_out, size);
        *payload_used += size;
    }
    status = 0;

done:
    if (build_event) clReleaseEvent(build_event);
    if (scan_event) clReleaseEvent(scan_event);
    free(padded);
    if (d_input) clReleaseMemObject(d_input);
    if (d_prefix) clReleaseMemObject(d_prefix);
    if (d_own) clReleaseMemObject(d_own);
    if (d_out) clReleaseMemObject(d_out);
    if (d_sizes) clReleaseMemObject(d_sizes);
    return status;
}

/* two-phase codec: a copy of lz4_tp_compress_to_file that uses the passed-in warm
 * ctx/queue and cached epoch32 kbuild/kscan; per-request cl_mem, no ctx/queue release.
 * queue must be profiling-enabled: kernel time is measured via CL events (pure GPU
 * time, matching lz4_tp_measure_one) so throughput reflects occupancy, not dispatch. */
static int lz4_tp_compress_daemon(cl_context ctx, cl_command_queue queue, cl_device_id dev,
                                  cl_kernel kbuild, cl_kernel kscan,
                                  const char* input_path, const char* output_path,
                                  int N, int hash_log, int block_size,
                                  size_t chunk_blocks, double* kernel_us_out) {
    if (kernel_us_out) *kernel_us_out = 0.0;
    struct stat st;
    if (!ctx || !queue || !dev || !kbuild || !kscan || !input_path || !output_path ||
        !*output_path || tp_paths_identify_same_file(input_path, output_path) ||
        stat(input_path, &st) != 0 || st.st_size <= 0 ||
        (uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "tp-daemon: invalid input\n"); return 1;
    }
    if ((N != 1 && N != 2 && N != 4 && N != 8) || block_size < 1 ||
        hash_log < LZ4_HASHLOG_MIN || hash_log > LZ4_HASHLOG_MAX) {
        fprintf(stderr, "tp-daemon: unsupported parameters\n"); return 1;
    }

    size_t orig_size = (size_t)st.st_size;
    size_t nblk = orig_size / (size_t)block_size + (orig_size % (size_t)block_size != 0);
    size_t seg_len_max = ((size_t)block_size + (size_t)N - 1) / (size_t)N;
    size_t seg_max_out = 0, meta_cnt = 0;
    if (nblk > UINT32_MAX ||
        tp_size_add(seg_len_max, seg_len_max / 255, &seg_max_out) != 0 ||
        tp_size_add(seg_max_out, 64, &seg_max_out) != 0 || seg_max_out > INT_MAX ||
        tp_size_mul(nblk, (size_t)N, &meta_cnt) != 0 ||
        meta_cnt > SIZE_MAX / sizeof(cl_uint)) {
        fprintf(stderr, "tp-daemon: frame dimensions overflow\n"); return 1;
    }
    size_t safe_chunk = tp_choose_chunk_blocks(dev, nblk, N, block_size, hash_log);
    if (chunk_blocks == 0 || chunk_blocks > safe_chunk) chunk_blocks = safe_chunk;
    size_t input_capacity = 0, chunk_meta_capacity = 0, payload_capacity = 0;
    if (chunk_blocks == 0 ||
        tp_size_mul(chunk_blocks, (size_t)block_size, &input_capacity) != 0 ||
        input_capacity > INT_MAX ||
        tp_size_mul(chunk_blocks, (size_t)N, &chunk_meta_capacity) != 0 ||
        tp_size_mul(chunk_meta_capacity, seg_max_out, &payload_capacity) != 0 ||
        payload_capacity > UINT32_MAX) {
        fprintf(stderr, "tp-daemon: no safe OpenCL chunk size\n"); return 1;
    }

    FILE* input_file = NULL;
    FILE* output_file = NULL;
    cl_uint* sizes = NULL;
    unsigned char* input_chunk = NULL;
    unsigned char* payload = NULL;
    char temp_path[PATH_MAX] = {0};
    int temp_active = 0;
    size_t payload_total = 0;
    int status = 1;
    double kernel_us = 0.0;

    sizes = (cl_uint*)calloc(meta_cnt, sizeof(cl_uint));
    input_chunk = (unsigned char*)malloc(input_capacity);
    payload = (unsigned char*)malloc(payload_capacity);
    input_file = fopen(input_path, "rb");
    if (!sizes || !input_chunk || !payload || !input_file ||
        tp_open_temp_output(output_path, temp_path, sizeof(temp_path), &output_file) != 0) {
        fprintf(stderr, "tp-daemon: host allocation or file open failed\n"); goto done;
    }
    temp_active = 1;
    cl_uint u_N = (cl_uint)N, u_hl = (cl_uint)hash_log;
    cl_uint u_bs = (cl_uint)block_size, u_nblk = (cl_uint)nblk;
    unsigned long long u_orig = (unsigned long long)orig_size;
    if (!tp_write_exact(output_file, LZ4TP_MAGIC, 8) ||
        !tp_write_exact(output_file, &u_N, 4) ||
        !tp_write_exact(output_file, &u_hl, 4) ||
        !tp_write_exact(output_file, &u_bs, 4) ||
        !tp_write_exact(output_file, &u_nblk, 4) ||
        !tp_write_exact(output_file, &u_orig, 8) ||
        !tp_write_exact(output_file, sizes, meta_cnt * sizeof(cl_uint))) {
        fprintf(stderr, "tp-daemon: header write failed\n"); goto done;
    }

    for (size_t base = 0; base < nblk; base += chunk_blocks) {
        size_t this_blocks = nblk - base;
        if (this_blocks > chunk_blocks) this_blocks = chunk_blocks;
        size_t nominal_bytes = this_blocks * (size_t)block_size;
        size_t consumed = base * (size_t)block_size;
        size_t remaining = orig_size - consumed;
        size_t input_bytes = remaining < nominal_bytes ? remaining : nominal_bytes;
        if (!tp_read_exact(input_file, input_chunk, input_bytes)) {
            fprintf(stderr, "tp-daemon: input changed or read failed\n"); goto done;
        }
        size_t payload_used = 0;
        if (tp_compress_chunk(ctx, queue, kbuild, kscan, input_chunk, input_bytes,
                              this_blocks, N, hash_log, block_size, (int)seg_max_out,
                              sizes + base * (size_t)N, payload, payload_capacity,
                              &payload_used, &kernel_us) != 0 ||
            !tp_write_exact(output_file, payload, payload_used) ||
            tp_size_add(payload_total, payload_used, &payload_total) != 0) {
            fprintf(stderr, "tp-daemon: chunk processing failed\n"); goto done;
        }
    }
    if (fgetc(input_file) != EOF || ferror(input_file) ||
        fseek(output_file, 32L, SEEK_SET) != 0 ||
        !tp_write_exact(output_file, sizes, meta_cnt * sizeof(cl_uint)) ||
        tp_finalize_temp_output(&output_file, temp_path, output_path) != 0) {
        fprintf(stderr, "tp-daemon: frame finalization failed\n"); goto done;
    }
    temp_active = 0;
    if (kernel_us_out) *kernel_us_out = kernel_us;
    status = 0;

done:
    if (input_file) fclose(input_file);
    if (output_file) fclose(output_file);
    if (temp_active) unlink(temp_path);
    free(sizes);
    free(input_chunk);
    free(payload);
    return status;
}

/* orchestrate a daemon two-phase compress: ensure program+kernels, select N, run, learn */
static int daemon_twophase_compress(worker_res_t* w, request_t* req) {
    int hash_log = daemon_hash_log_from_request(req);
    int block_size = req->block_size > 0 ? req->block_size : (64 * 1024);
    int slot = hash_log - LZ4_HASHLOG_MIN;
    if (slot < 0) slot = 0;
    if (slot >= LZ4_HASHLOG_COUNT) slot = LZ4_HASHLOG_COUNT - 1;

    if (block_size != 64 * 1024) {
        fprintf(stderr, "[tp] daemon two-phase currently requires a 64K block size\n");
        return -1;
    }
    if (g_tp_learn_enabled && hash_log != 14) {
        fprintf(stderr, "[tp] experimental live learning requires hash_log=14\n");
        return -1;
    }

    struct stat st;
    if (!req->input_path[0] || !req->output_path[0] ||
        tp_paths_identify_same_file(req->input_path, req->output_path) ||
        stat(req->input_path, &st) != 0 || st.st_size < 0 ||
        (uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "[tp] invalid input\n"); return -1;
    }
    if (st.st_size == 0) return tp_write_empty_frame(req->output_path, block_size, hash_log);
    size_t input_size = (size_t)st.st_size;
    size_t nblk = input_size / (size_t)block_size + (input_size % (size_t)block_size != 0);
    if (nblk > UINT32_MAX) { fprintf(stderr, "[tp] too many blocks\n"); return -1; }

    /* dedicated EPOCH32 tp program (forces -DLZ4_GPU_DICT_CLEAR=0 -DLZ4_GPU_DICT_ENTRY_BITS=32
     * via LZ4_GPU_EPOCH32, kept under compile_lock so no base compile observes the env). */
    pthread_mutex_lock(&g_state.compile_lock);
    if (!g_tp_program[slot]) {
        if (setenv("LZ4_GPU_EPOCH32", "1", 1) == 0) {
            g_tp_program[slot] = lz4_load_program(g_state.context, g_state.device, hash_log, (size_t)block_size);
            unsetenv("LZ4_GPU_EPOCH32");
        }
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
        if (!w->tp_queue || qe != CL_SUCCESS) {
            fprintf(stderr, "[tp] profiling command queue creation failed (%d)\n", qe);
            return -1;
        }
    }
    cl_command_queue q = w->tp_queue;

    uint32_t occ = 0;
    size_t chunk_blocks = 0;
    int N = tp_select_N(nblk, block_size, hash_log, &occ, &chunk_blocks);
    if (chunk_blocks == 0) {
        fprintf(stderr, "[tp] no safe OpenCL chunk size\n");
        return -1;
    }
    if (g_tp_learn_enabled && nblk > chunk_blocks) {
        fprintf(stderr, "[tp] experimental live learning requires a single-chunk input\n");
        return -1;
    }

    double kernel_us = 0.0;
    int rc = lz4_tp_compress_daemon(g_state.context, q, g_state.device,
                                    kbuild, kscan, req->input_path, req->output_path,
                                    N, hash_log, block_size, chunk_blocks, &kernel_us);
    if (rc != 0) return -1;

    double thr = (kernel_us > 0.0) ? ((double)st.st_size / kernel_us) : 0.0; /* bytes/us == MB/s */
    int W_sat_est = g_tp_learn.W_sat_est;
    if (g_tp_learn_enabled && thr > 0.0) tp_learn_record(nblk, N, thr, &W_sat_est);

    fprintf(stderr, "[tp] dev=%s nblk=%zu chunk=%zu N=%d occ=%u kernel=%.1fMB/s W_sat_est=%d learn=%d\n",
            g_tp_learn.devname[0] ? g_tp_learn.devname : "?", nblk, chunk_blocks, N, occ, thr,
            W_sat_est, g_tp_learn_enabled);
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

    for (int i = 0; i < g_state.worker_sync_count; i++) {
        pthread_mutex_lock(&g_state.workers[i].lock);
        pthread_cond_signal(&g_state.workers[i].cond);
        pthread_mutex_unlock(&g_state.workers[i].lock);
    }
    for (int i = 0; i < g_state.worker_thread_count; i++) {
        pthread_join(g_state.workers[i].thread, NULL);
    }
    g_state.worker_thread_count = 0;
    for (int i = 0; i < g_state.worker_sync_count; i++) {
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
    g_state.worker_sync_count = 0;

    if (g_state.compile_lock_initialized) {
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
        g_state.compile_lock_initialized = 0;
    }
    if (g_state.tp_learn_lock_initialized) {
        pthread_mutex_destroy(&g_tp_learn.lock);
        g_state.tp_learn_lock_initialized = 0;
    }

    if (g_state.context) {
        clReleaseContext(g_state.context);
        g_state.context = NULL;
    }
}

int init_resources(void) {
    uint64_t t1 = get_us();
    cl_int err;
    g_state.active_workers = 0;
    g_state.worker_sync_count = 0;
    g_state.worker_thread_count = 0;
    g_state.compile_lock_initialized = 0;
    g_state.tp_learn_lock_initialized = 0;
    err = lz4_select_opencl_platform_device(&g_state.platform, &g_state.device);
    if (err != CL_SUCCESS || g_state.device == NULL || g_state.platform == NULL) goto fail;

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
    if (err != CL_SUCCESS || !g_state.context) goto fail;
    g_ocl_init_us = get_us() - t1;

    if (pthread_mutex_init(&g_state.compile_lock, NULL) != 0) goto fail;
    g_state.compile_lock_initialized = 1;
    g_state.active_workers = choose_daemon_worker_count(g_state.device);
    g_state.server_sock = -1;

    for (int i = 0; i < g_state.active_workers; i++) {
        g_state.workers[i].id = i;
        g_state.workers[i].client_fd = -1;
        {
            cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
            g_state.workers[i].queue = clCreateCommandQueueWithProperties(g_state.context, g_state.device, props, &err);
        }
        if (err != CL_SUCCESS || !g_state.workers[i].queue) goto fail;
        lz4_gpu_workspace_init(&g_state.workers[i].ws);
        if (pthread_mutex_init(&g_state.workers[i].lock, NULL) != 0) {
            lz4_gpu_workspace_free(&g_state.workers[i].ws);
            clReleaseCommandQueue(g_state.workers[i].queue);
            g_state.workers[i].queue = NULL;
            goto fail;
        }
        if (pthread_cond_init(&g_state.workers[i].cond, NULL) != 0) {
            pthread_mutex_destroy(&g_state.workers[i].lock);
            lz4_gpu_workspace_free(&g_state.workers[i].ws);
            clReleaseCommandQueue(g_state.workers[i].queue);
            g_state.workers[i].queue = NULL;
            goto fail;
        }
        g_state.worker_sync_count++;
    }

    /* I3: profile-based selection; optional live learning is experimental. */
    if (pthread_mutex_init(&g_tp_learn.lock, NULL) != 0) goto fail;
    g_state.tp_learn_lock_initialized = 1;
    { int v; if (parse_env_int("LZ4_TP_LEARN", &v)) g_tp_learn_enabled = (v != 0); }
    tp_device_name(g_tp_learn.devname, sizeof(g_tp_learn.devname));
    g_tp_learn.W_sat_est = lz4_tp_profile_lookup(tp_profile_path(), g_tp_learn.devname,
                                                 64 * 1024, 14);
    fprintf(stderr, "[learn] init dev=%s learn=%d warm-start W_sat_est=%d\n",
            g_tp_learn.devname, g_tp_learn_enabled, g_tp_learn.W_sat_est);
    return 0;

fail:
    cleanup_resources();
    return -1;
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
    signal(SIGTERM, signal_handler); signal(SIGINT, signal_handler); signal(SIGPIPE, SIG_IGN);
    if (init_resources() < 0) {
        fprintf(stderr, "Daemon resource initialization failed\n");
        remove_pidfile();
        return 1;
    }
    g_state.server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_state.server_sock < 0) goto fail;
    struct sockaddr_un addr; memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX; strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    unlink(SOCKET_PATH);
    if (bind(g_state.server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0 ||
        listen(g_state.server_sock, 5) < 0) goto fail;
    g_state.running = 1;
    for (int i = 0; i < g_state.active_workers; i++) {
        if (pthread_create(&g_state.workers[i].thread, NULL, worker_thread,
                           &g_state.workers[i]) != 0) goto fail;
        g_state.worker_thread_count++;
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

fail:
    fprintf(stderr, "Daemon startup failed\n");
    cleanup_resources();
    remove_pidfile();
    return 1;
}
