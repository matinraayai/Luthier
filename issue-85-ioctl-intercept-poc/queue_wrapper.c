// A minimal LD_PRELOAD library that intercepts AMD KFD queue creation and puts
// Luthier between the application and the GPU, one layer below HSA. Works on
// programs that never touch the HSA runtime (issue #85).
//
// HOW TO USE
//   make                                     # builds libqueue_wrapper.so
//   LD_PRELOAD=$(pwd)/libqueue_wrapper.so <executable>
//
//   LUTHIER_VERBOSE=1          log every forwarded packet and callback.
//   LUTHIER_DEMO_ZERO_GRID=1   opt-in proof that a callback edit changes what the
//                              GPU runs; see luthier_rewrite_packet().
// 
//   1. Substitute the ring. At AMDKFD_IOC_CREATE_QUEUE we swap the queue's ring
//      buffer for one we allocate and register with the GPU, so the GPU reads
//      from our buffer while the program keeps writing its own.
//   2. Copy header-last. A background thread copies each finished packet into our
//      ring, runs the Luthier callback in between, and writes our copy's header
//      LAST. The GPU ignores a slot whose header says INVALID, so the callback is
//      guaranteed to run before the GPU can act on the packet.
//   3. Detect with two signals:
//        - The program's write pointer counts the # of claimed slots. It is
//          a safe upper bound on where to look, this bound the scan is what
//          keeps the copy loop finite.
//        - The slot's header says whether THAT slot is finished, because the
//          producer writes the header last. That is the commit test.
//      We also establish the "empty" marker ourselves and put it back
//      after copying each packet. The HSA runtime promises its callers that every
//      slot starts INVALID, but that is an HSA convention.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <linux/kfd_ioctl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define QW_LOG "[queue_wrapper] "

// The real ioctl
typedef int (*real_ioctl_fn)(int fd, unsigned long request, void *arg);
static real_ioctl_fn real_ioctl = NULL;

static void ensure_real_ioctl_resolved(void) {
  if (!real_ioctl) {
    real_ioctl = (real_ioctl_fn)dlsym(RTLD_NEXT, "ioctl");
    if (!real_ioctl) {
      fprintf(stderr, QW_LOG "dlsym(RTLD_NEXT, \"ioctl\") failed: %s\n",
              dlerror());
      abort();
    }
  }
}

// Identify /dev/kfd by device number
static dev_t kfd_rdev;
static int kfd_rdev_cached = 0;

static void ensure_kfd_rdev_cached(void) {
  if (kfd_rdev_cached) {
    return;
  }
  struct stat st;
  if (stat("/dev/kfd", &st) == 0 && S_ISCHR(st.st_mode)) {
    kfd_rdev = st.st_rdev;
  } else {
    fprintf(stderr, QW_LOG "stat(\"/dev/kfd\") failed; fd checks will fail\n");
    kfd_rdev = 0;
  }
  kfd_rdev_cached = 1;
}

static int fd_is_kfd(int fd) {
  ensure_kfd_rdev_cached();
  struct stat st;
  if (fstat(fd, &st) != 0) {
    return 0;
  }
  return S_ISCHR(st.st_mode) && st.st_rdev == kfd_rdev;
}

// Page-aligned anonymous allocation for the shadow ring
static void *alloc_shim_page(size_t min_size) {
  static long page_size = 0;
  if (page_size == 0) {
    page_size = sysconf(_SC_PAGESIZE);
  }
  size_t size = ((min_size + page_size - 1) / page_size) * page_size;
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    fprintf(stderr, QW_LOG "mmap(%zu) failed: %s\n", size, strerror(errno));
    abort();
  }
  return p;
}

// Register a shim ring with the GPU
// CREATE_QUEUE rejects an unregistered buffer (-EINVAL)
// (ALLOC_MEMORY_OF_GPU + MAP_MEMORY_TO_GPU) the shadow ring first. 
#define KFD_RING_ALLOC_FLAGS                                                   \
  (KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |        \
   KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |                                        \
   KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |  \
   KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)

static int register_shim_ring_with_gpu(int fd, void *va, size_t size,
                                       __u32 gpu_id) {
  struct kfd_ioctl_alloc_memory_of_gpu_args alloc_args;
  memset(&alloc_args, 0, sizeof(alloc_args));
  alloc_args.va_addr = (__u64)(uintptr_t)va;
  alloc_args.size = size;
  alloc_args.mmap_offset = (__u64)(uintptr_t)va;
  alloc_args.gpu_id = gpu_id;
  alloc_args.flags = KFD_RING_ALLOC_FLAGS;
  if (real_ioctl(fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc_args) != 0) {
    fprintf(stderr, QW_LOG "ALLOC_MEMORY_OF_GPU(0x%llx) failed: %s\n",
            (unsigned long long)alloc_args.va_addr, strerror(errno));
    return -1;
  }

  // Map the ring to this queue's own GPU node.
  __u32 device_ids[1] = {gpu_id};
  struct kfd_ioctl_map_memory_to_gpu_args map_args;
  memset(&map_args, 0, sizeof(map_args));
  map_args.handle = alloc_args.handle;
  map_args.device_ids_array_ptr = (__u64)(uintptr_t)device_ids;
  map_args.n_devices = 1;
  if (real_ioctl(fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map_args) != 0 ||
      map_args.n_success != 1) {
    fprintf(stderr, QW_LOG "MAP_MEMORY_TO_GPU(0x%llx) failed: %s (%u/1)\n",
            (unsigned long long)alloc_args.va_addr, strerror(errno),
            map_args.n_success);
    return -1;
  }
  return 0;
}

// AQL packet layout
// An AQL packet's first 2 bytes are the header whose low 8 bits are the packet
// TYPE. A KERNEL_DISPATCH carries the kernel entry address at byte offset 32.
#define AQL_PACKET_BYTES 64
#define AQL_HEADER_OFF 0
#define AQL_GRID_SIZE_X_OFF 12
#define AQL_KERNEL_OBJECT_OFF 32
#define AQL_HEADER_TYPE(h) ((unsigned)((h) & 0xFF))
#define HSA_PACKET_TYPE_INVALID 1
#define HSA_PACKET_TYPE_KERNEL_DISPATCH 2

// Per-wrapped-queue forwarding state
struct fwd_queue {
  volatile unsigned char *app_ring;  // app's ORIGINAL ring (app writes here)
  volatile unsigned char *shim_ring; // our registered ring (GPU reads here)
  volatile uint64_t *app_wptr;       // app's own packet counter (read-only)
  uint32_t nslots;                   // ring_bytes / 64
  __u32 gpu_id;
  __u32 queue_id;
  uint64_t forwarded; // next slot index to forward; also the total copied
  uint64_t n_dispatch; // how many of those were kernel dispatches
  int active;         // 0 after DESTROY_QUEUE
  int summarized;     // 1 once the teardown summary has been printed
};

#define FWD_MAX_QUEUES 64
static struct fwd_queue g_fwd[FWD_MAX_QUEUES];
static int g_fwd_count = 0;
static pthread_mutex_t g_fwd_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_fwd_thread;
static int g_fwd_thread_started = 0;

// Per-packet logging (LUTHIER_VERBOSE=1), off by default.
static int qw_verbose(void) {
  static int cache = -1;
  if (cache < 0) {
    const char *v = getenv("LUTHIER_VERBOSE");
    cache = (v && v[0] == '1') ? 1 : 0;
  }
  return cache;
}

static int demo_zero_grid_enabled(void) {
  static int cache = -1;
  if (cache < 0) {
    const char *v = getenv("LUTHIER_DEMO_ZERO_GRID");
    cache = (v && v[0] == '1') ? 1 : 0;
  }
  return cache;
}

// Luthier callback
// Runs on the shim-ring copy of a packet while the slot header is INVALID.
static void luthier_rewrite_packet(volatile unsigned char *slot, uint16_t hdr) {
  if (AQL_HEADER_TYPE(hdr) != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
    return;
  }
  if (qw_verbose()) {
    uint64_t kernel_object;
    memcpy(&kernel_object, (const void *)(slot + AQL_KERNEL_OBJECT_OFF),
           sizeof(kernel_object));
    fprintf(stderr, QW_LOG "callback: kernel dispatch, kernel_object=0x%llx\n",
            (unsigned long long)kernel_object);
  }

  if (demo_zero_grid_enabled()) {
    uint32_t old_grid;
    memcpy(&old_grid, (const void *)(slot + AQL_GRID_SIZE_X_OFF),
           sizeof(old_grid));
    uint32_t new_grid = 0;
    memcpy((void *)(slot + AQL_GRID_SIZE_X_OFF), &new_grid, sizeof(new_grid));
    fprintf(stderr,
            QW_LOG "callback DEMO: grid_size_x %u -> %u (kernel should not run)\n",
            old_grid, new_grid);
  }
}

// Forward one packet: app_ring[idx] -> shim_ring[idx], header-last
static void forward_one_packet(struct fwd_queue *q, uint64_t idx) {
  size_t slot = (size_t)(idx % q->nslots);
  volatile unsigned char *src = q->app_ring + slot * AQL_PACKET_BYTES;
  volatile unsigned char *dst = q->shim_ring + slot * AQL_PACKET_BYTES;

  // load the "valid" header from the app's ring
  uint16_t hdr = __atomic_load_n((const uint16_t *)(src + AQL_HEADER_OFF),
                                 __ATOMIC_ACQUIRE);

  // header-as-gate on the write side:
  // (1) close the shim slot (INVALID)
  // (2) copy the body
  // (3) run the callback
  // (4) commit the real header LAST with a release store.
  // Until (4) the GPU can't see the packet
  uint16_t invalid = HSA_PACKET_TYPE_INVALID;
  __atomic_store_n((uint16_t *)(dst + AQL_HEADER_OFF), invalid,
                   __ATOMIC_RELEASE);
  memcpy((void *)(dst + 2), (const void *)(src + 2), AQL_PACKET_BYTES - 2);
  luthier_rewrite_packet(dst, hdr);
  __atomic_store_n((uint16_t *)(dst + AQL_HEADER_OFF), hdr, __ATOMIC_RELEASE);

  if (AQL_HEADER_TYPE(hdr) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
    q->n_dispatch++;
  }

  if (qw_verbose()) {
    fprintf(stderr, QW_LOG "forwarded gpu=%u q=%u idx=%llu slot=%zu type=%u\n",
            q->gpu_id, q->queue_id, (unsigned long long)idx, slot,
            AQL_HEADER_TYPE(hdr));
  }
}

// Poll the app's ring headers
//  - The program's write pointer counts the # of claimed slots. It is
//    a safe upper bound on where to look, this bound the scan is what
//    keeps the copy loop finite.
//  - The slot's header says whether THAT slot is finished, because the
//    producer writes the header last. That is the commit test.
static void *forward_poller_main(void *unused) {
  (void)unused;
  for (;;) {
    pthread_mutex_lock(&g_fwd_lock);
    int n = g_fwd_count;
    pthread_mutex_unlock(&g_fwd_lock);

    for (int i = 0; i < n; i++) {
      struct fwd_queue *q = &g_fwd[i];
      if (!__atomic_load_n(&q->active, __ATOMIC_ACQUIRE)) {
        continue; // queue destroyed
      }

      uint64_t claimed =
          q->app_wptr ? __atomic_load_n(q->app_wptr, __ATOMIC_ACQUIRE) : 0;

      uint32_t budget = q->nslots;
      while (q->forwarded < claimed && budget > 0) {
        budget--;
        size_t slot = (size_t)(q->forwarded % q->nslots);
        volatile unsigned char *src = q->app_ring + slot * AQL_PACKET_BYTES;
        uint16_t hdr = __atomic_load_n((const uint16_t *)(src + AQL_HEADER_OFF),
                                       __ATOMIC_ACQUIRE);
        if (AQL_HEADER_TYPE(hdr) == HSA_PACKET_TYPE_INVALID) {
          break; // claimed, but the app hasn't finished writing it yet
        }
        forward_one_packet(q, q->forwarded);

        // Re-arm the marker so this slot reads "empty" again. Without this, the
        // next lap finds the previous packet's still-valid header and we cannot
        // differentiate reused header from freshly written one.

        __atomic_store_n((uint16_t *)(src + AQL_HEADER_OFF),
                         (uint16_t)HSA_PACKET_TYPE_INVALID, __ATOMIC_RELEASE);
        q->forwarded++;
      }
    }
    struct timespec ts = {0, 20 * 1000}; // 20 us poll
    nanosleep(&ts, NULL);
  }
  return NULL;
}

static void fwd_summarize_all(void); // defined below; registered with atexit()

// Claim a tracking slot BEFORE the queue's ring is substituted. Returns the
// slot index, or -1 if the table is full.
static int fwd_reserve_slot(void) {
  int idx = -1;
  pthread_mutex_lock(&g_fwd_lock);
  if (g_fwd_count < FWD_MAX_QUEUES) {
    idx = g_fwd_count++;
  }
  pthread_mutex_unlock(&g_fwd_lock);
  return idx;
}

// Fill in a reserved slot and publish it to the poller.
static void fwd_commit_slot(int idx, volatile unsigned char *app_ring,
                            volatile unsigned char *shim_ring,
                            volatile uint64_t *app_wptr, uint32_t ring_bytes,
                            __u32 gpu_id, __u32 queue_id) {
  struct fwd_queue *q = &g_fwd[idx];
  q->app_ring = app_ring;
  q->shim_ring = shim_ring;
  q->app_wptr = app_wptr;
  q->nslots = ring_bytes / AQL_PACKET_BYTES;
  q->gpu_id = gpu_id;
  q->queue_id = queue_id;
  q->forwarded = 0;
  q->n_dispatch = 0;
  q->summarized = 0;

  // Publish last, with a release store. The poller reads these fields without
  // holding the lock, and its acquire load of `active` pairs with this store to
  // guarantee it sees everything written above.
  __atomic_store_n(&q->active, 1, __ATOMIC_RELEASE);

  pthread_mutex_lock(&g_fwd_lock);
  if (!g_fwd_thread_started) {
    g_fwd_thread_started =
        (pthread_create(&g_fwd_thread, NULL, forward_poller_main, NULL) == 0);
    atexit(fwd_summarize_all);
  }
  pthread_mutex_unlock(&g_fwd_lock);
}

// One line per queue at teardown, so the numbers worth checking survive with
// per-packet logging off.
static void fwd_summarize_queue(struct fwd_queue *q) {
  if (__atomic_exchange_n(&q->summarized, 1, __ATOMIC_ACQ_REL)) {
    return;
  }
  fprintf(stderr,
          QW_LOG "summary gpu=%u q=%u: forwarded=%llu packets, "
                 "dispatches=%llu\n",
          q->gpu_id, q->queue_id, (unsigned long long)q->forwarded,
          (unsigned long long)q->n_dispatch);
}

// catch queues the program never destroyed before exiting.
static void fwd_summarize_all(void) {
  int n = __atomic_load_n(&g_fwd_count, __ATOMIC_ACQUIRE);
  for (int i = 0; i < n; i++) {
    if (__atomic_load_n(&g_fwd[i].active, __ATOMIC_ACQUIRE)) {
      fwd_summarize_queue(&g_fwd[i]);
    }
  }
}

// Teardown safety: stop the poller from touching a queue's app ring before
// the app frees it.
static void fwd_deactivate_queue(__u32 queue_id) {
  pthread_mutex_lock(&g_fwd_lock);
  for (int i = 0; i < g_fwd_count; i++) {
    if (g_fwd[i].queue_id == queue_id) {
      __atomic_store_n(&g_fwd[i].active, 0, __ATOMIC_RELEASE);
      fwd_summarize_queue(&g_fwd[i]);
    }
  }
  pthread_mutex_unlock(&g_fwd_lock);
}

// ioctl request decoding (can't include <sys/ioctl.h>)
#define KFD_IOC_NR(req) ((unsigned int)((req) & 0xFFu))
#define KFD_CREATE_QUEUE_NR (AMDKFD_IOC_CREATE_QUEUE & 0xFFu)
#define KFD_DESTROY_QUEUE_NR (AMDKFD_IOC_DESTROY_QUEUE & 0xFFu)

// Substitute the ring at queue creation, then register the queue for
// forwarding.
static int handle_create_queue(int fd, unsigned long request, void *arg) {
  struct kfd_ioctl_create_queue_args *q =
      (struct kfd_ioctl_create_queue_args *)arg;

  // Only wrap AQL compute queues
  if (q->queue_type != KFD_IOC_QUEUE_TYPE_COMPUTE_AQL) {
    return real_ioctl(fd, request, arg);
  }

  // Reserve tracking capacity BEFORE touching anything. If we cannot track this
  // queue we must not substitute its ring   
  int slot_idx = fwd_reserve_slot();
  if (slot_idx < 0) {
    fprintf(stderr,
            QW_LOG "WARNING: already tracking %d queues (FWD_MAX_QUEUES); "
                   "queue on gpu=%u created UNWRAPPED; its packets will not "
                   "reach the callback\n",
            FWD_MAX_QUEUES, q->gpu_id);
    return real_ioctl(fd, request, arg);
  }

  unsigned long long app_ring_va = (unsigned long long)q->ring_base_address;
  unsigned long long app_wptr_va =
      (unsigned long long)q->write_pointer_address;
  uint32_t ring_bytes = q->ring_size;

  // Build the shadow ring and allocate it
  void *shim_ring = alloc_shim_page(ring_bytes);
  uint16_t inv = HSA_PACKET_TYPE_INVALID;
  // pre-fill INVALID headers,
  for (uint32_t off = 0; off + AQL_PACKET_BYTES <= ring_bytes;
       off += AQL_PACKET_BYTES) {
    memcpy((unsigned char *)shim_ring + off + AQL_HEADER_OFF, &inv,
           sizeof(inv));
  }
  // register shadow ring with GPU
  if (register_shim_ring_with_gpu(fd, shim_ring, ring_bytes, q->gpu_id) != 0) {
    fprintf(stderr,
            QW_LOG "ring registration failed; queue created unwrapped\n");
    return real_ioctl(fd, request, arg);
  }
  // substitute ring address
  q->ring_base_address = (__u64)(uintptr_t)shim_ring;

  int ret = real_ioctl(fd, request, arg);
  int saved_errno = errno;
  if (ret != 0) {
    // The reserved slot is abandoned, the poller skips it forever. 
    fprintf(stderr, QW_LOG "CREATE_QUEUE failed ret=%d errno=%d (%s)\n", ret,
            saved_errno, strerror(saved_errno));
    return ret;
  }

  // Establish the "empty" marker in the APP's ring ourselves.
  for (uint32_t off = 0; off + AQL_PACKET_BYTES <= ring_bytes;
       off += AQL_PACKET_BYTES) {
    __atomic_store_n((uint16_t *)((uintptr_t)app_ring_va + off + AQL_HEADER_OFF),
                     (uint16_t)HSA_PACKET_TYPE_INVALID, __ATOMIC_RELEASE);
  }

  fprintf(stderr,
          QW_LOG "wrapped AQL compute queue gpu=%u queue_id=%u "
                 "app_ring=0x%llx shim_ring=%p ring_bytes=%u app_wptr=0x%llx\n",
          q->gpu_id, q->queue_id, app_ring_va, shim_ring, ring_bytes,
          app_wptr_va);
  fwd_commit_slot(slot_idx, (volatile unsigned char *)(uintptr_t)app_ring_va,
                  (volatile unsigned char *)shim_ring,
                  (volatile uint64_t *)(uintptr_t)app_wptr_va, ring_bytes,
                  q->gpu_id, q->queue_id);
  return ret;
}

int ioctl(int fd, unsigned long request, void *arg) {
  ensure_real_ioctl_resolved();

  int is_kfd = fd_is_kfd(fd);
  unsigned int nr = KFD_IOC_NR(request);

  if (is_kfd && nr == KFD_DESTROY_QUEUE_NR && arg != NULL) {
    fwd_deactivate_queue(*(const __u32 *)arg);
  }

  if (is_kfd && nr == KFD_CREATE_QUEUE_NR && arg != NULL) {
    return handle_create_queue(fd, request, arg);
  }

  return real_ioctl(fd, request, arg);
}
