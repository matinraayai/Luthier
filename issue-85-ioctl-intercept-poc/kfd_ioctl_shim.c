#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <linux/kfd_ioctl.h>

// fuction signature
typedef int (*real_ioctl_fn)(int fd, unsigned long request, void *arg);
static real_ioctl_fn real_ioctl = NULL;

// dlsym the real ioctl 
static void ensure_real_ioctl_resolved(void) {
  if (!real_ioctl) {
    real_ioctl = (real_ioctl_fn)dlsym(RTLD_NEXT, "ioctl");
    if (!real_ioctl) {
      fprintf(stderr, "kfd_ioctl_shim: dlsym(RTLD_NEXT, \"ioctl\") failed: %s\n",
              dlerror());
      abort();
    }
  }
}

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
    fprintf(stderr, "kfd_ioctl_shim: stat(\"/dev/kfd\") failed or not a "
                     "character device. subsequent fd checks will always "
                     "fail\n");
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

#define KFD_IOC_TYPE_NR_MASK 0xFFFFu

int ioctl(int fd, unsigned long request, void *arg) {
  ensure_real_ioctl_resolved();

  if ((request & KFD_IOC_TYPE_NR_MASK) ==
          (AMDKFD_IOC_CREATE_QUEUE & KFD_IOC_TYPE_NR_MASK) &&
      fd_is_kfd(fd)) {
    struct kfd_ioctl_create_queue_args *q =
        (struct kfd_ioctl_create_queue_args *)arg;

    fprintf(stderr, "[kfd_ioctl_shim] AMDKFD_IOC_CREATE_QUEUE, before real ioctl:\n");
    fprintf(stderr, "  ring_base_address = 0x%llx\n",
            (unsigned long long)q->ring_base_address);
    fprintf(stderr, "  ring_size         = %u\n", q->ring_size);
    fprintf(stderr, "  gpu_id            = %u\n", q->gpu_id);
    fprintf(stderr, "  queue_type        = %u\n", q->queue_type);
    fprintf(stderr, "  queue_percentage  = %u\n", q->queue_percentage);
    fprintf(stderr, "  queue_priority    = %u\n", q->queue_priority);
    fprintf(stderr, "  write_pointer_address = 0x%llx\n",
            (unsigned long long)q->write_pointer_address);
    fprintf(stderr, "  read_pointer_address  = 0x%llx\n",
            (unsigned long long)q->read_pointer_address);



    int ret = real_ioctl(fd, request, arg);

    fprintf(stderr, "[kfd_ioctl_shim] AMDKFD_IOC_CREATE_QUEUE, after real ioctl (ret=%d):\n",
            ret);
    fprintf(stderr, "  doorbell_offset   = 0x%llx\n",
            (unsigned long long)q->doorbell_offset);
    fprintf(stderr, "  queue_id          = %u\n", q->queue_id);


    return ret;
  }

  return real_ioctl(fd, request, arg);
}
