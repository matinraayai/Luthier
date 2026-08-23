CC := gcc
CFLAGS := -Wall -Wextra -fPIC -O0 -g

WRAPPER := libqueue_wrapper.so
SHIM := libkfd_ioctl_shim.so

.PHONY: all clean

all: $(WRAPPER) $(SHIM)

$(WRAPPER): queue_wrapper.c
	$(CC) $(CFLAGS) -shared -o $@ $< -ldl -lpthread

$(SHIM): kfd_ioctl_shim.c
	$(CC) $(CFLAGS) -rdynamic -shared -o $@ $< -ldl -lpthread

clean:
	rm -f $(WRAPPER) $(SHIM)
