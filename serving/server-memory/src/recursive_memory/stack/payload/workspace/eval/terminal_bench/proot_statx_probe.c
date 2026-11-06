#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/stat.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

/*
 * PRoot versions before statx translation support let the kernel resolve the
 * guest path against the host root.  That produces a false ENOENT even though
 * open(2), read(2), and older stat variants can see the same guest file.
 */
int main(void) {
    static const char path[] = "/tmp/harbor-proot-statx-probe";
    static const char payload[] = "statx-ok";
    struct statx metadata = {0};

    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) {
        fprintf(stderr, "open(%s): %s\n", path, strerror(errno));
        return 2;
    }
    ssize_t written = write(fd, payload, sizeof(payload) - 1);
    if (close(fd) != 0 || written != (ssize_t)(sizeof(payload) - 1)) {
        fprintf(stderr, "write(%s): %s\n", path, strerror(errno));
        return 3;
    }

    if (syscall(
            SYS_statx,
            AT_FDCWD,
            path,
            AT_STATX_SYNC_AS_STAT,
            STATX_TYPE | STATX_SIZE,
            &metadata) != 0) {
        fprintf(stderr, "statx(%s): %s\n", path, strerror(errno));
        return 4;
    }
    if (metadata.stx_size != sizeof(payload) - 1) {
        fprintf(
            stderr,
            "statx(%s): expected size %zu, got %llu\n",
            path,
            sizeof(payload) - 1,
            (unsigned long long)metadata.stx_size);
        return 5;
    }

    puts("proot-statx-ready");
    return 0;
}
