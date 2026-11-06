#define _GNU_SOURCE

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * Trusted host-side child launcher for proot_package_bridge.py.
 *
 * argv[1] is "--identity-gate-fd", argv[2] is a trusted inherited descriptor,
 * argv[3] is the broker-pinned rootfs, argv[4] the validated guest cwd,
 * argv[5] the broker-selected fixed guest executable, and argv[6...] the
 * validated command argv (including its preserved argv[0]).  The one-byte
 * gate prevents the command from running until the Python broker has pinned
 * and proved the identity of the host-side subreaper that owns this helper.
 */
int main(int argc, char **argv, char **environment) {
    if (argc < 7 || strcmp(argv[1], "--identity-gate-fd") != 0) {
        dprintf(STDERR_FILENO, "package bridge launcher: invalid argument count\n");
        return 126;
    }
    errno = 0;
    char *gate_end = NULL;
    long gate_value = strtol(argv[2], &gate_end, 10);
    if (errno != 0 || gate_end == argv[2] || *gate_end != '\0' ||
        gate_value < 3 || gate_value > 1048576) {
        dprintf(STDERR_FILENO, "package bridge launcher: invalid identity gate\n");
        return 126;
    }
    int gate_fd = (int)gate_value;
    unsigned char gate = 0;
    ssize_t received;
    do {
        received = read(gate_fd, &gate, 1);
    } while (received < 0 && errno == EINTR);
    close(gate_fd);
    if (received != 1 || gate != 'G') {
        dprintf(STDERR_FILENO, "package bridge launcher: identity gate denied\n");
        return 126;
    }
    if (argv[3][0] != '/' || argv[4][0] != '/' || argv[5][0] != '/') {
        dprintf(STDERR_FILENO, "package bridge launcher: paths must be absolute\n");
        return 126;
    }
    if (chroot(argv[3]) != 0) {
        dprintf(STDERR_FILENO, "package bridge launcher: chroot: %s\n", strerror(errno));
        return 126;
    }
    if (chdir(argv[4]) != 0) {
        dprintf(STDERR_FILENO, "package bridge launcher: chdir: %s\n", strerror(errno));
        return 126;
    }
    execve(argv[5], &argv[6], environment);
    dprintf(STDERR_FILENO, "package bridge launcher: execve: %s\n", strerror(errno));
    return 126;
}
