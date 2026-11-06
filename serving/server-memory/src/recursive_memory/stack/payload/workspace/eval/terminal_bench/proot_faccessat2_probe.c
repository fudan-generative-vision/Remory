#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

/* PRoot 5.3.1 leaves faccessat2 unhandled. The host kernel then checks the
 * host path instead of the guest file, so bash test -x fails even though
 * open/stat/access/exec on that same guest file succeed. Never substitute
 * access(2) in this probe: it would conceal the regression. */
static void check(int dirfd, const char *path, int mode, int expected_errno) {
    errno = 0;
    long result = syscall(SYS_faccessat2, dirfd, path, mode, AT_EACCESS);
    int actual_errno = errno;
    if ((expected_errno == 0 && result != 0) ||
        (expected_errno != 0 && (result != -1 || actual_errno != expected_errno))) {
        fprintf(stderr, "faccessat2(%s, mode=%d): result=%ld errno=%d (%s), "
                "expected errno=%d\n", path, mode, result, actual_errno,
                strerror(actual_errno), expected_errno);
        exit(2);
    }
}

int main(void) {
    /* Unique guest-only directory also prevents a coincidental host file from
     * making an untranslated syscall appear to work. */
    char directory[] = "/tmp/harbor-proot-faccessat2-XXXXXX";
    if (mkdtemp(directory) == NULL || chdir(directory) != 0) {
        perror("create guest directory");
        return 3;
    }
    int dirfd = open(".", O_RDONLY | O_DIRECTORY);
    int fd = open("executable", O_CREAT | O_EXCL | O_WRONLY, 0700);
    if (dirfd < 0 || fd < 0 || fchmod(fd, 0755) != 0 || close(fd) != 0) {
        perror("create guest executable");
        return 4;
    }
    char absolute[sizeof(directory) + 32];
    snprintf(absolute, sizeof(absolute), "%s/executable", directory);
    check(AT_FDCWD, absolute, X_OK, 0);
    check(AT_FDCWD, "executable", X_OK, 0);
    check(dirfd, "executable", X_OK, 0);
    check(dirfd, "missing", F_OK, ENOENT);

    /* Root may ignore R_OK/W_OK mode bits but must still require at least one
     * execute bit for X_OK on a regular file. Test the failure direction too. */
    if (chmod("executable", 0644) != 0) {
        perror("clear executable bits");
        return 5;
    }
    check(AT_FDCWD, absolute, X_OK, EACCES);
    check(dirfd, "executable", X_OK, EACCES);
    close(dirfd);
    puts("proot-faccessat2-ready");
    return 0;
}
