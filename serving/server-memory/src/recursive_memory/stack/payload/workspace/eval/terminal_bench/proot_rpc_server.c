#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_FIELD_SIZE (128U * 1024U * 1024U)
#define MAX_TRACKED_CHILDREN 4096U
#define REQUEST_DECISION_COMMIT ((unsigned char)'C')
#define REQUEST_DECISION_ABORT ((unsigned char)'A')
#define REQUEST_CLEANUP_ACK ((unsigned char)'A')
#define REQUEST_COMMIT_ACK ((unsigned char)'C')
#define SUPERVISOR_READY_ATTESTATION ((unsigned char)'R')
#define SUPERVISOR_STATUS_ATTESTATION ((unsigned char)'S')
#define SUPERVISOR_CLEANUP_ATTESTATION ((unsigned char)'K')
#define SUPERVISOR_IDENTITY_GATE_RELEASE ((unsigned char)'G')
#define REQUEST_SUPERVISOR_DEADLINE_MS 5000LL

struct buffer {
    unsigned char *data;
    size_t len;
    size_t cap;
};

static char fatal_reason[512];
static volatile sig_atomic_t supervisor_signal;

static void record_failure(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(fatal_reason, sizeof(fatal_reason), format, arguments);
    va_end(arguments);
}

static void record_errno_failure(const char *stage) {
    int saved_errno = errno;
    record_failure("%s: errno=%d (%s)", stage, saved_errno, strerror(saved_errno));
}

static int read_full(int fd, void *data, size_t size) {
    unsigned char *cursor = data;
    while (size > 0) {
        ssize_t got = read(fd, cursor, size);
        if (got == 0) return 0;
        if (got < 0) {
            if (errno == EINTR) continue;
            record_errno_failure("read_full");
            return -1;
        }
        cursor += got;
        size -= (size_t)got;
    }
    return 1;
}

static int write_full(int fd, const void *data, size_t size) {
    const unsigned char *cursor = data;
    while (size > 0) {
        ssize_t sent = write(fd, cursor, size);
        if (sent < 0) {
            if (errno == EINTR) continue;
            record_errno_failure("write_full");
            return -1;
        }
        cursor += sent;
        size -= (size_t)sent;
    }
    return 0;
}

static int read_u32(uint32_t *value) {
    uint32_t wire;
    int status = read_full(STDIN_FILENO, &wire, sizeof(wire));
    if (status <= 0) return status;
    *value = ntohl(wire);
    return 1;
}

static int write_u32(uint32_t value) {
    uint32_t wire = htonl(value);
    return write_full(STDOUT_FILENO, &wire, sizeof(wire));
}

static char *read_string(void) {
    uint32_t size;
    if (read_u32(&size) <= 0) {
        if (fatal_reason[0] == '\0') record_failure("truncated string length");
        return NULL;
    }
    if (size > MAX_FIELD_SIZE) {
        record_failure("request string exceeds MAX_FIELD_SIZE: %u", size);
        return NULL;
    }
    char *value = malloc((size_t)size + 1);
    if (value == NULL) {
        record_errno_failure("malloc request string");
        return NULL;
    }
    if (read_full(STDIN_FILENO, value, size) <= 0) {
        if (fatal_reason[0] == '\0') record_failure("truncated request string");
        free(value);
        return NULL;
    }
    value[size] = '\0';
    return value;
}

static int append_buffer(struct buffer *buffer, const void *data, size_t size) {
    if (size == 0) return 0;
    if (buffer->len + size > MAX_FIELD_SIZE) {
        record_failure("command output exceeds MAX_FIELD_SIZE");
        return -1;
    }
    if (buffer->len + size > buffer->cap) {
        size_t next = buffer->cap ? buffer->cap : 4096;
        while (next < buffer->len + size) next *= 2;
        unsigned char *grown = realloc(buffer->data, next);
        if (grown == NULL) {
            record_errno_failure("realloc command output");
            return -1;
        }
        buffer->data = grown;
        buffer->cap = next;
    }
    memcpy(buffer->data + buffer->len, data, size);
    buffer->len += size;
    return 0;
}

static long long monotonic_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (long long)value.tv_sec * 1000LL + value.tv_nsec / 1000000LL;
}

static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        record_errno_failure("fcntl nonblocking pipe");
        return -1;
    }
    return 0;
}

static int drain_fd(int fd, struct buffer *buffer, int *open_flag) {
    unsigned char chunk[65536];
    size_t budget = 1024U * 1024U;
    while (budget > 0) {
        size_t requested = budget < sizeof(chunk) ? budget : sizeof(chunk);
        ssize_t got = read(fd, chunk, requested);
        if (got > 0) {
            int append_status = append_buffer(buffer, chunk, (size_t)got);
            if (append_status != 0) return append_status;
            budget -= (size_t)got;
            continue;
        }
        if (got == 0) {
            close(fd);
            *open_flag = 0;
            return 0;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        close(fd);
        *open_flag = 0;
        record_errno_failure("read command output pipe");
        return -1;
    }
    return 0;
}

static int drain_fd_snapshot_and_close(int fd, struct buffer *buffer, int *open_flag) {
    int available = 0;
    if (ioctl(fd, FIONREAD, &available) != 0) {
        record_errno_failure("ioctl FIONREAD command output pipe");
        close(fd);
        *open_flag = 0;
        return -1;
    }
    unsigned char chunk[65536];
    size_t remaining = available > 0 ? (size_t)available : 0;
    while (remaining > 0) {
        size_t requested = remaining < sizeof(chunk) ? remaining : sizeof(chunk);
        ssize_t got = read(fd, chunk, requested);
        if (got > 0) {
            int append_status = append_buffer(buffer, chunk, (size_t)got);
            if (append_status != 0) {
                close(fd);
                *open_flag = 0;
                return append_status;
            }
            remaining -= (size_t)got;
            continue;
        }
        if (got < 0 && errno == EINTR) continue;
        if (got < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            record_errno_failure("read command output snapshot");
            close(fd);
            *open_flag = 0;
            return -1;
        }
        break;
    }
    close(fd);
    *open_flag = 0;
    return 0;
}

static int send_response(int return_code, struct buffer *out, struct buffer *err) {
    if (out->len > UINT32_MAX || err->len > UINT32_MAX) {
        record_failure("response length exceeds uint32 framing");
        return -1;
    }
    if (write_u32((uint32_t)(int32_t)return_code) != 0) return -1;
    if (write_u32((uint32_t)out->len) != 0) return -1;
    if (write_u32((uint32_t)err->len) != 0) return -1;
    if (write_full(STDOUT_FILENO, out->data, out->len) != 0) return -1;
    if (write_full(STDOUT_FILENO, err->data, err->len) != 0) return -1;
    return 0;
}

static int direct_children(pid_t *children, size_t capacity, size_t *count_out) {
#ifdef PROOT_RPC_TEST_CLEANUP_GATE
    const char *cleanup_gate = getenv("PROOT_RPC_TEST_CLEANUP_GATE");
    if (cleanup_gate != NULL && cleanup_gate[0] != '\0' &&
        access(cleanup_gate, F_OK) == 0) {
        errno = EIO;
        record_errno_failure("test-gated direct-children proc file");
        return -1;
    }
#endif
    char path[128];
    snprintf(path, sizeof(path), "/proc/self/task/%ld/children", (long)getpid());
    FILE *stream = fopen(path, "r");
    if (stream == NULL) {
        record_errno_failure("open direct-children proc file");
        return -1;
    }
    size_t count = 0;
    long value;
    while (fscanf(stream, "%ld", &value) == 1) {
        if (value <= 1) continue;
        if (count == capacity) {
            fclose(stream);
            record_failure("direct-child snapshot exceeds capacity %zu", capacity);
            return -1;
        }
        children[count++] = (pid_t)value;
    }
    if (ferror(stream)) {
        record_errno_failure("read direct-children proc file");
        fclose(stream);
        return -1;
    }
    fclose(stream);
    *count_out = count;
    return 0;
}

static void reap_exited_children(void) {
    int status;
    while (waitpid(-1, &status, WNOHANG) > 0) {
    }
}

static int read_operation_with_reaping(unsigned char *operation) {
    for (;;) {
        struct pollfd input = {STDIN_FILENO, POLLIN | POLLHUP, 0};
        int ready = poll(&input, 1, 100);
        if (ready == 0) {
            /* Persistent services committed by earlier requests are direct
             * children of this subreaper. Reap short-lived ones even while no
             * new protocol request is arriving. */
            reap_exited_children();
            continue;
        }
        if (ready < 0) {
            if (errno == EINTR) {
                reap_exited_children();
                continue;
            }
            record_errno_failure("poll operation input");
            return -1;
        }
        if (input.revents & (POLLERR | POLLNVAL)) {
            record_failure("operation input pipe reported an error");
            return -1;
        }
        return read_full(STDIN_FILENO, operation, 1);
    }
}

static int terminate_descendants(void) {
    const int signals[] = {SIGTERM, SIGKILL};
    const struct timespec pause = {0, 20 * 1000 * 1000};
    for (size_t phase = 0; phase < sizeof(signals) / sizeof(signals[0]); ++phase) {
        for (int attempt = 0; attempt < 25; ++attempt) {
            reap_exited_children();
            pid_t children[MAX_TRACKED_CHILDREN];
            size_t count = 0;
            if (direct_children(children, MAX_TRACKED_CHILDREN, &count) != 0) return -1;
            if (count == 0) break;
            pid_t own_group = getpgrp();
            for (size_t index = 0; index < count; ++index) {
                pid_t child = children[index];
                pid_t group = getpgid(child);
                /* Only group-signal a session/group led by this direct child.
                 * A foreign process group must never be targeted by a numeric
                 * PGID snapshot; iterative direct-child cleanup plus subreaper
                 * adoption handles every other shape without crossing owners. */
                if (group == child && group != own_group) {
                    kill(-group, signals[phase]);
                }
                kill(child, signals[phase]);
            }
            nanosleep(&pause, NULL);
        }
    }
    reap_exited_children();
    pid_t survivors[MAX_TRACKED_CHILDREN];
    size_t survivor_count = 0;
    if (direct_children(survivors, MAX_TRACKED_CHILDREN, &survivor_count) != 0) return -1;
    if (survivor_count != 0) {
        record_failure("descendant cleanup left %zu direct child processes", survivor_count);
        return -1;
    }
    return 0;
}

static int fatal_exit(int status, const char *fallback) {
    if (fatal_reason[0] == '\0') record_failure("%s", fallback);
    char original_reason[sizeof(fatal_reason)];
    snprintf(original_reason, sizeof(original_reason), "%s", fatal_reason);
    dprintf(STDERR_FILENO, "proot_rpc_server fatal: %s\n", original_reason);
    while (terminate_descendants() != 0) {
        /* Exiting the only host-side subreaper without a clean child scan can
         * orphan a stopped or setsid descendant to the host init.  Remain as a
         * quarantined owner and retry; Python will preserve the rootfs if this
         * cannot converge within its shutdown deadline. */
        dprintf(
            STDERR_FILENO,
            "proot_rpc_server cleanup incomplete; owner quarantined: %s\n",
            fatal_reason[0] == '\0' ? "unknown cleanup failure" : fatal_reason);
        const struct timespec pause = {0, 100 * 1000 * 1000};
        nanosleep(&pause, NULL);
    }
    return status;
}

static void record_supervisor_signal(int signum) {
    supervisor_signal = signum;
}

static void restore_default_signal_handlers(void) {
    const int signals[] = {SIGHUP, SIGINT, SIGPIPE, SIGTERM};
    struct sigaction action = {0};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    for (size_t index = 0; index < sizeof(signals) / sizeof(signals[0]); ++index) {
        sigaction(signals[index], &action, NULL);
    }
}

static int exit_like_child(int wait_status) {
    if (WIFEXITED(wait_status)) return WEXITSTATUS(wait_status);
    if (WIFSIGNALED(wait_status)) {
        int signum = WTERMSIG(wait_status);
        restore_default_signal_handlers();
        raise(signum);
        return 128 + signum;
    }
    return 1;
}

static int supervise_command(
    char **command_argv, int cleanup_ack_fd, int identity_gate_fd
) {
    const int strict_control_mode = cleanup_ack_fd >= 0 && identity_gate_fd >= 0;
    fatal_reason[0] = '\0';
    if (command_argv[0] == NULL) {
        record_failure("--supervise requires a command");
        return fatal_exit(64, "missing supervised command");
    }
    if (cleanup_ack_fd >= 0) {
        int flags = fcntl(cleanup_ack_fd, F_GETFD, 0);
        if (flags < 0 ||
            fcntl(cleanup_ack_fd, F_SETFD, flags | FD_CLOEXEC) < 0) {
            record_errno_failure("supervisor cleanup attestation fd setup");
            return fatal_exit(4, "supervisor could not protect cleanup attestation fd");
        }
    }
    if (identity_gate_fd >= 0) {
        int flags = fcntl(identity_gate_fd, F_GETFD, 0);
        if (flags < 0 ||
            fcntl(identity_gate_fd, F_SETFD, flags | FD_CLOEXEC) < 0) {
            record_errno_failure("supervisor identity gate fd setup");
            return fatal_exit(4, "supervisor could not protect identity gate fd");
        }
    }
    if (prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) {
        record_errno_failure("supervisor prctl PR_SET_CHILD_SUBREAPER");
        return fatal_exit(4, "supervisor could not become child subreaper");
    }

    struct sigaction action = {0};
    action.sa_handler = record_supervisor_signal;
    sigemptyset(&action.sa_mask);
    const int signals[] = {SIGHUP, SIGINT, SIGTERM};
    for (size_t index = 0; index < sizeof(signals) / sizeof(signals[0]); ++index) {
        if (sigaction(signals[index], &action, NULL) != 0) {
            record_errno_failure("supervisor sigaction");
            return fatal_exit(4, "supervisor could not install signal handlers");
        }
    }

    pid_t primary = fork();
    if (primary < 0) {
        record_errno_failure("supervisor fork");
        return fatal_exit(4, "supervisor could not fork command");
    }
    if (primary == 0) {
        restore_default_signal_handlers();
        if (cleanup_ack_fd >= 0) close(cleanup_ack_fd);
        if (identity_gate_fd >= 0) {
            unsigned char release = 0;
            int gate_status = read_full(identity_gate_fd, &release, 1);
            close(identity_gate_fd);
            if (gate_status != 1 || release != SUPERVISOR_IDENTITY_GATE_RELEASE) {
                dprintf(
                    STDERR_FILENO,
                    "proot_rpc_server supervisor identity gate was not released\n");
                _exit(126);
            }
        }
        execvp(command_argv[0], command_argv);
        dprintf(STDERR_FILENO, "proot_rpc_server supervisor exec failed for %s: %s\n",
                command_argv[0], strerror(errno));
        _exit(127);
    }

    /* The supervised command has the same uid as this host-side owner.  Without
     * a non-dumpable owner it could reopen the protected control pipe through
     * /proc/$PPID/fd and forge the cleanup record.  The primary is still blocked
     * on identity_gate_fd here, so a failure cannot race untrusted execution. */
    if (strict_control_mode && prctl(PR_SET_DUMPABLE, 0) != 0) {
        record_errno_failure("supervisor prctl PR_SET_DUMPABLE");
        close(identity_gate_fd);
        kill(primary, SIGKILL);
        (void)waitpid(primary, NULL, 0);
        return fatal_exit(4, "supervisor could not protect its control channel");
    }

    if (identity_gate_fd >= 0) close(identity_gate_fd);

    if (cleanup_ack_fd >= 0) {
        unsigned char ready = SUPERVISOR_READY_ATTESTATION;
        if (write_full(cleanup_ack_fd, &ready, 1) != 0) {
            kill(primary, SIGKILL);
            (void)waitpid(primary, NULL, 0);
            return fatal_exit(4, "supervisor could not attest readiness");
        }
    }

    /* The primary owns the protocol pipes.  The supervisor must not keep either
     * endpoint alive after an abrupt primary exit. */
    close(STDIN_FILENO);
    close(STDOUT_FILENO);

    int wait_status = 0;
    int forwarded_signal = 0;
    long long hard_kill_at = 0;
    for (;;) {
        pid_t waited = waitpid(primary, &wait_status, WNOHANG);
        if (waited == primary) break;
        if (waited < 0) {
            if (errno == EINTR) continue;
            record_errno_failure("supervisor waitpid primary");
            (void)terminate_descendants();
            return fatal_exit(4, "supervisor lost its primary child");
        }
        if (supervisor_signal != 0 && forwarded_signal == 0) {
            forwarded_signal = supervisor_signal;
            kill(primary, forwarded_signal);
            hard_kill_at = monotonic_ms() + 500;
        }
        if (forwarded_signal != 0 && monotonic_ms() >= hard_kill_at) {
            kill(primary, SIGKILL);
        }
        const struct timespec pause = {0, 10 * 1000 * 1000};
        nanosleep(&pause, NULL);
    }

    /* As a host-side subreaper, the supervisor adopts descendants even when the
     * PRoot tracer and in-guest RPC helper die together.  This closes the setsid
     * escape that a process-group-only cleanup cannot cover. */
    if (terminate_descendants() != 0) {
        return fatal_exit(4, "supervisor descendant cleanup failed");
    }
    if (cleanup_ack_fd >= 0) {
        if (strict_control_mode) {
            unsigned char attestation[6];
            uint32_t encoded_wait_status = htonl((uint32_t)wait_status);
            attestation[0] = SUPERVISOR_STATUS_ATTESTATION;
            memcpy(
                &attestation[1], &encoded_wait_status,
                sizeof(encoded_wait_status));
            attestation[5] = SUPERVISOR_CLEANUP_ATTESTATION;
            if (write_full(cleanup_ack_fd, attestation, sizeof(attestation)) != 0) {
                close(cleanup_ack_fd);
                return 4;
            }
        } else {
            unsigned char attestation = SUPERVISOR_CLEANUP_ATTESTATION;
            if (write_full(cleanup_ack_fd, &attestation, 1) != 0) {
                close(cleanup_ack_fd);
                return 4;
            }
        }
        close(cleanup_ack_fd);
    }
    /* In strict control mode Python accepts cleanup only when this owner exits
     * normally and the separate status frame is complete.  Preserve the legacy
     * child-like exit behavior for callers that do not use the identity gate. */
    if (strict_control_mode) return 0;
    return exit_like_child(wait_status);
}

static int terminate_request_primary(pid_t primary, int *wait_status) {
    const struct timespec pause = {0, 10 * 1000 * 1000};
    kill(-primary, SIGTERM);
    kill(primary, SIGTERM);
    long long hard_kill_at = monotonic_ms() + 500;
    for (;;) {
        pid_t waited = waitpid(primary, wait_status, WNOHANG);
        if (waited == primary) return 0;
        if (waited < 0) {
            if (errno == EINTR) continue;
            record_errno_failure("request supervisor waitpid primary");
            return -1;
        }
        if (monotonic_ms() >= hard_kill_at) {
            kill(-primary, SIGKILL);
            kill(primary, SIGKILL);
        }
        nanosleep(&pause, NULL);
    }
}

static int request_supervisor_main(
    int out_read, int out_write, int err_read, int err_write,
    int decision_read, int status_write, int ack_write,
    const char *cwd, const char *command, uint32_t env_count,
    char **keys, char **values
) {
    fatal_reason[0] = '\0';
    if (prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) {
        record_errno_failure("request supervisor prctl PR_SET_CHILD_SUBREAPER");
        return fatal_exit(4, "request supervisor could not become child subreaper");
    }
    if (set_nonblocking(decision_read) != 0) {
        return fatal_exit(4, "request supervisor decision pipe setup failed");
    }

    pid_t primary = fork();
    if (primary < 0) {
        record_errno_failure("request supervisor fork primary");
        return fatal_exit(4, "request supervisor could not fork command");
    }
    if (primary == 0) {
        restore_default_signal_handlers();
        close(decision_read);
        close(status_write);
        close(ack_write);
        if (setsid() < 0) {
            dprintf(STDERR_FILENO, "Unable to create command session: %s\n", strerror(errno));
            _exit(127);
        }
        int null_fd = open("/dev/null", O_RDONLY);
        if (null_fd >= 0) {
            dup2(null_fd, STDIN_FILENO);
            close(null_fd);
        }
        dup2(out_write, STDOUT_FILENO);
        dup2(err_write, STDERR_FILENO);
        close(out_read);
        close(out_write);
        close(err_read);
        close(err_write);
        for (uint32_t i = 0; i < env_count; ++i) setenv(keys[i], values[i], 1);
        if (chdir(cwd) != 0) {
            dprintf(STDERR_FILENO, "Unable to chdir to %s: %s\n", cwd, strerror(errno));
            _exit(127);
        }
        execl("/bin/bash", "bash", "-lc", command, (char *)NULL);
        dprintf(STDERR_FILENO, "Unable to execute /bin/bash: %s\n", strerror(errno));
        _exit(127);
    }

    close(out_read);
    close(out_write);
    close(err_read);
    close(err_write);
    int primary_done = 0;
    int primary_status = 0;
    int status_sent = 0;
    const struct timespec pause = {0, 10 * 1000 * 1000};
    for (;;) {
        unsigned char decision = 0;
        ssize_t decision_size = read(decision_read, &decision, 1);
        if (decision_size == 1) {
            if (decision == REQUEST_DECISION_COMMIT && primary_done) {
                if (write_full(ack_write, &decision, 1) != 0) return 4;
                return 0;
            }
            if (decision != REQUEST_DECISION_ABORT) {
                record_failure("invalid or premature request decision: %u", decision);
            }
            if (!primary_done) {
                if (terminate_request_primary(primary, &primary_status) != 0) return 4;
                primary_done = 1;
            }
            /* ABORT does not need to publish the primary status.  The outer
             * process may already have closed the status pipe after deciding on
             * timeout/overflow; cleanup and its ACK are the authoritative gate. */
            if (terminate_descendants() != 0) return 4;
            unsigned char ack = REQUEST_CLEANUP_ACK;
            if (write_full(ack_write, &ack, 1) != 0) return 4;
            return decision == REQUEST_DECISION_ABORT ? 0 : 4;
        }
        if (decision_size == 0) {
            if (!primary_done && terminate_request_primary(primary, &primary_status) != 0) {
                return 4;
            }
            if (terminate_descendants() != 0) return 4;
            return 4;
        }
        if (decision_size < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
            record_errno_failure("request supervisor read decision");
            if (!primary_done) (void)terminate_request_primary(primary, &primary_status);
            (void)terminate_descendants();
            return 4;
        }

        if (!primary_done) {
            pid_t waited = waitpid(primary, &primary_status, WNOHANG);
            if (waited == primary) {
                primary_done = 1;
            } else if (waited < 0 && errno != EINTR) {
                record_errno_failure("request supervisor waitpid primary");
                (void)terminate_descendants();
                return 4;
            }
            if (primary_done && !status_sent) {
                if (write_full(status_write, &primary_status, sizeof(primary_status)) != 0) {
                    (void)terminate_descendants();
                    return 4;
                }
                status_sent = 1;
            }
        }
        nanosleep(&pause, NULL);
    }
}

static int finish_request_supervisor(
    pid_t supervisor, int decision_write, int ack_read, unsigned char decision
) {
    unsigned char expected = decision == REQUEST_DECISION_COMMIT
        ? REQUEST_COMMIT_ACK : REQUEST_CLEANUP_ACK;
    if (write_full(decision_write, &decision, 1) != 0) {
        record_failure("request supervisor decision write failed");
        close(decision_write);
        close(ack_read);
        return -1;
    }
    close(decision_write);
    if (set_nonblocking(ack_read) != 0) {
        close(ack_read);
        return -1;
    }

    long long deadline = monotonic_ms() + REQUEST_SUPERVISOR_DEADLINE_MS;
    int supervisor_status = 0;
    int supervisor_done = 0;
    int ack_seen = 0;
    for (;;) {
        if (!ack_seen) {
            unsigned char ack = 0;
            ssize_t got = read(ack_read, &ack, 1);
            if (got == 1) {
                if (ack != expected) {
                    record_failure("request supervisor returned invalid ACK: %u", ack);
                    close(ack_read);
                    return -1;
                }
                ack_seen = 1;
            } else if (got == 0) {
                record_failure("request supervisor ACK pipe closed early");
                close(ack_read);
                return -1;
            } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                record_errno_failure("read request supervisor ACK");
                close(ack_read);
                return -1;
            }
        }
        if (!supervisor_done) {
            pid_t waited = waitpid(supervisor, &supervisor_status, WNOHANG);
            if (waited == supervisor) {
                supervisor_done = 1;
            } else if (waited < 0 && errno != EINTR) {
                record_errno_failure("waitpid request supervisor");
                close(ack_read);
                return -1;
            }
        }
        if (ack_seen && supervisor_done) {
            close(ack_read);
            if (!WIFEXITED(supervisor_status) || WEXITSTATUS(supervisor_status) != 0) {
                record_failure("request supervisor exited abnormally after ACK");
                return -1;
            }
            return 0;
        }
        if (monotonic_ms() >= deadline) {
            record_failure("request supervisor cleanup/commit ACK deadline expired");
            close(ack_read);
            return -1;
        }
        const struct timespec pause = {0, 10 * 1000 * 1000};
        nanosleep(&pause, NULL);
    }
}


static void close_pipe_pair(int pipe_pair[2]) {
    for (size_t index = 0; index < 2; ++index) {
        if (pipe_pair[index] >= 0) {
            close(pipe_pair[index]);
            pipe_pair[index] = -1;
        }
    }
}

static int execute_request(uint32_t timeout_ms) {
    fatal_reason[0] = '\0';
    char *cwd = NULL;
    char *command = NULL;
    char **keys = NULL;
    char **values = NULL;
    uint32_t env_count = 0;
    struct buffer out = {0}, err = {0};
    int result = -1;
    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    int decision_pipe[2] = {-1, -1};
    int status_pipe[2] = {-1, -1};
    int ack_pipe[2] = {-1, -1};
    pid_t supervisor = -1;

    cwd = read_string();
    command = read_string();
    if (cwd == NULL || command == NULL || read_u32(&env_count) <= 0 ||
        env_count > 65536) {
        if (fatal_reason[0] == '\0') record_failure("invalid request environment count");
        goto cleanup;
    }

    keys = calloc(env_count, sizeof(char *));
    values = calloc(env_count, sizeof(char *));
    if (keys == NULL || values == NULL) {
        record_errno_failure("calloc request environment");
        goto cleanup;
    }
    for (uint32_t index = 0; index < env_count; ++index) {
        keys[index] = read_string();
        values[index] = read_string();
        if (keys[index] == NULL || values[index] == NULL) {
            if (fatal_reason[0] == '\0') {
                record_failure("truncated request environment at index %u", index);
            }
            goto cleanup;
        }
    }

    if (pipe2(out_pipe, O_CLOEXEC) != 0 || pipe2(err_pipe, O_CLOEXEC) != 0 ||
        pipe2(decision_pipe, O_CLOEXEC) != 0 ||
        pipe2(status_pipe, O_CLOEXEC) != 0 ||
        pipe2(ack_pipe, O_CLOEXEC) != 0) {
        record_errno_failure("create request supervisor pipes");
        goto cleanup;
    }
    supervisor = fork();
    if (supervisor < 0) {
        record_errno_failure("fork request supervisor");
        goto cleanup;
    }
    if (supervisor == 0) {
        close(decision_pipe[1]);
        close(status_pipe[0]);
        close(ack_pipe[0]);
        int supervisor_result = request_supervisor_main(
            out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1], decision_pipe[0],
            status_pipe[1], ack_pipe[1], cwd, command, env_count, keys, values);
        _exit(supervisor_result);
    }

    close(out_pipe[1]);
    out_pipe[1] = -1;
    close(err_pipe[1]);
    err_pipe[1] = -1;
    close(decision_pipe[0]);
    decision_pipe[0] = -1;
    close(status_pipe[1]);
    status_pipe[1] = -1;
    close(ack_pipe[1]);
    ack_pipe[1] = -1;

    if (set_nonblocking(out_pipe[0]) != 0 || set_nonblocking(err_pipe[0]) != 0 ||
        set_nonblocking(status_pipe[0]) != 0) {
        (void)finish_request_supervisor(
            supervisor, decision_pipe[1], ack_pipe[0], REQUEST_DECISION_ABORT);
        decision_pipe[1] = -1;
        ack_pipe[0] = -1;
        supervisor = -1;
        goto cleanup;
    }

    int out_open = 1;
    int err_open = 1;
    int primary_done = 0;
    int wait_status = 0;
    int timed_out = 0;
    int request_failed = 0;
    int supervisor_reaped = 0;
    long long started = monotonic_ms();

    while (!primary_done) {
        ssize_t status_size = read(status_pipe[0], &wait_status, sizeof(wait_status));
        if (status_size == (ssize_t)sizeof(wait_status)) {
            primary_done = 1;
        } else if (status_size == 0) {
            record_failure("request supervisor status pipe closed before primary status");
            request_failed = 1;
            break;
        } else if (status_size > 0) {
            record_failure("request supervisor returned truncated primary status");
            request_failed = 1;
            break;
        } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
            record_errno_failure("read request supervisor primary status");
            request_failed = 1;
            break;
        }

        int supervisor_status = 0;
        pid_t supervisor_waited = waitpid(supervisor, &supervisor_status, WNOHANG);
        if (supervisor_waited == supervisor) {
            supervisor_reaped = 1;
            record_failure("request supervisor exited before a COMMIT/ABORT decision");
            request_failed = 1;
            break;
        }
        if (supervisor_waited < 0 && errno != EINTR) {
            record_errno_failure("waitpid request supervisor before decision");
            request_failed = 1;
            break;
        }

        if (primary_done) {
            if (out_open &&
                drain_fd_snapshot_and_close(out_pipe[0], &out, &out_open) != 0) {
                request_failed = 1;
            }
            if (err_open &&
                drain_fd_snapshot_and_close(err_pipe[0], &err, &err_open) != 0) {
                request_failed = 1;
            }
            break;
        }

        struct pollfd poll_fds[3];
        nfds_t count = 0;
        if (out_open) {
            poll_fds[count++] = (struct pollfd){out_pipe[0], POLLIN | POLLHUP, 0};
        }
        if (err_open) {
            poll_fds[count++] = (struct pollfd){err_pipe[0], POLLIN | POLLHUP, 0};
        }
        poll_fds[count++] = (struct pollfd){status_pipe[0], POLLIN | POLLHUP, 0};
        int poll_status = poll(poll_fds, count, 50);
        if (poll_status < 0 && errno != EINTR) {
            record_errno_failure("poll request supervisor pipes");
            request_failed = 1;
            break;
        }
        if (out_open && drain_fd(out_pipe[0], &out, &out_open) != 0) {
            request_failed = 1;
            break;
        }
        if (err_open && drain_fd(err_pipe[0], &err, &err_open) != 0) {
            request_failed = 1;
            break;
        }
        if (timeout_ms != 0 && monotonic_ms() - started >= timeout_ms) {
            timed_out = 1;
            break;
        }
    }

    if (out_open) {
        if (timed_out) (void)drain_fd(out_pipe[0], &out, &out_open);
        if (out_open) close(out_pipe[0]);
    }
    out_pipe[0] = -1;
    if (err_open) {
        if (timed_out) (void)drain_fd(err_pipe[0], &err, &err_open);
        if (err_open) close(err_pipe[0]);
    }
    err_pipe[0] = -1;
    close(status_pipe[0]);
    status_pipe[0] = -1;

    int output_overflow =
        request_failed && strcmp(fatal_reason, "command output exceeds MAX_FIELD_SIZE") == 0;
    if (timed_out || request_failed) {
        if (supervisor_reaped ||
            finish_request_supervisor(
                supervisor, decision_pipe[1], ack_pipe[0], REQUEST_DECISION_ABORT) != 0) {
            decision_pipe[1] = -1;
            ack_pipe[0] = -1;
            supervisor = -1;
            result = -1;
            goto cleanup;
        }
        decision_pipe[1] = -1;
        ack_pipe[0] = -1;
        supervisor = -1;

        if (output_overflow) {
            const char *message =
                "Command output exceeded the 128 MiB RPC field limit; command killed\n";
            free(err.data);
            err = (struct buffer){0};
            if (append_buffer(&err, message, strlen(message)) != 0) goto cleanup;
            result = send_response(125, &out, &err);
        } else if (timed_out && !request_failed) {
            const char *message = "Command timed out\n";
            if (append_buffer(&err, message, strlen(message)) != 0) goto cleanup;
            result = send_response(124, &out, &err);
        }
        goto cleanup;
    }

    int return_code = 1;
    if (WIFEXITED(wait_status)) {
        return_code = WEXITSTATUS(wait_status);
    } else if (WIFSIGNALED(wait_status)) {
        return_code = 128 + WTERMSIG(wait_status);
    }
    /* Never publish a successful command result before its process ownership is
     * committed.  If the response transport fails after COMMIT, main() retires
     * the whole RPC environment; no client can observe success for an
     * uncommitted request. */
    if (finish_request_supervisor(
            supervisor, decision_pipe[1], ack_pipe[0], REQUEST_DECISION_COMMIT) != 0) {
        result = -1;
    } else {
        result = send_response(return_code, &out, &err);
    }
    decision_pipe[1] = -1;
    ack_pipe[0] = -1;
    supervisor = -1;

cleanup:
    close_pipe_pair(out_pipe);
    close_pipe_pair(err_pipe);
    close_pipe_pair(decision_pipe);
    close_pipe_pair(status_pipe);
    close_pipe_pair(ack_pipe);
    if (supervisor > 0) {
        /* Any path that cannot complete the supervisor handshake is fatal to the
         * whole RPC process. main() will run global fail-closed cleanup. */
        record_failure("request supervisor was left without a completed decision");
        result = -1;
    }
    free(out.data);
    free(err.data);
    free(cwd);
    free(command);
    if (keys != NULL) {
        for (uint32_t index = 0; index < env_count; ++index) free(keys[index]);
    }
    if (values != NULL) {
        for (uint32_t index = 0; index < env_count; ++index) free(values[index]);
    }
    free(keys);
    free(values);
    return result;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    if (argc > 4 && strcmp(argv[1], "--supervise-control-fds") == 0) {
        char *cleanup_end = NULL;
        char *gate_end = NULL;
        errno = 0;
        long cleanup_ack_fd = strtol(argv[2], &cleanup_end, 10);
        long identity_gate_fd = strtol(argv[3], &gate_end, 10);
        if (errno != 0 || cleanup_end == argv[2] || *cleanup_end != '\0' ||
            gate_end == argv[3] || *gate_end != '\0' ||
            cleanup_ack_fd < 3 || cleanup_ack_fd > INT32_MAX ||
            identity_gate_fd < 3 || identity_gate_fd > INT32_MAX ||
            identity_gate_fd == cleanup_ack_fd) {
            dprintf(STDERR_FILENO, "invalid supervisor control fd\n");
            return 64;
        }
        return supervise_command(
            &argv[4], (int)cleanup_ack_fd, (int)identity_gate_fd);
    }
    if (argc > 3 && strcmp(argv[1], "--supervise-cleanup-fd") == 0) {
        char *end = NULL;
        errno = 0;
        long cleanup_ack_fd = strtol(argv[2], &end, 10);
        if (errno != 0 || end == argv[2] || *end != '\0' ||
            cleanup_ack_fd < 3 || cleanup_ack_fd > INT32_MAX) {
            dprintf(STDERR_FILENO, "invalid supervisor cleanup attestation fd\n");
            return 64;
        }
        return supervise_command(&argv[3], (int)cleanup_ack_fd, -1);
    }
    if (argc > 1 && strcmp(argv[1], "--supervise") == 0) {
        return supervise_command(&argv[2], -1, -1);
    }
    if (prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) {
        record_errno_failure("prctl PR_SET_CHILD_SUBREAPER");
        return fatal_exit(4, "could not become child subreaper");
    }
    for (;;) {
        /* COMMIT reparents a request's surviving services to this long-lived
         * subreaper. Reap services that have since exited before accepting the
         * next request, otherwise short-lived background jobs accumulate as
         * zombies until environment shutdown. */
        reap_exited_children();
        unsigned char operation;
        int status = read_operation_with_reaping(&operation);
        if (status == 0 || operation == 0) {
            if (terminate_descendants() != 0) {
                return fatal_exit(3, "shutdown descendant cleanup failed");
            }
            return 0;
        }
        if (status < 0) return fatal_exit(2, "could not read operation");
        if (operation != 1) {
            record_failure("unknown operation byte: %u", (unsigned int)operation);
            return fatal_exit(2, "unknown operation");
        }
        uint32_t timeout_ms;
        if (read_u32(&timeout_ms) <= 0) {
            return fatal_exit(2, "could not read request timeout");
        }
        /* A service may have exited while the protocol reader was blocked. */
        reap_exited_children();
        if (execute_request(timeout_ms) != 0) {
            return fatal_exit(3, "request execution failed");
        }
    }
}
