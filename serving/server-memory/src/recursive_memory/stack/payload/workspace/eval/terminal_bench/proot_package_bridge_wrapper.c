#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#ifndef PACKAGE_BRIDGE_SOCKET_PATH
#define PACKAGE_BRIDGE_SOCKET_PATH "/run/harbor-package-bridge/control.sock"
#endif

#define MAX_REQUEST_BYTES (8U * 1024U * 1024U)
#define MAX_RESPONSE_BYTES (64U * 1024U * 1024U)
#define MAX_STRING_BYTES (1024U * 1024U)
#define MAX_ARGUMENTS 512U
#define MAX_ENVIRONMENT_ENTRIES 32U

static const unsigned char REQUEST_MAGIC[8] = {'H', 'P', 'K', 'G', 'B', 'R', '1', 0};
static const unsigned char RESPONSE_MAGIC[8] = {'H', 'P', 'K', 'G', 'R', 'S', '1', 0};

struct buffer {
    unsigned char *data;
    size_t len;
    size_t cap;
};

static const char *allowed_commands[] = {
    "apt",
    "apt-get",
    "apt-cache",
    "apt-mark",
    "dpkg",
    "dpkg-query",
    "dpkg-deb",
    "dpkg-divert",
    "dpkg-statoverride",
    "debconf-set-selections",
    "dpkg-reconfigure",
};

static const char *preserved_environment[] = {
    "DEBIAN_FRONTEND",
    "DEBIAN_PRIORITY",
    "DEBCONF_NONINTERACTIVE_SEEN",
    "APT_LISTCHANGES_FRONTEND",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
};

static void local_error(const char *stage) {
    int saved = errno;
    dprintf(STDERR_FILENO, "package bridge wrapper: %s: %s\n", stage, strerror(saved));
}

static int append_data(struct buffer *buffer, const void *data, size_t size, size_t maximum) {
    if (size == 0) return 0;
    if (buffer->len > maximum || size > maximum - buffer->len) {
        errno = EOVERFLOW;
        return -1;
    }
    size_t needed = buffer->len + size;
    if (needed > buffer->cap) {
        size_t next = buffer->cap ? buffer->cap : 4096U;
        while (next < needed) {
            if (next > maximum / 2U) {
                next = maximum;
                break;
            }
            next *= 2U;
        }
        unsigned char *grown = realloc(buffer->data, next);
        if (grown == NULL) return -1;
        buffer->data = grown;
        buffer->cap = next;
    }
    memcpy(buffer->data + buffer->len, data, size);
    buffer->len += size;
    return 0;
}

static int append_u16(struct buffer *buffer, uint16_t value) {
    uint16_t wire = htons(value);
    return append_data(buffer, &wire, sizeof(wire), MAX_REQUEST_BYTES);
}

static int append_u32(struct buffer *buffer, uint32_t value) {
    uint32_t wire = htonl(value);
    return append_data(buffer, &wire, sizeof(wire), MAX_REQUEST_BYTES);
}

static int append_blob(struct buffer *buffer, const void *data, size_t size) {
    if (size > UINT32_MAX || size > MAX_REQUEST_BYTES) {
        errno = EOVERFLOW;
        return -1;
    }
    if (append_u32(buffer, (uint32_t)size) != 0) return -1;
    return append_data(buffer, data, size, MAX_REQUEST_BYTES);
}

static int append_string(struct buffer *buffer, const char *value) {
    size_t size = strlen(value);
    if (size > MAX_STRING_BYTES) {
        errno = EOVERFLOW;
        return -1;
    }
    return append_blob(buffer, value, size);
}

static int write_full(int fd, const void *data, size_t size) {
    const unsigned char *cursor = data;
    while (size > 0) {
        ssize_t sent = write(fd, cursor, size);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (sent == 0) {
            errno = EPIPE;
            return -1;
        }
        cursor += sent;
        size -= (size_t)sent;
    }
    return 0;
}

static int read_full(int fd, void *data, size_t size) {
    unsigned char *cursor = data;
    while (size > 0) {
        ssize_t got = read(fd, cursor, size);
        if (got < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (got == 0) {
            errno = ECONNRESET;
            return -1;
        }
        cursor += got;
        size -= (size_t)got;
    }
    return 0;
}

static const char *command_basename(const char *value) {
    const char *slash = strrchr(value, '/');
    const char *name = slash ? slash + 1 : value;
    return name[0] ? name : NULL;
}

static int command_allowed(const char *name) {
    for (size_t index = 0; index < sizeof(allowed_commands) / sizeof(allowed_commands[0]); ++index) {
        if (strcmp(name, allowed_commands[index]) == 0) return 1;
    }
    return 0;
}

static int connect_broker(void) {
    if (strlen(PACKAGE_BRIDGE_SOCKET_PATH) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    int descriptor = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (descriptor < 0) return -1;
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, PACKAGE_BRIDGE_SOCKET_PATH, strlen(PACKAGE_BRIDGE_SOCKET_PATH) + 1U);
    if (connect(descriptor, (struct sockaddr *)&address, sizeof(address)) != 0) {
        int saved = errno;
        close(descriptor);
        errno = saved;
        return -1;
    }
    return descriptor;
}

static int collect_stdin(struct buffer *input) {
    if (isatty(STDIN_FILENO)) return 0;
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (flags < 0 || fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK) != 0) return -1;
    unsigned char chunk[65536];
    for (;;) {
        struct pollfd descriptor = {STDIN_FILENO, POLLIN | POLLHUP, 0};
        /* A non-TTY input is a real pipe/file protocol.  Wait for EOF rather
         * than silently truncating a delayed producer; the outer command
         * timeout remains the cancellation boundary and the 8 MiB cap remains
         * the memory boundary. */
        int ready = poll(&descriptor, 1, -1);
        if (ready < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (descriptor.revents & (POLLERR | POLLNVAL)) {
            errno = EIO;
            return -1;
        }
        for (;;) {
            ssize_t got = read(STDIN_FILENO, chunk, sizeof(chunk));
            if (got > 0) {
                if (append_data(input, chunk, (size_t)got, MAX_REQUEST_BYTES) != 0) return -1;
                continue;
            }
            if (got == 0) return 0;
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            return -1;
        }
        if (descriptor.revents & POLLHUP) return 0;
    }
}

static int send_request(int socket_fd, int argc, char **argv, const char *cwd, const struct buffer *input) {
    struct buffer body = {0};
    if (argc < 1 || (unsigned int)argc > MAX_ARGUMENTS || append_u16(&body, (uint16_t)argc) != 0) {
        free(body.data);
        return -1;
    }
    for (int index = 0; index < argc; ++index) {
        if (append_string(&body, argv[index]) != 0) {
            free(body.data);
            return -1;
        }
    }
    if (append_string(&body, cwd) != 0) {
        free(body.data);
        return -1;
    }

    uint16_t environment_count = 0;
    for (size_t index = 0; index < sizeof(preserved_environment) / sizeof(preserved_environment[0]); ++index) {
        if (getenv(preserved_environment[index]) != NULL) environment_count++;
    }
    if (environment_count > MAX_ENVIRONMENT_ENTRIES || append_u16(&body, environment_count) != 0) {
        free(body.data);
        return -1;
    }
    for (size_t index = 0; index < sizeof(preserved_environment) / sizeof(preserved_environment[0]); ++index) {
        const char *value = getenv(preserved_environment[index]);
        if (value == NULL) continue;
        if (append_string(&body, preserved_environment[index]) != 0 || append_string(&body, value) != 0) {
            free(body.data);
            return -1;
        }
    }
    if (append_blob(&body, input->data, input->len) != 0 || body.len > UINT32_MAX) {
        free(body.data);
        return -1;
    }
    uint32_t body_size = htonl((uint32_t)body.len);
    int result = 0;
    if (write_full(socket_fd, REQUEST_MAGIC, sizeof(REQUEST_MAGIC)) != 0 ||
        write_full(socket_fd, &body_size, sizeof(body_size)) != 0 ||
        write_full(socket_fd, body.data, body.len) != 0) {
        result = -1;
    }
    free(body.data);
    return result;
}

static uint32_t read_wire_u32(const unsigned char *data) {
    uint32_t wire;
    memcpy(&wire, data, sizeof(wire));
    return ntohl(wire);
}

static int32_t read_wire_i32(const unsigned char *data) {
    return (int32_t)read_wire_u32(data);
}

static int receive_response(int socket_fd) {
    unsigned char header[12];
    if (read_full(socket_fd, header, sizeof(header)) != 0) return -1;
    if (memcmp(header, RESPONSE_MAGIC, sizeof(RESPONSE_MAGIC)) != 0) {
        errno = EPROTO;
        return -1;
    }
    uint32_t body_size = read_wire_u32(header + 8);
    if (body_size < 12U || body_size > MAX_RESPONSE_BYTES) {
        errno = EOVERFLOW;
        return -1;
    }
    unsigned char *body = malloc(body_size);
    if (body == NULL) return -1;
    if (read_full(socket_fd, body, body_size) != 0) {
        free(body);
        return -1;
    }
    int32_t return_code = read_wire_i32(body);
    uint32_t stdout_size = read_wire_u32(body + 4);
    uint32_t stderr_size = read_wire_u32(body + 8);
    if ((uint64_t)12U + stdout_size + stderr_size != body_size) {
        free(body);
        errno = EPROTO;
        return -1;
    }
    int result = 0;
    if (write_full(STDOUT_FILENO, body + 12, stdout_size) != 0 ||
        write_full(STDERR_FILENO, body + 12 + stdout_size, stderr_size) != 0) {
        result = -1;
    }
    free(body);
    if (result != 0) return -1;
    if (return_code < 0 || return_code > 255) return 126;
    return (int)return_code;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    if (argc < 1 || argv == NULL || argv[0] == NULL) {
        dprintf(STDERR_FILENO, "package bridge wrapper: invalid argv\n");
        return 126;
    }
    const char *name = command_basename(argv[0]);
    if (name == NULL || !command_allowed(name)) {
        dprintf(STDERR_FILENO, "package bridge wrapper: unsupported invocation name: %s\n", name ? name : "<empty>");
        return 126;
    }
    char *cwd = getcwd(NULL, 0);
    if (cwd == NULL) {
        local_error("getcwd");
        return 126;
    }
    struct buffer input = {0};
    if (collect_stdin(&input) != 0) {
        local_error("bounded stdin collection");
        free(input.data);
        free(cwd);
        return 126;
    }
    int socket_fd = connect_broker();
    if (socket_fd < 0) {
        local_error("connect broker");
        free(input.data);
        free(cwd);
        return 126;
    }
    if (send_request(socket_fd, argc, argv, cwd, &input) != 0) {
        local_error("send bounded request");
        close(socket_fd);
        free(input.data);
        free(cwd);
        return 126;
    }
    free(input.data);
    free(cwd);
    int return_code = receive_response(socket_fd);
    if (return_code < 0) {
        local_error("receive bounded response");
        close(socket_fd);
        return 126;
    }
    close(socket_fd);
    return return_code;
}
