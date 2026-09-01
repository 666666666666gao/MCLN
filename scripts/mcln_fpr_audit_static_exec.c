#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef REVIEWED_SOURCE_SHA256
#error "REVIEWED_SOURCE_SHA256 must be supplied at build time"
#endif

#ifndef TRUST_ROOT
#define TRUST_ROOT "/root/mcln_fpr_audit_trust/v1"
#endif
#define EXECUTOR_PATH TRUST_ROOT "/mcln_fpr_audit_static_exec.x86_64"
#define SOURCE_PATH TRUST_ROOT "/mcln_fpr_audit_static_exec.c"
#define LAUNCHER_PATH TRUST_ROOT "/run_nr3d_fpr_tv_density_audit.sh"
#ifndef SHARED_GPU_LOCK
#define SHARED_GPU_LOCK \
    "/root/autodl-tmp/DATA_ROOT/output/network_v99/single_gpu.lock"
#endif
#define LAUNCHER_DESCRIPTOR 3

static void fail_with_code(const char *message, int code)
{
    fprintf(stderr, "static FPR-TV audit executor: %s\n", message);
    exit(code);
}

static void fail(const char *message)
{
    fail_with_code(message, 2);
}

static bool is_hex_sha256(const char *value)
{
    size_t index;

    if (value == NULL || strlen(value) != 64) {
        return false;
    }
    for (index = 0; index < 64; ++index) {
        const char current = value[index];
        if (!((current >= '0' && current <= '9') ||
              (current >= 'a' && current <= 'f'))) {
            return false;
        }
    }
    return true;
}

static void close_inherited_descriptors(void)
{
#ifdef SYS_close_range
    if (syscall(SYS_close_range, 3U, ~0U, 0U) == 0) {
        return;
    }
    if (errno != ENOSYS && errno != EINVAL) {
        fail("cannot close inherited descriptors with close_range");
    }
#endif
    {
        struct rlimit limits;
        rlim_t descriptor;

        if (getrlimit(RLIMIT_NOFILE, &limits) != 0) {
            fail("cannot obtain descriptor limit");
        }
        for (descriptor = 3; descriptor < limits.rlim_cur; ++descriptor) {
            (void)close((int)descriptor);
        }
    }
}

static void require_protected_metadata(
    int descriptor, bool directory, bool executable
)
{
    struct stat info;

    if (fstat(descriptor, &info) != 0) {
        fail("cannot stat protected artifact");
    }
    if ((directory && !S_ISDIR(info.st_mode)) ||
        (!directory && !S_ISREG(info.st_mode))) {
        fail("protected path has the wrong file type");
    }
    if (info.st_uid != 0 || info.st_gid != 0 ||
        (info.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        fail("protected path is not root-owned and non-writable");
    }
    if (executable && (info.st_mode & S_IXUSR) == 0) {
        fail("protected executable is not owner-executable");
    }
}

static int open_protected_regular(
    const char *path, int open_flags, bool executable
)
{
    char copy[PATH_MAX];
    char *cursor;
    char *save_pointer = NULL;
    char *component;
    int directory_descriptor;

    if (path == NULL || path[0] != '/' || strlen(path) >= sizeof(copy) ||
        strstr(path, "//") != NULL || strstr(path, "/./") != NULL ||
        strstr(path, "/../") != NULL) {
        fail("protected path is not canonical");
    }
    memcpy(copy, path + 1, strlen(path));
    copy[strlen(path) - 1] = '\0';
    directory_descriptor = open(
        "/", O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
    );
    if (directory_descriptor < 0) {
        fail("cannot open filesystem root");
    }
    require_protected_metadata(directory_descriptor, true, false);
    cursor = copy;
    component = strtok_r(cursor, "/", &save_pointer);
    while (component != NULL) {
        char *next_component = strtok_r(NULL, "/", &save_pointer);

        if (component[0] == '\0' || strcmp(component, ".") == 0 ||
            strcmp(component, "..") == 0) {
            close(directory_descriptor);
            fail("protected path contains an invalid component");
        }
        if (next_component == NULL) {
            const int descriptor = openat(
                directory_descriptor,
                component,
                open_flags | O_NOFOLLOW | O_CLOEXEC
            );
            close(directory_descriptor);
            if (descriptor < 0) {
                fail("cannot open protected artifact");
            }
            require_protected_metadata(descriptor, false, executable);
            return descriptor;
        }
        {
            const int next_directory = openat(
                directory_descriptor,
                component,
                O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
            );
            close(directory_descriptor);
            if (next_directory < 0) {
                fail("cannot traverse protected artifact ancestry");
            }
            require_protected_metadata(next_directory, true, false);
            directory_descriptor = next_directory;
        }
        component = next_component;
    }
    close(directory_descriptor);
    fail("protected path has no final component");
    return -1;
}

static void sha256_descriptor(int descriptor, char output[65])
{
    struct stat before;
    struct stat after;
    int descriptors[2];
    pid_t pid;
    char buffer[256];
    size_t used = 0;
    int status;

    if (fstat(descriptor, &before) != 0 ||
        lseek(descriptor, 0, SEEK_SET) != 0) {
        fail("cannot prepare protected artifact for hashing");
    }
    if (pipe2(descriptors, O_CLOEXEC) != 0) {
        fail("cannot create SHA verification pipe");
    }
    pid = fork();
    if (pid < 0) {
        fail("cannot fork SHA verifier");
    }
    if (pid == 0) {
        char descriptor_path[64];
        char *arguments[4];
        char *const environment[] = {
            "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", NULL
        };
        int null_descriptor;
        int descriptor_flags;

        descriptor_flags = fcntl(descriptor, F_GETFD);
        if (descriptor_flags < 0 ||
            fcntl(descriptor, F_SETFD, descriptor_flags & ~FD_CLOEXEC) != 0) {
            _exit(119);
        }
        if (snprintf(descriptor_path, sizeof(descriptor_path),
                     "/proc/self/fd/%d", descriptor) < 0) {
            _exit(120);
        }
        arguments[0] = "/usr/bin/sha256sum";
        arguments[1] = "--";
        arguments[2] = descriptor_path;
        arguments[3] = NULL;
        if (dup2(descriptors[1], STDOUT_FILENO) < 0) {
            _exit(121);
        }
        null_descriptor = open("/dev/null", O_WRONLY | O_CLOEXEC);
        if (null_descriptor < 0 ||
            dup2(null_descriptor, STDERR_FILENO) < 0) {
            _exit(122);
        }
        close(descriptors[0]);
        close(descriptors[1]);
        execve(arguments[0], arguments, environment);
        _exit(123);
    }
    close(descriptors[1]);
    while (used < sizeof(buffer) - 1) {
        const ssize_t count = read(
            descriptors[0], buffer + used, sizeof(buffer) - 1 - used
        );
        if (count == 0) {
            break;
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("cannot read SHA verifier output");
        }
        used += (size_t)count;
    }
    close(descriptors[0]);
    buffer[used] = '\0';
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            fail("cannot reap SHA verifier");
        }
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || used < 65) {
        fail("SHA verifier failed");
    }
    memcpy(output, buffer, 64);
    output[64] = '\0';
    if (!is_hex_sha256(output) ||
        (buffer[64] != ' ' && buffer[64] != '\t')) {
        fail("SHA verifier returned malformed output");
    }
    if (lseek(descriptor, 0, SEEK_SET) != 0 ||
        fstat(descriptor, &after) != 0) {
        fail("cannot restore protected artifact after hashing");
    }
    if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec) {
        fail("protected artifact changed while hashing");
    }
}

static void read_start_ticks(char output[64])
{
    int descriptor;
    char buffer[4096];
    ssize_t count;
    char *cursor;
    int field;

    descriptor = open("/proc/self/stat", O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        fail("cannot open own process stat");
    }
    count = read(descriptor, buffer, sizeof(buffer) - 1);
    close(descriptor);
    if (count <= 0) {
        fail("cannot read own process stat");
    }
    buffer[count] = '\0';
    cursor = strrchr(buffer, ')');
    if (cursor == NULL || cursor[1] != ' ') {
        fail("own process stat is malformed");
    }
    cursor += 2;
    for (field = 3; field <= 22; ++field) {
        char *token;
        char *separator;

        while (*cursor == ' ') {
            ++cursor;
        }
        token = cursor;
        separator = strchr(cursor, ' ');
        if (separator == NULL && field != 22) {
            fail("own process stat is truncated");
        }
        if (field == 22) {
            const size_t length = separator == NULL
                ? strcspn(token, "\n")
                : (size_t)(separator - token);
            if (length == 0 || length >= 64) {
                fail("own process start ticks are malformed");
            }
            memcpy(output, token, length);
            output[length] = '\0';
            return;
        }
        cursor = separator + 1;
    }
    fail("own process start ticks were not found");
}

static void signal_formal_group(pid_t group_leader, int signal_number)
{
    if (kill(-group_leader, signal_number) != 0 && errno != ESRCH) {
        fail("cannot signal formal process group");
    }
}

static void terminate_formal_group(pid_t group_leader, long grace_nanoseconds)
{
    const struct timespec grace = {0, grace_nanoseconds};

    signal_formal_group(group_leader, SIGTERM);
    while (nanosleep(&grace, NULL) != 0 && errno == EINTR) {
        continue;
    }
    signal_formal_group(group_leader, SIGKILL);
}

static int process_exit_code(int status)
{
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 129;
}

static int supervise_formal_child(
    pid_t formal_pid,
    pid_t formal_group,
    int result_descriptor,
    const sigset_t *control_signals
)
{
    bool terminating = false;
    unsigned int termination_polls = 0;
    int formal_status;

    for (;;) {
        siginfo_t formal_information;

        memset(&formal_information, 0, sizeof(formal_information));
        if (waitid(P_PID, (id_t)formal_pid, &formal_information,
                   WEXITED | WNOHANG | WNOWAIT) != 0) {
            fail("cannot inspect formal launcher");
        }
        if (formal_information.si_pid == formal_pid) {
            while (waitpid(formal_pid, &formal_status, 0) < 0) {
                if (errno != EINTR) {
                    signal_formal_group(formal_group, SIGKILL);
                    _exit(137);
                }
            }
            /* Report Bash's status while this process remains the live group
             * leader.  The top process performs the final TERM/KILL sweep and
             * only then reaps us.  If the top process dies at any point, its
             * PDEATHSIG is consumed below and this still-live leader kills the
             * complete, identity-stable group itself. */
            {
                ssize_t count;
                do {
                    count = write(
                        result_descriptor,
                        &formal_status,
                        sizeof(formal_status)
                    );
                } while (count < 0 && errno == EINTR);
                if (count != (ssize_t)sizeof(formal_status)) {
                    signal_formal_group(formal_group, SIGKILL);
                    _exit(137);
                }
            }
            close(result_descriptor);
            for (;;) {
                siginfo_t information;
                const int signal_number = sigwaitinfo(
                    control_signals, &information
                );
                if (signal_number > 0) {
                    /* Normal top-parent cleanup and PDEATHSIG share this
                     * path.  SIGKILL includes this live group leader, so no
                     * unsupervised hand-off window exists. */
                    signal_formal_group(formal_group, SIGKILL);
                    _exit(137);
                }
                if (errno != EINTR) {
                    signal_formal_group(formal_group, SIGKILL);
                    _exit(137);
                }
            }
        }
        if (getppid() == 1) {
            /* A pending PDEATHSIG normally reaches sigtimedwait below.  This
             * explicit identity check also closes platforms where delivery is
             * delayed between polling iterations. */
            signal_formal_group(formal_group, SIGKILL);
            _exit(137);
        }
        if (terminating) {
            const struct timespec poll_delay = {0, 100000000L};
            (void)nanosleep(&poll_delay, NULL);
            ++termination_polls;
            if (termination_polls >= 50U) {
                signal_formal_group(formal_group, SIGKILL);
                _exit(137);
            }
            continue;
        }
        {
            const struct timespec timeout = {0, 200000000L};
            siginfo_t information;
            const int signal_number = sigtimedwait(
                control_signals, &information, &timeout
            );
            if (signal_number > 0) {
                signal_formal_group(formal_group, signal_number);
                terminating = true;
            } else if (errno != EAGAIN && errno != EINTR) {
                fail("cannot wait for formal control signal");
            }
        }
    }
}

static int supervise_group_leader(
    pid_t supervisor_pid,
    int result_descriptor,
    const sigset_t *control_signals
)
{
    int supervisor_status;
    int formal_status = 0;
    bool formal_status_received = false;

    for (;;) {
        siginfo_t supervisor_information;
        ssize_t result_count;

        do {
            result_count = read(
                result_descriptor, &formal_status, sizeof(formal_status)
            );
        } while (result_count < 0 && errno == EINTR);
        if (result_count == (ssize_t)sizeof(formal_status)) {
            formal_status_received = true;
        } else if (result_count == 0) {
            /* The writer can close only after publishing the complete atomic
             * integer.  EOF without a result means the supervisor failed. */
            formal_status_received = false;
        } else if (result_count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            terminate_formal_group(supervisor_pid, 200000000L);
            (void)waitpid(supervisor_pid, NULL, 0);
            close(result_descriptor);
            fail("cannot read formal launcher result");
        } else if (result_count > 0) {
            terminate_formal_group(supervisor_pid, 200000000L);
            (void)waitpid(supervisor_pid, NULL, 0);
            close(result_descriptor);
            fail("formal launcher result is truncated");
        }
        if (formal_status_received) {
            /* The supervisor is still alive and therefore reserves the PGID.
             * Keep it unreaped through both signals, then return Bash's exact
             * status rather than the deliberate supervisor SIGKILL status. */
            terminate_formal_group(supervisor_pid, 200000000L);
            while (waitpid(supervisor_pid, &supervisor_status, 0) < 0) {
                if (errno != EINTR) {
                    close(result_descriptor);
                    fail("cannot reap formal process-group supervisor");
                }
            }
            close(result_descriptor);
            return formal_status;
        }

        memset(&supervisor_information, 0, sizeof(supervisor_information));
        if (waitid(P_PID, (id_t)supervisor_pid, &supervisor_information,
                   WEXITED | WNOHANG | WNOWAIT) != 0) {
            fail("cannot inspect formal process-group supervisor");
        }
        if (supervisor_information.si_pid == supervisor_pid) {
            /* WNOWAIT keeps the group leader's numeric identity reserved while
             * every possible descendant is terminated. */
            terminate_formal_group(supervisor_pid, 200000000L);
            while (waitpid(supervisor_pid, &supervisor_status, 0) < 0) {
                if (errno != EINTR) {
                    fail("cannot reap formal process-group supervisor");
                }
            }
            close(result_descriptor);
            return supervisor_status;
        }
        {
            const struct timespec timeout = {0, 200000000L};
            siginfo_t information;
            const int signal_number = sigtimedwait(
                control_signals, &information, &timeout
            );
            if (signal_number > 0) {
                signal_formal_group(supervisor_pid, signal_number);
            } else if (errno != EAGAIN && errno != EINTR) {
                fail("cannot wait for formal control signal");
            }
        }
    }
}

int main(int argc, char **argv)
{
    const char *mode;
    const char *expected_executor_sha;
    const char *expected_launcher_sha;
    char self_path[PATH_MAX];
    char actual_executor_sha[65];
    char actual_launcher_sha[65];
    char actual_source_sha[65];
    char start_ticks[64];
    char parent_pid_environment[64];
    char parent_ticks_environment[128];
    char executor_sha_environment[128];
    char source_sha_environment[128];
    char launcher_sha_environment[128];
    char launcher_device_environment[96];
    char launcher_inode_environment[96];
    char formal_group_environment[64];
    char mode_environment[32];
    ssize_t self_length;
    const pid_t formal_parent_pid = getpid();
    int executor_descriptor;
    int source_descriptor;
    int launcher_descriptor;
    int lock_descriptor;
    int readiness_pipe[2];
    int result_pipe[2];
    struct stat launcher_info;
    sigset_t control_signals;
    sigset_t empty_signals;
    struct sigaction default_action;
    struct sigaction ignore_action;
    pid_t supervisor_pid;
    int status;

    close_inherited_descriptors();
    memset(&default_action, 0, sizeof(default_action));
    default_action.sa_handler = SIG_DFL;
    sigemptyset(&default_action.sa_mask);
    memset(&ignore_action, 0, sizeof(ignore_action));
    ignore_action.sa_handler = SIG_IGN;
    sigemptyset(&ignore_action.sa_mask);
    sigemptyset(&empty_signals);
    if (sigaction(SIGINT, &default_action, NULL) != 0 ||
        sigaction(SIGTERM, &default_action, NULL) != 0 ||
        sigaction(SIGHUP, &default_action, NULL) != 0 ||
        sigaction(SIGPIPE, &ignore_action, NULL) != 0 ||
        sigprocmask(SIG_SETMASK, &empty_signals, NULL) != 0) {
        fail("cannot normalize inherited signal state");
    }
    if (argc != 4) {
        fail("usage: <absolute-executor> preflight|backbone <executor-sha256> <launcher-sha256>");
    }
    if (strcmp(argv[0], EXECUTOR_PATH) != 0) {
        fail("executor must be invoked by its protected canonical path");
    }
    mode = argv[1];
    if (strcmp(mode, "preflight") != 0 && strcmp(mode, "backbone") != 0) {
        fail("mode must be preflight or backbone");
    }
    expected_executor_sha = argv[2];
    expected_launcher_sha = argv[3];
    if (!is_hex_sha256(expected_executor_sha) ||
        !is_hex_sha256(expected_launcher_sha)) {
        fail("reviewed artifact SHAs must be 64 lowercase hex characters");
    }

    self_length = readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
    if (self_length <= 0 || self_length >= (ssize_t)sizeof(self_path)) {
        fail("cannot resolve static executor identity");
    }
    self_path[self_length] = '\0';
    if (strcmp(self_path, EXECUTOR_PATH) != 0) {
        fail("running executable path changed");
    }
    executor_descriptor = open_protected_regular(
        EXECUTOR_PATH, O_RDONLY, true
    );
    sha256_descriptor(executor_descriptor, actual_executor_sha);
    close(executor_descriptor);
    source_descriptor = open_protected_regular(
        SOURCE_PATH, O_RDONLY, false
    );
    sha256_descriptor(source_descriptor, actual_source_sha);
    close(source_descriptor);
    launcher_descriptor = open_protected_regular(
        LAUNCHER_PATH, O_RDONLY, true
    );
    if (launcher_descriptor != LAUNCHER_DESCRIPTOR) {
        if (dup3(launcher_descriptor, LAUNCHER_DESCRIPTOR, O_CLOEXEC) < 0) {
            close(launcher_descriptor);
            fail("cannot assign fixed launcher descriptor");
        }
        close(launcher_descriptor);
        launcher_descriptor = LAUNCHER_DESCRIPTOR;
    }
    sha256_descriptor(launcher_descriptor, actual_launcher_sha);
    if (fstat(launcher_descriptor, &launcher_info) != 0) {
        fail("cannot stat opened launcher");
    }
    if (strcmp(actual_executor_sha, expected_executor_sha) != 0) {
        fail("static executor is not the externally reviewed artifact");
    }
    if (strcmp(actual_source_sha, REVIEWED_SOURCE_SHA256) != 0) {
        fail("static executor source changed");
    }
    if (strcmp(actual_launcher_sha, expected_launcher_sha) != 0) {
        fail("formal launcher changed");
    }

    lock_descriptor = open(
        SHARED_GPU_LOCK, O_RDWR | O_CLOEXEC | O_NOFOLLOW
    );
    if (lock_descriptor < 0) {
        fail("shared GPU lock is missing or unsafe");
    }
    require_protected_metadata(lock_descriptor, false, false);
    if (flock(lock_descriptor, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
            fail_with_code("another V99 job owns the GPU lock", 6);
        }
        fail("cannot acquire the shared GPU lock");
    }
    sigemptyset(&control_signals);
    sigaddset(&control_signals, SIGINT);
    sigaddset(&control_signals, SIGTERM);
    sigaddset(&control_signals, SIGHUP);
    if (sigprocmask(SIG_BLOCK, &control_signals, NULL) != 0) {
        fail("cannot block formal control signals");
    }
    if (pipe2(readiness_pipe, O_CLOEXEC) != 0) {
        fail("cannot create supervisor readiness pipe");
    }
    if (pipe2(result_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
        fail("cannot create formal result pipe");
    }
    umask(0077);
    supervisor_pid = fork();
    if (supervisor_pid < 0) {
        fail("cannot fork formal process-group supervisor");
    }
    if (supervisor_pid == 0) {
        int formal_readiness_pipe[2];
        pid_t formal_pid;
        const pid_t local_supervisor_pid = getpid();

        close(readiness_pipe[0]);
        close(result_pipe[0]);
        if (setpgid(0, 0) != 0 ||
            prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 ||
            getppid() != formal_parent_pid) {
            _exit(124);
        }
        read_start_ticks(start_ticks);
        if (snprintf(parent_pid_environment, sizeof(parent_pid_environment),
                     "MCLN_FPR_STATIC_PARENT_PID=%ld",
                     (long)local_supervisor_pid) < 0 ||
            snprintf(parent_ticks_environment, sizeof(parent_ticks_environment),
                     "MCLN_FPR_STATIC_PARENT_START_TICKS=%s", start_ticks) < 0 ||
            snprintf(executor_sha_environment, sizeof(executor_sha_environment),
                     "MCLN_FPR_STATIC_EXEC_SHA256=%s",
                     actual_executor_sha) < 0 ||
            snprintf(source_sha_environment, sizeof(source_sha_environment),
                     "MCLN_FPR_STATIC_SOURCE_SHA256=%s",
                     actual_source_sha) < 0 ||
            snprintf(launcher_sha_environment, sizeof(launcher_sha_environment),
                     "MCLN_REVIEWED_FPR_AUDIT_LAUNCHER_SHA256=%s",
                     actual_launcher_sha) < 0 ||
            snprintf(launcher_device_environment,
                     sizeof(launcher_device_environment),
                     "MCLN_FPR_LAUNCHER_DEVICE=%llu",
                     (unsigned long long)launcher_info.st_dev) < 0 ||
            snprintf(launcher_inode_environment,
                     sizeof(launcher_inode_environment),
                     "MCLN_FPR_LAUNCHER_INODE=%llu",
                     (unsigned long long)launcher_info.st_ino) < 0 ||
            snprintf(formal_group_environment, sizeof(formal_group_environment),
                     "MCLN_FPR_FORMAL_PGID=%ld",
                     (long)local_supervisor_pid) < 0 ||
            snprintf(mode_environment, sizeof(mode_environment),
                     "MODE=%s", mode) < 0) {
            _exit(125);
        }
        if (pipe2(formal_readiness_pipe, O_CLOEXEC) != 0) {
            _exit(126);
        }
        formal_pid = fork();
        if (formal_pid < 0) {
            _exit(127);
        }
        if (formal_pid == 0) {
            char *const arguments[] = {
                "/bin/bash", "--noprofile", "--norc",
                "/proc/self/fd/3", NULL
            };
            char *const environment[] = {
                "HOME=/root",
                "USER=root",
                "LOGNAME=root",
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                "PATH=/root/miniconda3/envs/bdetr/bin:/usr/local/cuda/bin:/usr/bin:/bin",
                mode_environment,
                "MCLN_FPR_TRUSTED_CLEAN_ENV=1",
                "MCLN_FPR_STATIC_EXEC_PATH=" EXECUTOR_PATH,
                "MCLN_FPR_STATIC_SOURCE_PATH=" SOURCE_PATH,
                "MCLN_FPR_LAUNCHER_FD=3",
                parent_pid_environment,
                parent_ticks_environment,
                executor_sha_environment,
                source_sha_environment,
                launcher_sha_environment,
                launcher_device_environment,
                launcher_inode_environment,
                formal_group_environment,
                "PYTHONNOUSERSITE=1",
                "PYTHONDONTWRITEBYTECODE=1",
                NULL
            };
            int descriptor_flags;
            const char ready = 'R';

            close(formal_readiness_pipe[0]);
            close(readiness_pipe[1]);
            if (getpgrp() != local_supervisor_pid ||
                prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 ||
                getppid() != local_supervisor_pid) {
                _exit(128);
            }
            descriptor_flags = fcntl(launcher_descriptor, F_GETFD);
            if (descriptor_flags < 0 ||
                fcntl(launcher_descriptor, F_SETFD,
                      descriptor_flags & ~FD_CLOEXEC) != 0) {
                _exit(129);
            }
            if (write(formal_readiness_pipe[1], &ready, 1) != 1) {
                _exit(130);
            }
            close(formal_readiness_pipe[1]);
            if (sigprocmask(SIG_SETMASK, &empty_signals, NULL) != 0) {
                _exit(131);
            }
            execve(arguments[0], arguments, environment);
            _exit(132);
        }

        close(formal_readiness_pipe[1]);
        close(launcher_descriptor);
        {
            char ready = '\0';
            ssize_t count;
            do {
                count = read(formal_readiness_pipe[0], &ready, 1);
            } while (count < 0 && errno == EINTR);
            close(formal_readiness_pipe[0]);
            if (count != 1 || ready != 'R' ||
                getpgid(formal_pid) != local_supervisor_pid) {
                signal_formal_group(local_supervisor_pid, SIGKILL);
                _exit(133);
            }
        }
        {
            const char ready = 'S';
            if (write(readiness_pipe[1], &ready, 1) != 1) {
                signal_formal_group(local_supervisor_pid, SIGKILL);
                _exit(134);
            }
        }
        close(readiness_pipe[1]);
        status = supervise_formal_child(
            formal_pid,
            local_supervisor_pid,
            result_pipe[1],
            &control_signals
        );
        close(lock_descriptor);
        _exit(process_exit_code(status));
    }

    close(readiness_pipe[1]);
    close(result_pipe[1]);
    close(launcher_descriptor);
    {
        char ready = '\0';
        ssize_t count;
        do {
            count = read(readiness_pipe[0], &ready, 1);
        } while (count < 0 && errno == EINTR);
        close(readiness_pipe[0]);
        if (count != 1 || ready != 'S' ||
            getpgid(supervisor_pid) != supervisor_pid) {
            if (getpgid(supervisor_pid) == supervisor_pid) {
                terminate_formal_group(supervisor_pid, 200000000L);
            } else {
                (void)kill(supervisor_pid, SIGKILL);
            }
            (void)waitpid(supervisor_pid, NULL, 0);
            fail("formal process-group supervisor handshake failed");
        }
    }
    status = supervise_group_leader(
        supervisor_pid, result_pipe[0], &control_signals
    );
    close(lock_descriptor);
    return process_exit_code(status);
}
