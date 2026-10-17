/*
 * ============================================================================
 *  Copy-on-Write Memory Demonstrator
 *  Operating Systems Project (Linux / Ubuntu / WSL, GCC, C11)
 * ============================================================================
 *
 *  Build :  gcc -std=c11 -Wall -Wextra -Wpedantic -O2 \
 *               cow_memory_demonstrator.c -o cow_memory_demonstrator
 *  Run   :  ./cow_memory_demonstrator
 *
 *  What it does
 *  ------------
 *  A menu-driven terminal program that uses REAL fork() processes and REAL
 *  memory writes to show that after fork() parent and child have independent
 *  views of private memory.  Linux implements this lazily with Copy-on-Write
 *  (COW): the pages are shared until one side writes, then the writer gets a
 *  private copy.
 *
 *  Honesty note
 *  ------------
 *  Ordinary C pointers are VIRTUAL addresses.  Equal virtual addresses in
 *  parent and child do NOT prove the same physical page is used, and different
 *  values do NOT by themselves prove a page copy happened at a given moment.
 *  Everything printed as "observed" is measured at run time.  Statements about
 *  kernel behaviour are labelled as explanations.  The optional /proc
 *  diagnostics are observations with limitations (see option 5).
 *
 *  Architecture
 *  ------------
 *  One generic engine, run_experiment(), does the fork and synchronisation:
 *    parent --fork()--> child
 *    child  : prints + sends its "initial" snapshot through a pipe
 *    parent : prints its own initial snapshot
 *    parent : (a) writes itself, or (b) tells the child 'W' to write
 *    the *other* process then re-reads its value and reports it
 *    parent : closes pipes, waitpid()s the child, collects exit status
 *  Two pipes (parent->child, child->parent) give deterministic ordering, so
 *  console output never interleaves.
 *
 *  Menu handlers (basic_cow_demo, multi_page_cow_demo, ...) only collect
 *  validated input, build an experiment_t and call the engine.
 * ============================================================================
 */

#define _GNU_SOURCE /* MAP_ANONYMOUS and POSIX APIs under -std=c11 */

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* ------------------------------------------------------------------------ */
/* Section 1: constants and types                                            */
/* ------------------------------------------------------------------------ */

#define MAX_REGION_BYTES (16UL * 1024UL * 1024UL) /* 16 MiB safety limit      */
#define TABLE_MAX 16                              /* pages shown in a table   */
#define HISTORY_MAX 8                             /* remembered runs          */
#define VALUE_MIN (-1000000L)                     /* accepted value range     */
#define VALUE_MAX 1000000L
#define INPUT_BUF 128

#define DEFAULT_INITIAL 100L
#define DEFAULT_NEW 999L
#define DEFAULT_PAGES 4L

typedef enum { DEMO_OK = 0, DEMO_EOF = -1, DEMO_ERROR = -2 } demo_status_t;
typedef enum { WRITER_PARENT = 1, WRITER_CHILD = 2 } writer_t;

/* Values read from /proc for the mapping that contains the demo region.
 * -1 means "not available".  These are OBSERVATIONS with limitations. */
typedef struct {
    bool available;      /* region's VMA found in /proc/self/smaps */
    long rss_kb;         /* Rss                                    */
    long pss_kb;         /* Pss (proportional set size)            */
    long shared_clean_kb;
    long shared_dirty_kb;
    long private_clean_kb;
    long private_dirty_kb;
    long vm_rss_kb;      /* whole-process VmRSS from /proc/self/status */
} proc_obs_t;

/* What one process observes about the target memory at one moment. */
typedef struct {
    pid_t pid;
    pid_t ppid;
    uintptr_t addr;            /* virtual address of the selected slot      */
    int value;                 /* value read from the selected slot         */
    size_t changed_pages;      /* pages whose slot != initial value (scan)  */
    size_t table_len;
    int table[TABLE_MAX];      /* slot value of the first table_len pages   */
    proc_obs_t proc;
} snapshot_t;

/* Description of one experiment (built in the parent before fork()). */
typedef struct {
    const char *name;
    volatile int *slot0;       /* the int being observed in page 0           */
    size_t stride_ints;        /* ints between the slot of page i and i+1    */
    size_t pages;              /* 1 for a plain variable                     */
    int init_base;             /* page i is initialised to init_base + i     */
    size_t sel_page;           /* page the writer modifies                   */
    int new_value;
    writer_t writer;
    bool collect_proc;
    void *region;              /* NULL for the plain-variable demos          */
    size_t region_size;
    size_t page_size;
} experiment_t;

/* Everything measured in one run; kept for the comparison / history. */
typedef struct {
    char name[48];
    writer_t writer;
    size_t pages, sel_page, page_size;
    int init_value, new_value;
    bool collect_proc;
    snapshot_t parent_pre, parent_init, child_init, parent_post, child_post;
    bool independent;
    int child_exit_status;
} run_record_t;

/* User configuration remembered between runs (cleared by Reset). */
typedef struct {
    long initial, modified, pages;
    int writer;
} settings_t;

static settings_t g_settings;
static run_record_t g_history[HISTORY_MAX];
static int g_history_count = 0;
static int g_live_mappings = 0; /* mmap()s not yet munmap()ped (should be 0) */

/* ------------------------------------------------------------------------ */
/* Section 2: small utilities                                                */
/* ------------------------------------------------------------------------ */

/* Print one line tagged with role and PID, flushed immediately so output
 * from different processes cannot be reordered by stdio buffering. */
static void say(const char *role, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void say(const char *role, const char *fmt, ...)
{
    va_list ap;
    printf("[%-6s pid=%d] ", role, (int)getpid());
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

static void print_banner(const char *title)
{
    printf("\n============================================================\n");
    printf("  %s\n", title);
    printf("============================================================\n");
}

/* write()/read() wrappers that finish partial transfers and retry on EINTR. */
static int write_full(int fd, const void *buf, size_t n)
{
    const unsigned char *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        p += (size_t)w;
        n -= (size_t)w;
    }
    return 0;
}

static int read_full(int fd, void *buf, size_t n)
{
    unsigned char *p = buf;
    while (n > 0) {
        ssize_t r = read(fd, p, n);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (r == 0) { /* peer closed the pipe early */
            errno = 0;
            return -1;
        }
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 0;
}

static size_t get_page_size(void)
{
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) {
        perror("sysconf(_SC_PAGESIZE)");
        return 0;
    }
    return (size_t)ps;
}

/* ------------------------------------------------------------------------ */
/* Section 3: safe input                                                     */
/* ------------------------------------------------------------------------ */

/* Read one line.  Returns 0 = ok, -1 = EOF, 1 = line too long (discarded). */
static int read_line(char *buf, size_t size)
{
    if (fgets(buf, (int)size, stdin) == NULL)
        return -1;
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') {
        buf[len - 1] = '\0';
        return 0;
    }
    if (len == size - 1) { /* buffer filled without newline: discard the rest */
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {
        }
        return 1;
    }
    return 0; /* last line without trailing newline */
}

/* Prompt until the user enters a whole number in [min, max].
 * Empty input selects the default when has_default is true.
 * Returns 0 on success, -1 on end-of-file. */
static int read_integer_safely(const char *prompt, long min, long max,
                               long def, bool has_default, long *out)
{
    char buf[INPUT_BUF];
    for (;;) {
        if (has_default)
            printf("%s (%ld..%ld, Enter = %ld): ", prompt, min, max, def);
        else
            printf("%s (%ld..%ld): ", prompt, min, max);
        fflush(stdout);

        int rc = read_line(buf, sizeof buf);
        if (rc < 0) {
            printf("\n[input] End of input reached.\n");
            return -1;
        }
        if (rc == 1) {
            printf("  Invalid: input too long.\n");
            continue;
        }

        char *start = buf;
        while (isspace((unsigned char)*start))
            start++;
        if (*start == '\0') {
            if (has_default) {
                *out = def;
                return 0;
            }
            printf("  Invalid: please enter a whole number.\n");
            continue;
        }

        errno = 0;
        char *end = NULL;
        long v = strtol(start, &end, 10);
        if (end == start || errno == ERANGE) {
            printf("  Invalid: '%s' is not a usable whole number.\n", start);
            continue;
        }
        while (isspace((unsigned char)*end))
            end++;
        if (*end != '\0') {
            printf("  Invalid: unexpected characters after the number.\n");
            continue;
        }
        if (v < min || v > max) {
            printf("  Out of range: allowed values are %ld..%ld.\n", min, max);
            continue;
        }
        *out = v;
        return 0;
    }
}

/* Ask a yes/no question.  Returns 0 ok, -1 EOF. */
static int read_yes_no(const char *prompt, bool def, bool *out)
{
    char buf[INPUT_BUF];
    for (;;) {
        printf("%s [%s]: ", prompt, def ? "Y/n" : "y/N");
        fflush(stdout);
        int rc = read_line(buf, sizeof buf);
        if (rc < 0) {
            printf("\n[input] End of input reached.\n");
            return -1;
        }
        if (rc == 1) {
            printf("  Invalid: input too long.\n");
            continue;
        }
        char *p = buf;
        while (isspace((unsigned char)*p))
            p++;
        if (*p == '\0') {
            *out = def;
            return 0;
        }
        if (tolower((unsigned char)*p) == 'y') {
            *out = true;
            return 0;
        }
        if (tolower((unsigned char)*p) == 'n') {
            *out = false;
            return 0;
        }
        printf("  Please answer y or n.\n");
    }
}

/* Read a value that must differ from `forbidden` (so the change is visible). */
static int read_distinct_value(const char *prompt, long def, long forbidden,
                               long *out)
{
    if (def == forbidden)
        def = (forbidden > VALUE_MIN) ? forbidden - 1 : forbidden + 1;
    for (;;) {
        int rc = read_integer_safely(prompt, VALUE_MIN, VALUE_MAX, def, true, out);
        if (rc != 0)
            return rc;
        if (*out != forbidden)
            return 0;
        printf("  The new value must differ from the initial value (%ld).\n",
               forbidden);
    }
}

static int read_writer_choice(long def, writer_t *out)
{
    long v;
    printf("Who modifies memory after fork()?  1 = Parent, 2 = Child\n");
    if (read_integer_safely("Choice", 1, 2, def, true, &v) != 0)
        return -1;
    *out = (v == 1) ? WRITER_PARENT : WRITER_CHILD;
    return 0;
}

static int pause_for_enter(void)
{
    char buf[INPUT_BUF];
    printf("\nPress Enter to return to the main menu...");
    fflush(stdout);
    int rc = read_line(buf, sizeof buf);
    if (rc < 0) {
        printf("\n");
        return DEMO_EOF;
    }
    return DEMO_OK;
}

/* ------------------------------------------------------------------------ */
/* Section 4: process helpers                                                */
/* ------------------------------------------------------------------------ */

/* Wait for `pid` with waitpid(), retrying on EINTR, and report how it ended.
 * Returns 0 if the child exited normally with status 0, otherwise -1.
 * *exit_status receives the exit code (or 128+signal). */
static int wait_for_child(pid_t pid, const char *label, int *exit_status)
{
    int status = 0;
    pid_t r;
    do {
        r = waitpid(pid, &status, 0);
    } while (r == -1 && errno == EINTR);

    if (r == -1) {
        perror("waitpid");
        if (exit_status)
            *exit_status = -1;
        return -1;
    }
    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        say("PARENT", "waitpid(): %s (pid %d) exited normally, status %d.",
            label, (int)pid, code);
        if (exit_status)
            *exit_status = code;
        return code == 0 ? 0 : -1;
    }
    if (WIFSIGNALED(status)) {
        say("PARENT", "waitpid(): %s (pid %d) was killed by signal %d.", label,
            (int)pid, WTERMSIG(status));
        if (exit_status)
            *exit_status = 128 + WTERMSIG(status);
        return -1;
    }
    if (exit_status)
        *exit_status = -1;
    return -1;
}

/* Collect any stray children (there should be none). Returns how many. */
static int reap_stray_children(void)
{
    int count = 0;
    for (;;) {
        pid_t r = waitpid(-1, NULL, WNOHANG);
        if (r > 0) {
            count++;
            continue;
        }
        if (r == -1 && errno == EINTR)
            continue;
        break; /* 0 = none finished, -1 = ECHILD (no children) */
    }
    return count;
}

/* ------------------------------------------------------------------------ */
/* Section 5: memory region helpers (mmap)                                   */
/* ------------------------------------------------------------------------ */

/* Private anonymous mapping: the right mapping type for ordinary COW.
 * (MAP_SHARED would create genuinely shared memory, which is NOT COW.) */
static void *create_private_region(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }
    g_live_mappings++;
    return p;
}

static void destroy_region(void *region, size_t bytes)
{
    if (region == NULL)
        return;
    if (munmap(region, bytes) == -1)
        perror("munmap");
    else
        g_live_mappings--;
}

static volatile int *slot_of(const experiment_t *cfg, size_t page)
{
    return cfg->slot0 + page * cfg->stride_ints;
}

/* Build an experiment over a fresh private region of `pages` pages.
 * Page i is filled with a byte pattern and its slot (the int at byte_offset
 * within the page) is set to init_base + i.  All arguments are re-validated. */
static int build_region_experiment(experiment_t *cfg, const char *name,
                                   size_t pages, size_t byte_offset,
                                   int init_base, size_t sel_page,
                                   int new_value, writer_t writer, bool proc)
{
    size_t ps = get_page_size();
    if (ps == 0)
        return DEMO_ERROR;
    if (pages == 0 || pages > MAX_REGION_BYTES / ps || byte_offset >= ps ||
        sel_page >= pages) {
        fprintf(stderr, "Internal validation failed (pages/offset/index).\n");
        return DEMO_ERROR;
    }
    size_t bytes = pages * ps;
    void *region = create_private_region(bytes);
    if (region == NULL)
        return DEMO_ERROR;

    memset(cfg, 0, sizeof *cfg);
    cfg->name = name;
    cfg->slot0 = (volatile int *)((unsigned char *)region +
                                  (byte_offset / sizeof(int)) * sizeof(int));
    cfg->stride_ints = ps / sizeof(int);
    cfg->pages = pages;
    cfg->init_base = init_base;
    cfg->sel_page = sel_page;
    cfg->new_value = new_value;
    cfg->writer = writer;
    cfg->collect_proc = proc;
    cfg->region = region;
    cfg->region_size = bytes;
    cfg->page_size = ps;

    for (size_t i = 0; i < pages; i++) {
        memset((unsigned char *)region + i * ps, (int)((i * 7 + 1) & 0xFF), ps);
        *slot_of(cfg, i) = init_base + (int)i;
    }
    return DEMO_OK;
}

static void release_experiment(experiment_t *cfg)
{
    destroy_region(cfg->region, cfg->region_size);
    cfg->region = NULL;
}

/* ------------------------------------------------------------------------ */
/* Section 6: optional /proc observations                                    */
/* ------------------------------------------------------------------------ */

/* Read memory statistics of the VMA containing `addr` from /proc/self/smaps
 * plus VmRSS from /proc/self/status.  Never fabricates: anything that cannot
 * be read stays -1 / available=false.  No root required. */
static void read_proc_obs(const void *addr, proc_obs_t *o)
{
    o->available = false;
    o->rss_kb = o->pss_kb = o->shared_clean_kb = o->shared_dirty_kb = -1;
    o->private_clean_kb = o->private_dirty_kb = o->vm_rss_kb = -1;

    FILE *f = fopen("/proc/self/smaps", "r");
    if (f != NULL) {
        char line[512];
        bool in_vma = false;
        bool found = false;
        unsigned long target = (unsigned long)(uintptr_t)addr;
        while (fgets(line, sizeof line, f) != NULL) {
            unsigned long s, e;
            char perms[8];
            long v;
            if (sscanf(line, "%lx-%lx %7s", &s, &e, perms) == 3) {
                in_vma = (target >= s && target < e);
                if (in_vma)
                    found = true;
                continue;
            }
            if (!in_vma)
                continue;
            if (sscanf(line, "Rss: %ld kB", &v) == 1)
                o->rss_kb = v;
            else if (sscanf(line, "Pss: %ld kB", &v) == 1)
                o->pss_kb = v;
            else if (sscanf(line, "Shared_Clean: %ld kB", &v) == 1)
                o->shared_clean_kb = v;
            else if (sscanf(line, "Shared_Dirty: %ld kB", &v) == 1)
                o->shared_dirty_kb = v;
            else if (sscanf(line, "Private_Clean: %ld kB", &v) == 1)
                o->private_clean_kb = v;
            else if (sscanf(line, "Private_Dirty: %ld kB", &v) == 1)
                o->private_dirty_kb = v;
        }
        fclose(f);
        o->available = found;
    }

    f = fopen("/proc/self/status", "r");
    if (f != NULL) {
        char line[256];
        long v;
        while (fgets(line, sizeof line, f) != NULL) {
            if (sscanf(line, "VmRSS: %ld kB", &v) == 1) {
                o->vm_rss_kb = v;
                break;
            }
        }
        fclose(f);
    }
}

static void fmt_kb(char *buf, size_t n, long v)
{
    if (v < 0)
        snprintf(buf, n, "n/a");
    else
        snprintf(buf, n, "%ld", v);
}

static void print_proc_row(const char *label, const proc_obs_t *o)
{
    char c[7][24];
    fmt_kb(c[0], sizeof c[0], o->rss_kb);
    fmt_kb(c[1], sizeof c[1], o->pss_kb);
    fmt_kb(c[2], sizeof c[2], o->shared_clean_kb);
    fmt_kb(c[3], sizeof c[3], o->shared_dirty_kb);
    fmt_kb(c[4], sizeof c[4], o->private_clean_kb);
    fmt_kb(c[5], sizeof c[5], o->private_dirty_kb);
    fmt_kb(c[6], sizeof c[6], o->vm_rss_kb);
    printf("  %-24s %7s %7s %8s %8s %8s %8s %9s\n", label, c[0], c[1], c[2],
           c[3], c[4], c[5], c[6]);
}

static void print_proc_table(const run_record_t *r)
{
    printf("\n--- Optional /proc observations (kB; labelled OBSERVATIONS) ---\n");
    if (!r->parent_pre.proc.available && !r->parent_post.proc.available) {
        printf("  /proc/self/smaps was not readable here; no data to show.\n");
        printf("  (The core demonstration above is unaffected.)\n");
        return;
    }
    printf("  %-24s %7s %7s %8s %8s %8s %8s %9s\n", "Moment", "Rss", "Pss",
           "ShClean", "ShDirty", "PrClean", "PrDirty", "VmRSS(all)");
    print_proc_row("parent, before fork", &r->parent_pre.proc);
    print_proc_row("parent, after fork", &r->parent_init.proc);
    print_proc_row("child,  after fork", &r->child_init.proc);
    print_proc_row("parent, after write", &r->parent_post.proc);
    print_proc_row("child,  after write", &r->child_post.proc);
    printf("  Fields: Rss = resident pages of the mapping; Pss = Rss with shared\n"
           "  pages divided among the processes mapping them; Shared_* = pages\n"
           "  mapped by more than one process; Private_* = pages mapped by only\n"
           "  this process; Dirty = modified, Clean = unmodified; VmRSS = whole\n"
           "  process.\n");
    printf("  Limitations: the numbers cover the whole VMA, which the kernel may\n"
           "  have merged with neighbouring anonymous mappings; huge pages, lazy\n"
           "  allocation and kernel version change the counts; values are in kB\n"
           "  and only approximate page counts.  They SUPPLEMENT the value\n"
           "  comparison and are not proof of an exact one-page copy.\n");
}

/* ------------------------------------------------------------------------ */
/* Section 7: the fork() engine                                              */
/* ------------------------------------------------------------------------ */

static void take_snapshot(const experiment_t *cfg, snapshot_t *s)
{
    memset(s, 0, sizeof *s);
    s->pid = getpid();
    s->ppid = getppid();

    volatile int *sel = slot_of(cfg, cfg->sel_page);
    s->addr = (uintptr_t)sel;
    s->value = *sel;

    for (size_t i = 0; i < cfg->pages; i++)
        if (*slot_of(cfg, i) != cfg->init_base + (int)i)
            s->changed_pages++;

    s->table_len = cfg->pages < TABLE_MAX ? cfg->pages : TABLE_MAX;
    for (size_t i = 0; i < s->table_len; i++)
        s->table[i] = *slot_of(cfg, i);

    if (cfg->collect_proc && cfg->region != NULL)
        read_proc_obs(cfg->region, &s->proc);
    else
        s->proc.available = false;
}

/* Code that runs only in the child.  It never returns to the menu: it ends
 * with _exit() so stdio buffers inherited from the parent are not flushed
 * twice and the menu loop is never executed by two processes. */
static void child_main(const experiment_t *cfg, int rd, int wr)
    __attribute__((noreturn));

static void child_main(const experiment_t *cfg, int rd, int wr)
{
    snapshot_t snap;
    take_snapshot(cfg, &snap);
    say("CHILD", "my PID=%d, my parent's PID=%d", (int)snap.pid, (int)snap.ppid);
    say("CHILD", "initial value = %d at virtual address 0x%" PRIxPTR, snap.value,
        snap.addr);
    if (write_full(wr, &snap, sizeof snap) != 0)
        _exit(3);

    char cmd;
    if (read_full(rd, &cmd, 1) != 0)
        _exit(4);

    volatile int *slot = slot_of(cfg, cfg->sel_page);
    if (cmd == 'W') {
        say("CHILD", "writing %d into my memory (this triggers COW)...",
            cfg->new_value);
        *slot = cfg->new_value;
    } else if (cmd == 'R') {
        say("CHILD", "parent has modified its memory; re-reading mine...");
    } else {
        _exit(5);
    }

    take_snapshot(cfg, &snap);
    say("CHILD", "value I observe now = %d", snap.value);
    if (write_full(wr, &snap, sizeof snap) != 0)
        _exit(3);

    close(rd);
    close(wr);
    if (cfg->region != NULL)
        munmap(cfg->region, cfg->region_size);
    fflush(stdout);
    _exit(0);
}

static const char *writer_name(writer_t w)
{
    return w == WRITER_PARENT ? "PARENT" : "CHILD";
}

/* Run one complete fork()-based experiment and fill `rec` with the results.
 * Returns DEMO_OK or DEMO_ERROR. */
static int run_experiment(const experiment_t *cfg, run_record_t *rec)
{
    memset(rec, 0, sizeof *rec);
    snprintf(rec->name, sizeof rec->name, "%s", cfg->name);
    rec->writer = cfg->writer;
    rec->pages = cfg->pages;
    rec->sel_page = cfg->sel_page;
    rec->page_size = cfg->page_size;
    rec->init_value = cfg->init_base + (int)cfg->sel_page;
    rec->new_value = cfg->new_value;
    rec->collect_proc = cfg->collect_proc;

    take_snapshot(cfg, &rec->parent_pre);
    printf("\n--- BEFORE fork() ---\n");
    say("PARENT", "my PID=%d", (int)rec->parent_pre.pid);
    say("PARENT", "value = %d at virtual address 0x%" PRIxPTR,
        rec->parent_pre.value, rec->parent_pre.addr);

    int p2c[2], c2p[2];
    if (pipe(p2c) == -1) {
        perror("pipe");
        return DEMO_ERROR;
    }
    if (pipe(c2p) == -1) {
        perror("pipe");
        close(p2c[0]);
        close(p2c[1]);
        return DEMO_ERROR;
    }

    printf("\n--- fork() ---\n");
    fflush(stdout); /* avoid duplicated buffered output in the child */
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(p2c[0]);
        close(p2c[1]);
        close(c2p[0]);
        close(c2p[1]);
        return DEMO_ERROR;
    }
    if (pid == 0) {
        close(p2c[1]);
        close(c2p[0]);
        child_main(cfg, p2c[0], c2p[1]); /* does not return */
    }

    /* ---- parent ---- */
    close(p2c[0]);
    close(c2p[1]);
    int rc = DEMO_OK;
    char cmd;

    if (read_full(c2p[0], &rec->child_init, sizeof rec->child_init) != 0) {
        fprintf(stderr, "Failed to receive the child's initial report.\n");
        rc = DEMO_ERROR;
        goto finish;
    }
    take_snapshot(cfg, &rec->parent_init);
    say("PARENT", "after fork, child PID=%d; my value = %d at virtual address "
                  "0x%" PRIxPTR,
        (int)pid, rec->parent_init.value, rec->parent_init.addr);

    printf("\n--- MODIFICATION (done by the %s) ---\n", writer_name(cfg->writer));
    fflush(stdout); /* header must appear before the child's reply */
    if (cfg->writer == WRITER_PARENT) {
        say("PARENT", "writing %d into my memory (this triggers COW)...",
            cfg->new_value);
        *slot_of(cfg, cfg->sel_page) = cfg->new_value;
        take_snapshot(cfg, &rec->parent_post);
        say("PARENT", "value I observe now = %d", rec->parent_post.value);
        cmd = 'R';
        if (write_full(p2c[1], &cmd, 1) != 0) {
            perror("write to child");
            rc = DEMO_ERROR;
            goto finish;
        }
        if (read_full(c2p[0], &rec->child_post, sizeof rec->child_post) != 0) {
            fprintf(stderr, "Failed to receive the child's final report.\n");
            rc = DEMO_ERROR;
            goto finish;
        }
    } else {
        cmd = 'W';
        if (write_full(p2c[1], &cmd, 1) != 0) {
            perror("write to child");
            rc = DEMO_ERROR;
            goto finish;
        }
        if (read_full(c2p[0], &rec->child_post, sizeof rec->child_post) != 0) {
            fprintf(stderr, "Failed to receive the child's final report.\n");
            rc = DEMO_ERROR;
            goto finish;
        }
        take_snapshot(cfg, &rec->parent_post);
        say("PARENT", "child has modified its memory; my value is now = %d",
            rec->parent_post.value);
    }

finish:
    /* Closing the pipe ends lets a blocked child see EOF and exit. */
    close(p2c[1]);
    close(c2p[0]);
    printf("\n--- WAITING for the child ---\n");
    if (wait_for_child(pid, "child", &rec->child_exit_status) != 0)
        rc = DEMO_ERROR;
    if (rc != DEMO_OK)
        return rc;

    const snapshot_t *mod = cfg->writer == WRITER_PARENT ? &rec->parent_post
                                                         : &rec->child_post;
    const snapshot_t *other = cfg->writer == WRITER_PARENT ? &rec->child_post
                                                           : &rec->parent_post;
    rec->independent = (mod->value == cfg->new_value) &&
                       (other->value == rec->init_value) &&
                       (mod->changed_pages == 1) && (other->changed_pages == 0);
    return DEMO_OK;
}

/* ------------------------------------------------------------------------ */
/* Section 8: history, results and explanations                              */
/* ------------------------------------------------------------------------ */

static void record_run(const run_record_t *rec)
{
    if (g_history_count == HISTORY_MAX) {
        memmove(&g_history[0], &g_history[1],
                sizeof g_history[0] * (HISTORY_MAX - 1));
        g_history_count--;
    }
    g_history[g_history_count++] = *rec;
}

static void print_verdict(const run_record_t *r)
{
    const snapshot_t *mod = r->writer == WRITER_PARENT ? &r->parent_post
                                                       : &r->child_post;
    const snapshot_t *other = r->writer == WRITER_PARENT ? &r->child_post
                                                         : &r->parent_post;
    printf("\n--- RESULT (values observed during this run) ---\n");
    printf("  Initial value (both processes) : %d / %d\n", r->parent_init.value,
           r->child_init.value);
    printf("  Written by the %-6s process   : %d\n", writer_name(r->writer),
           r->new_value);
    printf("  Writer observes                : %d\n", mod->value);
    printf("  Other process observes         : %d\n", other->value);
    printf("  Independent private copies?    : %s\n",
           r->independent ? "YES - the other process kept its original value"
                          : "NO - unexpected result, see output above");
}

static void print_page_table(const run_record_t *r)
{
    if (r->pages <= 1)
        return;
    printf("\n--- Per-page values (slot of each page; '*' = page that was written) "
           "---\n");
    printf("  %-6s %-14s %-14s %-14s %-14s\n", "Page", "parent(init)",
           "child(init)", "parent(now)", "child(now)");
    for (size_t i = 0; i < r->parent_init.table_len; i++)
        printf("  %-5zu%c %-14d %-14d %-14d %-14d\n", i,
               i == r->sel_page ? '*' : ' ', r->parent_init.table[i],
               r->child_init.table[i], r->parent_post.table[i],
               r->child_post.table[i]);
    if (r->pages > TABLE_MAX)
        printf("  (first %d of %zu pages shown; the scan below covers all of "
               "them)\n",
               TABLE_MAX, r->pages);
    printf("  Full scan, pages whose value differs from the initial pattern:\n"
           "    parent = %zu, child = %zu\n",
           r->parent_post.changed_pages, r->child_post.changed_pages);
}

/* Option 4 output: observed facts first, kernel explanation afterwards. */
static void print_comparison(const run_record_t *r)
{
    const snapshot_t *mod = r->writer == WRITER_PARENT ? &r->parent_post
                                                       : &r->child_post;
    const snapshot_t *other = r->writer == WRITER_PARENT ? &r->child_post
                                                         : &r->parent_post;
    print_banner("BEFORE / AFTER COMPARISON");
    printf("Experiment: %s\n", r->name);

    printf("\n[OBSERVED - measured by this run]\n");
    printf("  Parent PID                 : %d\n", (int)r->parent_init.pid);
    printf("  Child PID                  : %d (its getppid() = %d)\n",
           (int)r->child_init.pid, (int)r->child_init.ppid);
    printf("  Target                     : page %zu of %zu%s\n", r->sel_page,
           r->pages, r->page_size ? "" : " (plain variable)");
    printf("  Initial value              : parent %d, child %d\n",
           r->parent_init.value, r->child_init.value);
    printf("  Virtual address (parent)   : 0x%" PRIxPTR "\n", r->parent_init.addr);
    printf("  Virtual address (child)    : 0x%" PRIxPTR "  -> %s\n",
           r->child_init.addr,
           r->parent_init.addr == r->child_init.addr
               ? "same virtual address (normal after fork)"
               : "different virtual address");
    printf("  Process that modified      : %s (wrote %d)\n",
           writer_name(r->writer), r->new_value);
    printf("  Value seen by modifier     : %d\n", mod->value);
    printf("  Value seen by other process: %d\n", other->value);
    printf("  Data values independent?   : %s\n", r->independent ? "YES" : "NO");
    printf("  Child exit status          : %d\n", r->child_exit_status);
    print_page_table(r);
    if (r->collect_proc)
        print_proc_table(r);

    printf("\n[EXPLANATION - how Linux typically achieves this; not measured]\n");
    printf("  * After fork() both processes map the same private pages\n"
           "    read-only-for-COW; no page is copied yet.\n");
    printf("  * The first write by either process causes a write (page) fault.\n"
           "    The kernel then gives the writer its own private copy of that\n"
           "    one page and lets the write proceed on the copy.\n");
    printf("  * The other process still maps the original page, so it keeps its\n"
           "    old value - exactly what the observed values above show.\n");
    printf("  * Equal virtual addresses do not mean equal physical pages: each\n"
           "    process has its own page table translating the same virtual\n"
           "    address to a physical frame.\n");
}

static void explain_cow_concepts(void)
{
    print_banner("VIRTUAL ADDRESSES, PHYSICAL MEMORY AND COPY-ON-WRITE");
    printf(
        "1. Virtual address space\n"
        "   Every process sees its own private range of virtual addresses.\n"
        "   Pointers in C hold virtual addresses.  The CPU's MMU translates\n"
        "   them to physical addresses using the process's page tables.\n\n"
        "2. Physical memory pages\n"
        "   RAM is divided into fixed-size frames (usually 4096 bytes). The\n"
        "   kernel decides which frame backs each virtual page.\n\n"
        "3. fork() and address-space inheritance\n"
        "   fork() gives the child a logically identical copy of the parent's\n"
        "   address space: same layout, same virtual addresses, same contents.\n"
        "   Linux does not copy the data eagerly; it duplicates page tables\n"
        "   and marks private writable pages so a write will fault.\n\n"
        "4. Private mappings\n"
        "   Memory from mmap(MAP_PRIVATE | MAP_ANONYMOUS), the stack and the\n"
        "   heap is private: changes are never visible to other processes.\n"
        "   (MAP_SHARED is different - it really shares memory and is NOT\n"
        "   COW, so this program does not use it for the COW demos.)\n\n"
        "5. Copy-on-Write and write faults\n"
        "   Until someone writes, parent and child can use the same physical\n"
        "   frames.  When one of them writes, the CPU raises a page fault, the\n"
        "   kernel allocates a new frame, copies the old page into it, maps\n"
        "   it into the writer's page table as writable and resumes the\n"
        "   write.  The other process keeps the original frame.  COW works at\n"
        "   page granularity, so changing one int copies one page, not the\n"
        "   whole address space.\n\n"
        "6. Why one process's changes do not affect the other\n"
        "   After the write fault each process maps a different frame for the\n"
        "   written page, so later writes touch different physical memory.\n\n"
        "7. Identical virtual addresses prove nothing about physical pages\n"
        "   Parent and child normally print the SAME address for the same\n"
        "   variable, yet after a COW fault they use different frames.\n"
        "   Printing a pointer cannot show physical sharing.\n\n"
        "8. Why memory measurements are not a clean one-page jump\n"
        "   Kernel accounting is approximate: pages are allocated lazily,\n"
        "   page tables themselves use memory, transparent huge pages may\n"
        "   change the copy size, VMAs can merge, and the process has other\n"
        "   activity.  This program therefore compares VALUES for its main\n"
        "   evidence and treats /proc numbers as optional observations.\n\n"
        "Optional /proc fields used (from /proc/self/smaps, VmRSS from\n"
        "/proc/self/status): Rss, Pss, Shared_Clean/Dirty, Private_Clean/Dirty.\n"
        "Pss splits shared pages among the processes that map them, so after\n"
        "fork() a shared page often shows a lower Pss than Rss.  These are\n"
        "observations of one environment, may be unavailable, and never\n"
        "identify an individual physical page.\n");
}

/* ------------------------------------------------------------------------ */
/* Section 9: menu handlers                                                  */
/* ------------------------------------------------------------------------ */

static void display_main_menu(void)
{
    print_banner("COPY-ON-WRITE MEMORY DEMONSTRATOR");
    printf("  1. Basic Copy-on-Write Demonstration\n"
           "  2. Parent and Child Memory Modification\n"
           "  3. Demonstrate Multiple Memory Pages\n"
           "  4. Compare Memory Before and After Modification\n"
           "  5. Explain Virtual Addresses and Physical Memory\n"
           "  6. Run Complete COW Demonstration\n"
           "  7. Reset Demonstration\n"
           "  0. Exit\n"
           "------------------------------------------------------------\n");
}

/* Option 1: ordinary local variable, child modifies it. */
static int basic_cow_demo(void)
{
    print_banner("1. BASIC COPY-ON-WRITE DEMONSTRATION");
    printf("A local variable is set in the parent, fork() creates a child,\n"
           "the child changes its copy, and the parent checks its own value.\n\n");

    long initial;
    if (read_integer_safely("Initial value of the variable", VALUE_MIN, VALUE_MAX,
                            g_settings.initial, true, &initial) != 0)
        return DEMO_EOF;

    volatile int variable = (int)initial;
    experiment_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.name = "Basic demo (local variable, child writes)";
    cfg.slot0 = &variable;
    cfg.stride_ints = 0;
    cfg.pages = 1;
    cfg.init_base = (int)initial;
    cfg.sel_page = 0;
    cfg.new_value = (int)(initial + 100); /* sensible default modification */
    cfg.writer = WRITER_CHILD;

    run_record_t rec;
    if (run_experiment(&cfg, &rec) != DEMO_OK)
        return DEMO_ERROR;
    record_run(&rec);
    g_settings.initial = initial;
    print_verdict(&rec);

    printf("\n--- Explanation ---\n"
           "  The child's change did not alter the parent's variable, so the\n"
           "  processes have isolated private memory.  Note: a plain local\n"
           "  variable shows this isolation but does not by itself prove that\n"
           "  a physical page was shared first or that a page fault occurred;\n"
           "  those are kernel mechanisms (see option 5).  Both processes also\n"
           "  report the same virtual address (parent 0x%" PRIxPTR
           ", child 0x%" PRIxPTR ")\n"
           "  - that is expected, and it does not mean they share one physical "
           "page.\n",
           rec.parent_init.addr, rec.child_init.addr);
    return DEMO_OK;
}

/* Option 2: user picks who modifies and both values. */
static int parent_child_modification_demo(void)
{
    print_banner("2. PARENT AND CHILD MEMORY MODIFICATION");
    long initial, modified;
    writer_t writer;

    if (read_integer_safely("Initial value", VALUE_MIN, VALUE_MAX,
                            g_settings.initial, true, &initial) != 0)
        return DEMO_EOF;
    if (read_writer_choice(g_settings.writer, &writer) != 0)
        return DEMO_EOF;
    if (read_distinct_value("Modification value", g_settings.modified, initial,
                            &modified) != 0)
        return DEMO_EOF;

    volatile int variable = (int)initial;
    experiment_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.name = "Parent/child modification (local variable)";
    cfg.slot0 = &variable;
    cfg.pages = 1;
    cfg.init_base = (int)initial;
    cfg.new_value = (int)modified;
    cfg.writer = writer;

    run_record_t rec;
    if (run_experiment(&cfg, &rec) != DEMO_OK)
        return DEMO_ERROR;
    record_run(&rec);
    g_settings.initial = initial;
    g_settings.modified = modified;
    g_settings.writer = (int)writer;
    print_verdict(&rec);
    printf("\n  Virtual addresses observed: parent 0x%" PRIxPTR
           ", child 0x%" PRIxPTR "\n"
           "  Same virtual address is normal after fork(); each process\n"
           "  translates it through its own page table.\n",
           rec.parent_init.addr, rec.child_init.addr);
    return DEMO_OK;
}

/* Option 3: mmap()ed region of several pages. */
static int multi_page_cow_demo(void)
{
    print_banner("3. DEMONSTRATE MULTIPLE MEMORY PAGES");
    size_t ps = get_page_size();
    if (ps == 0)
        return DEMO_ERROR;
    long max_pages = (long)(MAX_REGION_BYTES / ps);
    printf("System page size : %zu bytes\n", ps);
    printf("Safety limit     : %lu bytes (16 MiB) = %ld pages\n\n",
           MAX_REGION_BYTES, max_pages);

    long pages, offset, sel_page, base, modified;
    writer_t writer;
    bool proc = false;

    if (read_integer_safely("Number of pages", 1, max_pages,
                            g_settings.pages < max_pages ? g_settings.pages
                                                         : max_pages,
                            true, &pages) != 0)
        return DEMO_EOF;
    printf("Total mapping size = %ld pages x %zu bytes = %zu bytes\n", pages, ps,
           (size_t)pages * ps);
    if (read_integer_safely("Page index to modify", 0, pages - 1, 0, true,
                            &sel_page) != 0)
        return DEMO_EOF;
    if (read_integer_safely("Byte offset inside that page (rounded down to an "
                            "int boundary)",
                            0, (long)ps - 1, 0, true, &offset) != 0)
        return DEMO_EOF;
    if (read_integer_safely("Initial value of page 0 (page i gets value+i)",
                            VALUE_MIN, VALUE_MAX, 1000, true, &base) != 0)
        return DEMO_EOF;
    if (read_writer_choice(g_settings.writer, &writer) != 0)
        return DEMO_EOF;
    if (read_distinct_value("Modification value", g_settings.modified,
                            base + sel_page, &modified) != 0)
        return DEMO_EOF;
    if (read_yes_no("Collect optional /proc observations?", true, &proc) != 0)
        return DEMO_EOF;

    experiment_t cfg;
    if (build_region_experiment(&cfg, "Multi-page private mapping",
                                (size_t)pages, (size_t)offset, (int)base,
                                (size_t)sel_page, (int)modified, writer,
                                proc) != DEMO_OK)
        return DEMO_ERROR;
    printf("\nMapped %zu bytes at virtual address %p (MAP_PRIVATE | "
           "MAP_ANONYMOUS) and initialised every page.\n",
           cfg.region_size, cfg.region);

    run_record_t rec;
    int rc = run_experiment(&cfg, &rec);
    release_experiment(&cfg);
    printf("\nMapping released with munmap() (live mappings: %d).\n",
           g_live_mappings);
    if (rc != DEMO_OK)
        return DEMO_ERROR;

    record_run(&rec);
    g_settings.pages = pages;
    g_settings.modified = modified;
    g_settings.writer = (int)writer;
    print_verdict(&rec);
    print_page_table(&rec);
    if (proc)
        print_proc_table(&rec);

    printf("\n--- Explanation ---\n"
           "  Only page %ld was written, and only that page's value diverged;\n"
           "  the other %ld page(s) hold identical values in both processes.\n"
           "  COW works at page granularity (here %zu bytes): the kernel does\n"
           "  not copy the whole address space when one value changes.  This\n"
           "  program does not claim an exact number of physical pages was\n"
           "  copied - it only observes which values differ.\n",
           sel_page, pages - 1, ps);
    return DEMO_OK;
}

/* Option 4: show the last recorded run, or run a fresh one. */
static int compare_memory_states(void)
{
    print_banner("4. COMPARE MEMORY BEFORE AND AFTER MODIFICATION");
    long choice = 2;

    if (g_history_count > 0) {
        printf("  1. Show the comparison of the most recent run (%s)\n"
               "  2. Run a new experiment and compare\n",
               g_history[g_history_count - 1].name);
        if (read_integer_safely("Choice", 1, 2, 1, true, &choice) != 0)
            return DEMO_EOF;
    } else {
        printf("No earlier run recorded; running a new experiment.\n");
    }
    if (choice == 1) {
        print_comparison(&g_history[g_history_count - 1]);
        return DEMO_OK;
    }

    long initial, modified;
    writer_t writer;
    bool proc = false;
    if (read_integer_safely("Initial value of page 0", VALUE_MIN, VALUE_MAX,
                            g_settings.initial, true, &initial) != 0)
        return DEMO_EOF;
    if (read_writer_choice(g_settings.writer, &writer) != 0)
        return DEMO_EOF;
    if (read_distinct_value("Modification value", g_settings.modified, initial,
                            &modified) != 0)
        return DEMO_EOF;
    if (read_yes_no("Collect optional /proc observations?", false, &proc) != 0)
        return DEMO_EOF;

    experiment_t cfg;
    if (build_region_experiment(&cfg, "Comparison run (2-page private mapping)",
                                2, 0, (int)initial, 0, (int)modified, writer,
                                proc) != DEMO_OK)
        return DEMO_ERROR;
    run_record_t rec;
    int rc = run_experiment(&cfg, &rec);
    release_experiment(&cfg);
    if (rc != DEMO_OK)
        return DEMO_ERROR;

    record_run(&rec);
    g_settings.initial = initial;
    g_settings.modified = modified;
    g_settings.writer = (int)writer;
    print_comparison(&rec);
    return DEMO_OK;
}

/* Option 6: end-to-end demonstration following the 12 specified steps. */
static int run_complete_demo(void)
{
    print_banner("6. COMPLETE COPY-ON-WRITE DEMONSTRATION");
    size_t ps = get_page_size();
    if (ps == 0)
        return DEMO_ERROR;
    long max_pages = (long)(MAX_REGION_BYTES / ps);
    long initial, pages, sel_page, modified;
    writer_t writer;
    bool proc = true;

    if (read_integer_safely("Initial value of page 0 (page i gets value+i)",
                            VALUE_MIN, VALUE_MAX, g_settings.initial, true,
                            &initial) != 0)
        return DEMO_EOF;
    if (read_integer_safely("Number of pages", 1, max_pages,
                            g_settings.pages < max_pages ? g_settings.pages
                                                         : max_pages,
                            true, &pages) != 0)
        return DEMO_EOF;
    if (read_integer_safely("Page index to modify", 0, pages - 1,
                            pages > 1 ? 1 : 0, true, &sel_page) != 0)
        return DEMO_EOF;
    if (read_writer_choice(g_settings.writer, &writer) != 0)
        return DEMO_EOF;
    if (read_distinct_value("Modification value", g_settings.modified,
                            initial + sel_page, &modified) != 0)
        return DEMO_EOF;
    if (read_yes_no("Collect optional /proc observations?", true, &proc) != 0)
        return DEMO_EOF;

    printf("\nStep 1  Configuration: initial=%ld, pages=%ld x %zu bytes = %zu "
           "bytes, page %ld will be written by the %s, new value=%ld.\n",
           initial, pages, ps, (size_t)pages * ps, sel_page,
           writer == WRITER_PARENT ? "parent" : "child", modified);

    experiment_t cfg;
    printf("Step 2  Creating and initialising the private memory region...\n");
    if (build_region_experiment(&cfg, "Complete demo (private mmap region)",
                                (size_t)pages, 0, (int)initial,
                                (size_t)sel_page, (int)modified, writer,
                                proc) != DEMO_OK)
        return DEMO_ERROR;
    printf("        Region at %p, %zu bytes, MAP_PRIVATE | MAP_ANONYMOUS.\n",
           cfg.region, cfg.region_size);
    printf("Steps 3-9 (parent info, fork, initial values, modification,\n"
           "        writer's view, other process's view, synchronisation) and\n"
           "Step 10 (waitpid) run live below, ordered by pipe handshakes.\n");

    run_record_t rec;
    int rc = run_experiment(&cfg, &rec);
    release_experiment(&cfg);
    printf("\nStep 12 Region released with munmap(); live mappings = %d.\n",
           g_live_mappings);
    if (rc != DEMO_OK)
        return DEMO_ERROR;

    record_run(&rec);
    g_settings.initial = initial;
    g_settings.pages = pages;
    g_settings.modified = modified;
    g_settings.writer = (int)writer;

    printf("\nStep 11 Final results and explanation:\n");
    print_comparison(&rec);
    printf("\n--- What the experiment demonstrates ---\n"
           "  Real processes created by fork() started with identical data.\n"
           "  After one of them wrote to a private page, only that process saw\n"
           "  the new value; the other kept the original.  This is the\n"
           "  behaviour Copy-on-Write must preserve.  The values prove the\n"
           "  isolation; that Linux achieved it by a lazy page copy at the\n"
           "  first write is the standard kernel mechanism, not something\n"
           "  this program's pointer values can prove.\n");
    return DEMO_OK;
}

/* Option 7: clear configuration and history; verify nothing is left over. */
static void reset_demo(void)
{
    print_banner("7. RESET DEMONSTRATION");
    g_history_count = 0;
    memset(g_history, 0, sizeof g_history);
    g_settings.initial = DEFAULT_INITIAL;
    g_settings.modified = DEFAULT_NEW;
    g_settings.pages = DEFAULT_PAGES;
    g_settings.writer = WRITER_CHILD;

    int stray = reap_stray_children();
    printf("Configuration and run history cleared (defaults restored).\n");
    printf("Stray child processes reaped : %d\n", stray);
    printf("Memory mappings still open   : %d\n", g_live_mappings);
    printf("Only this program's own state was reset; the operating system is "
           "untouched.\n");
}

/* ------------------------------------------------------------------------ */
/* Section 10: main                                                          */
/* ------------------------------------------------------------------------ */

int main(void)
{
    /* A write to a pipe whose reader died should return EPIPE, not kill us. */
    signal(SIGPIPE, SIG_IGN);
    g_settings.initial = DEFAULT_INITIAL;
    g_settings.modified = DEFAULT_NEW;
    g_settings.pages = DEFAULT_PAGES;
    g_settings.writer = WRITER_CHILD;

    for (;;) {
        display_main_menu();
        long choice;
        if (read_integer_safely("Select an option", 0, 7, 0, false, &choice) != 0)
            break;

        int status = DEMO_OK;
        switch (choice) {
        case 1: status = basic_cow_demo(); break;
        case 2: status = parent_child_modification_demo(); break;
        case 3: status = multi_page_cow_demo(); break;
        case 4: status = compare_memory_states(); break;
        case 5: explain_cow_concepts(); break;
        case 6: status = run_complete_demo(); break;
        case 7: reset_demo(); break;
        case 0:
            printf("Exiting.\n");
            goto done;
        default: break;
        }

        if (status == DEMO_EOF)
            break;
        if (status == DEMO_ERROR)
            printf("\nThe demonstration could not be completed (see messages "
                   "above). Returning to the menu.\n");
        if (pause_for_enter() == DEMO_EOF)
            break;
    }
done:;
    int stray = reap_stray_children();
    if (stray > 0)
        printf("Reaped %d stray child process(es) at exit.\n", stray);
    if (g_live_mappings != 0)
        printf("Warning: %d mapping(s) still open at exit.\n", g_live_mappings);
    return 0;
}
