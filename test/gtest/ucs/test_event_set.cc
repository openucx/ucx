/**
* Copyright (C) Hiroyuki Sato. 2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/event_set.h>
#include <pthread.h>
#include <sys/epoll.h>
}

#define MAX_BUF_LEN        255

static const char *UCS_EVENT_SET_TEST_STRING  = "ucs_event_set test string";
static const char *UCS_EVENT_SET_EXTRA_STRING = "ucs_event_set extra string";
static const int   UCS_EVENT_SET_EXTRA_NUM    = 0xFF;

enum {
    UCS_EVENT_SET_EXTERNAL_FD = UCS_BIT(0),
};

class test_event_set : public ucs::test_base,
                       public ::testing::TestWithParam<int> {
public:
    static const char *evfd_data;
    static pthread_barrier_t barrier;

    typedef void* (*event_set_pthread_callback_t)(void *arg);

    enum event_set_op_t {
        EVENT_SET_OP_ADD,
        EVENT_SET_OP_MOD,
        EVENT_SET_OP_DEL
    };

    UCS_TEST_BASE_IMPL;

protected:
    void init() {
        if (GetParam() & UCS_EVENT_SET_EXTERNAL_FD) {
            m_ext_fd = epoll_create(1);
            ASSERT_TRUE(m_ext_fd > 0);
        } else {
            m_ext_fd = -1;
        }
    }

    void cleanup() {
        if (GetParam() & UCS_EVENT_SET_EXTERNAL_FD) {
            ASSERT_NE(-1, m_ext_fd);
            close(m_ext_fd);
            m_ext_fd = -1;
        }
    }

    static void* event_set_read_func(void *arg) {
        int *fd = (int *)arg;
        int n;

        n = write(fd[1], evfd_data, strlen(test_event_set::evfd_data));
        if (n == -1) {
            ADD_FAILURE();
        }

        thread_barrier();
        return 0;
    }

    static void* event_set_tmo_func(void *arg) {
        thread_barrier();
        return 0;
    }

    void event_set_init(event_set_pthread_callback_t func) {
        ucs_status_t status;
        int ret;

        if (pipe(m_pipefd) == -1) {
            UCS_TEST_ABORT("pipe() failed with error - " <<
                           strerror(errno));
        }

        ret = pthread_barrier_init(&barrier, NULL, 2);
        if (ret) {
            UCS_TEST_ABORT("pthread_barrier_init() failed with error - " <<
                           strerror(errno));
        }

        ret = pthread_create(&m_tid, NULL, func, (void *)&m_pipefd);
        if (ret) {
            UCS_TEST_ABORT("pthread_create() failed with error - " <<
                           strerror(errno));
        }

        if (GetParam() & UCS_EVENT_SET_EXTERNAL_FD) {
            status = ucs_event_set_create_from_fd(&m_event_set, m_ext_fd);
        } else {
            status = ucs_event_set_create(&m_event_set);
        }
        ASSERT_UCS_OK(status);
        EXPECT_TRUE(m_event_set != NULL);
    }

    void event_set_cleanup() {
        ucs_event_set_cleanup(m_event_set);

        pthread_join(m_tid, NULL);
        pthread_barrier_destroy(&barrier);

        close(m_pipefd[0]);
        close(m_pipefd[1]);
    }

    void event_set_ctl(event_set_op_t op, int fd, ucs_event_set_types_t events) {
        ucs_status_t status = UCS_OK;

        switch (op) {
        case EVENT_SET_OP_ADD:
            status = ucs_event_set_add(m_event_set, fd, events,
                                       (void *)(uintptr_t)fd);
            break;
        case EVENT_SET_OP_MOD:
            status = ucs_event_set_mod(m_event_set, fd, events,
                                       (void *)(uintptr_t)fd);
            break;
        case EVENT_SET_OP_DEL:
            status = ucs_event_set_del(m_event_set, fd);
            break;
        default:
            UCS_TEST_ABORT("unknown event set operation - " << op);
        }

        EXPECT_UCS_OK(status);
    }

    void event_set_wait(unsigned exp_event, int timeout_ms,
                        ucs_event_set_handler_t handler, void *arg) {
        unsigned nread  = ucs_sys_event_set_max_wait_events;
        ucs_status_t status;

        /* Check for events on pipe fd */
        status = ucs_event_set_wait(m_event_set, &nread, 0, handler, arg);
        EXPECT_EQ(exp_event, nread);
        EXPECT_UCS_OK(status);
    }

    static void thread_barrier() {
        int ret = pthread_barrier_wait(&barrier);
        EXPECT_TRUE((ret == 0) || (ret == PTHREAD_BARRIER_SERIAL_THREAD));
    }

    int                  m_pipefd[2];
    int                  m_ext_fd;
    pthread_t            m_tid;
    ucs_sys_event_set_t *m_event_set;
};

const char *test_event_set::evfd_data = UCS_EVENT_SET_TEST_STRING;

pthread_barrier_t test_event_set::barrier;

static void event_set_func1(void *callback_data, ucs_event_set_types_t events,
                            void *arg)
{
    char buf[MAX_BUF_LEN];
    char *extra_str = (char *)((void**)arg)[0];
    int *extra_num = (int *)((void**)arg)[1];
    int n;
    int fd = (int)(uintptr_t)callback_data;
    memset(buf, 0, MAX_BUF_LEN);

    EXPECT_EQ(UCS_EVENT_SET_EVREAD, events);

    n = read(fd, buf, MAX_BUF_LEN - 1);
    if (n == -1) {
        ADD_FAILURE();
        return;
    }
    EXPECT_EQ(0, strncmp(UCS_EVENT_SET_TEST_STRING, buf, n));
    EXPECT_EQ(0, strncmp(UCS_EVENT_SET_EXTRA_STRING, extra_str,n ));
    EXPECT_EQ(UCS_EVENT_SET_EXTRA_NUM, *extra_num);
}

static void event_set_func2(void *callback_data, ucs_event_set_types_t events,
                            void *arg)
{
    EXPECT_EQ(UCS_EVENT_SET_EVWRITE, events);
}

static void event_set_func3(void *callback_data, ucs_event_set_types_t events,
                            void *arg)
{
    ADD_FAILURE();
}

static void event_set_func4(void *callback_data, ucs_event_set_types_t events,
                            void *arg)
{
    EXPECT_EQ(UCS_EVENT_SET_EVREAD, events);
}

UCS_TEST_P(test_event_set, ucs_event_set_read_thread) {
    void *arg[] = { (void*)UCS_EVENT_SET_EXTRA_STRING,
                    (void*)&UCS_EVENT_SET_EXTRA_NUM };

    event_set_init(event_set_read_func);
    event_set_ctl(EVENT_SET_OP_ADD, m_pipefd[0],
                  UCS_EVENT_SET_EVREAD);

    thread_barrier();

    event_set_wait(1u, -1, event_set_func1, arg);

    event_set_ctl(EVENT_SET_OP_DEL, m_pipefd[0], 0);
    event_set_cleanup();
}

UCS_TEST_P(test_event_set, ucs_event_set_write_thread) {
    event_set_init(event_set_read_func);
    event_set_ctl(EVENT_SET_OP_ADD, m_pipefd[1],
                  UCS_EVENT_SET_EVWRITE);

    thread_barrier();

    event_set_wait(1u, -1, event_set_func2, NULL);

    event_set_ctl(EVENT_SET_OP_DEL, m_pipefd[1], 0);
    event_set_cleanup();
}

UCS_TEST_P(test_event_set, ucs_event_set_tmo_thread) {
    event_set_init(event_set_tmo_func);
    event_set_ctl(EVENT_SET_OP_ADD, m_pipefd[0],
                  UCS_EVENT_SET_EVREAD);

    thread_barrier();

    event_set_wait(0u, 0, event_set_func3, NULL);

    event_set_ctl(EVENT_SET_OP_DEL, m_pipefd[0], 0);
    event_set_cleanup();
}

UCS_TEST_P(test_event_set, ucs_event_set_trig_modes) {
    void *arg[] = { (void*)UCS_EVENT_SET_EXTRA_STRING,
                    (void*)&UCS_EVENT_SET_EXTRA_NUM };

    event_set_init(event_set_read_func);
    event_set_ctl(EVENT_SET_OP_ADD, m_pipefd[0],
                  UCS_EVENT_SET_EVREAD);

    thread_barrier();

    /* Test level-triggered mode (default) */
    for (int i = 0; i < 10; i++) {
        event_set_wait(1u, 0, event_set_func4, NULL);
    }

    /* Test edge-triggered mode */
    /* Set edge-triggered mode */
    event_set_ctl(EVENT_SET_OP_MOD, m_pipefd[0],
                  UCS_EVENT_SET_EVREAD | UCS_EVENT_SET_EDGE_TRIGGERED);

    /* Should have only one event to read */
    event_set_wait(1u, 0, event_set_func4, NULL);

    /* Should not read nothing */
    for (int i = 0; i < 10; i++) {
        event_set_wait(0u, 0, event_set_func1, arg);
    }

    /* Call the function below directly to read
     * all outstanding data from pipe fd */
    event_set_func1((void*)(uintptr_t)m_pipefd[0], UCS_EVENT_SET_EVREAD, arg);

    event_set_ctl(EVENT_SET_OP_DEL, m_pipefd[0], 0);
    event_set_cleanup();
}

INSTANTIATE_TEST_SUITE_P(ext_fd, test_event_set,
                        ::testing::Values(static_cast<int>(
                                              UCS_EVENT_SET_EXTERNAL_FD)));
INSTANTIATE_TEST_SUITE_P(int_fd, test_event_set, ::testing::Values(0));
