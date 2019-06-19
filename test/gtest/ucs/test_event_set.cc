/**
* Copyright (C) Hiroyuki Sato. 2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/event_set.h>
#include <pthread.h>
}

#define MAX_BUF_LEN        255
#define MAX_EVENT_SET_SIZE 16

static const char *UCS_EVENT_SET_TEST_STRING  = "ucs_event_set test string";
static const char *UCS_EVENT_SET_EXTRA_STRING = "ucs_event_set extra string";
static const int   UCS_EVENT_SET_EXTRA_NUM    = 0xFF;

class test_event_set : public ucs::test {
public:
    static const char *evfd_data;

protected:
    static void* event_set_read_func(void *arg) {
        int *fd = (int *)arg;
        int n;
        usleep(10000);
        n = write(fd[1], evfd_data, strlen(test_event_set::evfd_data));
        if (n == -1) {
            ADD_FAILURE();
        }
        return 0;
    }

    static void* event_set_tmo_func(void *arg) {
        usleep(50);
        return 0;
    }
};

const char *test_event_set::evfd_data = UCS_EVENT_SET_TEST_STRING;

static void event_set_func1(void *callback_data, int events, void *arg)
{
    char buf[MAX_BUF_LEN];
    char *extra_str = (char *)((void**)arg)[0];
    int *extra_num = (int *)((void**)arg)[1];
    int n;
    int fd = (int)(uintptr_t)callback_data;
    memset(buf, 0, MAX_BUF_LEN);

    EXPECT_EQ(UCS_EVENT_SET_EVREAD, events);
    n = read(fd, buf, MAX_BUF_LEN);
    if (n==-1) {
        ADD_FAILURE();
        return;
    }
    EXPECT_TRUE(strcmp(UCS_EVENT_SET_TEST_STRING, buf) == 0);
    EXPECT_TRUE(strcmp(UCS_EVENT_SET_EXTRA_STRING, extra_str) == 0);
    EXPECT_EQ(*extra_num, UCS_EVENT_SET_EXTRA_NUM);
}

static void event_set_func2(void *callback_data, int events, void *arg)
{
    EXPECT_EQ(UCS_EVENT_SET_EVWRITE, events);
}

UCS_TEST_F(test_event_set, ucs_event_set_read_thread) {
    pthread_t tid;
    int ret;
    int pipefd[2];
    unsigned nread;
    void *arg[] = { (void*)UCS_EVENT_SET_EXTRA_STRING,
                    (void*)&UCS_EVENT_SET_EXTRA_NUM };
    ucs_sys_event_set_t *event_set = NULL;
    ucs_status_t status;

    if (pipe(pipefd) == -1) {
        UCS_TEST_MESSAGE << strerror(errno);
        throw ucs::test_abort_exception();
    }

    ret = pthread_create(&tid, NULL, event_set_read_func, (void *)&pipefd);
    if (ret) {
        UCS_TEST_MESSAGE << strerror(errno);
        throw ucs::test_abort_exception();
    }

    status = ucs_event_set_create(&event_set);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_TRUE(event_set != NULL);

    status = ucs_event_set_add(event_set, pipefd[0], UCS_EVENT_SET_EVREAD,
                               (void *)(uintptr_t)pipefd[0]);
    EXPECT_EQ(UCS_OK, status);

    status = ucs_event_set_wait(event_set, 1, 50,
                                event_set_func1, arg, &nread);
    EXPECT_EQ(1u, nread);
    EXPECT_EQ(UCS_OK, status);
    ucs_event_set_cleanup(event_set);

    pthread_join(tid, NULL);

    close(pipefd[0]);
    close(pipefd[1]);
}

UCS_TEST_F(test_event_set, ucs_event_set_write_thread) {
    pthread_t tid;
    int ret;
    int pipefd[2];
    ucs_sys_event_set_t *event_set = NULL;
    ucs_status_t status;
    unsigned nread;

    if (pipe(pipefd) == -1) {
        UCS_TEST_MESSAGE << strerror(errno);
        throw ucs::test_abort_exception();
    }

    ret = pthread_create(&tid, NULL, event_set_read_func, (void *)&pipefd);
    if (ret) {
        UCS_TEST_MESSAGE << strerror(errno);
        throw ucs::test_abort_exception();
    }

    status = ucs_event_set_create(&event_set);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_TRUE(event_set != NULL);

    status = ucs_event_set_add(event_set, pipefd[1], UCS_EVENT_SET_EVWRITE,
                               (void *)&pipefd[1]);
    EXPECT_EQ(UCS_OK, status);

    status = ucs_event_set_wait(event_set, 1, 50,
                                event_set_func2, NULL, &nread);
    EXPECT_EQ(1u, nread);
    EXPECT_EQ(UCS_OK, status);
    ucs_event_set_cleanup(event_set);

    pthread_join(tid, NULL);

    close(pipefd[0]);
    close(pipefd[1]);
}

