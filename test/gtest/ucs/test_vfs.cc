/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/vfs/sock/vfs_sock.h>
}

#include <sys/fcntl.h>
#include <time.h>


class test_vfs_sock : public ucs::test
{
protected:
    virtual void init()
    {
        int ret = socketpair(AF_UNIX, SOCK_STREAM, 0, m_sockets);
        if (ret < 0) {
            UCS_TEST_ABORT("failed to create socket pair: " << strerror(errno));
        }

        /* socket[1] will always receive credentials */
        ucs_vfs_sock_setopt_passcred(m_sockets[1]);
    }

    virtual void cleanup()
    {
        close(m_sockets[1]);
        close(m_sockets[0]);
    }

protected:
    void
    do_send_recv(ucs_vfs_sock_action_t action, int send_sockfd, int recv_sockfd,
                 int fd_in, ucs_vfs_sock_message_t *msg_out)
    {
        int ret;

        ucs_vfs_sock_message_t msg_in = {};
        msg_in.action                 = action;
        msg_in.fd                     = fd_in;
        ret = ucs_vfs_sock_send(send_sockfd, &msg_in);
        ASSERT_EQ(0, ret) << strerror(-ret);

        ret = ucs_vfs_sock_recv(recv_sockfd, msg_out);
        ASSERT_EQ(0, ret) << strerror(-ret);
        EXPECT_EQ(action, msg_out->action);
    }

    ino_t fd_inode(int fd)
    {
        struct stat st;
        int ret = fstat(fd, &st);
        if (ret < 0) {
            UCS_TEST_ABORT("stat() failed: " << strerror(errno));
        }
        return st.st_ino;
    }

    int m_sockets[2];
};

UCS_TEST_F(test_vfs_sock, send_recv_stop) {
    /* send stop/start commands from socket[0] to socket[1] */
    ucs_vfs_sock_message_t msg_out = {};
    do_send_recv(UCS_VFS_SOCK_ACTION_STOP, m_sockets[0], m_sockets[1], -1,
                 &msg_out);
}

UCS_TEST_F(test_vfs_sock, send_recv_mount) {
    /* send mount request from socket[0] to socket[1] */
    ucs_vfs_sock_message_t msg_out = {};
    do_send_recv(UCS_VFS_SOCK_ACTION_MOUNT, m_sockets[0], m_sockets[1], -1,
                 &msg_out);
    EXPECT_EQ(getpid(), msg_out.pid);
}

UCS_TEST_F(test_vfs_sock, send_recv_mount_reply) {
    /* open a file */
    int fd = open("/dev/null", O_WRONLY);
    if (fd < 0) {
        UCS_TEST_ABORT("failed to open /dev/null: " << strerror(errno));
    }

    /* send mount reply with fd from socket[1] to socket[0] */
    ucs_vfs_sock_message_t msg_out = {};
    do_send_recv(UCS_VFS_SOCK_ACTION_MOUNT_REPLY, m_sockets[1], m_sockets[0],
                 fd, &msg_out);

    UCS_TEST_MESSAGE << "send fd: " << fd << " recv fd: " << msg_out.fd;
    /* expect to have different fd but same inode */
    ASSERT_NE(msg_out.fd, fd);
    EXPECT_EQ(fd_inode(fd), fd_inode(msg_out.fd));

    close(msg_out.fd);
    close(fd);
}

UCS_TEST_F(test_vfs_sock, send_recv_nop) {
    /* send stop/start commands from socket[0] to socket[1] */
    ucs_vfs_sock_message_t msg_out = {};
    do_send_recv(UCS_VFS_SOCK_ACTION_NOP, m_sockets[0], m_sockets[1], -1,
                 &msg_out);
}
