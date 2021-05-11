/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/debug/memtrack.h>
#include <ucs/vfs/base/vfs_obj.h>
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

class test_vfs_obj : public ucs::test {
public:
    static void file_show_cb(void *obj, ucs_string_buffer_t *strb,
                             void *arg_ptr, uint64_t arg_u64)
    {
        ucs_string_buffer_appendf(strb, "%s", file_content().c_str());
    }

    static std::string file_content()
    {
        return "info";
    }

    static void list_dir_cb(const char *name, void *arg)
    {
        char *buffer = static_cast<char*>(arg);

        strcpy(buffer, name);
    }

    static void refresh_cb(void *obj)
    {
        ucs_vfs_obj_add_ro_file(obj, test_vfs_obj::file_show_cb, NULL, 0,
                                "info");
    }

    static void *create_simple_tree()
    {
        static char obj;
        ucs_vfs_obj_add_dir(NULL, &obj, "obj");
        ucs_vfs_obj_add_ro_file(&obj, test_vfs_obj::file_show_cb, NULL, 0,
                                "info");
        return &obj;
    }
};

UCS_MT_TEST_F(test_vfs_obj, simple_obj_tree, 4) {
    char obj1, obj2, obj3, obj4;

    /**
     * obj1
     * |
     * |____obj2
     * |    |
     * |    |____obj3
     * |
     * |____obj4
     */

    ucs_vfs_obj_add_dir(NULL, &obj1, "obj1");
    ucs_vfs_obj_add_dir(&obj1, &obj2, "obj2");
    ucs_vfs_obj_add_dir(&obj2, &obj3, "obj3");
    ucs_vfs_obj_add_dir(&obj1, &obj4, "obj4");
    ucs_vfs_obj_remove(&obj1);
}

UCS_MT_TEST_F(test_vfs_obj, remove_middle_obj, 4) {
    char obj1, obj2, obj3;

    ucs_vfs_obj_add_dir(NULL, &obj1, "obj1");
    ucs_vfs_obj_add_dir(&obj1, &obj2, "subdir/obj2");
    ucs_vfs_obj_add_dir(&obj2, &obj3, "obj3");
    ucs_vfs_obj_remove(&obj2);
    ucs_vfs_obj_remove(&obj1);
}

UCS_MT_TEST_F(test_vfs_obj, path_get_info, 4) {
    void *obj = create_simple_tree();

    ucs_vfs_path_info_t path_info;
    ucs_status_t status = ucs_vfs_path_get_info("/obj", &path_info);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_EQ(path_info.size, 1);
    EXPECT_TRUE(path_info.mode & S_IFDIR);

    status = ucs_vfs_path_get_info("/obj/info", &path_info);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_EQ(path_info.size, file_content().size());
    EXPECT_TRUE(path_info.mode & S_IFREG);

    status = ucs_vfs_path_get_info("invalid_path", &path_info);
    EXPECT_EQ(status, UCS_ERR_NO_ELEM);

    barrier();
    ucs_vfs_obj_remove(obj);
}

UCS_MT_TEST_F(test_vfs_obj, path_read_file, 4) {
    void *obj = create_simple_tree();

    ucs_string_buffer_t strb;
    ucs_string_buffer_init(&strb);
    ucs_status_t status = ucs_vfs_path_read_file("/obj", &strb);
    EXPECT_EQ(status, UCS_ERR_NO_ELEM);

    status = ucs_vfs_path_read_file("/obj/info", &strb);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_EQ(file_content(), ucs_string_buffer_cstr(&strb));

    status = ucs_vfs_path_read_file("invalid_path", &strb);
    EXPECT_EQ(status, UCS_ERR_NO_ELEM);
    ucs_string_buffer_cleanup(&strb);

    barrier();
    ucs_vfs_obj_remove(obj);
}

UCS_MT_TEST_F(test_vfs_obj, path_list_dir, 4) {
    void *obj = create_simple_tree();

    char buffer[32];
    ucs_status_t status = ucs_vfs_path_list_dir("/obj",
                                                test_vfs_obj::list_dir_cb,
                                                buffer);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_STREQ(buffer, "info");

    status = ucs_vfs_path_list_dir("/obj/info", test_vfs_obj::list_dir_cb,
                                   buffer);
    EXPECT_EQ(status, UCS_ERR_NO_ELEM);

    status = ucs_vfs_path_list_dir("invalid_path", test_vfs_obj::list_dir_cb,
                                   buffer);
    EXPECT_EQ(status, UCS_ERR_NO_ELEM);

    barrier();
    ucs_vfs_obj_remove(obj);
}

UCS_MT_TEST_F(test_vfs_obj, set_dirty_and_refresh, 4) {
    static char obj;
    ucs_vfs_obj_add_dir(NULL, &obj, "obj");

    ucs_vfs_path_info_t path_info;
    ucs_status_t status = ucs_vfs_path_get_info("/obj", &path_info);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_EQ(path_info.size, 0);

    barrier();
    ucs_vfs_obj_set_dirty(&obj, test_vfs_obj::refresh_cb);

    status = ucs_vfs_path_get_info("/obj", &path_info);
    EXPECT_EQ(status, UCS_OK);
    EXPECT_EQ(path_info.size, 1);

    barrier();
    ucs_vfs_obj_remove(&obj);
}
