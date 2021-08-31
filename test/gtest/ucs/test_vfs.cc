/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/debug/memtrack_int.h>
#include <ucs/vfs/base/vfs_cb.h>
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

    static ucs_status_t file_write_cb(void *obj, const char *buffer,
                                      size_t size, void *arg_ptr,
                                      uint64_t arg_u64)
    {
        int *arg = (int*)arg_ptr;
        *arg     = atoi(buffer);
        return UCS_OK;
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
    EXPECT_UCS_OK(ucs_vfs_path_get_info("/obj", &path_info));
    EXPECT_EQ(1, path_info.size);
    EXPECT_TRUE(path_info.mode & S_IFDIR);

    EXPECT_UCS_OK(ucs_vfs_path_get_info("/obj/info", &path_info));
    EXPECT_EQ(file_content().size(), path_info.size);
    EXPECT_TRUE(path_info.mode & S_IFREG);

    EXPECT_EQ(UCS_ERR_NO_ELEM,
              ucs_vfs_path_get_info("invalid_path", &path_info));

    barrier();
    ucs_vfs_obj_remove(obj);
}

UCS_MT_TEST_F(test_vfs_obj, path_read_file, 4) {
    void *obj = create_simple_tree();

    ucs_string_buffer_t strb;
    ucs_string_buffer_init(&strb);
    EXPECT_EQ(UCS_ERR_NO_ELEM, ucs_vfs_path_read_file("/obj", &strb));

    EXPECT_UCS_OK(ucs_vfs_path_read_file("/obj/info", &strb));
    EXPECT_EQ(file_content(), ucs_string_buffer_cstr(&strb));

    EXPECT_EQ(UCS_ERR_NO_ELEM, ucs_vfs_path_read_file("invalid_path", &strb));
    ucs_string_buffer_cleanup(&strb);

    barrier();
    ucs_vfs_obj_remove(obj);
}

UCS_MT_TEST_F(test_vfs_obj, path_list_dir, 4) {
    void *obj = create_simple_tree();

    char buffer[32];
    EXPECT_UCS_OK(
            ucs_vfs_path_list_dir("/obj", test_vfs_obj::list_dir_cb, buffer));

    EXPECT_STREQ("info", buffer);

    EXPECT_EQ(UCS_ERR_NO_ELEM,
              ucs_vfs_path_list_dir("/obj/info", test_vfs_obj::list_dir_cb,
                                    buffer));

    EXPECT_EQ(UCS_ERR_NO_ELEM,
              ucs_vfs_path_list_dir("invalid_path", test_vfs_obj::list_dir_cb,
                                    buffer));

    barrier();
    ucs_vfs_obj_remove(obj);
}

UCS_MT_TEST_F(test_vfs_obj, set_dirty_and_refresh, 4) {
    static char obj;
    ucs_vfs_obj_add_dir(NULL, &obj, "obj");

    ucs_vfs_path_info_t path_info;
    EXPECT_UCS_OK(ucs_vfs_path_get_info("/obj", &path_info));
    EXPECT_EQ(0, path_info.size);

    barrier();
    ucs_vfs_obj_set_dirty(&obj, test_vfs_obj::refresh_cb);

    EXPECT_UCS_OK(ucs_vfs_path_get_info("/obj", &path_info));
    EXPECT_EQ(1, path_info.size);

    barrier();
    ucs_vfs_obj_remove(&obj);
}

UCS_TEST_F(test_vfs_obj, check_ret) {
    char obj1, obj2;

    EXPECT_UCS_OK(ucs_vfs_obj_add_dir(NULL, &obj1, "obj"));
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS, ucs_vfs_obj_add_dir(NULL, &obj1, "obj"));
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, ucs_vfs_obj_add_dir(&obj2, &obj1, "obj"));

    EXPECT_UCS_OK(ucs_vfs_obj_add_ro_file(&obj1, test_vfs_obj::file_show_cb,
                                          NULL, 0, "info"));
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS,
              ucs_vfs_obj_add_ro_file(&obj1, test_vfs_obj::file_show_cb, NULL,
                                      0, "info"));
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucs_vfs_obj_add_ro_file(&obj2, test_vfs_obj::file_show_cb, NULL,
                                      0, "info"));

    /* Check creating dir with same name */
    EXPECT_UCS_OK(ucs_vfs_obj_add_dir(NULL, &obj2, "obj"));

    ucs_vfs_obj_remove(&obj1);
    ucs_vfs_obj_remove(&obj2);
}

UCS_MT_TEST_F(test_vfs_obj, add_sym_link, 4)
{
    static const char path_to_target[] = "target";

    static char target;
    ucs_vfs_obj_add_dir(NULL, &target, "target");
    ucs_vfs_obj_add_sym_link(NULL, &target, "link");

    ucs_vfs_path_info_t path_info;
    EXPECT_UCS_OK(ucs_vfs_path_get_info("/link", &path_info));
    EXPECT_EQ(strlen(path_to_target), path_info.size);
    EXPECT_TRUE(path_info.mode & S_IFLNK);

    ucs_string_buffer_t strb;
    ucs_string_buffer_init(&strb);
    EXPECT_UCS_OK(ucs_vfs_path_get_link("/link", &strb));
    EXPECT_STREQ(path_to_target, ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);

    barrier();
    ucs_vfs_obj_remove(&target);
}

UCS_TEST_F(test_vfs_obj, add_rw_check_ret) {
    int obj = 0, no_obj;

    EXPECT_UCS_OK(ucs_vfs_obj_add_dir(NULL, &obj, "obj"));

    EXPECT_UCS_OK(ucs_vfs_obj_add_rw_file(&obj, ucs_vfs_show_primitive,
                                          test_vfs_obj::file_write_cb, &obj,
                                          UCS_VFS_TYPE_INT, "info"));
    EXPECT_EQ(UCS_ERR_ALREADY_EXISTS,
              ucs_vfs_obj_add_rw_file(&obj, ucs_vfs_show_primitive,
                                      test_vfs_obj::file_write_cb, &obj,
                                      UCS_VFS_TYPE_INT, "info"));
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucs_vfs_obj_add_rw_file(&no_obj, ucs_vfs_show_primitive,
                                      test_vfs_obj::file_write_cb, &obj,
                                      UCS_VFS_TYPE_INT, "info"));

    ucs_vfs_obj_remove(&obj);
}

UCS_MT_TEST_F(test_vfs_obj, rw_file, 4)
{
    static int obj = 0;
    ucs_vfs_obj_add_dir(NULL, &obj, "obj");
    ucs_vfs_obj_add_rw_file(&obj, ucs_vfs_show_primitive,
                            test_vfs_obj::file_write_cb, &obj, UCS_VFS_TYPE_INT,
                            "info");

    ucs_vfs_path_info_t path_info;
    EXPECT_UCS_OK(ucs_vfs_path_get_info("/obj/info", &path_info));
    EXPECT_EQ(2, path_info.size);
    EXPECT_TRUE(path_info.mode & S_IWUSR);

    barrier();

    const char new_value[] = "777\n";
    EXPECT_UCS_OK(
            ucs_vfs_path_write_file("/obj/info", new_value, sizeof(new_value)));
    EXPECT_EQ(UCS_ERR_NO_ELEM,
              ucs_vfs_path_write_file("/obj", new_value, sizeof(new_value)));

    barrier();

    ucs_string_buffer_t strb;
    ucs_string_buffer_init(&strb);
    EXPECT_UCS_OK(ucs_vfs_path_read_file("/obj/info", &strb));
    EXPECT_STREQ(new_value, ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);

    barrier();

    ucs_vfs_obj_remove(&obj);
}
