/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/sys.h>
#include <ucs/type/spinlock.h>
#include <ucs/time/time.h>
}

#include <sys/mman.h>
#include <set>

class test_sys : public ucs::test {
protected:
    static int get_mem_prot(void *address, size_t size) {
        return ucs_get_mem_prot((uintptr_t)address, (uintptr_t)address + size);
    }
};

UCS_TEST_F(test_sys, uuid) {
    std::set<uint64_t> uuids;
    for (unsigned i = 0; i < 10000; ++i) {
        uint64_t uuid = ucs_generate_uuid(0);
        std::pair<std::set<uint64_t>::iterator, bool> ret = uuids.insert(uuid);
        ASSERT_TRUE(ret.second);
    }
}

UCS_TEST_F(test_sys, machine_guid) {
    uint64_t guid1 = ucs_machine_guid();
    uint64_t guid2 = ucs_machine_guid();
    EXPECT_EQ(guid1, guid2);
}

UCS_TEST_F(test_sys, spinlock) {
    ucs_spinlock_t lock;
    pthread_t self;

    self = pthread_self();

    ucs_spinlock_init(&lock);

    ucs_spin_lock(&lock);
    EXPECT_TRUE(ucs_spin_is_owner(&lock, self));

    /* coverity[double_lock] */
    ucs_spin_lock(&lock);
    EXPECT_TRUE(ucs_spin_is_owner(&lock, self));

    ucs_spin_unlock(&lock);
    EXPECT_TRUE(ucs_spin_is_owner(&lock, self));

    ucs_spin_unlock(&lock);
    EXPECT_FALSE(ucs_spin_is_owner(&lock, self));
}

UCS_TEST_F(test_sys, get_mem_prot) {
    int x;

    ASSERT_TRUE( get_mem_prot(&x, sizeof(x)) & PROT_READ );
    ASSERT_TRUE( get_mem_prot(&x, sizeof(x)) & PROT_WRITE );
    ASSERT_TRUE( get_mem_prot((void*)&get_mem_prot, 1) & PROT_EXEC );

    ucs_time_t start_time = ucs_get_time();
    get_mem_prot(&x, sizeof(x));
    ucs_time_t duration = ucs_get_time() - start_time;
    UCS_TEST_MESSAGE << "Time: " << ucs_time_to_usec(duration) << " us";
}

UCS_TEST_F(test_sys, fcntl) {
    ucs_status_t status;
    int fd, fl;

    fd = open("/dev/null", O_RDONLY);
    if (fd < 0) {
        FAIL();
    }

    /* Add */
    status = ucs_sys_fcntl_modfl(fd, O_NONBLOCK, 0);
    EXPECT_TRUE(status == UCS_OK);

    fl = fcntl(fd, F_GETFL);
    EXPECT_GE(fl, 0);
    EXPECT_TRUE(fl & O_NONBLOCK);

    /* Remove */
    status = ucs_sys_fcntl_modfl(fd, 0, O_NONBLOCK);
    EXPECT_TRUE(status == UCS_OK);

    fl = fcntl(fd, F_GETFL);
    EXPECT_GE(fl, 0);
    EXPECT_FALSE(fl & O_NONBLOCK);

    close(fd);
}

UCS_TEST_F(test_sys, memory) {
    size_t phys_size = ucs_get_phys_mem_size();
    UCS_TEST_MESSAGE << "Physical memory size: " << ucs::size_value(phys_size);
    EXPECT_GT(phys_size, 1ul * 1024 * 1024);
}
