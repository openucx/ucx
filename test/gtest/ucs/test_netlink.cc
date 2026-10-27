/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/netlink.h>
}

#include <net/if.h>
#include <libgen.h>
#include <limits.h>
#include <unistd.h>


class test_netlink : public ucs::test {
protected:
    static std::string sysfs_master_name(const char *if_name)
    {
        std::string path = std::string("/sys/class/net/") + if_name +
                           "/master";
        char target[PATH_MAX];
        ssize_t len;

        len = readlink(path.c_str(), target, sizeof(target) - 1);
        if (len < 0) {
            return "";
        }

        target[len] = '\0';
        return basename(target);
    }
};

UCS_TEST_F(test_netlink, get_vrf_master_index) {
    struct if_nameindex *ifs = if_nameindex();
    unsigned num_vrf_slaves  = 0;
    char master_name[IFNAMSIZ];

    ASSERT_NE(nullptr, ifs);

    for (struct if_nameindex *it = ifs; it->if_index != 0; ++it) {
        int master_index = ucs_netlink_get_vrf_master_index(it->if_index);
        if (master_index == -1) {
            continue;
        }

        /* A VRF slave must be linked to its master in sysfs */
        ASSERT_NE(nullptr, if_indextoname(master_index, master_name))
                << it->if_name;
        EXPECT_EQ(sysfs_master_name(it->if_name), master_name) << it->if_name;
        ++num_vrf_slaves;
    }

    if_freenameindex(ifs);

    EXPECT_EQ(-1, ucs_netlink_get_vrf_master_index(if_nametoindex("lo")));
    UCS_TEST_MESSAGE << num_vrf_slaves << " VRF slave interfaces found";
}
