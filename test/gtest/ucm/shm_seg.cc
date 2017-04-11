/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

extern "C" {
#include <ucm/util/sys.h>
}
#include <common/test.h>

#include <sstream>
#include <list>

static const char test_file_01[] = 
"00400000-00885000 r-xp 00000000 00:37 54889461                           /test/gtest/gtest\n"
"00a84000-00a85000 r--p 00484000 00:37 54889461                           /test/gtest/gtest\n"
"00a85000-00a88000 rw-p 00485000 00:37 54889461                           /test/gtest/gtest\n"
"00a88000-0a2db000 rw-p 00000000 00:00 0                                  [heap]\n"
"7ffff09fd000-7ffff0cee000 rw-p 00000000 00:00 0 \n"
"7ffff0cee000-7ffff0cef000 ---p 00000000 00:00 0 \n"
"7ffff0cef000-7ffff14ef000 rw-p 00000000 00:00 0                          [stack:17403]\n"
"7ffff14ef000-7ffff14f0000 ---p 00000000 00:00 0 \n"
"7ffff14f0000-7ffff1cf0000 rw-p 00000000 00:00 0 \n"
"7ffff1cf0000-7ffff1cf1000 ---p 00000000 00:00 0 \n"
"7ffff1cf1000-7ffff24f1000 rw-p 00000000 00:00 0                          [stack:17401]\n"
"7ffff24f1000-7ffff24f2000 ---p 00000000 00:00 0 \n"
"7ffff24f2000-7ffff2cf2000 rw-p 00000000 00:00 0                          [stack:17400]\n"
"7ffff2cf2000-7ffff2cf3000 ---p 00000000 00:00 0 \n"
"7ffff2cf3000-7ffff36e9000 rw-p 00000000 00:00 0                          [stack:17399]\n"
"7ffff36e9000-7ffff36ea000 ---p 00000000 00:00 0 \n"
"7ffff36ea000-7ffff3fe5000 rw-p 00000000 00:00 0 \n"
"7ffff3fe5000-7ffff3fe6000 ---p 00000000 00:00 0 \n"
"7ffff3fe6000-7ffff47e6000 rw-p 00000000 00:00 0                          [stack:17397]\n"
"7ffff47e6000-7ffff47e7000 ---p 00000000 00:00 0 \n"
"7ffff47e7000-7ffff4fe7000 rw-p 00000000 00:00 0 \n"
"7ffff4fe7000-7ffff5029000 r-xp 00000000 08:02 962567                     /usr/lib64/libmlx5-rdmav2.so\n"
"7ffff5029000-7ffff5228000 ---p 00042000 08:02 962567                     /usr/lib64/libmlx5-rdmav2.so\n"
"7ffff5228000-7ffff5229000 r--p 00041000 08:02 962567                     /usr/lib64/libmlx5-rdmav2.so\n"
"7ffff5229000-7ffff522b000 rw-p 00042000 08:02 962567                     /usr/lib64/libmlx5-rdmav2.so\n"
"7ffff522b000-7ffff524e000 r-xp 00000000 08:02 962565                     /usr/lib64/libmlx4-rdmav2.so\n"
"7ffff524e000-7ffff544d000 ---p 00023000 08:02 962565                     /usr/lib64/libmlx4-rdmav2.so\n"
"7ffff544d000-7ffff544e000 r--p 00022000 08:02 962565                     /usr/lib64/libmlx4-rdmav2.so\n"
"7ffff544e000-7ffff5450000 rw-p 00023000 08:02 962565                     /usr/lib64/libmlx4-rdmav2.so\n"
"7ffff5450000-7ffff549d000 r-xp 00000000 08:02 921329                     /usr/lib64/libnl.so.1.1.4\n"
"7ffff549d000-7ffff569d000 ---p 0004d000 08:02 921329                     /usr/lib64/libnl.so.1.1.4\n"
"7ffff569d000-7ffff569e000 r--p 0004d000 08:02 921329                     /usr/lib64/libnl.so.1.1.4\n"
"7ffff569e000-7ffff56a3000 rw-p 0004e000 08:02 921329                     /usr/lib64/libnl.so.1.1.4\n"
"7ffff56a3000-7ffff5859000 r-xp 00000000 08:02 918938                     /usr/lib64/libc-2.17.so\n"
"7ffff5859000-7ffff5a59000 ---p 001b6000 08:02 918938                     /usr/lib64/libc-2.17.so\n"
"7ffff5a59000-7ffff5a5d000 r--p 001b6000 08:02 918938                     /usr/lib64/libc-2.17.so\n"
"7ffff5a5d000-7ffff5a5f000 rw-p 001ba000 08:02 918938                     /usr/lib64/libc-2.17.so\n"
"7ffff5a5f000-7ffff5a64000 rw-p 00000000 00:00 0 \n"
"7ffff5a64000-7ffff5a79000 r-xp 00000000 08:02 913944                     /usr/lib64/libgcc_s-4.8.5-20150702.so.1\n"
"7ffff5a79000-7ffff5c78000 ---p 00015000 08:02 913944                     /usr/lib64/libgcc_s-4.8.5-20150702.so.1\n"
"7ffff5c78000-7ffff5c79000 r--p 00014000 08:02 913944                     /usr/lib64/libgcc_s-4.8.5-20150702.so.1\n"
"7ffff5c79000-7ffff5c7a000 rw-p 00015000 08:02 913944                     /usr/lib64/libgcc_s-4.8.5-20150702.so.1\n"
"7ffff5c7a000-7ffff5c90000 r-xp 00000000 08:02 920234                     /usr/lib64/libgomp.so.1.0.0\n"
"7ffff5c90000-7ffff5e8f000 ---p 00016000 08:02 920234                     /usr/lib64/libgomp.so.1.0.0\n"
"7ffff5e8f000-7ffff5e90000 r--p 00015000 08:02 920234                     /usr/lib64/libgomp.so.1.0.0\n"
"7ffff5e90000-7ffff5e91000 rw-p 00016000 08:02 920234                     /usr/lib64/libgomp.so.1.0.0\n"
"7ffff5e91000-7ffff5f92000 r-xp 00000000 08:02 918946                     /usr/lib64/libm-2.17.so\n"
"7ffff5f92000-7ffff6191000 ---p 00101000 08:02 918946                     /usr/lib64/libm-2.17.so\n"
"7ffff6191000-7ffff6192000 r--p 00100000 08:02 918946                     /usr/lib64/libm-2.17.so\n"
"7ffff6192000-7ffff6193000 rw-p 00101000 08:02 918946                     /usr/lib64/libm-2.17.so\n"
"7ffff6193000-7ffff627c000 r-xp 00000000 08:02 919244                     /usr/lib64/libstdc++.so.6.0.19\n"
"7ffff627c000-7ffff647c000 ---p 000e9000 08:02 919244                     /usr/lib64/libstdc++.so.6.0.19\n"
"7ffff647c000-7ffff6485000 r--p 000e9000 08:02 919244                     /usr/lib64/libstdc++.so.6.0.19\n"
"7ffff6485000-7ffff6487000 rw-p 000f2000 08:02 919244                     /usr/lib64/libstdc++.so.6.0.19\n"
"7ffff6487000-7ffff649c000 rw-p 00000000 00:00 0 \n"
"7ffff649c000-7ffff649f000 r-xp 00000000 08:02 918944                     /usr/lib64/libdl-2.17.so\n"
"7ffff649f000-7ffff669e000 ---p 00003000 08:02 918944                     /usr/lib64/libdl-2.17.so\n"
"7ffff669e000-7ffff669f000 r--p 00002000 08:02 918944                     /usr/lib64/libdl-2.17.so\n"
"7ffff669f000-7ffff66a0000 rw-p 00003000 08:02 918944                     /usr/lib64/libdl-2.17.so\n"
"7ffff66a0000-7ffff66a7000 r-xp 00000000 08:02 918968                     /usr/lib64/librt-2.17.so\n"
"7ffff66a7000-7ffff68a6000 ---p 00007000 08:02 918968                     /usr/lib64/librt-2.17.so\n"
"7ffff68a6000-7ffff68a7000 r--p 00006000 08:02 918968                     /usr/lib64/librt-2.17.so\n"
"7ffff68a7000-7ffff68a8000 rw-p 00007000 08:02 918968                     /usr/lib64/librt-2.17.so\n"
"7ffff68a8000-7ffff68be000 r-xp 00000000 08:02 918964                     /usr/lib64/libpthread-2.17.so\n"
"7ffff68be000-7ffff6abe000 ---p 00016000 08:02 918964                     /usr/lib64/libpthread-2.17.so\n"
"7ffff6abe000-7ffff6abf000 r--p 00016000 08:02 918964                     /usr/lib64/libpthread-2.17.so\n"
"7ffff6abf000-7ffff6ac0000 rw-p 00017000 08:02 918964                     /usr/lib64/libpthread-2.17.so\n"
"7ffff6ac0000-7ffff6ac4000 rw-p 00000000 00:00 0 \n"
"7ffff6ac4000-7ffff6ad9000 r-xp 00000000 08:02 919374                     /usr/lib64/libz.so.1.2.7\n"
"7ffff6ad9000-7ffff6cd8000 ---p 00015000 08:02 919374                     /usr/lib64/libz.so.1.2.7\n"
"7ffff6cd8000-7ffff6cd9000 r--p 00014000 08:02 919374                     /usr/lib64/libz.so.1.2.7\n"
"7ffff6cd9000-7ffff6cda000 rw-p 00015000 08:02 919374                     /usr/lib64/libz.so.1.2.7\n"
"7ffff6cda000-7ffff6cde000 r-xp 00000000 08:02 962570                     /usr/lib64/libibcm.so.1.0.0\n"
"7ffff6cde000-7ffff6edd000 ---p 00004000 08:02 962570                     /usr/lib64/libibcm.so.1.0.0\n"
"7ffff6edd000-7ffff6ede000 r--p 00003000 08:02 962570                     /usr/lib64/libibcm.so.1.0.0\n"
"7ffff6ede000-7ffff6edf000 rw-p 00004000 08:02 962570                     /usr/lib64/libibcm.so.1.0.0\n"
"7ffff6edf000-7ffff6ee9000 r-xp 00000000 08:02 920247                     /usr/lib64/libnuma.so.1\n"
"7ffff6ee9000-7ffff70e9000 ---p 0000a000 08:02 920247                     /usr/lib64/libnuma.so.1\n"
"7ffff70e9000-7ffff70ea000 r--p 0000a000 08:02 920247                     /usr/lib64/libnuma.so.1\n"
"7ffff70ea000-7ffff70eb000 rw-p 0000b000 08:02 920247                     /usr/lib64/libnuma.so.1\n"
"7ffff70eb000-7ffff7102000 r-xp 00000000 08:02 955368                     /usr/lib64/libibverbs.so.1.0.0\n"
"7ffff7102000-7ffff7301000 ---p 00017000 08:02 955368                     /usr/lib64/libibverbs.so.1.0.0\n"
"7ffff7301000-7ffff7302000 r--p 00016000 08:02 955368                     /usr/lib64/libibverbs.so.1.0.0\n"
"7ffff7302000-7ffff7303000 rw-p 00017000 08:02 955368                     /usr/lib64/libibverbs.so.1.0.0\n"
"7ffff7303000-7ffff735e000 r-xp 00000000 00:37 21507389                   /src/ucp/.libs/libucp.so.0.0.0\n"
"7ffff735e000-7ffff755d000 ---p 0005b000 00:37 21507389                   /src/ucp/.libs/libucp.so.0.0.0\n"
"7ffff755d000-7ffff755e000 r--p 0005a000 00:37 21507389                   /src/ucp/.libs/libucp.so.0.0.0\n"
"7ffff755e000-7ffff7560000 rw-p 0005b000 00:37 21507389                   /src/ucp/.libs/libucp.so.0.0.0\n"
"7ffff7560000-7ffff7575000 r-xp 00000000 00:37 51801771                   /src/ucm/.libs/libucm.so.0.0.0\n"
"7ffff7575000-7ffff7774000 ---p 00015000 00:37 51801771                   /src/ucm/.libs/libucm.so.0.0.0\n"
"7ffff7774000-7ffff7775000 r--p 00014000 00:37 51801771                   /src/ucm/.libs/libucm.so.0.0.0\n"
"7ffff7775000-7ffff7776000 rw-p 00015000 00:37 51801771                   /src/ucm/.libs/libucm.so.0.0.0\n"
"7ffff7776000-7ffff788b000 r-xp 00000000 00:37 55431371                   /src/uct/.libs/libuct.so.0.0.0\n"
"7ffff788b000-7ffff7a8a000 ---p 00115000 00:37 55431371                   /src/uct/.libs/libuct.so.0.0.0\n"
"7ffff7a8a000-7ffff7a8b000 r--p 00114000 00:37 55431371                   /src/uct/.libs/libuct.so.0.0.0\n"
"7ffff7a8b000-7ffff7a92000 rw-p 00115000 00:37 55431371                   /src/uct/.libs/libuct.so.0.0.0\n"
"7ffff7a92000-7ffff7bbc000 r-xp 00000000 00:37 51119717                   /src/ucs/.libs/libucs.so.0.0.0\n"
"7ffff7bbc000-7ffff7dbb000 ---p 0012a000 00:37 51119717                   /src/ucs/.libs/libucs.so.0.0.0\n"
"7ffff7dbb000-7ffff7dcd000 r--p 00129000 00:37 51119717                   /src/ucs/.libs/libucs.so.0.0.0\n"
"7ffff7dcd000-7ffff7dd4000 rw-p 0013b000 00:37 51119717                   /src/ucs/.libs/libucs.so.0.0.0\n"
"7ffff7dd4000-7ffff7ddb000 rw-p 00000000 00:00 0 \n"
"7ffff7ddb000-7ffff7dfc000 r-xp 00000000 08:02 918931                     /usr/lib64/ld-2.17.so\n"
"7ffff7e4c000-7ffff7e98000 r--s 00000000 08:02 153888                     /var/db/nscd/passwd\n"
"7ffff7e98000-7ffff7fcf000 rw-p 00000000 00:00 0 \n"
"7ffff7fed000-7ffff7fee000 rw-p 00000000 00:00 0 \n"
"7ffff7ff2000-7ffff7ff4000 rw-s 00000000 00:04 664993814                  /SYSV00000000 (deleted)\n"
"7ffff7ff5000-7ffff7ffa000 rw-p 00000000 00:00 0 \n"
"7ffff7ffa000-7ffff7ffc000 r-xp 00000000 00:00 0                          [vdso]\n"
"7ffff7ffc000-7ffff7ffd000 r--p 00021000 08:02 918931                     /usr/lib64/ld-2.17.so\n"
"7ffff7ffd000-7ffff7ffe000 rw-p 00022000 08:02 918931                     /usr/lib64/ld-2.17.so\n"
"7ffff7ffe000-7ffff7fff000 rw-p 00000000 00:00 0 \n"
"7ffffffdd000-7ffffffff000 rw-p 00000000 00:00 0                          [stack]\n"
"ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0                  [vsyscall]\n";


class shm_seg_test : public ucs::test {
public:
    UCS_TEST_BASE_IMPL;

    typedef std::list<void *>                 val_arr_t;
    typedef val_arr_t::const_iterator         val_iter_t;
    typedef std::pair<std::string, val_arr_t> test_case_t;
    typedef std::list<test_case_t>            test_case_arr_t;
    typedef test_case_arr_t::const_iterator   test_case_iter_t;

    shm_seg_test() {
        test_case_arr.push_back(test_case_t(test_file_01,
                                            build_val_arr(3ul,
                                                          0x00400000,
                                                          0x7ffff7ff2000,
                                                          0xffffffffff600000)));
    };

protected:
    void write2pipe(const std::string& str) {
        ASSERT_EQ(int(0), pipe(fd));
        ASSERT_TRUE(fd[0] && fd[1]);
        EXPECT_EQ(ssize_t(str.length() + 1),
                  write(fd[1], str.c_str(), str.length() + 1));
        close(fd[1]);
    }

    void test_pipe(void *seg_addr) {
        EXPECT_NE(0ul, ucm_get_shm_seg_size_fd(seg_addr, fd[0]));
        close(fd[0]);
    }

    static val_arr_t build_val_arr(size_t size, ...) {
        val_arr_t arr;
        va_list   va;

        va_start(va, size);
        for (size_t i = 0; i < size; ++i) {
            arr.push_back(va_arg(va, void *));
        }
        va_end(va);

        return arr;
    }

    test_case_arr_t test_case_arr;

    int fd[2];
};

UCS_TEST_F(shm_seg_test, parse) {
    size_t test_counter = 0;
    for (test_case_iter_t i = test_case_arr.begin(); i != test_case_arr.end(); ++i) {
        for (val_iter_t j = i->second.begin(); j != i->second.end(); ++j) {
            write2pipe(i->first);
            test_pipe(*j);
            test_counter++;
        }
    }
    EXPECT_EQ(3ul, test_counter);
}
