/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/module.h>
#include <ucs/sys/module_int.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/sock.h>
#include <ucs/type/spinlock.h>
#include <ucs/time/time.h>
#include <ucs/arch/cpu.h>
}

#include <sys/mman.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <fstream>
#include <link.h>
#include <set>

class test_sys : public ucs::test {
protected:
    static int get_mem_prot(void *address, size_t size) {
        return ucs_get_mem_prot((uintptr_t)address, (uintptr_t)address + size);
    }

    void test_dirname(char *path, int num_layers, const char *expected) {
        path = ucs_dirname(path, num_layers);
        EXPECT_EQ(std::string(expected), path);
    }

    void test_memunits(size_t size, const char *expected) {
        char buf[256];

        ucs_memunits_to_str(size, buf, sizeof(buf));
        EXPECT_EQ(std::string(expected), buf);
    }

    void test_read_file_number(const char *contents,
                               ucs_status_t expected_status,
                               long expected_value = 0)
    {
        char filename[PATH_MAX];
        long value = LONG_MIN;
        mode_t orig_umask;
        size_t length;
        ssize_t nwritten;
        ucs_status_t status;
        int fd;

        /* Create a temporary file to write the contents to */
        snprintf(filename, sizeof(filename), "%s/ucx_read_number_XXXXXX",
                 ucs_get_tmpdir());
        orig_umask = umask(S_IRWXG | S_IRWXO);
        fd         = mkstemp(filename);
        umask(orig_umask);
        if (fd < 0) {
            FAIL() << "failed to create temporary file";
            return;
        }

        length   = strlen(contents);
        nwritten = write(fd, contents, length);
        close(fd);
        if ((nwritten < 0) || ((size_t)nwritten != length)) {
            unlink(filename);
            FAIL() << "failed to write test file";
        }

        /* Read the contents of the file */
        status = ucs_read_file_number(&value, 1, "%s", filename);
        unlink(filename);

        EXPECT_EQ(expected_status, status);
        if (expected_status == UCS_OK) {
            EXPECT_EQ(expected_value, value);
        } else {
            EXPECT_EQ(LONG_MIN, value);
        }
    }

    static void check_cache_type(ucs_cpu_cache_type_t type, const char *name)
    {
        size_t cache;
        char memunits[32];

        cache = ucs_cpu_get_cache_size(type);

        ucs_memunits_to_str(cache, memunits, sizeof(memunits));
        UCS_TEST_MESSAGE << name << " cache: " << memunits;
    }
};

UCS_TEST_F(test_sys, read_file_number) {
    test_read_file_number("-1", UCS_OK, -1);
    test_read_file_number("0", UCS_OK, 0);
    test_read_file_number("-1\n", UCS_OK, -1);
    test_read_file_number("0\n", UCS_OK, 0);
    test_read_file_number("42", UCS_OK, 42);
    test_read_file_number("42\n", UCS_OK, 42);
    test_read_file_number("  0x2a \t\n", UCS_OK, 42);
    test_read_file_number("", UCS_ERR_INVALID_PARAM);
    test_read_file_number(" \t\n", UCS_ERR_INVALID_PARAM);
    test_read_file_number("not-a-number\n", UCS_ERR_INVALID_PARAM);
    test_read_file_number("42 trailing\n", UCS_ERR_INVALID_PARAM);
    test_read_file_number("42trailing\n", UCS_ERR_INVALID_PARAM);
    test_read_file_number("leading42\n", UCS_ERR_INVALID_PARAM);
    test_read_file_number("leading 42\n", UCS_ERR_INVALID_PARAM);
    test_read_file_number("42\n43", UCS_ERR_INVALID_PARAM);
    test_read_file_number("42\n43\n", UCS_ERR_INVALID_PARAM);
}

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
    ucs_recursive_spinlock_t lock;
    pthread_t self;

    self = pthread_self();

    ucs_recursive_spinlock_init(&lock, 0);

    ucs_recursive_spin_lock(&lock);
    EXPECT_TRUE(ucs_recursive_spin_is_owner(&lock, self));

    /* coverity[double_lock] */
    ucs_recursive_spin_lock(&lock);
    EXPECT_TRUE(ucs_recursive_spin_is_owner(&lock, self));

    ucs_recursive_spin_unlock(&lock);
    EXPECT_TRUE(ucs_recursive_spin_is_owner(&lock, self));

    /* coverity[double_unlock] */
    ucs_recursive_spin_unlock(&lock);
    EXPECT_FALSE(ucs_recursive_spin_is_owner(&lock, self));
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

extern "C" {
int test_module_loaded = 0;
}

UCS_TEST_F(test_sys, module) {
    UCS_MODULE_FRAMEWORK_DECLARE(test);

    EXPECT_EQ(0, test_module_loaded);
    UCS_MODULE_FRAMEWORK_LOAD(test, 0);
    EXPECT_EQ(1, test_module_loaded);
}

extern "C" {
int ucs_test_module_base_loaded      = 0;
int ucs_test_module_dependent_loaded = 0;
int ucs_test_module_dependency_value = 0;
int ucs_test_module_extra_loaded     = 0;
}

class test_module_load_order : public ucs::test {
protected:
    void init() override
    {
        char path[PATH_MAX];
        Dl_info library_info;
        const char *extension;
        void *load_modules;

        ucs::test::init();
        if (!ucs_sys_is_dynamic_lib()) {
            UCS_TEST_SKIP_R("module loading requires shared libraries");
        }

        load_modules = dlsym(RTLD_DEFAULT, "ucs_load_modules");
        ASSERT_NE(nullptr, load_modules);
        ASSERT_NE(0, dladdr(load_modules, &library_info));
        extension = strchr(ucs_basename(library_info.dli_fname), '.');
        ASSERT_NE(nullptr, extension);
        m_extension = extension;

        ucs_snprintf_safe(path, sizeof(path), "%s/ucx_module_order_XXXXXX",
                          ucs_get_tmpdir());
        ASSERT_NE(nullptr, mkdtemp(path));
        m_path = path;
        ASSERT_EQ(0, mkdir((m_path + "/first").c_str(), S_IRWXU));
        ASSERT_EQ(0, mkdir((m_path + "/second").c_str(), S_IRWXU));

        ASSERT_NO_FATAL_FAILURE(copy_module("dependent", "first", "dependent"));
        ASSERT_NO_FATAL_FAILURE(copy_module("base", "second", "base"));
        ASSERT_NO_FATAL_FAILURE(copy_module("extra", "first", "extra"));
        ASSERT_NO_FATAL_FAILURE(copy_module("extra", "first", "blocked"));

        modify_config("PLUGIN_PATH", m_path + "/first:" + m_path + "/second");
        modify_config("MODULES", "^blocked");
    }

    void cleanup() override
    {
        for (const auto &path : m_files) {
            unlink(path.c_str());
        }

        if (!m_path.empty()) {
            rmdir((m_path + "/second").c_str());
            rmdir((m_path + "/first").c_str());
            rmdir(m_path.c_str());
        }

        ucs::test::cleanup();
    }

    std::string module_filename(const char *name) const
    {
#ifdef UCX_MODULE_FILE_SUFFIX
        const char *file_suffix = UCX_MODULE_FILE_SUFFIX;
#else
        const char *file_suffix = "";
#endif

        return std::string("libtestorder_") + name + file_suffix + m_extension;
    }

    std::string module_path(const char *directory, const char *name) const
    {
        return m_path + "/" + directory + "/" + module_filename(name);
    }

    void copy_module(const char *source, const char *directory,
                      const char *name)
    {
        const std::string source_path =
                GTEST_MODULE_ORDER_LIB_DIR "/" + module_filename(source);
        const std::string target_path = module_path(directory, name);
        std::ifstream input(source_path, std::ios::binary);
        std::ofstream output(target_path, std::ios::binary);
        struct stat source_stat, target_stat;

        m_files.push_back(target_path);
        ASSERT_TRUE(input.is_open()) << source_path;
        ASSERT_TRUE(output.is_open()) << target_path;
        output << input.rdbuf();
        output.close();
        ASSERT_TRUE(output.good());

        ASSERT_EQ(0, stat(source_path.c_str(), &source_stat));
        ASSERT_EQ(0, stat(target_path.c_str(), &target_stat));
        ASSERT_EQ(source_stat.st_size, target_stat.st_size);
        ASSERT_FALSE((source_stat.st_dev == target_stat.st_dev) &&
                     (source_stat.st_ino == target_stat.st_ino));
    }

    static int count_modules(struct dl_phdr_info *info, size_t, void *arg)
    {
        auto *count = static_cast<std::pair<std::string, unsigned>*>(arg);

        if ((info->dlpi_name != nullptr) &&
            (strncmp(info->dlpi_name, count->first.c_str(),
                     count->first.length()) == 0)) {
            ++count->second;
        }

        return 0;
    }

    static unsigned module_count(const std::string &prefix)
    {
        std::pair<std::string, unsigned> count(prefix, 0);

        dl_iterate_phdr(count_modules, &count);
        return count.second;
    }

    std::string m_path, m_extension;
    std::vector<std::string> m_files;
};

UCS_TEST_F(test_module_load_order, dependency_copy)
{
    const std::string empty_path = m_path + "/first/libtestorder_" +
                                   m_extension;
    const int base_before      = ucs_test_module_base_loaded;
    const int dependent_before = ucs_test_module_dependent_loaded;
    const int extra_before     = ucs_test_module_extra_loaded;
    ucs_init_once_t init_once  = UCS_INIT_ONCE_INITIALIZER;

    ASSERT_NO_FATAL_FAILURE(copy_module("dependent", "first", ""));
    m_files.push_back(empty_path);
    ASSERT_EQ(0, rename(module_path("first", "").c_str(), empty_path.c_str()));

    /* The dependent is in an earlier directory, while its RUNPATH points to
     * the original build-tree base. Load the copied base first, so DT_NEEDED
     * reuses it rather than introducing a second physical base library. */
    ucs_load_modules("testorder", ":missing::base:dependent:base:blocked:",
                      &init_once, 0);
    EXPECT_EQ(0, pthread_mutex_destroy(&init_once.lock));

    EXPECT_EQ(base_before + 1, ucs_test_module_base_loaded);
    EXPECT_EQ(dependent_before + 1, ucs_test_module_dependent_loaded);
    EXPECT_EQ(extra_before + 1, ucs_test_module_extra_loaded);
    EXPECT_EQ(42, ucs_test_module_dependency_value);
    EXPECT_EQ(1u, module_count(module_path("second", "base")));
    EXPECT_EQ(1u, module_count(module_path("first", "dependent")));
    EXPECT_EQ(1u, module_count(module_path("first", "extra")));
    EXPECT_EQ(0u, module_count(empty_path));
    EXPECT_EQ(0u, module_count(module_path("first", "blocked")));
    EXPECT_EQ(0u, module_count(GTEST_MODULE_ORDER_LIB_DIR
                               "/libtestorder_base"));
}

UCS_TEST_F(test_module_load_order, plain_override)
{
#ifndef UCX_MODULE_FILE_SUFFIX
    UCS_TEST_SKIP_R("module file suffix is not configured");
#else
    const std::string plain_path = m_path + "/first/libtestorder_base" +
                                   m_extension;
    const int base_before        = ucs_test_module_base_loaded;
    ucs_init_once_t init_once    = UCS_INIT_ONCE_INITIALIZER;

    ASSERT_NO_FATAL_FAILURE(copy_module("base", "first", "base"));
    m_files.push_back(plain_path);
    ASSERT_EQ(0, rename(module_path("first", "base").c_str(),
                        plain_path.c_str()));

    /* A plain override in an earlier directory takes priority over a private
     * module filename in a later directory. */
    ucs_load_modules("testorder", ":base:dependent", &init_once, 0);
    EXPECT_EQ(0, pthread_mutex_destroy(&init_once.lock));

    EXPECT_EQ(base_before + 1, ucs_test_module_base_loaded);
    EXPECT_EQ(1u, module_count(plain_path));
    EXPECT_EQ(0u, module_count(module_path("second", "base")));
#endif
}

UCS_TEST_F(test_sys, module_file_suffix) {
#ifndef UCX_MODULE_FILE_SUFFIX
    UCS_TEST_SKIP_R("module file suffix is not configured");
#else
    char suffixed_base[] = "libtest_module" UCX_MODULE_FILE_SUFFIX;
    char plain_base[]    = "libtest_module";
    char other_base[]    = "libother_module" UCX_MODULE_FILE_SUFFIX;

    ucs_module_normalize_base(suffixed_base, "libtest_");
    ucs_module_normalize_base(plain_base, "libtest_");
    ucs_module_normalize_base(other_base, "libtest_");

    EXPECT_STREQ("libtest_module", suffixed_base);
    EXPECT_STREQ("libtest_module", plain_base);
    EXPECT_STREQ("libother_module" UCX_MODULE_FILE_SUFFIX, other_base);
#endif
}

UCS_TEST_F(test_sys, dirname) {
    char path[] = "/sys/devices/pci0000:00/0000:00:00.0";
    test_dirname(path, 3, "/sys");
}

UCS_TEST_F(test_sys, memunits_to_str) {
    test_memunits(256, "256");
    test_memunits(1256, "1256");
    test_memunits(UCS_KBYTE, "1K");
    test_memunits(UCS_MBYTE + UCS_KBYTE, "1025K");
    test_memunits(UCS_GBYTE, "1G");
    test_memunits(2 * UCS_GBYTE, "2G");
    test_memunits(UCS_TBYTE, "1T");
    test_memunits(UCS_TBYTE * 1024, "1P");
}

UCS_TEST_F(test_sys, cpu_cache) {
    check_cache_type(UCS_CPU_CACHE_L1d, "L1d");
    check_cache_type(UCS_CPU_CACHE_L1i, "L1i");
    check_cache_type(UCS_CPU_CACHE_L2, "L2");
    check_cache_type(UCS_CPU_CACHE_L3, "L3");
}
