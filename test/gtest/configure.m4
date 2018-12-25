#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

test_modules=""
m4_include([test/gtest/ucm/test_dlopen/configure.m4])
m4_include([test/gtest/ucs/test_module/configure.m4])
AC_DEFINE_UNQUOTED([test_MODULES], ["${test_modules}"], [Test loadable modules])
AC_CONFIG_FILES([test/gtest/Makefile])
