#
# Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#


ucg_modules=":builtin"
m4_include([src/ucg/base/configure.m4])
m4_include([src/ucg/builtin/configure.m4])
m4_include([src/ucg/hicoll/configure.m4])
AC_DEFINE_UNQUOTED([ucg_MODULES], ["${ucg_modules}"], [UCG loadable modules])

AC_CONFIG_FILES([src/ucg/Makefile])
