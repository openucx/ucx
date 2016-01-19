# Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

dnl GTEST_LIB_CHECK([minimum version [,
dnl                  action if found [,action if not found]]])
dnl
dnl Google C++ Testing Framework is a part of project source code.
dnl So ignore version check and just process if gtest is enable.
AC_DEFUN([GTEST_LIB_CHECK],
[
dnl Provide a flag to enable or disable Google Test usage.
AC_ARG_ENABLE([gtest],
  [AS_HELP_STRING([--enable-gtest],
                  [Enable tests using the Google C++ Testing Framework.
                  (Default is disabled.)])],
  [enable_gtest=$enableval],
  [enable_gtest=no])
AC_MSG_CHECKING([for using Google C++ Testing Framework])
AC_MSG_RESULT([$enable_gtest])
AM_CONDITIONAL([HAVE_GTEST],[test "x$enable_gtest" = "xyes"])
])
