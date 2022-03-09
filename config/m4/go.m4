#
# Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# Check for Golang support
#

go_happy="no"
AC_ARG_WITH([go],
            [AC_HELP_STRING([--with-go=(PATH)],
                            [Compile GO UCX bindings (default is guess).])
            ], [], [with_go=guess])

AS_IF([test "x$with_go" != xno],
      [
            AC_CHECK_PROG(GOBIN, go, yes)
            AS_IF([test "x${GOBIN}" = "xyes"],
                  [AS_VERSION_COMPARE([1.16], [`go version | awk '{print substr($3, 3, length($3)-2)}'`],
                                      [go_happy="yes"], [go_happy="yes"], [go_happy=no])],
                  [go_happy=no])
            AS_IF([test "x$go_happy" == xno],
                  [AS_IF([test "x$with_go" = "xguess"],
                         [AC_MSG_WARN([Disabling GO support - GO compiler version 1.16 or newer not found.])],
                         [AC_MSG_ERROR([GO support was explicitly requested, but go compiler not found.])])])
      ],
      [
            AC_MSG_WARN([GO support was explicitly disabled.])
      ])

AM_CONDITIONAL([HAVE_GO], [test "x$go_happy" != "xno"])
AM_COND_IF([HAVE_GO],
           [AC_SUBST([GO], ["go"])
           build_bindings="${build_bindings}:go"])
