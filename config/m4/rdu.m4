#
# Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_RDU],[

AS_IF([test "x$rdu_checked" != "xyes"],
   [
    # RDU device memory is registered by the IB memory domain
    UCX_CHECK_VERBS

    AC_ARG_WITH([rdu],
                [AS_HELP_STRING([--with-rdu], [Enable the use of SambaNova RDU (default is guess).])],
                [], [with_rdu=guess])

    AS_IF([test "x$with_rdu" = "xno"],
        [
         rdu_happy="no"
        ],
        [test "x$with_ib" != "xyes"],
        [
         # The RDU modules are useful only together with IB
         rdu_happy="no"
         AS_IF([test "x$with_rdu" != "xguess"],
               [AC_MSG_ERROR([RDU support was requested but IB support is not available])])
        ],
        [
         save_LIBS="$LIBS"

         RDU_LIBS=""

         # The SambaNova rdu_devmem runtime library is loaded at run time, so no
         # runtime header or library is needed to build the RDU modules.
         # dlopen() needs libdl on older toolchains; newer glibc folds it into
         # libc, so only add the library when the linker actually requires it.
         AC_SEARCH_LIBS([dlopen], [dl],
                        [AS_IF([test "x$ac_cv_search_dlopen" != "xnone required"],
                               [RDU_LIBS="$ac_cv_search_dlopen"])])

         # With 'guess', enable the RDU modules only where the runtime library
         # is installed, so the default build is unchanged on other hosts.  An
         # explicit --with-rdu enables them without the runtime, e.g. for build
         # coverage in CI.
         AS_IF([test "x$with_rdu" = "xguess"],
               [AC_CHECK_LIB([rdu_devmem], [rdu_detect],
                             [rdu_happy="yes"], [rdu_happy="no"])],
               [rdu_happy="yes"])

         LIBS="$save_LIBS"

         AS_IF([test "x$rdu_happy" = "xyes"],
               [AC_SUBST([RDU_LIBS], ["$RDU_LIBS"])
                AC_DEFINE([HAVE_RDU], 1, [Enable SambaNova RDU support])],
               [AC_MSG_WARN([RDU runtime not found])])

        ]) # "x$with_rdu" = "xno"

        AM_CONDITIONAL([HAVE_RDU], [test "x$rdu_happy" = "xyes"])
        rdu_checked=yes

   ]) # "x$rdu_checked" != "xyes"

]) # UCX_CHECK_RDU
