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
         # The modules load the library by its SONAME, so check that it loads
         # rather than link with it: the runtime does not need to install the
         # unversioned development link.
         AS_IF([test "x$with_rdu" = "xguess"],
               [AC_MSG_CHECKING([whether the rdu_devmem runtime library loads])
                LIBS="$RDU_LIBS $LIBS"
                # Run the test program with its output discarded, since the
                # runtime library may print when it is loaded
                rdu_happy="no"
                AS_IF([test "x$cross_compiling" != "xyes"],
                      [AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <dlfcn.h>
                                                         #include <stddef.h>]],
                                         [[return (dlopen("librdu_devmem.so.1",
                                                          RTLD_LAZY) == NULL) &&
                                                  (dlopen("librdu_devmem.so",
                                                          RTLD_LAZY) == NULL);]])],
                                      [AS_IF([./conftest$ac_exeext >/dev/null 2>&1],
                                             [rdu_happy="yes"])])])
                AC_MSG_RESULT([$rdu_happy])],
               [rdu_happy="yes"])

         LIBS="$save_LIBS"

         AS_IF([test "x$rdu_happy" = "xyes"],
               [AC_SUBST([RDU_LIBS], ["$RDU_LIBS"])
                AC_DEFINE([HAVE_RDU], 1, [Enable SambaNova RDU support])
                AC_DEFINE([UCX_RDU_DEVMEM_LIBS],
                          [{"librdu_devmem.so.1", "librdu_devmem.so"}],
                          [Names of the SambaNova rdu_devmem runtime library,
                           the versioned SONAME first])],
               [AC_MSG_WARN([RDU runtime not found])])

        ]) # "x$with_rdu" = "xno"

        AM_CONDITIONAL([HAVE_RDU], [test "x$rdu_happy" = "xyes"])
        rdu_checked=yes

   ]) # "x$rdu_checked" != "xyes"

]) # UCX_CHECK_RDU
