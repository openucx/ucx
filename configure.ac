#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and The University
#               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
AC_PREREQ([2.63])

define([ucx_ver_major], 1)
define([ucx_ver_minor], 5)
define([ucx_ver_patch], 0)
define([ts], esyscmd([sh -c "date +%Y%m%d%H%M%S"]))

# This is the API version (see libtool library versioning)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
# current:rev:age
define([libucx_so_version], 0:0:0)

AC_INIT([ucx], [ucx_ver_major.ucx_ver_minor])
: ${CFLAGS=""}
: ${CXXFLAGS=""}
config_flags="$*"
valgrind_libpath=""

AC_USE_SYSTEM_EXTENSIONS
AC_GNU_SOURCE
AC_CONFIG_HEADERS([config.h])

AC_CHECK_PROG(GITBIN,git,yes)
AS_IF([test x"${GITBIN}" == x"yes"],
      [AC_SUBST(SCM_VERSION, esyscmd([sh -c 'git rev-parse --short=7 HEAD']))],
      [AC_SUBST(SCM_VERSION, "0000000")])

AH_TOP([
#ifndef UCX_CONFIG_H
#define UCX_CONFIG_H
])

AH_BOTTOM([
#endif /* UCX_CONFIG_H */
])

AM_INIT_AUTOMAKE([1.10 foreign tar-ustar subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([config/m4])

define([ucx_prefix], [/usr])
AC_PREFIX_DEFAULT([ucx_prefix])

top_top_srcdir=$srcdir
AC_SUBST(top_top_srcdir)

MAJOR_VERSION=ucx_ver_major
MINOR_VERSION=ucx_ver_minor
PATCH_VERSION=ucx_ver_patch
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$PATCH_VERSION
SOVERSION=libucx_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(PATCH_VERSION)
AC_SUBST(SCM_VERSION)
AC_SUBST(SOVERSION)

AC_PROG_CC
AC_PROG_CXX
AC_OPENMP
AM_PROG_AS
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
LT_LIB_M
AC_C_RESTRICT
AC_FUNC_STRERROR_R

m4_include([config/m4/pkg.m4])
PKG_PROG_PKG_CONFIG


#
# Save config flags for version dump tool
#
AC_DEFINE_UNQUOTED([UCX_CONFIGURE_FLAGS], ["$config_flags"], [UCX configure flags])


#
# Provide the functionality of AS_VAR_APPEND if Autoconf does not have it.
#
m4_ifdef([AS_VAR_APPEND],
[m4_copy([AS_VAR_APPEND], [ucx_AS_VAR_APPEND])],
[m4_define([ucx_AS_VAR_APPEND],
[AS_VAR_SET([$1], [AS_VAR_GET([$1])$2])])])

#
# Configure modules
#
m4_include([config/m4/ax_prog_doxygen.m4])
m4_include([config/m4/graphviz.m4])
AC_ARG_WITH([docs_only],
        AS_HELP_STRING([--with-docs-only],
                       [Compile only the docs and not the rest of UCX. [default=NO]]),
        ,[:],[with_docs_only=no])

AS_IF([test "x$with_docs_only" == xyes],
    [AS_MESSAGE([dxonly])
     AM_CONDITIONAL([DOCS_ONLY], [true])
     AM_CONDITIONAL([HAVE_GTEST], [false])
     AM_CONDITIONAL([HAVE_STATS], [false])
     AM_CONDITIONAL([HAVE_TUNING], [false])
     AM_CONDITIONAL([HAVE_MEMTRACK], [false])
     AM_CONDITIONAL([HAVE_IB], [false])
     AM_CONDITIONAL([HAVE_MLX5_HW], [false])
     AM_CONDITIONAL([HAVE_MLX5_HW_UD], [false])
     AM_CONDITIONAL([HAVE_MLX5_DV], [false])
     AM_CONDITIONAL([HAVE_TL_RC], [false])
     AM_CONDITIONAL([HAVE_TL_DC], [false])
     AM_CONDITIONAL([HAVE_TL_UD], [false])
     AM_CONDITIONAL([HAVE_TL_CM], [false])
     AM_CONDITIONAL([HAVE_IBV_EX_HW_TM], [false])
     AM_CONDITIONAL([HAVE_CRAY_UGNI], [false])
     AM_CONDITIONAL([HAVE_CUDA], [false])
     AM_CONDITIONAL([HAVE_GDR_COPY], [false])
     AM_CONDITIONAL([HAVE_ROCM], [false])
     AM_CONDITIONAL([HAVE_XPMEM], [false])
     AM_CONDITIONAL([HAVE_CMA], [false])
     AM_CONDITIONAL([HAVE_KNEM], [false])
     AM_CONDITIONAL([HAVE_RDMACM], [false])
     AM_CONDITIONAL([HAVE_MPI], [false])
     AM_CONDITIONAL([HAVE_MPIRUN], [false])
     AM_CONDITIONAL([HAVE_MPICC], [false])
     AM_CONDITIONAL([HAVE_PROFILING], [false])
     AM_CONDITIONAL([HAVE_UCM_PTMALLOC286], [false])
     AM_CONDITIONAL([HAVE_JAVA], [false])
     AM_CONDITIONAL([HAVE_CXX11], [false])
     AM_CONDITIONAL([HAVE_TCMALLOC], [false])
    ],
    [
     AM_CONDITIONAL([DOCS_ONLY], [false])
     m4_include([config/m4/compiler.m4])
     m4_include([config/m4/sysdep.m4])
     m4_include([config/m4/ucs.m4])
     m4_include([config/m4/ucm.m4])
     m4_include([config/m4/ib.m4])
     m4_include([config/m4/rdmacm.m4])
     m4_include([config/m4/cray_ugni.m4])
     m4_include([config/m4/mpi.m4])
     m4_include([config/m4/rte.m4])
     m4_include([config/m4/java.m4])
     m4_include([config/m4/cuda.m4])
     m4_include([config/m4/gdrcopy.m4])
     m4_include([config/m4/rocm.m4])
     m4_include([config/m4/cma.m4])
     m4_include([config/m4/knem.m4])
     m4_include([config/m4/xpmem.m4])

     #
     # Compile code with frame pointer. Optimizations usually omit the frame pointer,
     # but if we are profiling the code with callgraph we need it.
     # This option may affect perofrmance so it is off by default.
     #
     AC_ARG_ENABLE([frame-pointer],
         AS_HELP_STRING([--enable-frame-pointer],
                        [Compile with frame pointer, useful for profiling, default: NO]),
         [],
         [enable_frame_pointer=no])

     AS_IF([test "x$enable_frame_pointer" == xyes],
         [AS_MESSAGE([compiling with frame pointer])
          BASE_CFLAGS="$BASE_CFLAGS -fno-omit-frame-pointer"],
         [:])


     #
     # Enable fault injection code
     #
     AC_ARG_ENABLE([fault-injection],
         AS_HELP_STRING([--enable-fault-injection],
                        [Enable fault injection code, default: NO]),
         [],
         [enable_fault_injection=no])
     AS_IF([test "x$enable_fault_injection" == xyes],
        [AS_MESSAGE([enabling with fault injection code])
         AC_DEFINE([ENABLE_FAULT_INJECTION], [1], [Enable fault injection code])],
        [:]
       )


     #
     # Disable checking user parameters
     #
     AC_ARG_ENABLE([params-check],
         AS_HELP_STRING([--disable-params-check],
                        [Disable checking user parameters passed to API, default: NO]),
         [AC_DEFINE([ENABLE_PARAMS_CHECK], [0])],
         [AC_DEFINE([ENABLE_PARAMS_CHECK], [1], [Enable checking user parameters])])


     #
     # Enable collecting data to ease debugging
     #
     AC_ARG_ENABLE([debug-data],
                   AS_HELP_STRING([--enable-debug-data],
                                  [Enable collecting data to ease debugging, default: NO]),
         [
          AC_DEFINE([ENABLE_DEBUG_DATA], [1], [Enable collecting data])
          AC_DEFINE([UCT_UD_EP_DEBUG_HOOKS], [1],
                    [Enable packet header inspection/rewriting in UCT/UD])
         ],
         [AC_DEFINE([ENABLE_DEBUG_DATA], [0])])


     #
     # Enable multithreading support
     #
     AC_ARG_ENABLE([mt],
                   AS_HELP_STRING([--enable-mt],
                                  [Enable thread support in UCP and UCT, default: NO]),
         [AC_DEFINE([ENABLE_MT], [1], [Enable thread support in UCP and UCT])],
         [AC_DEFINE([ENABLE_MT], [0])])


     #
     # Path for valgrind-enabled libraries
     #
     AC_SUBST([VALGRIND_LIBPATH], [${valgrind_libpath}])

    ]) # Docs only


#
#Doxygen options
#
DX_PS_FEATURE(OFF)
DX_HTML_FEATURE(ON)
DX_MAN_FEATURE(ON)
DX_PDF_FEATURE(ON)
DX_INIT_DOXYGEN([UCX],[doc/doxygen/ucxdox],[doc/doxygen-doc])

#
# Print which transports are built
#
AC_MSG_NOTICE([Supported transports: $transports])

#
# Final output
#
AC_CONFIG_FILES([Makefile
                 doc/doxygen/header.tex
                 src/uct/api/version.h
                 ])
AS_IF([test "x$with_docs_only" == xyes], [], [
AC_CONFIG_LINKS([
                 debian/compat:debian/compat
                 debian/copyright:debian/copyright
                 debian/ucx.prerm:debian/ucx.prerm
                 ])
AC_CONFIG_FILES([
                 ucx.spec
                 ucx.pc
                 debian/rules
                 debian/control
                 debian/changelog
                 src/ucs/Makefile
                 src/uct/Makefile
                 src/ucp/Makefile
                 src/ucp/api/ucp_version.h
                 src/ucp/core/ucp_version.c
                 src/ucm/Makefile
                 src/tools/info/Makefile
                 src/tools/perf/Makefile
                 src/tools/profile/Makefile
                 test/apps/Makefile
                 test/apps/sockaddr/Makefile
                 test/examples/Makefile
                 test/gtest/Makefile
                 test/mpi/Makefile
                 bindings/java/src/main/native/Makefile
                 ])

AC_CONFIG_FILES([test/mpi/run_mpi.sh], [chmod a+x test/mpi/run_mpi.sh])
])

AC_OUTPUT
