#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and The University
#               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016-2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
AC_PREREQ([2.63])

define([ucx_ver_major], 1)
define([ucx_ver_minor], 11)
define([ucx_ver_patch], 0)
define([ts], esyscmd([sh -c "date +%Y%m%d%H%M%S"]))

# This is the API version (see libtool library versioning)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
# current:rev:age
define([libucx_so_version], 0:0:0)

AC_INIT([ucx], [ucx_ver_major.ucx_ver_minor])
: ${CPPFLAGS=""}
: ${CFLAGS=""}
: ${CXXFLAGS=""}
config_flags="$*"
valgrind_libpath=""

AC_USE_SYSTEM_EXTENSIONS
AC_GNU_SOURCE
AC_CONFIG_HEADERS([config.h])

AC_CHECK_PROG(GITBIN, git, yes)
AS_IF([test x"${GITBIN}" = x"yes"],
      [# remove preceding "refs/heads/" (11 characters) for symbolic ref
       AC_SUBST(SCM_BRANCH,  esyscmd([sh -c 'git symbolic-ref --quiet   HEAD | sed "s/^.\{11\}//"']))
       AC_SUBST(SCM_VERSION, esyscmd([sh -c 'git rev-parse    --short=7 HEAD']))],
      [AC_SUBST(SCM_BRANCH,  "<unknown>")
       AC_SUBST(SCM_VERSION, "0000000")])

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
AC_PROG_LN_S
AC_PROG_MKDIR_P
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
LT_LIB_M
AC_C_RESTRICT
AC_FUNC_STRERROR_R

AC_PATH_TOOL([PKG_CONFIG], [pkg-config], [pkg-config])


#
# Force link_all_deplibs=yes for libtool, otherwise it will not
# link against dependency libs
#
link_all_deplibs=yes


#
# Check if 'ln' supports creating relative links
#
AC_MSG_CHECKING([if ${LN_S} supports --relative])
AS_IF([${LN_S} --relative symlinktest 2>/dev/null],
      [AC_MSG_RESULT([yes])
       AC_SUBST([LN_RS], ["${LN_S} --relative"])
       rm symlinktest],
      [AC_MSG_RESULT([no])
       AC_SUBST([LN_RS], [${LN_S}])])


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
# Paths for loadable modules
#
AC_SUBST([modulesubdir],   [${PACKAGE_NAME}])               # module directory names
AC_SUBST([moduledir],      [${libdir}/${modulesubdir}])     # module installation directory
AC_SUBST([localmoduledir], ['$(abs_top_builddir)/modules']) # local directory for module symlinks
AC_SUBST([objdir],         [${objdir}])                     # libtool objects dir, usually .libs
AC_SUBST([shrext],         [${shrext_cmds}])                # libtool shared library extension
AC_DEFINE_UNQUOTED([UCX_MODULE_SUBDIR], ["${modulesubdir}"], [UCX module sub-directory])


#
# Additional m4 files
#
m4_include([config/m4/ax_prog_doxygen.m4])
m4_include([config/m4/graphviz.m4])
m4_include([config/m4/ucg.m4])

AC_ARG_WITH([docs_only],
        AS_HELP_STRING([--with-docs-only],
                       [Compile only the docs and not the rest of UCX. [default=NO]]),
        ,[:],[with_docs_only=no])

AC_DEFUN([UCX_DX_ENABLE_CHECK],
         [AS_IF([DX_TEST_FEATURE($1)],
                [],
                [AS_IF([test "x$enable_doxygen_$1" == xyes],
                       [AC_MSG_ERROR([--enable-doxygen-$1 was specified, but $1 tools were not found])],
                       [])])])

#
# Doxygen options
#
DX_PS_FEATURE(OFF)
DX_HTML_FEATURE(ON)
DX_MAN_FEATURE(ON)
DX_PDF_FEATURE(ON)
DX_INIT_DOXYGEN([UCX],[docs/doxygen/ucxdox],[docs/doxygen-doc])

AS_IF([test "x$with_docs_only" = xyes],
    [AS_MESSAGE([Documents only requested])
     AS_IF([DX_TEST_FEATURE(doc)],
           [],
           [AC_MSG_ERROR([--with-only-docs was specified, but doxygen was not found])])
     UCX_DX_ENABLE_CHECK([html])
     UCX_DX_ENABLE_CHECK([man])
     UCX_DX_ENABLE_CHECK([pdf])
     AM_CONDITIONAL([DOCS_ONLY], [true])
     AM_CONDITIONAL([HAVE_GTEST], [false])
     AM_CONDITIONAL([HAVE_STATS], [false])
     AM_CONDITIONAL([HAVE_TUNING], [false])
     AM_CONDITIONAL([HAVE_MEMTRACK], [false])
     AM_CONDITIONAL([HAVE_IB], [false])
     AM_CONDITIONAL([HAVE_MLX5_HW], [false])
     AM_CONDITIONAL([HAVE_MLX5_HW_UD], [false])
     AM_CONDITIONAL([HAVE_MLX5_DV], [false])
     AM_CONDITIONAL([HAVE_DEVX], [false])
     AM_CONDITIONAL([HAVE_EXP], [false])
     AM_CONDITIONAL([HAVE_TL_RC], [false])
     AM_CONDITIONAL([HAVE_TL_DC], [false])
     AM_CONDITIONAL([HAVE_DC_DV], [false])
     AM_CONDITIONAL([HAVE_DC_EXP], [false])
     AM_CONDITIONAL([HAVE_TL_UD], [false])
     AM_CONDITIONAL([HAVE_TL_CM], [false])
     AM_CONDITIONAL([HAVE_CRAY_UGNI], [false])
     AM_CONDITIONAL([HAVE_CUDA], [false])
     AM_CONDITIONAL([HAVE_CUDA_STATIC], [false])
     AM_CONDITIONAL([HAVE_GDR_COPY], [false])
     AM_CONDITIONAL([HAVE_ROCM], [false])
     AM_CONDITIONAL([HAVE_HIP], [false])
     AM_CONDITIONAL([HAVE_XPMEM], [false])
     AM_CONDITIONAL([HAVE_CMA], [false])
     AM_CONDITIONAL([HAVE_KNEM], [false])
     AM_CONDITIONAL([HAVE_RDMACM], [false])
     AM_CONDITIONAL([HAVE_RDMACM_QP_LESS], [false])
     AM_CONDITIONAL([HAVE_MPI], [false])
     AM_CONDITIONAL([HAVE_MPIRUN], [false])
     AM_CONDITIONAL([HAVE_MPICC], [false])
     AM_CONDITIONAL([HAVE_PROFILING], [false])
     AM_CONDITIONAL([HAVE_UCM_PTMALLOC286], [false])
     AM_CONDITIONAL([HAVE_JAVA], [false])
     AM_CONDITIONAL([HAVE_CXX11], [false])
     AM_CONDITIONAL([HAVE_GNUXX11], [false])
     AM_CONDITIONAL([HAVE_GLIBCXX_NOTHROW], [false])
     AM_CONDITIONAL([HAVE_TCMALLOC], [false])
     AM_CONDITIONAL([ENABLE_EXPERIMENTAL_API], [false])
     AM_CONDITIONAL([INSTALL_DEVEL_HEADERS], [false])
     AM_CONDITIONAL([HAVE_EXAMPLES], [false])
     AM_CONDITIONAL([HAVE_AARCH64_THUNDERX2], [false])
     AM_CONDITIONAL([HAVE_AARCH64_THUNDERX1], [false])
     AM_CONDITIONAL([HAVE_AARCH64_HI1620], [false])
     AM_CONDITIONAL([HAVE_FUSE3], [false])
    ],
    [
     AM_CONDITIONAL([DOCS_ONLY], [false])
     m4_include([config/m4/compiler.m4])
     m4_include([config/m4/sysdep.m4])
     m4_include([config/m4/ucm.m4])
     m4_include([config/m4/mpi.m4])
     m4_include([config/m4/rte.m4])
     m4_include([config/m4/fuse3.m4])
     m4_include([config/m4/java.m4])
     m4_include([config/m4/cuda.m4])
     m4_include([config/m4/rocm.m4])
     m4_include([config/m4/gdrcopy.m4])
     m4_include([src/ucm/configure.m4])
     m4_include([src/ucs/configure.m4])
     m4_include([src/uct/configure.m4])
     m4_include([src/tools/perf/configure.m4])
     m4_include([test/gtest/configure.m4])


     #
     # Enable fault injection code
     #
     AC_ARG_ENABLE([fault-injection],
         AS_HELP_STRING([--enable-fault-injection],
                        [Enable fault injection code, default: NO]),
         [],
         [enable_fault_injection=no])
     AS_IF([test "x$enable_fault_injection" = xyes],
        [AS_MESSAGE([enabling with fault injection code])
         AC_DEFINE([ENABLE_FAULT_INJECTION], [1], [Enable fault injection code])],
        [:])


     #
     # Disable checking user parameters
     #
     AC_ARG_ENABLE([params-check],
         AS_HELP_STRING([--disable-params-check],
                        [Disable checking user parameters passed to API, default: NO]),
         [],
         [enable_params_check=yes])
     AS_IF([test "x$enable_params_check" = xyes],
           [AC_DEFINE([ENABLE_PARAMS_CHECK], [1], [Enable checking user parameters])],
           [AC_DEFINE([ENABLE_PARAMS_CHECK], [0])])

     #
     # Enable collecting data to ease debugging
     #
     AC_ARG_ENABLE([debug-data],
                   AS_HELP_STRING([--enable-debug-data],
                                  [Enable collecting data to ease debugging, default: NO]),
		   [],
		   [enable_debug_data=no])
     AS_IF([test "x$enable_debug_data" = xyes],
           [AC_DEFINE([ENABLE_DEBUG_DATA], [1], [Enable collecting data])
            AC_DEFINE([UCT_UD_EP_DEBUG_HOOKS], [1],
                      [Enable packet header inspection/rewriting in UCT/UD])],
           [AC_DEFINE([ENABLE_DEBUG_DATA], [0])
            AC_DEFINE([UCT_UD_EP_DEBUG_HOOKS], [0])])


     #
     # Enable multithreading support
     #
     AC_ARG_ENABLE([mt],
                   AS_HELP_STRING([--enable-mt],
                                  [Enable thread support in UCP and UCT, default: NO]),
                   [],
                   [enable_mt=no])
     AS_IF([test "x$enable_mt" = xyes],
           [AC_DEFINE([ENABLE_MT], [1], [Enable thread support in UCP and UCT])
            mt_enable=enabled],
           [AC_DEFINE([ENABLE_MT], [0])
            mt_enable=disabled])


     #
     # Enable experimental header
     #
     AC_ARG_ENABLE([experimental-api],
                   AS_HELP_STRING([--enable-experimental-api],
                                  [Enable installing experimental APIs, default: NO]),
                   [],
                   [enable_experimental_api=no])
     AM_CONDITIONAL([ENABLE_EXPERIMENTAL_API], [test "x$enable_experimental_api" = "xyes"])


     #
     # Install development headers
     #
     AC_ARG_ENABLE([devel-headers],
                   AS_HELP_STRING([--enable-devel-headers],
                                  [Enable installing development headers, default: NO]),
                    [],
                    [enable_devel_headers=no])
     AM_CONDITIONAL([INSTALL_DEVEL_HEADERS], [test "x$enable_devel_headers" = "xyes"])


     #
     # Path for valgrind-enabled libraries
     #
     AC_SUBST([VALGRIND_LIBPATH], [${valgrind_libpath}])


     #
     # Enable examples build
     #
     AC_ARG_ENABLE([examples],
                   AS_HELP_STRING([--enable-examples],
                                  [Enable examples build, default: NO]),
                   [],
                   [enable_examples=no])
    AM_CONDITIONAL([HAVE_EXAMPLES], [test "x$enable_examples" = "xyes"])
    ]) # Docs only

#
# Print which transports are built
#
build_modules="${uct_modules}"
build_modules="${build_modules}${ucs_modules}"
build_modules="${build_modules}${uct_ib_modules}"
build_modules="${build_modules}${uct_cuda_modules}"
build_modules="${build_modules}${ucm_modules}"
build_modules="${build_modules}${ucx_perftest_modules}"
build_modules="${build_modules}${uct_rocm_modules}"
build_modules="${build_modules}${ucg_modules}"
AC_SUBST([build_modules], [${build_modules}])
AC_SUBST([build_bindings], [${build_bindings}])

#
# Final output
#
AC_CONFIG_FILES([Makefile
                 docs/doxygen/header.tex
                 src/uct/api/version.h
                 ])
AS_IF([test "x$with_docs_only" = xyes], [], [
AC_CONFIG_LINKS([
                 debian/compat:debian/compat
                 debian/copyright:debian/copyright
                 debian/ucx.prerm:debian/ucx.prerm
                 ])
AC_CONFIG_FILES([
                 ucx.spec
                 ucx.pc
                 contrib/rpmdef.sh
                 debian/rules
                 debian/control
                 debian/changelog
                 src/ucp/Makefile
                 src/ucp/api/ucp_version.h
                 src/ucp/core/ucp_version.c
                 src/tools/vfs/Makefile
                 src/tools/info/Makefile
                 src/tools/profile/Makefile
                 test/apps/Makefile
                 test/apps/iodemo/Makefile
                 test/apps/sockaddr/Makefile
                 test/apps/profiling/Makefile
                 test/mpi/Makefile
                 bindings/java/Makefile
                 bindings/java/pom.xml
                 bindings/java/src/main/native/Makefile
                 examples/Makefile
                 ])

AC_CONFIG_FILES([test/mpi/run_mpi.sh], [chmod a+x test/mpi/run_mpi.sh])
])


AC_OUTPUT


#
# Print build condiguration
#
AC_MSG_NOTICE([=========================================================])
AS_IF([test "x$with_docs_only" = xyes],
[
AC_MSG_NOTICE([Building documents only])
],
[
AC_MSG_NOTICE([UCX build configuration:])
AC_MSG_NOTICE([      Build prefix:   ${prefix}])
AC_MSG_NOTICE([Preprocessor flags:   ${BASE_CPPFLAGS}])
AC_MSG_NOTICE([        C compiler:   ${CC} ${BASE_CFLAGS}])
AC_MSG_NOTICE([      C++ compiler:   ${CXX} ${BASE_CXXFLAGS}])
AC_MSG_NOTICE([      Multi-thread:   ${mt_enable}])
AC_MSG_NOTICE([         MPI tests:   ${mpi_enable}])
AC_MSG_NOTICE([       VFS support:   ${vfs_enable}])
AC_MSG_NOTICE([     Devel headers:   ${enable_devel_headers}])
AC_MSG_NOTICE([          Bindings:   <$(echo ${build_bindings}|tr ':' ' ') >])
AC_MSG_NOTICE([       UCS modules:   <$(echo ${ucs_modules}|tr ':' ' ') >])
AC_MSG_NOTICE([       UCT modules:   <$(echo ${uct_modules}|tr ':' ' ') >])
AC_MSG_NOTICE([      CUDA modules:   <$(echo ${uct_cuda_modules}|tr ':' ' ') >])
AC_MSG_NOTICE([      ROCM modules:   <$(echo ${uct_rocm_modules}|tr ':' ' ') >])
AC_MSG_NOTICE([        IB modules:   <$(echo ${uct_ib_modules}|tr ':' ' ') >])
AC_MSG_NOTICE([       UCM modules:   <$(echo ${ucm_modules}|tr ':' ' ') >])
AC_MSG_NOTICE([      Perf modules:   <$(echo ${ucx_perftest_modules}|tr ':' ' ') >])
AS_IF([test "x$enable_ucg" != "xno"], [
    AC_MSG_NOTICE([       UCG modules:   <$(echo ${ucg_modules}|tr ':' ' ') >])])
])
AC_MSG_NOTICE([=========================================================])
