#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
AC_PREREQ([2.63])

define([scm_r], esyscmd([sh -c "git rev-parse --short=7 HEAD"]))
define([ucx_ver_major], 1)
define([ucx_ver_minor], 0)
define([ts], esyscmd([sh -c "date +%Y%m%d%H%M%S"]))
define([revcount], esyscmd([git rev-list HEAD | wc -l | sed -e 's/ *//g' | xargs -n1 printf]))

# This is the API version (see libtool library versioning)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
# current:rev:age
define([libucx_so_version], 2:1:0)

AC_INIT([ucx], [ucx_ver_major.ucx_ver_minor])

config_flags="$*"
valgrind_libpath=""

AC_USE_SYSTEM_EXTENSIONS
AC_GNU_SOURCE
AC_CONFIG_HEADERS([config.h])

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

define([ucx_prefix], [/opt/ucx])
AC_PREFIX_DEFAULT([ucx_prefix])

RPM_RELEASE=1
BUILD_TS=ts
MAJOR_VERSION=ucx_ver_major
MINOR_VERSION=ucx_ver_minor
MICRO_VERSION=scm_r
REV_COUNT=revcount
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$REV_COUNT
SOVERSION=libucx_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(MICRO_VERSION)
AC_SUBST(REV_COUNT)
AC_SUBST(RPM_RELEASE)
AC_SUBST(SOVERSION)
AC_SUBST(BUILD_TS)

AC_PROG_CC
AC_PROG_CXX
AM_PROG_AS
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
LT_LIB_M

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
m4_include([config/m4/compiler.m4])
m4_include([config/m4/sysdep.m4])
m4_include([config/m4/ucs.m4])
m4_include([config/m4/ib.m4])
m4_include([config/m4/cray_ugni.m4])
m4_include([config/m4/mpi.m4])
m4_include([config/m4/rte.m4])
m4_include([config/m4/ax_prog_doxygen.m4])
m4_include([config/m4/cuda.m4])
m4_include([config/m4/graphviz.m4])
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
	   CFLAGS="$CFLAGS -fno-omit-frame-pointer"],
	  [:]
  )


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
     AC_DEFINE([UCT_UD_EP_DEBUG_HOOKS], [1], [Enable packet header inspection/rewriting in UCT/UD])
    ],
	[AC_DEFINE([ENABLE_DEBUG_DATA], [0])])


#
# Configure tcmalloc
#

AC_ARG_ENABLE([tcmalloc],
        AS_HELP_STRING([--enable-tcmalloc],
                       [Compile with tcmalloc, libtcmalloc will be installed in the same place as UCX. default: YES]),
        [if test "x$enableval" = xyes; then
	   tcmalloc_enabled=yes
	 else
	   tcmalloc_enabled=no
	 fi
	 ],
        [tcmalloc_enabled=yes])

AS_IF([test "x$tcmalloc_enabled" == xyes],
          [AS_MESSAGE([compiling tcmalloc])
	   AC_SUBST([BUILD_TCMALLOC], [yes])],
          [AC_SUBST([BUILD_TCMALLOC], [no])]
  )

#
# subdir must always be configured to make sure that dist checks work
#

AC_CONFIG_SUBDIRS([external/gperftools])
AM_CONDITIONAL([BUILD_TCMALLOC],[test "x$BUILD_TCMALLOC" = "xyes"])

#
# Path for valgrind-enabled libraries
#
AC_SUBST([VALGRIND_LIBPATH], [${valgrind_libpath}])

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
AC_CONFIG_FILES([
				 Makefile
                 ucx.spec
                 ucx.pc
                 debian/rules
                 debian/control
                 debian/changelog
                 debian/ucx.postinst
                 src/ucs/Makefile
                 src/uct/Makefile
                 src/uct/api/version.h
                 src/ucp/Makefile
                 src/tools/hook/Makefile
                 src/tools/info/Makefile
                 src/tools/perf/Makefile
                 test/examples/Makefile
                 test/gtest/Makefile
                 ])
AC_CONFIG_FILES([test/mpi/run_mpi.sh], [chmod a+x test/mpi/run_mpi.sh])

AC_OUTPUT
