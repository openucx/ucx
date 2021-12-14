#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#


AC_FUNC_ALLOCA


#
# SystemV shared memory
#
#IPC_INFO
AC_CHECK_LIB([rt], [shm_open],     [], AC_MSG_ERROR([librt not found]))
AC_CHECK_LIB([rt], [timer_create], [], AC_MSG_ERROR([librt not found]))


#
# Extended string functions
#
AC_CHECK_HEADERS([libgen.h])
AC_CHECK_DECLS([asprintf, basename, fmemopen], [],
				AC_MSG_ERROR([GNU string extensions not found]), 
				[#define _GNU_SOURCE 1
				 #include <string.h>
				 #include <stdio.h>
				 #ifdef HAVE_LIBGEN_H
				 #include <libgen.h>
				 #endif
				 ])


#
# CPU-sets 
#
AC_CHECK_HEADERS([sys/cpuset.h])
AC_CHECK_DECLS([CPU_ZERO, CPU_ISSET], [], 
				AC_MSG_ERROR([CPU_ZERO/CPU_ISSET not found]), 
				[#define _GNU_SOURCE 1
				 #include <sys/types.h>
				 #include <sched.h>
				 #ifdef HAVE_SYS_CPUSET_H
				 #include <sys/cpuset.h>
				 #endif
				 ])
AC_CHECK_TYPES([cpu_set_t, cpuset_t], [], [],
			   [#define _GNU_SOURCE 1
			    #include <sys/types.h>
			    #include <sched.h>
			    #ifdef HAVE_SYS_CPUSET_H
			    #include <sys/cpuset.h>
			    #endif])


#
# Type for sighandler
#
AC_CHECK_TYPES([sighandler_t,  __sighandler_t], [], [],
			      [#define _GNU_SOURCE 1
			       #include <signal.h>])


#
# pthread
#
AC_CHECK_HEADERS([pthread_np.h])
AC_SEARCH_LIBS(pthread_create, pthread)
AC_SEARCH_LIBS(pthread_atfork, pthread)


#
# Misc. Linux-specific functions
#
AC_CHECK_FUNCS([clearenv])
AC_CHECK_FUNCS([malloc_trim])
AC_CHECK_FUNCS([memalign])
AC_CHECK_FUNCS([posix_memalign])
AC_CHECK_FUNCS([mremap])
AC_CHECK_FUNCS([sched_setaffinity sched_getaffinity])
AC_CHECK_FUNCS([cpuset_setaffinity cpuset_getaffinity])


#
# Route file descriptor signal to specific thread
#
AC_CHECK_DECLS([F_SETOWN_EX], [], [], [#define _GNU_SOURCE 1
#include <fcntl.h>])


#
# Ethtool definitions
#
AC_CHECK_DECLS([ethtool_cmd_speed, SPEED_UNKNOWN], [], [],
               [#include <linux/ethtool.h>])


#
# PowerPC "sys/platform/ppc.h" header
#
AC_CHECK_HEADERS([sys/platform/ppc.h])


#
# PowerPC query for getting TB and frequency
#
AC_CHECK_DECLS([__ppc_get_timebase_freq, __ppc_get_timebase], [], [],
               [[#include <sys/platform/ppc.h>]])


#
# Google Testing framework
#
GTEST_LIB_CHECK([1.5.0], [true], [true])


#
# Valgrind support
#
AC_ARG_WITH([valgrind],
    AC_HELP_STRING([--with-valgrind],
                   [Enable Valgrind annotations (small runtime overhead, default NO)]),
    [],
    [with_valgrind=no]
)
AS_IF([test "x$with_valgrind" = xno],
      [AC_DEFINE([NVALGRIND], 1, [Define to 1 to disable Valgrind annotations.])],
      [AS_IF([test ! -d $with_valgrind], 
              [AC_MSG_NOTICE([Valgrind path was not defined, guessing ...])
               with_valgrind=/usr], [:])
        AC_CHECK_HEADER([$with_valgrind/include/valgrind/memcheck.h], [],
                       [AC_MSG_ERROR([Valgrind memcheck support requested, but <valgrind/memcheck.h> not found, install valgrind-devel rpm.])])
        CPPFLAGS="$CPPFLAGS -I$with_valgrind/include"
      ]
)


#
# NUMA support
#
AC_ARG_ENABLE([numa],
    AC_HELP_STRING([--disable-numa], [Disable NUMA support]),
    [],
    [enable_numa=guess])
AS_IF([test "x$enable_numa" = xno],
    [
     AC_MSG_NOTICE([NUMA support is explictly disabled])
     numa_enable=disabled
    ],
    [
     save_LDFLAGS="$LDFLAGS"

     numa_happy=yes
     AC_CHECK_HEADERS([numa.h numaif.h], [], [numa_happy=no])
     AC_CHECK_LIB(numa, mbind,
                  [AC_SUBST(NUMA_LIBS, [-lnuma])],
                  [numa_happy=no])
     AC_CHECK_TYPES([struct bitmask], [], [numa_happy=no], [[#include <numa.h>]])

     LDFLAGS="$save_LDFLAGS"

     AS_IF([test "x$numa_happy" = xyes],
           [
            AC_DEFINE([HAVE_NUMA], 1, [Define to 1 to enable NUMA support])
            numa_enable=enabled
           ],
           [
            AC_DEFUN([NUMA_W1], [NUMA support not found])
            AC_DEFUN([NUMA_W2], [Please consider installing libnuma-devel package.])
            AS_IF([test "x$enable_numa" = xyes],
                  [AC_MSG_ERROR([NUMA_W1. NUMA_W2])],
                  [
                   AC_MSG_WARN([NUMA_W1, this many impact library performance.])
                   AC_MSG_WARN([NUMA_W2])
                  ])
            numa_enable=disabled
           ])
    ])


#
# Malloc hooks
#
AC_MSG_CHECKING([malloc hooks])
SAVE_CFLAGS=$CFLAGS
CFLAGS="$CFLAGS $CFLAGS_NO_DEPRECATED"
CHECK_CROSS_COMP([AC_LANG_SOURCE([#include <malloc.h>
                                  static int rc = 1;
                                  void *ptr;
                                  void *myhook(size_t size, const void *caller) {
                                      rc = 0;
                                      return NULL;
                                  }
                                  int main(int argc, char** argv) {
                                      __malloc_hook = myhook;
                                      ptr = malloc(1);
                                      return rc;
                                  }])],
                 [AC_MSG_RESULT([yes])
                  AC_DEFINE([HAVE_MALLOC_HOOK], 1, [malloc hooks support])],
                 [AC_MSG_RESULT([no])
                  AC_MSG_WARN([malloc hooks are not supported])]
                )
CFLAGS=$SAVE_CFLAGS


#
# Check for capability.h header (usually comes from libcap-devel package) and
# make sure it defines the types we need
#
AC_CHECK_HEADERS([sys/capability.h],
                 [AC_CHECK_TYPES([cap_user_header_t, cap_user_data_t], [],
                                 [AC_DEFINE([HAVE_SYS_CAPABILITY_H], [0], [Linux capability API support])],
                                 [[#include <sys/capability.h>]])]
                 )

#
# Check for PR_SET_PTRACER
#
AC_CHECK_DECLS([PR_SET_PTRACER], [], [], [#include <sys/prctl.h>])


#
# ipv6 s6_addr32/__u6_addr32 shortcuts for in6_addr
# ip header structure layout name
#
AC_CHECK_MEMBER(struct in6_addr.s6_addr32,
	[AC_DEFINE([HAVE_IN6_ADDR_S6_ADDR32], [1],
		[struct in6_addr has s6_addr32 member])],
	[],
	[#include <netinet/in.h>])
AC_CHECK_MEMBER(struct in6_addr.__u6_addr.__u6_addr32,
	[AC_DEFINE([HAVE_IN6_ADDR_U6_ADDR32], [1],
	        [struct in6_addr is BSD-style])],
	[],
	[#include <netinet/in.h>])
AC_CHECK_MEMBER(struct iphdr.daddr.s_addr,
	[AC_DEFINE([HAVE_IPHDR_DADDR], [1],
		[struct iphdr has daddr member])],
	[],
	[#include <linux/ip.h>])
AC_CHECK_MEMBER(struct ip.ip_dst.s_addr,
	[AC_DEFINE([HAVE_IP_IP_DST], [1],
	        [struct ip has ip_dst member])],
	[],
	[#include <sys/types.h>
	 #include <netinet/in.h>
	 #include <netinet/ip.h>])


#
# struct sigevent reporting thread id
#
AC_CHECK_MEMBER(struct sigevent._sigev_un._tid,
	[AC_DEFINE([HAVE_SIGEVENT_SIGEV_UN_TID], [1],
		[struct sigevent has _sigev_un._tid])],
	[],
	[#include <signal.h>])
AC_CHECK_MEMBER(struct sigevent.sigev_notify_thread_id,
	[AC_DEFINE([HAVE_SIGEVENT_SIGEV_NOTIFY_THREAD_ID], [1],
	        [struct sigevent has sigev_notify_thread_id])],
	[],
	[#include <signal.h>])


#
# sa_restorer is something that only Linux has
#
AC_CHECK_MEMBER(struct sigaction.sa_restorer,
	[AC_DEFINE([HAVE_SIGACTION_SA_RESTORER], [1],
		[struct sigaction has sa_restorer member])],
	[],
	[#include <signal.h>])


#
# epoll vs. kqueue
#
AC_CHECK_HEADERS([sys/epoll.h])
AC_CHECK_HEADERS([sys/eventfd.h])
AC_CHECK_HEADERS([sys/event.h])


#
# FreeBSD-specific threading functions
#
AC_CHECK_HEADERS([sys/thr.h])


#
# malloc headers are Linux-specific
#
AC_CHECK_HEADERS([malloc.h])
AC_CHECK_HEADERS([malloc_np.h])


#
# endianess
#
AC_CHECK_HEADERS([endian.h, sys/endian.h])


#
# Linux-only headers
#
AC_CHECK_HEADERS([linux/mman.h])
AC_CHECK_HEADERS([linux/ip.h])
AC_CHECK_HEADERS([linux/futex.h])


#
# Networking headers
#
AC_CHECK_HEADERS([net/ethernet.h], [], [],
	[#include <sys/types.h>])
AC_CHECK_HEADERS([netinet/ip.h], [], [],
	[#include <sys/types.h>
	 #include <netinet/in.h>])
