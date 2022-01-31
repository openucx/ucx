#
# Copyright (C) 2022 NVIDIA CORPORATION & AFFILIATES. All Rights Reserved.
# See file LICENSE for terms.
#

# helper macro
m4_define([_CHECK_MODULE_MAKE_HELP_STRING],
          [m4_default(m4_argn($1, $2), [$3])])

m4_ifndef([PKG_CHECK_MODULES_STATIC],
          [AC_DEFUN([PKG_CHECK_MODULES_STATIC],
                    [AC_REQUIRE([PKG_PROG_PKG_CONFIG])
                     save_PKG_CONFIG=$PKG_CONFIG
                     PKG_CONFIG="$PKG_CONFIG --static"
                     PKG_CHECK_MODULES([$1], [$2], [$3], [$4])
                     PKG_CONFIG=$save_PKG_CONFIG
                    ])])

m4_define([_CHECK_MODULE_DEFVAR],
          [
             m4_pushdef([mod_header], [m4_argn([1], $1)])
             m4_pushdef([mod_decls], [m4_argn([2], $1)])
             m4_pushdef([mod_includes], [m4_argn([3], $1)])
             m4_pushdef([mod_libs], [m4_argn([4], $1)])
             m4_pushdef([mod_funcs], [m4_argn([5], $1)])
             m4_pushdef([mod_ldflags], [m4_argn([6], $1)])
          ])

m4_define([_CHECK_MODULE_UNDEFVAR],
          [
             m4_popdef([mod_ldflags])dnl
             m4_popdef([mod_funcs])dnl
             m4_popdef([mod_libs])dnl
             m4_popdef([mod_includes])dnl
             m4_popdef([mod_decls])dnl
             m4_popdef([mod_header])dnl
          ])

#
# Init progress: set 'progress' variable to one of values yes/no/guess/skip/dir
#                based on 'mod-dir' value. In case of 'dir' value is set then
#                also set dir-int, inc-flag, dir-lib and dir-flag variables or call
#                'if-failed' macro if some of directories are missing
#
# Usage: _CHECK_MODULE_INIT_PROGRESS([mod-dir], [progress],
#                                    [dir-int], [inc-flag], [dir-lib], [dir-flag], [if-failed])
#
m4_define([_CHECK_MODULE_INIT_PROGRESS],
         [AS_CASE(["x$1"],
                  ["xyes"],   [$2=yes],
                  ["xno"],    [$2=skip],
                  ["xguess"], [$2=guess],
                  ["x"],      [$2=skip],
                              [$2=dir
                               _CHECK_MODULE_SET_DIR([$1], [$3], [$4], [$5],
                                                     [$6], [$7])])
         ])

#
# Configure module directories:
#     MOD_INC_DIR=$MOD_DIR/include
#     MOD_CFLAGS=-I$MOD_DIR/include
#     MOD_LIB_DIR=$MOD_DIR/lib64 or $MOD_DIR/MOD_LIB_DIR
#     MOD_LDFLAGS=-L$MOD_LIB_DIR (from previous step)
#     if any of directories is missing - call action_if_failed macro
#
# Usage: _CHECK_MODULE_SET_DIR([MOD_DIR], [MOD_INC_DIR], [MOD_CFLAGS], [MOD_LIB_DIR], [MOD_LDFLAGS], [action_if_failed])
#
m4_define([_CHECK_MODULE_SET_DIR],
          [
             $2=$1/include
             $3=-I$1/include
             # use macro unwind
             m4_pushdef([dir_lib], [$4])
             _CHECK_MODULE_FIXUP_LIB([$1], [$4], [$5=-L$dir_lib], [$6])
             m4_popdef([dir_lib])
          ])

#
# Run 'macro' if status is in allowed list or 'if-no-macro' in all other cases
#
# Usage: _CHECK_MODULE_RUN([status], [allowed-statuses-list], [macro], [if-no-macro])
#
m4_define([_CHECK_MODULE_RUN],
          [
             mod_run_happy=no
             m4_foreach([stat], [$2],
                 [
                     AS_IF([test "$1" = "stat"], [mod_run_happy=yes])dnl
                 ])
             AS_IF([test $mod_run_happy = yes], [$3], [$4])
          ])

#
# Check if all directories listed in $1 exist
#
# Usage: _CHECK_MODULE_IS_DIR([dir1, dir2, ...], [if-yes], [if-no])
#
AC_DEFUN([_CHECK_MODULE_IS_DIR],
         [
             AS_IF([test -n "$1"], [module_check_dir_happy=yes], [module_check_dir_happy=no])dnl
             m4_foreach([dir], [$1],
                        [
                            AS_IF([test -d "dir"], [], [module_check_dir_happy=no])dnl
                        ])
             AS_IF([test $module_check_dir_happy = yes], [$2], [$3])dnl
         ])

#
# Fixup for lib path: check if $lib/lib64 exists, then set dest_var to $lib/lib64,
#                     if not - check if $lib/lib exists, then set dest_var to
#                     $lib/lib value. If one of directories exists call
#                     if-success macro, otherwise call call if-not macro.
#
# Usage: _CHECK_MODULE_FIXUP_LIB([lib], [dest_var], [if-success], [if-not])
#
AC_DEFUN([_CHECK_MODULE_FIXUP_LIB],
         [
             _CHECK_MODULE_IS_DIR(
                 [$1/lib64],
                 [$2=$1/lib64
                  $3],
                 [_CHECK_MODULE_IS_DIR([$1/lib],
                                       [$2=$1/lib
                                        $3], [$4])])
         ])

#
# Check module headers
#
# Usage: _CHECK_MODULE_HEADERS([header], [cppflags], [if-found], [if-not-found], [includes])
#
AC_DEFUN([_CHECK_MODULE_HEADERS],
         [
             mod_header_saved_CPPFLAGS=$CPPFLAGS
             mod_header_happy=yes
             CPPFLAGS="$2 $CPPFLAGS"

             m4_foreach([header], [$1],
                 [
                  # have to clean check header cache to allow multiple checks
                  AS_VAR_PUSHDEF([header_cache], [ac_cv_header_[]header])dnl
                  unset header_cache
                  mod_header_name=header
                  AC_CHECK_HEADER([$mod_header_name], [], [mod_header_happy=no], [$5])
                  AS_VAR_POPDEF([header_cache])])dnl

             CPPFLAGS=$mod_header_saved_CPPFLAGS

             AS_IF([test $mod_header_happy = yes], [$3], [$4])
         ])

#
# Check module declarations
#
# Usage: _CHECK_MODULE_DECLS([decls], [cppflags], [if-found], [if-not-found], [includes])
#
AC_DEFUN([_CHECK_MODULE_DECLS],
         [
             m4_foreach([decl], [$1],
                 [
                  # have to clean declaration cache to allow multiple checks
                  AS_VAR_PUSHDEF([decl_cache], [ac_cv_have_decl_[]decl])dnl
                  unset decl_cache
                  AS_VAR_POPDEF([decl_cache])])dnl

             mod_decls_saved_CPPFLAGS=$CPPFLAGS
             mod_decls_happy=yes
             CPPFLAGS="$2 $CPPFLAGS"

             AS_IF([test -n "$1"],
                   [AC_CHECK_DECLS([$1], [], [mod_decls_happy=no], [$5])])

             CPPFLAGS=$mod_decls_saved_CPPFLAGS

             # in case if decls is empty list - then assume that test is successfull
             AS_IF([test $mod_decls_happy = yes], [$3], [$4])
         ])

#
# Check module libraries
#
# Usage: _CHECK_MODULE_LIBS([libs], [libdir], [funcs], [if-found], [if-not-found], [otherlibs])
#
AC_DEFUN([_CHECK_MODULE_LIBS],
         [
             mod_libs_saved_LDFLAGS=$LDFLAGS
             mod_libs_saved_LIBS=$LIBS # save it because AC_SEARCH_LIBS breaks LIBS var
             mod_libs_happy=yes
             LDFLAGS="$2 $LDFLAGS"

             m4_foreach([func], [$3],
                 [
                  # have to clean func check cache to allow multiple checks
                  AS_VAR_PUSHDEF([func_cache], [ac_cv_search_[]func])dnl
                  unset func_cache
                  mod_func_name=func
                  AC_SEARCH_LIBS([$mod_func_name], [$1], [], [mod_libs_happy=no], [$6])
                  AS_VAR_POPDEF([func_cache])])dnl

             LIBS=$mod_libs_saved_LIBS
             LDFLAGS=$mod_libs_saved_LDFLAGS

             AS_IF([test $mod_libs_happy = yes], [$4], [$5])
         ])

#
# Check module API
#
# Usage: _CHECK_MODULE_API([incdir], [libdir], [if-found], [if-not-found],
#                          [[header], [decls], [includes], [libs], [funcs], [otherlibs]])
#
AC_DEFUN([_CHECK_MODULE_API],
         [
             _CHECK_MODULE_DEFVAR([$5])
             mod_api_happy=no

             _CHECK_MODULE_HEADERS(mod_header, [$1],
                     [_CHECK_MODULE_DECLS(mod_decls, [$1],
                              [_CHECK_MODULE_LIBS(mod_libs, [$2], mod_funcs,
                                      [
                                          $3
                                          mod_api_happy=yes
                                      ],
                                      [], [mod_ldflags])],
                                      [], mod_includes)])

             _CHECK_MODULE_UNDEFVAR
             AS_IF([test $mod_api_happy = no], [$4])
         ])

#
# Check module package
#
# Usage: _CHECK_MODULE_PACKAGE([pkgid], [prefix], [libdir], [static], [if-found], [if-not-found],
#                              [[header], [decls], [includes], [libs], [funcs], [ldflags]])
#
# DO NOT!!!!! use macro substitution for 'prefix' var of  _CHECK_MODULE_PACKAGE
# because m4 can't expand macro in PKG_CHECK_MODULES
#
AC_DEFUN([_CHECK_MODULE_PACKAGE],
         [
             mod_pkg_saved_PKG_CONFIG_PATH="$PKG_CONFIG_PATH"
             mod_package_happy=yes

             AS_IF([test -z "$1"], [mod_package_happy=no])

             AS_IF([test -n "$3"],
                   [_CHECK_MODULE_IS_DIR([$3/pkgconfig],
                                         [export PKG_CONFIG_PATH="$3/pkgconfig:$PKG_CONFIG_PATH"],
                                         [mod_package_happy=no])])

             AS_IF([test $mod_package_happy = yes],
                   [AS_IF([test "$4" = "static"],
                          [PKG_CHECK_MODULES_STATIC([$2], [$1], [], [mod_package_happy=no])],
                          [PKG_CHECK_MODULES([$2], [$1], [], [mod_package_happy=no])])])

             export PKG_CONFIG_PATH="$mod_pkg_saved_PKG_CONFIG_PATH"

             AS_IF([test $mod_package_happy = yes],
                   [_CHECK_MODULE_API([$[$2_CFLAGS]], [$[$2_LIBS]],
                                      [], [mod_package_happy=no], [$7])])

             AS_IF([test $mod_package_happy = yes], [$5], [$6])
         ])

# Check if module available
# Result: set and substitute variables prefix_CFLAGS and prefix_LIBS
#
# Algo:
#     - if set static lib
#       1. check it using package configuration (--static) and try to
#          detect whole set of dependencies
#       2. check it using includes/ldflags provided by user
#     - for dynamic lib
#       1. check for lib using declarations/link
#       2. check for package
#
# Usage: UCX_CHECK_MODULE([id], [name], [prefix], [pkgname],
#                         [[help], [help-lib], [help-static], [fail-msg]],
#                         [action-if-found], [action-if-not-found],
#                         [[headers], [decls], [includes], [libs], [funcs], [otherlibs]],
#                         [alt-dirs])
#
# Parameters:
#   id                 - module ID, used to generate runtime configuration parameters --with-id
#   name               - readable module name
#   prefix             - variable prefix, in success variables prefix_CFLAGS and
#                        prefix_LIBS will store module specific flags
#   pkgname            - package name used to lookup module using pkg-config tool, optional
#   help               - help string for module or empty argument to generate it automatically
#   help-lib           - help string for module library dir or empty argument to generate
#                        it automatically
#   help-static        - help string for static module library or empty argument to
#                        generate it automatically, if static link is not needed set it
#                        to '-' value (see usage example)
#   action-if-found    - macro called if module configuration found
#   action-if-notfound - macro called if module configuration not found or user disabled module
#   headers            - list of header files to check, coma-separated, optional
#   decls              - declarations to check, coma-separated, optional
#   includes           - additional include directives to check headers and declarations, optional
#   libs               - list of libraries to check, whitespace-separated
#   funcs              - list of the symbols to look for in the libraries, coma-separated
#   otherlibs          - extra libraries (including '-l' prefix) and linker flags to
#                        check libraries, optional
#   alt-dirs           - alternative directories to lookup: in case if user didn't
#                        specify exact path to module then script will use this list of
#                        directories, optional
#
# Example:
# UCX_CHECK_MODULE([xpmem], [XPMEM], [XPMEM], [xpmem],
#                  [[], [], [-], []], # no static link required
#                   [have_xpmem=true], [have_xpmem=false],
#                   [[xpmem.h], [xpmem_version], [[#include <xpmem.h>]], [xpmem], [xpmem_init], []],
#                   [/opt/xpmem, /usr/local, /usr/local/xpmem])
#
AC_DEFUN([UCX_CHECK_MODULE],
         [
             # AS_VAR_PUSHDEF doesn't work here due to processed in autoconf.
             # use m4_pushdef instead to instantiate id
             m4_pushdef([mod_hlp], [$5])
             AC_ARG_WITH([$1],
                         [AS_HELP_STRING([--with-$1=(DIR)],
                                         _CHECK_MODULE_MAKE_HELP_STRING([1], [mod_hlp], [Enable the use of $2 (default is guess).]))],
                         [], [with_$1=guess])

             AC_ARG_WITH([$1-libdir],
                         [AS_HELP_STRING([--with-$1-libdir=(DIR)],
                                         _CHECK_MODULE_MAKE_HELP_STRING([2], [mod_hlp], [Search $2 library in DIR.]))],
                         [], [with_$1_libdir=""])

             m4_if(m4_argn(3, $5), [-], [], [
                 AC_ARG_WITH([$1-static],
                             [AS_HELP_STRING([--with-$1-static=(DIR)],
                                             _CHECK_MODULE_MAKE_HELP_STRING([3], [mod_hlp], [Static link $2 (default is no).]))],
                             [], [with_$1_static=no])])

             # define 'local' variables
             AS_VAR_PUSHDEF([mod_name], [$2])
             AS_VAR_PUSHDEF([mod_prefix], [$3])
             AS_VAR_PUSHDEF([mod_pkg], [$4])
             m4_pushdef([mod_alt_dir], [$9])

             _CHECK_MODULE_DEFVAR([$8])

             AS_VAR_PUSHDEF([mod_with], [with_$1])
             AS_VAR_PUSHDEF([mod_with_libdir], [with_$1_libdir])
             AS_VAR_PUSHDEF([mod_with_static], [with_$1_static])
             AS_VAR_PUSHDEF([mod_dir_inc], [include_$1])
             AS_VAR_PUSHDEF([mod_dir_inc_flag], [include_flag_$1])
             AS_VAR_PUSHDEF([mod_dir_lib], [lib_$1])
             AS_VAR_PUSHDEF([mod_dir_flag], [lib_flag_$1])

             mod_dir_inc=""
             mod_dir_inc_flag=""
             mod_dir_lib=""
             mod_dir_flag=""

             mod_progress=start

             # static module check
             _CHECK_MODULE_RUN([$mod_progress], [start],
                     [
                         _CHECK_MODULE_INIT_PROGRESS([$mod_with_static], [mod_progress],
                             [mod_dir_inc], [mod_dir_inc_flag], [mod_dir_lib], [mod_dir_flag], [mod_progress=fatal])
                     ])

             _CHECK_MODULE_RUN([$mod_progress], [dir],
                     [
                         _CHECK_MODULE_IS_DIR([$mod_with_static, $mod_dir_inc, $mod_dir_lib],
                                              [], [mod_progress=fatal])
                     ])

             _CHECK_MODULE_RUN([$mod_progress], [yes, guess, dir],
                     [
                         _CHECK_MODULE_PACKAGE([mod_pkg], [$3], [$mod_dir_lib], [static],
                                               [mod_progress=complete], [], [$8])
                     ])

             _CHECK_MODULE_RUN([$mod_progress], [yes, guess, dir],
                     [
                         _CHECK_MODULE_API($mod_dir_inc_flag, $mod_dir_flag,
                                           [mod_progress=success], [], [$8])
                     ])

             # test is failed
             _CHECK_MODULE_RUN([$mod_progress], [yes, dir], [mod_progress=fatal])
             # test is failed or skipped but can continue
             _CHECK_MODULE_RUN([$mod_progress], [guess, skip], [mod_progress=start])

             # dynamic module check
             _CHECK_MODULE_RUN([$mod_progress], [start],
                     [
                         _CHECK_MODULE_INIT_PROGRESS([$mod_with], [mod_progress],
                             [mod_dir_inc], [mod_dir_inc_flag], [mod_dir_lib], [mod_dir_flag], [])
                         # do not 'fatal' here because lib dir may be missed
                         # will check directories later
                     ])

             _CHECK_MODULE_RUN([$mod_progress], [yes, guess, dir],
                     [
                         AS_IF([test -n "$mod_with_libdir"],
                               [_CHECK_MODULE_IS_DIR([$mod_with_libdir],
                                                     [mod_dir_lib=$mod_with_libdir],
                                                     [mod_progress=fatal])])dnl
                     ])

             _CHECK_MODULE_RUN([$mod_progress], [dir],
                     [
                         _CHECK_MODULE_IS_DIR([$mod_with, $mod_dir_inc, $mod_dir_lib],
                                              [], [mod_progress=fatal])
                     ])

             _CHECK_MODULE_RUN([$mod_progress], [yes, guess, dir],
                     [
                         _CHECK_MODULE_API($mod_dir_inc_flag, $mod_dir_flag,
                                           [mod_progress=success], [], [$8])
                     ])

             _CHECK_MODULE_RUN([$mod_progress], [yes, guess, dir],
                     [
                         _CHECK_MODULE_PACKAGE([mod_pkg], [$3], [$mod_dir_lib], [],
                                               [mod_progress=complete], [], [$8])
                     ])

             # try to search in alternative directories
             _CHECK_MODULE_RUN([$mod_progress], [yes, guess],
                     [
                         alt_progress=start

                         m4_foreach([alt_dir_m], [mod_alt_dir],
                             [
                                 alt_dir=alt_dir_m
                                 _CHECK_MODULE_RUN([$alt_progress], [start, skip, dir],
                                         [
                                             _CHECK_MODULE_INIT_PROGRESS([$alt_dir], [alt_progress],
                                                 [mod_dir_inc], [mod_dir_inc_flag], [mod_dir_lib], [mod_dir_flag],
                                                 [alt_progress=skip])
                                         ])

                                 _CHECK_MODULE_RUN([$alt_progress], [dir],
                                         [
                                             _CHECK_MODULE_IS_DIR([$alt_dir, $mod_dir_inc, $mod_dir_lib],
                                                                  [], [alt_progress=skip])
                                         ])

                                 _CHECK_MODULE_RUN([$alt_progress], [dir],
                                         [
                                             _CHECK_MODULE_API($mod_dir_inc_flag, $mod_dir_flag,
                                                               [alt_progress=success
                                                                mod_progress=success], [], [$8])
                                         ])

                                 _CHECK_MODULE_RUN([$alt_progress], [dir],
                                         [
                                             _CHECK_MODULE_PACKAGE([mod_pkg], [$3], [$mod_dir_lib], [],
                                                                   [alt_progress=complete
                                                                    mod_progress=complete], [], [$8])
                                         ])
                             ])
                     ])

             # processing results
             _CHECK_MODULE_RUN([$mod_progress], [success],
                     [
                         mod_prefix[]_CFLAGS=$mod_dir_inc_flag
                         # add -l prefix to libs
                         mod_prefix[]_LIBS="$mod_dir_flag m4_map_args_w(mod_libs, [ -l])"
                         mod_progress=complete
                     ])

             _CHECK_MODULE_RUN([$mod_progress], [skip], [$7], [AC_MSG_CHECKING(mod_name)])
             _CHECK_MODULE_RUN([$mod_progress], [complete],
                               [AC_MSG_RESULT(yes)
                                AC_SUBST(mod_prefix[]_LIBS)
                                AC_SUBST(mod_prefix[]_CFLAGS)
                                $6])
             _CHECK_MODULE_RUN([$mod_progress], [guess],
                               [AC_MSG_RESULT(no)
                                $7])
             _CHECK_MODULE_RUN([$mod_progress], [fatal, yes, dir],
                               [AC_MSG_RESULT(no)
                                AC_MSG_ERROR(m4_default(m4_argn(4, $5),
                                                        [$2 could not be found]))])

             AS_VAR_POPDEF([mod_dir_flag])dnl
             AS_VAR_POPDEF([mod_dir_lib])dnl
             AS_VAR_POPDEF([mod_dir_inc_flag])dnl
             AS_VAR_POPDEF([mod_dir_inc])dnl
             AS_VAR_POPDEF([mod_with_libdir])dnl
             AS_VAR_POPDEF([mod_with_static])dnl
             AS_VAR_POPDEF([mod_with])dnl

             _CHECK_MODULE_UNDEFVAR

             m4_popdef([mod_alt_dir])dnl
             AS_VAR_POPDEF([mod_pkg])dnl
             AS_VAR_POPDEF([mod_prefix])dnl
             AS_VAR_POPDEF([mod_name])dnl

             m4_popdef([mod_hlp])dnl
         ])

