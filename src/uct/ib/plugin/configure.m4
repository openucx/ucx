#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Check for plugin source directory existence
# Plugin source should be in src/uct/ib/plugin/ucx_plugin/
# The entire plugin repository is copied here
#
plugin_dir="$srcdir/src/uct/ib/plugin/ucx_plugin"

AS_IF([test -d "$plugin_dir" -a -d "$plugin_dir/src" -a -f "$plugin_dir/src/ucx_plugin.c"],
      [
       have_plugin=yes
       AC_DEFINE([HAVE_IB_PLUGIN], [1], [IB plugin support])
       AC_MSG_NOTICE([IB plugin source found in $plugin_dir])
      ],
      [
       have_plugin=no
       AC_MSG_NOTICE([IB plugin source not found, using stub implementation])
      ])

AM_CONDITIONAL([HAVE_IB_PLUGIN], [test "x$have_plugin" = "xyes"])
AC_CONFIG_FILES([src/uct/ib/plugin/Makefile
                 src/uct/ib/plugin/ucx-ib-plugin.pc])
