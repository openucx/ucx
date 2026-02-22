#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

# Plugin is built separately and dynamically loaded at runtime.
# No need to check for plugin source in UCX tree.
# Plugin library (libucx_plugin_ib.so) should be installed separately.
# UCX will discover and load it automatically via the plugin infrastructure.

AC_CONFIG_FILES([src/uct/ib/plugin/Makefile
                 src/uct/ib/plugin/ucx-ib-plugin.pc])
