#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
#


ucs_modules="${ucs_modules}:topo_sysfs:topo_default"
AC_CONFIG_FILES([src/ucs/sys/topo/providers/sysfs/Makefile
                 src/ucs/sys/topo/providers/sysfs/ucx-topo-sysfs.pc
                 src/ucs/sys/topo/providers/default/Makefile
                 src/ucs/sys/topo/providers/default/ucx-topo-default.pc])
