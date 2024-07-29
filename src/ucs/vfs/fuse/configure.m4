#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
#


AS_IF([test "x$fuse3_happy" = "xyes"], [ucs_modules="${ucs_modules}:fuse"])
AC_CONFIG_FILES([src/ucs/vfs/fuse/Makefile
                 src/ucs/vfs/fuse/ucx-fuse.pc])
