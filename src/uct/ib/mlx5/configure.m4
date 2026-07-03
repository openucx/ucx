#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# Add IB mlx5 provider support
#

AC_CONFIG_FILES([src/uct/ib/mlx5/Makefile src/uct/ib/mlx5/ucx-ib-mlx5.pc])

m4_include([src/uct/ib/mlx5/gdaki/configure.m4])

AC_DEFINE_UNQUOTED([uct_ib_mlx5_MODULES],
                   ["${uct_ib_mlx5_modules}"],
                   [IB MLX5 loadable modules])
