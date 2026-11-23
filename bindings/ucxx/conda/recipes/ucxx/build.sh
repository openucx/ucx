#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

ucxx_ROOT="$(realpath ./cpp/build)"
export ucxx_ROOT
./build.sh -n -v libucxx libucxx_python libucxx_benchmarks libucxx_examples libucxx_tests --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
