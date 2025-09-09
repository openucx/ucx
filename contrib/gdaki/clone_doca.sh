#!/bin/bash

[ -d doca ] && { echo "doca directory already exists"; exit 1; }

set -e

git clone ssh://git-nbu.nvidia.com:12023/doca/doca

#First commit, later reverted
#git checkout -B verbs-gpunetio ea9a643c2f27fd790d6d2197bc6e10af055caf45

cd doca
# git fetch origin refs/changes/75/1253175/2
git checkout -B master origin/master
git am ../doca-00*.patch
cd -
