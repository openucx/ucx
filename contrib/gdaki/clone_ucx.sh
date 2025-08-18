#!/bin/bash

git clone ssh://git@gitlab-master.nvidia.com:12051/ofarjon/ucx-fork.git
cd ucx-fork
git checkout origin/ak_wip
git am <../ucx-0*.patch
cd -
