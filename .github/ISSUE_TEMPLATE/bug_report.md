---
name: Bug report
about: Report an unexpected or erroneous behavior
title: ''
labels: Bug
assignees: ''

---

### Describe the bug
A clear and concise description of what the bug is.

### Steps to Reproduce
- Command line
- UCX version used (from github branch XX or release YY) + UCX configure flags (can be checked by `ucx_info -v`)
- **Any UCX environment variables used**

### Setup and versions
- OS version (e.g Linux distro) + CPU architecture (x86_64/aarch64/ppc64le/...)
   - `cat /etc/issue` or `cat /etc/redhat-release` + `uname -a`
   - For Nvidia Bluefield SmartNIC include `cat /etc/mlnx-release` (the string identifies software and firmware setup)
- For RDMA/IB/RoCE related issues:
    - Driver version:
        - `rpm -q rdma-core` or `rpm -q libibverbs`
        - or: MLNX_OFED version `ofed_info -s`
   - HW information from `ibstat` or `ibv_devinfo -vv` command
- For GPU related issues:
  - GPU type
  - Cuda: 
      - Drivers version
      - Check if peer-direct is loaded: `lsmod|grep nv_peer_mem` and/or gdrcopy: `lsmod|grep gdrdrv`

### Additional information (depending on the issue)
- OpenMPI version
- Output of `ucx_info -d` to show transports and devices recognized by UCX
- Configure result - config.log
- Log file - configure UCX with "--enable-logging" - and run with "UCX_LOG_LEVEL=data"
