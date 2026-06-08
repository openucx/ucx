# Azure wrapper around rapidsai/ci-wheel: chmod /pyenv so the non-root UID Azure runs
# steps as can write there (rapidsai owns it as root); + adds gdb for stack capture.
# Default base = cuda13; cuda12 image built by overriding BASE_IMAGE to the cuda12.9.1 tag.

ARG BASE_IMAGE=rapidsai/ci-wheel:26.08-cuda13.2.0-rockylinux8-py3.11
FROM ${BASE_IMAGE}

RUN chmod -R o+rwX /pyenv \
 && dnf install -y gdb \
 && dnf clean all
