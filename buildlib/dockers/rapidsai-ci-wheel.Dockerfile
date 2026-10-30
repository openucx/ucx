# Azure wrapper around rapidsai/ci-wheel: chmod /pyenv so the non-root UID Azure runs
# steps as can write there (rapidsai owns it as root); + adds gdb for stack capture.
# One image per CUDA major: the default BASE_IMAGE is the cuda13 base, the cuda12
# image overrides it with rapidsai/ci-wheel:26.12-cuda12.9.2-rockylinux8-py3.12.
#
# Build + push both arches (from the repo root; <REV> is new for every rebuild):
#   docker buildx build --platform linux/amd64,linux/arm64 --push \
#       -t harbor.mellanox.com/ucx/rapidsai-ci-wheel:26.12-cuda13-azp-<REV> \
#       -f buildlib/dockers/rapidsai-ci-wheel.Dockerfile buildlib/dockers
#   docker buildx build --platform linux/amd64,linux/arm64 --push \
#       --build-arg BASE_IMAGE=rapidsai/ci-wheel:26.12-cuda12.9.2-rockylinux8-py3.12 \
#       -t harbor.mellanox.com/ucx/rapidsai-ci-wheel:26.12-cuda12-azp-<REV> \
#       -f buildlib/dockers/rapidsai-ci-wheel.Dockerfile buildlib/dockers
# CI pulls these tags through the rdmz-harbor.rdmz.labs.mlnx mirror; keep the
# image refs in buildlib/pr/main.yml in sync with the pushed tags.

ARG BASE_IMAGE=rapidsai/ci-wheel:26.12-cuda13.3.0-rockylinux8-py3.12
FROM ${BASE_IMAGE}

RUN chmod -R o+rwX /pyenv \
 && dnf install -y gdb \
 && dnf clean all
