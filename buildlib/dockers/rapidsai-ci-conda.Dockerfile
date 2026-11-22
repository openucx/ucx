# Azure wrapper around rapidsai/ci-conda: chmod /opt/conda so the non-root UID Azure runs
# steps as can use conda/python (rapidsai owns it as root); + adds gdb for stack capture.
#
# Build + push both arches (from the repo root; <REV> is new for every rebuild):
#   docker buildx build --platform linux/amd64,linux/arm64 --push \
#       -t harbor.mellanox.com/ucx/rapidsai-ci-conda:26.12-azp-<REV> \
#       -f buildlib/dockers/rapidsai-ci-conda.Dockerfile buildlib/dockers
# CI pulls this tag through the rdmz-harbor.rdmz.labs.mlnx mirror; keep the
# image refs in buildlib/pr/main.yml in sync with the pushed tag.

ARG BASE_IMAGE=rapidsai/ci-conda:26.12-latest
FROM ${BASE_IMAGE}

RUN chmod -R o+rwX /opt/conda \
 && apt-get update \
 && apt-get install -y --no-install-recommends gdb \
 && rm -rf /var/lib/apt/lists/*
