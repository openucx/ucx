# Azure wrapper around rapidsai/ci-conda: chmod /opt/conda so the non-root UID Azure runs
# steps as can use conda/python (rapidsai owns it as root); + adds gdb for stack capture.

ARG BASE_IMAGE=rapidsai/ci-conda:26.08-latest
FROM ${BASE_IMAGE}

RUN chmod -R o+rwX /opt/conda \
 && apt-get update \
 && apt-get install -y --no-install-recommends gdb \
 && rm -rf /var/lib/apt/lists/*
