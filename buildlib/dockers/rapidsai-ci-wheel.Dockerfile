# Azure Pipelines wrapper around rapidsai/ci-wheel.
# Opens /pyenv so the Azure-injected step user can write shims.

ARG BASE_IMAGE=rapidsai/ci-wheel:26.06-latest
FROM ${BASE_IMAGE}

RUN chmod -R o+rwX /pyenv
