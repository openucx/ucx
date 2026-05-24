# Azure Pipelines wrapper around rapidsai/ci-conda.
# Opens /opt/conda so the Azure-injected step user can use conda/python.

ARG BASE_IMAGE=rapidsai/ci-conda:26.06-latest
FROM ${BASE_IMAGE}

RUN chmod -R o+rwX /opt/conda
