# Azure Pipelines wrapper around rapidsai/ci-conda: adds sudo (Azure
# container job contract) and opens /opt/conda to the auto-created step user.

ARG BASE_IMAGE=rapidsai/ci-conda:26.06-latest
FROM ${BASE_IMAGE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends sudo passwd \
 && rm -rf /var/lib/apt/lists/*

RUN chmod -R o+rwX /opt/conda
