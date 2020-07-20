#!/bin/bash -eE

# shellcheck disable=SC2086
basedir=$(cd "$(dirname $0)" && pwd)

registry=harbor.mellanox.com/ucx
tag=1

images=$(awk '/image:/ {print $2}' "${basedir}/docker-compose.yml")
for img in $images; do
    target_name="${registry}/${img}:${tag}"
    docker tag ${img}:latest ${target_name}
    docker push ${target_name}
done
