#!/bin/bash -eE

# shellcheck disable=SC2086
basedir=$(cd "$(dirname $0)" && pwd)

registry=harbor.mellanox.com/ucx

images=$(awk '/image:/ {print $2}' "${basedir}/docker-compose.yml")
for img in $images; do
    target_name="${registry}/${img}"
    docker tag ${img} ${target_name}
    docker push ${target_name}
done
