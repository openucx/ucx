#!/bin/bash -eEx

# shellcheck disable=SC2086
basedir=$(cd "$(dirname $0)" && pwd)
ARCH=$(uname -m)
registry=harbor.mellanox.com/ucx/${ARCH}

images=$(awk '!/#/ && /image:/ {print $2}' "${basedir}/docker-compose-${ARCH}.yml")
for img in $images; do
    target_name="${registry}/${img}"
    docker tag ${img} ${target_name}
    docker push ${target_name}
done
