#!/usr/bin/env bash
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

set -eEu -o pipefail

range="${1?Provide commit range like, for example: orig/v1.17..orig/v1.18}"

if [ ! -w AUTHORS ]
then
    echo "AUTHORS file is not writable"
    exit 1
fi

# Failure message triggered if range is not valid
lines=$(git log --no-merges --pretty=format:"%an%x09%ae" "$range" | sort -u)
if [ -z "$lines" ]
then
    echo "Error: provided range \"$range\" is empty"
    exit 1
fi

tmp=$(mktemp --tmpdir=./)
grep @ AUTHORS >"$tmp"

echo "Names:"

echo -e "$lines" | \
while IFS=$'\t' read -r name email; do
    line="$name <$email>"

    # Check for known email, or identical name
    if ! grep -iqw "$email" "$tmp" && ! grep -iq "$name" "$tmp"; then
        echo "$line" >>"$tmp"
        echo "++ $line"
    else
        echo "   $line"
    fi
done

LC_COLLATE=C sort -o "$tmp"{,}
grep -v @ AUTHORS >>"$tmp"
mv "$tmp" AUTHORS
