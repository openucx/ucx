#!/usr/bin/env awk -f
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

BEGIN {
    OFS="\t"
}

function emit(type, name) {
    print name, FILENAME, "/^" $0 "$/;\"", "f", "typeref:typename:"type, "file:"
}

function fail(line) {
    print "Failed to get function name for:" | "cat >&2"
    print line | "cat >&2"
}

/UCS_PROFILE_FUNC_VOID[\t ]*\(/ {
    match($0, /UCS_PROFILE_FUNC_VOID\([\t ]*([^, \t]+)/, name)
    if (name[1] == "") {
        fail($0)
        next
    }

    emit("void", name[1]);
}

/UCS_PROFILE_FUNC[\t ]*\(/ {
    match($0, /UCS_PROFILE_FUNC\([\t ]*([^, \t]+)/, type)
    match($0, /UCS_PROFILE_FUNC\([^,]+,[\t ]*([^, \t]+)/, name)
    if (type[1] == "" || name[1] == "") {
        fail($0)
        next
    }

    emit(type[1], name[1]);
}
