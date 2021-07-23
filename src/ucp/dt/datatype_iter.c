/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "datatype_iter.inl"


void ucp_datatype_iter_str(const ucp_datatype_iter_t *dt_iter,
                           ucs_string_buffer_t *strb)
{
    size_t iov_index, offset;
    const ucp_dt_iov_t *iov;
    const char *sysdev_name;
    char buffer[32];

    if (dt_iter->mem_info.type != UCS_MEMORY_TYPE_HOST) {
        ucs_string_buffer_appendf(
                strb, "%s ", ucs_memory_type_names[dt_iter->mem_info.type]);
    }

    if (dt_iter->mem_info.sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        sysdev_name = ucs_topo_sys_device_bdf_name(dt_iter->mem_info.sys_dev,
                                                   buffer, sizeof(buffer));
        ucs_string_buffer_appendf(strb, "%s ", sysdev_name);
    }

    ucs_string_buffer_appendf(strb, "%zu/%zu %s", dt_iter->offset,
                              dt_iter->length,
                              ucp_datatype_class_names[dt_iter->dt_class]);

    switch (dt_iter->dt_class) {
    case UCP_DATATYPE_CONTIG:
        ucs_string_buffer_appendf(strb, " buffer:%p",
                                  dt_iter->type.contig.buffer);
        break;
    case UCP_DATATYPE_IOV:
        iov_index = 0;
        offset    = 0;
        while (offset < dt_iter->length) {
            iov = &dt_iter->type.iov.iov[iov_index];
            ucs_string_buffer_appendf(strb, " [%zu]", iov_index);
            if (iov_index == dt_iter->type.iov.iov_index) {
                ucs_string_buffer_appendf(strb, " *{%p,%zu/%zu}", iov->buffer,
                                          dt_iter->type.iov.iov_offset,
                                          iov->length);
            } else {
                ucs_string_buffer_appendf(strb, " {%p, %zu}", iov->buffer,
                                          iov->length);
            }
            offset += iov->length;
            ++iov_index;
        }
        break;
    case UCP_DATATYPE_GENERIC:
        ucs_string_buffer_appendf(strb, " dt_gen:%p state:%p",
                                  dt_iter->type.generic.dt_gen,
                                  dt_iter->type.generic.state);

        break;
    default:
        break;
    }
}