/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "libstats.h"

#include <ucs/debug/log.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/sys/compiler.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <errno.h>
#include <inttypes.h>


/* Class table */
#define UCS_STATS_CLS_HASH_SIZE      127
#define UCS_STATS_CLSID_HASH(a)      (   (uintptr_t)((a)->cls)   )
#define UCS_STATS_CLSID_CMP(a, b)    (  ((long)((a)->cls)) - ((long)((b)->cls))  )
#define UCS_STATS_CLSID_SENTINEL     UINT8_MAX

/* Encode counter size */
#define UCS_STATS_BITS_PER_COUNTER   2
#define UCS_STATS_COUNTER_ZERO       0
#define UCS_STATS_COUNTER_U16        1
#define UCS_STATS_COUNTER_U32        2
#define UCS_STATS_COUNTER_U64        3


/* Compression mode */
#define UCS_STATS_COMPRESSION_NONE   0
#define UCS_STATS_COMPRESSION_BZIP2  1


/* Statistics data header */
typedef struct ucs_stats_data_header {
    uint32_t   version;
    uint32_t   reserved;
    uint32_t   compression;
    uint32_t   num_classes;
} ucs_stats_data_header_t;


/* Class id record */
typedef struct ucs_stats_clsid         ucs_stats_clsid_t;
struct ucs_stats_clsid {
    uint8_t              clsid;
    ucs_stats_class_t    *cls;
    ucs_stats_clsid_t    *next;
};


/* Save pointer to class table near the root node */
typedef struct ucs_stats_root_storage {
    ucs_stats_class_t **classes;
    unsigned          num_classes;
    ucs_stats_node_t  node;
} ucs_stats_root_storage_t;


SGLIB_DEFINE_LIST_PROTOTYPES(ucs_stats_clsid_t, UCS_STATS_CLSID_CMP, next)
SGLIB_DEFINE_LIST_FUNCTIONS(ucs_stats_clsid_t, UCS_STATS_CLSID_CMP, next)
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(ucs_stats_clsid_t, UCS_STATS_CLS_HASH_SIZE, UCS_STATS_CLSID_HASH)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(ucs_stats_clsid_t, UCS_STATS_CLS_HASH_SIZE, UCS_STATS_CLSID_HASH)

#define FREAD(_buf, _size, _stream) \
    { \
        size_t nread = fread(_buf, 1, _size, _stream); \
        assert(nread == _size); \
    }

#define FWRITE(_buf, _size, _stream) \
    { \
        size_t nwrite = fwrite(_buf, 1, _size, _stream); \
        assert(nwrite == _size); \
    }

#define FREAD_ONE(_ptr, _stream) \
    FREAD(_ptr, sizeof(*(_ptr)), _stream)

#define FWRITE_ONE(_ptr, _stream) \
    FWRITE(_ptr, sizeof(*(_ptr)), _stream)


static unsigned ucs_stats_get_all_classes_recurs(ucs_stats_node_t *node,
                                                 ucs_stats_children_sel_t sel,
                                                 ucs_stats_clsid_t **cls_hash)
{
    ucs_stats_clsid_t *elem, search;
    ucs_stats_node_t *child;
    unsigned count;

    search.cls = node->cls;
    if (!sglib_hashed_ucs_stats_clsid_t_find_member(cls_hash, &search)) {
        elem = malloc(sizeof *elem);
        elem->cls = node->cls;
        sglib_hashed_ucs_stats_clsid_t_add(cls_hash, elem);
        count = 1;
    } else {
        count = 0;
    }

    ucs_list_for_each(child, &node->children[sel], list) {
        count += ucs_stats_get_all_classes_recurs(child, sel, cls_hash);
    }

    return count;
}

static char * ucs_stats_read_str(FILE *stream)
{
    uint8_t tmp;
    char *str;

    FREAD_ONE(&tmp, stream);
    /* coverity[tainted_data] */
    str = malloc(tmp + 1);
    FREAD(str, tmp, stream);
    str[tmp] = '\0';
    return str;
}

static void ucs_stats_write_str(const char *str, FILE *stream)
{
    uint8_t tmp = strlen(str);

    FWRITE_ONE(&tmp, stream);
    FWRITE(str, tmp, stream);
}

static void ucs_stats_read_counters(ucs_stats_counter_t *counters,
                                    unsigned num_counters,
                                    FILE *stream)
{
    const unsigned counters_per_byte = 8 / UCS_STATS_BITS_PER_COUNTER;
    uint16_t value16;
    uint32_t value32;
    uint64_t value64;
    uint8_t *counter_desc, v;
    size_t counter_desc_size;
    unsigned i;

    counter_desc_size = ((num_counters + counters_per_byte - 1) / counters_per_byte);
    counter_desc = ucs_alloca(counter_desc_size);
    FREAD(counter_desc, counter_desc_size, stream);

    for (i = 0; i < num_counters; ++i) {
        v = (counter_desc[i / counters_per_byte] >>
                        ((i % counters_per_byte) * UCS_STATS_BITS_PER_COUNTER)) & 0x3;
        switch (v) {
        case UCS_STATS_COUNTER_ZERO:
            counters[i] = 0;
            break;
        case UCS_STATS_COUNTER_U16:
            FREAD_ONE(&value16, stream);
            counters[i] = value16;
            break;
        case UCS_STATS_COUNTER_U32:
            FREAD_ONE(&value32, stream);
            counters[i] = value32;
            break;
        case UCS_STATS_COUNTER_U64:
            FREAD_ONE(&value64, stream);
            counters[i] = value64;
            break;
        }
    }
}

static void ucs_stats_write_counters(ucs_stats_counter_t *counters,
                                     unsigned num_counters,
                                     FILE *stream)
{
    const unsigned counters_per_byte = 8 / UCS_STATS_BITS_PER_COUNTER;
    ucs_stats_counter_t value;
    uint8_t *counter_desc, v;
    void *counter_data, *pos;
    size_t counter_desc_size;
    unsigned i;

    UCS_STATIC_ASSERT((8 % UCS_STATS_BITS_PER_COUNTER) == 0);
    counter_desc_size = ((num_counters + counters_per_byte - 1) / counters_per_byte);
    counter_desc = ucs_alloca(counter_desc_size);
    counter_data = ucs_alloca(num_counters * sizeof(ucs_stats_counter_t));

    memset(counter_desc, 0, counter_desc_size);
    pos = counter_data;

    /*
     * First, we have an array with 2 bits per counter describing its size:
     *  (0 - empty, 1 - 16bit, 2 - 32bit, 3 - 64bit)
     * Then, an array of all counters, each one occupying the size listed before.
     */
    for (i = 0; i < num_counters; ++i) {
        value = counters[i];
        if (value == 0) {
            v = UCS_STATS_COUNTER_ZERO;
        } else if (value <= USHRT_MAX) {
            v = UCS_STATS_COUNTER_U16;
            *(uint16_t*)(pos) = value;
            pos += sizeof(uint16_t);
        } else if (value <= UINT_MAX) {
            v = UCS_STATS_COUNTER_U32;
            *(uint32_t*)(pos) = value;
            pos += sizeof(uint32_t);
        } else {
            v = UCS_STATS_COUNTER_U64;
            *(uint64_t*)(pos) = value;
            pos += sizeof(uint64_t);
        }
        counter_desc[i / counters_per_byte] |=
                        v << ((i % counters_per_byte) * UCS_STATS_BITS_PER_COUNTER);
    }

    FWRITE(counter_desc, counter_desc_size,  stream);
    FWRITE(counter_data, pos - counter_data, stream);
}

static void
ucs_stats_serialize_binary_recurs(FILE *stream, ucs_stats_node_t *node,
                                  ucs_stats_children_sel_t sel,
                                  ucs_stats_clsid_t **cls_hash)
{
    ucs_stats_class_t *cls = node->cls;
    ucs_stats_clsid_t *elem, search;
    ucs_stats_node_t *child;
    uint8_t sentinel;

    /* Search the class */
    search.cls = cls;
    elem = sglib_hashed_ucs_stats_clsid_t_find_member(cls_hash, &search);
    assert(elem != NULL);

    /* Write class ID */
    FWRITE_ONE(&elem->clsid, stream);

    /* Name */
    ucs_stats_write_str(node->name, stream);

    /* Counters */
    ucs_stats_write_counters(node->counters, cls->num_counters, stream);

    /* Children */
    ucs_list_for_each(child, &node->children[sel], list) {
        ucs_stats_serialize_binary_recurs(stream, child, sel, cls_hash);
    }

    /* Write sentinel which is not valid class id to mark end of children */
    sentinel = UCS_STATS_CLSID_SENTINEL;
    FWRITE_ONE(&sentinel, stream);
}

static ucs_status_t
ucs_stats_serialize_binary(FILE *stream, ucs_stats_node_t *root,
                           ucs_stats_children_sel_t sel)
{
    ucs_stats_clsid_t* cls_hash[UCS_STATS_CLS_HASH_SIZE];
    struct sglib_hashed_ucs_stats_clsid_t_iterator it;
    ucs_stats_class_t *cls;
    ucs_stats_clsid_t *elem;
    ucs_stats_data_header_t hdr;
    unsigned index, counter;

    sglib_hashed_ucs_stats_clsid_t_init(cls_hash);

    /* Write header */
    hdr.version     = 1;
    hdr.compression = UCS_STATS_COMPRESSION_NONE;
    hdr.reserved    = 0;
    hdr.num_classes = ucs_stats_get_all_classes_recurs(root, sel, cls_hash);
    assert(hdr.num_classes < UINT8_MAX);
    FWRITE_ONE(&hdr, stream);

    /* Write stats node classes */
    index = 0;
    for (elem = sglib_hashed_ucs_stats_clsid_t_it_init(&it, cls_hash);
         elem != NULL; elem = sglib_hashed_ucs_stats_clsid_t_it_next(&it))
    {
        cls = elem->cls;
        ucs_stats_write_str(cls->name, stream);
        FWRITE_ONE(&cls->num_counters, stream);
        for (counter = 0; counter < cls->num_counters; ++counter) {
            ucs_stats_write_str(cls->counter_names[counter], stream);
        }
        elem->clsid = index++;
    }

    assert(index == hdr.num_classes);

    /* Write stats nodes */
    ucs_stats_serialize_binary_recurs(stream, root, sel, cls_hash);

    /* Free classes */
    for (elem = sglib_hashed_ucs_stats_clsid_t_it_init(&it, cls_hash);
         elem != NULL; elem = sglib_hashed_ucs_stats_clsid_t_it_next(&it))
    {
        free(elem);
    }

    return UCS_OK;
}

static ucs_status_t
ucs_stats_serialize_text_recurs_filtered(FILE *stream,
                                         ucs_stats_filter_node_t *filter_node,
                                         unsigned indent)
{
    ucs_stats_filter_node_t *filter_child;
    ucs_stats_node_t *node;
    unsigned i;
    int is_sum = ucs_global_opts.stats_format == UCS_STATS_SUMMARY;
    char *nl = is_sum ? "" : "\n";
    char *space =  is_sum ? "" : " ";
    char *left_b = is_sum ? "{" : "";
    char *rigth_b = is_sum ? "} " : "";

    if (!filter_node->ref_count) {
        return UCS_OK;
    }

    if (ucs_list_is_empty(&filter_node->type_list_head)) {
        ucs_error("no node is associated with node filter");
        return UCS_OK;
    }

    node = ucs_list_head(&filter_node->type_list_head,
                         ucs_stats_node_t,
                         type_list);
    if (filter_node->type_list_len > 1) {
        fprintf(stream, "%*s%s*:%s", UCS_STATS_INDENT(is_sum, indent),
                node->cls->name, nl);
    } else {
        if (ucs_global_opts.stats_format == UCS_STATS_SUMMARY) {
            fprintf(stream, "%*s%s:%s",
                    UCS_STATS_INDENT(is_sum, indent),
                    strlen(node->cls->name) ? node->cls->name : node->name, nl);

        } else {
            fprintf(stream, "%*s"UCS_STATS_NODE_FMT":%s",
                    UCS_STATS_INDENT(is_sum, indent),
                    UCS_STATS_NODE_ARG(node), nl);
        }
    }

    /* Root shouldn't be with brackets.*/
    if (filter_node->parent) {
        fputs(left_b, stream);
    }

    for (i = 0; (i < node->cls->num_counters) && (i < 64); ++i) {
        ucs_stats_counter_t counters_acc = 0;
        if (filter_node->counters_bitmask & UCS_BIT(i)) {
            ucs_stats_node_t * temp_node;
            ucs_list_for_each(temp_node, &filter_node->type_list_head, type_list) {
                counters_acc += temp_node->counters[i];
            }

            fprintf(stream, "%*s%s:%s%"PRIu64"%s",
                    UCS_STATS_INDENT(is_sum, indent + 1),
                    node->cls->counter_names[i],
                    space, counters_acc, nl);

            /* Don't print space on last counter */
            if (UCS_STATS_IS_LAST_COUNTER(filter_node->counters_bitmask, i) &&
                is_sum) {
                fputs(" ", stream);
            }
        }
    }

    ucs_list_for_each(filter_child, &filter_node->children, list) {
        ucs_stats_serialize_text_recurs_filtered(stream, filter_child,
                                                 indent + 1);
    }

    if (filter_node->parent) {
        /* Root shouldn't be with parent brackets.*/
        fputs(rigth_b, stream);
    } else {
        /* End report with new line.*/
        fputs("\n", stream);
    }

    return UCS_OK;
}

ucs_status_t ucs_stats_serialize(FILE *stream, ucs_stats_node_t *root, int options)
{
    ucs_stats_children_sel_t sel =
                    (options & UCS_STATS_SERIALIZE_INACTVIVE) ?
                                    UCS_STATS_INACTIVE_CHILDREN :
                                    UCS_STATS_ACTIVE_CHILDREN;

    if (options & UCS_STATS_SERIALIZE_BINARY) {
        return ucs_stats_serialize_binary(stream, root, sel);
    } else {
        return ucs_stats_serialize_text_recurs_filtered(stream,
                                                        root->filter_node,
                                                        0);
    }
}

static ucs_status_t
ucs_stats_deserialize_recurs(FILE *stream, ucs_stats_class_t **classes,
                             unsigned num_classes, size_t headroom,
                             ucs_stats_node_t **p_root)
{
    ucs_stats_node_t *node, *child;
    ucs_stats_class_t *cls;
    uint8_t clsid, namelen;
    ucs_status_t status;
    void *ptr;

    if (headroom >= UINT_MAX) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (feof(stream)) {
        ucs_error("Error parsing statistics - premature end of stream");
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    FREAD_ONE(&clsid, stream);
    if (clsid == UCS_STATS_CLSID_SENTINEL) {
        return UCS_ERR_NO_MESSAGE; /* Sentinel */
    }

    if (clsid >= num_classes) {
        ucs_error("Error parsing statistics - class id out of range");
        return UCS_ERR_OUT_OF_RANGE;
    }

    FREAD_ONE(&namelen, stream);
    if (namelen >= UCS_STAT_NAME_MAX) {
        ucs_error("Error parsing statistics - node name too long");
        return UCS_ERR_OUT_OF_RANGE; /* Name too long */
    }

    cls = classes[clsid];
    ptr = malloc(headroom + sizeof *node + sizeof(ucs_stats_counter_t) * cls->num_counters);
    if (ptr == NULL) {
        ucs_error("Failed to allocate statistics counters (headroom %zu, %u counters)",
                  headroom, cls->num_counters);
        return UCS_ERR_NO_MEMORY;
    }

    node = ptr + headroom;

    node->cls = cls;
    FREAD(node->name, namelen, stream);
    node->name[namelen] = '\0';
    ucs_list_head_init(&node->children[UCS_STATS_INACTIVE_CHILDREN]);
    ucs_list_head_init(&node->children[UCS_STATS_ACTIVE_CHILDREN]);

    /* Read counters */
    ucs_stats_read_counters(node->counters, cls->num_counters, stream);

    /* Read children */
    do {
        status = ucs_stats_deserialize_recurs(stream, classes, num_classes, 0,
                                              &child);
        if (status == UCS_OK) {
            ucs_list_add_tail(&node->children[UCS_STATS_ACTIVE_CHILDREN], &child->list);
        } else if (status == UCS_ERR_NO_MESSAGE) {
            break; /* Sentinel */
        } else {
            ucs_error("ucs_stats_deserialize_recurs returned %s", ucs_status_string(status));
            free(ptr); /* Error TODO free previous children */
            return status;
        }
    } while (1);

    *p_root = node;
    return UCS_OK;
}

static void ucs_stats_free_classes(ucs_stats_class_t **classes, unsigned num_classes)
{
    unsigned i, j;

    for (i = 0; i < num_classes; ++i) {
        free((char*)classes[i]->name);
        for (j = 0; j < classes[i]->num_counters; ++j) {
            free((char*)classes[i]->counter_names[j]);
        }
        free(classes[i]);
    }
    free(classes);
}

ucs_status_t ucs_stats_deserialize(FILE *stream, ucs_stats_node_t **p_root)
{
    ucs_stats_data_header_t hdr;
    ucs_stats_root_storage_t *s;
    ucs_stats_class_t **classes, *cls;
    unsigned i, j, num_counters;
    ucs_status_t status;
    size_t nread;
    char *name;

    nread = fread(&hdr, 1, sizeof(hdr), stream);
    if (nread == 0) {
        status = UCS_ERR_NO_ELEM;
        goto err;
    }

    if (hdr.version != 1) {
        ucs_error("invalid file version");
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    if (!(hdr.num_classes < UINT8_MAX)) {
        ucs_error("invalid num classes");
        status = UCS_ERR_OUT_OF_RANGE;
        goto err;
    }

    /* Read classes */
    classes = malloc(hdr.num_classes * sizeof(*classes));
    for (i = 0; i < hdr.num_classes; ++i) {
        name = ucs_stats_read_str(stream);
        FREAD_ONE(&num_counters, stream);

        /* coverity[tainted_data] */
        cls = malloc(sizeof *cls + num_counters * sizeof(cls->counter_names[0]));
        cls->name = name;
        cls->num_counters = num_counters;

        /* coverity[tainted_data] */
        for (j = 0; j < cls->num_counters; ++j) {
            cls->counter_names[j] = ucs_stats_read_str(stream);
        }
        classes[i] = cls;

    }

    /* Read nodes */
    status = ucs_stats_deserialize_recurs(stream, classes, hdr.num_classes,
                                         sizeof(ucs_stats_root_storage_t) - sizeof(ucs_stats_node_t),
                                         p_root);
    if (status != UCS_OK) {
        if (status == UCS_ERR_NO_MESSAGE) {
            ucs_error("Error parsing statistics - misplaced sentinel");
        }
        goto err_free;
    }

    s = ucs_container_of(*p_root, ucs_stats_root_storage_t, node);
    s->num_classes = hdr.num_classes;
    s->classes     = classes;
    return UCS_OK;

err_free:
    ucs_stats_free_classes(classes, hdr.num_classes);
err:
    return status;
}

static void ucs_stats_free_recurs(ucs_stats_node_t *node)
{
    ucs_stats_node_t *child, *tmp;

    ucs_list_for_each_safe(child, tmp, &node->children[UCS_STATS_ACTIVE_CHILDREN], list) {
        ucs_stats_free_recurs(child);
        free(child);
    }
    ucs_list_for_each_safe(child, tmp, &node->children[UCS_STATS_INACTIVE_CHILDREN], list) {
        ucs_stats_free_recurs(child);
        free(child);
    }
}

void ucs_stats_free(ucs_stats_node_t *root)
{
    ucs_stats_root_storage_t *s;

    s = ucs_container_of(root, ucs_stats_root_storage_t, node);
    ucs_stats_free_recurs(&s->node);
    ucs_stats_free_classes(s->classes, s->num_classes);
    free(s);
}

