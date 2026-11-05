/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TYPE_QOS_LEVEL_H
#define UCS_TYPE_QOS_LEVEL_H


/**
 * @ingroup UCS_RESOURCE
 * @brief Quality of service level
 *
 * Specifies the relative priority of the endpoint's traffic. The level is
 * abstract: each transport maps it to whatever prioritization mechanism the
 * underlying fabric provides, and transports which do not support
 * prioritization ignore it.
 *
 * The levels are ordered from the lowest priority to the highest, so a
 * transport which distinguishes fewer levels than are defined here maps the
 * levels it cannot represent to the lowest priority it supports.
 */
typedef enum {
    /**
     * Lowest priority. Traffic of this endpoint yields to traffic of
     * endpoints at any other level.
     */
    UCS_QOS_LEVEL_LOW,

    /**
     * Intermediate priority.
     */
    UCS_QOS_LEVEL_MEDIUM,

    /**
     * Highest priority. Traffic of this endpoint is served before traffic of
     * endpoints at any other level.
     */
    UCS_QOS_LEVEL_HIGH,

    UCS_QOS_LEVEL_LAST
} ucs_qos_level_t;


#endif
