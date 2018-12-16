/*
* Copyright (C) Huawei Technologies Co., Ltd. 2018.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/


/**
 * Construct a UCG version identifier from major and minor version numbers.
 */
#define UCG_VERSION(_major, _minor) \
	(((_major) << UCG_VERSION_MAJOR_SHIFT) | \
	 ((_minor) << UCG_VERSION_MINOR_SHIFT))
#define UCG_VERSION_MAJOR_SHIFT    24
#define UCG_VERSION_MINOR_SHIFT    16


/**
 * UCG API version is 1.6
 */
#define UCG_API_MAJOR    1
#define UCG_API_MINOR    6
#define UCG_API_VERSION  UCG_VERSION(1, 6)
