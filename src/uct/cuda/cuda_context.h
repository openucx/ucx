/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_CUDA_CONTEXT_H
#define UCT_CUDA_CONTEXT_H

#define UCT_CUDA_TL_NAME           "cuda"

/*
 * Helper function to list cuda resources
 */
ucs_status_t uct_cuda_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p);

#endif
