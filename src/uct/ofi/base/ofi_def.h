/**
* Copyright (C) UT-Battelle, LLC. 2022. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_OFI_DEF_H
#define UCT_OFI_DEF_H

#define UCT_OFI_MD_NAME "ofi"
#define UCT_OFI_EPS_PER_AV 256

#define UCT_OFI_CHECK_ERROR(_x, _error_str, _ret) do { \
        if (_x) {                                      \
            ucs_error("UCT OFI error: '%s' ofi error: '%s'", _error_str, fi_strerror(_x)); \
            return (_ret);}                           \
 } while(0)

#endif
