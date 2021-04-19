AS_IF([test "x$with_efa_dv" != xno],
     [AC_CHECK_HEADER([infiniband/efadv.h], [], [with_efa_dv=no])])

AS_IF([test "x$with_efa_dv" != xno],
     [AC_CHECK_LIB([efa], [efadv_query_device],
                   [AC_SUBST(LIB_EFA, [-lefa])
                    AC_DEFINE([HAVE_EFA_DV], [1], [EFA device support])],
                   [with_efa_dv=no], [-libverbs])])

AS_IF([test "x$with_efa_dv" != xno],
     [AC_CHECK_DECLS([EFADV_DEVICE_ATTR_CAPS_RDMA_READ],
                     [AC_DEFINE([HAVE_DECL_EFA_DV_RDMA_READ], [1], [HAVE EFA device with RDMA READ support])],
                     [], [[#include <infiniband/efadv.h>]])])

