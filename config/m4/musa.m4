#
# Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_MUSA],[
    AS_IF([test "x$musa_checked" != "xyes"], [
        AC_ARG_WITH([musa],
                    [AS_HELP_STRING([--with-musa=(DIR)],
                                    [Enable MUSA support (default is guess).])],
                    [],
                    [with_musa=guess])

        musa_happy=no
        MUSA_CPPFLAGS=""
        MUSA_LDFLAGS=""
        MUSA_LIBS=""
        MUSA_ROOT=""

        AS_IF([test "x$with_musa" = "xno"], [
            AC_MSG_NOTICE([MUSA support was explicitly disabled])
        ], [
            AS_IF([test "x$with_musa" = "xyes" ||
                   test "x$with_musa" = "xguess"], [
                AS_IF([test -d /usr/local/musa],
                      [MUSA_ROOT=/usr/local/musa])
            ], [
                MUSA_ROOT="$with_musa"
            ])

            AS_IF([test -n "$MUSA_ROOT"], [
                MUSA_CPPFLAGS="-I$MUSA_ROOT/include"
                MUSA_LDFLAGS="-L$MUSA_ROOT/lib -L$MUSA_ROOT/lib64"
            ])
            MUSA_LIBS="-lmusart -lmusa"

            musa_happy=yes
            save_CPPFLAGS="$CPPFLAGS"
            save_LDFLAGS="$LDFLAGS"
            save_LIBS="$LIBS"

            CPPFLAGS="$MUSA_CPPFLAGS $CPPFLAGS"
            LDFLAGS="$MUSA_LDFLAGS $LDFLAGS"
            LIBS="$MUSA_LIBS $LIBS"

            AC_CHECK_HEADERS([musa_runtime_api.h],
                             [],
                             [musa_happy=no])
            AC_CHECK_HEADERS([musa.h],
                             [],
                             [musa_happy=no])

            AS_IF([test "x$musa_happy" = "xyes"], [
                AC_CHECK_LIB([musart], [musaGetDeviceCount],
                             [],
                             [musa_happy=no])
            ])
            AS_IF([test "x$musa_happy" = "xyes"], [
                AC_CHECK_LIB([musa], [muMemGetAddressRange],
                             [],
                             [musa_happy=no])
            ])
            AS_IF([test "x$musa_happy" = "xyes"], [
                # Mandatory symbols: referenced directly by the MUSA
                # components and their tests, so the build must fail early
                # when one of them is missing.
                AC_LINK_IFELSE(
                    [AC_LANG_PROGRAM(
                        [[#include <stddef.h>
                          #include <musa.h>
                          #include <musa_runtime_api.h>]],
                        [[int count = 0;
                          int device = 0;
                          void *address = 0;
                          MUdeviceptr base = 0;
                          size_t size = 0;
                          musaStream_t runtime_stream = 0;
                          musaEvent_t event = 0;
                          char pci_bus_id[64];
                          MUcontext context = 0;
                          unsigned memory_type = 0;
                          unsigned is_managed = 0;
                          musaGetDeviceCount(&count);
                          musaGetDevice(&device);
                          musaSetDevice(device);
                          musaGetErrorString(musaSuccess);
                          musaGetLastError();
                          musaPeekAtLastError();
                          musaDeviceGetPCIBusId(pci_bus_id,
                                                sizeof(pci_bus_id),
                                                device);
                          muPointerGetAttribute(
                              &memory_type,
                              MU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                              (MUdeviceptr)address);
                          muPointerGetAttribute(
                              &is_managed, MU_POINTER_ATTRIBUTE_IS_MANAGED,
                              (MUdeviceptr)address);
                          muPointerGetAttribute(
                              &device, MU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                              (MUdeviceptr)address);
                          muMemGetAddressRange(&base, &size,
                                               (MUdeviceptr)address);
                          if (muCtxGetCurrent(&context) ==
                              MUSA_ERROR_NOT_INITIALIZED) {
                              count = 0;
                          }
                          if (muMemHostRegister(
                                  address, size,
                                  MU_MEMHOSTREGISTER_PORTABLE) ==
                              MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
                              count = 0;
                          }
                          muMemHostUnregister(address);
                          musaMalloc(&address, size);
                          musaMallocHost(&address, size);
                          musaMallocManaged(&address, size, 0);
                          musaFreeHost(address);
                          musaFree(address);
                          musaStreamCreateWithFlags(&runtime_stream, 0);
                          musaEventCreateWithFlags(&event, 0);
                          musaMemcpyAsync(address, address, size,
                                          musaMemcpyDeviceToDevice,
                                          runtime_stream);
                          musaEventRecord(event, runtime_stream);
                          if (muEventQuery((MUevent)event) ==
                              MUSA_ERROR_NOT_READY) {
                              count = 0;
                          }
                          musaEventDestroy(event);
                          musaMemcpy(address, address, size,
                                     musaMemcpyDeviceToDevice);
                          musaMemset(address, 0, size);
                          musaDeviceSynchronize();
                          musaStreamSynchronize(runtime_stream);
                          musaStreamDestroy(runtime_stream);
                          return count < 0;]])],
                    [],
                    [musa_happy=no])
            ])
            AS_IF([test "x$musa_happy" = "xyes"], [
                # Optional non-async allocation family (driver _v2 alloc/free,
                # pitch and managed allocations): the memory hooks resolve
                # these symbols by name and skip the ones the runtime does not
                # provide, so an older MUSA SDK must not disable MUSA support
                # when they are missing. It is checked separately from the
                # async/pool family below, so that an SDK which provides the
                # plain driver _v2 APIs still gets the corresponding memory
                # hooks.
                AC_LINK_IFELSE(
                    [AC_LANG_PROGRAM(
                        [[#include <stddef.h>
                          #include <musa.h>
                          #include <musa_runtime_api.h>]],
                        [[void *address = 0;
                          size_t size = 0;
                          size_t pitch = 0;
                          MUdeviceptr base = 0;
                          musaMallocPitch(&address, &pitch, size, 1);
                          muMemAlloc_v2(&base, size);
                          muMemAllocPitch_v2(&base, &pitch, size, 1, 1);
                          muMemAllocManaged(&base, size, 0);
                          muMemFree_v2(base);
                          return 0;]])],
                    [AC_DEFINE([HAVE_MUSA_MEM_ALLOC_V2], [1],
                               [MUSA _v2/pitch/managed allocation APIs are available])],
                    [AC_MSG_NOTICE([MUSA _v2/pitch/managed allocation APIs not found; \
the corresponding memory hooks will be skipped])])
            ])
            AS_IF([test "x$musa_happy" = "xyes"], [
                # Optional async/pool allocation family: the memory hooks
                # resolve these symbols by name and skip the ones the runtime
                # does not provide, and the async allocation paths of the test
                # code are compiled only when the whole family is available,
                # so an older MUSA SDK must not disable MUSA support when they
                # are missing.
                AC_LINK_IFELSE(
                    [AC_LANG_PROGRAM(
                        [[#include <stddef.h>
                          #include <musa.h>
                          #include <musa_runtime_api.h>]],
                        [[void *address = 0;
                          size_t size = 0;
                          musaStream_t runtime_stream = 0;
                          musaMemPool_t runtime_pool = 0;
                          MUstream driver_stream = 0;
                          MUmemoryPool driver_pool = 0;
                          MUdeviceptr base = 0;
                          musaMallocAsync(&address, size, runtime_stream);
                          musaMallocFromPoolAsync(&address, size,
                                                  runtime_pool,
                                                  runtime_stream);
                          musaFreeAsync(address, runtime_stream);
                          muMemAllocAsync(&base, size, driver_stream);
                          muMemAllocFromPoolAsync(&base, size, driver_pool,
                                                  driver_stream);
                          muMemFreeAsync(base, driver_stream);
                          return 0;]])],
                    [AC_DEFINE([HAVE_MUSA_MEM_POOL_ALLOC], [1],
                               [MUSA async/pool allocation APIs are available])],
                    [AC_MSG_NOTICE([MUSA async/pool allocation APIs not found; \
the corresponding memory hooks will be skipped])])
            ])
            AS_IF([test "x$musa_happy" = "xyes"], [
                AC_CHECK_DECLS(
                    [MU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD],
                    [AC_LINK_IFELSE(
                        [AC_LANG_PROGRAM(
                            [[#include <musa.h>]],
                            [[int fd = -1;
                              int supported = 0;
                              muDeviceGetAttribute(
                                  &supported,
                                  MU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                                  0);
                              return (int)muMemGetHandleForAddressRange(
                                  &fd, 0, 0,
                                  MU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                  0);]])],
                        [AC_DEFINE(
                            [HAVE_MUSA_DMABUF], [1],
                            [MUSA supports DMA-BUF export])],
                        [])],
                    [],
                    [[#include <musa.h>]])
            ])

            CPPFLAGS="$save_CPPFLAGS"
            LDFLAGS="$save_LDFLAGS"
            LIBS="$save_LIBS"

            AS_IF([test "x$musa_happy" = "xyes"], [
                AC_DEFINE([HAVE_MUSA], [1], [Enable MUSA support])
                AC_SUBST([MUSA_CPPFLAGS])
                AC_SUBST([MUSA_LDFLAGS])
                AC_SUBST([MUSA_LIBS])
                AC_SUBST([MUSA_ROOT])
            ], [
                AS_IF([test "x$with_musa" != "xguess"], [
                    AC_MSG_ERROR([MUSA support was requested but MUSA headers and libraries were not found])
                ], [
                    AC_MSG_WARN([MUSA headers and libraries not found])
                ])
            ])
        ])

        musa_checked=yes
        AM_CONDITIONAL([HAVE_MUSA], [test "x$musa_happy" = "xyes"])
    ])
])
