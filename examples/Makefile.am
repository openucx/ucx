#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

examplesdir = $(pkgdatadir)/examples
dist_examples_DATA =   \
	hello_world_util.h \
	ucp_hello_world.c  \
	uct_hello_world.c  \
	ucp_client_server.c

if HAVE_CUDA
EXAMPLE_CUDA_LDFLAGS = $(CUDA_LDFLAGS)
# cuda.h couldn't be compiled with -pedantic flag
EXAMPLE_CUDA_CFLAGS =
EXAMPLE_CUDA_CPPFLAGS = $(CUDA_CPPFLAGS) -DHAVE_CUDA
else
EXAMPLE_CUDA_LDFLAGS =
EXAMPLE_CUDA_CFLAGS = $(CFLAGS_PEDANTIC)
EXAMPLE_CUDA_CPPFLAGS =
endif

EXAMPLE_CCLD_FLAGS = -lucs -I$(includedir) -L$(libdir) -Wall -Werror -Wl,-rpath,$(libdir) \
                     $(EXAMPLE_CUDA_LDFLAGS) $(EXAMPLE_CUDA_CPPFLAGS)

installcheck-local:
	@echo "INSTALLCHECK: Compiling examples with installed library"
	$(CC) -o uct_hello_world   $(examplesdir)/uct_hello_world.c   -luct $(EXAMPLE_CCLD_FLAGS)
	$(CC) -o ucp_hello_world   $(examplesdir)/ucp_hello_world.c   -lucp $(EXAMPLE_CCLD_FLAGS)
	$(CC) -o ucp_client_server $(examplesdir)/ucp_client_server.c -lucp $(EXAMPLE_CCLD_FLAGS)
	$(RM) *.o uct_hello_world ucp_hello_world ucp_client_server

if HAVE_EXAMPLES

bin_PROGRAMS = \
	ucp_hello_world \
	uct_hello_world \
	ucp_client_server

ucp_hello_world_SOURCES  = ucp_hello_world.c
ucp_hello_world_CFLAGS   = $(BASE_CFLAGS) $(EXAMPLE_CUDA_CFLAGS)
ucp_hello_world_CPPFLAGS = $(BASE_CPPFLAGS) $(EXAMPLE_CUDA_CPPFLAGS)
ucp_hello_world_LDADD    = $(top_builddir)/src/ucs/libucs.la \
                           $(top_builddir)/src/ucp/libucp.la \
                           $(EXAMPLE_CUDA_LDFLAGS)

uct_hello_world_SOURCES  = uct_hello_world.c
uct_hello_world_CFLAGS   = $(BASE_CFLAGS) $(EXAMPLE_CUDA_CFLAGS)
uct_hello_world_CPPFLAGS = $(BASE_CPPFLAGS) $(EXAMPLE_CUDA_CPPFLAGS)
uct_hello_world_LDADD    = $(top_builddir)/src/ucs/libucs.la \
                           $(top_builddir)/src/uct/libuct.la \
                           $(EXAMPLE_CUDA_LDFLAGS)

ucp_client_server_SOURCES  = ucp_client_server.c
ucp_client_server_CFLAGS   = $(BASE_CFLAGS) $(CFLAGS_PEDANTIC)
ucp_client_server_CPPFLAGS = $(BASE_CPPFLAGS)
ucp_client_server_LDADD    = $(top_builddir)/src/ucs/libucs.la \
                             $(top_builddir)/src/ucp/libucp.la

endif
