#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

examplesdir = $(pkgdatadir)/examples
dist_examples_DATA = \
	ucx_hello_world.h \
	ucp_hello_world.c \
	uct_hello_world.c \
	ucx_profiling.c \
	ucp_client_server.c

EXAMPLE_CC_FLAGS = -lucs -lucm -I$(includedir) -L$(libdir) -Wall -Werror -Wl,-rpath,$(libdir)

installcheck-local:
	@echo "INSTALLCHECK: Compiling examples with installed library"
	$(CC) -o uct_hello_world   $(examplesdir)/uct_hello_world.c   -luct $(EXAMPLE_CC_FLAGS) -pedantic
	$(CC) -o ucp_hello_world   $(examplesdir)/ucp_hello_world.c   -lucp $(EXAMPLE_CC_FLAGS) -pedantic
	$(CC) -o ucp_client_server $(examplesdir)/ucp_client_server.c -lucp $(EXAMPLE_CC_FLAGS) -pedantic
	$(CC) -o ucx_profiling     $(examplesdir)/ucx_profiling.c     -lucs $(EXAMPLE_CC_FLAGS) -Wall -lm
	$(RM) *.o uct_hello_world ucp_hello_world ucp_client_server ucx_profiling

if HAVE_EXAMPLES

EXAMPLE_INCLUDE_FLAGS = \
	-I$(abs_top_srcdir)/src \
	-I$(abs_top_builddir)/src

EXAMPLE_CPPFLAGS = \
	$(EXAMPLE_INCLUDE_FLAGS)

noinst_PROGRAMS = \
	ucp_hello_world \
	uct_hello_world \
	ucx_profiling \
	ucp_client_server

ucp_hello_world_SOURCES  = ucp_hello_world.c
ucp_hello_world_CFLAGS   = $(BASE_CFLAGS)
ucp_hello_world_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
ucp_hello_world_LDADD    = $(top_builddir)/src/ucp/libucp.la

uct_hello_world_SOURCES  = uct_hello_world.c
uct_hello_world_CFLAGS   = $(BASE_CFLAGS)
uct_hello_world_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
uct_hello_world_LDADD    = $(top_builddir)/src/uct/libuct.la

ucp_client_server_SOURCES  = ucp_client_server.c
ucp_client_server_CFLAGS   = $(BASE_CFLAGS)
ucp_client_server_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
ucp_client_server_LDADD    = $(top_builddir)/src/ucp/libucp.la

ucx_profiling_SOURCES  = ucx_profiling.c
ucx_profiling_CFLAGS   = $(BASE_CFLAGS)
ucx_profiling_CPPFLAGS = $(EXAMPLE_CPPFLAGS)
ucx_profiling_LDADD    = $(top_builddir)/src/ucs/libucs.la

endif
