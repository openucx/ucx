#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

emulatedir = $(pkgdatadir)/examples/emulate-distribute-storage/
dist_emulate_DATA = \
        ucp_tag_server_read.c \
        ucp_tag_client_read.c

if HAVE_CUDA
EXAMPLE_CUDA_CFLAGS =
EXAMPLE_CUDA_CPPFLAGS = $(CUDA_CPPFLAGS) -DHAVE_CUDA
else
EXAMPLE_CUDA_CFLAGS = $(CFLAGS_PEDANTIC)
EXAMPLE_CUDA_CPPFLAGS =
endif

EXAMPLE_CCLD_FLAGS = -lucs -I$(includedir) -L$(libdir) -Wall -Werror -Wl,-rpath,$(libdir) \
                     $(EXAMPLE_CUDA_LDFLAGS) $(EXAMPLE_CUDA_CPPFLAGS)

installcheck-local:
        @echo "INSTALLCHECK: Compiling examples with installed library"
        $(CC) -o ucp_tag_server_read $(emulatedir)/ucp_tag_server_read.c -lucp $(EXAMPLE_CCLD_FLAGS)
        $(CC) -o ucp_tag_server_read $(emulatedir)/ucp_tag_client_read.c -lucp $(EXAMPLE_CCLD_FLAGS)
        $(RM) *.o ucp_tag_server_read ucp_tag_client_read

if HAVE_EXAMPLES

noinst_PROGRAMS = \
        ucp_tag_server_read \
        ucp_tag_client_read

ucp_tag_server_read_SOURCES  = ucp_tag_server_read.c
ucp_tag_server_read_CFLAGS   = $(BASE_CFLAGS) $(CFLAGS_PEDANTIC)
ucp_tag_server_read_CPPFLAGS = $(BASE_CPPFLAGS)
ucp_tag_server_read_LDADD    = $(top_builddir)/src/ucs/libucs.la \
                             $(top_builddir)/src/ucp/libucp.la

ucp_tag_client_read_SOURCES  = ucp_tag_client_read.c
ucp_tag_client_read_CFLAGS   = $(BASE_CFLAGS) $(CFLAGS_PEDANTIC)
ucp_tag_client_read_CPPFLAGS = $(BASE_CPPFLAGS)
ucp_tag_client_read_LDADD    = $(top_builddir)/src/ucs/libucs.la \
                             $(top_builddir)/src/ucp/libucp.la

endif
