#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
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
