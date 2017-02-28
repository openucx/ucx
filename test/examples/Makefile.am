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
	uct_hello_world.c

installcheck-local:
	@echo "INSTALLCHECK: Compiling examples with installed library"
	$(CC) -o uct_hello_world $(examplesdir)/uct_hello_world.c -luct -lucs -I$(includedir) -L$(libdir) -pedantic -Werror -Wl,-rpath,$(libdir)
	$(CC) -o ucp_hello_world $(examplesdir)/ucp_hello_world.c -lucp -lucs -I$(includedir) -L$(libdir) -pedantic -Werror -Wl,-rpath,$(libdir)
	$(RM) *.o uct_hello_world ucp_hello_world
