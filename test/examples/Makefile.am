#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

examplesdir = $(pkgdatadir)/examples
dist_examples_DATA = \
	ucp_hello_world.c \
	uct_hello_world.c \
	active_message.c 

installcheck-local:
	@echo "INSTALLCHECK: Compiling examples with installed library"
	$(CC) -o uct_hello_world $(examplesdir)/uct_hello_world.c -luct -I$(includedir) -L$(libdir)
	$(CC) -o ucp_hello_world $(examplesdir)/ucp_hello_world.c -lucp -I$(includedir) -L$(libdir)
if HAVE_MPI
	$(MPICC) -o active_message $(examplesdir)/active_message.c -luct -lucs -I$(includedir) -L$(libdir)
endif
	$(RM) *.o uct_hello_world ucp_hello_world active_message
