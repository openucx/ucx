#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

MPICC = /opt/ompi/bin/mpicc

examplesdir = $(pkgdatadir)/examples
dist_examples_DATA = \
	uct_hello_world.c \
	active_message.c 

installcheck-local:
	@echo "INSTALLCHECK: Compiling examples with installed library"
	$(MPICC) -o uct_hello_world $(examplesdir)/uct_hello_world.c -luct -lucs -I$(includedir) -L$(libdir)
	$(MPICC) -o active_message $(examplesdir)/active_message.c -luct -lucs -I$(includedir) -L$(libdir)
	$(RM) *.o uct_hello_world active_message 
