#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

examplesdir = $(pkgdatadir)/examples
dist_examples_DATA = \
	uct_hello_world.c

installcheck-local:
	@echo "INSTALLCHECK: Compiling examples with installed library"
	$(CC) $(examplesdir)/uct_hello_world.c -luct -I$(includedir) -L$(libdir)
	$(RM) a.out

