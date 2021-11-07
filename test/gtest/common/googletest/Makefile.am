#
# Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
# Copyright (C) NextSilicon Ltd. 2021.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_GTEST

noinst_LTLIBRARIES = libgtest.la

libgtest_la_CPPFLAGS = \
	$(BASE_CPPFLAGS) \
	-DTOP_SRCDIR=\"$(abs_top_srcdir)\" \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/test \
	-I$(top_builddir)/src \
	-I$(top_srcdir)/test/gtest \
	$(GTEST_CPPFLAGS) \
	$(OPENMP_CFLAGS)

libgtest_la_LDFLAGS  = $(GTEST_LDFLAGS) -no-install $(LDFLAGS_DYNAMIC_LIST_DATA)
libgtest_la_CFLAGS   = $(BASE_CFLAGS)
libgtest_la_CXXFLAGS = \
	$(BASE_CXXFLAGS) $(GTEST_CXXFLAGS) -std=c++11

libgtest_la_SOURCES = \
	gtest.cc \
	gtest-death-test.cc \
	gtest-filepath.cc \
	gtest-port.cc \
	gtest-printers.cc \
	gtest-test-part.cc \
	gtest-typed-test.cc

noinst_HEADERS = \
	gtest.h \
	gtest-death-test.h \
	gtest-internal-inl.h \
	gtest-matchers.h \
	gtest-message.h \
	gtest-param-test.h \
	gtest_pred_impl.h \
	gtest-printers.h \
	gtest_prod.h \
	gtest-spi.h \
	gtest-test-part.h \
	gtest-typed-test.h \
	internal/gtest-death-test-internal.h \
	internal/gtest-filepath.h \
	internal/gtest-internal.h \
	internal/gtest-param-util.h \
	internal/gtest-port.h \
	internal/gtest-string.h \
	internal/gtest-type-util.h \
	internal/gtest-port-arch.h \
	internal/custom/gtest-port.h \
	internal/custom/gtest-printers.h \
	internal/custom/gtest.h

endif
