
LIBMOPSY = libmopsy.so
TEST     = test/test
CC       = gcc
RM       = rm -f
CPPFLAGS = -Isrc
CFLAGS   = -O3 -g

LIBMOPSY_HEADERS  = \
	src/mopsy/tl/base/tl_base.h \
	src/mopsy/tl/base/tl_def.h \
	src/mopsy/tl/base/tl.h \
	src/services/debug/log.h \
	src/services/sys/compiler.h \
	src/services/sys/error.h \

LIBMOPSY_SOURCES  = \
	src/mopsy/tl/base/tl.c
	
TEST_SOURCES      = \
	test/test.c

all: $(LIBMOPSY) $(TEST)

clean:
	$(RM) $(LIBMOPSY) $(TEST)

$(LIBMOPSY): $(LIBMOPSY_SOURCES) $(LIBMOPSY_HEADERS) Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(LIBMOPSY_SOURCES) -o $(LIBMOPSY) -shared -fPIC

$(TEST): $(TEST_SOURCES) $(TEST_HEADERS) $(LIBMOPSY) Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(TEST_SOURCES) -o $(TEST) $(LIBMOPSY) -Wl,-rpath $(PWD)
	