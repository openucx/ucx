
LIBUCT     = libuct.so
TEST       = test/test
CC         = gcc
RM         = rm -f
CPPFLAGS   = -Isrc
CFLAGS     = -O3 -g

LIBUCT_HEADERS  = \
	src/uct/api/tl.h \
	src/uct/api/uct_def.h \
	src/uct/api/uct.h

LIBUCT_SOURCES  = \
	src/uct/tl/tl.c
	
TEST_SOURCES      = \
	test/test.c

.PHONY: clean all

all: $(LIBUCT) $(TEST)

clean:
	$(RM) $(LIBUCT) $(TEST)

$(LIBUCT): $(LIBUCT_SOURCES) $(LIBUCT_HEADERS) Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(LIBUCT_SOURCES) -o $(LIBUCT) -shared -fPIC

$(TEST): $(TEST_SOURCES) $(TEST_HEADERS) $(LIBUCT) Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(TEST_SOURCES) -o $(TEST) $(LIBUCT) -Wl,-rpath $(PWD)
	
