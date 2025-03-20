#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

CC ?= gcc
CFLAGS = -ggdb3 -fPIC -Wall -Wextra -Werror

INCLUDES ?=
BUILD ?= ./build

MAJOR = 1
MINOR = 1
VERSION = $(MAJOR).$(MINOR)

OBJS_IBVERBS = $(addprefix $(BUILD)/, verbs.o config.o stubs.o)
OBJS_EFA = $(addprefix $(BUILD)/, efa.o config.o)

.PHONY: all clean make_dir

all: make_dir $(BUILD)/libibverbs.so.1 $(BUILD)/libefa.so.1

make_dir: $(BUILD)/
$(BUILD)/:
	mkdir -p $(BUILD)

$(OBJS_EFA) $(OBJS_IBVERBS): *.h

$(BUILD)/libibverbs.so: $(OBJS_IBVERBS)
$(BUILD)/libefa.so: $(OBJS_EFA)
$(BUILD)/%.so.$(MAJOR): $(BUILD)/%.so

$(BUILD)/%.o: %.c
	$(CC) -c $(INCLUDES) $(CFLAGS) -o $@ $<

$(BUILD)/%.so:
	$(CC) -shared -o $@ -Wl,--soname='$@.$(VERSION)' $^

$(BUILD)/%.so.$(MAJOR): $(BUILD)/%.so
	ln -sf $(shell basename $<) $@

clean:
	rm -f $(addprefix $(BUILD)/, *.o *.so.* *.so)
