#
# Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_GO

GOOBJDIR=$(abs_top_builddir)/bindings/go/$(objdir)
GOPATH=$(abs_top_srcdir)/bindings/go/
CGOCFLAGS=-I$(abs_top_builddir)/src -I$(abs_top_srcdir)/src
CGOLDFLAGS=-L$(abs_top_builddir)/src/ucp/$(objdir) -lucp -L$(abs_top_builddir)/src/ucs/$(objdir) -lucs
UCX_SOPATH=$(abs_top_builddir)/src/ucp/$(objdir):$(abs_top_builddir)/src/ucs/$(objdir):$(abs_top_builddir)/src/ucm/$(objdir):$(abs_top_builddir)/src/uct/$(objdir)
GOTMPDIR=$(GOOBJDIR)/tmp
GOCACHE=$(GOTMPDIR)

export GOTMPDIR
export GOCACHE
export GOPATH
export CGO_CFLAGS=$(CGOCFLAGS)
export CGO_LDFLAGS=$(CGOLDFLAGS)

$(GOTMPDIR):
	$(AM_V_at)mkdir -p $(GOTMPDIR)

build: $(GOTMPDIR)
	cd $(abs_top_srcdir)/bindings/go/src/ucx ;\
	$(GO) build

clean-local:
	$(AM_V_at)-rm -rf $(GOOBJDIR)

test: $(GOTMPDIR)
	cd $(abs_top_srcdir)/bindings/go/tests ;\
	$(GO) env -w GO111MODULE=off ; \
	LD_LIBRARY_PATH=$(UCX_SOPATH):${LD_LIBRARY_PATH} $(GO) test -v

bench: $(GOTMPDIR)
	cd $(abs_top_srcdir)/bindings/go/tests ;\
	$(GO) env -w GO111MODULE=off ; \
	LD_LIBRARY_PATH=$(UCX_SOPATH):${LD_LIBRARY_PATH} $(GO) test -v -bench=.

goperftest: $(GOTMPDIR)
	$(GO) env -w GO111MODULE=off ; \
	cd $(abs_top_srcdir)/bindings/go/src/examples/perftest ;\
	$(GO) build -o ${GOTMPDIR}/goperftest

run-perftest:
	cd $(abs_top_srcdir)/bindings/go/src/examples/perftest ;\
	LD_LIBRARY_PATH=$(UCX_SOPATH):${LD_LIBRARY_PATH} ${GOTMPDIR}/goperftest ${ARGS}

install-exec-hook:
	$(INSTALL) ${GOTMPDIR}/goperftest $(DESTDIR)$(bindir)

all: goperftest build

.PHONY: all build run_perftest test bench

endif
