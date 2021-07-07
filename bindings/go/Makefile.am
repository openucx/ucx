#
# Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_GO

GOPATH=$(abs_top_builddir)/bindings/go/$(objdir)
CGOCFLAGS=-I$(abs_top_builddir)/src -I$(top_srcdir)/src
CGOLDFLAGS=-L$(abs_top_builddir)/src/ucp/$(objdir) -lucp -L$(abs_top_builddir)/src/ucs/$(objdir) -lucs
UCX_SOPATH=$(abs_top_builddir)/src/ucp/$(objdir):$(abs_top_builddir)/src/ucs/$(objdir):$(abs_top_builddir)/src/ucm/$(objdir):$(abs_top_builddir)/src/uct/$(objdir)
GOTMPDIR=$(GOPATH)/tmp

export GOTMPDIR
export GOPATH
export CGO_CFLAGS=$(CGOCFLAGS)
export CGO_LDFLAGS=$(CGOLDFLAGS)

$(GOTMPDIR):
	$(AM_V_at)mkdir -p $(GOTMPDIR)

build: $(GOTMPDIR)
	$(AM_V_at)$(GO) build $(top_srcdir)/bindings/go/*.go

clean-local:
	$(AM_V_at)-rm -rf $(GOPATH)

test: $(GOTMPDIR)
	$(AM_V_at)LD_LIBRARY_PATH=$(UCX_SOPATH) $(GO) test -v $(top_srcdir)/bindings/go/*.go

bench: $(GOTMPDIR)
	$(AM_V_at)LD_LIBRARY_PATH=$(UCX_SOPATH) $(GO) test -v -bench=. $(top_srcdir)/bindings/go/*.go

all: build

endif
