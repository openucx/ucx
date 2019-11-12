#
# Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_JAVA

topdir=$(abs_top_builddir)
java_build_dir=$(builddir)/build-java
javadir=$(top_srcdir)/bindings/java
MVNCMD=$(MVN) -B -T 1C -f $(topdir)/bindings/java/pom.xml -Dmaven.repo.local=$(java_build_dir)/.deps \
              -Dorg.slf4j.simpleLogger.log.org.apache.maven.cli.transfer.Slf4jMavenTransferListener=warn

BUILT_SOURCES = org_openucx_jucx_ucp_UcpConstants.h \
                org_openucx_jucx_ucp_UcpContext.h \
                org_openucx_jucx_ucp_UcpEndpoint.h \
                org_openucx_jucx_ucp_UcpWorker.h \
                org_openucx_jucx_ucs_UcsConstants.h \
                org_openucx_jucx_ucp_UcpListener.h

DISTCLEANFILES = org_openucx_jucx_ucp_UcpConstants.h \
                 org_openucx_jucx_ucp_UcpContext.h \
                 org_openucx_jucx_ucp_UcpEndpoint.h \
                 org_openucx_jucx_ucp_UcpWorker.h \
                 org_openucx_jucx_ucs_UcsConstants.h \
                 org_openucx_jucx_ucp_UcpListener.h

org_openucx_jucx_ucp_UcpListener.h:
org_openucx_jucx_ucp_UcpConstants.h:
org_openucx_jucx_ucp_UcpEndpoint.h:
org_openucx_jucx_ucp_UcpWorker.h:
org_openucx_jucx_ucs_UcsConstants.h:
org_openucx_jucx_ucp_UcpContext.h:
	$(MVNCMD) compile native:javah

lib_LTLIBRARIES = libjucx.la

libjucx_la_CPPFLAGS = -I$(JDK)/include -I$(JDK)/include/linux \
                      -I$(topdir)/src -I$(top_srcdir)/src

libjucx_la_SOURCES = context.cc \
                     endpoint.cc \
                     jucx_common_def.cc \
                     listener.cc \
                     memory.cc \
                     ucp_constants.cc \
                     ucs_constants.cc \
                     worker.cc

libjucx_la_CXXFLAGS = -fPIC -DPIC -Werror

libjucx_la_LIBADD = $(topdir)/src/ucs/libucs.la \
                    $(topdir)/src/uct/libuct.la \
                    $(topdir)/src/ucm/libucm.la \
                    $(topdir)/src/ucp/libucp.la

libjucx_la_DEPENDENCIES = Makefile.am Makefile.in Makefile

# Compile Java source code and pack to jar
package:
	$(MVNCMD) package -DskipTests

# Maven install phase
install-data-hook: package
	cp -fp $(java_build_dir)/jucx-@VERSION@.jar $(DESTDIR)@libdir@

# Remove all compiled Java files
clean-local:
	-rm -rf $(java_build_dir)

test:
	$(MVNCMD) test
docs:
	$(MVNCMD) javadoc:javadoc

endif
