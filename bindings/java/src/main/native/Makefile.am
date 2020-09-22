#
# Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_JAVA

jardir         = $(libdir)
topdir         = $(abs_top_builddir)
java_build_dir = $(builddir)/build-java
maven_repo    ?= $(java_build_dir)/.deps
jarfile        = $(java_build_dir)/jucx-@VERSION@.jar
javadir        = $(top_srcdir)/bindings/java

MVNCMD = $(MVN) -B -T 1C -f \
         $(topdir)/bindings/java/pom.xml \
         -Dmaven.repo.local=$(maven_repo) \
         -Dorg.slf4j.simpleLogger.log.org.apache.maven.cli.transfer.Slf4jMavenTransferListener=warn

JUCX_GENERATED_H_FILES = org_openucx_jucx_ucp_UcpConstants.h             \
                         org_openucx_jucx_ucp_UcpContext.h               \
                         org_openucx_jucx_ucp_UcpEndpoint.h              \
                         org_openucx_jucx_ucp_UcpListener.h              \
                         org_openucx_jucx_ucp_UcpMemory.h                \
                         org_openucx_jucx_ucp_UcpRequest.h               \
                         org_openucx_jucx_ucp_UcpRemoteKey.h             \
                         org_openucx_jucx_ucp_UcpWorker.h                \
                         org_openucx_jucx_ucs_UcsConstants_ThreadMode.h  \
                         org_openucx_jucx_ucs_UcsConstants.h

BUILT_SOURCES = $(JUCX_GENERATED_H_FILES)

STAMP_FILE = native_headers.stamp

MOSTLYCLEANFILES = $(JUCX_GENERATED_H_FILES) $(STAMP_FILE)

#
# Create a timestamp file to avoid regenerating header files every time
# See https://www.gnu.org/software/automake/manual/html_node/Multiple-Outputs.html
#
$(STAMP_FILE): \
		$(javadir)/src/main/java/org/openucx/jucx/ucs/*.java \
		$(javadir)/src/main/java/org/openucx/jucx/ucp/*.java
	$(MVNCMD) compile
	touch $(STAMP_FILE)

$(JUCX_GENERATED_H_FILES): $(STAMP_FILE)

lib_LTLIBRARIES = libjucx.la

libjucx_la_CPPFLAGS = -I$(JDK)/include -I$(JDK)/include/linux \
                      -I$(topdir)/src -I$(top_srcdir)/src

noinst_HEADERS = jucx_common_def.h

libjucx_la_SOURCES = context.cc \
                     endpoint.cc \
                     jucx_common_def.cc \
                     listener.cc \
                     memory.cc \
                     request.cc \
                     ucp_constants.cc \
                     ucs_constants.cc \
                     worker.cc

libjucx_la_CXXFLAGS = -fPIC -DPIC -Werror -std=gnu++98

libjucx_la_LIBADD = $(topdir)/src/ucs/libucs.la \
                    $(topdir)/src/uct/libuct.la \
                    $(topdir)/src/ucm/libucm.la \
                    $(topdir)/src/ucp/libucp.la

libjucx_la_DEPENDENCIES = Makefile.am Makefile.in Makefile

# Compile Java source code and pack to jar
$(jarfile):
	$(MVNCMD) package -DskipTests

package : $(jarfile)

.PHONY: package

# Maven install phase
jar_DATA = $(jarfile)

# Remove all compiled Java files
clean-local:
	-rm -rf $(java_build_dir)

set-version:
	$(MVNCMD) versions:set -DnewVersion=${JUCX_VERSION}

# Publish JUCX jar to maven central
publish-snapshot:
	@make set-version JUCX_VERSION=@VERSION@-SNAPSHOT
	@make publish

publish-release:
	@make set-version JUCX_VERSION=${JUCX_VERSION}
	@make publish

publish:
	$(MVNCMD) deploy -DskipTests ${ARGS}

test:
	$(MVNCMD) test -DargLine="-XX:OnError='cat hs_err_pid%p.log'"
docs:
	$(MVNCMD) javadoc:javadoc

endif
