#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_JAVA

jardir         = $(libdir)
topdir         = $(abs_top_builddir)
java_build_dir = $(builddir)/build-java
maven_repo    ?= $(java_build_dir)/.deps
jarfile        = $(java_build_dir)/jucx-@VERSION@$(suffix).jar
javadir        = $(top_srcdir)/bindings/java

MVNCMD = $(MVN) -B -T 1C -f \
         $(topdir)/bindings/java/pom.xml \
         -Dmaven.repo.local=$(maven_repo) \
         -Dorg.slf4j.simpleLogger.log.org.apache.maven.cli.transfer.Slf4jMavenTransferListener=warn

JUCX_GENERATED_H_FILES = org_openucx_jucx_ucp_UcpConnectionRequest.h     \
                         org_openucx_jucx_ucp_UcpConstants.h             \
                         org_openucx_jucx_ucp_UcpContext.h               \
                         org_openucx_jucx_ucp_UcpEndpoint.h              \
                         org_openucx_jucx_ucp_UcpListener.h              \
                         org_openucx_jucx_ucp_UcpMemory.h                \
                         org_openucx_jucx_ucp_UcpRemoteKey.h             \
                         org_openucx_jucx_ucp_UcpWorker.h                \
                         org_openucx_jucx_ucs_UcsConstants_ThreadMode.h  \
                         org_openucx_jucx_ucs_UcsConstants.h             \
                         org_openucx_jucx_UcxUtils.h

BUILT_SOURCES = $(JUCX_GENERATED_H_FILES)

STAMP_FILE = native_headers.stamp

MOSTLYCLEANFILES = $(JUCX_GENERATED_H_FILES) $(STAMP_FILE)

#
# Create a timestamp file to avoid regenerating header files every time
# See https://www.gnu.org/software/automake/manual/html_node/Multiple-Outputs.html
#
$(STAMP_FILE): \
		$(javadir)/src/main/java/org/openucx/jucx/ucs/*.java \
		$(javadir)/src/main/java/org/openucx/jucx/ucp/*.java \
		$(javadir)/src/main/java/org/openucx/jucx/examples/*.java
	$(MVNCMD) compile
	touch $(STAMP_FILE)

$(JUCX_GENERATED_H_FILES): $(STAMP_FILE)

# Return CPU architecture in Java format
arch = $(if $(filter x86_64,$(shell uname -p)),amd64,$(shell uname -p))

# Append non-x86_64 arch as filename suffix
suffix = $(if $(filter-out x86_64,$(shell uname -p)),-$(shell uname -p))

libjucx = $(objdir)/libjucx_$(arch).so

lib_LTLIBRARIES = libjucx.la

libjucx_la_CPPFLAGS = -I$(JDK)/include -I$(JDK)/include/linux \
                      -I$(topdir)/src -I$(top_srcdir)/src

noinst_HEADERS = jucx_common_def.h

libjucx_la_SOURCES = context.cc \
                     endpoint.cc \
                     jucx_common_def.cc \
                     listener.cc \
                     memory.cc \
                     ucp_constants.cc \
                     ucs_constants.cc \
                     worker.cc

libjucx_la_CXXFLAGS = $(BASE_CXXFLAGS) -std=gnu++98

libjucx_la_LIBADD = $(topdir)/src/ucs/libucs.la \
                    $(topdir)/src/uct/libuct.la \
                    $(topdir)/src/ucm/libucm.la \
                    $(topdir)/src/ucp/libucp.la

libjucx_la_DEPENDENCIES = Makefile.am Makefile.in Makefile

# Compile Java source code and pack to jar
$(libjucx): libjucx.la
	cp $(objdir)/libjucx.so $(libjucx)

$(jarfile): $(libjucx)
	$(MVNCMD) package -DskipTests -Djar.finalName=jucx-@VERSION@$(suffix)
	jar tf $(jarfile)

package : $(jarfile)

.PHONY: package

# Maven install phase
jar_DATA = $(jarfile)

# Remove all compiled Java files
clean-local:
	-rm -rf $(java_build_dir)

set-version:
	$(MVNCMD) versions:set -DnewVersion=${JUCX_VERSION}

# Add multi-platform support (CI only)
repack-jar: $(jarfile)
	$(RM) -r $(java_build_dir)/repack_dir
	unzip -o $(jarfile) -d $(java_build_dir)/repack_dir
	\cp -v $(topdir)/libjucx_*.so/libjucx_*.so $(topdir)/bindings/java/resources
	\cp -v $(topdir)/libjucx_*.so/libjucx_*.so $(java_build_dir)/repack_dir
	jar -cf $(jarfile) -C $(java_build_dir)/repack_dir .
	jar tf $(jarfile)

multi-arch:
	@make set-version JUCX_VERSION=${JUCX_VERSION}
	@make repack-jar

check-jar:
	@test $(shell jar tf $(jarfile) | grep -q libjucx_amd64.so)
	@test $(shell jar tf $(jarfile) | grep -q libjucx_aarch64.so)

# Publish JUCX jar to maven central
publish-snapshot:
	@make set-version JUCX_VERSION=${JUCX_VERSION}-SNAPSHOT
	@make publish

publish-release:
	@make set-version JUCX_VERSION=${JUCX_VERSION}
	@make publish

publish-local:
	@make set-version JUCX_VERSION=${JUCX_VERSION}
	$(MVNCMD) deploy -DskipTests ${ARGS} -Plocal-deploy

publish:
	$(MVNCMD) deploy -DskipTests ${ARGS}

test:
	$(MVNCMD) test -DargLine="-XX:OnError='cat hs_err_pid%p.log'"
docs:
	$(MVNCMD) javadoc:javadoc

endif
