topdir=$(abs_top_builddir)
javadir=$(topdir)/bindings/java
MVNCMD=@cd $(javadir) && $(MVN)

BUILT_SOURCES = org_ucx_jucx_UcxConstants.h \
                org_ucx_jucx_ucp_UcpContext.h

DISTCLEANFILES = org_ucx_jucx_UcxConstants.h \
                 org_ucx_jucx_ucp_UcpContext.h

org_ucx_jucx_UcxConstants.h: org_ucx_jucx_ucp_UcpContext.h

org_ucx_jucx_ucp_UcpContext.h:
	$(MVNCMD) compile native:javah

lib_LTLIBRARIES = libjucx.la

libjucx_la_CPPFLAGS = -I$(JDK)/include -I$(JDK)/include/linux \
                      -I$(topdir)/src

libjucx_la_SOURCES = ucp_constants.cc context.cc jucx_common_def.cc

noinst_libjucx_la_SOURCES = org_ucx_jucx_UcxConstants.h \
                            org_ucx_jucx_ucp_UcpContext.h

libjucx_la_CXXFLAGS = -fPIC -DPIC -Werror -std=c++11

libjucx_la_LIBADD = $(topdir)/src/ucs/libucs.la \
                    $(topdir)/src/uct/libuct.la \
                    $(topdir)/src/ucm/libucm.la \
                    $(topdir)/src/ucp/libucp.la

libjucx_la_DEPENDENCIES = Makefile.am Makefile.in Makefile

#Compile Java source code and pack to jar
package:
	$(MVNCMD) package -DskipTests

#Maven install phase
install-data-hook:
	$(MVNCMD) install -DskipTests

#Remove all compiled Java files
clean-local:
	$(MVNCMD) clean

test:
	$(MVNCMD) test
