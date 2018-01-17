topdir=$(abs_top_builddir)
javadir=$(topdir)/bindings/java
MVNCMD=@cd $(javadir) && $(MVN)

lib_LTLIBRARIES = libjucx.la

libjucx_la_CPPFLAGS = -I$(JDK)/include -I$(JDK)/include/linux \
                      -I$(topdir)/src

libjucx_la_SOURCES = bridge.cc request_util.cc

libjucx_la_CXXFLAGS = -fPIC -DPIC -Werror -std=c++0x

libjucx_la_LIBADD = $(topdir)/src/ucs/libucs.la \
                    $(topdir)/src/uct/libuct.la \
                    $(topdir)/src/ucm/libucm.la \
                    $(topdir)/src/ucp/libucp.la

libjucx_la_DEPENDENCIES = Makefile.am Makefile.in Makefile


#Compile Java source code and pack to jar
all:
	$(MVNCMD) package

#Maven install phase
install-data-hook:
	$(MVNCMD) install

#Remove all compiled Java files
clean-local:
	$(MVNCMD) clean
