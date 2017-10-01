#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#
#
#
# Check for Java support
#
#
java_happy="no"
AC_ARG_WITH([java],
            [AC_HELP_STRING([--with-java],
                            [Compile Java UCX (default is NO).]
                           )], [with_java=yes], [with_java=no])


AS_IF([test "x$with_java" != xno],
      [AS_IF([test "x$with_java" == xyes],
             [AS_IF([test -n "$JAVA_HOME"],
                    [],
                    [
                     READLINK_FKNOWN=$(readlink -f xxx 1>/dev/null 2>&1 && echo yes)
                     AS_IF([test "x${READLINK_FKNOWN}" == xyes],
                           [
                            AC_SUBST([JAVA], [$(readlink -f $(type java | awk '{print $3;}'))])
                            AC_SUBST([JAVA_HOME], [${JAVA%*/jre*}])
                            AC_MSG_WARN([Please set JAVA_HOME=$JAVA_HOME])
                           ],
                           [
                            AC_MSG_ERROR([Plesae install readlink or set JAVA_HOME=<path-to-java>])
                           ]
                          )
                    ]
                   )
              with_java=$JAVA_HOME

              AC_CHECK_PROG(MVNBIN, mvn, yes)
              AS_IF([test x"${MVNBIN}" != x"yes"],
                    [AC_MSG_ERROR([Unable to find mvn on your path])])
             ],
            )

       AS_IF([test -d "$with_java"],
             [],
             [AC_MSG_ERROR([Please set JAVA_HOME=<path-to-java>])]
            )

       save_CPPFLAGS="$CPPFLAGS"

       CPPFLAGS="-I$with_java/include/linux $CPPFLAGS"

        AC_CHECK_HEADER([$with_java/include/linux/jni_md.h],
                        [
                         AC_CHECK_HEADER([$with_java/include/jni.h],
                                         [java_happy="yes"],
                                         [
                                          AC_MSG_ERROR([jni.h file not found])
                                         ]
                                        )
                        ],
                        [
                         AC_MSG_ERROR([jni_md.h file not found])
                        ]
                       )

        CPPFLAGS="$save_CPPFLAGS"
      ]
     )

AM_CONDITIONAL([HAVE_JAVA], [test "x$java_happy" != "xno"])
