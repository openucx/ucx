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
            [AC_HELP_STRING([--with-java=(PATH)],
                            [Compile Java UCX (default is guess).])
            ], [], [with_java=guess])

AS_IF([test "x$with_java" != xno],
      [
       AC_CHECK_PROG(MVNBIN,  mvn,  yes)
       AC_CHECK_PROG(JAVABIN, java, yes)
       AS_IF([test "x${MVNBIN}" = "xyes" -a "x${JAVABIN}" = "xyes"],
             [
              AS_IF([test -n "$with_java" -a "x$with_java" != "xyes" -a "x$with_java" != "xguess"],
                    [java_dir=$with_java],
                    [
                     AS_IF([test -n "$JAVA_HOME"],
                           [],
                           [
                            AC_CHECK_PROG(READLINK, readlink, yes)
                            AS_IF([test "x${READLINK}" = xyes],
                                  [
                                   JAVA_BIN_FOLDER=`AS_DIRNAME([$(readlink -f $(type -P javac))])`
                                   JAVA_HOME=`AS_DIRNAME([$JAVA_BIN_FOLDER])`
                                   AC_MSG_NOTICE([Setting JAVA_HOME=$JAVA_HOME])
                                  ],
                                  [
                                   AS_IF(
                                         [test "x$with_java" = "xguess"],
                                         [AC_MSG_WARN([For Java support please install readlink or set JAVA_HOME=<path-to-java>])],
                                         [AC_MSG_ERROR([Java support requested, but couldn't find path; please set JAVA_HOME=<path-to-java>])]
                                        )
                                  ]
                                 )
                           ]
                          )
                     java_dir=$JAVA_HOME
                    ]
                   )
              save_CPPFLAGS="$CPPFLAGS"
              CPPFLAGS="-I$java_dir/include/linux -I$java_dir/include $CPPFLAGS"
              AC_CHECK_HEADERS([jni_md.h jni.h],
                              [
                               java_happy="yes"
                              ],
                              [
                               AS_IF([test "x$with_java" = "xguess"],
                                     [AC_MSG_WARN([Couldn't find jni headers.])],
                                     [AC_MSG_ERROR([Java support requested, but couldn't find jni headers in $java_dir])]
                                    )
                              ]
                             )

              CPPFLAGS="$save_CPPFLAGS"
             ],
             [
              AS_IF([test "x$with_java" = "xguess"],
                    [AC_MSG_WARN([Disabling Java support - java or mvn not in path.])],
                    [AC_MSG_ERROR([Java support was explicitly requested, but java or mvn not in path.])]
                   )
             ]
            )
      ],
      [AC_MSG_WARN([Java support was explicitly disabled.])]
     )

AC_SUBST([JDK], [${java_dir}])
AM_CONDITIONAL([HAVE_JAVA], [test "x$java_happy" != "xno"])
#Set MVN according to whether user has Java and Maven or not
AM_COND_IF([HAVE_JAVA],
           [AC_SUBST([MVN], ["mvn"])
           build_bindings="${build_bindings}:java"]
          )
