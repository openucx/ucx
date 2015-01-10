AC_ARG_WITH([rte],
            [AC_HELP_STRING([--with-rte(=DIR)],
                            [Where to find the RTE libraries and header
                            files]
                           )], [], [with_rte=no])

AS_IF([test "x$with_rte" != xno],
      [
      AC_CHECK_HEADERS([$with_rte/include/rte.h], [rte_happy="yes"], [rte_happy="no"])
      AS_IF([test "x$rte_happy" == xyes],
            [
            AC_SUBST(RTE_CPPFLAGS,  "-I$with_rte/include")
            AC_SUBST(RTE_LDFLAGS,   "-L$with_rte/lib -lrte")
            AC_DEFINE([HAVE_RTE], [1], [RTE support])
            ], [])],
      [])
