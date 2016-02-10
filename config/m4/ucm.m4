#
# Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#


#
# Enable overriding library symbols
#
AC_ARG_ENABLE([symbol-override],
	AS_HELP_STRING([--disable-symbol-override], [Disable overriding library symbols, default: NO]),
	[],
	[enable_symbol_override=yes])
	
AS_IF([test "x$enable_symbol_override" == xyes], 
	[AC_DEFINE([ENABLE_SYMBOL_OVERRIDE], [1], [Enable symbol override])]
	[:]
)

