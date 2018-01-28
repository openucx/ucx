/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.util;

import java.util.HashMap;
import java.util.Map;

/**
 * Utility class to handle command line arguments.
 */
public class Options {
    public static final char EOF = (char) -1;

    private String                  optsStr;
    private String[]                args;
    private int                     optCnt;
    private Map<Character, Boolean> options;
    private String                  nonOption;

    public String optArg;

    public Options(String opts, String[] args) {
        this.args = args;
        optsStr = opts;
        optCnt = 0;
        options = null;
        optArg = null;
        nonOption = null;
    }

    public String getNonOptionArgument() {
        return nonOption;
    }

    public char getOpt() {
        if (options == null) {
            options = new HashMap<>();
            parseOpts();
        }

        if (args == null || args.length == 0)
            return EOF;

        if (optCnt == 0 && !isOption(args[optCnt]))
            nonOption = args[optCnt++];

        if (optCnt >= args.length)
            return EOF;

        char opt = args[optCnt++].charAt(1);
        if (!options.containsKey(opt))
            return opt;

        if (options.get(opt))
            optArg = args[optCnt++];
        else
            optArg = null;

        return opt;
    }

    private void parseOpts() {
        if (optsStr == null)
            return;

        int len = optsStr.length();
        for (int i = 0; i < len; i++) {
            char ch = optsStr.charAt(i);

            if (ch == ':')
                continue;

            if (i < len - 1 && optsStr.charAt(i + 1) == ':')
                options.put(ch, true);
            else
                options.put(ch, false);
        }
    }

    private boolean isOption(String option) {
        return option.startsWith("-");
    }
}
