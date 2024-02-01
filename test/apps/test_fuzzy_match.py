#!/usr/bin/env python

#
# Copyright (C) 2022 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.

import os
import commands
import re
import argparse
import sys
import logging


class Environment(object):
    '''Handles environment variables setup and cleanup'''
    def __init__(self, env_vars):
        logging.info('Using env vars: %s' % env_vars)        
        self.env_vars = env_vars;
        
    def __enter__(self):
        self.cleanup()
        for var_name in self.env_vars:
            os.environ[var_name] = 'value'
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        ucx_vars = [var for var in os.environ.keys() if var.startswith('UCX_')]
        for var in ucx_vars:
            del os.environ[var]

    
class TestRunner:
    '''Main test runner'''
    def __init__(self, ucx_info, verbose):
        self.ucx_info = ucx_info
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
       
    def run(self, test_case):
        with Environment(test_case.keys()):
            matches = self.get_fuzzy_matches()

            if matches != test_case:
                raise Exception('Wrong fuzzy list: got: %s, expected: %s' % (matches, test_case))
        
            logging.info('found all expected matches: %s' % test_case)
        
    def exec_ucx_info(self):
        cmd = self.ucx_info + ' -u m -w'
        logging.info('running cmd: %s' % cmd)
    
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            raise Exception('Received unexpected exit code from ucx_info: ' + str(status))
            
        logging.info(output)
        return output
    
    def get_fuzzy_matches(self):
        output = self.exec_ucx_info()
        warn_msg = output.splitlines()[0]
        
        # This text is printed from 'parser.c' file (updates should be synced properly).
        warn_match = re.match('.*unused environment variables?: (.*)', warn_msg)
        if not warn_match:
            raise Exception('"unused vars" message was not found')
            
        output_vars = warn_match.group(1).split(';')
        matches = [re.match(r'(\w+)(?: \(maybe: (.*)\?\))?', var.strip()) for var in output_vars]
        if None in matches:
            raise Exception('Unexpected warning message format: %s' % warn_msg)
        
        return {m.group(1) : [x.strip() for x in m.group(2).split(',')] if m.group(2) else [] for m in matches}
    
def has_ib():
    status, output = commands.getstatusoutput('ibv_devinfo')
    if status != 0:
        return False
        
    return 'No IB devices found' not in output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tester for config vars fuzzy matching')
    parser.add_argument('--ucx_info', help="Path to ucx_info binary", required=True)
    parser.add_argument('-v', '--verbose', help="Add some debug prints", action='store_true', default=False)
    args = parser.parse_args()

    try:    
        runner = TestRunner(args.ucx_info, args.verbose)
        test_cases =  [{'UCX_LOF_LEVEL' : ['UCX_LOG_LEVEL']}, 
                       {'UCX_LOF_LEVEL' : ['UCX_LOG_LEVEL'], 'UCX_MOFULE_D' : ['UCX_MODULE_DIR', 'UCX_MODULES']},
                       {'UCX_SOME_VAR' : [], 'UCX_SOME_VAR2' : [],  'UCX_SOME_VAR3' : [],  'UCX_SOME_VAR4' : []},
                       {'UCX_SOME_VAR' : [], 'UCX_MOFULE_D' : ['UCX_MODULE_DIR', 'UCX_MODULES'], 'UCX_SOME_VAR2' : [], 'UCX_LOF_LEVEL' : ['UCX_LOG_LEVEL']},
                       {'UCX_RLS' : ['UCX_TLS']}]
        
        if has_ib():
            test_cases += [{'UCX_RC_VERBS_RX_MAX_BUF' : ['UCX_RC_VERBS_TX_MAX_BUFS', 'UCX_RC_VERBS_RX_MAX_BUFS', 'UCX_UD_VERBS_RX_MAX_BUFS']}]
            
        for test_case in test_cases:
            runner.run(test_case)
            
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)
        
