# -*- coding: utf-8 -*-
#
# Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
# Copyright (C) Arm Ltd. 2021.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# Project information
#

project   = u'OpenUCX'
copyright = u'2019, UCF'
author    = u'UCF'

#
# General options
#

extensions         = ['recommonmark', 'breathe'] # For processing Markdown pages
templates_path     = ['_templates']
source_suffix      = ['.rst', '.md']
master_doc         = 'index'
language           = None
exclude_patterns   = [u'_build']
pygments_style     = None

#
# Breathe configuration
#

# breathe_separate_member_pages      = False
# breathe_show_enumvalue_initializer = False
# breathe_show_define_initializer    = False
breathe_order_parameters_first     = True
breathe_projects = {"openucx":"build/xml/",}

#                  
# HTML options     
#                  

html_theme         = 'sphinx_rtd_theme'
html_logo          = '_static/ucxlogo.png'
html_theme_options = {
    'style_external_links': True
}
html_static_path   = ['_static']
htmlhelp_basename  = 'OpenUCXdoc'


#
# UCX custom configuration
#

# For the local testing set cached_version to the latest released version (e.g 'v1.10.1').
# Otherwise, GitHub may block you for a frequent access to GitHub API (DDOS protection).
cached_version = ''

def getLatestVersion():
    import requests

    global cached_version
    if cached_version == '':
        request = requests.get('https://api.github.com/repos/openucx/ucx/releases/latest')
        cached_version = request.json()["name"]

    return cached_version

def substituteVersion(app, docname, source):
    #
    # Updating these 2 variables will automatically update all download and API
    # documentation links.
    # We don't use the normal RST substitution because it cannot substitute text
    # inside code blocks and URL links.
    #
    version_name  = getLatestVersion()
    clean_version = version_name.lstrip('v')        # remove leading 'v' for tag name
    api_version   = clean_version.rsplit('.', 1)[0] # take only MAJOR.MINOR
    result        = source[0].replace("{VERSION}", api_version) \
                             .replace("{RELEASE}", clean_version)
    source[0] = result

def run_doxygen(doxygen_home):
    import subprocess, sys

    try:
        ret = subprocess.call('cd %s && doxygen && cd -' % doxygen_home, shell=True)
        if ret < 0:
            sys.stderr.write('doxygen error code %s' % (-ret))
    except OSError as e:
        sys.stderr.write('doxygen execution failed: %s' % e)

def run_apidoc(api_dir, xml_dir):
    import subprocess, sys

    try:
        # We don't generate groups since we create those manually
        ret = subprocess.call('breathe-apidoc -m -o %s -p openucx %s -g struct,file' % (api_dir, xml_dir), shell=True)
        if ret < 0:
            sys.stderr.write('breathe-apidoc error code %s' % (-ret))
    except OSError as e:
        sys.stderr.write('breathe-apidoc execution failed: %s' % e)

def config_doxygen_file(dox_dir, src_dir, output_dir):
    with open(dox_dir +'/ucxdox', 'r') as file :
        doxygenfile = file.read()
    # Using upstream doxygen configuration to generate "breathe" friendly
    # configration. Since autotools are not availible at RTD we are doing this
    # python way
    doxygenfile = doxygenfile.replace('$(PROJECT)', 'UCX')
    doxygenfile = doxygenfile.replace('$(VERSION)', getLatestVersion())
    doxygenfile = doxygenfile.replace('$(SRCDIR)', src_dir)
    doxygenfile = doxygenfile.replace('$(DOCDIR)', output_dir)
    doxygenfile = doxygenfile.replace('$(GENERATE_HTML)', 'NO')
    doxygenfile = doxygenfile.replace('$(GENERATE_HTMLHELP)', 'NO')
    doxygenfile = doxygenfile.replace('$(GENERATE_CHI)', 'NO')
    doxygenfile = doxygenfile.replace('$(GENERATE_LATEX)', 'NO')
    doxygenfile = doxygenfile.replace('$(GENERATE_RTF)', 'NO')
    doxygenfile = doxygenfile.replace('$(GENERATE_MAN)', 'NO')
    doxygenfile = doxygenfile.replace('$(GENERATE_XML)', 'YES')
    doxygenfile = doxygenfile.replace('$(PERL_PATH)', '/usr/bin/perl')
    doxygenfile = doxygenfile.replace('$(HAVE_DOT)', 'NO')
    doxygenfile = doxygenfile.replace('docs/doxygen/header.tex', '')
    # Exclude UCT documentation
    doxygenfile = doxygenfile.replace('../..//src/uct/api/', '')

    with open(dox_dir + '/Doxyfile', 'w') as file:
        file.write(doxygenfile)

def generate_api(app):
    import os

    read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'
    if read_the_docs_build:
        dox_dir = '../doxygen'
        api_dir = 'api'
        xml_dir = 'build/xml/'
    else:
        dox_dir = 'doxygen'
        api_dir = 'source/api'
        xml_dir = 'source/build/xml/'

    output_dir = '../source/build'
    src_dir = '../../'

    config_doxygen_file(dox_dir, src_dir, output_dir)
    run_doxygen(dox_dir)
    run_apidoc(api_dir, xml_dir)

def setup(app):
    app.connect('builder-inited', generate_api)
    app.connect('source-read', substituteVersion)
