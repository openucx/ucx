# -*- coding: utf-8 -*-
#
# Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
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

extensions         = ['recommonmark'] # For processing Markdown pages
templates_path     = ['_templates']
source_suffix      = ['.rst', '.md']
master_doc         = 'index'
language           = None
exclude_patterns   = [u'_build']
pygments_style     = None


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

def getLatestVersion():
    import requests
    request = requests.get('https://api.github.com/repos/openucx/ucx/releases/latest')
    return request.json()["name"]

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

def setup(app):
   app.connect('source-read', substituteVersion)
