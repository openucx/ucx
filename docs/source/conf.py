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
