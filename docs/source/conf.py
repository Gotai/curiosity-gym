# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'curiosity-gym'
copyright = '2024, Chris Reimann'
author = 'Chris Reimann'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    ]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = 'Curiosity Gym'
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/chrisreimann/curiosity-gym",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    'navigation_depth': 2,
    'navbar_align': 'left',
    'logo': {
        'image_light': 'images/curiosity_logo_black.png',
        'image_dark': 'images/curiosity_logo_white.png',
    }
}

print("conf.py done.")
