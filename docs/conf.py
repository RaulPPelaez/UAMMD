# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../src/'))
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'UAMMD'
copyright = '2022, Raul P. Pelaez'
author = 'Raul P. Pelaez'

# The full version, including alpha/beta/rc tags
release = '2.0'


# -- General configuration ---------------------------------------------------

primary_domain = 'cpp'
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx_rtd_theme',
    'sphinx.ext.imgmath',
    'sphinxcontrib.inkscapeconverter',
    'sphinx.ext.todo',
    "sphinx_disqus.disqus"
]

imgmath_image_format='svg'
imgmath_use_preview=True


disqus_shortname = 'uammd-readthedocs'

todo_include_todos=True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'global.rst', 'README.md', 'uammd_cpp_lexer.py']


cpp_id_attributes = ["__device__","__host__"]
cpp_paren_attributes = ["__device__","__host__"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


html_sidebars = { '**': ['about.html','navigation.html', 'relations.html','searchbox.html',] }

html_theme_options = {
#    'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
#    'analytics_anonymize_ip': False,
#    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
#    'vcs_pageview_mode': '',
#    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_logo = "img/logo.png"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

rst_prolog = open('global.rst', 'r').read()


imgmath_latex_preamble = r'''
\usepackage{bm}
\usepackage{svg}
\usepackage{graphicx}
\graphicspath{{img/}}
\renewcommand{\vec}[1]{\bm{#1}}
\newcommand{\kT}{k_B T}
\newcommand{\tens}[1]{\bm{\mathcal{#1}}}
\newcommand{\oper}[1]{\mathcal{#1}}
\newcommand{\dt}{\delta t}
\newcommand{\sinc}{\textrm{sinc}}
\newcommand{\floor}{\textrm{floor}}
\newcommand{\near}{\textrm{near}}
\newcommand{\far}{\textrm{far}}
\newcommand{\half}{\frac{1}{2}}
\newcommand{\red}[1]{{\color{red}#1}}
\newcommand{\fou}[1]{\widehat{#1}}
\newcommand{\noise}{\widetilde{W}}
\DeclareMathOperator{\erf}{erf}

\newcommand{\ppos}{q}
\newcommand{\pvel}{u}
\newcommand{\fpos}{r}
\newcommand{\fvel}{v}

\newcommand{\corr}{\text{corr}}
\newcommand{\dpr}{\text{\tiny DP}}
\newcommand{\qtd}{\text{\tiny q2D}}

'''

latex_elements = {'preamble': imgmath_latex_preamble}


from sphinx.highlighting import lexers
import uammd_cpp_lexer

lexers['cpp'] = uammd_cpp_lexer.UAMMDCppLexer()
lexers['c++'] = lexers['cpp']

#pygments_style='pastie'
#pygments_style='lovelace'
