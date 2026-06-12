# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nicgiprep'
copyright = '2026, Adrià Casamitjana, Agustín Cartaya Lathulerie, Clara Lisazo, Rachika E. Hamadache'
author = 'Adrià Casamitjana, Agustín Cartaya Lathulerie, Clara Lisazo, Rachika E. Hamadache'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # NumPy & Google docstring support
    'sphinx.ext.viewcode',   # adds [source] links
    'sphinx.ext.autosummary',
]

# Napoleon settings for NumPy style
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Theme
html_theme = 'sphinx_rtd_theme'

import os, sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../nicgiprep'))

templates_path = ['_templates']
exclude_patterns = []



# # -- Options for HTML output -------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']
