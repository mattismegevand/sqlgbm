import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'sqlgbm'
copyright = '2025, Mattis Megevand'
author = 'Mattis Megevand'

extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'myst_parser'
]

html_theme = 'sphinx_rtd_theme' 