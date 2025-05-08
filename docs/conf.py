import os
import sys

# Add the project directory to the system path
sys.path.insert(0, os.path.abspath("../"))

# Project information
project = "BertNado"
copyright = "2025, Catherine Chahrour"
author = "Catherine Chahrour"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# Options for HTML output
# Use the Furo theme for a modern look
html_theme = "furo"

# Add custom CSS for further styling
html_static_path = ["_static"]
html_css_files = ["custom.css"]
