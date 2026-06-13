# docs/conf.py
from __future__ import annotations
import sys
from unittest.mock import MagicMock

# Mock heavy deps so RTD doesn't need to install torch/numpy
for mod in ["torch", "torch.nn", "torch.nn.functional", "numpy", "box"]:
    sys.modules.setdefault(mod, MagicMock())
    
project = "irelium"
author = "xqyn"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "autoapi.extension",
]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

autoapi_dirs = ["../irelium"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch":  ("https://pytorch.org/docs/stable", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
}

#html_theme = "furo"
# html_theme = "pydata_sphinx_theme"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/xqyn/irelium",
    "navbar_end": ["navbar-icon-links"],
    "logo": {"text": "irelium"},
}

html_static_path = ["_static"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}