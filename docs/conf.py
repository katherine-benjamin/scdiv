"""Sphinx configuration for scdiv documentation."""

from __future__ import annotations

import importlib.metadata

project = "scdiv"
author = "Katherine Benjamin"
copyright = f"2026, {author}"  # noqa: A001
version = importlib.metadata.version("scdiv")
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "myst_parser",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
default_role = "any"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
typehints_defaults = "comma"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

html_theme = "sphinx_book_theme"
html_title = "scdiv"
html_static_path = ["_static"]
html_extra_path = ["_extra"]
html_theme_options = {
    "repository_url": "https://github.com/katherine-benjamin/scdiv",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": True,
}
