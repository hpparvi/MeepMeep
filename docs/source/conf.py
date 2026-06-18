# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MeepMeep'
copyright = '2026, Hannu Parviainen'
author = 'Hannu Parviainen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx_design',
    'nbsphinx',
    'numpydoc'
]

autosummary_generate = True

# numpydoc parses the NumPy-style docstrings (Parameters / Returns / Notes /
# Examples) used throughout the package. The class-member summary table that
# numpydoc inserts is disabled because the API pages drive their own member
# listings through autosummary :toctree: directives; leaving it on duplicates
# those entries and emits "toctree references nonexisting document" warnings.
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# With xref_param_type on, numpydoc tries to cross-reference every token in a
# type field. These words are prose/structure that leak out of NumPy-style
# type strings ("ndarray, shape (N,), optional", "matplotlib color") and have
# no resolvable target; list them so the refs are never generated (cleaner
# than suppressing the resulting warnings after the fact).
numpydoc_xref_ignore = {
    'shape', 'optional', 'default', 'of', 'or', 'type',
    'N', 'npt', 'color', 'matplotlib',
}

# Docstring validation is available via numpydoc_validation_checks (e.g.
# {"GL08", "PR01", "RT01"}); left unset for now to avoid flooding the build
# while the refactor is in progress. Enable once the baseline is curated.

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

templates_path = ['_templates']
exclude_patterns = []

root_doc = 'index'

# Treat every unresolved cross-reference as a warning so that stale
# ``:func:`` / ``:class:`` targets (e.g. a renamed dispatcher) are caught
# at build time rather than silently rendered as plain text. External
# references that have no intersphinx target are listed in
# ``nitpick_ignore_regex`` below.
nitpicky = True

# Targets that legitimately have no resolvable inventory. Each entry
# silences a structural false positive, not genuine cross-reference rot:
#   - the numba API and numpy typing internals (no usable objects.inv);
#   - the array-type aliases used in docstring signatures;
#   - the private dispatcher kernels (``_pos_osd`` etc.) that the public
#     ``*_o`` / ``*_od`` docstrings mention but that stay undocumented.
# Prose/structure tokens that leak out of NumPy-style type fields
# ("ndarray, shape (N,), optional") are handled upstream by
# ``numpydoc_xref_ignore`` so the refs are never generated in the first place.
nitpick_ignore_regex = [
    (r'py:.*', r'numba\..*'),
    (r'py:.*', r'numpy\._typing.*'),
    (r'py:.*', r'numpy\..*_ScalarT$'),
    (r'py:.*', r'(NDArray|ndarray)$'),
    (r'py:.*', r'.*\.ndarray$'),
    (r'py:.*', r'Optional$'),
    (r'py:func', r'_[a-z_]+_o[sv]d?$'),
    (r'py:meth', r'.*\(.*\)$'),
]

# NOTE: the residual nitpicky warnings (cross-references to
# ``meepmeep.backends.numba.*`` paths) are a known, deferred backlog: those
# objects are documented under the public ``meepmeep.numba2d`` /
# ``meepmeep.numba3d`` aggregators, so the backend-path refs do not resolve.
# nitpicky is kept on as a tripwire for *new* rot; see the project notes.


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
