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
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

autosummary_generate = True

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

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
#   - bare tokens that NumPy-style "ndarray, shape (N, 3, 5)" type fields
#     leak into the type position (``shape``, ``N``, ``5``, ``default ...``);
#   - the array-type aliases used in docstring signatures;
#   - the private dispatcher kernels (``_pos_osd`` etc.) that the public
#     ``*_o`` / ``*_od`` docstrings mention but that stay undocumented.
nitpick_ignore_regex = [
    (r'py:.*', r'numba\..*'),
    (r'py:.*', r'numpy\._typing.*'),
    (r'py:.*', r'numpy\..*_ScalarT$'),
    (r'py:.*', r'(NDArray|ndarray)$'),
    (r'py:.*', r'.*\.ndarray$'),
    (r'py:.*', r'Optional$'),
    (r'py:class', r'shape$'),
    (r'py:class', r'optional$'),
    (r'py:class', r'default .*'),
    (r'py:class', r'[A-Z]$'),
    (r'py:class', r'\d+$'),
    (r'py:class', r"['{].*"),  # "{'mm', 'ea', 'ta'}" choice literals
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

html_theme = 'alabaster'
html_static_path = ['_static']
