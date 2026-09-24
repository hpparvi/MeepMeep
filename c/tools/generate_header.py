#!/usr/bin/env python3
"""Regenerate the prototype block of ``c/include/meepmeep.h``.

The C library is a second compile target of the shared sources in
``meepmeep/backends/opencl/*.cl``; their function signatures are the C API.
This script extracts every ``MM_INLINE`` definition (with the comment that
precedes it), rewrites it as a C99 prototype (``REAL`` -> ``double``,
``MM_GLOBAL`` dropped) and splices the result between the
``BEGIN``/``END GENERATED PROTOTYPES`` markers of the header. The rest of the
header (constants, status codes, and the functions implemented under
``c/src/``) is hand-written.

Usage::

    python c/tools/generate_header.py          # rewrite the header in place
    python c/tools/generate_header.py --check  # exit 1 if the header is stale

``meepmeep/tests/test_c_library.py`` runs the ``--check`` form, so a signature
change in a ``.cl`` file fails the test suite until the header is regenerated.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / 'meepmeep' / 'backends' / 'opencl'
HEADER = ROOT / 'c' / 'include' / 'meepmeep.h'

# Dependency order; solve_kernels.cl is OpenCL-only and deliberately absent.
SOURCE_FILES = ('common.cl', 'solve2d.cl', 'solve3d.cl', 'point2d.cl', 'point2dd.cl',
                'point3d.cl', 'point3dd.cl', 'orbit3d.cl', 'orbit3dd.cl')

BEGIN = '/* BEGIN GENERATED PROTOTYPES -- edit c/tools/generate_header.py, not this block. */'
END = '/* END GENERATED PROTOTYPES */'

# An optional comment block immediately (whitespace only) before the definition.
_DEFINITION = re.compile(
    r'(?P<comment>/\*(?:(?!\*/).)*\*/[ \t]*\n)?'
    r'^MM_INLINE[ \t]+(?P<signature>[^{;]*?\))[ \t]*\{',
    re.S | re.M)
_WIDTH = 80


def _prototype(signature: str) -> str:
    """Rewrite one ``MM_INLINE`` signature as a wrapped C99 prototype."""
    flat = ' '.join(signature.split())
    flat = re.sub(r'\bMM_GLOBAL\s+', '', flat)
    flat = re.sub(r'\bREAL\b', 'double', flat)
    head, _, rest = flat.partition('(')
    args = [arg.strip() for arg in rest[:-1].split(',')]
    indent = ' ' * (len(head) + 1)
    lines, current = [], head + '('
    for k, arg in enumerate(args):
        piece = arg + (',' if k < len(args) - 1 else ');')
        candidate = current + ('' if current.endswith('(') else ' ') + piece
        if len(candidate) > _WIDTH and not current.endswith('('):
            lines.append(current)
            current = indent + piece
        else:
            current = candidate
    lines.append(current)
    return '\n'.join(lines)


def generate() -> str:
    """Return the text that belongs between the markers."""
    out = []
    for name in SOURCE_FILES:
        out.append(f'\n/* ---- {name} {"-" * (_WIDTH - 12 - len(name))} */\n')
        text = (SOURCE_DIR / name).read_text()
        for match in _DEFINITION.finditer(text):
            comment = match.group('comment')
            if comment:
                out.append(comment.rstrip() + '\n')
            out.append(_prototype(match.group('signature')) + '\n\n')
    return ''.join(out)


def splice(header: str, block: str) -> str:
    start = header.index(BEGIN) + len(BEGIN)
    end = header.index(END)
    return header[:start] + '\n' + block + header[end:]


def main(argv) -> int:
    current = HEADER.read_text()
    updated = splice(current, generate())
    if '--check' in argv:
        if updated != current:
            print(f'{HEADER.relative_to(ROOT)} is stale; run python c/tools/generate_header.py')
            return 1
        return 0
    HEADER.write_text(updated)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
