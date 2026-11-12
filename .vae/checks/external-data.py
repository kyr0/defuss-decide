#!/usr/bin/env python3
"""Every function that saves an ONNX graph with external data removes an existing data file first.

VERIFIED: onnx.save appends to an existing external data file and points the graph at the appended copy (onnx 1.23.2,
2026-11-12); a second run of python/quantize_tasks.py into the same folder doubled the published ner q8 and vision-box q8
data files (596 MB and 101 MB of bytes no graph reads)."""
import ast, sys
from pathlib import Path

bad = []
for f in sorted([*Path('python').rglob('*.py'), *Path('scripts').glob('*.py')]):
    for fn in ast.walk(ast.parse(f.read_text())):
        if not isinstance(fn, ast.FunctionDef): continue
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
        unlinks = [c.lineno for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == 'unlink']
        for c in calls:
            if any(k.arg == 'save_as_external_data' for k in c.keywords) and not any(n <= c.lineno for n in unlinks):
                bad.append(f'{f}:{c.lineno}')
if bad:
    print('FIX: unlink the data file (missing_ok=True) before saving with save_as_external_data in: ' + ', '.join(bad))
    sys.exit(1)
print('every save with external data removes the old data file first')
