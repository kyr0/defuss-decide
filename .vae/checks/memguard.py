#!/usr/bin/env python3
"""Every Python job under python/ that loads models (imports torch or onnxruntime and has a main) starts the memory
guard, so no export, quantization or repair run can push the machine into swap unnoticed (python/memguard.py)."""
import re, sys
from pathlib import Path

bad = []
for f in sorted(Path('python').glob('*.py')):
    s = f.read_text()
    if f.name == 'memguard.py' or 'def main(' not in s or not re.search(r'^import (torch|onnxruntime)|^from (torch|onnxruntime)', s, re.M): continue
    if not re.search(r'^\s*\w*\.?parse_args\(\);?\s*guard\(|guard\(a(rgs)?\.min_free_gb\)', s, re.M): bad.append(str(f))
if bad:
    print('FIX: call guard(<args>.min_free_gb) from python/memguard.py right after parse_args() in: ' + ', '.join(bad))
    sys.exit(1)
print('memory guard wired into every model job under python/')
