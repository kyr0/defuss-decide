#!/usr/bin/env python3
"""Create a source ZIP with reports and a SHA-256 index; exclude checkpoints, models, dependencies and runtime output.

The index (SOURCE_SHA256.json) is written into the ZIP only: a copy in the working tree would be stale after the
next edit, and git already records the content of every committed file. VERIFIED: the former committed copy still
listed docs/ADAPTER_VALIDATION.md after the page moved to documentation/.
"""
import argparse,hashlib,json,os,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[1];exclude={'node_modules','models','dist','checkpoints','__pycache__','output','tmp','var','input','coverage','test-results','playwright-report'};files=[]
for base,dirs,names in os.walk(root):
 dirs[:]=[d for d in dirs if d not in exclude and not d.startswith('.')]
 for name in names:
  if name.startswith('.') or name.endswith(('.pyc','.onnx','.onnx.data','.safetensors','.zip','.tgz')):continue
  f=Path(base)/name
  if f.stat().st_size>10_000_000:raise RuntimeError(f'Unexpected large source artifact: {f}')
  files.append(f)
files.sort();hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};
args.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for f in files:z.write(f,root.name+'/'+str(f.relative_to(root)))
 z.writestr(root.name+'/SOURCE_SHA256.json',json.dumps(hashes,indent=2)+'\n')
with zipfile.ZipFile(args.out) as z:assert z.testzip() is None
print(json.dumps({'file':str(args.out.resolve()),'files':len(files)+1,'bytes':args.out.stat().st_size,'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()},indent=2))
