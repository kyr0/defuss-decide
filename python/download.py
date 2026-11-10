#!/usr/bin/env python3
"""Download pinned decision or task adapter checkpoints."""
import argparse,json
from pathlib import Path
from huggingface_hub import snapshot_download
PINS={'fast':('vllm-sr/vela3-fast','fa60e3aa41132c60bf05a84cabfdced39cb6d4f7'),
      'vision':('vllm-sr/vela3-fast-vision','c1f651cafddca43860a855ee434a35697949bce1')}
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=PINS,required=True);p.add_argument('--out',required=True);p.add_argument('--tasks',action='store_true',help='Include all span/OCR/region adapters');a=p.parse_args()
repo,rev=PINS[a.model];files=list(json.loads((Path(__file__).parent/'source-lock.json').read_text())[a.model])+['LICENSE','LICENSE-APACHE']
if a.tasks: files+=list(json.loads((Path(__file__).parent/'task-source-lock.json').read_text())[a.model])
print(snapshot_download(repo,revision=rev,allow_patterns=files,local_dir=a.out))
