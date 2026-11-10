#!/usr/bin/env python3
"""Inspect the exact lite dependency. Does not claim to export Qwen3.5.

Vela3-lite's ordinary decisions delegate to the pinned d3-lite engine, which
uses non-causal full-attention layers interleaved with causal Gated DeltaNet.
An off-the-shelf causal language-model ONNX export is not equivalent.
"""
import json
from huggingface_hub import hf_hub_download
VELA_REV='c789afd6fbeaceb3c734b95b9ca3ad9772a7f434'
p=hf_hub_download('vllm-sr/vela3-lite','vela3_config.json',revision=VELA_REV)
c=json.load(open(p));base=c['base']
bp=hf_hub_download(base['repo_id'],'config.json',revision=base['revision']);cfg=json.load(open(bp))
print(json.dumps({'vela_revision':VELA_REV,'base':base,'model_type':cfg['model_type'],
                 'layer_types':cfg.get('text_config',{}).get('layer_types'),
                 'status':'not exported in this PoC; choose vela3-fast for text decisions',
                 'reason':'Must preserve the released d3 PromptEncoder attention semantics and recurrent scan, then prove parity.'},indent=2))
