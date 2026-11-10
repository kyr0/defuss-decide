#!/usr/bin/env python3
"""Store ONNX tensors outside protobuf to bound browser session-loading memory."""
import argparse,json
from pathlib import Path
import onnx

def externalize(path):
    """Move every tensor above 1 KiB of one graph into a sibling `.onnx.data` file; return its manifest entry.

    VERIFIED: Chromium failed to create a session from the embedded-weight FP32 graph (`std::bad_alloc`, see
    documentation/ADAPTER_VALIDATION.md); with external data the protobuf stays small. Tensor values are unchanged."""
    path=Path(path);m=onnx.load(str(path));data=path.with_suffix('.onnx.data');data.unlink(missing_ok=True)
    onnx.save_model(m,str(path),save_as_external_data=True,all_tensors_to_one_file=True,location=data.name,size_threshold=1024,convert_attribute=False)
    del m
    onnx.checker.check_model(str(path));m=onnx.load(str(path),load_external_data=False)
    external=sorted({v.value for t in m.graph.initializer for v in t.external_data if v.key=='location'})
    return {'file':path.name,'externalData':external,'sizeBytes':path.stat().st_size,'totalSizeBytes':path.stat().st_size+sum((path.parent/f).stat().st_size for f in external)}

def main():
    """Externalize every graph of a bundle in place. The manifest is removed first and rewritten last, so an
    interrupted run leaves no loadable half-converted bundle."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);a=p.parse_args();file=a.bundle/'manifest.json';m=json.loads(file.read_text());file.unlink()
    for key,g in m['graphs'].items():m['graphs'][key]=externalize(a.bundle/g['file'])
    file.write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
