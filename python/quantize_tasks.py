#!/usr/bin/env python3
"""Quantize a validated FP32 task bundle (v0.2 adapters) to weight-only Q8 or Q4.

Q8: signed per-output-channel INT8 weights + DequantizeLinear (opset 17, portable).
Q4: signed per-output-channel INT4 weights + DequantizeLinear (opset 21 bump).
Activations stay FP32 in both modes; embeddings and non-MatMul tensors are untouched.
External data layout (.onnx + .onnx.data) is preserved for browser loading.

The output manifest says validated: false. scripts/verify-quantized.ts compares the variant's decoded spans/boxes with
the FP32 fixtures in Node.js and sets validated: true only when they are identical (scores within 0.02).
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
import onnx
import ml_dtypes
from onnx import numpy_helper as nh, helper, TensorProto


def quantize_graph(source, destination, bits, group_size=0):
    """Rewrite one graph with quantized MatMul weights; returns (count, raw bytes, packed bytes).

    Symmetric per output channel, or per block of `group_size` rows along the reduction axis, with zero points 0, so
    the standard DequantizeLinear operator reproduces `q * scale` and no custom operator is needed.

    An existing data file of the destination is removed first, and the written one must hold only what the graph
    reads: a rerun into the same folder then gives the same bundle, instead of one that carries the old weights too."""
    source, destination = Path(source), Path(destination)
    m = onnx.load(str(source))  # resolves external data next to the file
    if bits == 4:
        # INT4 DequantizeLinear needs opset 21; the converter rewrites ops whose
        # schema changed since 17 (e.g. ReduceMean axes attribute -> input).
        m = onnx.version_converter.convert_version(m, 21)
    initial = {x.name: x for x in m.graph.initializer}
    candidates = {n.input[1] for n in m.graph.node if n.op_type == 'MatMul' and len(n.input) > 1 and n.input[1] in initial}
    nodes, count, raw_bytes, packed_bytes = [], 0, 0, 0
    qmax = 127 if bits == 8 else 7
    qmin = -127 if bits == 8 else -8
    for name in sorted(candidates):
        t = initial[name]
        if t.data_type != TensorProto.FLOAT or len(t.dims) != 2:
            continue
        w = nh.to_array(t)
        rows, cols = w.shape
        grouped = group_size > 0 and rows % group_size == 0
        if grouped:
            # Blocks along the reduction axis: one scale per (row-group, column).
            wb = w.reshape(rows // group_size, group_size, cols)
            scale = np.maximum(np.max(np.abs(wb), axis=1) / qmax, np.finfo(np.float32).tiny).astype(np.float32)
            q = np.clip(np.rint(wb / scale[:, None, :]), qmin, qmax).reshape(rows, cols)
            axis, block = 0, group_size
        else:
            scale = np.maximum(np.max(np.abs(w), axis=0) / qmax, np.finfo(np.float32).tiny).astype(np.float32)
            q = np.clip(np.rint(w / scale), qmin, qmax)
            axis, block = 1, 0
        qn, sn = name + '__wq', name + '__scale'
        if bits == 8:
            qa = q.astype(np.int8)
            qt = nh.from_array(qa, qn)
            zt = nh.from_array(np.zeros(scale.shape, dtype=np.int8), name + '__zero')
        else:
            qa = q.astype(ml_dtypes.int4)
            qt = nh.from_array(qa, qn)
            zt = nh.from_array(ml_dtypes.int4(np.zeros(scale.shape, dtype=np.int8)), name + '__zero')
        m.graph.initializer.extend([qt, nh.from_array(scale, sn), zt])
        attrs = {'axis': axis}
        if block:
            attrs['block_size'] = block
        nodes.append(helper.make_node('DequantizeLinear', [qn, sn, name + '__zero'], [name], name=name + '__dequant', **attrs))
        m.graph.initializer.remove(t)
        count += 1
        raw_bytes += w.nbytes
        packed_bytes += qt.raw_data.__len__() + scale.nbytes
    old = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes + old)
    data_name = destination.name + '.data'
    # onnx.save appends to an existing external data file and points the graph at the appended copy.
    # VERIFIED: a second save doubled a 4 MB data file with offset 4,000,000 (onnx 1.23.2, 2026-11-12); the published
    # ner q8 and vision-box q8 bundles carried 596 MB and 101 MB of such unread bytes from a second run into their folder.
    (destination.parent / data_name).unlink(missing_ok=True)
    onnx.save(m, str(destination), save_as_external_data=True, all_tensors_to_one_file=True,
              location=data_name, size_threshold=1024, convert_attribute=False)
    onnx.checker.check_model(str(destination))
    data = destination.parent / data_name
    if data.exists():
        saved = onnx.load(str(destination), load_external_data=False)
        used = sum(int(e.value) for t in saved.graph.initializer for e in t.external_data if e.key == 'length')
        if data.stat().st_size > used * 1.01:
            raise RuntimeError(f'{data} holds {data.stat().st_size} bytes, the graph reads {used}')
    return count, raw_bytes, packed_bytes


def main():
    """Quantize every graph of a validated FP32 bundle (or `--only` one) and write an unvalidated manifest that names
    the gate command; scripts/verify-quantized.ts decides whether the variant becomes loadable."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bits', type=int, choices=[4, 8], required=True)
    ap.add_argument('--group-size', type=int, default=0,
                    help='Block-wise scales along the reduction axis (0 = per-channel)')
    ap.add_argument('--only', choices=['task', 'vision'], default=None,
                    help='Quantize only this graph; copy the other FP32 graph unchanged')
    a = ap.parse_args()
    src, out = a.source.resolve(), a.out.resolve()
    if src == out:
        ap.error('Use a new output directory')
    m = json.loads((src / 'manifest.json').read_text())
    if not m['validated'] or m['precision'] != 'fp32':
        raise ValueError('Expected validated FP32 bundle')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').unlink(missing_ok=True)
    for p in src.iterdir():
        if p.name in ['tokenizer.json', 'tokenizer_config.json', 'fixtures.json', 'parity-python.json'] or p.suffix == '.png' or p.name.startswith('LICENSE'):
            shutil.copyfile(p, out / p.name)
    total = {'count': 0, 'raw': 0, 'packed': 0}
    for name, g in m['graphs'].items():
        if a.only and name != a.only:
            for suffix in ['', '.data']:
                f = g['file'] + suffix
                if (src / f).exists():
                    shutil.copyfile(src / f, out / f)
            continue
        print(f"Quantizing {name} -> Q{a.bits} g{a.group_size}", flush=True)
        count, raw, packed = quantize_graph(src / g['file'], out / g['file'], a.bits, a.group_size)
        total['count'] += count; total['raw'] += raw; total['packed'] += packed
        g['sizeBytes'] = (out / g['file']).stat().st_size
        # VERIFIED: onnx.save writes the .data file only when some tensor exceeds size_threshold (reproduced on a
        # fixture-bundle copy, 2026-11-02); a graph of small tensors stays self-contained and lists no external data.
        data = out / (g['file'] + '.data')
        g['externalData'] = [data.name] if data.exists() else []
        g['totalSizeBytes'] = g['sizeBytes'] + (data.stat().st_size if data.exists() else 0)
    m['precision'] = f'w{a.bits}a32'
    m['quantization'] = (f'weight-only signed INT{a.bits} MatMul + DequantizeLinear, ' +
                         (f'group size {a.group_size} along reduction axis' if a.group_size else 'per-output-channel') +
                         (f'; only graph "{a.only}" quantized' if a.only else ''))
    # Relative, so a shipped manifest never carries a machine's home directory.
    m['quantizedFrom'] = os.path.relpath(src, out)
    # Not loadable yet: the decoded-output gate runs in JavaScript (scripts/verify-quantized.ts), and only a pass
    # there may flip this flag. A variant that fails stays unloadable instead of relying on someone deleting it.
    m['validated'] = False
    m['pendingVerification'] = 'node scripts/verify-quantized.ts ' + os.path.relpath(out)
    dump = (out / 'manifest.json')
    dump.write_text(json.dumps(m, indent=1) + '\n')
    print(json.dumps({'out': str(out), 'matmuls': total['count'],
                      'weightsMB': round(total['raw'] / 1e6, 1), 'packedMB': round(total['packed'] / 1e6, 1)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
