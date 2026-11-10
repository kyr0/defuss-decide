#!/usr/bin/env python3
"""Which backbone matrices make the q4 text decision head drift? Restore one layer, or one matrix type, to FP32.

    uv run python python/quant_sensitivity.py --source checkpoints/fast --inputs <out>/backbone/fast/q4/inputs.json \
        --out reports/quantization-decision-sensitivity.json

Quantizes all 168 backbone matrices with GPTQ int4 (groups of 32, as python/quantize_gptq.py; Hessians from the decision
inputs only), then restores one layer (24 runs) or one matrix type in every layer (7 runs) to FP32 and measures the
decision head against FP32 on the held-out inputs, the 56 labelled text cases of tests/fixtures/decide-cases.json and
the bundle's recorded fixtures. The run whose restore removes the most drift names the matrices to keep at int8.
VERIFIED (reports/quantization-decision-sensitivity.json, 2026-11-08): no layer or matrix type carries the drift; restoring
any one to FP32 removed at most 13 % of the held-out drift, and none kept the `noul` fixture's FP32 option. Layers 16
to 23 change the answers only in the fifth decimal.
HYPOTHESIS: restoring a matrix to FP32 approximates keeping it int8 per channel, because int8 per channel on all 168
matrices moved no decision on 131 held-out inputs (tmp/quant-probe, 2026-11-08); falsifier: the gate of an export with
`python/quantize_gptq.py --int8` for the chosen matrices.
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memguard import guard
from quantize_gptq import REPO, Hessians, check_sources, decision_inputs, decision_model, dequant4, fast_args, gptq, linears, lines, log, paragraphs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, help='inputs.json of a python/quantize_gptq.py run; default: the inputs it generates')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--min-free-gb', type=float, default=8.0, help='Stop the job when the machine has less memory available (python/memguard.py)')
    a = ap.parse_args(); torch.set_num_threads(6)
    guard(a.min_free_gb)
    root = a.source.resolve(); check_sources(root); sys.path.insert(0, str(root))
    if a.inputs: inputs = json.loads(a.inputs.read_text())['decision']
    else:
        calib, held = paragraphs(); inputs = {'calibration': decision_inputs(calib, 1, 80), 'heldOut': decision_inputs(held, 3, 80)}
    d = decision_model(root)
    H = Hessians()
    def run():
        with torch.no_grad():
            for s, q in inputs['calibration']:
                rec = d.encode(s, q); ids = torch.tensor([rec['ids']]); d.encoder(input_ids=ids, attention_mask=torch.ones_like(ids))
    H.collect(list(linears(d)), run)
    fp32 = {name: mod.weight.detach().clone() for name, mod in linears(d)}
    q4 = {name: torch.from_numpy(dequant4(*gptq(w, H[name]))) for name, w in fp32.items()}
    log('INFO', 'gptq matrices=168')

    labelled = [c for c in json.loads((REPO / 'tests/fixtures/decide-cases.json').read_text())['cases'] if 'images' not in c]
    def question(c):
        if not c.get('options'): return {'type': 'noul', 'instructions': c['instructions']}
        o = c['options']; return {'type': 'choice', 'instructions': c['instructions'], 'criteria': o if isinstance(o, dict) else {x: None for x in o}}
    fixtures = json.loads((REPO / 'models/decision/fast/fixtures.json').read_text())
    sets = {'heldOut': [d.encode(s, q) for s, q in inputs['heldOut']],
            'labelled': [d.encode(lines(c['state']), question(c)) for c in labelled],
            'fixtures': [{'ids': f['ids'], 'pool': f['pool'], 'opt_pos': f['optionPositions'], 'names': f['names']} for f in fixtures]}
    from vela_onnx.portable import FastDecision
    wrapper = FastDecision(d).eval()
    def probs():
        with torch.no_grad(): return {k: [torch.softmax(wrapper(*fast_args(r)), -1).numpy() for r in recs] for k, recs in sets.items()}
    def load(restore):
        with torch.no_grad():
            for name, mod in linears(d): mod.weight.copy_(fp32[name] if restore(name) else q4[name])
    load(lambda n: True); ref = probs()
    def metrics(label):
        p = probs(); row = {'run': label}
        for k in sets:
            shifts = [float(np.abs(x - y).max()) for x, y in zip(p[k], ref[k])]
            row[k] = {'flips': sum(int(x.argmax() != y.argmax()) for x, y in zip(p[k], ref[k])), 'meanShift': round(float(np.mean(shifts)), 4), 'maxShift': round(float(np.max(shifts)), 4)}
        row['labelled']['correct'] = sum(int(r['names'][int(x.argmax())] == c['expected']) for x, r, c in zip(p['labelled'], sets['labelled'], labelled))
        log('INFO', json.dumps(row)); return row
    rows = []
    load(lambda n: False); rows.append(metrics('all int4'))
    for layer in range(24):
        load(lambda n: n.startswith(f'encoder.layers.{layer}.')); rows.append(metrics(f'layer {layer} FP32'))
    for kind in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
        load(lambda n: n.endswith(f'.{kind}_proj')); rows.append(metrics(f'{kind}_proj FP32 in every layer'))
    base = rows[0]['heldOut']['meanShift']
    for r in rows[1:]: r['heldOutMeanShiftRemoved'] = round(1 - r['heldOut']['meanShift'] / base, 3)
    fp32_correct = sum(int(r['names'][int(x.argmax())] == c['expected']) for x, r, c in zip(ref['labelled'], sets['labelled'], labelled))
    a.out.write_text(json.dumps({'date': time.strftime('%Y-%m-%d'), 'tool': 'python/quant_sensitivity.py', 'fp32LabelledCorrect': fp32_correct,
                                 'counts': {k: len(v) for k, v in sets.items()}, 'runs': rows}, indent=1) + '\n')
    log('INFO', f'wrote {a.out}')


if __name__ == '__main__':
    sys.exit(main())
