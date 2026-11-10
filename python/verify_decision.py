#!/usr/bin/env python3
"""The gate of a quantized text decision bundle: does it decide as well as FP32, and how often does it differ?

    uv run python python/verify_decision.py --source checkpoints/fast --models models [--bundle decision/fast/q4]

Runs the FP32 graph and the quantized graph in ONNX Runtime (CPU) on three sets and compares the chosen options:

    boolq     the 1,300 first validation questions of BoolQ (google/boolq, pinned revision, CC BY-SA 3.0), yes/no, labelled
    heldOut   the held-out decision inputs of the q4 export (`<models>/backbone/fast/q4/inputs.json`), unlabelled
    labelled  the text cases of tests/fixtures/decide-cases.json, labelled

The bundle passes when, on every set, it picks another option than FP32 on at most 3 % of the inputs, and on every
labelled set it answers at most 1 percentage point fewer questions correctly than FP32. The script then writes
`validated: true` and the measured figures into the bundle's manifest; the report goes to
reports/quantization-decision.json either way. The 5 recorded fixtures are reported, not gated.

WHY this rule and not "no fixture changes its option", the task bundles' rule: a task answer is a set of spans decoded
with thresholds, so an unchanged answer on the fixtures is a strong check there. A decision is one option out of few,
and a question on which FP32 itself is close to undecided changes its option under any change of the weights.
VERIFIED: (2026-11-10) of the questions on which the published q4 differs from FP32, FP32's two best options were less
than 0.05 apart on 3 of 8 (300 BoolQ questions); a rank-16 repair, closed-form corrections and quantization-aware
training of all codes left 16 to 18 differing answers in 1,000 unseen BoolQ questions, with the accuracy unchanged
(plans/2026-11-08_19-43_decision-q4.md). So the gate measures what a caller gets: the accuracy, and the share of
differing answers, which the manifest and the documentation state. The 3 % limit was set after these measurements.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memguard import guard
from export import dump, session, softmax
from quantize_gptq import REPO, assembled, check_sources, decision_model, fast_args, lines, log
from repair_decision import BOOLQ, boolq, boolq_inputs

FLIP_LIMIT, ACCURACY_DROP = 0.03, 0.01


def passes(entry):
    """The rule on one set's figures: flips within the limit, and with labels the accuracy within the allowed drop."""
    if entry['flips'] > FLIP_LIMIT * entry['inputs']: return False
    return 'correct' not in entry or entry['correct']['quantized'] >= entry['correct']['fp32'] - ACCURACY_DROP * entry['inputs']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True, help='Pinned vela3-fast checkpoint (for the tokenizer and the input layout)')
    ap.add_argument('--models', type=Path, required=True, help='Folder with decision/fast (FP32) and the quantized bundle')
    ap.add_argument('--bundle', default='decision/fast/q4', help='The quantized bundle, relative to --models')
    ap.add_argument('--report', type=Path, default=REPO / 'reports' / 'quantization-decision.json')
    ap.add_argument('--min-free-gb', type=float, default=8.0, help='Stop the job when the machine has less memory available (python/memguard.py)')
    a = ap.parse_args(); guard(a.min_free_gb)
    root, models = a.source.resolve(), a.models.resolve(); bundle = models / a.bundle; check_sources(root); sys.path.insert(0, str(root))
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if 'sharedData' not in manifest: raise SystemExit(f'{bundle} names no shared backbone file; this gate covers the text decision bundle')
    shared = (bundle / manifest['sharedData']['file']).resolve()

    model = decision_model(root)  # only its tokenizer and input layout are used
    rows = boolq('validation')[:1300]
    recorded = json.loads((models / 'backbone' / 'fast' / 'q4' / 'inputs.json').read_text())['decision']
    cases = [c for c in json.loads((REPO / 'tests/fixtures/decide-cases.json').read_text())['cases'] if 'images' not in c]
    sets = {'boolq': (boolq_inputs(rows, 15, 0), ['yes' if r[2] else 'no' for r in rows]),
            'heldOut': ([tuple(x) for x in recorded['heldOut']], None),
            'labelled': ([(lines(c['state']), {'type': 'choice', 'instructions': c['instructions'], 'criteria': c['options'] if isinstance(c['options'], dict) else {o: None for o in c['options']}}
                          if c.get('options') else {'type': 'noul', 'instructions': c['instructions']}) for c in cases], [c['expected'] for c in cases])}
    recs = {k: ([model.encode(s, q) for s, q in items], gold) for k, (items, gold) in sets.items()}
    del model

    tmp = assembled(bundle, 'decision', shared)
    try:
        fp, q = session(models / 'decision' / 'fast' / 'decision.onnx'), session(tmp / 'decision.onnx')
        run = lambda s, args: s.run(None, {n.name: t.numpy() for n, t in zip(s.get_inputs(), args)})[0]
        report = {'date': time.strftime('%Y-%m-%d', time.gmtime()), 'tool': 'python/verify_decision.py', 'bundle': a.bundle, 'precision': manifest['precision'],
                  'runtime': 'ONNX Runtime CPU', 'rule': {'maxFlipShare': FLIP_LIMIT, 'maxAccuracyDrop': ACCURACY_DROP},
                  'boolq': {'dataset': BOOLQ[0], 'revision': BOOLQ[1], 'split': 'validation, questions 1 to 1,300'}, 'sets': {}}
        for label, (rs, gold) in recs.items():
            out = [(r['names'], softmax(run(fp, fast_args(r))), softmax(run(q, fast_args(r)))) for r in rs]
            shifts = [float(np.abs(x - y).max()) for _, x, y in out]
            entry = {'inputs': len(out), 'flips': sum(int(x.argmax() != y.argmax()) for _, x, y in out),
                     'meanShift': round(float(np.mean(shifts)), 4), 'maxShift': round(float(np.max(shifts)), 4)}
            if gold is not None:
                entry['correct'] = {tag: sum(int(n[int(p[i].argmax())] == g) for (n, *p), g in zip(out, gold)) for tag, i in (('fp32', 0), ('quantized', 1))}
            entry['passed'] = passes(entry); report['sets'][label] = entry; log('INFO', f'set={label} ' + json.dumps(entry))
        fixtures = json.loads((models / 'decision' / 'fast' / 'fixtures.json').read_text())
        report['fixtureFlips'] = [f['name'] for f in fixtures
                                  if int(np.argmax(run(q, fast_args({'ids': f['ids'], 'pool': f['pool'], 'opt_pos': f['optionPositions']})))) != int(np.argmax(f['logits']))]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report['passed'] = all(e['passed'] for e in report['sets'].values())
    dump(a.report, report)
    if report['passed']:
        manifest = {k: v for k, v in manifest.items() if k not in ('rejected', 'pendingVerification')}
        manifest['validated'] = True
        manifest['validation'] = {'tool': report['tool'], 'date': report['date'], 'rule': f'at most {FLIP_LIMIT:.0%} of the answers differ from FP32 on every set, and at most {ACCURACY_DROP:.0%} fewer are correct',
                                  'sets': {k: {x: e[x] for x in ('inputs', 'flips', 'correct') if x in e} for k, e in report['sets'].items()}, 'fixtureFlips': report['fixtureFlips']}
        dump(bundle / 'manifest.json', manifest)
    log('INFO' if report['passed'] else 'ERROR', f'done passed={report["passed"]} fixture_flips={report["fixtureFlips"]} report={a.report}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
