#!/usr/bin/env python3
"""Reproduce the q4 vision bundles: GPTQ int4 on every MatMul of the image encoder and the task or decision graph.

    uv run python python/quantize_gptq_vision.py --source checkpoints/vision --models models --out <root> [--bundles vision-span,vision-ocr]

Writes `<out>/vela3-fast-onnx/<task>/q4/` for vision-span, vision-ocr and vision-box and `<out>/decision/vision/q4/`.
VERIFIED: the content hashes of the FP32 graphs (2026-11-09) show that vision-span and vision-ocr hold the same image encoder (73
matrices), 140 of their 168 text matrices and the same token table; vision-box (backbone v2) and decision/vision
(backbone h2) share no matrix with them or with each other. So vision-span and vision-ocr read two shared files,
`<out>/backbone/vision/q4/image.onnx.data` (image encoder) and `text.onnx.data` (text backbone and token table), one per
graph, so each ONNX Runtime session downloads only the weights it runs; vision-box and decision/vision stay
self-contained. The runtime format is python/quantize_gptq.py's: `MatMulNBits` (4 bits, groups of 32, fp16 scales,
4-bit zero points), the token table packed with exact rows from id 49152, LoRA matrices int8 per channel.

WHY Hessians from ONNX Runtime and not PyTorch hooks as for the text bundles: each vision family has its own
upstream model class and wrapper; exposing every MatMul input of the FP32 graph as an extra output gives the same
X^T X from the exported graphs themselves, for all four bundles with one code path.

The calibration and held-out inputs are generated here, deterministic per seed: rendered documents (DejaVu Sans),
shapes on plain and noisy backgrounds, descriptions with wrong details, OCR texts with corrupted characters, region
instructions and decision questions. Every q4 graph must reproduce the FP32 graph with the same dequantized weights
(ONNX Runtime, optimization level basic). decision/vision must also keep FP32's option on its 4 recorded fixtures;
task bundles stay unvalidated until `scripts/verify-quantized.ts` passes, vision-box until `scripts/verify-accuracy.ts` does.
"""
import argparse, hashlib, json, os, random, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper as nh
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memguard import guard
from export import compare, dump, softmax, vision_args
from quantize_gptq import (GROUP, MARKER_BASE, REPO, SHARED, DataFile, dequant4, exact_session, gptq, int8, key, log, rewrite, rtn4, sha256,
                          tree_relpath)

FAMILIES = {'raw': ['vision-span', 'vision-ocr'], 'v2': ['vision-box'], 'h2': ['decision/vision']}
COLORS = {'red': (220, 30, 30), 'green': (30, 160, 60), 'blue': (30, 60, 220), 'yellow': (240, 210, 30), 'black': (20, 20, 20), 'white': (250, 250, 250), 'orange': (240, 130, 20), 'purple': (130, 50, 170)}
SHAPES = ['square', 'circle', 'triangle']
WORDS = ('invoice total amount due date customer order number shipping address payment received balance account '
         'reference item quantity price tax subtotal delivery note signature company street city phone email '
         'report summary meeting agenda project status budget review approved pending invoice receipt').split()


def src_dir(models, b): return models / 'decision' / 'vision' if b == 'decision/vision' else models / 'vela3-fast-onnx' / b / 'fp32'
def dst_dir(out, b): return out / 'decision' / 'vision' / 'q4' if b == 'decision/vision' else out / 'vela3-fast-onnx' / b / 'q4'


# ---- generated inputs --------------------------------------------------------------------------------------------
def document(r):
    """A rendered document image and its text."""
    from PIL import Image, ImageDraw, ImageFont
    w, h = r.choice([(512, 512), (768, 1024), (1024, 768), (640, 480)])
    im = Image.new('RGB', (w, h), r.choice([(255, 255, 255), (245, 242, 230), (230, 230, 230)])); d = ImageDraw.Draw(im)
    size = r.randint(18, 40); font = ImageFont.truetype(str(REPO / 'DejaVuSans.ttf'), size)
    lines = []
    for _ in range(r.randint(2, 7)):
        k = r.random()
        lines.append(f'{r.choice(WORDS).title()}: {r.randint(1, 9999)}.{r.randint(0, 99):02d}' if k < 0.4 else
                     f'{r.choice(WORDS).title()} {r.choice(WORDS)} {r.randint(1, 31)}/{r.randint(1, 12)}/20{r.randint(10, 29)}' if k < 0.6 else
                     ' '.join(r.choice(WORDS) for _ in range(r.randint(3, 6))).capitalize())
    text = '\n'.join(lines); d.multiline_text((r.randint(10, 60), r.randint(10, 80)), text, font=font, fill=(r.randint(0, 60),) * 3, spacing=size // 3)
    return im, text


def shapes(r):
    """An image with one to three coloured shapes; returns the image and (colour, shape) of each."""
    from PIL import Image, ImageDraw
    w, h = r.choice([(512, 512), (800, 600), (600, 800), (1024, 1024)]); bg = r.choice(['white', 'black', 'yellow', 'green', 'blue'])
    im = Image.new('RGB', (w, h), COLORS[bg])
    if r.random() < 0.4:
        noise = np.random.default_rng(r.randint(0, 1 << 30)).integers(-40, 40, (h, w, 3))
        im = Image.fromarray(np.clip(np.asarray(im, np.int32) + noise, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im); items = []
    for _ in range(r.randint(1, 3)):
        c, s = r.choice([k for k in COLORS if k != bg]), r.choice(SHAPES); x, y, e = r.randint(0, w - 120), r.randint(0, h - 120), r.randint(60, min(w, h) // 2)
        box = (x, y, min(w - 1, x + e), min(h - 1, y + e))
        if s == 'square': d.rectangle(box, fill=COLORS[c])
        elif s == 'circle': d.ellipse(box, fill=COLORS[c])
        else: d.polygon([(box[0], box[3]), (box[2], box[3]), ((box[0] + box[2]) // 2, box[1])], fill=COLORS[c])
        items.append((c, s))
    return im, items


def corrupt(r, text):
    """An OCR reading of `text` with a few wrong characters."""
    chars = list(text)
    for _ in range(r.randint(0, 4)):
        i = r.randrange(len(chars))
        if chars[i].isalnum(): chars[i] = r.choice('0123456789') if chars[i].isdigit() else r.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(chars)


def examples(bundle, proc, seed, n):
    """n upstream examples of one bundle (the dicts uvb_core builds) with the image list they came from."""
    import uvb_core as C
    r = random.Random(seed); out = []
    while len(out) < n:
        try:
            if bundle == 'vision-ocr':
                im, text = document(r)
                out.append(C.span_example(proc, im, 'User request: Check this OCR extraction against the scanned document.', corrupt(r, text)))
            elif bundle == 'vision-span':
                if r.random() < 0.7:
                    im, items = shapes(r); c, s = items[0]; c = c if r.random() < 0.5 else r.choice(list(COLORS))
                    answer = f'The image shows a {c} {s}' + (f' and a {items[1][0]} {items[1][1]}' if len(items) > 1 else '') + f' on a plain background.'
                else:
                    im, text = document(r); answer = f'The document is a {r.choice(["receipt", "letter", "report"])} that mentions {r.choice(WORDS)} and {r.choice(WORDS)}.'
                out.append(C.span_example(proc, im, 'User request: ' + r.choice(['Describe the picture.', 'What is shown?', 'Summarize the image.']), answer))
            elif bundle == 'vision-box':
                if r.random() < 0.7:
                    im, items = shapes(r); c, s = r.choice(items); q = f'the {c} {s}'
                else:
                    im, text = document(r); q = 'the word ' + r.choice(text.split())
                out.append(C.box_example(proc, im, q))
            else:
                k = r.random()
                if k < 0.6:
                    im, items = shapes(r); imgs = [im]
                    if r.random() < 0.5: q, opts = 'What is the dominant color?', r.sample(list(COLORS), 3)
                    else: q, opts = f'Is there a {r.choice(list(COLORS))} {r.choice(SHAPES)} in the image?', ['No / false', 'Yes / true']
                elif k < 0.85:
                    im, _ = document(r); imgs = [im]
                    q, opts = r.choice([('What kind of document is this?', ['invoice', 'letter', 'report']), ('Does the document show a total amount?', ['No / false', 'Yes / true'])])
                else:
                    imgs = [shapes(r)[0], shapes(r)[0]]; q, opts = 'Do both images show the same shape?', ['No / false', 'Yes / true']
                ex = C.decision_example(proc, imgs, '', q, opts); out.append(ex)
        except ValueError as e:
            log('WARN', f'skipped example error={e}')
    return out


# ---- ONNX Runtime helpers ----------------------------------------------------------------------------------------
def quantizable(m):
    """{ONNX initializer name: array} of the MatMul weights (2-D initializers) of a loaded FP32 model, and the token table."""
    init = {i.name: i for i in m.graph.initializer}; mats, table = {}, None
    for node in m.graph.node:
        if node.op_type == 'MatMul' and len(node.input) > 1 and node.input[1] in init and len(init[node.input[1]].dims) == 2:
            mats[node.input[1]] = node.input[0]
        if node.op_type == 'Gather' and node.input[0] in init and len(init[node.input[0]].dims) == 2 and init[node.input[0]].dims[0] > MARKER_BASE:
            table = node.input[0]
    return mats, table


def capture_session(m, inputs_of):
    """An ONNX Runtime session of model `m` that also returns every MatMul input named in `inputs_of`."""
    import onnxruntime as ort
    m = onnx.ModelProto.FromString(m.SerializeToString())
    have = {o.name for o in m.graph.output}
    for name in sorted(set(inputs_of) - have): m.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC; so.intra_op_num_threads = 6
    return ort.InferenceSession(m.SerializeToString(), sess_options=so, providers=['CPUExecutionProvider'])


def feed(s, args): return {i.name: np.asarray(a) for i, a in zip(s.get_inputs(), args)}


def task_args(bundle, ex, feats, image_id):
    if bundle == 'decision/vision': return [t.numpy() if hasattr(t, 'numpy') else t for t in vision_args(ex, torch.from_numpy(feats), image_id)]
    idx, k = [], 0
    for t in ex['input_ids']:
        if t == image_id: k += 1; idx.append(k)
        else: idx.append(0)
    pos = ex['img_pos'] if bundle == 'vision-box' else ex['word_pos']
    return [np.array([ex['input_ids']], np.int64), np.ones((1, len(ex['input_ids'])), np.int64), feats.astype(np.float32), np.array([idx], np.int64), np.asarray(pos, np.int64)]


def tiles(ex): return [] if ex.get('pv') is None else [t[None].numpy() for t in ex['pv']]


def features(vs, ex):
    out = [vs.run(['image_features'], {'pixel_values': t})[0] for t in tiles(ex)]
    return np.concatenate(out).reshape(-1, 768).astype(np.float32) if out else np.zeros((1, 768), np.float32)


def text_graph(bundle): return 'decision' if bundle == 'decision/vision' else 'task'


# ---- main --------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True, help='Pinned vela3-fast-vision checkpoint (python/download.py --model vision --tasks)')
    ap.add_argument('--models', type=Path, required=True, help='Folder with the validated FP32 vision bundles')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bundles', default=','.join(b for f in FAMILIES.values() for b in f))
    ap.add_argument('--calibration', type=int, default=48, help='Generated inputs per bundle for the Hessians')
    ap.add_argument('--held-out', type=int, default=24)
    ap.add_argument('--min-free-gb', type=float, default=8.0, help='Stop the job when the machine has less memory available (python/memguard.py)')
    a = ap.parse_args(); torch.set_num_threads(6)
    guard(a.min_free_gb)
    root, models, out = a.source.resolve(), a.models.resolve(), a.out.resolve(); wanted = [b for b in a.bundles.split(',') if b]
    sys.path.insert(0, str(root))
    import uvb_core as C
    proc = C.Proc.get(root, 1024)
    report = {'date': time.strftime('%Y-%m-%d'), 'tool': 'python/quantize_gptq_vision.py', 'bundles': {}, 'shared': {}}
    for family, members in FAMILIES.items():
        members = [b for b in members if b in wanted]
        if not members: continue
        log('INFO', f'family={family} bundles={members}')
        models_fp = {(b, g): onnx.load(str(src_dir(models, b) / f'{g}.onnx')) for b in members for g in ('vision', text_graph(b))}
        man = {b: json.loads((src_dir(models, b) / 'manifest.json').read_text()) for b in members}
        # 1. Hessians: X^T X / n per MatMul input, per bundle; a weight's Hessian is the mean over the bundles holding it.
        # One float32 array per input (q, k and v read the same one), not per weight, and no copies of the weights: the
        # per-weight float64 arrays took 4 GB for decision/vision (python/memguard.py); gptq computes in float32 anyway.
        Hx, sources, names, tables = {}, {}, {}, {}
        inputs = {b: {'calibration': examples(b, proc, 100 + i, a.calibration), 'heldOut': examples(b, proc, 200 + i, a.held_out)} for i, b in enumerate(members)}
        for b in members:
            for g in ('vision', text_graph(b)):
                m = models_fp[(b, g)]; mats, table = quantizable(m); arrays = {i.name: i for i in m.graph.initializer}
                for w, x in mats.items():
                    k = key(nh.to_array(arrays[w])); names.setdefault(k, (b, g, w)); sources.setdefault(k, []).append((b, g, x))
                if table: tables[(b, g)] = nh.to_array(arrays[table])
            vs = capture_session(models_fp[(b, 'vision')], [])
            for g in ('vision', text_graph(b)):
                m = models_fp[(b, g)]; mats, _ = quantizable(m); xs = sorted(set(mats.values()))
                s = capture_session(m, xs); acc, cnt = {}, {}
                for ex in inputs[b]['calibration']:
                    runs = [[t] for t in tiles(ex)] if g == 'vision' else [task_args(b, ex, features(vs, ex), man[b]['imageId'])]
                    for args in runs:
                        res = dict(zip([o.name for o in s.get_outputs()], s.run(None, feed(s, args))))
                        for x in xs:
                            v = res[x].reshape(-1, res[x].shape[-1]).astype(np.float64); acc[x] = acc.get(x, 0) + v.T @ v; cnt[x] = cnt.get(x, 0) + v.shape[0]
                for x in xs: Hx[(b, g, x)] = (acc.pop(x) / cnt[x]).astype(np.float32)
                del s
                log('INFO', f'hessians bundle={b} graph={g} matrices={len(mats)} inputs={len(inputs[b]["calibration"])}')
        # 2. Quantize: LoRA matrices (a dimension equal to the rank) int8 per channel, every other matrix GPTQ int4.
        consumers = {}
        for (b, g), m in models_fp.items():
            mats, _ = quantizable(m); arrays = {i.name: i for i in m.graph.initializer}
            for w in mats: consumers.setdefault(key(nh.to_array(arrays[w])), set()).add(b)
        quant = {}
        for k, (b, g, wname) in names.items():
            w = nh.to_array(next(i for i in models_fp[(b, g)].graph.initializer if i.name == wname))
            rank = man[b].get('taskConfig', {}).get('lora_rank', 0); shared = len(consumers[k]) > 1
            if rank and rank in w.shape: q, s = int8(torch.from_numpy(w.T.copy())); quant[k] = ('int8', q, s, shared, f'm.{k[:16]}')
            else: quant[k] = ('int4', *gptq(torch.from_numpy(w.T.copy()), torch.from_numpy(sum(Hx[src] for src in sources[k]) / len(sources[k]))), shared, f'm.{k[:16]}')
        del Hx
        log('INFO', f'quantized family={family} matrices={len(quant)} shared={sum(len(c) > 1 for c in consumers.values())}')
        table = next(iter(tables.values()))
        if any(not np.array_equal(t, table) for t in tables.values()): raise ValueError(f'family {family}: the bundles hold different token tables')
        q, s, z = rtn4(torch.from_numpy(table[:MARKER_BASE]))
        fq = np.concatenate([q, np.zeros((table.shape[0] - MARKER_BASE, q.shape[1]), np.uint8)]); fs = np.concatenate([s, np.ones((table.shape[0] - MARKER_BASE, s.shape[1]), np.float16)])
        fz = np.concatenate([z, np.zeros((table.shape[0] - MARKER_BASE, z.shape[1]), np.uint8)]); deq = dequant4(fq, fs, fz)
        resid = np.zeros((1 + table.shape[0] - MARKER_BASE, table.shape[1]), np.float32); resid[1:] = table[MARKER_BASE:] - deq[MARKER_BASE:]
        quant['embed'] = ('embed', fq, fs, fz, resid)
        # 3. Rewrite; a family with more than one bundle writes one shared file per graph kind.
        shared_dir = out / 'backbone' / 'vision' / 'q4'
        shared = None
        if len(members) > 1:
            shared_dir.mkdir(parents=True, exist_ok=True); shared = {kind: DataFile(shared_dir / f'{kind}.onnx.data', SHARED) for kind in ('image', 'text')}
        counts = {}
        for b in members:
            d = dst_dir(out, b); d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.suffix in ('.onnx', '.data') or f.name == 'manifest.json': f.unlink()
            for g in ('vision', text_graph(b)):
                kind = 'image' if g == 'vision' else 'text'
                counts[(b, g)] = rewrite(src_dir(models, b) / f'{g}.onnx', d, g, quant, shared[kind] if shared else None)
                log('INFO', f'rewrote bundle={b} graph={g} ' + json.dumps(counts[(b, g)]))
        if shared:
            for s_ in shared.values(): s_.close()
        # 4. Check every graph against the FP32 graph with the same dequantized weights, and measure the drift.
        import onnxruntime as ort
        def reference(m):
            r = onnx.ModelProto.FromString(m.SerializeToString())
            for t in r.graph.initializer:
                arr = nh.to_array(t) if t.data_type == TensorProto.FLOAT and len(t.dims) == 2 else None
                if arr is None: continue
                q_ = quant.get(key(arr))
                if q_ and q_[0] == 'int4': t.CopyFrom(nh.from_array(dequant4(q_[1], q_[2], q_[3]).T.astype(np.float32).copy(), t.name))
                elif q_ and q_[0] == 'int8': t.CopyFrom(nh.from_array((q_[1].astype(np.float32) * q_[2][:, None]).T.copy(), t.name))
                elif arr.shape[0] > MARKER_BASE: t.CopyFrom(nh.from_array((deq + np.pad(resid[1:], ((MARKER_BASE, 0), (0, 0)))).astype(np.float32), t.name))
            so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC; so.intra_op_num_threads = 6
            return ort.InferenceSession(r.SerializeToString(), sess_options=so, providers=['CPUExecutionProvider'])
        def plain(m):
            so = ort.SessionOptions(); so.intra_op_num_threads = 6
            return ort.InferenceSession(m.SerializeToString(), sess_options=so, providers=['CPUExecutionProvider'])
        for b in members:
            d = dst_dir(out, b); tmp = Path(tempfile.mkdtemp(prefix='q4v-', dir=REPO / 'tmp'))
            try:
                for g in ('vision', text_graph(b)):
                    for f in (f'{g}.onnx', f'{g}.onnx.data'): os.link(d / f, tmp / f)
                sess = {}
                for g in ('vision', text_graph(b)):
                    if shared:
                        p = tmp / g; p.mkdir(); [os.link(tmp / f, p / f) for f in (f'{g}.onnx', f'{g}.onnx.data')]
                        os.link(shared_dir / f'{"image" if g == "vision" else "text"}.onnx.data', p / SHARED); path = p / f'{g}.onnx'
                    else: path = tmp / f'{g}.onnx'
                    sess[g] = (exact_session(path), plain(models_fp[(b, g)]), reference(models_fp[(b, g)]), ort.InferenceSession(str(path), providers=['CPUExecutionProvider']))
                tg = text_graph(b); checks, drift = [], []
                for i, ex in enumerate(inputs[b]['heldOut']):
                    q4x, fp, ref, q4 = sess['vision']
                    feats_ref = features(fp, ex); feats_q = features(q4, ex)
                    if i < 2 and tiles(ex): checks.append(compare(ref.run(None, {'pixel_values': tiles(ex)[0]})[0], q4x.run(None, {'pixel_values': tiles(ex)[0]})[0], f'{b}/vision/{i}/q4-vs-dequantized', atol=6e-3))
                    q4x, fp, ref, q4 = sess[tg]
                    a_fp, a_q = task_args(b, ex, feats_ref, man[b]['imageId']), task_args(b, ex, feats_q, man[b]['imageId'])
                    if i < 2: checks.append(compare(ref.run(None, feed(ref, a_fp))[0], q4x.run(None, feed(q4x, a_fp))[0], f'{b}/{tg}/{i}/q4-vs-dequantized', atol=6e-3))
                    x, y = fp.run(None, feed(fp, a_fp))[0], q4.run(None, feed(q4, a_q))[0]
                    drift.append((x, y))
                entry = {'counts': {g: counts[(b, g)] for g in ('vision', tg)}, 'parity': checks}
                if b == 'decision/vision':
                    ps = [(softmax(x), softmax(y)) for x, y in drift]; shifts = [float(np.abs(p - q).max()) for p, q in ps]
                    entry['heldOut'] = {'inputs': len(ps), 'flips': sum(int(p.argmax() != q.argmax()) for p, q in ps), 'meanShift': round(float(np.mean(shifts)), 4), 'maxShift': round(float(np.max(shifts)), 4)}
                    flipped = []
                    for f in json.loads((src_dir(models, b) / 'fixtures.json').read_text()):
                        q_ = f['question']; opts = list(q_['criteria']) if q_['type'] == 'choice' else ['No / false', 'Yes / true']
                        ex = C.decision_example(proc, [C.load_image(str(src_dir(models, b) / n)) for n in f.get('images', [])], '', q_['instructions'], opts)
                        if ex['input_ids'] != f['ids']: raise AssertionError(f'fixture {f["name"]}: token ids differ from the recorded ones')
                        y = sess[tg][3].run(None, feed(sess[tg][3], task_args(b, ex, features(sess['vision'][3], ex), man[b]['imageId'])))[0]
                        if int(np.argmax(y)) != int(np.argmax(f['logits'])): flipped.append(f['name'])
                    entry['fixtureFlips'] = flipped
                    extra = {'validated': not flipped, **({'rejected': f'chosen option differs from FP32 on fixtures {flipped}'} if flipped else {})}
                else:
                    diffs = [float(np.abs(x - y).max()) for x, y in drift]
                    entry['heldOut'] = {'inputs': len(diffs), 'meanMaxLogitDiff': round(float(np.mean(diffs)), 4), 'maxLogitDiff': round(float(np.max(diffs)), 4)}
                    # vision-box points at another cell of the same region in q4, so its gate is the share of points in the
                    # labelled region (scripts/verify-accuracy.ts) and not "no fixture changes" (scripts/verify-quantized.ts).
                    # VERIFIED: vision-box q4 failed the fixture check on its point and passed the accuracy gate (2026-11-12).
                    gate = 'verify-accuracy.ts' if b == 'vision-box' else 'verify-quantized.ts'
                    extra = {'validated': False, 'pendingVerification': f'node scripts/{gate} ' + os.path.relpath(d, REPO)}
                # The manifest: the FP32 one with the q4 graphs; files other than graphs and parity records are copied.
                m = dict(man[b]); src = src_dir(models, b)
                for f in src.iterdir():
                    if f.suffix in ('.png', '.json') and f.name not in ('manifest.json',) and not f.name.startswith('parity-') or f.name.startswith('LICENSE'): shutil.copyfile(f, d / f.name)
                m['graphs'] = {}
                for g in ('vision', tg):
                    ext = [f'{g}.onnx.data']; total = (d / f'{g}.onnx').stat().st_size + (d / f'{g}.onnx.data').stat().st_size
                    if shared:
                        sp = shared_dir / f'{"image" if g == "vision" else "text"}.onnx.data'; ext.append({'path': SHARED, 'file': os.path.relpath(sp, d)}); total += sp.stat().st_size
                    m['graphs'][g] = {'file': f'{g}.onnx', 'externalData': ext, 'sizeBytes': (d / f'{g}.onnx').stat().st_size, 'totalSizeBytes': total}
                m['precision'] = 'w4a32'; m['quantizedFrom'] = tree_relpath(src, d, (models, out))
                m['quantization'] = 'GPTQ int4 (groups of 32, fp16 scales, 4-bit zero points, MatMulNBits) on every image-encoder and text matrix; token table int4 groups of 32 with exact rows from id 49152; LoRA int8 per channel; norms, biases and heads FP32'
                if shared: m['sharedData'] = [{'path': SHARED, 'file': os.path.relpath(shared_dir / f'{k}.onnx.data', d), 'bytes': (shared_dir / f'{k}.onnx.data').stat().st_size, 'sha256': sha256(shared_dir / f'{k}.onnx.data')} for k in ('image', 'text')]
                m.update(extra); dump(d / 'manifest.json', m)
                dump(d / 'parity-python.json', {'passed': True, 'checks': checks, 'scope': 'q4 graphs in ONNX Runtime CPU vs the FP32 graphs with the same dequantized weights'})
                entry['bytes'] = {g: m['graphs'][g]['totalSizeBytes'] for g in ('vision', tg)}
                report['bundles'][b] = entry; log('INFO', f'checked bundle={b} ' + json.dumps({k: v for k, v in entry.items() if k not in ('parity', 'counts')}))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if shared: report['shared'] = {k: {'file': os.path.relpath(shared_dir / f'{k}.onnx.data', out), 'bytes': (shared_dir / f'{k}.onnx.data').stat().st_size} for k in ('image', 'text')}
        del models_fp
    dump(out / f'report-vision-q4.json', report)
    log('INFO', 'done')


if __name__ == '__main__':
    sys.exit(main())
