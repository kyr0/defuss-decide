#!/usr/bin/env python3
"""Reproduce the q4 text bundles: one GPTQ int4 backbone shared by the decision head and the five text task adapters.

    uv run python python/quantize_gptq.py --source checkpoints/fast --models models --out models

Reads the validated FP32 graphs (`<models>/decision/fast/decision.onnx`, `<models>/vela3-fast-onnx/<task>/fp32/
task.onnx`) and the pinned checkpoint, and writes

    <out>/backbone/fast/q4/backbone.onnx.data       the shared weights: 168 backbone matrices and the token table
    <out>/decision/fast/q4/                         decision graph + its own weights (head, norms)
    <out>/vela3-fast-onnx/<task>/q4/                task graph + its own weights (LoRA, layers 20 to 23, head)

Each graph lists both data files in its manifest; the loaders pass the shared file from its own folder, so a client
downloads it once (`src/web.ts`, `src/node.ts`). The graphs keep their structure: every backbone matrix becomes a
`MatMulNBits` (4 bits, groups of 32, fp16 scales behind a `Cast`, 4-bit zero points), the token table a lookup of
packed bytes with the 8 marker rows exact, the LoRA matrices int8 per channel (`DequantizeLinear`, the task q8
format); heads and norms stay FP32. `--int8` keeps named backbone matrices int8 per channel instead.
VERIFIED: the decision graph and the text task graphs hold the same backbone weights byte for byte (layer 0 attention
and the token table compared, 2026-11-08), so one shared file serves all six. VERIFIED: `MatMulNBits` and the packed
lookup were exact in Node.js (CPU, WebGPU) and Chromium and WebKit (WASM, WebGPU); blocked `DequantizeLinear` with 4-bit
zero points was wrong on the Node.js CPU at optimization level extended and `GatherBlockQuantized` has no WASM kernel
(.vae/MEMORY.md, plans/2026-11-08_19-43_decision-q4.md).

GPTQ uses Hessians from inputs this script builds from the repository's documents in the decision format and in each
task's format: a backbone matrix from the decision model and all five task nets (each source weighted equally), a
task's layers 20 to 23 from its own net. Every graph must reproduce PyTorch with the same quantized weights; the
decision bundle must also keep the chosen option of its recorded fixtures. Task bundles stay unvalidated until
`scripts/verify-quantized.ts` passes, as for the other quantized task bundles.
"""
import argparse, hashlib, json, os, random, re, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper as nh
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memguard import guard
from export import compare, dump, session, softmax

REPO = Path(__file__).resolve().parent.parent

TASKS = ['evidence', 'ner', 'relation', 'span', 'span_set']
GROUP, MARKER_BASE = 32, 49152
SHARED = 'backbone.onnx.data'


def log(level, msg):
    """One line per event, ISO-8601 UTC first."""
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + f' {level} {msg}', flush=True)


def check_sources(root):
    """Refuse a checkpoint that differs from the pinned files (python/source-lock.json, task-source-lock.json)."""
    here = Path(__file__).parent
    locks = {**json.loads((here / 'source-lock.json').read_text())['fast'], **json.loads((here / 'task-source-lock.json').read_text()).get('fast', {})}
    for name, expected in locks.items():
        with (root / name).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != expected: raise ValueError(f'Source checksum mismatch: {name}')


# ---- calibration and held-out inputs ------------------------------------------------------------------------------
def paragraphs():
    """English paragraphs of this repository's documents, shuffled with a fixed seed; the first half calibrates, the
    second half is held out."""
    out = []
    files = sorted(REPO.glob('documentation/*.md')) + [REPO / f for f in ('README.md', 'ARCH.md', 'models/README.md')] + sorted(REPO.glob('plans/*.md'))
    for f in files:
        for p in re.split(r'\n\s*\n', f.read_text()):
            p = p.strip()
            if 200 < len(p) < 1500 and not p.startswith(('```', '|', '<', '#')): out.append(p)
    random.Random(0).shuffle(out)
    return out[: len(out) // 2], out[len(out) // 2:]


def lines(state):
    """The provider's `path: value` rendering of a state without observations (src/decide.ts decisionContext)."""
    if isinstance(state, str): return state
    out = []
    def walk(v, p):
        if isinstance(v, dict):
            for k, x in v.items(): walk(x, f'{p}.{k}' if p else k)
        elif isinstance(v, list):
            for i, x in enumerate(v): walk(x, f'{p}.{i}' if p else str(i))
        else:
            out.append(f"{p}: {'true' if v is True else 'false' if v is False else 'null' if v is None else v}" if p else str(v))
    walk(state, ''); return '\n'.join(out)


def decision_inputs(paras, seed, extra):
    """(state, question) pairs in the decision format: documents with yes/no and choice questions, and generated
    records with yes/no questions about their fields."""
    r = random.Random(seed)
    words = sorted({w.lower() for p in paras for w in re.findall(r'[A-Za-z]{6,}', p)})
    topics = ['installation', 'accuracy', 'privacy', 'performance', 'licensing', 'browser support', 'quantization', 'testing', 'images', 'decisions']
    names = ['Ada', 'Bob', 'Chen', 'Dana', 'Emil', 'Fatima', 'Gus', 'Hana']
    statuses = ['open', 'closed', 'pending', 'delivered', 'cancelled', 'refunded', 'shipped']
    def record():
        kind = r.choice(['order', 'ticket', 'user', 'payment'])
        st = {kind: {'id': r.randint(1, 9999), 'status': r.choice(statuses), 'owner': r.choice(names), 'amount': round(r.uniform(1, 500), 2),
                     'items': [{'sku': f'S{r.randint(1, 99)}', 'qty': r.randint(0, 5)} for _ in range(r.randint(1, 3))]}}
        f = r.choice(['status', 'owner']); v = st[kind][f] if r.random() < 0.5 else r.choice(statuses if f == 'status' else names)
        return lines(st), {'type': 'noul', 'instructions': f'Is the {kind} {f} {v}?' if f == 'status' else f'Does {v} own the {kind}?'}
    out = []
    for p in paras:
        k = r.random()
        if k < 0.4: out.append((p, {'type': 'noul', 'instructions': f'Does the text mention {r.choice(words)}?'}))
        elif k < 0.7: out.append((p, {'type': 'choice', 'instructions': 'Which topic fits the text best?', 'criteria': {t: None for t in r.sample(topics, 3)}}))
        else: out.append(record())
    out += [record() for _ in range(extra)]
    r.shuffle(out); return out


def task_inputs(task, paras, seed):
    """(state, question) pairs in a task's span format, one per paragraph."""
    r = random.Random(seed)
    labels = {'ner': ['person', 'organization', 'city', 'country', 'date', 'product'],
              'span_set': ['EMAIL_ADDRESS', 'PHONE_NUMBER', 'PERSON', 'LOCATION', 'DATE_TIME', 'URL']}
    out = []
    for p in paras:
        if task == 'span': q = {'criteria': {'tool': 'A software tool or library', 'number': 'A number with its unit', 'format': 'A file format'}}
        elif task == 'ner': q = {'criteria': r.sample(labels['ner'], 3)}
        elif task == 'relation': q = {'instructions': 'Which spans name what the library runs on?', 'criteria': ['runtime']}
        elif task == 'evidence':
            q = {'instructions': f'Which spans support the claim that {p.split(".")[0].strip()[:120]}?', 'criteria': ['evidence']} if r.random() < 0.5 \
                else {'target': 'context', 'criteria': ['evidence']}
            if 'target' in q: p = {'user': 'What does the text say?', 'context': p, 'answer': p.split('.')[0]}
        else: q = {'criteria': r.sample(labels['span_set'], 2)} if r.random() < 0.6 else {'criteria': ['unsupported']}
        if task == 'span_set' and q['criteria'] == ['unsupported']: p = {'context': p, 'answer': p.split('.')[0] + ' is false.'}
        out.append((p, {'type': 'span', **q}))
    return out


# ---- models ------------------------------------------------------------------------------------------------------
def decision_model(root):
    from vela3_encoder import DecisionEncoder
    d = DecisionEncoder.load(root, device='cpu', max_len=2048); d.encoder.set_attn_implementation('eager'); return d.eval()


def task_net(root, task):
    """A text task net on a fresh decision backbone: building an adapter wraps the backbone's modules in place."""
    import vela3_schema as S
    from vela3_encoder import Vela3Encoder
    cfg = json.loads((root / 'vela3_config.json').read_text())
    net = S.build_adapter(decision_model(root), root / cfg['adapters'][task]['file'], 'cpu'); net.encoder.set_attn_implementation('eager')
    eng = S.SchemaEngine(root, net, 'cpu', max_len=2048)
    v = Vela3Encoder.__new__(Vela3Encoder); v.cfg = cfg; v._VS = S; v.span = v.span_ev = v.span_set = eng; v.task = {'ner': eng, 'relation': eng}; v.entity_route = 'ner'
    return net.eval(), eng, v


def task_records(eng, v, cases):
    """The token ids and positions the upstream engine builds for each case (as python/export_tasks.py records them)."""
    captured, original, recs = {}, eng.predict_raw, []
    def capture(rows):
        captured['rec'] = eng._assemble(eng._prep(rows[0])); return original(rows)
    eng.predict_raw = capture
    try:
        for state, q in cases:
            captured.pop('rec', None)
            try: v.span_answer(state, q)
            except Exception as e: log('WARN', f'skipped calibration case error={type(e).__name__}'); continue
            if 'rec' in captured: recs.append(captured['rec'])
    finally: eng.predict_raw = original
    return recs


def task_args(rec):
    return (torch.tensor([rec['ids']]), torch.ones(1, len(rec['ids']), dtype=torch.long), torch.tensor(rec['w_pos']), torch.tensor(rec['e_pos']))


def fast_args(rec):
    n = len(rec['ids']); pool = torch.zeros(1, n); pool[:, rec['pool'][0]:rec['pool'][1]] = 1
    return torch.tensor([rec['ids']]), torch.ones(1, n, dtype=torch.long), torch.tensor(rec['opt_pos']), pool


# ---- Hessians and quantizers -------------------------------------------------------------------------------------
LINEAR = re.compile(r'^encoder\.layers\.(\d+)\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj(\.base)?$')


def linears(model):
    """(canonical name, nn.Linear) of the 168 backbone matrices; in a task net the LoRA module's `base`."""
    for name, mod in model.named_modules():
        m = LINEAR.match(name)
        if m and isinstance(mod, torch.nn.Linear): yield name.removesuffix('.base'), mod


class Hessians:
    """X^T X / n of each matrix's input, averaged over sources so every consumer of a shared matrix counts equally."""
    def __init__(self): self.sum, self.sources = {}, {}
    def collect(self, modules, run):
        acc, cnt, hooks = {}, {}, []
        for key, mod in modules:
            def hook(m, inp, out, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float(); acc[key] = acc.get(key, 0) + x.T @ x; cnt[key] = cnt.get(key, 0) + x.shape[0]
            hooks.append(mod.register_forward_hook(hook))
        try: run()
        finally: [h.remove() for h in hooks]
        for key in acc:
            self.sum[key] = self.sum.get(key, 0) + acc[key] / cnt[key]; self.sources[key] = self.sources.get(key, 0) + 1
    def __getitem__(self, key): return self.sum[key] / self.sources[key]


def gptq(W, H, g=GROUP, damp=0.01, block=128):
    """GPTQ int4 of W (out, in) with asymmetric min-max groups of `g` inputs; scales are rounded to fp16 before use,
    as stored. Returns codes (out, in) uint8 0..15, scales (out, in/g) fp16, zero points (out, in/g) uint8."""
    W = W.detach().float().clone(); n = W.shape[1]; H = H.clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(n)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W, dtype=torch.uint8); S = torch.zeros(W.shape[0], n // g); Z = torch.zeros(W.shape[0], n // g)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % g == 0:
                grp = W1[:, i:i + g]; lo, hi = grp.amin(1), grp.amax(1)
                s = ((hi - lo) / 15).clamp_min(1e-8).half().float(); z = (-lo / s).round().clamp(0, 15)
                S[:, (i1 + i) // g] = s; Z[:, (i1 + i) // g] = z
            w = W1[:, i]; q = (w / s + z).round().clamp(0, 15); Q[:, i1 + i] = q.to(torch.uint8)
            e = (w - (q - z) * s) / Hi[i, i]; W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]; E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q.numpy(), S.half().numpy(), Z.to(torch.uint8).numpy()


def rtn4(W, g=GROUP):
    """Round-to-nearest int4 with the same groups and storage as gptq(), for the token table."""
    W = W.detach().float(); o, n = W.shape; w = W.reshape(o, n // g, g); lo, hi = w.amin(-1), w.amax(-1)
    s = ((hi - lo) / 15).clamp_min(1e-8).half().float(); z = (-lo / s).round().clamp(0, 15)
    q = (w / s[..., None] + z[..., None]).round().clamp(0, 15)
    return q.reshape(o, n).to(torch.uint8).numpy(), s.half().numpy(), z.to(torch.uint8).numpy()


def dequant4(q, s, z, g=GROUP):
    return (q.astype(np.float32) - np.repeat(z, g, 1).astype(np.float32)) * np.repeat(s.astype(np.float32), g, 1)


def int8(W):
    """Symmetric int8 per output row of W (torch layout), the task q8 format; returns codes and fp32 scales."""
    W = W.detach().float().numpy(); s = np.maximum(np.abs(W).max(1) / 127, np.finfo(np.float32).tiny).astype(np.float32)
    return np.clip(np.rint(W / s[:, None]), -127, 127).astype(np.int8), s


def key(a):
    """Content address of an FP32 matrix in ONNX layout (in, out)."""
    return hashlib.sha1(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


# ---- graph rewrite -----------------------------------------------------------------------------------------------
class DataFile:
    """Tensors appended to one external data file at 64-byte aligned offsets."""
    def __init__(self, path, location): self.f = open(path, 'wb'); self.location = location; self.index = {}
    def add(self, name, array):
        if name in self.index: return self.index[name]
        off = (self.f.tell() + 63) // 64 * 64; self.f.seek(off); raw = np.ascontiguousarray(array).tobytes(); self.f.write(raw)
        self.index[name] = (off, len(raw), array.dtype, array.shape); return self.index[name]
    def close(self): self.f.close()


def ext_tensor(name, data, entry):
    off, length, dtype, shape = entry
    t = TensorProto(); t.name = name; t.data_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype)); t.dims.extend(shape)
    t.data_location = TensorProto.EXTERNAL
    for k, v in (('location', data.location), ('offset', str(off)), ('length', str(length))):
        e = t.external_data.add(); e.key, e.value = k, v
    return t


def nbits_tensors(q, s, z):
    """MatMulNBits inputs from codes (N, K): B [N, K/32, 16] low nibble first, scales fp16 [N*K/32], zero points
    [N, ceil(K/32/2)] packed alike."""
    N, K = q.shape; nb = K // GROUP; qb = q.reshape(N, nb, GROUP)
    B = (qb[..., 0::2] | (qb[..., 1::2] << 4)).astype(np.uint8)
    zp = np.zeros((N, (nb + 1) // 2), np.uint8)
    for j in range(nb): zp[:, j // 2] |= (z[:, j] << (4 * (j % 2))).astype(np.uint8)
    return B, s.reshape(-1).astype(np.float16), zp


def rewrite(src, dst_dir, graph_name, quant, shared, own_bytes_min=1024):
    """Write `<dst_dir>/<graph_name>.onnx` and `<graph_name>.onnx.data` from the FP32 graph `src`.

    `quant` maps the content address of an FP32 MatMul weight (ONNX layout) to ('int4', q, s, z, shared?, name[,
    own scales?]) or
    ('int8', q, s); the token table (a Gather table with more than MARKER_BASE rows) to ('embed', ...). Shared tensors
    go to `shared` (written once), the rest to the graph's own file; with `shared` None everything goes to the own
    file (a self-contained bundle). Returns the counts per kind."""
    m = onnx.load(str(src)); g = m.graph
    init = {i.name: i for i in g.initializer}
    own = DataFile(dst_dir / f'{graph_name}.onnx.data', f'{graph_name}.onnx.data')
    new_nodes, drop, add, counts = [], set(), [], {'int4 shared': 0, 'int4 own': 0, 'int8': 0, 'fp32': 0, 'embed': 0}
    one = lambda name, arr: add.append(nh.from_array(np.asarray(arr), name)) or name
    consts = {}
    def const(name, arr):
        if name not in consts: consts[name] = one(name, arr)
        return consts[name]
    for node in g.node:
        w = node.input[1] if node.op_type == 'MatMul' and len(node.input) > 1 and node.input[1] in init else None
        if w is not None and len(init[w].dims) == 2:
            q = quant.get(key(nh.to_array(init[w])))
            if q and q[0] == 'int4':
                _, codes, s, z, is_shared, cname, *opt = q; own_scales = bool(opt and opt[0])  # own_scales: python/repair_decision.py
                B, S16, ZP = nbits_tensors(codes, s, z)
                store = shared if is_shared and shared is not None else own
                names = [f'{cname}.B', f'{cname}.scales16' + ('.own' if own_scales else ''), f'{cname}.zeros']
                for n_, a in zip(names, (B, S16, ZP)):
                    st = own if own_scales and n_ == names[1] else store
                    if not any(t.name == n_ for t in add): add.append(ext_tensor(n_, st, st.add(n_, a)))
                N, K = codes.shape
                new_nodes.append(helper.make_node('Cast', [names[1]], [f'{node.name or w}.scales'], to=TensorProto.FLOAT))
                new_nodes.append(helper.make_node('MatMulNBits', [node.input[0], names[0], f'{node.name or w}.scales', names[2]], list(node.output),
                                                  domain='com.microsoft', K=K, N=N, bits=4, block_size=GROUP, name=node.name))
                drop.add(w); counts['int4 shared' if is_shared else 'int4 own'] += 1; continue
            if q and q[0] == 'int8':
                _, codes, s, is_shared, cname = q  # codes (out, in) torch layout; ONNX weight is (in, out): scale axis 1
                store = shared if is_shared and shared is not None else own; qn, sn = f'{cname or w}.int8', f'{cname or w}.scale'
                for n_, arr in ((qn, codes.T.copy()), (sn, s)):
                    if not any(t.name == n_ for t in add): add.append(ext_tensor(n_, store, store.add(n_, arr)))
                new_nodes.append(helper.make_node('DequantizeLinear', [qn, sn], [f'{w}.dq'], axis=1))
                new_nodes.append(helper.make_node('MatMul', [node.input[0], f'{w}.dq'], list(node.output), name=node.name))
                drop.add(w); counts['int8'] += 1; continue
        table = init.get(node.input[0]) if node.op_type == 'Gather' else None
        if table is not None and 'embed' in quant and len(table.dims) == 2 and table.dims[0] > MARKER_BASE:
            _, q, s, z, resid = quant['embed']
            V, K = q.shape; ids = node.input[1]; o = node.output[0]; p = f'{o}.emb'
            packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8); store = shared if shared is not None else own
            for n_, a in (('embed.codes', packed), ('embed.scales16', s), ('embed.zeros', z), ('embed.markers', resid)):
                if not any(t.name == n_ for t in add): add.append(ext_tensor(n_, store, store.add(n_, a)))
            c16, cK, cg, cG, cmk = const('c16', np.float32(16)), const('cK', np.array([K], np.int64)), const('cg', np.array([K // GROUP, GROUP], np.int64)), \
                const('cneg1', np.array([-1], np.int64)), const('cmarker', np.float32(MARKER_BASE - 1))
            new_nodes += [
                helper.make_node('Cast', ['embed.scales16'], [f'{p}.stab'], to=TensorProto.FLOAT),
                helper.make_node('Cast', ['embed.zeros'], [f'{p}.ztab'], to=TensorProto.FLOAT),
                helper.make_node('Gather', ['embed.codes', ids], [f'{p}.packed']),
                helper.make_node('Cast', [f'{p}.packed'], [f'{p}.pf'], to=TensorProto.FLOAT),
                helper.make_node('Div', [f'{p}.pf', c16], [f'{p}.d']), helper.make_node('Floor', [f'{p}.d'], [f'{p}.hi']),
                helper.make_node('Mul', [f'{p}.hi', c16], [f'{p}.hi16']), helper.make_node('Sub', [f'{p}.pf', f'{p}.hi16'], [f'{p}.lo']),
                helper.make_node('Unsqueeze', [f'{p}.lo', cG], [f'{p}.lo1']), helper.make_node('Unsqueeze', [f'{p}.hi', cG], [f'{p}.hi1']),
                helper.make_node('Concat', [f'{p}.lo1', f'{p}.hi1'], [f'{p}.pair'], axis=-1),
                helper.make_node('Shape', [ids], [f'{p}.ishape']),
                helper.make_node('Concat', [f'{p}.ishape', cK], [f'{p}.shape'], axis=0),
                helper.make_node('Concat', [f'{p}.ishape', cg], [f'{p}.gshape'], axis=0),
                helper.make_node('Reshape', [f'{p}.pair', f'{p}.shape'], [f'{p}.q']),
                helper.make_node('Gather', [f'{p}.stab', ids], [f'{p}.sg']), helper.make_node('Gather', [f'{p}.ztab', ids], [f'{p}.zg']),
                helper.make_node('Unsqueeze', [f'{p}.sg', cG], [f'{p}.s1']), helper.make_node('Unsqueeze', [f'{p}.zg', cG], [f'{p}.z1']),
                helper.make_node('Expand', [f'{p}.s1', f'{p}.gshape'], [f'{p}.s2']), helper.make_node('Expand', [f'{p}.z1', f'{p}.gshape'], [f'{p}.z2']),
                helper.make_node('Reshape', [f'{p}.s2', f'{p}.shape'], [f'{p}.s']), helper.make_node('Reshape', [f'{p}.z2', f'{p}.shape'], [f'{p}.z']),
                helper.make_node('Sub', [f'{p}.q', f'{p}.z'], [f'{p}.qz']), helper.make_node('Mul', [f'{p}.qz', f'{p}.s'], [f'{p}.deq']),
                # Marker rows (ids >= 49152) exact: row 0 of `embed.markers` is zero, rows 1..8 hold exact minus dequantized.
                helper.make_node('Cast', [ids], [f'{p}.idf'], to=TensorProto.FLOAT), helper.make_node('Sub', [f'{p}.idf', cmk], [f'{p}.mrel']),
                helper.make_node('Relu', [f'{p}.mrel'], [f'{p}.mr']), helper.make_node('Cast', [f'{p}.mr'], [f'{p}.mi'], to=TensorProto.INT64),
                helper.make_node('Gather', ['embed.markers', f'{p}.mi'], [f'{p}.r']),
                helper.make_node('Add', [f'{p}.deq', f'{p}.r'], [o]),
            ]
            drop.add(node.input[0]); counts['embed'] += 1; continue
        new_nodes.append(node)
    keep = []
    for t in g.initializer:
        if t.name in drop: continue
        a = nh.to_array(t)
        if a.nbytes >= own_bytes_min:
            keep.append(ext_tensor(t.name, own, own.add(t.name, a))); counts['fp32'] += 1
        else: keep.append(t)
    del g.node[:]; g.node.extend(new_nodes)
    del g.initializer[:]; g.initializer.extend(keep + add)
    if not any(o.domain == 'com.microsoft' for o in m.opset_import): m.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    own.close()
    onnx.save(m, str(dst_dir / f'{graph_name}.onnx'))
    return counts


def assembled(dst_dir, graph_name, shared_path):
    """A temporary folder with the graph and both data files hard-linked next to it: ONNX Runtime rejects a data
    file outside the graph's folder by path. (No onnx.checker: it has no schema for the com.microsoft operators;
    ONNX Runtime's load is the check.)"""
    t = Path(tempfile.mkdtemp(prefix='q4-', dir=REPO / 'tmp'))
    for f in (f'{graph_name}.onnx', f'{graph_name}.onnx.data'): os.link(dst_dir / f, t / f)
    os.link(shared_path, t / SHARED)
    return t


def exact_session(path):
    """A CPU session at optimization level `basic`. WHY: at `extended` the CPU rewrites the int8 LoRA matrices into
    `MatMulNBits` and moves task logits by up to 0.0072 (VERIFIED on evidence, 2026-11-08), so an export check at the
    default level would mix the runtime's rewrite into the export's correctness. The drift figures use the default
    level, which is what callers get."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC; so.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), sess_options=so, providers=['CPUExecutionProvider'])


def sha256(path):
    with open(path, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def tree_relpath(path, start, roots):
    """`path` relative to the folder `start`, both taken at their place in the models tree: a file under any of `roots`
    (--models, --out) sits where copying --out into --models puts it. WHY: a bundle written to tmp/ and installed by
    copying kept references relative to tmp/ (VERIFIED: `quantizedFrom` `../../../../../models/...` in the installed text
    q4 manifests, 2026-11-09)."""
    def placed(p): return p.relative_to(max((r for r in roots if p.is_relative_to(r)), key=lambda r: len(r.parts)))
    return os.path.relpath(placed(path), placed(start))


def manifest_for(src_dir, dst_dir, graph_name, shared_path, extra, roots):
    """The FP32 manifest with the q4 graph, its two data files and the shared file's size and hash; paths as in
    `tree_relpath` over `roots`."""
    m = json.loads((src_dir / 'manifest.json').read_text())
    for p in src_dir.iterdir():
        if p.name in ['tokenizer.json', 'tokenizer_config.json', 'fixtures.json'] or p.name.startswith('LICENSE'): shutil.copyfile(p, dst_dir / p.name)
    own = dst_dir / f'{graph_name}.onnx.data'
    m['graphs'][graph_name] = {'file': f'{graph_name}.onnx', 'externalData': [own.name, {'path': SHARED, 'file': tree_relpath(shared_path, dst_dir, roots)}],
                               'sizeBytes': (dst_dir / f'{graph_name}.onnx').stat().st_size,
                               'totalSizeBytes': (dst_dir / f'{graph_name}.onnx').stat().st_size + own.stat().st_size + shared_path.stat().st_size}
    m['precision'] = 'w4a32'
    m['quantization'] = ('GPTQ int4 (groups of 32, fp16 scales, 4-bit zero points, MatMulNBits) on the 168 backbone matrices and layers 20-23; '
                         'token table int4 groups of 32 with exact marker rows; LoRA int8 per channel; heads and norms FP32')
    m['quantizedFrom'] = tree_relpath(src_dir, dst_dir, roots)
    m['sharedData'] = {'path': SHARED, 'file': tree_relpath(shared_path, dst_dir, roots), 'bytes': shared_path.stat().st_size, 'sha256': sha256(shared_path)}
    m.update(extra)
    return m


# ---- main --------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True, help='Pinned vela3-fast checkpoint (python/download.py --model fast --tasks)')
    ap.add_argument('--models', type=Path, required=True, help='Folder with the validated FP32 bundles')
    ap.add_argument('--out', type=Path, required=True, help='Root to write backbone/, decision/ and vela3-fast-onnx/ into')
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--int8', default='', help='Regular expression of backbone matrices kept int8 per channel instead of int4, e.g. "layers\\.(22|23)\\." (reports/quantization-decision-sensitivity.json)')
    ap.add_argument('--inputs', type=Path, help='Replay the calibration and held-out inputs of an earlier run (its inputs.json)')
    ap.add_argument('--min-free-gb', type=float, default=8.0, help='Stop the job when the machine has less memory available (python/memguard.py)')
    a = ap.parse_args(); torch.set_num_threads(6); torch.manual_seed(0)
    guard(a.min_free_gb)
    keep8 = re.compile(a.int8) if a.int8 else None
    def quantize(w, h, is_shared, cname):
        """GPTQ int4, or int8 per channel for the matrices `--int8` names."""
        if keep8 and keep8.search(cname): q, s = int8(w); return ('int8', q, s, is_shared, cname)
        return ('int4', *gptq(w, h), is_shared, cname)
    root, models, out = a.source.resolve(), a.models.resolve(), a.out.resolve(); tasks = [t for t in a.tasks.split(',') if t]
    check_sources(root); sys.path.insert(0, str(root))
    if a.inputs: inputs = json.loads(a.inputs.read_text())
    else:
        calib_paras, held_paras = paragraphs()
        inputs = {'decision': {'calibration': decision_inputs(calib_paras, 1, 80), 'heldOut': decision_inputs(held_paras, 3, 80)},
                  **{t: {'calibration': task_inputs(t, calib_paras, 2), 'heldOut': task_inputs(t, held_paras, 4)[:60]} for t in TASKS}}
    log('INFO', 'inputs ' + ' '.join(f'{k}={len(v["calibration"])}/{len(v["heldOut"])}' for k, v in inputs.items()))
    H = Hessians(); quant = {}

    # 1. The decision model: backbone Hessians on decision inputs, and the reference weights of the shared matrices.
    d = decision_model(root)
    dec_recs = [d.encode(s, q) for s, q in inputs['decision']['calibration']]
    base = {name: mod.weight.detach().clone() for name, mod in linears(d)}
    base_keys = {name: key(w.T.numpy()) for name, w in base.items()}
    def run_decision():
        with torch.no_grad():
            for rec in dec_recs:
                ids = torch.tensor([rec['ids']]); d.encoder(input_ids=ids, attention_mask=torch.ones_like(ids))
    H.collect(list(linears(d)), run_decision)
    log('INFO', f'hessians source=decision inputs={len(dec_recs)} tokens={sum(len(r["ids"]) for r in dec_recs)}')
    E = d.encoder.embed_tokens.weight.detach().clone(); del d

    # 2. Each task net: shared Hessians (its inputs through the shared matrices), its own layers' Hessians and GPTQ,
    #    its LoRA int8.
    for task in tasks:
        net, eng, v = task_net(root, task)
        recs = task_records(eng, v, inputs[task]['calibration'])
        own = {name: mod for name, mod in linears(net) if not torch.equal(mod.weight, base[name])}
        Ht = Hessians()
        def run_task():
            from vela_onnx.tasks import FastSpan
            w = FastSpan(net).eval()
            with torch.no_grad():
                for rec in recs: w(*task_args(rec))
        shared_mods = [(n, m) for n, m in linears(net) if n not in own]
        both = shared_mods + [(f'{task}:{n}', m) for n, m in own.items()]
        H_all = Hessians(); H_all.collect(both, run_task)
        for k_ in H_all.sum:
            if ':' in k_: Ht.sum[k_.split(':', 1)[1]] = H_all.sum[k_]; Ht.sources[k_.split(':', 1)[1]] = 1
            else: H.sum[k_] = H.sum.get(k_, 0) + H_all.sum[k_]; H.sources[k_] = H.sources.get(k_, 0) + 1
        for name, mod in own.items():
            quant[key(mod.weight.detach().T.numpy())] = quantize(mod.weight, Ht[name], False, f'{task}.{name}')
        for mname, mod in net.named_modules():
            for pname in ('lora_A', 'lora_B'):
                p = getattr(mod, pname, None)
                if isinstance(p, torch.nn.Parameter):
                    q, s = int8(p); quant[key(p.detach().T.numpy())] = ('int8', q, s, False, None)
        log('INFO', f'task={task} inputs={len(recs)} own_matrices={len(own)} shared_matrices={len(shared_mods)}')
        del net, eng, v

    # 3. GPTQ of the shared matrices, the token table, and the shared file.
    for name, w in base.items():
        quant[base_keys[name]] = quantize(w, H[name], True, f'backbone.{name}')
    q, s, z = rtn4(E[:MARKER_BASE])
    full_q = np.concatenate([q, np.zeros((E.shape[0] - MARKER_BASE, q.shape[1]), np.uint8)])
    full_s = np.concatenate([s, np.ones((E.shape[0] - MARKER_BASE, s.shape[1]), np.float16)])
    full_z = np.concatenate([z, np.zeros((E.shape[0] - MARKER_BASE, z.shape[1]), np.uint8)])
    deq = dequant4(full_q, full_s, full_z)
    resid = np.zeros((1 + E.shape[0] - MARKER_BASE, E.shape[1]), np.float32); resid[1:] = E[MARKER_BASE:].numpy() - deq[MARKER_BASE:]
    quant['embed'] = ('embed', full_q, full_s, full_z, resid)
    log('INFO', f'gptq shared_matrices={len(base)}')
    shared_dir = out / 'backbone' / 'fast' / 'q4'; shared_dir.mkdir(parents=True, exist_ok=True)
    shared = DataFile(shared_dir / SHARED, SHARED)

    # 4. Rewrite every graph, then check it against PyTorch with the same quantized weights.
    from vela_onnx.portable import FastDecision
    from vela_onnx.tasks import FastSpan
    def fake_quant(model):
        """Load the dequantized weights into a PyTorch model, so it computes what the q4 graph should."""
        with torch.no_grad():
            for name, mod in linears(model):
                q_ = quant[key(mod.weight.detach().T.numpy())]
                mod.weight.copy_(torch.from_numpy(dequant4(q_[1], q_[2], q_[3]) if q_[0] == 'int4' else q_[1].astype(np.float32) * q_[2][:, None]))
            for mod in model.modules():
                for pname in ('lora_A', 'lora_B'):
                    p = getattr(mod, pname, None)
                    if isinstance(p, torch.nn.Parameter):
                        q_ = quant[key(p.detach().T.numpy())]; p.copy_(torch.from_numpy(q_[1].astype(np.float32) * q_[2][:, None]))
            emb = model.encoder.embed_tokens.weight; emb.copy_(torch.from_numpy(deq)); emb[MARKER_BASE:] = E[MARKER_BASE:]
        return model

    report = {'date': time.strftime('%Y-%m-%d'), 'tool': 'python/quantize_gptq.py', 'graphs': {}}
    graphs = [('decision', models / 'decision' / 'fast', out / 'decision' / 'fast' / 'q4', 'decision')] + \
             [(task, models / 'vela3-fast-onnx' / task / 'fp32', out / 'vela3-fast-onnx' / task / 'q4', 'task') for task in tasks]
    pending = []
    for label, src_dir, dst_dir, gname in graphs:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for stale in ('manifest.json', 'task.onnx', 'task.onnx.data', 'decision.onnx', 'decision.onnx.data'): (dst_dir / stale).unlink(missing_ok=True)
        counts = rewrite(src_dir / f'{gname}.onnx', dst_dir, gname, quant, shared)
        log('INFO', f'rewrote graph={label} ' + ' '.join(f'{k.replace(" ", "_")}={v}' for k, v in counts.items()))
        pending.append((label, src_dir, dst_dir, gname, counts))
    shared.close()

    held_dec = inputs['decision']['heldOut']
    labelled = [c for c in json.loads((REPO / 'tests/fixtures/decide-cases.json').read_text())['cases'] if 'images' not in c]
    for label, src_dir, dst_dir, gname, counts in pending:
        tmp = assembled(dst_dir, gname, shared_dir / SHARED)
        try:
            q4, fp, q4x = session(tmp / f'{gname}.onnx'), session(src_dir / f'{gname}.onnx'), exact_session(tmp / f'{gname}.onnx')
            fixtures = json.loads((src_dir / 'fixtures.json').read_text()); checks, entry = [], {'counts': counts}
            if label == 'decision':
                model = fake_quant(decision_model(root)); wrapper = FastDecision(model).eval(); flipped = []
                for f in fixtures:
                    args = fast_args({'ids': f['ids'], 'pool': f['pool'], 'opt_pos': f['optionPositions']})
                    with torch.no_grad(): ref = wrapper(*args).numpy()
                    got = q4x.run(None, {n.name: t.numpy() for n, t in zip(q4x.get_inputs(), args)})[0]
                    checks.append(compare(ref, got, f['name'] + '/onnx-vs-pytorch'))
                    if int(np.argmax(got)) != int(np.argmax(f['logits'])): flipped.append(f['name'])
                def drift(items):
                    rows = []
                    for s_, q_ in items:
                        rec = model.encode(s_, q_); args = [t.numpy() for t in fast_args(rec)]
                        a_ = softmax(fp.run(None, {n.name: t for n, t in zip(fp.get_inputs(), args)})[0])
                        b_ = softmax(q4.run(None, {n.name: t for n, t in zip(q4.get_inputs(), args)})[0])
                        rows.append((rec['names'], a_, b_))
                    shifts = [float(np.abs(a_ - b_).max()) for _, a_, b_ in rows]
                    return rows, {'inputs': len(rows), 'flips': sum(int(a_.argmax() != b_.argmax()) for _, a_, b_ in rows),
                                  'meanShift': round(float(np.mean(shifts)), 4), 'maxShift': round(float(np.max(shifts)), 4)}
                _, entry['heldOut'] = drift(held_dec)
                rows, entry['labelled'] = drift([(lines(c['state']), {'type': 'choice', 'instructions': c['instructions'], 'criteria': c['options'] if isinstance(c['options'], dict) else {o: None for o in c['options']}}
                                                  if c.get('options') else {'type': 'noul', 'instructions': c['instructions']}) for c in labelled])
                entry['labelled']['correctFp32'] = sum(int(n[int(a_.argmax())] == c['expected']) for (n, a_, _), c in zip(rows, labelled))
                entry['labelled']['correctQ4'] = sum(int(n[int(b_.argmax())] == c['expected']) for (n, _, b_), c in zip(rows, labelled))
                # The decision gate, like the task gate: every recorded fixture keeps FP32's chosen option.
                extra = {'validated': not flipped, **({'rejected': f'chosen option differs from FP32 on fixtures {flipped}'} if flipped else {})}
                entry['fixtureFlips'] = flipped
            else:
                net, eng, v = task_net(root, label); net = fake_quant(net); wrapper = FastSpan(net).eval()
                for f in fixtures:
                    args = (torch.tensor([f['ids']]), torch.ones(1, len(f['ids']), dtype=torch.long), torch.tensor(f['wordPositions']), torch.tensor(f['labelPositions']))
                    with torch.no_grad(): ref = wrapper(*args).numpy()
                    got = q4x.run(None, {n.name: t.numpy() for n, t in zip(q4x.get_inputs(), args)})[0]
                    checks.append(compare(ref, got, f['name'] + '/onnx-vs-pytorch'))
                diffs = []
                for rec in task_records(eng, v, inputs[label]['heldOut']):
                    args = [t.numpy() for t in task_args(rec)]
                    a_ = fp.run(None, {n.name: t for n, t in zip(fp.get_inputs(), args)})[0]; b_ = q4.run(None, {n.name: t for n, t in zip(q4.get_inputs(), args)})[0]
                    diffs.append(float(np.abs(a_ - b_).max()))
                entry['heldOut'] = {'inputs': len(diffs), 'meanMaxLogitDiff': round(float(np.mean(diffs)), 4), 'maxLogitDiff': round(float(np.max(diffs)), 4)}
                del net, eng, v
                extra = {'validated': False, 'pendingVerification': 'node scripts/verify-quantized.ts ' + os.path.relpath(dst_dir, REPO)}
            entry['parity'] = checks
            if a.int8: extra['int8Matrices'] = a.int8  # the --int8 pattern: these backbone matrices are int8 per channel
            m = manifest_for(src_dir, dst_dir, gname, shared_dir / SHARED, extra, (models, out))
            dump(dst_dir / 'manifest.json', m); dump(dst_dir / 'parity-python.json', {'passed': True, 'checks': checks, 'scope': 'q4 graph in ONNX Runtime CPU vs PyTorch with the same quantized weights'})
            entry['bytes'] = {'graph': m['graphs'][gname]['sizeBytes'], 'own': (dst_dir / f'{gname}.onnx.data').stat().st_size, 'shared': m['sharedData']['bytes']}
            report['graphs'][label] = entry; log('INFO', f'checked graph={label} ' + json.dumps({k: v for k, v in entry.items() if k not in ('parity', 'counts')}))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    report['shared'] = {'file': os.path.relpath(shared_dir / SHARED, out), 'bytes': (shared_dir / SHARED).stat().st_size, 'sha256': sha256(shared_dir / SHARED)}
    dump(out / 'backbone' / 'fast' / 'q4' / 'report.json', report); dump(out / 'backbone' / 'fast' / 'q4' / 'inputs.json', inputs)
    log('INFO', 'done')


if __name__ == '__main__':
    sys.exit(main())
