#!/usr/bin/env python3
"""Repair the q4 text decision bundle by distillation from FP32, without touching the shared backbone file.

    uv run python python/repair_decision.py --source checkpoints/fast --models models --out tmp/repair [--steps 6000]

The student is the q4 decision model as published: its int4 matrices and token table are decoded from
`<models>/backbone/fast/q4/backbone.onnx.data` through `<models>/decision/fast/q4/decision.onnx`, so the repair works on
the exact shared codes the four validated text task bundles also read. Trainable are only tensors the decision bundle
owns: a rank-16 correction per backbone matrix (a LoRA branch, as in the task adapters), the encoder's normalization
vectors and the decision head. The FP32 model is the teacher; the loss is the KL divergence of the option
probabilities plus the relative squared error of the final hidden states on every token, which anchors the
representation the head reads, not only the few option logits.

Data (label-free for training): BoolQ train (google/boolq, pinned revision, CC BY-SA 3.0) as yes/no and choice
questions, plus the calibration inputs of the q4 export (`inputs.json`: repository paragraphs and generated records). The checkpoint with the
smallest probability shift on the first 300 BoolQ validation questions is kept. The test sets are disjoint from
both: BoolQ validation questions 300 to 1,300 (labelled), the held-out inputs of the q4 export (`inputs.json`), the 56
labelled cases of tests/fixtures/decide-cases.json and the 5 recorded fixtures.

The export runs the repaired model through the same ONNX exporter as the FP32 bundle (`FastDecision`) and
`quantize_gptq.rewrite`: the backbone matrices and the token table map to the existing shared tensors (byte-checked
against the file), the corrections to int8 per channel in the decision's own file. Gate, proposed in
plans/2026-11-08_19-43_decision-q4.md (M4): every fixture keeps FP32's option and at most 1 % of the decisions flip on
the repository held-out set and on the BoolQ test set. `<out>/decision/fast/q4/` gets `validated: true` only then.
VERIFIED (smoke runs, 2026-11-09): the exported graph reproduces the PyTorch student within 1e-5 on the 5 fixtures,
and the shared codes and zero points it references equal the shared file byte for byte, with and without own scales.
HYPOTHESIS: training the group scales (`--scales-lr`) removes more drift than the rank-16 corrections alone, which
moved the selection shift only from 0.0169 to 0.0161 in 1,600 inputs; falsifier: the selection shift of a
`--scales-lr 1e-4` run of the same length.
"""
import argparse, gc, hashlib, json, math, random, shutil, sys, time
from pathlib import Path
import numpy as np
import onnx
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memguard import guard
from export import compare, dump, export_graph, session, softmax
from quantize_gptq import (GROUP, MARKER_BASE, REPO, SHARED, assembled, check_sources, decision_model, dequant4, exact_session,
                           fast_args, int8, key, lines, linears, log, manifest_for, rewrite)

BOOLQ = ('google/boolq', '35b264d03638db9f4ce671b711558bf7ff0f80d5',
         {'train': '4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623', 'validation': '52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8'})
RANK, ALPHA = 16, 32


def boolq(split):
    """[(passage, question, answer)] of one BoolQ split at the pinned revision; refuses a file with another hash."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(BOOLQ[0], f'data/{split}-00000-of-00001.parquet', repo_type='dataset', revision=BOOLQ[1])
    with open(path, 'rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != BOOLQ[2][split]: raise ValueError(f'BoolQ {split}: checksum mismatch')
    return [(r['passage'][:1500], r['question'], bool(r['answer'])) for r in pq.read_table(path).to_pylist()]


def boolq_inputs(rows, seed, choice_share=0.3):
    """BoolQ rows as decision inputs: a yes/no question, or the same question as a choice between yes, no and unclear."""
    r = random.Random(seed); out = []
    for passage, q, _ in rows:
        text = q[0].upper() + q[1:] + '?'
        out.append((passage, {'type': 'choice', 'instructions': text, 'criteria': {'yes': None, 'no': None, 'unclear': None}} if r.random() < choice_share
                    else {'type': 'noul', 'instructions': text}))
    return out


# ---- the student -------------------------------------------------------------------------------------------------
class Repaired(torch.nn.Module):
    """A backbone matrix as published: the shared int4 codes (frozen) times its group scales, plus a trainable
    rank-r correction. With `scales` the group scales are trainable too (as a log-factor on the published ones); the
    decision bundle then stores its own fp16 copy, while the codes and zero points stay in the shared file. In export
    mode it computes with the FP32 weight instead, so the exported MatMul carries the FP32 weight's content address
    and `quantize_gptq.rewrite` swaps in the shared int4 tensors; the correction is exported as two MatMuls."""
    def __init__(self, base, q, s, z, scales=False, rank=RANK, alpha=ALPHA):
        super().__init__()
        self.base, self.export, self.fixed = base, False, None
        # codes minus zero points fit int8 (-15..15): a quarter of the memory of float32 (python/memguard.py)
        self.register_buffer('qz', torch.from_numpy(q.astype(np.int8) - np.repeat(z, GROUP, 1).astype(np.int8)))
        self.register_buffer('s0', torch.from_numpy(s.astype(np.float32)))
        self.log_s = torch.nn.Parameter(torch.zeros_like(self.s0)) if scales else None
        self.lora_A = torch.nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank)); self.scale = alpha / rank
        base.weight.requires_grad_(False)
    def scales16(self):
        """The group scales as stored: fp16."""
        s = self.s0 if self.log_s is None else self.s0 * self.log_s.exp()
        return s.detach().cpu().half().numpy()
    def scales(self):
        if self.fixed is not None: return self.fixed
        return self.s0 if self.log_s is None else self.s0 * self.log_s.exp()
    def forward(self, x):
        y = F.linear(x, self.base.weight) if self.export else QMatMul.apply(x, self.qz, self.scales())
        return y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def dequant(qz, s):
    """The weight (out, in) of int8 codes `qz` (out, in) and group scales `s` (out, in / GROUP)."""
    return (qz.view(*s.shape, GROUP).float() * s[..., None]).view(qz.shape)


class QMatMul(torch.autograd.Function):
    """x times the transposed weight of codes `qz` and scales `s`. It keeps only x and the int8 codes for the backward
    pass and rebuilds the weight there, one matrix at a time. WHY: plain autograd on the dequantized weight keeps fp32
    copies of every backbone matrix until the backward pass (vela3-fast: 327M weights, 1.3 GB per copy). VERIFIED: with
    them a scale-training probe drove the machine under memguard's floor on 2026-11-09; this function gives the same
    values and gradients (torch.autograd.gradcheck, differences 0.0 in float32)."""
    @staticmethod
    def forward(ctx, x, qz, s):
        ctx.save_for_backward(x, qz, s)
        return F.linear(x, dequant(qz, s))

    @staticmethod
    def backward(ctx, gy):
        x, qz, s = ctx.saved_tensors; gx = gs = None
        if ctx.needs_input_grad[0]: gx = gy @ dequant(qz, s)
        if ctx.needs_input_grad[2]:
            gw = gy.reshape(-1, gy.shape[-1]).T @ x.reshape(-1, x.shape[-1])
            gs = (gw.view(*s.shape, GROUP) * qz.view(*s.shape, GROUP)).sum(-1)
        return gx, None, gs


def published_q4(models):
    """The int4 weights of the published decision graph, decoded from its MatMulNBits tensors and the packed token
    table: {name: W (out, in)}, the token table, and the shared tensors' locations in backbone.onnx.data."""
    q4dir = models / 'decision' / 'fast' / 'q4'; shared_path = models / 'backbone' / 'fast' / 'q4' / SHARED
    m = onnx.load(str(q4dir / 'decision.onnx'), load_external_data=False)
    where = {t.name: {e.key: e.value for e in t.external_data} for t in m.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL}
    index = {t.name: (int(where[t.name]['offset']), int(where[t.name]['length']), onnx.helper.tensor_dtype_to_np_dtype(t.data_type), tuple(t.dims))
             for t in m.graph.initializer if where.get(t.name, {}).get('location') == SHARED}
    arr = {}
    with open(shared_path, 'rb') as f:
        for name, (off, length, dtype, shape) in index.items():
            f.seek(off); arr[name] = np.frombuffer(f.read(length), dtype).reshape(shape)
    codes = {}
    for name in [n[len('backbone.'):-len('.B')] for n in arr if n.endswith('.B')]:
        B, s16, zp = arr[f'backbone.{name}.B'], arr[f'backbone.{name}.scales16'], arr[f'backbone.{name}.zeros']
        N, nb, _ = B.shape
        q = np.empty((N, nb, GROUP), np.uint8); q[..., 0::2] = B & 15; q[..., 1::2] = B >> 4; q = q.reshape(N, nb * GROUP)
        z = np.stack([(zp[:, j // 2] >> (4 * (j % 2))) & 15 for j in range(nb)], 1).astype(np.uint8); s = s16.reshape(N, nb)
        codes[name] = (q, s, z)
    packed = arr['embed.codes']; eq = np.empty((packed.shape[0], packed.shape[1] * 2), np.uint8); eq[:, 0::2] = packed & 15; eq[:, 1::2] = packed >> 4
    table = dequant4(eq, arr['embed.scales16'], arr['embed.zeros']); table[MARKER_BASE:] += arr['embed.markers'][1:]
    return codes, (eq, arr['embed.scales16'], arr['embed.zeros'], arr['embed.markers']), torch.from_numpy(table), index, shared_path


def student_from(root, codes, table, scales, teacher):
    """The student: a decision model whose backbone matrices are `Repaired` around the teacher's own (frozen) FP32
    Linear modules, so the FP32 weights the export needs exist once in memory, not twice."""
    d = decision_model(root); fp32 = dict(linears(teacher))
    for name, _ in list(linears(d)):
        parent, attr = name.rsplit('.', 1)
        setattr(d.get_submodule(parent), attr, Repaired(fp32[name], *codes[name], scales=scales))
    with torch.no_grad(): d.encoder.embed_tokens.weight.copy_(table)
    for p in d.parameters(): p.requires_grad_(False)
    trainable = {'lora': [], 'scales': [], 'other': []}
    for name, p in d.named_parameters():
        if 'lora_' in name: trainable['lora'].append(p)
        elif name.endswith('log_s'): trainable['scales'].append(p)
        elif 'norm' in name or not name.startswith('encoder.'):
            if name != 'encoder.embed_tokens.weight': trainable['other'].append(p)
    for group in trainable.values():
        for p in group: p.requires_grad_(True)
    return d, trainable


def both(wrapper, args):
    """FastDecision's logits and the final hidden states it reads them from."""
    ids, mask, opt, pool = args; d = wrapper.decision  # the arithmetic of FastDecision.forward
    h = wrapper.stack(d.encoder.embed_tokens(ids), mask).float()
    pooled = (h * pool[:, :, None]).sum(1) / pool.sum(1, keepdim=True); option = h[0].index_select(0, opt)
    v = F.normalize(d.w_p(d.norm(pooled)), dim=-1); u = F.normalize(d.w_o(d.norm(option)) + d.w_q(d.norm(h[:, 1])), dim=-1)
    return (u * v).sum(-1) / d.log_tau.exp() + d.cls_mlp(option).squeeze(-1), h


class SharedIndex:
    """The existing shared file as a `quantize_gptq.DataFile`: `add` returns a tensor's existing location after
    checking that the bytes rewrite would write equal the bytes in the file."""
    def __init__(self, path, index): self.path, self.index, self.location = path, index, SHARED
    def add(self, name, array):
        off, length, dtype, shape = self.index[name]; raw = np.ascontiguousarray(array).tobytes()
        with open(self.path, 'rb') as f: f.seek(off); stored = f.read(length)
        if raw != stored or np.dtype(array.dtype) != np.dtype(dtype) or tuple(array.shape) != shape: raise ValueError(f'shared tensor {name} differs from {self.path}')
        return self.index[name]


# ---- main --------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True, help='Pinned vela3-fast checkpoint')
    ap.add_argument('--models', type=Path, required=True, help='Folder with decision/fast (FP32), decision/fast/q4 and backbone/fast/q4')
    ap.add_argument('--out', type=Path, required=True, help='Root to write decision/fast/q4 and the run state into')
    ap.add_argument('--inputs', type=Path, help='The decision inputs of the q4 export (default <models>/backbone/fast/q4/inputs.json): its calibration inputs join the training data, its held-out inputs are a test set')
    ap.add_argument('--steps', type=int, default=6000, help='Training inputs seen (batch 1, gradient accumulation 8)')
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=2e-5, help='Learning rate of the corrections; normalization vectors and head use a tenth')
    ap.add_argument('--scales-lr', type=float, default=0.0, help='Learning rate of the group scales (log-factor); 0 keeps the shared ones, else the bundle stores its own fp16 copy')
    ap.add_argument('--kl-weight', type=float, default=1.0, help='Weight of the option KL divergence against the hidden-state error')
    ap.add_argument('--resume', action='store_true', help='Continue from <out>/repair/state.pt')
    ap.add_argument('--export-only', action='store_true', help='Export and gate the best saved state')
    ap.add_argument('--limit', type=int, help='Smoke run: cut the selection and test sets to this many inputs each (the result is not a gate)')
    ap.add_argument('--min-free-gb', type=float, default=8.0, help='Stop the job when the machine has less memory available (python/memguard.py)')
    a = ap.parse_args(); torch.manual_seed(0); torch.set_num_threads(6)
    guard(a.min_free_gb)
    root, models, out = a.source.resolve(), a.models.resolve(), a.out.resolve()
    if out == models: ap.error('--out must differ from --models: the gate compares against the unrepaired bundle there; copy the result after the gate')
    run_dir = out / 'repair'; run_dir.mkdir(parents=True, exist_ok=True)
    check_sources(root); sys.path.insert(0, str(root))
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    from vela_onnx.portable import FastDecision

    codes, embed, table, index, shared_path = published_q4(models)
    teacher = decision_model(root)
    student, trainable = student_from(root, codes, table, a.scales_lr > 0, teacher)
    log('INFO', f'student matrices={len(codes)} trainable ' + ' '.join(f'{k}={sum(p.numel() for p in v)}' for k, v in trainable.items()) + f' device={dev}')

    recorded = json.loads((a.inputs or models / 'backbone' / 'fast' / 'q4' / 'inputs.json').read_text())['decision']
    train_rows, val_rows = boolq('train'), boolq('validation')
    train = boolq_inputs(train_rows, 11) + [tuple(x) for x in recorded['calibration']]
    random.Random(13).shuffle(train)
    select = boolq_inputs(val_rows[:300], 14, 0)
    test = {'boolq': (boolq_inputs(val_rows[300:1300], 15, 0), [x[2] for x in val_rows[300:1300]]),
            'heldOut': ([tuple(x) for x in recorded['heldOut']], None)}
    labelled = [c for c in json.loads((REPO / 'tests/fixtures/decide-cases.json').read_text())['cases'] if 'images' not in c]
    test['labelled'] = ([(lines(c['state']), {'type': 'choice', 'instructions': c['instructions'], 'criteria': c['options'] if isinstance(c['options'], dict) else {o: None for o in c['options']}}
                          if c.get('options') else {'type': 'noul', 'instructions': c['instructions']}) for c in labelled], [c['expected'] for c in labelled])
    if a.limit: select, test = select[:a.limit], {k: (v[0][:a.limit], v[1][:a.limit] if v[1] else None) for k, v in test.items()}
    enc = lambda items: [teacher.encode(s, q) for s, q in items]
    train_recs, select_recs = enc(train), enc(select)
    test_recs = {k: (enc(items), gold) for k, (items, gold) in test.items()}  # encoded now, so the models can go before the ONNX checks
    log('INFO', f'inputs train={len(train_recs)} select={len(select_recs)} ' + ' '.join(f'{k}={len(v[0])}' for k, v in test.items()))

    sw, tw = FastDecision(student).eval().to(dev), FastDecision(teacher).eval().to(dev)
    args_of = lambda rec: tuple(t.to(dev) for t in fast_args(rec))

    def probs(wrapper, recs):
        with torch.no_grad(): return [torch.softmax(wrapper(*args_of(r)), -1).cpu().numpy() for r in recs]

    ref_select = probs(tw, select_recs)
    def score():
        p = probs(sw, select_recs)
        return {'flips': sum(int(x.argmax() != y.argmax()) for x, y in zip(p, ref_select)), 'meanShift': float(np.mean([np.abs(x - y).max() for x, y in zip(p, ref_select)]))}

    state_file, best_file = run_dir / 'state.pt', run_dir / 'best.pt'
    params = trainable['lora'] + trainable['scales'] + trainable['other']
    opt = torch.optim.AdamW([{'params': trainable['lora'], 'lr': a.lr}, {'params': trainable['scales'], 'lr': a.scales_lr}, {'params': trainable['other'], 'lr': a.lr / 10}], weight_decay=0.0)
    step, best = 0, None
    if (a.resume or a.export_only) and state_file.exists():
        st = torch.load(state_file, map_location=dev); [p.data.copy_(x) for p, x in zip(params, st['params'])]; opt.load_state_dict(st['opt']); step, best = st['step'], st['best']
        log('INFO', f'resumed step={step} best={best}')
    if not a.export_only:
        if best is None: best = {'step': 0, **score()}; torch.save([p.detach().cpu() for p in params], best_file); log('INFO', 'select ' + json.dumps(best))
        accum, total = 8, a.steps; t0 = time.time(); acc = {'kl': 0.0, 'rep': 0.0}
        sched = lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
        while step < total:
            rec = train_recs[step % len(train_recs)]; args = args_of(rec)
            with torch.no_grad(): t_logits, t_h = both(tw, args)
            s_logits, s_h = both(sw, args)
            kl = F.kl_div(F.log_softmax(s_logits, -1), F.softmax(t_logits, -1), reduction='sum')
            rep = ((s_h - t_h) ** 2).sum(-1).mean() / (t_h ** 2).sum(-1).mean()
            ((a.kl_weight * kl + rep) / accum).backward(); acc['kl'] += kl.item(); acc['rep'] += rep.item(); step += 1
            if step % accum == 0:
                for g in opt.param_groups: g.setdefault('base_lr', g['lr']); g['lr'] = g['base_lr'] * sched(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                log('INFO', f'step={step} kl={acc["kl"] / 200:.5f} rep={acc["rep"] / 200:.5f} seconds={time.time() - t0:.0f}'); acc = {'kl': 0.0, 'rep': 0.0}
            if step % a.eval_every == 0 or step == total:
                s = {'step': step, **score()}; log('INFO', 'select ' + json.dumps(s))
                if (s['flips'], s['meanShift']) < (best['flips'], best['meanShift']): best = s; torch.save([p.detach().cpu() for p in params], best_file)
                torch.save({'params': [p.detach().cpu() for p in params], 'opt': opt.state_dict(), 'step': step, 'best': best}, state_file)
        log('INFO', 'trained best=' + json.dumps(best))

    # ---- export the best state and gate it ----
    for p, x in zip(params, torch.load(best_file)): p.data.copy_(x.to(p.device))
    sw = sw.cpu(); student.eval()
    with torch.no_grad():
        for mod in student.modules():
            if isinstance(mod, Repaired):
                if mod.log_s is not None: mod.fixed = torch.from_numpy(mod.scales16().astype(np.float32))  # as stored: fp16
                mod.lora_B.mul_(mod.scale); mod.scale = 1.0
                for pname in ('lora_A', 'lora_B'):  # the exported student computes with the int8 corrections the graph stores
                    q, s = int8(getattr(mod, pname)); getattr(mod, pname).copy_(torch.from_numpy(q.astype(np.float32) * s[:, None]))
    quant = {}
    for name, mod in linears(student):
        q, s, z = codes[name]; rep_mod = student.get_submodule(name)
        own = rep_mod.log_s is not None  # repaired scales: the bundle's own fp16 copy, codes and zero points stay shared
        quant[key(mod.weight.detach().T.numpy())] = ('int4', q, rep_mod.scales16() if own else s, z, True, f'backbone.{name}', own)
    for mod in student.modules():
        if isinstance(mod, Repaired):
            for pname in ('lora_A', 'lora_B'):
                p = getattr(mod, pname); q, s = int8(p); quant[key(p.detach().T.numpy())] = ('int8', q, s, False, None)
    quant['embed'] = ('embed', *embed)
    fp32_table = teacher.encoder.embed_tokens.weight.detach().cpu().clone()
    fixtures = json.loads((models / 'decision' / 'fast' / 'fixtures.json').read_text())
    with torch.no_grad():
        ref = {f['name']: sw(*fast_args({'ids': f['ids'], 'pool': f['pool'], 'opt_pos': f['optionPositions']})).numpy() for f in fixtures}
        for mod in student.modules():
            if isinstance(mod, Repaired): mod.export = True
        student.encoder.embed_tokens.weight.copy_(fp32_table)
    graph_dir = run_dir / 'fp32-graph'; shutil.rmtree(graph_dir, ignore_errors=True); graph_dir.mkdir()
    names = ['input_ids', 'attention_mask', 'option_positions', 'pool_mask']
    axes = {'input_ids': {1: 'sequence'}, 'attention_mask': {1: 'sequence'}, 'option_positions': {0: 'options'}, 'pool_mask': {1: 'sequence'}, 'logits': {0: 'options'}}
    f0 = fixtures[0]; export_graph(sw, fast_args({'ids': f0['ids'], 'pool': f0['pool'], 'opt_pos': f0['optionPositions']}), graph_dir / 'decision.onnx', names, axes)
    dst = out / 'decision' / 'fast' / 'q4'; dst.mkdir(parents=True, exist_ok=True)
    for stale in ('manifest.json', 'decision.onnx', 'decision.onnx.data'): (dst / stale).unlink(missing_ok=True)
    # Free both PyTorch models before the rewrite and the ONNX Runtime checks load their own copies (python/memguard.py).
    del sw, tw, student, teacher, trainable, params, opt, codes; gc.collect()
    if dev.type == 'mps': torch.mps.empty_cache()
    counts = rewrite(graph_dir / 'decision.onnx', dst, 'decision', quant, SharedIndex(shared_path, index))
    shutil.rmtree(graph_dir, ignore_errors=True)
    log('INFO', 'rewrote ' + json.dumps(counts))

    tmp, tmp_old = assembled(dst, 'decision', shared_path), assembled(models / 'decision' / 'fast' / 'q4', 'decision', shared_path)
    try:
        q4x, q4, fp = exact_session(tmp / 'decision.onnx'), session(tmp / 'decision.onnx'), session(models / 'decision' / 'fast' / 'decision.onnx')
        unrepaired = session(tmp_old / 'decision.onnx')
        run = lambda s, args: s.run(None, {n.name: t.numpy() for n, t in zip(s.get_inputs(), args)})[0]
        checks, flipped = [], []
        for f in fixtures:
            args = fast_args({'ids': f['ids'], 'pool': f['pool'], 'opt_pos': f['optionPositions']}); got = run(q4x, args)
            checks.append(compare(ref[f['name']], got, f['name'] + '/onnx-vs-pytorch'))
            if int(np.argmax(run(q4, args))) != int(np.argmax(f['logits'])): flipped.append(f['name'])
        report = {'date': time.strftime('%Y-%m-%d'), 'tool': 'python/repair_decision.py', 'rank': RANK, 'alpha': ALPHA, 'steps': a.steps, 'best': best,
                  'counts': counts, 'parity': checks, 'fixtureFlips': flipped, 'sets': {}}
        for label, (recs, gold) in test_recs.items():
            rows = []
            for rec in recs:
                args = fast_args(rec)
                rows.append((rec['names'], softmax(run(fp, args)), softmax(run(q4, args)), softmax(run(unrepaired, args))))
            entry = {'inputs': len(rows)}
            for tag, i in (('repaired', 2), ('unrepaired', 3)):
                shifts = [float(np.abs(r[1] - r[i]).max()) for r in rows]
                entry[tag] = {'flips': sum(int(r[1].argmax() != r[i].argmax()) for r in rows), 'meanShift': round(float(np.mean(shifts)), 4), 'maxShift': round(float(np.max(shifts)), 4)}
            if gold is not None:
                pick = lambda r, i: r[0][int(r[i].argmax())]
                truth = [('yes' if g else 'no') if isinstance(g, bool) else g for g in gold]
                entry['correct'] = {tag: sum(int(pick(r, i) == t) for r, t in zip(rows, truth)) for tag, i in (('fp32', 1), ('repaired', 2), ('unrepaired', 3))}
            report['sets'][label] = entry; log('INFO', f'set={label} ' + json.dumps(entry))
    finally:
        shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(tmp_old, ignore_errors=True)
    rates = {k: report['sets'][k]['repaired']['flips'] / report['sets'][k]['inputs'] for k in ('heldOut', 'boolq')}
    passed = not flipped and all(r <= 0.01 for r in rates.values()) and not a.limit
    extra = {'validated': passed, 'repair': {'tool': 'python/repair_decision.py', 'rank': RANK,
                                             'trained': 'rank-16 corrections (int8), normalization vectors, head' + (', group scales (own fp16 copy)' if a.scales_lr > 0 else ''),
                                             'data': 'BoolQ train (google/boolq@35b264d0), generated records, repository paragraphs'}}
    if not passed: extra['rejected'] = f'fixture flips {flipped}, flip rates {rates}'
    m = manifest_for(models / 'decision' / 'fast', dst, 'decision', shared_path, extra, (models, out))
    m['quantization'] += '; rank-16 repair corrections int8 per channel'
    dump(dst / 'manifest.json', m)
    dump(dst / 'parity-python.json', {'passed': True, 'checks': checks, 'scope': 'repaired q4 graph in ONNX Runtime CPU vs the PyTorch student'})
    report['validated'] = passed; dump(run_dir / 'report.json', report)
    log('INFO', f'done validated={passed} report={run_dir / "report.json"}')


if __name__ == '__main__':
    sys.exit(main())
