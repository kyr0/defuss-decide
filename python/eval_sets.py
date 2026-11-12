#!/usr/bin/env python3
"""The labelled sets of the accuracy gate for quantized task bundles (scripts/verify-accuracy.ts).

    uv run python python/eval_sets.py [--out input/eval]

Writes
    <out>/ner/cases.json   3,000 test sentences of Few-NERD (supervised split) with their coarse entity spans
    <out>/box/cases.json   1,500 validation expressions of RefCOCO with the labelled box, and <out>/box/images/

WHY public labelled sets and not the bundles' few fixtures: a q4 answer can differ from FP32's on an input where FP32
itself is close to a threshold, so "no fixture changes" rejects a bundle that is as accurate as FP32.
VERIFIED: (2026-11-12, Node.js, WebGPU) ner q4 has an entity F1 of 0.560 against 0.557 for FP32 on these sentences while
24 % of them get another entity set, 94 % of the dropped spans with an FP32 score below 0.7 (threshold 0.6); vision-box
q4 points into the labelled region for 70.3 % of these expressions against 70.8 %.

Both selections are deterministic: a seeded shuffle of the pinned parquet file, whose checksum is checked. The COCO
images come over plain HTTP, so their checksums are checked too. Few-NERD: CC BY-SA 4.0. RefCOCO: annotations of
COCO train2014 images; the files stay in the gitignored <out> and are not redistributed.
"""
import argparse, hashlib, json, random, sys, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FEWNERD = ('DFKI-SLT/few-nerd', '205f3e9c9f3577ea2561d43f2f62dc249ab92d5b', 'supervised/test-00000-of-00001.parquet',
           'b7ad746fcbeb68fcc235ba7142d7c3723ea2dc39930089e947284defecf300c6')
REFCOCO = ('jxu124/refcoco', '9fba8200c5326e996f789191f095bd464ef1d09e', 'data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet',
           'df03b1b16873f92727f3df010afb3c7260e8396a86592d3c4fee053daca08c05')
COCO = 'http://images.cocodataset.org/train2014/'
# sha256 over the sorted lines "<file name> <sha256 of the file>" of the images the 1,500 expressions use.
IMAGES_SHA256 = 'e51838ccdb5bfde75bbab8a73f666bce791b74d9828828f4c3371f22cf86a43b'
SENTENCES, EXPRESSIONS, SEED = 3000, 1500, 7
# Few-NERD's coarse types as the labels the question asks for; "other" has no label a caller would name.
LABELS = {'art': 'work of art', 'building': 'building', 'event': 'event', 'location': 'location', 'organization': 'organization',
          'person': 'person', 'product': 'product'}
TYPES = ['O', 'art', 'building', 'event', 'location', 'organization', 'other', 'person', 'product']


def log(level, msg):
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}.{int(time.time() * 1000) % 1000:03d}Z {level} {msg}', flush=True)


def sha256(path):
    with open(path, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(source):
    """The rows of one pinned parquet file; refuses a file with another checksum."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    repo, revision, file, digest = source
    path = hf_hub_download(repo, file, repo_type='dataset', revision=revision)
    if sha256(path) != digest: raise ValueError(f'{repo} {file}: checksum mismatch')
    return pq.read_table(path).to_pylist()


def sentences(table):
    """Few-NERD sentences of 5 to 60 tokens as text with character spans: tokens joined by one space, a run of equal
    coarse tags is one entity (the set's own convention, it has no B- and I- tags)."""
    random.Random(SEED).shuffle(table); cases = []
    for r in table:
        if len(cases) == SENTENCES: break
        tokens, tags = r['tokens'], r['ner_tags']
        if not 5 <= len(tokens) <= 60: continue
        starts, pos = [], 0
        for t in tokens: starts.append(pos); pos += len(t) + 1
        gold, i = [], 0
        while i < len(tokens):
            j = i
            while j + 1 < len(tokens) and tags[j + 1] == tags[i]: j += 1
            if TYPES[tags[i]] in LABELS: gold.append({'start': starts[i], 'end': starts[j] + len(tokens[j]), 'label': LABELS[TYPES[tags[i]]]})
            i = j + 1
        cases.append({'id': r['id'], 'text': ' '.join(tokens), 'gold': gold})
    return cases


def expressions(table, images):
    """RefCOCO expressions (the first sentence of each reference) with the labelled box as fractions of the image, and
    the image fetched into `images` unless it is there."""
    random.Random(SEED).shuffle(table); cases = []
    for r in table[:EXPRESSIONS]:
        info, ann = json.loads(r['raw_image_info']), json.loads(r['raw_anns'])
        path = images / info['file_name']
        if not path.exists():
            with urllib.request.urlopen(COCO + info['file_name'], timeout=60) as response: path.write_bytes(response.read())
        x, y, w, h = ann['bbox']; W, H = info['width'], info['height']
        cases.append({'id': r['ref_id'], 'image': f'images/{path.name}', 'expression': r['sentences'][0]['raw'],
                      'box': {'left': x / W, 'top': y / H, 'right': (x + w) / W, 'bottom': (y + h) / H}})
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=REPO / 'input' / 'eval')
    out = ap.parse_args().out.resolve()
    for d in (out / 'ner', out / 'box' / 'images'): d.mkdir(parents=True, exist_ok=True)

    cases = sentences(rows(FEWNERD))
    (out / 'ner' / 'cases.json').write_text(json.dumps({
        'set': {'name': 'Few-NERD, supervised test split', 'dataset': FEWNERD[0], 'revision': FEWNERD[1], 'license': 'CC BY-SA 4.0',
                'selection': f'the first {SENTENCES} sentences of 5 to 60 tokens after a shuffle with seed {SEED}; the coarse types except "other"'},
        'labels': sorted(LABELS.values()), 'cases': cases}))
    log('INFO', f'ner cases={len(cases)} entities={sum(len(c["gold"]) for c in cases)} file={out / "ner" / "cases.json"}')

    cases = expressions(rows(REFCOCO), out / 'box' / 'images')
    names = sorted({c['image'] for c in cases})
    digest = hashlib.sha256(''.join(f'{Path(n).name} {sha256(out / "box" / n)}\n' for n in names).encode()).hexdigest()
    if digest != IMAGES_SHA256: raise ValueError(f'the COCO images have checksum {digest}, expected {IMAGES_SHA256}; remove {out / "box" / "images"} and run again')
    (out / 'box' / 'cases.json').write_text(json.dumps({
        'set': {'name': 'RefCOCO, validation split', 'dataset': REFCOCO[0], 'revision': REFCOCO[1], 'images': 'COCO train2014', 'imagesSha256': digest,
                'selection': f'the first {EXPRESSIONS} references after a shuffle with seed {SEED}, each with its first expression'},
        'cases': cases}))
    log('INFO', f'box cases={len(cases)} images={len(names)} file={out / "box" / "cases.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
