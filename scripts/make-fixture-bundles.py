#!/usr/bin/env python3
"""Build the committed fixture bundles in tests/fixtures/bundles/ from the locally exported real bundles.

WHY fixture bundles exist: the real bundles are 0.2–1.8 GB each and need a multi-GB Python export, so CI cannot have
them. A fixture bundle keeps everything that defines the JavaScript contract (the real manifest configuration, the
real tokenizer and the real upstream Python fixtures with their reference ids, offsets, logits and answers) and only
swaps the neural network for a tiny deterministic ONNX graph with the same input and output signature. Tests can then
check tokenization, formatting, routing and decoding against upstream Python in CI, and run the complete
load → tensors → ONNX Runtime → decode → dispose path in Node.js, Bun, Chromium and WebKit. Weight parity with the real
checkpoints stays the job of `make parity`, which needs the real bundles.

Run after an export changed a manifest, tokenizer or fixture (needs the bundles under models/ and the `onnx` package):

    uv run --frozen scripts/make-fixture-bundles.py

The graphs are written with external data (`.onnx` + `.onnx.data`), the same layout the real bundles use, so the
browser loader's external-data path is exercised too.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "vela3-fast-onnx"
OUT = ROOT / "tests" / "fixtures" / "bundles"
TEXT_TASKS = ["span", "span_set", "ner", "relation", "evidence"]
VISION_TASKS = ["span", "ocr", "box"]
# opset 17 and IR 8 match the real exports, so the fixture graphs load under the same ONNX Runtime constraints.
OPSET = [helper.make_opsetid("", 17)]


def save(graph, path: Path) -> dict:
    """Save `graph` with its float weights in a sibling `.onnx.data` file and return its manifest graph entry.

    Integer initializers (axes, shapes, pads) become Constant nodes first: ONNX Runtime's shape inference must read
    them while loading, before external data is resolved, exactly as in the real exports where only weights are external.
    """
    model = fixture_model(graph)
    data = path.name + ".data"
    (path.parent / data).unlink(missing_ok=True)
    onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location=data, size_threshold=0)
    return {"file": path.name, "externalData": [data], "sizeBytes": path.stat().st_size,
            "totalSizeBytes": path.stat().st_size + (path.parent / data).stat().st_size}


def fixture_model(graph):
    """The checked model of `graph`, with its integer initializers (axes, shapes, pads) turned into Constant nodes:
    ONNX Runtime's shape inference must read them while loading, before external data is resolved, exactly as in the
    real exports where only weights are external."""
    for t in [t for t in graph.initializer if t.data_type == TensorProto.INT64]:
        graph.node.insert(0, helper.make_node("Constant", [], [t.name], value=t))
        graph.initializer.remove(t)
    model = helper.make_model(graph, opset_imports=OPSET, producer_name="defuss-decision fixture")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def save_split(graph, path: Path, shared: Path, names: set) -> dict:
    """Like save(), but the weights in `names` go to `shared`, a file in another folder that several bundles use, as
    the q4 bundles of python/quantize_gptq.py share their backbone; the manifest entry names it as {path, file}.
    VERIFIED: the split bundle answers its fixtures exactly like fast-decision in Node.js, Bun, Chromium and WebKit
    (tests/unit/engine.test.ts, tests/browser/web.test.ts)."""
    model = fixture_model(graph)
    own, files = path.name + ".data", {}
    for t in model.graph.initializer:
        target = (shared, shared.name) if t.name in names else (path.parent / own, own)
        f = files.setdefault(target[0], open(target[0], "wb"))
        raw = numpy_helper.to_array(t).tobytes(); offset = f.tell(); f.write(raw)
        t.ClearField("raw_data"); t.data_location = TensorProto.EXTERNAL
        for key, value in (("location", target[1]), ("offset", str(offset)), ("length", str(len(raw)))):
            entry = t.external_data.add(); entry.key, entry.value = key, value
    for f in files.values():
        f.close()
    onnx.save(model, str(path))
    return {"file": path.name, "externalData": [own, {"path": shared.name, "file": os.path.relpath(shared, path.parent)}],
            "sizeBytes": path.stat().st_size, "totalSizeBytes": path.stat().st_size + (path.parent / own).stat().st_size + shared.stat().st_size}


def const(name, values, dtype=np.float32):
    """Initializer from a Python value (float32 unless given); integers become Constant nodes in save()."""
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name)


# Symbolic dimensions mirror the real graphs' dynamic axes, so ONNX Runtime accepts any sequence, word or label count.
SHAPES = {"input_ids": [1, "sequence"], "attention_mask": [1, "sequence"], "image_indices": [1, "sequence"],
          "pool_mask": [1, "sequence"], "word_positions": ["words"], "label_positions": ["labels"],
          "readout_positions": ["readouts"], "option_positions": ["options"], "image_features": ["image_tokens", 768],
          "option_mask": ["options", "sequence"]}


def ints(name):
    """INT64 graph input with the symbolic shape of SHAPES."""
    return helper.make_tensor_value_info(name, TensorProto.INT64, SHAPES[name])


def floats(name, shape=None):
    """FLOAT graph value with an explicit shape or the one in SHAPES."""
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape or SHAPES[name])


def mask_scale(nodes):
    """1.0 from attention_mask: every declared input takes part, as in the real graphs."""
    nodes += [helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
              helper.make_node("ReduceMax", ["mask_f"], ["mask_one"], keepdims=0)]


def token_values(nodes):
    """input_ids as a float vector [S], scaled by the attention mask."""
    nodes += [helper.make_node("Squeeze", ["input_ids", "axis0"], ["ids"]),
              helper.make_node("Cast", ["ids"], ["ids_raw"], to=TensorProto.FLOAT),
              helper.make_node("Mul", ["ids_raw", "mask_one"], ["ids_f"])]


def fast_task_graph():
    """logits[w, l] = 6·sin(0.37·id(word w) + 1.3·position(label l)) − 1: deterministic spans for every label."""
    nodes = []
    mask_scale(nodes)
    token_values(nodes)
    nodes += [
        helper.make_node("Gather", ["ids_f", "word_positions"], ["word_ids"]),
        helper.make_node("Cast", ["label_positions"], ["label_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["word_ids", "axis1"], ["w_col"]),
        helper.make_node("Unsqueeze", ["label_f", "axis0"], ["l_row"]),
        helper.make_node("Mul", ["w_col", "word_freq"], ["w_term"]),
        helper.make_node("Mul", ["l_row", "label_freq"], ["l_term"]),
        helper.make_node("Add", ["w_term", "l_term"], ["phase"]),
        helper.make_node("Sin", ["phase"], ["wave"]),
        helper.make_node("Mul", ["wave", "amplitude"], ["scaled"]),
        helper.make_node("Sub", ["scaled", "offset"], ["logits"]),
    ]
    init = [const("axis0", [0], np.int64), const("axis1", [1], np.int64), const("word_freq", [0.37]),
            const("label_freq", [1.3]), const("amplitude", [6.0]), const("offset", [1.0])]
    return helper.make_graph(nodes, "fixture_fast_task",
                             [ints("input_ids"), ints("attention_mask"), ints("word_positions"), ints("label_positions")],
                             [floats("logits", ["words", "labels"])], init)


def vision_encoder_graph():
    """image_features[1, 64, 768] = per-cell (R − (G + B) / 2) redness of an 8×8 grid, spread over 768 channels."""
    projection = np.outer([1.0, -0.5, -0.5], 1 + np.arange(768) / 768).astype(np.float32)
    nodes = [
        helper.make_node("AveragePool", ["pixel_values"], ["cells"], kernel_shape=[64, 64], strides=[64, 64]),
        helper.make_node("Reshape", ["cells", "cells_shape"], ["flat"]),
        helper.make_node("Transpose", ["flat"], ["per_cell"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["per_cell", "projection"], ["image_features"]),
    ]
    init = [const("cells_shape", [1, 3, 64], np.int64), const("projection", projection)]
    return helper.make_graph(nodes, "fixture_vision_encoder",
                             [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 512, 512])],
                             [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, [1, 64, 768])], init)


def token_redness(nodes, index_input):
    """Mean feature per index: index k > 0 reads feature row k − 1, index 0 (a text token) reads a zero row."""
    nodes += [
        helper.make_node("Pad", ["image_features", "pad_rows"], ["padded"]),
        helper.make_node("Gather", ["padded", index_input], ["token_features"]),
        helper.make_node("ReduceMean", ["token_features"], ["redness"], axes=[1], keepdims=0),
    ]


def vision_task_graph():
    """logits[r] = 8·redness(readout r) + 5·sin(0.37·id(readout r)): boxes follow red pixels, text spans follow ids."""
    nodes = []
    mask_scale(nodes)
    token_values(nodes)
    nodes += [helper.make_node("Squeeze", ["image_indices", "axis0"], ["indices"]),
              helper.make_node("Gather", ["indices", "readout_positions"], ["readout_indices"])]
    token_redness(nodes, "readout_indices")
    nodes += [
        helper.make_node("Gather", ["ids_f", "readout_positions"], ["readout_ids"]),
        helper.make_node("Mul", ["readout_ids", "id_freq"], ["phase"]),
        helper.make_node("Sin", ["phase"], ["wave"]),
        helper.make_node("Mul", ["wave", "text_amplitude"], ["text_term"]),
        helper.make_node("Mul", ["redness", "image_amplitude"], ["image_term"]),
        helper.make_node("Add", ["text_term", "image_term"], ["logits"]),
    ]
    init = [const("axis0", [0], np.int64), const("pad_rows", [1, 0, 0, 0], np.int64), const("id_freq", [0.37]),
            const("text_amplitude", [5.0]), const("image_amplitude", [8.0])]
    return helper.make_graph(nodes, "fixture_vision_task",
                             [ints("input_ids"), ints("attention_mask"), floats("image_features"), ints("image_indices"),
                              ints("readout_positions")], [floats("logits", ["readouts"])], init)


def fast_decision_graph():
    """logits[o] = 3·sin(0.01·id(first option token) + 0.001·mean id of the pooled state)."""
    nodes = []
    mask_scale(nodes)
    token_values(nodes)
    nodes += [
        helper.make_node("Squeeze", ["pool_mask", "axis0"], ["pool"]),
        helper.make_node("Mul", ["ids_f", "pool"], ["pooled_ids"]),
        helper.make_node("ReduceSum", ["pooled_ids"], ["pool_sum"], keepdims=0),
        helper.make_node("ReduceSum", ["pool"], ["pool_count"], keepdims=0),
        helper.make_node("Div", ["pool_sum", "pool_count"], ["pool_mean"]),
        helper.make_node("Add", ["option_positions", "one"], ["first_tokens"]),
        helper.make_node("Gather", ["ids_f", "first_tokens"], ["option_ids"]),
        helper.make_node("Mul", ["option_ids", "option_freq"], ["o_term"]),
        helper.make_node("Mul", ["pool_mean", "state_freq"], ["s_term"]),
        helper.make_node("Add", ["o_term", "s_term"], ["phase"]),
        helper.make_node("Sin", ["phase"], ["wave"]),
        helper.make_node("Mul", ["wave", "amplitude"], ["logits"]),
    ]
    init = [const("axis0", [0], np.int64), const("one", [1], np.int64), const("option_freq", [0.01]),
            const("state_freq", [0.001]), const("amplitude", [3.0])]
    return helper.make_graph(nodes, "fixture_fast_decision",
                             [ints("input_ids"), ints("attention_mask"), ints("option_positions"), floats("pool_mask")],
                             [floats("logits", ["options"])], init)


def vision_decision_graph():
    """logits[o] = mean over option o's tokens of (4·redness + sin(0.37·id))."""
    nodes = []
    mask_scale(nodes)
    token_values(nodes)
    nodes += [helper.make_node("Squeeze", ["image_indices", "axis0"], ["indices"])]
    token_redness(nodes, "indices")
    nodes += [
        helper.make_node("Mul", ["ids_f", "id_freq"], ["phase"]),
        helper.make_node("Sin", ["phase"], ["wave"]),
        helper.make_node("Mul", ["redness", "image_amplitude"], ["image_term"]),
        helper.make_node("Add", ["wave", "image_term"], ["token_value"]),
        helper.make_node("Unsqueeze", ["token_value", "axis1"], ["token_col"]),
        helper.make_node("MatMul", ["option_mask", "token_col"], ["option_sum2"]),
        helper.make_node("Squeeze", ["option_sum2", "axis1"], ["option_sum"]),
        # ReduceSum takes axes as an input since opset 13 (ReduceMean keeps the attribute until opset 18).
        helper.make_node("ReduceSum", ["option_mask", "axis1"], ["option_len"], keepdims=0),
        helper.make_node("Div", ["option_sum", "option_len"], ["logits"]),
    ]
    init = [const("axis0", [0], np.int64), const("axis1", [1], np.int64), const("pad_rows", [1, 0, 0, 0], np.int64),
            const("id_freq", [0.37]), const("image_amplitude", [4.0])]
    return helper.make_graph(nodes, "fixture_vision_decision",
                             [ints("input_ids"), ints("attention_mask"), floats("image_features"), ints("image_indices"),
                              floats("option_mask")], [floats("logits", ["options"])], init)


def fixture_manifest(real: dict, name: str) -> dict:
    """The real manifest minus machine-local fields, marked so no one mistakes it for model weights."""
    m = {k: v for k, v in real.items() if k not in ("quantizedFrom", "quantization", "graphs")}
    m.update(precision="fp32", fixture=True, graphs={},
             note="Test fixture: real configuration, tokenizer and Python fixtures; tiny synthetic ONNX graphs. "
                  "Outputs are meaningless; see scripts/make-fixture-bundles.py.")
    m["tokenizer"] = "../tokenizer-" + real["kind"] + ".json"
    m["tokenizerConfig"] = "../tokenizer_config-" + real["kind"] + ".json"
    m["fixtureName"] = name
    return m


def copy_shared(real_dir: Path, kind: str):
    """Copy the real tokenizer once per model kind; the fixture manifests reference it as ../tokenizer-<kind>.json."""
    shutil.copyfile(real_dir / "tokenizer.json", OUT / f"tokenizer-{kind}.json")
    shutil.copyfile(real_dir / "tokenizer_config.json", OUT / f"tokenizer_config-{kind}.json")


def main():
    """Rebuild every fixture bundle from the exported real bundles: their manifests, tokenizers and fixtures, new graphs.

    VERIFIED: rerun in the uv environment on 2026-11-02, it reproduced tests/fixtures/bundles byte for byte."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.parse_args()
    if not (MODELS / "ner" / "fp32" / "manifest.json").is_file():
        raise SystemExit("needs the exported real bundles under models/vela3-fast-onnx/ (see README: Getting the models)")
    OUT.mkdir(parents=True, exist_ok=True)
    copy_shared(MODELS / "ner" / "fp32", "fast")
    copy_shared(MODELS / "vision-span" / "fp32", "vision")
    for kind, tasks in (("fast", TEXT_TASKS), ("vision", VISION_TASKS)):
        for task in tasks:
            name = task if kind == "fast" else f"vision-{task}"
            src, dst = MODELS / name / "fp32", OUT / name
            dst.mkdir(parents=True, exist_ok=True)
            real = json.loads((src / "manifest.json").read_text())
            m = fixture_manifest(real, name)
            if kind == "fast":
                m["graphs"]["task"] = save(fast_task_graph(), dst / "task.onnx")
            else:
                m["graphs"]["vision"] = save(vision_encoder_graph(), dst / "vision.onnx")
                m["graphs"]["task"] = save(vision_task_graph(), dst / "task.onnx")
                shutil.copyfile(src / "sample.png", dst / "sample.png")
            shutil.copyfile(src / "fixtures.json", dst / "fixtures.json")
            (dst / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    # v0.1 decision bundles: no local export exists, so their configuration comes from the task manifests (same
    # tokenizer, markers, image processor and prefixes) and their fixtures from the recorded v0.1 reports.
    task_fast = json.loads((MODELS / "ner" / "fp32" / "manifest.json").read_text())
    task_vision = json.loads((MODELS / "vision-span" / "fp32" / "manifest.json").read_text())
    for kind, real, graph in (("fast", task_fast, fast_decision_graph), ("vision", task_vision, vision_decision_graph)):
        dst = OUT / f"{kind}-decision"
        dst.mkdir(parents=True, exist_ok=True)
        m = fixture_manifest(real, f"{kind}-decision")
        for key in ("task", "taskConfig", "spanConfig", "maxAnswerTokens"):
            m.pop(key, None)
        m.update(capabilities=["choice", "noul"], scope="ordinary decisions; no task adapters",
                 maxImages=3 if kind == "vision" else 0)
        if kind == "vision":
            m["graphs"]["vision"] = save(vision_encoder_graph(), dst / "vision.onnx")
            for png in ("red.png", "blue.png"):
                shutil.copyfile(ROOT / "reports" / "vision" / png, dst / png)
        m["graphs"]["decision"] = save(graph(), dst / "decision.onnx")
        shutil.copyfile(ROOT / "reports" / kind / "fixtures.json", dst / "fixtures.json")
        (dst / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
        if kind == "fast":
            # The same graph with two weights in a shared file of the parent folder, like the shared tokenizers: the loaders'
            # shared-file path.
            split = OUT / "fast-decision-split"
            split.mkdir(exist_ok=True)
            m = {**m, "fixtureName": "fast-decision-split", "graphs": {
                "decision": save_split(graph(), split / "decision.onnx", OUT / "backbone.onnx.data", {"option_freq", "amplitude"})}}
            shutil.copyfile(ROOT / "reports" / kind / "fixtures.json", split / "fixtures.json")
            (split / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "bundles": sorted(p.name for p in OUT.iterdir() if p.is_dir())}))


if __name__ == "__main__":
    main()
