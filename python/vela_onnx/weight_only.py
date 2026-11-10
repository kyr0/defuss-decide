"""Portable W8A32: signed per-output-channel weights + standard DequantizeLinear.
No activation quantization and no proprietary operators. Reduces transfer size;
CPU backends may expand weights to FP32, so this is not a RAM/speed guarantee.
"""
import numpy as np
import onnx
from onnx import numpy_helper as nh, helper

def quantize_weights(source,destination):
    """Replace every 2-D FP32 MatMul weight by INT8 values, a per-output-channel scale and a DequantizeLinear node.

    Symmetric (zero point 0, range -127..127), so the dequantized weight is exactly `q * scale`; the scale floor keeps
    all-zero columns finite. Used by the v0.1 decision experiment (python/quantize.py); returns the number of weights. VERIFIED: that candidate
    failed the text parity gate (reports/quantization-rejected.json)."""
    m=onnx.load(str(source));initial={x.name:x for x in m.graph.initializer}
    candidates={n.input[1] for n in m.graph.node if n.op_type=='MatMul' and len(n.input)>1 and n.input[1] in initial}
    nodes=[];count=0
    for name in sorted(candidates):
        t=initial[name]
        if t.data_type!=onnx.TensorProto.FLOAT or len(t.dims)!=2:continue
        w=nh.to_array(t);scale=np.maximum(np.max(np.abs(w),axis=0)/127,np.finfo(np.float32).tiny).astype(np.float32)
        q=np.clip(np.rint(w/scale),-127,127).astype(np.int8)
        qn,sn,zn=name+'__w8',name+'__scale',name+'__zero'
        m.graph.initializer.extend([nh.from_array(q,qn),nh.from_array(scale,sn),nh.from_array(np.zeros_like(scale,dtype=np.int8),zn)])
        nodes.append(helper.make_node('DequantizeLinear',[qn,sn,zn],[name],name=name+'__dequant',axis=1))
        m.graph.initializer.remove(t);count+=1
    old=list(m.graph.node);del m.graph.node[:];m.graph.node.extend(nodes+old)
    onnx.save(m,str(destination));onnx.checker.check_model(str(destination))
    return count
