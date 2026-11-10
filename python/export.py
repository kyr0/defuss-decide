#!/usr/bin/env python3
"""Export decision-only Vela packages, with mandatory PyTorch/ORT parity gates."""
import argparse, gc, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from memguard import guard
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from vela_onnx.portable import FastDecision, VisionEncoder, VisionDecision

PINS = {'fast': ('vllm-sr/vela3-fast','fa60e3aa41132c60bf05a84cabfdced39cb6d4f7'),
        'vision': ('vllm-sr/vela3-fast-vision','c1f651cafddca43860a855ee434a35697949bce1')}

def dump(path, value):
    """Write JSON as UTF-8 text (`ensure_ascii=False`) so fixtures keep their Unicode readable."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def softmax(x):
    """Softmax in float64, the reference the JavaScript `probabilities` is compared with."""
    x = np.asarray(x, np.float64); p = np.exp(x-x.max()); return p/p.sum()

def compare(reference, actual, label, atol=3e-3):
    """Fail on any non-finite or out-of-tolerance value, else return the measured maximum error for the report."""
    a,b=np.asarray(reference),np.asarray(actual)
    if not np.isfinite(b).all(): raise AssertionError(label+': non-finite result')
    error=float(np.max(np.abs(a-b)))
    np.testing.assert_allclose(b,a,atol=atol,rtol=3e-4,err_msg=label)
    return {'case':label,'max_abs_error':error}

def export_graph(model, args, path, names, axes):
    """Export one decision graph at opset 17 with the TorchScript exporter and external data, then run the ONNX
    checker."""
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model,args,str(path),input_names=names,output_names=['logits' if 'decision' in path.name else 'image_features'],
                          dynamic_axes=axes,opset_version=17,dynamo=False,external_data=True,do_constant_folding=True)
    import onnx
    onnx.checker.check_model(str(path))

def session(path):
    """CPU ONNX Runtime session with telemetry off and two intra-op threads."""
    import onnxruntime as ort
    ort.disable_telemetry_events()
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])

def run(s,args):
    """Run a session on torch tensors, matched to the graph inputs by position."""
    return s.run(None,{n.name:t.detach().cpu().numpy() for n,t in zip(s.get_inputs(),args)})[0]

def resolve(kind,local):
    """The checkpoint folder: `--source` when given, else a download of only the decision files at the pinned revision."""
    if local: return Path(local).resolve()
    from huggingface_hub import snapshot_download
    repo,rev=PINS[kind]
    patterns=['*.py','*.json','LICENSE*'] + (['model.safetensors'] if kind=='fast' else ['backbone_h2.safetensors','decision_adapter.safetensors'])
    return Path(snapshot_download(repo,revision=rev,allow_patterns=patterns,ignore_patterns=['text/*']))

def fast_args(rec):
    """Graph inputs of one rendered text decision: ids, attention mask, option positions and the state pooling mask."""
    n=len(rec['ids']);pool=torch.zeros(1,n);pool[:,rec['pool'][0]:rec['pool'][1]]=1
    return torch.tensor([rec['ids']]),torch.ones(1,n,dtype=torch.long),torch.tensor(rec['opt_pos']),pool

def vision_args(example,image_features,image_id):
    """Graph inputs of one rendered vision decision. The k-th image token gets index k (text tokens 0), and each
    option's token span becomes one row of the pooling mask."""
    ids=example['input_ids'];idx=[];k=0
    for t in ids:
        if t==image_id:k+=1;idx.append(k)
        else:idx.append(0)
    masks=torch.zeros(len(example['spans']),len(ids))
    for j,(a,b) in enumerate(example['spans']):masks[j,a:b]=1
    return torch.tensor([ids]),torch.ones(1,len(ids),dtype=torch.long),image_features.reshape(-1,768),torch.tensor([idx]),masks

def main():
    """Check the source hashes, export the graphs, compare PyTorch with ONNX Runtime on every fixture, and only
    then write `manifest.json`: a bundle without a passed comparison is never loadable. VERIFIED: VelaDecision.load
    refuses bundles without `validated: true` (tests/unit/engine.test.ts)."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=PINS,required=True);ap.add_argument('--source',help='Already downloaded pinned snapshot')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--max-tokens',type=int,default=2048)
    ap.add_argument('--min-free-gb',type=float,default=8.0,help='Stop the job when the machine has less memory available (python/memguard.py)')
    args=ap.parse_args();guard(args.min_free_gb);torch.set_num_threads(2)
    if not 64<=args.max_tokens<=32768:ap.error('--max-tokens must be 64..32768')
    root=resolve(args.model,args.source)
    locks=json.loads((Path(__file__).parent/'source-lock.json').read_text())[args.model]
    for name,expected in locks.items():
        with (root/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise ValueError(f'Source checksum mismatch: {name}')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    # Never leave a stale ready manifest after an interrupted or failed re-export.
    (out/'manifest.json').unlink(missing_ok=True)
    sys.path.insert(0,str(root));report=[];fixtures=[]
    cfg=AutoConfig.from_pretrained(root,trust_remote_code=True)
    for f in ['tokenizer.json','tokenizer_config.json','LICENSE','LICENSE-APACHE']:
        if (root/f).exists():shutil.copyfile(root/f,out/f)
    manifest={'format':'defuss-vela-onnx/1','model':PINS[args.model][0],'revision':PINS[args.model][1],
              'kind':args.model,'precision':'fp32','maxTokens':args.max_tokens,'maxImages':3 if args.model=='vision' else 0,
              'capabilities':['choice','noul'],'graphs':{},'tokenizer':'tokenizer.json','tokenizerConfig':'tokenizer_config.json',
              'scope':'ordinary decisions; no task adapters','validated':False}
    if args.model=='fast':
        from vela3_encoder import DecisionEncoder
        d=DecisionEncoder.load(root,device='cpu',max_len=args.max_tokens);d.encoder.set_attn_implementation('eager')
        wrapper=FastDecision(d).eval()
        cases=[('billing','My invoice includes a duplicate charge.',{'type':'choice','instructions':'Which team should handle this request?','criteria':{'billing':None,'technical':None,'sales':None}}),
               ('noul','The printer does not turn on.',{'type':'noul','instructions':'Does the user need technical support?'}),
               ('json',{'user':'Grüße 👋','items':[1,True,None]},{'type':'choice','instructions':{'task':'classify'},'criteria':{'unicode':'Text contains non-ASCII characters','ascii':'Only ASCII','other':None}}),
               ('long','A garden contains trees. '*70,{'type':'choice','instructions':'What is described?','criteria':{'garden':'plants and trees','server':'computers'}}),
               ('empty','',{'type':'choice','instructions':'Choose an option.','criteria':{'only':None}})]
        first=fast_args(d.encode(cases[0][1],cases[0][2]))
        names=['input_ids','attention_mask','option_positions','pool_mask']
        axes={'input_ids':{1:'sequence'},'attention_mask':{1:'sequence'},'option_positions':{0:'options'},'pool_mask':{1:'sequence'},'logits':{0:'options'}}
        print('Exporting fast decision graph',flush=True);export_graph(wrapper,first,out/'decision.onnx',names,axes)
        s=session(out/'decision.onnx')
        for name,state,q in cases:
            rec=d.encode(state,q);tensors=fast_args(rec)
            with torch.no_grad():
                original=d._logits(d.encoder(input_ids=tensors[0],attention_mask=tensors[1]).last_hidden_state,[rec])[0].numpy()
                adapted=wrapper(*tensors).numpy()
            report.append(compare(original,adapted,name+'/wrapper'));actual=run(s,tensors);report.append(compare(original,actual,name+'/onnx'))
            report.append(compare(softmax(original),softmax(actual),name+'/probabilities',atol=3e-4))
            if np.argmax(original)!=np.argmax(actual):raise AssertionError('argmax changed: '+name)
            fixtures.append({'name':name,'state':state,'question':q,'ids':rec['ids'],'optionPositions':rec['opt_pos'],'pool':rec['pool'],'names':rec['names'],'logits':original.tolist()})
        manifest.update(markers=d.mk,bosId=d.tok.bos_token_id,eosId=d.tok.eos_token_id,padId=d.tok.pad_token_id)
    else:
        from PIL import Image, ImageDraw
        import uvb_core as C
        # Load only the decision-trained backbone: raw/box backbones are different weights.
        enc=AutoModel.from_config(cfg,trust_remote_code=True,attn_implementation='eager',dtype=torch.float32)
        enc.load_state_dict(load_file(str(root/'backbone_h2.safetensors')),strict=True);enc.eval()
        head=C.Head();weights=load_file(str(root/'decision_adapter.safetensors'))
        head.load_state_dict({k[5:]:v for k,v in weights.items() if k.startswith('head.')},strict=True)
        if any(not k.startswith('head.') for k in weights):raise RuntimeError('Unexpected non-head decision adapter tensors')
        head.eval();vg=VisionEncoder(enc).eval();dg=VisionDecision(enc,head).eval()
        proc=C.Proc.get(root,1024)
        # Exact native input processor; pixel fixtures are also checked in JS.
        images=[]
        for name,color,size in [('red',(230,25,25),(256,192)),('blue',(25,25,230),(192,256))]:
            im=Image.new('RGB',size,color);draw=ImageDraw.Draw(im);draw.rectangle((20,20,80,80),fill=(240,240,240));im.save(out/(name+'.png'));images.append(im)
        e=C.decision_example(proc,[images[0]],'', 'What is the dominant color?', ['red','blue','green'])
        print('Exporting image encoder (one 512px tile per call)',flush=True)
        # Fixed tile batch 1 bounds browser peak memory. JS repeats this graph over tiles.
        export_graph(vg,(e['pv'][:1],),out/'vision.onnx',['pixel_values'],{})
        vs=session(out/'vision.onnx')
        with torch.no_grad():features=torch.cat([vg(tile[None]) for tile in e['pv']],0)
        a=vision_args(e,features,cfg.image_token_id)
        names=['input_ids','attention_mask','image_features','image_indices','option_mask']
        axes={'input_ids':{1:'sequence'},'attention_mask':{1:'sequence'},'image_features':{0:'image_tokens'},'image_indices':{1:'sequence'},'option_mask':{0:'options',1:'sequence'},'logits':{0:'options'}}
        print('Exporting visual decision graph',flush=True);export_graph(dg,a,out/'decision.onnx',names,axes)
        ds=session(out/'decision.onnx')
        for name,imgs,question in [('red',[images[0]],{'type':'choice','instructions':'What is the dominant color?','criteria':{'red':None,'blue':None,'green':None}}),
                                   ('blue',[images[1]],{'type':'noul','instructions':'Is the image mostly blue?'}),
                                   ('two',[images[0],images[1]],{'type':'choice','instructions':'What colors are shown?','criteria':{'red and blue':None,'green and yellow':None}}),
                                   ('text',[],{'type':'choice','instructions':'Choose red.','criteria':{'red':None,'blue':None}})]:
            opts=list(question['criteria']) if question['type']=='choice' else ['No / false','Yes / true']
            ex=C.decision_example(proc,imgs,'',question['instructions'],opts)
            if imgs:
                np.save(out/(name+'-pixels.npy'), ex['pv'].numpy())
                f=[];rf=[]
                for tile in ex['pv']:
                    with torch.no_grad():r=vg(tile[None]).numpy()
                    o=run(vs,(tile[None],));report.append(compare(r,o,name+'/vision-tile',atol=6e-3));rf.append(r);f.append(o)
                feats=np.concatenate(f,0).reshape(-1,768);ref_feats=np.concatenate(rf,0).reshape(-1,768)
            else:feats=ref_feats=np.zeros((1,768),np.float32)
            tensors=vision_args(ex,torch.from_numpy(feats),cfg.image_token_id)
            with torch.no_grad():
                native=enc(**C.collate([ex],'cpu')).last_hidden_state[0]
                original=head(torch.stack([native[a:b].mean(0) for a,b in ex['spans']]))[:,0].numpy()
                adapted=dg(*vision_args(ex,torch.from_numpy(ref_feats),cfg.image_token_id)).numpy()
            report.append(compare(original,adapted,name+'/wrapper',atol=6e-3));actual=run(ds,tensors);report.append(compare(original,actual,name+'/onnx',atol=6e-3))
            report.append(compare(softmax(original),softmax(actual),name+'/probabilities',atol=5e-4))
            if np.argmax(original)!=np.argmax(actual):raise AssertionError('argmax changed: '+name)
            # Small raw images accompany the fixture, encoded losslessly as PNG.
            fixtures.append({'name':name,'state':'','question':question,'images':(['red.png'] if name=='red' else ['blue.png'] if name=='blue' else ['red.png','blue.png'] if name=='two' else []),
                             'ids':ex['input_ids'],'spans':ex['spans'],'logits':original.tolist()})
        # Prefix IDs are generated by the official processor for every supported 1024px tile grid.
        grids={}
        for rows,cols in [(1,2),(2,1),(2,2)]:
            im=Image.new('RGB',(cols*512,rows*512),(100,100,100))
            grids[f'{rows}x{cols}']=C.image_prefix(proc,[im])[0]
        manifest.update(imageId=cfg.image_token_id,sepId=proc.tokenizer.sep_token_id,unknownTextId=C.Text(proc).q,
                        imagePrefixes=grids,vision={'edge':1024,'tileSize':512,'featuresPerTile':64,'hiddenSize':768,'resize':'pillow-lanczos'})
        manifest['graphs']['vision']={'file':'vision.onnx','externalData':[]}
    manifest['graphs']['decision']={'file':'decision.onnx','externalData':[]}
    import onnx
    for g in manifest['graphs'].values():
        model=onnx.load(str(out/g['file']),load_external_data=False)
        g['externalData']=sorted({v.value for t in model.graph.initializer for v in t.external_data if v.key=='location'})
        g['sizeBytes']=(out/g['file']).stat().st_size
    dump(out/'fixtures.json',fixtures)
    dump(out/'parity-python.json',{'passed':True,'checks':report,'torch':torch.__version__,'tests':'real checkpoint; eager FP32 vs portable wrapper vs ONNX Runtime CPU'})
    manifest['validated']=True;dump(out/'manifest.json',manifest)
    print(json.dumps({'out':str(out),'passed':True,'checks':len(report)}),flush=True)

if __name__=='__main__':main()
