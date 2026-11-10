#!/usr/bin/env python3
"""Export one real adapter per bundle, with mandatory original-PyTorch/ONNX parity."""
import argparse, gc, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
from memguard import guard
from export import PINS, dump, compare, session, run
from vela_onnx.portable import VisionEncoder
from vela_onnx.tasks import FastSpan, VisionTask
TASKS={'fast':['span','span_set','ner','relation','evidence'],'vision':['span','ocr','box']}

def graph(net,args,path,names,axes,output='logits'):
    """Export one task graph at opset 17 and move its weights to external data (see externalize.py)."""
    with torch.no_grad():
        torch.onnx.export(net.eval(),args,str(path),input_names=names,output_names=[output],dynamic_axes=axes,opset_version=17,dynamo=False,external_data=True)
    from externalize import externalize
    return externalize(path)

def fast(a,root,out,m):
    """Export one text task adapter and record its fixtures.

    The upstream engine runs each case and its raw-logit call is intercepted, so the fixture keeps exactly the token
    ids, positions, offsets and logits the original code used; the portable wrapper must reproduce those logits.
    Marker tokens are removed from the exported tokenizer's added tokens, so text that contains a marker string is
    tokenized as text and markers enter only by id."""
    import vela3_schema as S
    from vela3_encoder import DecisionEncoder,Vela3Encoder
    cfg=json.loads((root/'vela3_config.json').read_text()); base=DecisionEncoder.load(root,device='cpu')
    net=S.build_adapter(base,root/cfg['adapters'][a.task]['file'],'cpu');net.encoder.set_attn_implementation('eager')
    eng=S.SchemaEngine(root,net,'cpu',max_len=a.max_tokens)
    v=Vela3Encoder.__new__(Vela3Encoder);v.cfg=cfg;v._VS=S;v.span=v.span_ev=v.span_set=eng;v.task={'ner':eng,'relation':eng};v.entity_route='ner'
    cases={
      'span': [('description','Ada Lovelace wrote notes about computing.',{'criteria':{'person':'A named human being'}}),('unicode','Grüße 👋 — Zoë in 東京. Email: zoe@example.org',{'criteria':{'person':'A named human','city':'A city','email':'An email address'}})],
      'ner':[('entities','Ada Lovelace lived in London.',{'criteria':['person','city']}),('unicode','👋 Zoë besucht München und 東京.',{'criteria':['person','city']})],
      'relation':[('employment','Ada works at Acme in London.',{'instructions':'Which spans are the employer of Ada?','criteria':['employer']}),('multi','Ada works at Acme in London.',{'instructions':'Find the related employer and city for Ada.','criteria':['employer','city']})],
      'evidence':[('evidence',{'user':'Where does Ada live?','context':'Ada lives in London. Bob lives in Paris.','answer':'London'}, {'target':'context','criteria':['evidence']}),('support','The Moon orbits Earth. Mars orbits the Sun.',{'instructions':'Which spans support the claim that the Moon orbits Earth?','criteria':{'support':'Supporting text'}})],
      'span_set':[('pii','Contact Ada at ada@example.org or call +1 202 555 0147.',{'criteria':['EMAIL_ADDRESS','PHONE_NUMBER']}),('halu',{'context':'The object is red.','answer':'The object is blue.'},{'criteria':['unsupported']})]
    }[a.task]
    wrapper=FastSpan(net).eval(); fixtures=[]; records=[]; checks=[]
    original_predict=eng.predict_raw
    captured={}
    def capture(rows):
        captured['rec']=eng._assemble(eng._prep(rows[0]));r=original_predict(rows);captured['raw']=r[0];return r
    eng.predict_raw=capture
    for name,state,qq in cases:
        q={'type':'span',**qq}; answer,_=v.span_answer(state,q);rec=captured['rec'];raw=captured['raw']['span']
        args=(torch.tensor([rec['ids']]),torch.ones(1,len(rec['ids']),dtype=torch.long),torch.tensor(rec['w_pos']),torch.tensor(rec['e_pos']))
        reference=np.asarray(raw['logits'],np.float32)
        with torch.no_grad():checks.append(compare(reference,wrapper(*args).numpy(),name+'/wrapper'))
        fixtures.append({'name':name,'state':state,'question':q,'ids':[int(x) for x in rec['ids']],'wordPositions':rec['w_pos'].tolist(),'labelPositions':rec['e_pos'],'offsets':rec['w_offs'].tolist(),'names':rec['names'],'logits':reference.tolist(),'answer':answer})
        records.append((args,reference,name))
    m.update(markers=eng.mk,bosId=eng.bos,eosId=eng.eos,padId=eng.pad,spanConfig=cfg['span'])
    # Markers must be inserted by ID, never recognized inside user text.
    tj=json.loads((root/'tokenizer.json').read_text());tj['added_tokens']=[t for t in tj['added_tokens'] if t['content'] not in eng.mk];dump(out/'tokenizer.json',tj)
    names=['input_ids','attention_mask','word_positions','label_positions']
    axes={'input_ids':{1:'sequence'},'attention_mask':{1:'sequence'},'word_positions':{0:'words'},'label_positions':{0:'labels'},'logits':{0:'words',1:'labels'}}
    m['graphs']['task']=graph(wrapper,records[0][0],out/'task.onnx',names,axes)
    return fixtures,records,checks

def vision(a,root,out,m):
    """Export one vision task adapter (image encoder plus task graph) with a rendered sample image as its fixture.

    The tiles are saved and the full image-to-logits chain is checked against ONNX Runtime after PyTorch is released,
    so PyTorch's memory is released before the ONNX Runtime sessions load."""
    from PIL import Image,ImageDraw,ImageFont
    from safetensors.torch import load_file
    import uvb_core as C
    from vela3v_encoder import Vela3VEncoder
    cfg=json.loads((root/'vela3v_config.json').read_text()); tc=cfg['tasks'][a.task]
    backbone=load_file(str(root/tc['backbone_file'])) if tc.get('backbone_file') else None
    net=C.TaskNet(root,tc['lora_rank'],tc['lora_alpha'],load_file(str(root/tc['file'])),backbone);net.enc.set_attn_implementation('eager')
    del backbone
    proc=C.Proc.get(root,1024);v=Vela3VEncoder.__new__(Vela3VEncoder);v.cfg=cfg;v.nets={a.task:net};v.device='cpu';v.bf16=False;v._proc=lambda task:proc
    im=Image.new('RGB',(512,512),'white');d=ImageDraw.Draw(im)
    if a.task=='ocr':
        font=ImageFont.truetype(str(Path(__file__).resolve().parents[1]/'DejaVuSans.ttf'),32);d.text((32,100),'Invoice 123\nTotal: EUR 42.00',font=font,fill='black');state={'ocr':'Invoice 128\nTotal: EUR 92.00'};q={'type':'span','task':'ocr_check','criteria':['ocr_error']};context='User request: Check this OCR extraction against the scanned document.';text=state['ocr']
    elif a.task=='span':
        d.rectangle((100,100,400,400),fill='red');state={'user':'Describe the picture.','answer':'A blue circle on a white background.'};q={'type':'span','criteria':['unsupported']};context='User request: '+state['user'];text=state['answer']
    else:
        d.rectangle((100,100,400,400),fill='red');state='';q={'type':'box','instructions':'the red square'}
    im.save(out/'sample.png')
    ex=C.box_example(proc,im,q['instructions']) if a.task=='box' else C.span_example(proc,im,context,text)
    # Original high-level answer and raw per-word / per-cell logits.
    answer,_=getattr(v,a.task)(state,q,[im]);reference=net.logits('box' if a.task=='box' else 'span',[ex],'cpu',False)[0]
    vg=VisionEncoder(net.enc).eval();wrapper=VisionTask(net).eval()
    m['graphs']['vision']=graph(vg,(ex['pv'][:1],),out/'vision.onnx',['pixel_values'],{},'image_features')
    feats=[]
    with torch.no_grad():
        for tile in ex['pv']:feats.append(vg(tile[None]).numpy())
    feats=np.concatenate(feats).reshape(-1,768);idx=[];k=0
    for t in ex['input_ids']:
        if t==net.enc.config.image_token_id:k+=1;idx.append(k)
        else:idx.append(0)
    positions=ex['img_pos'] if a.task=='box' else ex['word_pos'];n=len(idx)
    args=(torch.tensor([ex['input_ids']]),torch.ones(1,n,dtype=torch.long),torch.from_numpy(feats),torch.tensor([idx]),torch.tensor(positions))
    with torch.no_grad():checks=[compare(reference,wrapper(*args).numpy(),a.task+'/wrapper',atol=6e-3)]
    names=['input_ids','attention_mask','image_features','image_indices','readout_positions'];axes={'input_ids':{1:'sequence'},'attention_mask':{1:'sequence'},'image_features':{0:'image_tokens'},'image_indices':{1:'sequence'},'readout_positions':{0:'readouts'},'logits':{0:'readouts'}}
    m['graphs']['task']=graph(wrapper,args,out/'task.onnx',names,axes)
    # Store tiles temporarily, validate a complete image -> logits chain after PyTorch is released.
    np.save(out/'validation-pixels.npy',ex['pv'].numpy())
    prefixes={}
    for rows,cols in [(1,2),(2,1),(2,2)]:prefixes[f'{rows}x{cols}']=C.image_prefix(proc,[Image.new('RGB',(cols*512,rows*512),'gray')])[0]
    m.update(taskConfig=tc,imageId=net.enc.config.image_token_id,sepId=proc.tokenizer.sep_token_id,unknownTextId=C.Text(proc).q,imagePrefixes=prefixes)
    f={'name':a.task,'state':state,'question':q,'images':['sample.png'],'ids':ex['input_ids'],'readoutPositions':positions.tolist(),'offsets':ex.get('w_offs',np.empty((0,2),int)).tolist(),'logits':reference.tolist(),'answer':answer}
    return [f],[(args,reference,a.task)],checks

def main():
    """Check the source hashes, export one adapter, compare original PyTorch, the portable wrapper and ONNX
    Runtime on its fixtures, and only then write `manifest.json` with `validated: true`. VERIFIED: all eight adapters
    passed these comparisons in the recorded exports (documentation/ADAPTER_VALIDATION.md)."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=TASKS,required=True);p.add_argument('--task',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-tokens',type=int,default=2048);p.add_argument('--min-free-gb',type=float,default=8.0,help='Stop the job when the machine has less memory available (python/memguard.py)');a=p.parse_args();guard(a.min_free_gb)
    if a.task not in TASKS[a.model]:p.error(f'tasks: {TASKS[a.model]}')
    if not 512<=a.max_tokens<=8192:p.error('--max-tokens must be 512..8192; runtime rejects overflow')
    torch.set_num_threads(2);root=a.source.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);(out/'manifest.json').unlink(missing_ok=True)
    locks=json.loads((Path(__file__).parent/'task-source-lock.json').read_text())[a.model]
    for name,expected in locks.items():
        if name.endswith('.safetensors') and name not in ['model.safetensors',f'{a.task}_adapter.safetensors','span_sets_adapter.safetensors' if a.task=='span_set' else '', 'backbone_v2.safetensors' if a.task=='box' else '']:continue
        with (root/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Source checksum mismatch: '+name)
    sys.path.insert(0,str(root))
    for file in ['tokenizer.json','tokenizer_config.json','LICENSE','LICENSE-APACHE']:
        if (root/file).exists():shutil.copyfile(root/file,out/file)
    m={'format':'defuss-vela-onnx/1','model':PINS[a.model][0],'revision':PINS[a.model][1],'kind':a.model,'task':a.task,'precision':'fp32','maxTokens':a.max_tokens,'maxImages':int(a.model=='vision'),'maxAnswerTokens':768,'tokenizer':'tokenizer.json','tokenizerConfig':'tokenizer_config.json','graphs':{},'validated':False}
    print('Exporting',a.model,a.task,flush=True)
    fixtures,records,checks=(fast if a.model=='fast' else vision)(a,root,out,m)
    gc.collect()
    import ctypes
    try:ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:pass
    if a.model=='vision':
        vs=session(out/'vision.onnx');tiles=np.load(out/'validation-pixels.npy');features=[]
        for i,tile in enumerate(tiles):
            actual=vs.run(None,{'pixel_values':tile[None]})[0];features.append(actual)
        actual=np.concatenate(features).reshape(-1,768)
        checks.append(compare(records[0][0][2].numpy(),actual,'vision-features',atol=6e-3));records[0][0][2].copy_(torch.from_numpy(actual));del vs
    s=session(out/'task.onnx')
    for tensors,reference,name in records:checks.append(compare(reference,run(s,tensors),name+'/onnx',atol=6e-3))
    dump(out/'fixtures.json',fixtures);dump(out/'parity-python.json',{'passed':True,'model':a.model,'task':a.task,'torch':torch.__version__,'checks':checks,'scope':'original checkpoint vs portable wrapper and ONNX CPU; JS verification separately checks decoding'})
    m['validated']=True;dump(out/'manifest.json',m);(out/'validation-pixels.npy').unlink(missing_ok=True)
    print(json.dumps({'passed':True,'out':str(out),'checks':checks}),flush=True)
if __name__=='__main__':main()
