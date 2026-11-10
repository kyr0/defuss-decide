"""Portable task readouts; preserve the checkpoint's LoRA and unfrozen layers."""
import torch
from torch import nn
from torch.nn import functional as F
from .portable import TextStack

class FastSpan(nn.Module):
    """Span readout of a text task adapter: cosine of each word's first token against each label marker, over the
    learned span temperature. Returns (words, labels) logits, which src/spans.ts decodes into spans. VERIFIED:
    tests/unit/task-format.test.ts decodes the recorded logits of every fixture to the Python answers."""
    def __init__(self, net):
        super().__init__(); self.net=net; self.stack=TextStack(net.encoder,legacy=True)
    def forward(self,input_ids,attention_mask,word_positions,label_positions):
        n=self.net; h=self.stack(n.encoder.embed_tokens(input_ids),attention_mask)[0].float()
        t=F.normalize(n.w_t(n.norm(h.index_select(0,word_positions))),dim=-1)
        e=F.normalize(n.w_e(n.norm(h.index_select(0,label_positions))),dim=-1)
        return (t @ e.T)/n.log_tau_span.exp()

class VisionTask(nn.Module):
    """Readout of a vision task adapter (span, OCR or region). Image features are spliced in as in VisionDecision,
    then the head scores the hidden states at `readout_positions`: the answer or OCR words, or the image cells for
    regions."""
    def __init__(self, net):
        super().__init__(); self.stack=TextStack(net.enc.text_model); self.head=net.head
    def forward(self,input_ids,attention_mask,image_features,image_indices,readout_positions):
        h=self.stack.model.embed_tokens(input_ids)
        features=torch.cat((torch.zeros_like(image_features[:1]),image_features),0)
        visual=features.index_select(0,image_indices.reshape(-1)).reshape_as(h)
        h=torch.where((image_indices>0)[:,:,None],visual,h)
        h=self.stack(h,attention_mask)[0].index_select(0,readout_positions)
        return self.head(h.float()).squeeze(-1)
