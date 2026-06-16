# -*- coding: utf-8 -*-
"""
generate_transfer.py — pakai model hasil TRANSFER LEARNING (pretrain gaming umum -> fine-tune FPS).

Memuat checkpoint outputs/ckpt_finetune.pt (default) lalu membangkitkan teks. Tokenizer BPE
dibangun ulang deterministik dari corpus_pretrain.txt + corpus_fps.txt (vocab identik dgn notebook).

Contoh:
  python generate_transfer.py --prompt "game fps " --tokens 200
  python generate_transfer.py --ckpt scratch --prompt "senjata "    # pembanding (model menghafal)
  python generate_transfer.py --prompt "machine learning aim "      # OOV aman (subword, tanpa <unk>)
"""
import os, io, re, argparse, collections
import torch, torch.nn as nn
from torch.nn import functional as F

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"outputs")
device="cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE,EMBED_DIM,N_HEADS,N_LAYERS=128,256,8,6

class BPETokenizer:
    def __init__(s,num_merges=500): s.num_merges=num_merges
    def fit(s,text):
        words=re.findall(r"[a-zA-Z]+",text.lower())
        vocab=collections.Counter(" ".join(list(w))+" </w>" for w in words); merges=[]
        for _ in range(s.num_merges):
            pairs=collections.Counter()
            for w,f in vocab.items():
                a=w.split()
                for j in range(len(a)-1): pairs[(a[j],a[j+1])]+=f
            if not pairs: break
            best=max(pairs,key=pairs.get); merges.append(best)
            patt=re.compile(r"(?<!\S)"+re.escape(" ".join(best))+r"(?!\S)")
            vocab={patt.sub("".join(best),w):f for w,f in vocab.items()}
        s.ranks={p:i for i,p in enumerate(merges)}; units=set()
        for w in re.findall(r"[a-zA-Z]+",text.lower()): units.update(s._enc(w))
        vl=["<unk>"," "]+sorted(units|set(text.lower())|{"</w>"})
        s.stoi={t:i for i,t in enumerate(dict.fromkeys(vl))}; s.itos={i:t for t,i in s.stoi.items()}
        s.vocab_size=len(s.stoi); return s
    def _enc(s,word):
        a=list(word)+["</w>"]
        while len(a)>1:
            best,br=None,None
            for i in range(len(a)-1):
                r=s.ranks.get((a[i],a[i+1]))
                if r is not None and (br is None or r<br): br,best=r,(a[i],a[i+1])
            if best is None: break
            m,i=[],0
            while i<len(a):
                if i<len(a)-1 and (a[i],a[i+1])==best: m.append(a[i]+a[i+1]); i+=2
                else: m.append(a[i]); i+=1
            a=m
        return a
    def encode(s,t):
        ids=[]
        for tok in re.findall(r"[a-zA-Z]+|\s+|[^a-zA-Z\s]",t.lower()):
            if tok.isspace(): ids.append(s.stoi.get(" ",0))
            elif tok.isalpha():
                for u in s._enc(tok): ids.append(s.stoi.get(u,0))
            else: ids.append(s.stoi.get(tok,0))
        return ids
    def decode(s,ids): return "".join(s.itos.get(i,"") for i in ids).replace("</w>"," ")

class CausalSelfAttention(nn.Module):
    def __init__(s,e,h,d):
        super().__init__(); s.qkv=nn.Linear(e,3*e,bias=False); s.proj=nn.Linear(e,e); s.drop=nn.Dropout(d); s.h=h; s.hd=e//h; s.d=d
    def forward(s,x):
        B,T,C=x.shape; q,k,v=s.qkv(x).split(C,2)
        q=q.view(B,T,s.h,s.hd).transpose(1,2); k=k.view(B,T,s.h,s.hd).transpose(1,2); v=v.view(B,T,s.h,s.hd).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
        return s.drop(s.proj(y.transpose(1,2).contiguous().view(B,T,C)))
class FeedForward(nn.Module):
    def __init__(s,e,d): super().__init__(); s.net=nn.Sequential(nn.Linear(e,4*e),nn.ReLU(),nn.Linear(4*e,e),nn.Dropout(d))
    def forward(s,x): return s.net(x)
class Block(nn.Module):
    def __init__(s,e,h,d): super().__init__(); s.sa=CausalSelfAttention(e,h,d); s.ff=FeedForward(e,d); s.n1=nn.LayerNorm(e); s.n2=nn.LayerNorm(e)
    def forward(s,x): x=x+s.sa(s.n1(x)); return x+s.ff(s.n2(x))
class TinyGPT(nn.Module):
    def __init__(s,vocab,dropout=0.2):
        super().__init__(); s.tok_emb=nn.Embedding(vocab,EMBED_DIM); s.pos_emb=nn.Embedding(BLOCK_SIZE,EMBED_DIM)
        s.blocks=nn.ModuleList([Block(EMBED_DIM,N_HEADS,dropout) for _ in range(N_LAYERS)])
        s.ln_f=nn.LayerNorm(EMBED_DIM); s.head=nn.Linear(EMBED_DIM,vocab)
    def forward(s,idx):
        B,T=idx.shape; x=s.tok_emb(idx)+s.pos_emb(torch.arange(T,device=idx.device))
        for b in s.blocks: x=b(x)
        return s.head(s.ln_f(x))
    @torch.no_grad()
    def generate(s,idx,n_new,temperature=0.8,top_p=0.9,repetition_penalty=1.3):
        for _ in range(n_new):
            logits=s(idx[:,-BLOCK_SIZE:])[:,-1,:]
            if repetition_penalty and repetition_penalty!=1.0:
                for b in range(idx.size(0)):
                    seen=torch.unique(idx[b]); sc=logits[b,seen]
                    logits[b,seen]=torch.where(sc>0,sc/repetition_penalty,sc*repetition_penalty)
            logits=logits/temperature
            if top_p and top_p<1.0:
                sl,si=torch.sort(logits,descending=True,dim=-1); cum=torch.cumsum(F.softmax(sl,dim=-1),dim=-1)
                drop=cum>top_p; drop[...,1:]=drop[...,:-1].clone(); drop[...,0]=False
                for b in range(logits.size(0)): logits[b,si[b,drop[b]]]=-float("inf")
            p=F.softmax(logits,dim=-1); idx=torch.cat([idx,torch.multinomial(p,1)],dim=1)
        return idx

def main():
    ap=argparse.ArgumentParser(description="Generate dari model transfer-learning gaming/FPS.")
    ap.add_argument("--ckpt",default="finetune",choices=["finetune","scratch","pretrain"],
                    help="checkpoint: finetune (transfer, default), scratch (pembanding), pretrain")
    ap.add_argument("--prompt",default="game fps ")
    ap.add_argument("--tokens",type=int,default=200)
    ap.add_argument("--temp",type=float,default=0.8)
    ap.add_argument("--top_p",type=float,default=0.9)
    ap.add_argument("--rep",type=float,default=1.3)
    a=ap.parse_args()
    ck=os.path.join(OUT,f"ckpt_{a.ckpt}.pt")
    if not os.path.exists(ck):
        raise SystemExit(f"Checkpoint tidak ada: {ck}\nJalankan notebook tinygpt_gaming_transfer.ipynb (Run All) dulu.")
    text=io.open("corpus_pretrain.txt",encoding="utf-8").read()+"\n"+io.open("corpus_fps.txt",encoding="utf-8").read()
    tok=BPETokenizer(500).fit(text)
    model=TinyGPT(tok.vocab_size).to(device); model.load_state_dict(torch.load(ck,map_location=device)); model.eval()
    ids=tok.encode(a.prompt) or [0]
    ctx=torch.tensor([ids],dtype=torch.long,device=device)
    out=model.generate(ctx,a.tokens,temperature=a.temp,top_p=a.top_p,repetition_penalty=a.rep)
    print(f"[ckpt={a.ckpt} | device={device} | vocab={tok.vocab_size}]")
    print(f"prompt: {a.prompt!r}\n"); print(tok.decode(out[0].tolist()))

if __name__=="__main__": main()
