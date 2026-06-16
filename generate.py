# -*- coding: utf-8 -*-
"""
generate.py — pakai model Tiny GPT yang sudah dilatih, TANPA melatih ulang.

Memuat checkpoint outputs/ckpt_<mode>.pt (dibuat oleh notebook tinygpt_gaming.ipynb)
lalu membangkitkan teks dari prompt. Tokenizer dibangun ulang secara deterministik
dari corpus_gaming.txt (untuk spm: memuat outputs/spm_gaming.model), sehingga vocab-nya
identik dengan saat pelatihan.

Contoh:
  python generate.py --mode word --prompt "game fps " --tokens 200
  python generate.py --mode bpe  --prompt "piala dunia "       --temp 0.7 --topk 20
  python generate.py --mode char                                # pakai prompt default
"""
import os, io, re, argparse, collections
import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "outputs")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- HARUS sama dengan notebook saat pelatihan (versi SCALE-UP) ----
BLOCK_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, DROPOUT = 128, 256, 8, 6, 0.2

# ============================ TOKENIZER ============================
class CharTokenizer:
    name = "char"
    def fit(self, full_text):
        chars = sorted(set(full_text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars); return self
    def encode(self, s): return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids): return "".join(self.itos[i] for i in ids)

class WordTokenizer:
    name = "word"; pat = re.compile(r"[a-zA-Z]+|[^a-zA-Z\s]|\s+")
    def _split(self, s): return [t for t in self.pat.findall(s.lower())]
    def fit(self, train_text):
        vocab = ["<unk>"] + sorted(set(self._split(train_text)))
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(vocab); return self
    def encode(self, s): return [self.stoi.get(t, 0) for t in self._split(s)]
    def decode(self, ids): return "".join(self.itos[i] for i in ids)

class BPETokenizer:
    name = "bpe"
    def __init__(self, num_merges=250): self.num_merges = num_merges
    def fit(self, train_text, full_text=None):
        words = re.findall(r"[a-zA-Z]+", train_text.lower())
        vocab = collections.Counter(" ".join(list(w)) + " </w>" for w in words)
        merges = []
        for _ in range(self.num_merges):
            pairs = collections.Counter()
            for word, fr in vocab.items():
                sym = word.split()
                for j in range(len(sym) - 1): pairs[(sym[j], sym[j + 1])] += fr
            if not pairs: break
            best = max(pairs, key=pairs.get); merges.append(best)
            patt = re.compile(r"(?<!\S)" + re.escape(" ".join(best)) + r"(?!\S)")
            vocab = {patt.sub("".join(best), w): fr for w, fr in vocab.items()}
        self.ranks = {p: i for i, p in enumerate(merges)}
        units = set()
        for w in re.findall(r"[a-zA-Z]+", (full_text or train_text).lower()):
            units.update(self._enc_word(w))
        base = set(ch for ch in (full_text or train_text).lower())
        vocab_list = ["<unk>", " "] + sorted(units | base | {"</w>"})
        self.stoi = {t: i for i, t in enumerate(dict.fromkeys(vocab_list))}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(self.stoi); return self
    def _enc_word(self, word):
        sym = list(word) + ["</w>"]
        while len(sym) > 1:
            best, br = None, None
            for i in range(len(sym) - 1):
                r = self.ranks.get((sym[i], sym[i + 1]))
                if r is not None and (br is None or r < br): br, best = r, (sym[i], sym[i + 1])
            if best is None: break
            merged, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    merged.append(sym[i] + sym[i + 1]); i += 2
                else: merged.append(sym[i]); i += 1
            sym = merged
        return sym
    def encode(self, s):
        ids = []
        for tok in re.findall(r"[a-zA-Z]+|\s+|[^a-zA-Z\s]", s.lower()):
            if tok.isspace(): ids.append(self.stoi.get(" ", 0))
            elif tok.isalpha():
                for u in self._enc_word(tok): ids.append(self.stoi.get(u, 0))
            else: ids.append(self.stoi.get(tok, 0))
        return ids
    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids).replace("</w>", " ")

class SPMTokenizer:
    name = "spm"
    def load(self):
        import sentencepiece as spm
        self.model_path = os.path.join(OUT, "spm_gaming.model")
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.vocab_size = self.sp.get_piece_size(); return self
    def encode(self, s): return self.sp.encode(s, out_type=int)
    def decode(self, ids): return self.sp.decode(ids)

# ============================ MODEL ============================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.resid_drop = nn.Dropout(dropout)
        self.n_heads = n_heads; self.head_dim = embed_dim // n_heads; self.dropout = dropout
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.sa = CausalSelfAttention(embed_dim, n_heads, dropout)
        self.ffwd = FeedForward(embed_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim); self.ln2 = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.sa(self.ln1(x)); x = x + self.ffwd(self.ln2(x)); return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[Block(EMBED_DIM, N_HEADS, DROPOUT) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size)
    def forward(self, idx):
        B, T = idx.shape
        x = self.blocks(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        return self.lm_head(self.ln_f(x))
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 top_p=0.9, repetition_penalty=1.3):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -BLOCK_SIZE:])[:, -1, :]
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    sc = logits[b, seen]
                    logits[b, seen] = torch.where(sc > 0, sc / repetition_penalty,
                                                  sc * repetition_penalty)
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p and top_p < 1.0:
                s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
                cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
                drop = cum > top_p
                drop[..., 1:] = drop[..., :-1].clone(); drop[..., 0] = False
                for b in range(logits.size(0)):
                    logits[b, s_idx[b, drop[b]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ============================ MAIN ============================
def build_tokenizer(mode):
    raw = io.open(os.path.join(HERE, "corpus_gaming.txt"), encoding="utf-8").read()
    train_text = raw[:int(0.9 * len(raw))]
    if mode == "char": return CharTokenizer().fit(raw)
    if mode == "word": return WordTokenizer().fit(train_text)
    if mode == "bpe":  return BPETokenizer(250).fit(train_text, full_text=raw)
    if mode == "spm":  return SPMTokenizer().load()
    raise ValueError(f"mode tak dikenal: {mode}")

def main():
    ap = argparse.ArgumentParser(description="Generate teks dari Tiny GPT terlatih.")
    ap.add_argument("--mode", default="word", choices=["char", "word", "bpe", "spm"])
    ap.add_argument("--prompt", default="game fps ")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=0, help="0 = nonaktif (pakai top_p)")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.3, help="repetition penalty (>1 menekan pengulangan)")
    args = ap.parse_args()

    ckpt = os.path.join(OUT, f"ckpt_{args.mode}.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint tidak ditemukan: {ckpt}\n"
                         f"Jalankan dulu notebook tinygpt_gaming.ipynb (Run All) untuk membuatnya.")

    tok = build_tokenizer(args.mode)
    model = TinyGPT(tok.vocab_size).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ids = tok.encode(args.prompt) or [0]
    ctx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(ctx, args.tokens, temperature=args.temp,
                         top_k=(args.topk or None), top_p=args.top_p,
                         repetition_penalty=args.rep)
    text = tok.decode(out[0].tolist())

    print(f"[mode={args.mode} | device={device} | vocab={tok.vocab_size}]")
    print(f"prompt: {args.prompt!r}\n")
    print(text)

if __name__ == "__main__":
    main()
