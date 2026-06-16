# -*- coding: utf-8 -*-
"""
cek_corpus.py — menguji & memeriksa corpus_gaming.txt.

Menampilkan statistik korpus (jumlah kata/karakter/kalimat), pengecekan syarat
tugas, pembagian train/val, kata yang paling sering muncul, serta bagaimana tiap
mode tokenisasi memecah korpus (ukuran vocab & rasio kompresi).

Jalankan:  python cek_corpus.py
"""
import os, io, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
raw = io.open(os.path.join(HERE, "corpus_gaming.txt"), encoding="utf-8").read()

words = raw.split()
n_words = len(words)
sentences = [s for s in re.split(r"[.!?]+", raw) if s.strip()]
paras = [p for p in raw.split("\n\n") if p.strip()]

print("=" * 60)
print("STATISTIK CORPUS  -  corpus_gaming.txt")
print("=" * 60)
print(f"Jumlah kata            : {n_words:,}")
print(f"Jumlah karakter        : {len(raw):,}")
print(f"Kata unik (lowercase)  : {len(set(w.lower() for w in words)):,}")
print(f"Karakter unik          : {len(set(raw))}")
print(f"Jumlah paragraf        : {len(paras)}")
print(f"Jumlah kalimat         : {len(sentences)}")
print(f"Rata-rata kata/kalimat : {n_words / max(1, len(sentences)):.1f}")

REQ = 2000
status = "LULUS" if n_words >= REQ else "BELUM CUKUP"
print(f"\nSyarat tugas (>= {REQ} kata) : {status}  ({n_words:,} kata)")

cut = int(0.9 * len(raw))
print(f"Split 90/10            : train {cut:,} char | val {len(raw) - cut:,} char")

# kata paling sering
stop = set("yang dan di ke dari untuk dengan pada adalah ini itu juga atau dalam "
           "tidak para sangat menjadi serta akan oleh dapat lebih".split())
freq = collections.Counter(re.sub(r"[^a-z]", "", w.lower()) for w in words)
freq.pop("", None)
print("\n15 kata paling sering:")
for w, c in freq.most_common(15):
    print(f"   {w:18} {c}")
print("\n10 kata 'isi' paling sering (tanpa kata umum):")
isi = [(w, c) for w, c in freq.most_common(80) if w not in stop][:10]
for w, c in isi:
    print(f"   {w:18} {c}")

# tokenisasi per mode (memakai tokenizer dari generate.py)
print("\n" + "=" * 60)
print("TOKENISASI PER MODE  (rasio token/kata = makin kecil makin padat)")
print("=" * 60)
try:
    from generate import build_tokenizer
    print(f"{'mode':<6}{'vocab':>8}{'n_token':>10}{'token/kata':>12}")
    print("-" * 36)
    for mode in ["char", "word", "bpe", "spm"]:
        tk = build_tokenizer(mode)
        n_tok = len(tk.encode(raw))
        print(f"{mode:<6}{tk.vocab_size:>8}{n_tok:>10,}{n_tok / n_words:>12.2f}")
    print("\n(spm butuh outputs/spm_gaming.model - dibuat saat menjalankan notebook)")
except Exception as e:
    print("  (tokenisasi dilewati:", e, ")")
