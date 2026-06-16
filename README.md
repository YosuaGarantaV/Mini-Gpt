# Tiny GPT dengan Mini Corpus — Tema Gaming / FPS 🎮

> Proyek **Data Mining ST167 — Modul 9: Membangun Tiny GPT dengan Mini Corpus**
> Fakultas Ilmu Komputer, Universitas Amikom Yogyakarta.

Membangun sebuah **GPT (Transformer decoder-only)** dari nol dengan PyTorch, lalu melatihnya
di atas korpus teks berbahasa Indonesia bertema **gaming, khususnya FPS (first person shooter)**
dengan cakupan luas (**±18.500 kata**). Ini adalah **versi scale-up** ala Tiny GPT: korpus & model
dibesarkan agar hasilnya lebih baik. Proyek membandingkan **empat mode tokenisasi** dan menganalisis
performanya. Seluruh proses pelatihan beserta **log dan grafik** tersimpan langsung di dalam notebook
[`tinygpt_gaming.ipynb`](tinygpt_gaming.ipynb).

**Skala model:** `embed=256, layers=6, heads=8, konteks=128` (~5–6 juta parameter), dilatih
**4.000 iterasi** + **warmup/cosine LR**, sampling **top_p + repetition penalty**.

---

## 🎯 Sesuai instruksi tugas

| No | Instruksi tugas | Status |
|----|-----------------|--------|
| 1 | Korpus total kata **min 2.000** (bebas domain) | ✅ **±18.500 kata**, tema gaming/FPS |
| 2 | Latih model dengan korpus tersebut | ✅ GPT decoder-only (~5–6 juta parameter) |
| 3 | Coba **beberapa mode tokenisasi** | ✅ `char`, `word`, `bpe`, `spm` (SentencePiece) |
| 4 | Tampilkan hasil + analisis performa | ✅ Tabel metrik, kurva loss, perpleksitas, bits/char, contoh generate |

Tambahan: pelatihan berjalan di **GPU (CUDA)** dan seluruh **log training tampil di dalam notebook**.

---

## 📁 Struktur berkas (file pathing)

```
tinygpt_gaming/
├── README.md                 # dokumen ini
├── LAPORAN.md                # laporan + analisis performa (untuk submission)
├── HASIL_TRAINING.md         # semua output 1 run: chart, tabel, log, contoh generate
├── LAPORAN_GAMING.docx       # laporan versi Word (.docx)
├── LAPORAN_GAMING.pdf        # laporan versi PDF (siap cetak/kirim)
├── requirements.txt          # daftar dependensi
├── .gitignore
├── corpus_gaming.txt         # korpus gaming/FPS (±18.500 kata)  ← input utama
├── tinygpt_gaming.ipynb      # NOTEBOOK UTAMA (4 mode tokenisasi, log + chart)
├── generate.py               # pakai model terlatih dari CLI (tanpa latih ulang)
├── cek_corpus.py             # uji/inspeksi corpus (statistik + tokenisasi)
├── build_nb.py               # script penyusun notebook (opsional)
│   # --- Upgrade transfer learning (anti-hafalan, tetap Tiny GPT) ---
├── tinygpt_gaming_transfer.ipynb  # NOTEBOOK transfer: pretrain gaming umum -> fine-tune FPS
├── build_transfer_nb.py      # penyusun notebook transfer
├── generate_transfer.py      # inferensi dari model hasil transfer (ckpt_finetune)
├── corpus_pretrain.txt       # gaming umum (pretrain, ±12.700 kata)
├── corpus_fps.txt            # FPS (fine-tune, ±5.700 kata)
└── outputs/                  # artefak hasil eksekusi
    ├── chart_loss_curves.png / chart_ppl.png / chart_bpc_vocab.png   # notebook utama
    ├── chart_transfer_loss.png / chart_transfer_compare.png          # notebook transfer
    ├── hasil_gaming.json / hasil_transfer.json                       # ringkasan metrik
    ├── ckpt_char/word/bpe/spm.pt                # checkpoint 4 mode (notebook utama)
    ├── ckpt_pretrain/finetune/scratch.pt        # checkpoint transfer
    └── spm_gaming.model / spm_gaming.vocab      # tokenizer SentencePiece
```

---

## 🚀 Cara menjalankan

### 1. Pasang dependensi
```bash
pip install -r requirements.txt
# GPU NVIDIA (mis. RTX 50-series): pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 2. Buka & jalankan notebook
```bash
jupyter notebook tinygpt_gaming.ipynb
# Jalankan semua sel: menu  Run > Run All Cells
```
Notebook akan otomatis memakai **GPU** bila tersedia, dan **CPU** bila tidak.

### 3. (Opsional) bangun ulang notebook dari script
```bash
python build_nb.py        # menyusun ulang tinygpt_gaming.ipynb
```

---

## 🧪 Cara memakai & menguji model terlatih

**A. Di dalam notebook** — setelah *Run All*, pakai sel **"9. Coba Model Sendiri"**:
```python
coba("game fps ", mode="word", max_new_tokens=200, temperature=0.8, top_p=0.9, repetition_penalty=1.3)
```

**B. Dari terminal (tanpa latih ulang)** — pakai [`generate.py`](generate.py):
```bash
python generate.py --mode word --prompt "game fps "       --tokens 200
python generate.py --mode bpe  --prompt "esports adalah " --temp 0.7 --top_p 0.9 --rep 1.3
```

**C. Uji/inspeksi korpus:**
```bash
python cek_corpus.py
```

> **Parameter:** `temp` rendah (0.5–0.7) = lebih aman; tinggi (0.9–1.1) = lebih kreatif/acak.
> `--top_p` = nucleus sampling. `--rep` = repetition penalty (>1 menekan pengulangan).

---

## 🧠 Ringkasan teknis

- **Arsitektur:** token embedding + positional embedding → 6× Block (Multi-Head Self-Attention
  kausal + FeedForward, residual & LayerNorm) → LayerNorm → linear head.
  Attention memakai `F.scaled_dot_product_attention` (flash-attention di GPU).
- **Hyperparameter (scale-up):** `block=128, embed=256, heads=8, layers=6, dropout=0.2, lr=1e-3
  (warmup+cosine), weight_decay=0.1, iters=4000`, **early stopping**.
- **Sampling:** `top_p` (nucleus) + `repetition_penalty` untuk teks lebih variatif & tidak berulang.
- **Mode tokenisasi:** `char`, `word`, `bpe` (BPE implementasi sendiri), `spm` (SentencePiece).
- **Metrik pembanding utama: bits/char** — dinormalisasi ke jumlah karakter validasi yang sama
  sehingga **adil dibandingkan antar-mode**.

📊 **Tabel hasil & grafik lengkap ada di [LAPORAN.md](LAPORAN.md), [HASIL_TRAINING.md](HASIL_TRAINING.md),
laporan Word [LAPORAN_GAMING.docx](LAPORAN_GAMING.docx), serta di dalam notebook.**

---

## 🧪 Upgrade: Transfer Learning (anti-hafalan, TETAP Tiny GPT)

Notebook [`tinygpt_gaming_transfer.ipynb`](tinygpt_gaming_transfer.ipynb) mengurangi **hafalan** tanpa
keluar dari konsep Tiny GPT (tetap dari nol, bukan model pra-terlatih raksasa): **pretrain** Tiny GPT
pada **gaming umum** (`corpus_pretrain.txt`) lalu **fine-tune** ke **FPS** (`corpus_fps.txt`) dengan
LR kecil + dropout tinggi + sebagian layer dibekukan + early stopping.

| Model | val loss | **gap (val−train)** | PPL | bits/char |
|-------|---------:|--------------------:|----:|----------:|
| SCRATCH (FPS dari nol) | 2.79 | 1.42 | 16.3 | 1.75 |
| **TRANSFER (pretrain+FT)** | **2.21** | **0.36** | **9.1** | **1.38** |

➡️ **Overfitting turun ~4× (gap 1.42 → 0.36)** = jauh lebih sedikit menghafal, generalisasi lebih baik.

```bash
python build_transfer_nb.py          # susun notebook transfer
python generate_transfer.py --prompt "game fps "          # pakai model transfer
python generate_transfer.py --ckpt scratch --prompt "game fps "   # pembanding
```

---

## ⬆️ Cara unggah ke GitHub

Dari dalam folder `tinygpt_gaming/`:
```bash
git init
git add .
git commit -m "Tiny GPT scale-up - mini corpus gaming/FPS (Modul 9 ST167)"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```
> Notebook sudah berisi output (log + chart) sehingga langsung tampil rapi di halaman GitHub.

---

## 📚 Referensi
Modul mengikuti struktur [annaamikom/tinyGPT](https://github.com/annaamikom/tinyGPT)
(`transformer_blocks.py`, `corpus.txt`, `tinygpt.py`) — di sini diadaptasi menjadi satu notebook
mandiri bertema gaming/FPS dengan model & korpus yang diperbesar.
