# 🧾 Auditor Akuntansi AI

Aplikasi Streamlit untuk menganalisis **risiko, rasio, dan tren** dari data buku besar/jurnal akuntansi,
lengkap dengan **chat AI (Groq)** dan **ekspor Excel**. Data bisa dari Kaggle, upload CSV/Excel, atau data demo.

```
Sumber data (Kaggle / upload / demo) → data_client.py → analysis.py → app.py → Pengguna
                                                                        ↕
                                                                    Groq API (gpt-oss-120b)
Claude (bantu tulis kode) → GitHub (simpan kode) → Streamlit Cloud (hosting)
```

| File | Peran |
|---|---|
| `data_client.py` | Ambil data (Kaggle/upload/demo), deteksi kolom otomatis, normalisasi |
| `analysis.py` | Ringkasan, rasio, tren + proyeksi, 6 uji risiko, uji Benford, konteks untuk AI |
| `report.py` | Ekspor Excel multi-sheet |
| `app.py` | Tampilan 5 tab, chat Groq |
| `tests/` | Uji otomatis (`pytest`) |

## Uji risiko
Jurnal tidak seimbang · Duplikat · Nilai ekstrem · Tepat di bawah ambang persetujuan · Angka bulat besar · Posting akhir pekan · Uji Benford.
Hasilnya **indikator untuk ditelusuri**, bukan bukti kecurangan.

## Menjalankan di komputer sendiri
```bash
pip install -r requirements.txt
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # isi GROQ_API_KEY
streamlit run app.py
pytest -q tests        # opsional
```

## Deploy (GitHub → Streamlit Cloud)
1. Buat repo baru di github.com, upload **semua file** folder ini (kecuali `secrets.toml` asli - sudah ada di `.gitignore`).
2. Ambil API key gratis di <https://console.groq.com/keys>.
3. Buka <https://share.streamlit.io> → **New app** → pilih repo, branch `main`, file utama `app.py`.
4. **Advanced settings → Secrets**, tempel:
   ```toml
   GROQ_API_KEY = "gsk_..."
   KAGGLE_USERNAME = "..."   # opsional, bila dataset butuh login
   KAGGLE_KEY = "..."        # opsional
   ```
5. Deploy. API key hanya ada di Secrets Streamlit, tidak pernah di kode GitHub.

## Memakai dataset Kaggle
Sidebar → **Dataset Kaggle** → isi slug (dari URL `kaggle.com/datasets/PEMILIK/NAMA`) → Unduh → pilih file.
Kolom dikenali otomatis (Indonesia/Inggris): tanggal, no jurnal, akun, kategori, deskripsi, debit/kredit atau jumlah.
Bila salah tebak, perbaiki di panel **Pemetaan kolom**. Dataset paling cocok adalah buku besar/jurnal/transaksi
dengan tanggal dan nominal; maksimum 200.000 baris dibaca.

## Mode penjualan (e-commerce / ritel)
Jika data tidak punya debit/kredit tetapi punya harga/kuantitas atau kategori produk, aplikasi otomatis masuk **mode penjualan**:
semua baris dianggap pendapatan, laba/margin tidak dihitung (tidak ada data biaya), transaksi berstatus gagal/batal dibuang,
retur/refund dihitung negatif. Contoh dataset: slug `smayanj/e-commerce-transactions-dataset`.
Dataset berbahasa USD? Ubah **Simbol mata uang** di sidebar menjadi `$`.
