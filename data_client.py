"""
data_client.py  -  Mengambil data (setara djpk_client.py pada diagram)

Tiga sumber data:
  1. Demo      : buku besar sintetis (tanpa internet, langsung jalan)
  2. Upload    : CSV / Excel dari komputer
  3. Kaggle    : unduh dataset memakai slug "pemilik/nama-dataset"

Apa pun sumbernya, hasilnya dinormalkan ke kolom standar:
    tanggal, no_jurnal, akun, kategori, deskripsi, debit, kredit, nilai, nominal
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

KATEGORI = ["Pendapatan", "HPP", "Beban", "Aset", "Liabilitas", "Ekuitas", "Lainnya"]

# Nama kolom yang dikenali otomatis (huruf kecil, tanpa spasi/simbol)
ALIAS = {
    "tanggal": ["date", "tanggal", "tgl", "postingdate", "transactiondate", "transdate",
                "datetime", "invoicedate", "entrydate", "txndate", "period", "timestamp",
                "orderdate", "purchasedate", "transactiontimestamp", "createdat", "time"],
    "no_jurnal": ["journalid", "journal", "jurnal", "nojurnal", "entryid", "journalentry",
                  "transactionid", "voucher", "voucherno", "reference", "ref", "refno",
                  "id", "invoice", "invoiceno", "txnid"],
    "akun": ["account", "akun", "accountname", "namaakun", "glaccount", "ledger",
             "accounttitle", "accountdescription", "coa"],
    "kategori": ["category", "kategori", "type", "accounttype", "class", "group",
                 "jenis", "accountclass", "accountcategory"],
    "deskripsi": ["description", "deskripsi", "keterangan", "memo", "narration",
                  "details", "particulars", "notes", "remarks"],
    "debit": ["debit", "dr", "debet", "debitamount"],
    "kredit": ["credit", "cr", "kredit", "creditamount"],
    "jumlah": ["totalamount", "totalprice", "totalsales", "salesamount", "grandtotal", "finalamount",
               "totalvalue", "amount", "jumlah", "nilai", "value", "total", "netamount",
               "transactionamount", "amt", "sales", "revenue"],
    # --- khusus data penjualan / e-commerce ---
    "harga": ["price", "unitprice", "harga", "priceperunit", "sellingprice", "itemprice"],
    "qty": ["quantity", "qty", "units", "unitssold", "jumlahbarang", "itemsbought"],
    "produk": ["productname", "product", "produk", "item", "itemname", "sku", "productid", "productcategory"],
    "pelanggan": ["customerid", "userid", "customer", "pelanggan", "user", "clientid", "buyerid", "customername"],
    "metode": ["paymentmethod", "metodebayar", "payment", "paymenttype"],
    "status": ["status", "paymentstatus", "orderstatus", "transactionstatus"],
}

# Kata kunci untuk menebak kategori dari nama akun (Indonesia + Inggris)
KATA_KUNCI = [
    ("HPP", ["hpp", "harga pokok", "pokok penjualan", "cogs", "cost of goods", "cost of sales",
             "cost of revenue", "pembelian", "purchase"]),
    ("Pendapatan", ["pendapatan", "penjualan", "revenue", "sales", "income", "omzet", "fee income"]),
    ("Beban", ["beban", "biaya", "expense", "gaji", "salary", "wage", "sewa", "rent", "utilit",
               "marketing", "depresiasi", "depreciation", "listrik", "admin", "travel", "cost"]),
    ("Liabilitas", ["utang", "hutang", "payable", "liabilit", "loan", "pinjaman", "accrued", "tax payable"]),
    ("Ekuitas", ["modal", "equity", "laba ditahan", "retained", "capital", "dividen"]),
    ("Aset", ["kas", "cash", "bank", "piutang", "receivable", "persediaan", "inventory",
              "aset", "asset", "prepaid", "equipment", "peralatan"]),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def deteksi_kolom(df: pd.DataFrame) -> dict:
    """Tebak kolom mana berisi apa. Hasil: {kolom_standar: nama_kolom_asli atau None}."""
    normal = {_norm(c): c for c in df.columns}
    hasil, dipakai = {}, set()
    for std, daftar in ALIAS.items():
        ketemu = None
        for a in daftar:                       # cocok persis
            if a in normal and normal[a] not in dipakai:
                ketemu = normal[a]
                break
        if ketemu is None:                     # cocok sebagian (alias >= 5 huruf)
            for a in daftar:
                if len(a) < 6:
                    continue
                for n, asli in normal.items():
                    if a in n and asli not in dipakai:
                        ketemu = asli
                        break
                if ketemu:
                    break
        hasil[std] = ketemu
        if ketemu:
            dipakai.add(ketemu)
    return hasil


def tebak_kategori(nama: str) -> str:
    n = str(nama).lower()
    for kat, kata in KATA_KUNCI:
        if any(k in n for k in kata):
            return kat
    return "Lainnya"


def _angka(s: pd.Series) -> pd.Series:
    if s.dtype == object or str(s.dtype).startswith("str"):
        s = s.astype(str).str.replace(r"[^\d,.\-()]", "", regex=True)
        neg = s.str.startswith("(") & s.str.endswith(")")
        s = s.str.replace(r"[()]", "", regex=True)
        # "1.234.567,89" (format Indonesia) -> "1234567.89"
        eu = s.str.contains(r"\d\.\d{3},\d") | s.str.match(r"^-?\d{1,3}(\.\d{3})+$")
        s = s.where(~eu, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        s = s.str.replace(",", "", regex=False)
        out = pd.to_numeric(s, errors="coerce")
        out = out.where(~neg, -out)
        return out.fillna(0.0)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def normalisasi(df: pd.DataFrame, peta: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Ubah dataframe mentah menjadi skema standar. Mengembalikan (df, info).

    Dua mode otomatis:
      - "buku besar" : ada debit/kredit, atau akun-akun bernama akuntansi
      - "penjualan"  : data transaksi ritel/e-commerce (ada harga/kuantitas, atau kategori produk);
                       semua baris dianggap Pendapatan, tanpa HPP/beban.
    """
    peta = peta or deteksi_kolom(df)
    g = lambda k: df[peta[k]] if peta.get(k) and peta[k] in df.columns else None
    info = {"peta": peta, "baris_awal": len(df), "catatan": [], "mode": "buku besar"}
    out = pd.DataFrame(index=df.index)

    if g("tanggal") is None:
        raise ValueError("Kolom tanggal tidak ditemukan. Pilih kolom tanggal secara manual di panel Pemetaan kolom.")
    out["tanggal"] = pd.to_datetime(g("tanggal"), errors="coerce")

    punya_dk = g("debit") is not None and g("kredit") is not None
    produk, kat_mentah, akun_mentah = g("produk"), g("kategori"), g("akun")

    # --- tentukan mode ---
    penjualan = False
    if not punya_dk:
        contoh = kat_mentah if kat_mentah is not None else akun_mentah
        asing = contoh is not None and contoh.astype(str).map(tebak_kategori).eq("Lainnya").mean() > 0.8
        penjualan = (g("qty") is not None or g("harga") is not None or asing) and \
                    (g("jumlah") is not None or (g("harga") is not None and g("qty") is not None))
    if penjualan:
        info["mode"] = "penjualan"

    # --- akun & kategori ---
    if penjualan:
        akun = kat_mentah if kat_mentah is not None else (akun_mentah if akun_mentah is not None else produk)
        out["akun"] = akun.astype(str).str.strip() if akun is not None else "Penjualan"
        out["kategori"] = "Pendapatan"
        info["catatan"].append("Mode PENJUALAN: data dibaca sebagai transaksi penjualan (semua baris = pendapatan). "
                               "Tidak ada HPP/beban, jadi laba dan margin tidak dihitung.")
    else:
        if akun_mentah is not None:
            out["akun"] = akun_mentah.astype(str).str.strip()
        elif kat_mentah is not None:
            out["akun"] = kat_mentah.astype(str).str.strip()
        else:
            out["akun"] = "Tidak diketahui"
            info["catatan"].append("Kolom akun tidak ada - semua baris diberi akun 'Tidak diketahui'.")
        if kat_mentah is not None and peta.get("kategori") != peta.get("akun"):
            out["kategori"] = kat_mentah.astype(str).map(tebak_kategori)
            if (out["kategori"] == "Lainnya").mean() > 0.8:
                out["kategori"] = out["akun"].map(tebak_kategori)
        else:
            out["kategori"] = out["akun"].map(tebak_kategori)

    # --- deskripsi, jurnal, pelanggan ---
    if penjualan:
        bagian = [x.astype(str) for x in (produk, g("pelanggan"), g("metode")) if x is not None]
        out["deskripsi"] = bagian[0].str.cat(bagian[1:], sep=" | ") if bagian else ""
    else:
        out["deskripsi"] = g("deskripsi").astype(str) if g("deskripsi") is not None else ""
    out["no_jurnal"] = g("no_jurnal").astype(str) if g("no_jurnal") is not None else np.nan
    out["pelanggan"] = g("pelanggan").astype(str) if g("pelanggan") is not None else np.nan

    # --- nominal ---
    if punya_dk:
        out["debit"] = _angka(g("debit")).abs()
        out["kredit"] = _angka(g("kredit")).abs()
        kredit_normal = out["kategori"].isin(["Pendapatan", "Liabilitas", "Ekuitas"])
        out["nilai"] = np.where(kredit_normal, out["kredit"] - out["debit"], out["debit"] - out["kredit"])
        out["nominal"] = out[["debit", "kredit"]].max(axis=1)
    else:
        if g("jumlah") is not None:
            amt = _angka(g("jumlah"))
        elif g("harga") is not None and g("qty") is not None:
            amt = _angka(g("harga")) * _angka(g("qty"))
            info["catatan"].append("Nilai transaksi dihitung = harga x kuantitas.")
        else:
            raise ValueError("Tidak ada kolom debit/kredit maupun jumlah. Pilih kolom nominal secara manual.")
        out["debit"] = 0.0
        out["kredit"] = 0.0
        out["nilai"] = amt
        out["nominal"] = amt.abs()
        if not penjualan:
            info["catatan"].append("Hanya ada kolom jumlah (bukan debit/kredit): uji jurnal tidak seimbang dinonaktifkan.")

    # --- status (hanya mode penjualan): gagal dibuang, retur jadi negatif ---
    if penjualan and g("status") is not None:
        st_ = g("status").astype(str).str.lower()
        gagal = st_.str.contains(r"fail|cancel|gagal|batal|declin|reject", regex=True)
        retur = st_.str.contains(r"refund|return|retur", regex=True)
        out.loc[retur, "nilai"] = -out.loc[retur, "nominal"]
        if gagal.any():
            info["catatan"].append(f"{int(gagal.sum()):,} transaksi berstatus gagal/batal dikeluarkan dari analisis.")
        if retur.any():
            info["catatan"].append(f"{int(retur.sum()):,} transaksi retur/refund dihitung sebagai nilai negatif.")
        out = out[~gagal]

    sebelum = len(out)
    out = out.dropna(subset=["tanggal"]).reset_index(drop=True)
    if sebelum - len(out):
        info["catatan"].append(f"{sebelum - len(out):,} baris dibuang karena tanggal tidak valid.")
    info["baris_akhir"] = len(out)
    return out, info


# ---------------------------------------------------------------- sumber data
def baca_file(file_or_path, nama: str | None = None, maks_baris: int = 200_000) -> pd.DataFrame:
    nama = (nama or getattr(file_or_path, "name", str(file_or_path))).lower()
    if nama.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_or_path, nrows=maks_baris)
    return pd.read_csv(file_or_path, nrows=maks_baris, low_memory=False, encoding_errors="replace")


def daftar_file_kaggle(slug: str, username: str | None = None, key: str | None = None) -> list[str]:
    """Unduh dataset Kaggle (di-cache oleh kagglehub) lalu daftar file CSV/Excel di dalamnya."""
    import kagglehub  # import di sini agar app tetap jalan walau paket belum terpasang

    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
    folder = kagglehub.dataset_download(slug.strip())
    files = [f for ext in ("csv", "xlsx", "xls")
             for f in glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)]
    return sorted(files)


# ---------------------------------------------------------------- data demo
def buat_demo(seed: int = 7, ambang: float = 50_000_000) -> pd.DataFrame:
    """Buku besar sintetis 22 bulan, lengkap dengan beberapa anomali yang sengaja disisipkan."""
    rng = np.random.default_rng(seed)
    hari = pd.bdate_range("2025-01-01", "2026-10-30")
    baris, n = [], 0

    def jurnal(tgl, deskripsi, kiri, kanan, jumlah):
        nonlocal n
        n += 1
        no = f"JV-{n:05d}"
        baris.append([tgl, no, kiri, deskripsi, round(jumlah), 0])
        baris.append([tgl, no, kanan, deskripsi, 0, round(jumlah)])

    for tgl in hari:
        musim = 1 + 0.25 * np.sin((tgl.month - 1) / 12 * 2 * np.pi) + 0.012 * (tgl.year - 2025) * 12
        for _ in range(rng.poisson(2.2)):
            j = rng.lognormal(16.3, 0.8) * musim
            jurnal(tgl, "Penjualan produk", "Kas dan Bank" if rng.random() < .6 else "Piutang Usaha",
                   "Penjualan Produk", j)
            jurnal(tgl, "HPP penjualan", "Harga Pokok Penjualan", "Persediaan Barang", j * rng.uniform(.58, .68))
        if rng.random() < .55:
            akun = rng.choice(["Beban Gaji", "Beban Sewa", "Beban Listrik dan Air", "Beban Pemasaran",
                               "Beban Administrasi", "Beban Perjalanan Dinas"],
                              p=[.25, .1, .15, .22, .18, .1])
            jurnal(tgl, f"Pembayaran {akun.lower()}", akun, "Kas dan Bank", rng.lognormal(15.2, 0.9))
        if rng.random() < .03:
            jurnal(tgl, "Pembayaran utang usaha", "Utang Usaha", "Kas dan Bank", rng.lognormal(16.0, .7))

    df = pd.DataFrame(baris, columns=["Tanggal", "No Jurnal", "Akun", "Keterangan", "Debit", "Kredit"])
    # --- anomali yang sengaja disisipkan (agar fitur risiko ada yang ditemukan) ---
    ekstra = []
    for k in range(8):   # tepat di bawah ambang persetujuan
        t = rng.choice(hari)
        v = round(ambang * rng.uniform(.93, .995))
        ekstra += [[t, f"JV-X{k:03d}", "Beban Administrasi", "Biaya konsultan (tanpa PO)", v, 0],
                   [t, f"JV-X{k:03d}", "Kas dan Bank", "Biaya konsultan (tanpa PO)", 0, v]]
    for k in range(6):   # jurnal tidak seimbang
        t = rng.choice(hari)
        ekstra += [[t, f"JV-U{k:03d}", "Beban Pemasaran", "Penyesuaian manual", 12_500_000, 0],
                   [t, f"JV-U{k:03d}", "Kas dan Bank", "Penyesuaian manual", 0, 12_500_000 - rng.integers(1, 9) * 750_000]]
    for k in range(6):   # nilai ekstrem
        t = rng.choice(hari)
        v = round(rng.uniform(2.5e8, 4.5e8))
        ekstra += [[t, f"JV-E{k:03d}", "Beban Perjalanan Dinas", "Biaya perjalanan luar biasa", v, 0],
                   [t, f"JV-E{k:03d}", "Kas dan Bank", "Biaya perjalanan luar biasa", 0, v]]
    for k in range(5):   # diposting akhir pekan, angka bulat
        d = pd.Timestamp(rng.choice(hari))
        t = d + pd.Timedelta(days=int((5 - d.weekday()) % 7))      # Sabtu terdekat
        v = int(rng.choice([100_000_000, 150_000_000, 200_000_000]))
        ekstra += [[t, f"JV-W{k:03d}", "Beban Administrasi", "Pembayaran vendor akhir pekan", v, 0],
                   [t, f"JV-W{k:03d}", "Kas dan Bank", "Pembayaran vendor akhir pekan", 0, v]]
    ekstra = pd.DataFrame(ekstra, columns=df.columns)
    # duplikat: salin 10 jurnal beban pemasaran lengkap (kedua sisi) dengan nomor jurnal baru
    nomor = df.loc[df["Akun"] == "Beban Pemasaran", "No Jurnal"].sample(10, random_state=seed)
    dup = df[df["No Jurnal"].isin(nomor)].copy()
    dup["No Jurnal"] = dup["No Jurnal"].str.replace("JV-", "JD-", regex=False)
    df = pd.concat([df, ekstra, dup], ignore_index=True).sort_values("Tanggal").reset_index(drop=True)
    df[["Debit", "Kredit"]] = df[["Debit", "Kredit"]].round(0)
    return df
