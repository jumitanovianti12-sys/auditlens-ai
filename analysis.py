"""
analysis.py  -  Risiko, rasio, tren (tanpa Streamlit, jadi mudah diuji)
Input: dataframe berskema standar dari data_client.normalisasi()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BOBOT = {  # skor risiko per temuan
    "Jurnal tidak seimbang": 4,
    "Duplikat": 3,
    "Nilai ekstrem": 3,
    "Tepat di bawah ambang": 3,
    "Angka bulat besar": 1,
    "Posting akhir pekan": 1,
    "Retur / nilai negatif": 1,
}


# ---------------------------------------------------------------- ringkasan & rasio
def ringkasan(df: pd.DataFrame) -> dict:
    per = df.groupby("kategori")["nilai"].sum()
    g = lambda k: float(per.get(k, 0.0))
    pend, hpp, beban = g("Pendapatan"), g("HPP"), g("Beban")
    laba_kotor = pend - hpp
    laba_bersih = laba_kotor - beban
    pct = lambda a, b: (a / b * 100) if b else np.nan
    return {
        "Pendapatan": pend, "HPP": hpp, "Laba Kotor": laba_kotor, "Beban Operasional": beban,
        "Laba Bersih": laba_bersih,
        "Margin Kotor %": pct(laba_kotor, pend),
        "Margin Bersih %": pct(laba_bersih, pend),
        "Rasio HPP %": pct(hpp, pend),
        "Rasio Beban %": pct(beban, pend),
        "Jumlah Transaksi": int(len(df)),
        "Periode": f"{df['tanggal'].min():%d %b %Y} - {df['tanggal'].max():%d %b %Y}",
    }


def per_akun(df: pd.DataFrame) -> pd.DataFrame:
    t = (df.groupby(["kategori", "akun"])
           .agg(jumlah_baris=("nilai", "size"), total_nilai=("nilai", "sum"), rata2=("nominal", "mean"))
           .reset_index().sort_values("total_nilai", ascending=False))
    return t


# ---------------------------------------------------------------- tren
def tren_bulanan(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(bulan=df["tanggal"].dt.to_period("M").dt.to_timestamp())
    p = d.pivot_table(index="bulan", columns="kategori", values="nilai", aggfunc="sum", fill_value=0.0)
    for k in ("Pendapatan", "HPP", "Beban"):
        if k not in p:
            p[k] = 0.0
    p["Laba Kotor"] = p["Pendapatan"] - p["HPP"]
    p["Laba Bersih"] = p["Laba Kotor"] - p["Beban"]
    p["Margin Bersih %"] = np.where(p["Pendapatan"] != 0, p["Laba Bersih"] / p["Pendapatan"] * 100, np.nan)
    p["Pendapatan MoM %"] = p["Pendapatan"].pct_change() * 100
    return p[["Pendapatan", "HPP", "Beban", "Laba Kotor", "Laba Bersih", "Margin Bersih %", "Pendapatan MoM %"]].reset_index()


def proyeksi(tren: pd.DataFrame, kolom: str = "Pendapatan", bulan: int = 3) -> pd.DataFrame | None:
    """Proyeksi linear sederhana. Bulan terakhir dibuang bila datanya belum penuh."""
    t = tren.copy()
    if len(t) < 6:
        return None
    y = t[kolom].to_numpy(dtype=float)
    x = np.arange(len(y))
    b, a = np.polyfit(x, y, 1)
    sisa = y - (a + b * x)
    se = sisa.std(ddof=2) if len(y) > 2 else 0
    xs = np.arange(len(y), len(y) + bulan)
    tgl = pd.date_range(t["bulan"].iloc[-1] + pd.offsets.MonthBegin(1), periods=bulan, freq="MS")
    return pd.DataFrame({"bulan": tgl, "proyeksi": a + b * xs,
                         "batas_bawah": a + b * xs - 1.96 * se, "batas_atas": a + b * xs + 1.96 * se})


# ---------------------------------------------------------------- risiko
def _bulat(nom: pd.Series, p90: float) -> pd.Series:
    """Angka 'terlalu bulat' (<= 2 digit signifikan) dan termasuk 10% terbesar."""
    n = nom.round(0)
    pangkat = np.floor(np.log10(n.where(n > 0, 1))).astype(int)
    pembagi = (10.0 ** (pangkat - 1)).clip(lower=100)
    return (n >= p90) & (n > 0) & ((n % pembagi) == 0)


def temuan_risiko(df: pd.DataFrame, ambang: float = 50_000_000, z_batas: float = 3.5,
                  akhir_pekan: bool = True, retur: bool = False) -> pd.DataFrame:
    """Tandai transaksi mencurigakan. Mengembalikan df asli + kolom: temuan, skor, level."""
    d = df.copy()
    d["_i"] = np.arange(len(d))
    flags = {k: pd.Series(False, index=d.index) for k in BOBOT}
    nom = d["nominal"]
    ada_nom = nom > 0

    # 1. Duplikat
    flags["Duplikat"] = ada_nom & d.duplicated(["tanggal", "akun", "nominal", "deskripsi"], keep=False)

    # 2. Nilai ekstrem (robust z-score pada log nominal, per kategori)
    ln = np.log1p(nom.where(ada_nom))
    for _, idx in d.groupby("kategori").groups.items():
        x = ln.loc[idx].dropna()
        if len(x) < 30:
            continue
        med = x.median()
        mad = (x - med).abs().median()
        if mad == 0:
            continue
        z = 0.6745 * (x - med) / mad
        flags["Nilai ekstrem"].loc[z[z > z_batas].index] = True

    # 3. Tepat di bawah ambang persetujuan (90% - <100%)
    if ambang and ambang > 0:
        flags["Tepat di bawah ambang"] = (nom >= 0.9 * ambang) & (nom < ambang)

    # 4. Angka bulat besar
    if ada_nom.sum() >= 30:
        flags["Angka bulat besar"] = _bulat(nom, nom[ada_nom].quantile(0.90))

    # 5. Posting akhir pekan
    if akhir_pekan:
        flags["Posting akhir pekan"] = ada_nom & (d["tanggal"].dt.weekday >= 5)

    # 5b. Retur / nilai negatif (relevan untuk data penjualan)
    if retur:
        flags["Retur / nilai negatif"] = d["nilai"] < 0

    # 6. Jurnal tidak seimbang (hanya jika data double-entry & ada nomor jurnal)
    if (d["debit"].sum() + d["kredit"].sum()) > 0 and d["no_jurnal"].notna().any():
        sel = d.groupby("no_jurnal")[["debit", "kredit"]].sum()
        buruk = sel.index[(sel["debit"] - sel["kredit"]).abs() > 1]
        flags["Jurnal tidak seimbang"] = d["no_jurnal"].isin(buruk)

    d["temuan"] = ""
    d["skor"] = 0
    for k, f in flags.items():
        d.loc[f, "temuan"] = d.loc[f, "temuan"].where(d.loc[f, "temuan"] == "", d.loc[f, "temuan"] + "; ") + k
        d.loc[f, "skor"] += BOBOT[k]
    d["level"] = pd.cut(d["skor"], [-1, 0, 1, 3, 99], labels=["Aman", "Rendah", "Sedang", "Tinggi"]).astype(str)
    return d.drop(columns="_i")


def ringkas_temuan(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in BOBOT:
        m = d["temuan"].str.contains(k, regex=False)
        rows.append({"Jenis temuan": k, "Jumlah baris": int(m.sum()),
                     "Total nominal": float(d.loc[m, "nominal"].sum()), "Bobot": BOBOT[k]})
    return pd.DataFrame(rows).sort_values(["Bobot", "Jumlah baris"], ascending=False)


# ---------------------------------------------------------------- Benford
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def benford(df: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    n = df.loc[df["nominal"] >= 10, "nominal"]
    if len(n) < 100:
        return pd.DataFrame(), np.nan, "Data kurang dari 100 transaksi - uji Benford tidak andal."
    digit = n.astype(float).abs().astype(str).str.lstrip("0.").str[0].astype(int)
    obs = digit.value_counts(normalize=True).reindex(range(1, 10), fill_value=0).to_numpy()
    mad = float(np.mean(np.abs(obs - BENFORD)))
    if mad < 0.006:
        v = "Sangat sesuai Benford"
    elif mad < 0.012:
        v = "Cukup sesuai Benford"
    elif mad < 0.015:
        v = "Sedikit menyimpang - perlu perhatian"
    else:
        v = "Tidak sesuai Benford - telaah lebih lanjut"
    t = pd.DataFrame({"Digit": range(1, 10), "Teramati %": obs * 100, "Benford %": BENFORD * 100})
    return t, mad, v


# ---------------------------------------------------------------- konteks untuk AI
def _ribuan(teks: str, simbol: str) -> str:
    """Rp -> 1.234.567 (gaya Indonesia); mata uang lain -> 1,234,567."""
    return teks.replace(",", "X").replace(".", ",").replace("X", ".") if simbol == "Rp" else teks


def rp(x: float, simbol: str = "Rp") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return _ribuan(f"{simbol} {x:,.0f}", simbol)


def rp_ringkas(x: float, simbol: str = "Rp") -> str:
    """Format pendek untuk kartu angka. Rp: M (miliar)/jt/T; lainnya: K/M/B."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s, v = ("-" if x < 0 else ""), abs(x)
    skala = ((1e12, " T"), (1e9, " M"), (1e6, " jt")) if simbol == "Rp" else ((1e9, "B"), (1e6, "M"), (1e3, "K"))
    for batas, sufiks in skala:
        if v >= batas:
            return _ribuan(f"{s}{simbol} {v / batas:,.1f}{sufiks}", simbol)
    return _ribuan(f"{s}{simbol} {v:,.0f}", simbol)


def konteks_ai(df: pd.DataFrame, tem: pd.DataFrame, ambang: float, simbol: str = "Rp", mode: str = "buku besar") -> str:
    r = ringkasan(df)
    tr = tren_bulanan(df).tail(6)
    _, mad, verdict = benford(df)
    rs = ringkas_temuan(tem)
    top = tem[tem["skor"] >= 3].sort_values(["skor", "nominal"], ascending=False).head(12)
    akun = per_akun(df).head(8)
    fmt = lambda k, v: f"{v:.1f}%" if ("%" in k and pd.notna(v)) else rp(v, simbol)
    L = []
    if mode == "penjualan":
        L.append("JENIS DATA: transaksi PENJUALAN ritel/e-commerce. Tidak ada HPP/beban: JANGAN menyimpulkan laba, "
                 "margin, atau efisiensi biaya. 'Pendapatan' = total penjualan, 'akun' = kategori produk.")
    L += [f"PERIODE: {r['Periode']} | {r['Jumlah Transaksi']:,} baris",
         "RINGKASAN: " + "; ".join(f"{k}={fmt(k, v)}" for k, v in r.items()
                                  if k not in ("Periode", "Jumlah Transaksi")
                                  and not (mode == "penjualan" and k != "Pendapatan"))
         + (f"; Rata-rata per transaksi={rp(df['nilai'].mean(), simbol)}" if mode == "penjualan" else ""),
         "TREN 6 BULAN TERAKHIR " + ("(bulan | penjualan):" if mode == "penjualan" else "(bulan | pendapatan | beban | laba bersih):")]
    for _, x in tr.iterrows():
        L.append(f"  {x['bulan']:%Y-%m} | {rp(x['Pendapatan'], simbol)}" if mode == "penjualan" else
                 f"  {x['bulan']:%Y-%m} | {rp(x['Pendapatan'], simbol)} | {rp(x['Beban'], simbol)} | {rp(x['Laba Bersih'], simbol)}")
    L.append("AKUN TERBESAR: " + "; ".join(f"{a.akun} ({a.kategori}) {rp(a.total_nilai, simbol)}" for a in akun.itertuples()))
    L.append(f"AMBANG PERSETUJUAN: {rp(ambang, simbol)}")
    L.append("TEMUAN RISIKO: " + "; ".join(f"{a._1}={a._2} baris" for a in rs.itertuples() if a._2))
    L.append(f"BENFORD: MAD={mad:.4f} -> {verdict}" if not np.isnan(mad) else f"BENFORD: {verdict}")
    L.append("TRANSAKSI RISIKO TERTINGGI:")
    L += [f"  {x.tanggal:%Y-%m-%d} | {x.no_jurnal} | {x.akun} | {rp(x.nominal, simbol)} | {x.temuan}" for x in top.itertuples()]
    return "\n".join(L)
