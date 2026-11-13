"""
app.py  -  Tampilan, chat AI, dan ekspor Excel
Auditor Akuntansi AI: analisis risiko, rasio, dan tren buku besar + tanya-jawab dengan Groq.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import data_client as dc
import report

st.set_page_config(page_title="Auditor Akuntansi AI", page_icon="🧾", layout="wide")

MODEL = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
SISTEM = """Anda adalah asisten auditor dan analis akuntansi. Jawab dalam Bahasa Indonesia yang ringkas dan terstruktur.
Gunakan HANYA angka pada konteks data yang diberikan; jangan mengarang angka. Jika data tidak cukup, katakan terang-terangan.
Temuan risiko adalah INDIKASI yang perlu diverifikasi, bukan bukti kecurangan. Beri langkah tindak lanjut yang konkret
(dokumen yang diminta, pihak yang dikonfirmasi, uji tambahan).
Jika konteks menyebut data PENJUALAN, jangan membahas laba/beban/margin karena datanya tidak ada."""


# ------------------------------------------------------------------ util
def rahasia(nama: str):
    try:
        v = st.secrets[nama]
    except Exception:
        v = None
    return v or os.getenv(nama)


def tanya_ai(pesan: list[dict], model: str) -> str:
    key = rahasia("GROQ_API_KEY")
    if not key:
        return "⚠️ GROQ_API_KEY belum diisi. Tambahkan di Streamlit → Settings → Secrets."
    try:
        from groq import Groq
        r = Groq(api_key=key).chat.completions.create(model=model, messages=pesan, temperature=0.2, max_tokens=1800)
        return r.choices[0].message.content
    except Exception as e:  # jaringan, kuota, model salah, dll.
        return f"⚠️ Permintaan AI gagal: {e}"


@st.cache_data(show_spinner=False)
def muat_demo():
    return dc.buat_demo()


@st.cache_data(show_spinner="Mengunduh dataset dari Kaggle...")
def muat_kaggle(slug, user, key):
    return dc.daftar_file_kaggle(slug, user, key)


@st.cache_data(show_spinner="Membaca file...")
def baca_path(path):
    return dc.baca_file(path)


@st.cache_data(show_spinner=False)
def hitung_risiko(df, ambang, penjualan):
    return an.temuan_risiko(df, ambang=ambang, akhir_pekan=not penjualan, retur=penjualan)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    sumber = st.radio("Sumber data", ["Data demo", "Upload CSV/Excel", "Dataset Kaggle"])
    mentah, nama_data = None, ""

    if sumber == "Data demo":
        mentah, nama_data = muat_demo(), "Buku besar demo (sintetis)"
    elif sumber == "Upload CSV/Excel":
        f = st.file_uploader("File buku besar / jurnal", type=["csv", "xlsx", "xls"])
        if f:
            mentah, nama_data = dc.baca_file(f), f.name
    else:
        slug = st.text_input("Slug dataset Kaggle", placeholder="pemilik/nama-dataset",
                             help="Salin dari URL: kaggle.com/datasets/PEMILIK/NAMA-DATASET")
        if st.button("Unduh dataset", disabled=not slug):
            try:
                st.session_state["kg_files"] = muat_kaggle(slug, rahasia("KAGGLE_USERNAME"), rahasia("KAGGLE_KEY"))
                st.session_state["kg_slug"] = slug
            except Exception as e:
                st.error(f"Gagal mengunduh: {e}")
        files = st.session_state.get("kg_files", [])
        if files:
            pilih = st.selectbox("File", files, format_func=os.path.basename)
            mentah, nama_data = baca_path(pilih), f"Kaggle: {st.session_state.get('kg_slug')} / {os.path.basename(pilih)}"
        elif "kg_files" in st.session_state:
            st.warning("Tidak ada file CSV/Excel di dataset ini.")

    st.divider()
    model = st.selectbox("Model AI (Groq)", MODEL)
    simbol = st.text_input("Simbol mata uang", "Rp", help="Dataset e-commerce Kaggle biasanya dalam USD: isi $")
    ambang = st.number_input("Ambang persetujuan (0 = uji dimatikan)",
                             min_value=0.0, value=float(st.session_state.get("ambang_default", 50_000_000.0)),
                             step=1_000.0, format="%.0f", key=f"ambang_{st.session_state.get('mode', 'bb')}")
    st.caption("API key Groq dibaca dari Secrets Streamlit, tidak ada di kode.")
    if not rahasia("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY belum diisi - fitur AI nonaktif, analisis lain tetap jalan.")

# ------------------------------------------------------------------ header
st.title("🧾 Auditor Akuntansi AI")
st.caption("Risiko · rasio · tren dari data buku besar, plus tanya-jawab dengan AI. Cocok untuk auditor, bagian keuangan, dan pembelajaran.")

if mentah is None:
    st.info("Pilih sumber data di sidebar untuk mulai. Coba **Data demo** agar langsung melihat hasilnya.")
    st.stop()

# ------------------------------------------------------------------ pemetaan kolom
auto = dc.deteksi_kolom(mentah)
with st.expander("🔎 Pemetaan kolom (cek jika hasil terlihat aneh)", expanded=False):
    opsi = [None] + list(mentah.columns)
    DAFTAR = [("tanggal", "Tanggal"), ("no_jurnal", "No. jurnal / ID transaksi"), ("akun", "Akun"),
              ("kategori", "Kategori / tipe akun"), ("deskripsi", "Deskripsi"),
              ("debit", "Debit"), ("kredit", "Kredit"), ("jumlah", "Jumlah / total (jika tak ada debit-kredit)"),
              ("harga", "Harga satuan (penjualan)"), ("qty", "Kuantitas (penjualan)"), ("produk", "Produk (penjualan)"),
              ("pelanggan", "Pelanggan (penjualan)"), ("metode", "Metode bayar (penjualan)"), ("status", "Status transaksi")]
    peta, kolom = {}, st.columns(4)
    for i, (std, label) in enumerate(DAFTAR):
        awal = opsi.index(auto[std]) if auto[std] in opsi else 0
        peta[std] = kolom[i % 4].selectbox(label, opsi, index=awal, key=f"m_{std}_{nama_data}")

try:
    df, info = dc.normalisasi(mentah, peta)
except ValueError as e:
    st.error(str(e))
    st.stop()
for c in info["catatan"]:
    st.info(c)
if df.empty:
    st.error("Tidak ada baris valid setelah pembersihan.")
    st.stop()

penjualan = info["mode"] == "penjualan"
if st.session_state.get("mode") != info["mode"]:     # ganti mode -> sesuaikan ambang bawaan, muat ulang sekali
    st.session_state["mode"] = info["mode"]
    st.session_state["ambang_default"] = 0.0 if penjualan else 50_000_000.0
    st.rerun()
tem = hitung_risiko(df, ambang, penjualan)
r = an.ringkasan(df)
tren = an.tren_bulanan(df)
benf, mad, verdict = an.benford(df)
st.write(f"**Data:** {nama_data} · {r['Jumlah Transaksi']:,} baris · {r['Periode']}")

# ------------------------------------------------------------------ tab
t1, t2, t3, t4, t5 = st.tabs(["📊 Ringkasan", "📈 Tren", "🚩 Risiko", "💬 Tanya AI", "📥 Data & Excel"])

with t1:
    st.caption("Singkatan angka: Rp → M = miliar, jt = juta, T = triliun. Mata uang lain → K = ribu, M = juta, B = miliar. Angka lengkap ada di tab Data & Excel.")
    if penjualan:
        total = df["nilai"].sum()
        a, b, c, d = st.columns(4)
        a.metric("Total penjualan (bersih)", an.rp_ringkas(total, simbol))
        b.metric("Jumlah transaksi", f"{len(df):,}")
        c.metric("Rata-rata per transaksi", an.rp_ringkas(df["nilai"].mean(), simbol))
        d.metric("Pelanggan unik", f"{df['pelanggan'].nunique():,}" if df["pelanggan"].notna().any() else "-")
        kiri, kanan = st.columns(2)
        pk = df.groupby("akun")["nilai"].sum().sort_values().reset_index()
        fig = px.bar(pk.tail(15), x="nilai", y="akun", orientation="h", title="Penjualan per kategori", labels={"nilai": "Total", "akun": ""})
        fig.update_layout(height=360)
        kiri.plotly_chart(fig, width="stretch")
        ph = df.assign(hari=df["tanggal"].dt.day_name()).groupby("hari")["nilai"].sum()
        urut = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig = px.bar(ph.reindex(urut).reset_index(), x="hari", y="nilai", title="Penjualan per hari dalam seminggu", labels={"nilai": "Total", "hari": ""})
        fig.update_layout(height=360)
        kanan.plotly_chart(fig, width="stretch")
        st.info("Data penjualan tidak berisi biaya, jadi laba/margin tidak dihitung. Fokus analisis: pola penjualan, retur, dan transaksi tidak wajar.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Pendapatan", an.rp_ringkas(r["Pendapatan"], simbol))
        b.metric("Laba kotor", an.rp_ringkas(r["Laba Kotor"], simbol), f"{r['Margin Kotor %']:.1f}% margin" if pd.notna(r["Margin Kotor %"]) else None)
        c.metric("Beban operasional", an.rp_ringkas(r["Beban Operasional"], simbol), f"{r['Rasio Beban %']:.1f}% dari pendapatan" if pd.notna(r["Rasio Beban %"]) else None, delta_color="off")
        d.metric("Laba bersih", an.rp_ringkas(r["Laba Bersih"], simbol), f"{r['Margin Bersih %']:.1f}% margin" if pd.notna(r["Margin Bersih %"]) else None)
        if r["Pendapatan"] == 0:
            st.warning("Tidak ada baris berkategori Pendapatan. Periksa pemetaan kolom 'Kategori/tipe akun' atau nama akun.")
        kiri, kanan = st.columns(2)
        wf = go.Figure(go.Waterfall(
            x=["Pendapatan", "HPP", "Beban", "Laba bersih"], measure=["absolute", "relative", "relative", "total"],
            y=[r["Pendapatan"], -r["HPP"], -r["Beban Operasional"], 0], connector={"line": {"color": "#888"}}))
        wf.update_layout(title="Dari pendapatan ke laba bersih", height=360, showlegend=False)
        kiri.plotly_chart(wf, width="stretch")
        pa = an.per_akun(df)
        pa = pa[pa["kategori"].isin(["Beban", "HPP"])].head(10)
        if len(pa):
            fig = px.bar(pa.sort_values("total_nilai"), x="total_nilai", y="akun", orientation="h", color="kategori",
                         title="Akun biaya terbesar", labels={"total_nilai": "Total", "akun": ""})
            fig.update_layout(height=360)
            kanan.plotly_chart(fig, width="stretch")
        st.subheader("Rasio utama")
        st.dataframe(pd.DataFrame({"Rasio": ["Margin kotor", "Margin bersih", "HPP / pendapatan", "Beban / pendapatan"],
                                   "Nilai (%)": [r["Margin Kotor %"], r["Margin Bersih %"], r["Rasio HPP %"], r["Rasio Beban %"]]})
                     .style.format({"Nilai (%)": "{:.1f}"}), hide_index=True, width="stretch")

with t2:
    if len(tren) < 2:
        st.info("Data kurang dari 2 bulan - tren tidak dapat ditampilkan.")
    else:
        fig = px.line(tren, x="bulan", y=["Pendapatan"] if penjualan else ["Pendapatan", "HPP", "Beban", "Laba Bersih"], markers=True,
                      title="Tren penjualan bulanan" if penjualan else "Tren bulanan")
        pr = an.proyeksi(tren, "Pendapatan", 3)
        if pr is not None:
            fig.add_scatter(x=pr["bulan"], y=pr["proyeksi"], mode="lines+markers", name="Proyeksi pendapatan",
                            line=dict(dash="dash"))
            fig.add_scatter(x=pd.concat([pr["bulan"], pr["bulan"][::-1]]),
                            y=pd.concat([pr["batas_atas"], pr["batas_bawah"][::-1]]),
                            fill="toself", opacity=0.15, line=dict(width=0), name="Rentang 95%", hoverinfo="skip")
        fig.update_layout(height=460, legend_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Proyeksi = regresi linear sederhana, hanya gambaran arah - bukan peramalan formal. Bulan terakhir bisa belum penuh.")
        st.dataframe(tren.assign(bulan=tren["bulan"].dt.strftime("%Y-%m")).style.format(
            {c: "{:,.0f}" for c in ["Pendapatan", "HPP", "Beban", "Laba Kotor", "Laba Bersih"]} |
            {"Margin Bersih %": "{:.1f}", "Pendapatan MoM %": "{:+.1f}"}), hide_index=True, width="stretch")

with t3:
    sr = an.ringkas_temuan(tem)
    n_t, n_s = int((tem["level"] == "Tinggi").sum()), int((tem["level"] == "Sedang").sum())
    a, b, c = st.columns(3)
    a.metric("Baris risiko tinggi", f"{n_t:,}")
    b.metric("Baris risiko sedang", f"{n_s:,}")
    c.metric("Uji Benford (MAD)", f"{mad:.4f}" if pd.notna(mad) else "-")
    c.caption(verdict)
    kiri, kanan = st.columns([3, 2])
    fig = px.bar(sr, x="Jumlah baris", y="Jenis temuan", orientation="h", title="Jumlah baris per jenis temuan")
    fig.update_layout(height=340, yaxis={"categoryorder": "total ascending"})
    kiri.plotly_chart(fig, width="stretch")
    if len(benf):
        fb = go.Figure([go.Bar(x=benf["Digit"], y=benf["Teramati %"], name="Teramati"),
                        go.Scatter(x=benf["Digit"], y=benf["Benford %"], name="Harapan Benford", mode="lines+markers")])
        fb.update_layout(title="Digit pertama vs Hukum Benford", height=340, xaxis=dict(dtick=1))
        kanan.plotly_chart(fb, width="stretch")
    with st.expander("Apa arti tiap temuan?"):
        st.markdown("""
- **Jurnal tidak seimbang** - total debit ≠ kredit dalam satu nomor jurnal (bobot 4)
- **Duplikat** - tanggal, akun, nominal, dan keterangan sama persis (bobot 3)
- **Nilai ekstrem** - jauh di luar kebiasaan kategorinya (robust z-score > 3,5 pada skala log) (bobot 3)
- **Tepat di bawah ambang** - 90-99,9% dari ambang persetujuan; pola khas pemecahan transaksi (bobot 3)
- **Angka bulat besar** - nominal sangat bulat dan termasuk 10% terbesar (bobot 1)
- **Posting akhir pekan** - dicatat Sabtu/Minggu (bobot 1; dimatikan untuk data penjualan)
- **Retur / nilai negatif** - transaksi refund/retur, hanya untuk data penjualan (bobot 1)

Skor: 1 = rendah, 2-3 = sedang, ≥4 = tinggi. Ini **indikator untuk ditelusuri**, bukan bukti kecurangan.""")
    level = st.multiselect("Tampilkan level", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang"])
    daftar = tem[tem["level"].isin(level)].sort_values(["skor", "nominal"], ascending=False)
    st.dataframe(daftar[["tanggal", "no_jurnal", "akun", "deskripsi", "debit", "kredit", "temuan", "skor", "level"]],
                 hide_index=True, width="stretch", height=380)

with t4:
    konteks = an.konteks_ai(df, tem, ambang, simbol, info["mode"])
    if "chat" not in st.session_state:
        st.session_state.chat = []
    c1, c2 = st.columns([1, 1])
    if c1.button("✨ Buat ringkasan eksekutif AI", width="stretch"):
        with st.spinner("AI menganalisis..."):
            jawab = tanya_ai([{"role": "system", "content": SISTEM},
                              {"role": "user", "content": f"DATA:\n{konteks}\n\nBuat ringkasan eksekutif: (1) kondisi kinerja, (2) 3 temuan risiko terpenting beserta alasannya, (3) rekomendasi tindak lanjut audit berurutan prioritas."}],
                             model)
        st.session_state.ringkasan_ai = jawab
    if c2.button("🔄 Hapus riwayat chat", width="stretch"):
        st.session_state.chat = []
    if st.session_state.get("ringkasan_ai"):
        with st.container(border=True):
            st.markdown(st.session_state.ringkasan_ai)
    st.divider()
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Tanyakan sesuatu, mis. 'Kenapa margin bersih turun?' atau 'Jurnal mana yang harus diperiksa dulu?'")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        pesan = [{"role": "system", "content": SISTEM + "\n\nKONTEKS DATA:\n" + konteks}] + st.session_state.chat[-10:]
        with st.spinner("AI berpikir..."):
            jawab = tanya_ai(pesan, model)
        st.session_state.chat.append({"role": "assistant", "content": jawab})
        st.rerun()
    st.caption("AI hanya melihat ringkasan statistik dan 12 transaksi berisiko tertinggi, bukan seluruh baris data.")

with t5:
    st.subheader("Unduh laporan Excel")
    st.caption("Berisi sheet: Ringkasan, Tren Bulanan, Per Akun, Ringkasan Risiko, Daftar Temuan, Benford" +
               (", Analisis AI." if st.session_state.get("ringkasan_ai") else ". (Buat ringkasan AI di tab 'Tanya AI' dulu agar ikut tercantum.)"))
    st.download_button("📥 Unduh Excel", report.buat_excel(df, tem, ambang, st.session_state.get("ringkasan_ai", "")),
                       file_name="laporan_audit_akuntansi.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.subheader("Data setelah dinormalkan")
    st.dataframe(tem.head(2000), hide_index=True, width="stretch", height=360)
