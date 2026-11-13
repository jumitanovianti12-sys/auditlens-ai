"""report.py - Ekspor hasil analisis ke satu file Excel berisi beberapa sheet."""
from __future__ import annotations

import io

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import analysis as an

HEAD = PatternFill("solid", fgColor="1F2A44")
FONT = Font(color="FFFFFF", bold=True)


def _tulis(wr, df: pd.DataFrame, nama: str) -> None:
    df.to_excel(wr, sheet_name=nama, index=False)
    ws = wr.sheets[nama]
    for c in ws[1]:
        c.fill, c.font, c.alignment = HEAD, FONT, Alignment(horizontal="center", vertical="center")
    for i, col in enumerate(df.columns, 1):
        lebar = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)]) + 2
        ws.column_dimensions[get_column_letter(i)].width = min(lebar, 48)
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            for cell in ws[get_column_letter(i)][1:]:
                cell.number_format = "#,##0.0" if "%" in str(col) else "#,##0"
    ws.freeze_panes = "A2"


def buat_excel(df: pd.DataFrame, tem: pd.DataFrame, ambang: float, ai_teks: str = "") -> bytes:
    ring = an.ringkasan(df)
    ringkas = pd.DataFrame({"Indikator": list(ring), "Nilai": [str(v) if isinstance(v, str) else v for v in ring.values()]})
    tren = an.tren_bulanan(df)
    tren["bulan"] = tren["bulan"].dt.strftime("%Y-%m")
    risiko = tem[tem["skor"] > 0].sort_values(["skor", "nominal"], ascending=False)
    risiko = risiko[["tanggal", "no_jurnal", "akun", "kategori", "deskripsi", "debit", "kredit", "temuan", "skor", "level"]]
    risiko["tanggal"] = risiko["tanggal"].dt.strftime("%Y-%m-%d")
    bf, mad, verdict = an.benford(df)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
        _tulis(wr, ringkas, "Ringkasan")
        _tulis(wr, tren, "Tren Bulanan")
        _tulis(wr, an.per_akun(df), "Per Akun")
        _tulis(wr, an.ringkas_temuan(tem), "Ringkasan Risiko")
        _tulis(wr, risiko, "Daftar Temuan")
        if len(bf):
            _tulis(wr, bf, "Benford")
        if ai_teks:
            _tulis(wr, pd.DataFrame({"Analisis AI": ai_teks.split("\n")}), "Analisis AI")
            wr.sheets["Analisis AI"].column_dimensions["A"].width = 120
    return buf.getvalue()
