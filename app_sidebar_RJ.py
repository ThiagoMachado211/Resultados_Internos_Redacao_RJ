# app.py — Notas por Regional (menu lateral, fontes 24, marcadores 12)
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_numeric_dtype

# ===================== CONFIG =====================
PAGE_TITLE = "Notas por Regional: Rio de Janeiro"
FONT_SIZE = 24
MARKER_SIZE = 12
DEFAULT_XLSX = "data/Comparativo_RJ.xlsx"  # ajuste se quiser outro arquivo padrão
# ==================================================

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.caption("")

# ---------- CSS: deixa UI com fonte 24 e menu lateral vertical ----------
st.markdown(f"""
<style>
/* Fonte base ampla */
html, body, [class*="css"] {{ font-size: {FONT_SIZE}px !important; }}

/* Label e opções do selectbox */
.stSelectbox label {{ font-size: {FONT_SIZE}px !important; }}
.stSelectbox div[data-baseweb="select"] div {{ font-size: {FONT_SIZE}px !important; }}

/* Radio (menu) — rótulos grandes */
div[role="radiogroup"] label {{ font-size: {FONT_SIZE}px !important; }}
div[role="radiogroup"] p {{ font-size: {FONT_SIZE}px !important; }}
</style>
""", unsafe_allow_html=True)

# ---------- Utilitários ----------
def resolve_excel_path(default_rel_path: str = DEFAULT_XLSX) -> Path | None:
    """
    Resolve o caminho do Excel:
      1) se existir st.secrets["EXCEL_PATH"], usa (relativo à pasta do app se não for absoluto);
      2) senão, usa default_rel_path relativo à pasta do app.
    """
    here = Path(__file__).parent
    # 1) tentar via secrets
    try:
        secret_path = st.secrets.get("EXCEL_PATH", None)
    except Exception:
        secret_path = None
    if secret_path:
        p = Path(str(secret_path))
        if not p.is_absolute():
            p = here / p
        return p
    # 2) padrão
    return here / default_rel_path

def parse_br_number(x):
    """Converte '452,74', '1.234,56', '1,234.56' em float; mantém floats/ints."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("\xa0", "").replace(" ", "")
    for sym in ["R$", "%"]:
        s = s.replace(sym, "")
    if "," in s and "." in s:
        # último símbolo define o decimal
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")   # 1.234,56 -> 1234.56
        else:
            s = s.replace(",", "")                     # 1,234.56 -> 1234.56
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")       # 452,74 -> 452.74
    else:
        s = s.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def ler_abas_local(caminho: Path) -> dict[str, pd.DataFrame]:
    """Lê todas as sheets do Excel."""
    return pd.read_excel(caminho, sheet_name=None, engine="openpyxl")

def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza, remove última linha se for 'presença' e devolve df_long (Regional, Avaliação, Nota)."""
    # Garante nome da primeira coluna
    if df.columns[0] != "Regional":
        df = df.rename(columns={df.columns[0]: "Regional"})

    # Se a última linha for de presença (coluna A contém 'presen'), descarta
    ultima_rotulo = str(df.iloc[-1, 0]).lower()
    if "presen" in ultima_rotulo:
        df = df.iloc[:-1].copy()

    # Normaliza números em colunas de avaliação
    aval_cols = list(df.columns[1:])
    for c in aval_cols:
        if not is_numeric_dtype(df[c]):
            df[c] = df[c].map(parse_br_number)

    # Formato longo
    df_long = df.melt(id_vars="Regional", value_vars=aval_cols,
                      var_name="Avaliação", value_name="Nota")
    df_long["Avaliação"] = pd.Categorical(df_long["Avaliação"],
                                          categories=aval_cols, ordered=True)
    return df_long

def montar_base(df_long: pd.DataFrame, regional: str) -> pd.DataFrame:
    """Filtra regional, ordena por avaliação e calcula Δ e Δ%."""
    base = df_long[df_long["Regional"] == regional].sort_values("Avaliação").copy()
    base["Nota_anterior"] = base["Nota"].shift(1)
    base["Delta"] = base["Nota"] - base["Nota_anterior"]
    base["Delta_pct"] = (base["Delta"] / base["Nota_anterior"]) * 100

    def fsgn(x): return "—" if pd.isna(x) else f"{x:+.2f}"
    def fnum(x): return "—" if pd.isna(x) else f"{x:.2f}"

    # Rótulo visível (sem presença)
    base["label_text"] = np.where(
        base["Nota_anterior"].isna(),
        base["Nota"].map(fnum).radd("Nota "),
        base.apply(lambda r: f"Nota {fnum(r['Nota'])} (Δ {fsgn(r['Delta'])}; {fsgn(r['Delta_pct'])}%)", axis=1)
    )

    # Tooltip
    base["hover_text"] = (
        "<b>" + base["Avaliação"].astype(str) + "</b>"
        + "<br>Nota: " + base["Nota"].map(fnum)
        + "<br>Variação Absoluta: " + base["Delta"].map(fsgn)
        + "<br>Variação Percentual: " + base["Delta_pct"].map(fsgn) + "%"
    )
    return base

def grafico(base: pd.DataFrame, titulo: str):
    fig = px.line(base, x="Avaliação", y="Nota", markers=True, title=titulo)
    fig.update_traces(
        marker=dict(size=MARKER_SIZE),
        text=base["label_text"],
        textposition="top center",
        textfont=dict(size=FONT_SIZE),
        hovertext=base["hover_text"],
        hovertemplate="%{hovertext}<extra></extra>"
    )
    fig.update_layout(
        font=dict(size=FONT_SIZE),
        xaxis_title="", yaxis_title="",
        xaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE), range=[0.95*float(base["Nota"].min()),  1.05*float(base["Nota"].max())] ),
        legend=dict(font=dict(size=FONT_SIZE)),
        hovermode="x unified",
        hoverlabel=dict(font_size=FONT_SIZE),
        height=450,
        width=300
    )
    return fig

# ---------- Main ----------
excel_path = resolve_excel_path()
if not excel_path.exists():
    st.error(
        f"Arquivo .xlsx não encontrado em: `{excel_path}`.\n\n"
        "• Coloque o arquivo no repositório nesse caminho OU\n"
        "• Defina `EXCEL_PATH` em Settings → Secrets (ex.: EXCEL_PATH=\"data/Comparativo_MG.xlsx\")."
    )
    st.stop()

abas = ler_abas_local(excel_path)
tab_names = list(abas.keys())

# Layout: coluna esquerda = menu; direita = conteúdo
col_nav, col_main = st.columns([1,4], gap="large")

with col_nav:
    st.markdown("")
    aba_sel = st.radio(
        label="Abas",                  # rótulo não-vazio (evita warning de acessibilidade)
        options=tab_names,
        index=0,
        key="aba_radio",
        label_visibility="collapsed"   # oculta o texto do rótulo
    )

with col_main:
    st.subheader(aba_sel)
    df_sheet = abas[aba_sel].copy()
    df_long = preparar_df(df_sheet)

    regionais = df_long["Regional"].dropna().unique()
    regional = st.selectbox(
        "Regional",
        sorted(regionais),
        key=f"reg_{aba_sel}",
        label_visibility="visible"
    )

    base = montar_base(df_long, regional)
    fig = grafico(base, f"")

    st.plotly_chart(fig, use_container_width=True)







