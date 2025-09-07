# app.py — Notas por Regional (menu lateral espaçado, fonte maior, 2 linhas no gráfico 2)
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_numeric_dtype

# ===================== CONFIG =====================
PAGE_TITLE = "Notas por Regional: Rio de Janeiro"
FONT_SIZE = 26          # ↑ fonte geral
MARKER_SIZE = 12
DEFAULT_XLSX = "data/Comparativo_RJ.xlsx"
# ==================================================

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.caption("")

# ---------- CSS: fonte maior e espaçamento vertical no menu da esquerda ----------
st.markdown(
    f"""
<style>
/* Fonte base ampla */
html, body, [class*="css"] {{ font-size: {FONT_SIZE}px !important; }}

/* Labels maiores */
.stSelectbox label {{ font-size: {FONT_SIZE}px !important; }}
.stSelectbox div[data-baseweb="select"] div {{ font-size: {FONT_SIZE}px !important; }}
div[role="radiogroup"] label {{ font-size: {FONT_SIZE}px !important; }}
div[role="radiogroup"] p {{ font-size: {FONT_SIZE}px !important; }}

/* Espaçamento vertical suave entre as opções do radio (menu lateral) */
div[role="radiogroup"] > * {{ margin-bottom: 10px !important; }}
/* alguns layouts do Streamlit usam divs internos; garantimos o padding também */
div[role="radiogroup"] > div {{ padding: 4px 0 !important; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Utilitários ----------
def resolve_excel_path(default_rel_path: str = DEFAULT_XLSX) -> Path | None:
    here = Path(__file__).parent
    try:
        secret_path = st.secrets.get("EXCEL_PATH", None)
    except Exception:
        secret_path = None
    if secret_path:
        p = Path(str(secret_path))
        if not p.is_absolute():
            p = here / p
        return p
    return here / default_rel_path

def parse_br_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("\xa0", "").replace(" ", "")
    for sym in ["R$", "%"]:
        s = s.replace(sym, "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def ler_aba(excel_path: Path, sheet_name: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        return None

def ler_abas_local(caminho: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(caminho, sheet_name=None, engine="openpyxl")

def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns[0] != "Regional":
        df = df.rename(columns={df.columns[0]: "Regional"})
    ultima_rotulo = str(df.iloc[-1, 0]).lower()
    if "presen" in ultima_rotulo:
        df = df.iloc[:-1].copy()
    aval_cols = list(df.columns[1:])
    for c in aval_cols:
        if not is_numeric_dtype(df[c]):
            df[c] = df[c].map(parse_br_number)
    df_long = df.melt(
        id_vars="Regional", value_vars=aval_cols,
        var_name="Avaliação", value_name="Valor"
    )
    df_long["Avaliação"] = pd.Categorical(df_long["Avaliação"],
                                          categories=aval_cols, ordered=True)
    return df_long

def montar_base(df_long: pd.DataFrame, regional: str) -> pd.DataFrame:
    base = df_long[df_long["Regional"] == regional].sort_values("Avaliação").copy()
    base["Anterior"] = base["Valor"].shift(1)
    base["Delta"] = base["Valor"] - base["Anterior"]
    base["Delta_pct"] = (base["Delta"] / base["Anterior"]) * 100

    def fsgn(x): return "—" if pd.isna(x) else f"{x:+.2f}"
    def fnum(x): return "—" if pd.isna(x) else f"{x:.2f}"

    base["hover_text"] = (
        "<b>" + base["Avaliação"].astype(str) + "</b>"
        + "<br>Valor: " + base["Valor"].map(fnum)
        + "<br>Δ abs.: " + base["Delta"].map(fsgn)
        + "<br>Δ %: " + base["Delta_pct"].map(fsgn) + "%"
    )
    return base

# ---------- Gráficos ----------
def grafico_notas(base_notas: pd.DataFrame, titulo: str):
    # Textos só com o número da nota, sempre acima
    txt = base_notas["Valor"].map(lambda v: f"{v:.2f}")
    fig = px.line(base_notas, x="Avaliação", y="Valor", markers=True, title=titulo)
    fig.update_traces(
        marker=dict(size=MARKER_SIZE),
        text=txt,
        textposition="top center",
        textfont=dict(size=FONT_SIZE),
        hovertext=base_notas["hover_text"],
        hovertemplate="%{hovertext}<extra></extra>",
        line=dict(width=3),
    )
    y_min = 0.95 * float(base_notas["Valor"].min())
    y_max = 1.05 * float(base_notas["Valor"].max())
    fig.update_layout(
        font=dict(size=FONT_SIZE),
        xaxis_title="", yaxis_title="",
        xaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE),
                   range=[y_min, y_max]),
        legend=dict(font=dict(size=FONT_SIZE)),
        hovermode="x unified",
        hoverlabel=dict(font_size=FONT_SIZE),
        height=470,
    )
    return fig

def grafico_participacao_insuficiente(base_part: pd.DataFrame,
                                      base_insuf: pd.DataFrame,
                                      titulo: str):
    # Combina duas séries em um único DF longo
    a = base_part[["Avaliação", "Valor"]].copy()
    a["Métrica"] = "Participação (%)"
    b = base_insuf[["Avaliação", "Valor"]].copy()
    b["Métrica"] = "Texto insuficiente (%)"
    df_plot = pd.concat([a, b], ignore_index=True)

    # tooltips simples em %
    def fnum(x): return "—" if pd.isna(x) else f"{x:.2f}%"
    hover = (
        "<b>" + df_plot["Avaliação"].astype(str) + "</b>"
        + "<br>" + df_plot["Métrica"] + ": "
        + df_plot["Valor"].map(lambda v: f"{v:.2f}")
        + "%"
    )
    df_plot["hover_text"] = hover

    fig = px.line(
        df_plot, x="Avaliação", y="Valor", color="Métrica",
        markers=True, title=titulo
    )
    fig.update_traces(
        marker=dict(size=MARKER_SIZE),
        hovertext=df_plot["hover_text"],
        hovertemplate="%{hovertext}<extra></extra>",
        line=dict(width=3),
    )

    # eixo Y de 0 a 100% com margem
    y_min = max(0, df_plot["Valor"].min() - 5)
    y_max = min(100, df_plot["Valor"].max() + 5)
    fig.update_layout(
        font=dict(size=FONT_SIZE),
        xaxis_title="", yaxis_title="",
        xaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE),
                   range=[y_min, y_max]),
        legend=dict(font=dict(size=FONT_SIZE)),
        hovermode="x unified",
        hoverlabel=dict(font_size=FONT_SIZE),
        height=470,
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

col_nav, col_main = st.columns([1, 2], gap="large")

with col_nav:
    st.markdown("")
    aba_sel = st.radio(
        label="Abas",
        options=tab_names,
        index=0,
        width=300,
        key="aba_radio",
        label_visibility="collapsed"
    )

with col_main:
    st.subheader(aba_sel)

    # ----- NOTAS -----
    df_sheet = abas[aba_sel].copy()
    df_long_notas = preparar_df(df_sheet)

    regionais = df_long_notas["Regional"].dropna().unique()
    regional = st.selectbox(
        "Regional",
        sorted(regionais),
        key=f"reg_{aba_sel}",
        label_visibility="visible"
    )

    base_notas = montar_base(df_long_notas, regional)
    fig1 = grafico_notas(base_notas, "Evolução das Notas")
    st.plotly_chart(fig1, use_container_width=True)

    # ----- PARTICIPAÇÃO e TEXTO INSUFICIENTE -----
    # Procuramos por abas-irmãs no mesmo xlsx
    sheet_part = f"{aba_sel} - Participação"
    sheet_insu = f"{aba_sel} - Texto Insuficiente"

    df_part = ler_aba(excel_path, sheet_part)
    df_insu = ler_aba(excel_path, sheet_insu)

    if (df_part is None) or (df_insu is None):
        st.info(
            "Para o 2º gráfico, eu procuro as abas "
            f"`{sheet_part}` e `{sheet_insu}` no mesmo Excel.\n"
            "Se não existirem, crie-as (estrutura igual às abas de notas, com Regionais nas linhas e avaliações nas colunas)."
        )
    else:
        df_long_part = preparar_df(df_part)
        df_long_insu = preparar_df(df_insu)
        base_part = montar_base(df_long_part, regional)
        base_insu = montar_base(df_long_insu, regional)

        fig2 = grafico_participacao_insuficiente(
            base_part, base_insu, "Participação e Textos Insuficientes"
        )
        st.plotly_chart(fig2, use_container_width=True)
