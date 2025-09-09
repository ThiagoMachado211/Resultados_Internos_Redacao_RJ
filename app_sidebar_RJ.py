# app_sidebar_RJ.py — Três gráficos padronizados (médias, participação, insuficiente)
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_numeric_dtype

# ===================== CONFIG =====================
PAGE_TITLE = "Notas por Regional: Rio de Janeiro"
FONT_SIZE = 24          # <- fonte padrão (ticks, labels, legenda, título, hover)
MARKER_SIZE = 12        # <- tamanho padrão dos marcadores
DEFAULT_XLSX = "data/Comparativo_RJ.xlsx"
Y_PAD_PCT = 0.05        # padding vertical do gráfico de médias
# ==================================================

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.caption("")

# ---------- CSS: fonte da UI ===========
st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-size: {FONT_SIZE}px !important; }}
.stSelectbox label {{ font-size: {FONT_SIZE}px !important; }}
.stSelectbox div[data-baseweb="select"] div {{ font-size: {FONT_SIZE}px !important; }}
div[role="radiogroup"] label {{ font-size: {FONT_SIZE}px !important; }}
div[role="radiogroup"] p {{ font-size: {FONT_SIZE}px !important; }}
</style>
""", unsafe_allow_html=True)

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
    """Converte '452,74', '1.234,56', '1,234.56', '57%' em float; mantém floats/ints."""
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

def as_percent(series: pd.Series) -> pd.Series:
    """Garante 0–100. Se os dados vierem 0–1, multiplica por 100."""
    s = pd.to_numeric(series, errors="coerce")
    return s * 100 if s.max() is not np.nan and s.max() <= 1.2 else s

def ler_abas_local(caminho: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(caminho, sheet_name=None, engine="openpyxl")

def preparar_df_medias(df: pd.DataFrame) -> pd.DataFrame:
    """Médias: retorna df_long (Regional, Avaliação, Nota). Remove linha 'presença' se houver."""
    if df.columns[0] != "Regional":
        df = df.rename(columns={df.columns[0]: "Regional"})
    ultima_rotulo = str(df.iloc[-1, 0]).lower()
    if "presen" in ultima_rotulo:
        df = df.iloc[:-1].copy()
    aval_cols = list(df.columns[1:])
    for c in aval_cols:
        if not is_numeric_dtype(df[c]):
            df[c] = df[c].map(parse_br_number)
    df_long = df.melt(id_vars="Regional", value_vars=aval_cols,
                      var_name="Avaliação", value_name="Nota")
    df_long["Avaliação"] = pd.Categorical(df_long["Avaliação"],
                                          categories=aval_cols, ordered=True)
    return df_long

def preparar_df_percentual(df: pd.DataFrame) -> pd.DataFrame:
    """Participação / Insuf.: devolve df_long (Regional, Avaliação, Valor[0–100])."""
    # Primeira coluna deve ser Regional
    if df.columns[0] != "Regional":
        df = df.rename(columns={df.columns[0]: "Regional"})
    aval_cols = list(df.columns[1:])
    for c in aval_cols:
        if not is_numeric_dtype(df[c]):
            df[c] = df[c].map(parse_br_number)
    df_long = df.melt(id_vars="Regional", value_vars=aval_cols,
                      var_name="Avaliação", value_name="Valor")
    df_long["Avaliação"] = pd.Categorical(df_long["Avaliação"],
                                          categories=aval_cols, ordered=True)
    df_long["Valor"] = as_percent(df_long["Valor"])
    return df_long

def montar_base_linha(df_long: pd.DataFrame, regional: str, valor_col: str) -> pd.DataFrame:
    """Calcula deltas e labels (genérico para médias e percentuais)."""
    base = df_long[df_long["Regional"] == regional].sort_values("Avaliação").copy()
    base["Anterior"] = base[valor_col].shift(1)
    base["Delta"] = base[valor_col] - base["Anterior"]
    base["Delta_pct"] = (base["Delta"] / base["Anterior"]) * 100

    def fsgn(x): return "—" if pd.isna(x) else f"{x:+.2f}"
    def fnum(x): return "—" if pd.isna(x) else f"{x:.2f}"

    base["hover_text"] = (
        "<b>" + base["Avaliação"].astype(str) + "</b>"
        + f"<br>Valor: " + base[valor_col].map(fnum)
        + "<br>Variação Absoluta: " + base["Delta"].map(fsgn)
        + "<br>Variação Percentual: " + base["Delta_pct"].map(fsgn) + "%"
    )
    return base

def cores_por_delta(deltas):
    cores = []
    for i, d in enumerate(deltas):
        if i == 0 or pd.isna(d):
            cores.append("black")
        else:
            cores.append("royalblue" if d > 0 else "crimson")
    return cores

# ===================== GRÁFICOS PADRONIZADOS =====================
def grafico_medias(base: pd.DataFrame, titulo: str):
    """Linha única de médias (usa coluna 'Nota')."""
    ymin = float(base["Nota"].min()) * (1 - Y_PAD_PCT)
    ymax = float(base["Nota"].max()) * (1 + Y_PAD_PCT)

    fig = px.line(base, x="Avaliação", y="Nota", markers=True, title=titulo)
    fig.update_traces(
        marker=dict(size=MARKER_SIZE),
        text=base["Nota"].map(lambda v: f"Nota {v:.2f}"),
        textposition="top center",
        textfont=dict(size=FONT_SIZE, color=cores_por_delta(base["Delta"].tolist())),
        hovertext=base["hover_text"],
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False
    )
    fig.update_layout(
        font=dict(size=FONT_SIZE),
        xaxis_title="", yaxis_title="",
        xaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE), range=[ymin, ymax]),
        legend=dict(font=dict(size=FONT_SIZE)),
        title_font=dict(size=FONT_SIZE),
        hovermode="x unified",
        hoverlabel=dict(font_size=FONT_SIZE),
        height=600
    )
    return fig

def _grafico_percentual(base: pd.DataFrame, titulo: str, valor_col: str):
    """Linha única percentual (0–100%) usando 'Valor'."""
    base = base.copy()
    base["Label"] = base[valor_col].map(lambda v: f"{v:.2f}%")

    fig = px.line(base, x="Avaliação", y=valor_col, markers=True, text="Label", title=titulo)
    fig.update_traces(
        mode="lines+markers+text",
        marker=dict(size=MARKER_SIZE),
        line=dict(width=3),
        textposition="top center",
        textfont=dict(size=FONT_SIZE, color=cores_por_delta(base["Delta"].tolist())),
        hovertext=base["hover_text"],
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False
    )
    fig.update_layout(
        font=dict(size=FONT_SIZE),
        xaxis_title="", yaxis_title="",
        xaxis=dict(tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE)),
        yaxis=dict(
            tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE),
            range=[-15, 115], ticksuffix="%"
        ),
        title_font=dict(size=FONT_SIZE),
        hovermode="x unified",
        hoverlabel=dict(font_size=FONT_SIZE),
        height=600
    )
    return fig

def grafico_participacao(base_part: pd.DataFrame, titulo: str = "Participação (%)"):
    return _grafico_percentual(base_part, titulo, valor_col="Valor")

def grafico_texto_insuficiente(base_insuf: pd.DataFrame, titulo: str = "Texto insuficiente (%)"):
    return _grafico_percentual(base_insuf, titulo, valor_col="Valor")

# ---------- helpers de busca de sheet ----------
def achar_sheet(abas_dict: dict, *keywords) -> str | None:
    """Retorna o nome da sheet cujo título contém todas as keywords (case-insensitive)."""
    keys = [k.lower() for k in abas_dict.keys()]
    for name in abas_dict.keys():
        low = name.lower()
        if all(k in low for k in keywords):
            return name
    return None

# ===================== MAIN =====================
excel_path = resolve_excel_path()
if not excel_path.exists():
    st.error(
        f"Arquivo .xlsx não encontrado em: `{excel_path}`.\n\n"
        "• Coloque o arquivo nesse caminho OU\n"
        "• Defina EXCEL_PATH em Settings → Secrets (ex.: EXCEL_PATH=\"data/Comparativo_RJ.xlsx\")."
    )
    st.stop()

abas = ler_abas_local(excel_path)

tab_medias, tab_part, tab_insu = st.tabs(["Médias", "Participação", "Texto insuficiente"])

# ---------- TAB: MÉDIAS ----------
with tab_medias:
    tab_names = list(abas.keys())
    col_nav, col_main = st.columns([1, 4], gap="large")

    with col_nav:
        st.markdown("")
        aba_sel = st.radio(
            label="Abas",
            options=tab_names,
            index=0,
            key="aba_radio_medias",
            label_visibility="collapsed"
        )

    with col_main:
        st.subheader(aba_sel)
        df_sheet = abas[aba_sel].copy()
        df_long = preparar_df_medias(df_sheet)
        regionais = df_long["Regional"].dropna().unique()
        regional = st.selectbox(
            "Regional", sorted(regionais), key=f"reg_{aba_sel}_medias", label_visibility="visible"
        )
        base = montar_base_linha(df_long, regional, valor_col="Nota")
        fig = grafico_medias(base, "")
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB: PARTICIPAÇÃO ----------
with tab_part:
    nome_sheet_part = achar_sheet(abas, "particip")
    if not nome_sheet_part:
        st.info("Sheet de **Participação** não encontrada no Excel (procuro por 'particip' no nome).")
    else:
        dfp = preparar_df_percentual(abas[nome_sheet_part].copy())
        regionais = dfp["Regional"].dropna().unique()
        regp = st.selectbox("Regional", sorted(regionais), key="reg_part", label_visibility="visible")
        basep = montar_base_linha(dfp, regp, valor_col="Valor")
        figp = grafico_participacao(basep, "Participação (%)")
        st.plotly_chart(figp, use_container_width=True)

# ---------- TAB: TEXTO INSUFICIENTE ----------
with tab_insu:
    nome_sheet_insu = achar_sheet(abas, "insuf")
    if not nome_sheet_insu:
        st.info("Sheet de **Texto insuficiente** não encontrada no Excel (procuro por 'insuf' no nome).")
    else:
        dfi = preparar_df_percentual(abas[nome_sheet_insu].copy())
        regionais = dfi["Regional"].dropna().unique()
        regi = st.selectbox("Regional", sorted(regionais), key="reg_insu", label_visibility="visible")
        basei = montar_base_linha(dfi, regi, valor_col="Valor")
        figi = grafico_texto_insuficiente(basei, "Texto insuficiente (%)")
        st.plotly_chart(figi, use_container_width=True)
