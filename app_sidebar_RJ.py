# app.py — Notas por Regional (mantém opções originais; 2 linhas no gráfico 2)
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_numeric_dtype

# ===================== CONFIG =====================
PAGE_TITLE = "Notas por Regional: Rio de Janeiro"
FONT_SIZE = 36         # fonte maior
MARKER_SIZE = 12
DEFAULT_XLSX = "data/Comparativo_RJ.xlsx"
# ==================================================


st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)
st.caption("")


# ---------- CSS: fonte maior + espaçamento no menu esquerdo ----------
st.markdown(
    f"""
<style>
html, body, [class*="css"] {{ font-size: {FONT_SIZE}px !important; }}
.stSelectbox label {{ font-size: {FONT_SIZE}px !important; }}
.stSelectbox div[data-baseweb="select"] div {{ font-size: {FONT_SIZE}px !important; }}
div[role="radiogroup"] label, div[role="radiogroup"] p {{ font-size: {FONT_SIZE}px !important; }}

/* respiro vertical nas opções do radio */
div[role="radiogroup"] > * {{ margin-bottom: 10px !important; }}
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


def ler_abas_local(caminho: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(caminho, sheet_name=None, engine="openpyxl")


def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns[0] != "Regional":
        df = df.rename(columns={df.columns[0]: "Regional"})
    # se a última linha for algo como "Presença" (não-regional), descarta
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
    df_long["Avaliação"] = pd.Categorical(
        df_long["Avaliação"], categories=aval_cols, ordered=True
    )
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


# --------- localizar abas companheiras (sem aparecer no menu) ----------
def achar_companheiras(todas_abas: dict, base_name: str):
    """Encontra as abas de 'Participação' e 'Texto Insuficiente' correspondentes ao base_name.
    É tolerante a variações: 'participa', 'texto insuf', nomes cortados pelo limite do Excel, etc."""
    base_lower = base_name.lower()
    part_name = None
    insuf_name = None
    for nome in todas_abas.keys():
        low = nome.lower()
        if low == base_lower:
            continue
        if low.startswith(base_lower) and "particip" in low:
            part_name = nome
        if low.startswith(base_lower) and ("texto" in low and "insuf" in low):
            insuf_name = nome
    return part_name, insuf_name



# ---------- Gráficos ----------
def grafico_notas(base_notas: pd.DataFrame, titulo: str):
    # texto: somente a nota, sempre acima
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


def grafico_participacao_insuficiente(base_part: pd.DataFrame, base_insuf: pd.DataFrame, titulo: str):
    a = base_part[["Avaliação", "Valor"]].copy();  a["Métrica"] = "Participação (%)"
    b = base_insuf[["Avaliação", "Valor"]].copy(); b["Métrica"] = "Texto insuficiente (%)"
    df_plot = pd.concat([a, b], ignore_index=True)

    df_plot["hover_text"] = (
        "<b>" + df_plot["Avaliação"].astype(str) + "</b>"
        + "<br>" + df_plot["Métrica"] + ": "
        + df_plot["Valor"].map(lambda v: f"{v:.2f}") + "%"
    )

    fig = px.line(df_plot, x="Avaliação", y="Valor", color="Métrica", markers=True, title=titulo)
    fig.update_traces(
        marker=dict(size=MARKER_SIZE),
        hovertext=df_plot["hover_text"],
        hovertemplate="%{hovertext}<extra></extra>",
        line=dict(width=3),
    )
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
# ---------- Gráficos ----------------


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

# *** MANTÉM APENAS AS OPÇÕES ORIGINAIS NO MENU ***
# Filtra fora nomes que contaminem o menu (participação / texto insuficiente)
tab_names = [
    n for n in abas.keys()
    if ("particip" not in n.lower()) and not ("texto" in n.lower() and "insuf" in n.lower())
]

col_nav, col_main = st.columns([3, 2], gap="small")

with col_nav:
    st.markdown("")
    aba_sel = st.radio(
        label="Abas",
        options=tab_names,
        index=0,
        width=320,
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
    st.plotly_chart(grafico_notas(base_notas, "Evolução das Notas"), use_container_width=True)

    # ----- PARTICIPAÇÃO e TEXTO INSUFICIENTE (busca abas companheiras sem mostrar no menu) -----
    aba_part, aba_insu = achar_companheiras(abas, aba_sel)

    if not aba_part or not aba_insu:
        st.info(
            "Para o 2º gráfico, procuro abas companheiras que comecem com "
            f"`{aba_sel}` e contenham “participa” e “texto insuf…”.\n"
            "Ex.: `2º Ano - Regular - Participação` e `2º Ano - Regular - Texto Insuficiente`.\n"
            "Se os nomes estiverem diferentes, me diga que eu ajusto o detector."
        )
    else:
        df_long_part = preparar_df(abas[aba_part])
        df_long_insu = preparar_df(abas[aba_insu])
        base_part = montar_base(df_long_part, regional)
        base_insu = montar_base(df_long_insu, regional)
        st.plotly_chart(
            grafico_participacao_insuficiente(
                base_part, base_insu, "Participação e Textos Insuficientes"
            ),
            use_container_width=True
        )




