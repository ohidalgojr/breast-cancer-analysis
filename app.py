from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

st.set_page_config(
    page_title="Trabalho Final de Estatística | Câncer de Mama no SUS (2024)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
DATA_POP_PATH = ROOT / "data" / "processed" / "populacao_mama_2024.csv"

HAS_WIDTH_PARAM_PLOTLY = "width" in inspect.signature(st.plotly_chart).parameters
HAS_WIDTH_PARAM_DF = "width" in inspect.signature(st.dataframe).parameters

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

PALETTE = {
    "primary": "#0f172a",
    "pop": "#1d4ed8",
    "pop_light": "#93c5fd",
    "sample": "#b45309",
    "sample_light": "#fcd34d",
    "alert": "#b91c1c",
    "ok": "#15803d",
    "grid": "#e2e8f0",
    "bg_plot": "#ffffff",
}


def fmt_br(val: float, decimals: int = 2) -> str:
    s = f"{float(val):,.{decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_int(val: int | float) -> str:
    return f"{int(val):,}".replace(",", ".")


def render_chart(fig: go.Figure, key: str) -> None:
    if HAS_WIDTH_PARAM_PLOTLY:
        st.plotly_chart(fig, width="stretch", theme=None, key=key, config=PLOTLY_CONFIG)
    else:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key, config=PLOTLY_CONFIG)


def render_table(df: pd.DataFrame, key: str | None = None) -> None:
    if HAS_WIDTH_PARAM_DF:
        st.dataframe(df, width="stretch", hide_index=True, key=key)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, key=key)


def apply_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.6rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }
        h1, h2, h3, h4 {
            color: #0f172a;
            font-weight: 600;
        }
        .main-header {
            border-bottom: 2px solid #0f172a;
            padding-bottom: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .main-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
            line-height: 1.3;
        }
        .main-subtitle {
            font-size: 0.94rem;
            color: #475569;
            margin-top: 0.25rem;
            margin-bottom: 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #cbd5e1;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.55rem 1rem;
            font-weight: 600;
            color: #334155;
        }
        .stTabs [aria-selected="true"] {
            color: #0f172a;
            border-bottom: 2px solid #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_and_prepare_data() -> dict:
    dtypes = {
        "idade": "int16",
        "faixa_etaria": "string",
        "uf_residencia": "string",
        "regiao_residencia": "string",
        "estadiamento": "string",
        "primeiro_tratamento": "string",
        "tempo_tratamento_dias": "float32",
        "tratado_ate_60_dias": "bool",
    }
    df_pop = pd.read_csv(DATA_POP_PATH, usecols=list(dtypes.keys()), dtype=dtypes)
    df_pop["faixa_rastreamento_50_69"] = df_pop["idade"].between(50, 69)

    n_pop = len(df_pop)
    z_val = 1.96
    e_val = 0.05
    p_est = 0.50
    n0 = (z_val**2 * p_est * (1.0 - p_est)) / (e_val**2)
    n_exact = (n_pop * n0) / (n_pop + n0 - 1.0)
    n_sample = int(np.ceil(n_exact))

    df_sample = df_pop.sample(n=n_sample, random_state=42).reset_index(drop=True)

    return {
        "df_pop": df_pop,
        "df_sample": df_sample,
        "N": n_pop,
        "n": n_sample,
        "n0": n0,
        "n_exact": n_exact,
    }


def compute_quant_summary(series: pd.Series, is_population: bool) -> dict:
    clean = series.dropna().astype(float)
    ddof = 0 if is_population else 1
    mean_val = float(clean.mean())
    median_val = float(clean.median())
    mode_val = float(clean.mode().iloc[0])
    var_val = float(clean.var(ddof=ddof))
    std_val = float(clean.std(ddof=ddof))
    cv_val = (std_val / mean_val) * 100.0
    min_val = float(clean.min())
    q1_val = float(clean.quantile(0.25))
    q3_val = float(clean.quantile(0.75))
    max_val = float(clean.max())
    iqr_val = q3_val - q1_val
    lower_limit = max(0.0, q1_val - 1.5 * iqr_val) if min_val >= 0 else q1_val - 1.5 * iqr_val
    upper_limit = q3_val + 1.5 * iqr_val
    non_outliers = clean[(clean >= (q1_val - 1.5 * iqr_val)) & (clean <= upper_limit)]
    lower_fence = float(non_outliers.min())
    upper_fence = float(non_outliers.max())
    outliers_low = clean[clean < (q1_val - 1.5 * iqr_val)]
    outliers_high = clean[clean > upper_limit]
    return {
        "n": len(clean),
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "var": var_val,
        "std": std_val,
        "cv": cv_val,
        "min": min_val,
        "q1": q1_val,
        "q3": q3_val,
        "max": max_val,
        "iqr": iqr_val,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "n_outliers": int(len(outliers_low) + len(outliers_high)),
        "outliers_values": pd.concat([outliers_low, outliers_high]),
        "skew": float(clean.skew()),
    }


def build_histogram_figure(
    series: pd.Series,
    stats_dict: dict,
    title: str,
    x_label: str,
    unit: str,
    bar_color: str,
    is_population: bool,
    nbins: int = 34,
    ref_line_value: float | None = None,
    ref_line_label: str | None = None,
    show_normal_fit: bool = False,
) -> go.Figure:
    clean = series.dropna().astype(float).to_numpy()
    min_x = float(np.floor(clean.min()))
    max_x = float(np.ceil(clean.max()))
    counts, bin_edges = np.histogram(clean, bins=nbins, range=(min_x, max_x))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    pcts = (counts / len(clean)) * 100.0

    custom_hover = np.column_stack([bin_edges[:-1], bin_edges[1:], counts, pcts])
    mean_sym = "μ" if is_population else "x̄"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            width=bin_widths * 0.90,
            name="Frequência observada",
            marker=dict(color=bar_color, line=dict(color="#0f172a", width=0.5)),
            customdata=custom_hover,
            hovertemplate=(
                f"Faixa: %{{customdata[0]:.1f}} a %{{customdata[1]:.1f}} {unit}<br>"
                "Pacientes: %{customdata[2]:.0f}<br>"
                "Proporção: %{customdata[3]:.2f}%<extra></extra>"
            ),
        )
    )

    if show_normal_fit and stats_dict["std"] > 0:
        x_grid = np.linspace(min_x, max_x, 180)
        bin_w = float(bin_widths[0])
        y_norm = stats.norm.pdf(x_grid, loc=stats_dict["mean"], scale=stats_dict["std"]) * len(clean) * bin_w
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_norm,
                mode="lines",
                name="Curva Normal Teórica",
                line=dict(color="#0f172a", width=2.0, dash="dot"),
                hovertemplate=f"{x_label}: %{{x:.1f}} {unit}<br>Frequência teórica: %{{y:.1f}}<extra></extra>",
            )
        )

    max_y = float(counts.max()) * 1.18

    fig.add_trace(
        go.Scatter(
            x=[stats_dict["mean"], stats_dict["mean"]],
            y=[0, max_y * 0.95],
            mode="lines",
            name=f"Média ({mean_sym} = {fmt_br(stats_dict['mean'], 1)} {unit})",
            line=dict(color=PALETTE["alert"], width=2.2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[stats_dict["median"], stats_dict["median"]],
            y=[0, max_y * 0.95],
            mode="lines",
            name=f"Mediana ({fmt_br(stats_dict['median'], 1)} {unit})",
            line=dict(color=PALETTE["ok"], width=2.2, dash="solid"),
        )
    )

    if ref_line_value is not None and ref_line_label is not None:
        fig.add_trace(
            go.Scatter(
                x=[ref_line_value, ref_line_value],
                y=[0, max_y * 0.95],
                mode="lines",
                name=f"{ref_line_label} ({fmt_int(ref_line_value)} {unit})",
                line=dict(color="#6d28d9", width=2.0, dash="dashdot"),
            )
        )

    x_pad = (max_x - min_x) * 0.03
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.96, yanchor="top", font=dict(size=14.5, color="#0f172a")),
        xaxis=dict(
            title=f"{x_label} ({unit})",
            range=[max(0.0, min_x - x_pad) if min_x >= 0 else min_x - x_pad, max_x + x_pad],
            gridcolor=PALETTE["grid"],
            automargin=True,
        ),
        yaxis=dict(
            title="Nº de pacientes",
            rangemode="tozero",
            range=[0, max_y],
            gridcolor=PALETTE["grid"],
            automargin=True,
        ),
        plot_bgcolor=PALETTE["bg_plot"],
        paper_bgcolor=PALETTE["bg_plot"],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=50, b=95),
        height=420,
        bargap=0.05,
    )
    return fig


def build_boxplot_figure(
    stats_dict: dict,
    title: str,
    x_label: str,
    unit: str,
    box_color: str,
    is_population: bool,
) -> go.Figure:
    label_y = "População" if is_population else "Amostra"
    mean_sym = "μ" if is_population else "x̄"

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=[label_y],
            q1=[stats_dict["q1"]],
            median=[stats_dict["median"]],
            q3=[stats_dict["q3"]],
            lowerfence=[stats_dict["lower_fence"]],
            upperfence=[stats_dict["upper_fence"]],
            mean=[stats_dict["mean"]],
            sd=[stats_dict["std"]],
            orientation="h",
            name=f"Quartis (Q1={fmt_br(stats_dict['q1'], 1)}, Md={fmt_br(stats_dict['median'], 1)}, Q3={fmt_br(stats_dict['q3'], 1)})",
            marker=dict(color=box_color),
            fillcolor=box_color,
            line=dict(color="#0f172a", width=1.5),
            boxmean=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[stats_dict["mean"]],
            y=[label_y],
            mode="markers",
            name=f"Média ({mean_sym} = {fmt_br(stats_dict['mean'], 2)} {unit})",
            marker=dict(color=PALETTE["alert"], size=10, symbol="diamond", line=dict(color="#ffffff", width=1)),
            hovertemplate=f"Média ({mean_sym}): {fmt_br(stats_dict['mean'], 2)} {unit}<extra></extra>",
        )
    )

    outliers_ser = stats_dict["outliers_values"]
    if len(outliers_ser) > 0:
        vc = outliers_ser.value_counts().sort_index()
        out_x = vc.index.to_numpy(dtype=float)
        out_counts = vc.to_numpy(dtype=int)
        fig.add_trace(
            go.Scatter(
                x=out_x,
                y=[label_y] * len(out_x),
                mode="markers",
                name=f"Outliers ({fmt_int(stats_dict['n_outliers'])} casos)",
                marker=dict(color="#dc2626", size=6.5, symbol="circle-open", line=dict(width=1.4, color="#dc2626")),
                customdata=out_counts,
                hovertemplate=(
                    f"Valor: %{{x:.1f}} {unit}<br>"
                    "Pacientes neste valor: %{customdata}<br>"
                    f"Haste superior: {fmt_br(stats_dict['upper_limit'], 1)} {unit}<extra></extra>"
                ),
            )
        )

    span = stats_dict["max"] - stats_dict["min"]
    pad = max(span * 0.05, 3.0)
    min_range = max(0.0, stats_dict["min"] - pad) if stats_dict["min"] >= 0 else stats_dict["min"] - pad
    max_range = stats_dict["max"] + pad

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.95, yanchor="top", font=dict(size=14.5, color="#0f172a")),
        xaxis=dict(
            title=f"{x_label} ({unit})",
            range=[min_range, max_range],
            gridcolor=PALETTE["grid"],
            automargin=True,
        ),
        yaxis=dict(title="", automargin=True),
        plot_bgcolor=PALETTE["bg_plot"],
        paper_bgcolor=PALETTE["bg_plot"],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=80, r=30, t=45, b=85),
        height=290,
    )
    return fig


def build_categorical_bar_figure(
    series: pd.Series,
    title: str,
    category_label: str,
    bar_color: str,
    is_population: bool,
    custom_labels: dict | None = None,
) -> go.Figure:
    s = series.dropna().astype(str)
    if custom_labels:
        s = s.map(lambda x: custom_labels.get(x, x))

    vc = s.value_counts(ascending=False)
    total = int(vc.sum())
    cats = vc.index.tolist()
    counts = vc.values.astype(int).tolist()
    pcts = [(c / total) * 100.0 for c in counts]
    text_labels = [f"{fmt_int(c)} ({fmt_br(p, 1)}%)" for c, p in zip(counts, pcts)]
    trace_name = "População (N)" if is_population else "Amostra (n)"
    max_count = max(counts) if counts else 1

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cats,
            y=counts,
            name=f"Frequência e proporção ({trace_name})",
            marker=dict(color=bar_color, line=dict(color="#0f172a", width=0.5)),
            text=text_labels,
            textposition="outside",
            cliponaxis=False,
            customdata=pcts,
            hovertemplate=(
                f"{category_label}: %{{x}}<br>"
                "Frequência absoluta: %{y:,.0f} pacientes<br>"
                "Proporção: %{customdata:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.96, yanchor="top", font=dict(size=14.5, color="#0f172a")),
        xaxis=dict(title=category_label, automargin=True),
        yaxis=dict(
            title="Nº de pacientes",
            rangemode="tozero",
            range=[0, max_count * 1.22],
            gridcolor=PALETTE["grid"],
            automargin=True,
        ),
        plot_bgcolor=PALETTE["bg_plot"],
        paper_bgcolor=PALETTE["bg_plot"],
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=60, r=30, t=50, b=85),
        height=390,
    )
    return fig


def build_comparison_bar_figure(
    pop_series: pd.Series,
    sam_series: pd.Series,
    title: str,
    category_label: str,
    custom_labels: dict | None = None,
) -> go.Figure:
    s_pop = pop_series.dropna().astype(str)
    s_sam = sam_series.dropna().astype(str)
    if custom_labels:
        s_pop = s_pop.map(lambda x: custom_labels.get(x, x))
        s_sam = s_sam.map(lambda x: custom_labels.get(x, x))

    pop_pct = (s_pop.value_counts(normalize=True) * 100.0).sort_values(ascending=False)
    cats = pop_pct.index.tolist()
    sam_pct_map = (s_sam.value_counts(normalize=True) * 100.0).to_dict()

    pop_vals = [float(pop_pct[c]) for c in cats]
    sam_vals = [float(sam_pct_map.get(c, 0.0)) for c in cats]
    max_val = max(max(pop_vals, default=1.0), max(sam_vals, default=1.0))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cats,
            y=pop_vals,
            name="População (p)",
            marker=dict(color=PALETTE["pop"]),
            text=[f"{fmt_br(v, 1)}%" for v in pop_vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"{category_label}: %{{x}}<br>População (p): %{{y:.2f}}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=cats,
            y=sam_vals,
            name="Amostra (p̂)",
            marker=dict(color=PALETTE["sample"]),
            text=[f"{fmt_br(v, 1)}%" for v in sam_vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"{category_label}: %{{x}}<br>Amostra (p̂): %{{y:.2f}}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.96, yanchor="top", font=dict(size=14.5, color="#0f172a")),
        barmode="group",
        xaxis=dict(title=category_label, automargin=True),
        yaxis=dict(
            title="Proporção (%)",
            rangemode="tozero",
            range=[0, max_val * 1.24],
            gridcolor=PALETTE["grid"],
            automargin=True,
        ),
        plot_bgcolor=PALETTE["bg_plot"],
        paper_bgcolor=PALETTE["bg_plot"],
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=60, r=30, t=50, b=85),
        height=390,
    )
    return fig


def build_ci_comparison_figure(
    param_pop: float,
    est_sample: float,
    ci_low: float,
    ci_high: float,
    null_val: float,
    title: str,
    x_label: str,
    unit: str,
    is_proportion: bool = False,
) -> go.Figure:
    scale = 100.0 if is_proportion else 1.0
    p_pop = param_pop * scale
    p_est = est_sample * scale
    c_low = ci_low * scale
    c_high = ci_high * scale
    h0_v = null_val * scale

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[c_low, c_high],
            y=["IC 95%"],
            mode="lines+markers",
            name=f"IC 95% ([{fmt_br(c_low, 2)}{unit}; {fmt_br(c_high, 2)}{unit}])",
            line=dict(color=PALETTE["sample"], width=5),
            marker=dict(size=10, symbol="line-ns-open", line=dict(width=3, color=PALETTE["sample"])),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p_est],
            y=["IC 95%"],
            mode="markers",
            name=f"Estimativa Amostral ({fmt_br(p_est, 2)}{unit})",
            marker=dict(color=PALETTE["sample"], size=12, symbol="circle", line=dict(color="#ffffff", width=1.5)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p_pop],
            y=["IC 95%"],
            mode="markers",
            name=f"Parâmetro Populacional ({fmt_br(p_pop, 2)}{unit})",
            marker=dict(color=PALETTE["pop"], size=14, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[h0_v],
            y=["IC 95%"],
            mode="markers",
            name=f"Referência H0 ({fmt_br(h0_v, 1)}{unit})",
            marker=dict(color=PALETTE["alert"], size=12, symbol="x", line=dict(color=PALETTE["alert"], width=2)),
        )
    )

    all_vals = [c_low, c_high, p_pop, p_est, h0_v]
    min_x, max_x = min(all_vals), max(all_vals)
    span = max(max_x - min_x, 2.0)
    pad = span * 0.22

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", y=0.95, yanchor="top", font=dict(size=14, color="#0f172a")),
        xaxis=dict(
            title=f"{x_label} ({unit.strip()})",
            range=[min_x - pad, max_x + pad],
            gridcolor=PALETTE["grid"],
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(title="", automargin=True),
        plot_bgcolor=PALETTE["bg_plot"],
        paper_bgcolor=PALETTE["bg_plot"],
        legend=dict(orientation="h", yanchor="top", y=-0.32, xanchor="center", x=0.5, font=dict(size=10.5)),
        margin=dict(l=70, r=30, t=45, b=90),
        height=275,
    )
    return fig


def main() -> None:
    apply_css()

    st.markdown(
        """
        <div class="main-header">
            <p class="main-title">Análise Estatística e Inferencial: Câncer de Mama no SUS (PAINEL-Oncologia 2024)</p>
            <p class="main-subtitle">
                Comparação entre parâmetros populacionais (N = 48.866) e estimativas de amostra aleatória simples (n = 382) no monitoramento da Lei nº 12.732/2012 (Lei dos 60 dias).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Carregando base de dados do DATASUS e calculando estatísticas da população e da amostra..."):
        data_bundle = load_and_prepare_data()
        df_pop = data_bundle["df_pop"]
        df_sample = data_bundle["df_sample"]
        n_pop = data_bundle["N"]
        n_sample = data_bundle["n"]

        pop_idade = compute_quant_summary(df_pop["idade"], is_population=True)
        pop_tempo = compute_quant_summary(df_pop["tempo_tratamento_dias"], is_population=True)
        sam_idade = compute_quant_summary(df_sample["idade"], is_population=False)
        sam_tempo = compute_quant_summary(df_sample["tempo_tratamento_dias"], is_population=False)

        p_pop_60 = float(df_pop["tratado_ate_60_dias"].mean())
        p_sam_60 = float(df_sample["tratado_ate_60_dias"].mean())
        p_pop_rast = float(df_pop["faixa_rastreamento_50_69"].mean())
        p_sam_rast = float(df_sample["faixa_rastreamento_50_69"].mean())

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. Base de Dados e Amostragem",
            "2. Painel 1: População (N = 48.866)",
            "3. Painel 2: Amostra (n = 382)",
            "4. Painel 3: Inferência e Confronto",
            "5. Referências e Extração",
        ]
    )

    # =========================================================================
    # ABA 1: BASE DE DADOS E AMOSTRAGEM (CONCISA E DIRETA)
    # =========================================================================
    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("População de Interesse (N)", fmt_int(n_pop), "Mulheres (CID C50) em 2024")
        m2.metric("Tamanho da Amostra (n)", fmt_int(n_sample), "Fração amostral: 0,78%")
        m3.metric("Nível de Confiança e Erro", "95% | e = 5%", "Proporção p = 0,50")
        m4.metric("Sorteio Amostral (AAS)", "random_state = 42", "Sem reposição")

        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                st.markdown("#### 1. Escolha da Base e População de Interesse")
                st.markdown(
                    """
                    - **Fonte e Link Oficial:** Ministério da Saúde / DATASUS, sistema **PAINEL-Oncologia** ([Portal de Transferência de Arquivos DATASUS](https://datasus.saude.gov.br/transferencia-de-arquivos/)).
                    - **Contexto e Objetivo:** Monitorar o cumprimento da **Lei nº 12.732/2012 (Lei dos 60 dias)**, que estabelece o prazo máximo de 60 dias após o laudo diagnóstico para o início do primeiro tratamento oncológico no SUS.
                    - **Número de Observações e Variáveis:** **48.866 observações** na população de interesse (recorte dos 64.776 registros de CID C50 em 2024) e **7 variáveis selecionadas** para análise (a partir das 17 colunas da base tratada).
                    - **População de Interesse (N = 48.866):** Mulheres adultas (18 a 115 anos) diagnosticadas com câncer de mama maligno (CID C50) em 2024 no Brasil, com primeiro tratamento registrado no SUS a partir da data do laudo.
                    - **Unidade Amostral:** 1 paciente diagnosticada com câncer de mama no SUS em 2024.
                    """
                )

        with col_b:
            with st.container(border=True):
                st.markdown("#### 2. Dimensionamento e Seleção da Amostra (AAS)")
                st.markdown(
                    """
                    O tamanho amostral foi calculado conforme a calculadora de tamanho amostral da **SurveyMonkey** (fórmula para população finita):
                    - **Tamanho da população (N):** 48.866 pacientes.
                    - **Nível de confiança:** 95% (escore crítico Z = 1,96).
                    - **Margem de erro (e):** 5% (0,05).
                    - **Proporção estimada (p):** 50% (0,50, ponto de variância máxima que gera a amostra mais conservadora).
                    """
                )
                st.latex(
                    r"n_0 = \frac{1,96^2 \cdot 0,50 \cdot 0,50}{0,05^2} = 384,16 \quad \Longrightarrow \quad n = \frac{48.866 \cdot 384,16}{48.866 + 384,16 - 1} = 381,17 \approx \mathbf{382}"
                )
                st.markdown(
                    "**Seleção da Amostra Aleatória Simples:** Sorteio equiprovável sem reposição via `df_pop.sample(n=382, random_state=42)`, garantindo que todas as pacientes tenham a mesma probabilidade de seleção."
                )

        st.markdown("#### 3. Variáveis Selecionadas para a Análise")
        df_vars = pd.DataFrame(
            [
                {
                    "Variável": "idade",
                    "Tipo Estatístico": "Quantitativa Discreta",
                    "Descrição": "Idade da paciente no diagnóstico em anos completos (18 a 103 anos).",
                    "Aplicação no Trabalho": "Painéis 1 e 2 (descritiva) e Painel 3 (Hipótese de Média M1).",
                },
                {
                    "Variável": "tempo_tratamento_dias",
                    "Tipo Estatístico": "Quantitativa Contínua",
                    "Descrição": "Dias corridos entre o laudo diagnóstico e o 1º tratamento no SUS (0 a 887 dias).",
                    "Aplicação no Trabalho": "Painéis 1 e 2 (descritiva) e Painel 3 (Hipótese de Média M2).",
                },
                {
                    "Variável": "tratado_ate_60_dias",
                    "Tipo Estatístico": "Qualitativa Nominal (Binária)",
                    "Descrição": "Cumprimento da Lei nº 12.732/2012: Até 60 dias vs. Acima de 60 dias.",
                    "Aplicação no Trabalho": "Painéis 1 e 2 (proporções) e Painel 3 (Hipótese de Proporção P1).",
                },
                {
                    "Variável": "faixa_rastreamento_50_69",
                    "Tipo Estatístico": "Qualitativa Nominal (Binária)",
                    "Descrição": "Indica se a paciente está na faixa-alvo de mamografia do INCA (50 a 69 anos).",
                    "Aplicação no Trabalho": "Painéis 1 e 2 (proporções) e Painel 3 (Hipótese de Proporção P2).",
                },
                {
                    "Variável": "primeiro_tratamento",
                    "Tipo Estatístico": "Qualitativa Nominal",
                    "Descrição": "Modalidade terapêutica inicial (Quimioterapia, Radioterapia, Cirurgia, Quimio + Radio).",
                    "Aplicação no Trabalho": "Painéis 1 e 2 (frequências e proporções).",
                },
                {
                    "Variável": "regiao_residencia",
                    "Tipo Estatístico": "Qualitativa Nominal",
                    "Descrição": "Macrorregião brasileira de residência da paciente (Sudeste, Nordeste, Sul, Centro-Oeste, Norte).",
                    "Aplicação no Trabalho": "Painéis 1 e 2 (frequências e proporções).",
                },
                {
                    "Variável": "faixa_etaria",
                    "Tipo Estatístico": "Qualitativa Ordinal",
                    "Descrição": "Faixa etária agrupada (<30, 30-39, 40-49, 50-59, 60-69, 70-79, 80+).",
                    "Aplicação no Trabalho": "Painéis 1 e 2 (frequências e proporções).",
                },
            ]
        )
        render_table(df_vars, key="tbl_vars_tab1")

    # =========================================================================
    # ABA 2: PAINEL 1 - ANÁLISE DESCRITIVA DA POPULAÇÃO
    # =========================================================================
    with tab2:
        st.subheader("Painel 1: Análise Descritiva da População (N = 48.866)")

        kp1, kp2, kp3, kp4 = st.columns(4)
        kp1.metric(
            "Idade Média Populacional (μ)",
            f"{fmt_br(pop_idade['mean'])} anos",
            f"Mediana: {fmt_br(pop_idade['median'], 1)} | σ = {fmt_br(pop_idade['std'])} anos",
            delta_color="off",
        )
        kp2.metric(
            "Tempo Médio até Tratamento (μ)",
            f"{fmt_br(pop_tempo['mean'])} dias",
            f"Mediana: {fmt_br(pop_tempo['median'], 1)} | σ = {fmt_br(pop_tempo['std'])} dias",
            delta_color="off",
        )
        kp3.metric(
            "Tratamento em Até 60 Dias (p)",
            f"{fmt_br(p_pop_60 * 100)}%",
            f"{fmt_int(df_pop['tratado_ate_60_dias'].sum())} de {fmt_int(n_pop)} pacientes",
            delta_color="off",
        )
        kp4.metric(
            "Faixa de Rastreamento 50-69 Anos (p)",
            f"{fmt_br(p_pop_rast * 100)}%",
            f"{fmt_int(df_pop['faixa_rastreamento_50_69'].sum())} de {fmt_int(n_pop)} pacientes",
            delta_color="off",
        )

        st.markdown("#### 1. Medidas Descritivas das Variáveis Quantitativas (População)")
        df_quant_pop = pd.DataFrame(
            [
                {
                    "Variável": "Idade no diagnóstico (anos)",
                    "N": fmt_int(pop_idade["n"]),
                    "Média (μ)": fmt_br(pop_idade["mean"]),
                    "Mediana (Md)": fmt_br(pop_idade["median"]),
                    "Moda": fmt_int(pop_idade["mode"]),
                    "Desvio-Padrão (σ)": fmt_br(pop_idade["std"]),
                    "CV (%)": f"{fmt_br(pop_idade['cv'])}%",
                    "Mínimo": fmt_int(pop_idade["min"]),
                    "Q1 (25%)": fmt_br(pop_idade["q1"], 1),
                    "Q3 (75%)": fmt_br(pop_idade["q3"], 1),
                    "Máximo": fmt_int(pop_idade["max"]),
                    "IQR": fmt_br(pop_idade["iqr"], 1),
                    "Outliers": fmt_int(pop_idade["n_outliers"]),
                },
                {
                    "Variável": "Tempo até 1º tratamento (dias)",
                    "N": fmt_int(pop_tempo["n"]),
                    "Média (μ)": fmt_br(pop_tempo["mean"]),
                    "Mediana (Md)": fmt_br(pop_tempo["median"]),
                    "Moda": fmt_int(pop_tempo["mode"]),
                    "Desvio-Padrão (σ)": fmt_br(pop_tempo["std"]),
                    "CV (%)": f"{fmt_br(pop_tempo['cv'])}%",
                    "Mínimo": fmt_int(pop_tempo["min"]),
                    "Q1 (25%)": fmt_br(pop_tempo["q1"], 1),
                    "Q3 (75%)": fmt_br(pop_tempo["q3"], 1),
                    "Máximo": fmt_int(pop_tempo["max"]),
                    "IQR": fmt_br(pop_tempo["iqr"], 1),
                    "Outliers": fmt_int(pop_tempo["n_outliers"]),
                },
            ]
        )
        render_table(df_quant_pop, key="tbl_quant_pop")

        st.markdown("#### 2. Distribuição das Variáveis Quantitativas (População)")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            render_chart(
                build_histogram_figure(
                    df_pop["idade"],
                    pop_idade,
                    title="Histograma da Idade no Diagnóstico (População)",
                    x_label="Idade",
                    unit="anos",
                    bar_color=PALETTE["pop_light"],
                    is_population=True,
                    nbins=34,
                    ref_line_value=50.0,
                    ref_line_label="Início Rastreio",
                    show_normal_fit=True,
                ),
                key="hist_idade_pop",
            )
            render_chart(
                build_boxplot_figure(
                    pop_idade,
                    title="Boxplot da Idade (18 a 103 Anos)",
                    x_label="Idade",
                    unit="anos",
                    box_color=PALETTE["pop_light"],
                    is_population=True,
                ),
                key="box_idade_pop",
            )

        with col_q2:
            render_chart(
                build_histogram_figure(
                    df_pop["tempo_tratamento_dias"],
                    pop_tempo,
                    title="Histograma do Tempo até o Tratamento (0 a 887 Dias)",
                    x_label="Tempo até o 1º tratamento",
                    unit="dias",
                    bar_color="#60a5fa",
                    is_population=True,
                    nbins=45,
                    ref_line_value=60.0,
                    ref_line_label="Prazo Legal",
                    show_normal_fit=False,
                ),
                key="hist_tempo_pop",
            )
            render_chart(
                build_boxplot_figure(
                    pop_tempo,
                    title="Boxplot do Tempo até o Tratamento (0 a 887 Dias)",
                    x_label="Tempo até o 1º tratamento",
                    unit="dias",
                    box_color="#60a5fa",
                    is_population=True,
                ),
                key="box_tempo_pop",
            )

        with st.container(border=True):
            st.markdown(
                f"**Resumo das variáveis quantitativas:** A idade tem distribuição quase simétrica, com média (μ = {fmt_br(pop_idade['mean'])} anos) próxima da mediana ({fmt_br(pop_idade['median'], 1)} anos) e 39 outliers superiores entre 95 e 103 anos. O tempo até o tratamento tem forte assimetria positiva à direita (CV = {fmt_br(pop_tempo['cv'])}%): os {fmt_int(pop_tempo['n_outliers'])} casos acima de 300,5 dias (chegando a 887 dias) puxam a média para {fmt_br(pop_tempo['mean'])} dias, enquanto a mediana fica em {fmt_br(pop_tempo['median'], 1)} dias."
            )

        st.markdown("#### 3. Frequências e Proporções das Variáveis Qualitativas (População)")
        c_cat1, c_cat2 = st.columns(2)
        with c_cat1:
            render_chart(
                build_categorical_bar_figure(
                    df_pop["tratado_ate_60_dias"],
                    title="Cumprimento da Lei dos 60 Dias (População)",
                    category_label="Prazo até o 1º tratamento",
                    bar_color=PALETTE["pop"],
                    is_population=True,
                    custom_labels={
                        "False": "Acima de 60 dias (Fora do prazo)",
                        "True": "Até 60 dias (Dentro do prazo)",
                    },
                ),
                key="bar_lei_pop",
            )
            render_chart(
                build_categorical_bar_figure(
                    df_pop["primeiro_tratamento"],
                    title="Modalidade do Primeiro Tratamento (População)",
                    category_label="Primeiro tratamento registrado",
                    bar_color=PALETTE["pop"],
                    is_population=True,
                ),
                key="bar_trat_pop",
            )

        with c_cat2:
            render_chart(
                build_categorical_bar_figure(
                    df_pop["regiao_residencia"],
                    title="Distribuição por Região de Residência (População)",
                    category_label="Região geográfica",
                    bar_color=PALETTE["pop"],
                    is_population=True,
                ),
                key="bar_reg_pop",
            )
            render_chart(
                build_categorical_bar_figure(
                    df_pop["faixa_etaria"],
                    title="Distribuição por Faixa Etária (População)",
                    category_label="Faixa etária (anos)",
                    bar_color=PALETTE["pop"],
                    is_population=True,
                ),
                key="bar_fx_pop",
            )

    # =========================================================================
    # ABA 3: PAINEL 2 - ANÁLISE DESCRITIVA DA AMOSTRA
    # =========================================================================
    with tab3:
        st.subheader("Painel 2: Análise Descritiva da Amostra (n = 382)")

        ks1, ks2, ks3, ks4 = st.columns(4)
        ks1.metric(
            "Idade Média Amostral (x̄)",
            f"{fmt_br(sam_idade['mean'])} anos",
            f"Dif. p/ μ: {fmt_br(sam_idade['mean'] - pop_idade['mean'], 2)} anos | s = {fmt_br(sam_idade['std'])}",
            delta_color="off",
        )
        ks2.metric(
            "Tempo Médio Amostral (x̄)",
            f"{fmt_br(sam_tempo['mean'])} dias",
            f"Dif. p/ μ: +{fmt_br(sam_tempo['mean'] - pop_tempo['mean'], 2)} dias | s = {fmt_br(sam_tempo['std'])}",
            delta_color="off",
        )
        ks3.metric(
            "Tratamento em Até 60 Dias (p̂)",
            f"{fmt_br(p_sam_60 * 100)}%",
            f"126 de 382 | Dif. p/ p: {fmt_br((p_sam_60 - p_pop_60)*100)} p.p.",
            delta_color="off",
        )
        ks4.metric(
            "Faixa 50-69 Anos na Amostra (p̂)",
            f"{fmt_br(p_sam_rast * 100)}%",
            f"185 de 382 | Dif. p/ p: {fmt_br((p_sam_rast - p_pop_rast)*100)} p.p.",
            delta_color="off",
        )

        st.markdown("#### 1. Medidas Descritivas das Variáveis Quantitativas (Amostra)")
        df_quant_sam = pd.DataFrame(
            [
                {
                    "Variável": "Idade no diagnóstico (anos)",
                    "n": fmt_int(sam_idade["n"]),
                    "Média (x̄)": fmt_br(sam_idade["mean"]),
                    "Mediana (Md)": fmt_br(sam_idade["median"]),
                    "Moda": fmt_int(sam_idade["mode"]),
                    "Desvio-Padrão (s)": fmt_br(sam_idade["std"]),
                    "CV (%)": f"{fmt_br(sam_idade['cv'])}%",
                    "Mínimo": fmt_int(sam_idade["min"]),
                    "Q1 (25%)": fmt_br(sam_idade["q1"], 1),
                    "Q3 (75%)": fmt_br(sam_idade["q3"], 1),
                    "Máximo": fmt_int(sam_idade["max"]),
                    "IQR": fmt_br(sam_idade["iqr"], 1),
                    "Outliers": fmt_int(sam_idade["n_outliers"]),
                },
                {
                    "Variável": "Tempo até 1º tratamento (dias)",
                    "n": fmt_int(sam_tempo["n"]),
                    "Média (x̄)": fmt_br(sam_tempo["mean"]),
                    "Mediana (Md)": fmt_br(sam_tempo["median"]),
                    "Moda": fmt_int(sam_tempo["mode"]),
                    "Desvio-Padrão (s)": fmt_br(sam_tempo["std"]),
                    "CV (%)": f"{fmt_br(sam_tempo['cv'])}%",
                    "Mínimo": fmt_int(sam_tempo["min"]),
                    "Q1 (25%)": fmt_br(sam_tempo["q1"], 1),
                    "Q3 (75%)": fmt_br(sam_tempo["q3"], 1),
                    "Máximo": fmt_int(sam_tempo["max"]),
                    "IQR": fmt_br(sam_tempo["iqr"], 1),
                    "Outliers": fmt_int(sam_tempo["n_outliers"]),
                },
            ]
        )
        render_table(df_quant_sam, key="tbl_quant_sam")

        st.markdown("#### 2. Distribuição das Variáveis Quantitativas (Amostra)")
        col_sq1, col_sq2 = st.columns(2)
        with col_sq1:
            render_chart(
                build_histogram_figure(
                    df_sample["idade"],
                    sam_idade,
                    title="Histograma da Idade na Amostra (n = 382)",
                    x_label="Idade",
                    unit="anos",
                    bar_color=PALETTE["sample_light"],
                    is_population=False,
                    nbins=22,
                    ref_line_value=50.0,
                    ref_line_label="Início Rastreio",
                    show_normal_fit=True,
                ),
                key="hist_idade_sam",
            )
            render_chart(
                build_boxplot_figure(
                    sam_idade,
                    title="Boxplot da Idade na Amostra",
                    x_label="Idade",
                    unit="anos",
                    box_color=PALETTE["sample_light"],
                    is_population=False,
                ),
                key="box_idade_sam",
            )

        with col_sq2:
            render_chart(
                build_histogram_figure(
                    df_sample["tempo_tratamento_dias"],
                    sam_tempo,
                    title="Histograma do Tempo até o Tratamento na Amostra",
                    x_label="Tempo até o 1º tratamento",
                    unit="dias",
                    bar_color="#fbbf24",
                    is_population=False,
                    nbins=26,
                    ref_line_value=60.0,
                    ref_line_label="Prazo Legal",
                    show_normal_fit=False,
                ),
                key="hist_tempo_sam",
            )
            render_chart(
                build_boxplot_figure(
                    sam_tempo,
                    title="Boxplot do Tempo até o Tratamento na Amostra",
                    x_label="Tempo até o 1º tratamento",
                    unit="dias",
                    box_color="#fbbf24",
                    is_population=False,
                ),
                key="box_tempo_sam",
            )

        st.markdown("#### 3. Frequências das Variáveis Qualitativas na Amostra e Comparação com a População")
        cs_cat1, cs_cat2 = st.columns(2)
        with cs_cat1:
            render_chart(
                build_categorical_bar_figure(
                    df_sample["tratado_ate_60_dias"],
                    title="Cumprimento da Lei dos 60 Dias (Amostra)",
                    category_label="Prazo até o 1º tratamento",
                    bar_color=PALETTE["sample"],
                    is_population=False,
                    custom_labels={
                        "False": "Acima de 60 dias (Fora do prazo)",
                        "True": "Até 60 dias (Dentro do prazo)",
                    },
                ),
                key="bar_lei_sam",
            )
            render_chart(
                build_comparison_bar_figure(
                    df_pop["regiao_residencia"],
                    df_sample["regiao_residencia"],
                    title="Comparativo de Proporções por Região: População (p) vs. Amostra (p̂)",
                    category_label="Região de residência",
                ),
                key="comp_reg_bar",
            )

        with cs_cat2:
            render_chart(
                build_categorical_bar_figure(
                    df_sample["primeiro_tratamento"],
                    title="Modalidade do Primeiro Tratamento (Amostra)",
                    category_label="Primeiro tratamento registrado",
                    bar_color=PALETTE["sample"],
                    is_population=False,
                ),
                key="bar_trat_sam",
            )
            render_chart(
                build_comparison_bar_figure(
                    df_pop["faixa_etaria"],
                    df_sample["faixa_etaria"],
                    title="Comparativo por Faixa Etária: População (p) vs. Amostra (p̂)",
                    category_label="Faixa etária (anos)",
                ),
                key="comp_fx_bar",
            )

        with st.container(border=True):
            st.markdown(
                "**Diferenças entre População e Amostra:** As estimativas da amostra (n = 382) acompanharam de perto as distribuições da população (N = 48.866). A idade média amostral diferiu em 0,76 ano da população (57,46 vs. 56,70 anos) e o tempo mediano diferiu em 3,5 dias (85,5 vs. 82,0 dias). Nas proporções, a diferença foi de 4,33 pontos percentuais no cumprimento da lei (32,98% vs. 37,31%) e de 1,69 ponto percentual na faixa de 50 a 69 anos (48,43% vs. 50,12%), ambos dentro da margem de erro de 5 pontos percentuais."
            )

    # =========================================================================
    # ABA 4: PAINEL 3 - INFERÊNCIA ESTATÍSTICA E CONFRONTO
    # =========================================================================
    with tab4:
        st.subheader("Painel 3: Inferência Estatística e Comparação dos Resultados")

        with st.container(border=True):
            st.markdown(
                "**Condições de aplicabilidade verificadas:** (1) Amostra Aleatória Simples (n = 382 < 5% de N = 48.866); (2) Para os testes Z de proporção, o número de sucessos e fracassos é superior a 5 nas duas hipóteses (126 e 256 em P1; 185 e 197 em P2); (3) Para os testes t de média, como n = 382 ≥ 30, o Teorema Central do Limite assegura a normalidade da distribuição amostral da média x̄ mesmo para variáveis assimétricas como o tempo de espera."
            )

        z_crit = float(stats.norm.ppf(0.975))
        t_crit = float(stats.t.ppf(0.975, df=n_sample - 1))

        # Cálculos P1
        se_ci_p1 = float(np.sqrt(p_sam_60 * (1.0 - p_sam_60) / n_sample))
        ci_p1 = (p_sam_60 - z_crit * se_ci_p1, p_sam_60 + z_crit * se_ci_p1)
        z_stat_p1 = (p_sam_60 - 0.50) / float(np.sqrt(0.50 * 0.50 / n_sample))
        pval_p1 = float(stats.norm.cdf(z_stat_p1))

        # Cálculos P2
        se_ci_p2 = float(np.sqrt(p_sam_rast * (1.0 - p_sam_rast) / n_sample))
        ci_p2 = (p_sam_rast - z_crit * se_ci_p2, p_sam_rast + z_crit * se_ci_p2)
        z_stat_p2 = (p_sam_rast - 0.50) / float(np.sqrt(0.50 * 0.50 / n_sample))
        pval_p2 = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat_p2))))

        # Cálculos M1
        se_m1 = sam_idade["std"] / float(np.sqrt(n_sample))
        ci_m1 = (sam_idade["mean"] - t_crit * se_m1, sam_idade["mean"] + t_crit * se_m1)
        t_stat_m1 = (sam_idade["mean"] - 50.0) / se_m1
        pval_m1 = float(1.0 - stats.t.cdf(t_stat_m1, df=n_sample - 1))

        # Cálculos M2
        se_m2 = sam_tempo["std"] / float(np.sqrt(n_sample))
        ci_m2 = (sam_tempo["mean"] - t_crit * se_m2, sam_tempo["mean"] + t_crit * se_m2)
        t_stat_m2 = (sam_tempo["mean"] - 60.0) / se_m2
        pval_m2 = float(1.0 - stats.t.cdf(t_stat_m2, df=n_sample - 1))

        st.markdown("#### A) Inferência para Dois Parâmetros de Proporção (Nível de Significância α = 5%)")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            with st.container(border=True):
                st.markdown("##### Hipótese de Proporção 1 (P1): Tratamento em Até 60 Dias")
                st.markdown(
                    f"""
                    - **Variável e evento:** `tratado_ate_60_dias` (iniciar tratamento em até 60 dias).
                    - **Hipóteses:** H₀: p ≥ 0,50 vs. H₁: p < 0,50 (unilateral à esquerda).
                    - **Estimativa amostral (p̂):** {fmt_br(p_sam_60*100)}% (126/382) | **Parâmetro real (p):** {fmt_br(p_pop_60*100)}%.
                    - **IC 95%:** [{fmt_br(ci_p1[0]*100)}%; {fmt_br(ci_p1[1]*100)}%]. Temos 95% de confiança de que o nosso intervalo contém (abraça) a verdadeira proporção populacional p (o valor real de 37,31% está contido no intervalo).
                    - **Teste Z:** Z = {fmt_br(z_stat_p1, 4)} | valor-p = {pval_p1:.2e} (α = 0,05).
                    - **Conclusão:** Como valor-p < 0,05, **rejeita-se H₀**. Há evidência estatística de que menos da metade das pacientes inicia o tratamento em até 60 dias, decisão que concorda com o parâmetro populacional (37,31%).
                    """
                )
                render_chart(
                    build_ci_comparison_figure(
                        p_pop_60,
                        p_sam_60,
                        ci_p1[0],
                        ci_p1[1],
                        0.50,
                        title="P1: Intervalo de Confiança (95%) vs. Parâmetro Real (p)",
                        x_label="Proporção tratada em até 60 dias",
                        unit="%",
                        is_proportion=True,
                    ),
                    key="ci_chart_p1",
                )

        with col_p2:
            with st.container(border=True):
                st.markdown("##### Hipótese de Proporção 2 (P2): Faixa de Rastreamento (50 a 69 Anos)")
                st.markdown(
                    f"""
                    - **Variável e evento:** `faixa_rastreamento_50_69` (diagnóstico entre 50 e 69 anos).
                    - **Hipóteses:** H₀: p = 0,50 vs. H₁: p ≠ 0,50 (bilateral).
                    - **Estimativa amostral (p̂):** {fmt_br(p_sam_rast*100)}% (185/382) | **Parâmetro real (p):** {fmt_br(p_pop_rast*100)}%.
                    - **IC 95%:** [{fmt_br(ci_p2[0]*100)}%; {fmt_br(ci_p2[1]*100)}%]. Temos 95% de confiança de que o nosso intervalo contém (abraça) a verdadeira proporção populacional p (o valor real de 50,12% está contido no intervalo).
                    - **Teste Z:** Z = {fmt_br(z_stat_p2, 4)} | valor-p = {fmt_br(pval_p2, 4)} (α = 0,05).
                    - **Conclusão:** Como valor-p (0,5392) ≥ 0,05, **não se rejeita H₀**. A amostra não traz evidência suficiente para afirmar que a proporção na faixa de 50 a 69 anos difere de 50%, resultado coerente com a população (50,12%).
                    """
                )
                render_chart(
                    build_ci_comparison_figure(
                        p_pop_rast,
                        p_sam_rast,
                        ci_p2[0],
                        ci_p2[1],
                        0.50,
                        title="P2: Intervalo de Confiança (95%) vs. Parâmetro Real (p)",
                        x_label="Proporção na faixa de 50 a 69 anos",
                        unit="%",
                        is_proportion=True,
                    ),
                    key="ci_chart_p2",
                )

        st.markdown("#### B) Inferência para Dois Parâmetros de Média (Nível de Significância α = 5%)")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            with st.container(border=True):
                st.markdown("##### Hipótese de Média 1 (M1): Idade Média no Diagnóstico")
                st.markdown(
                    f"""
                    - **Variável:** `idade` (anos completos no diagnóstico).
                    - **Hipóteses:** H₀: μ ≤ 50 anos vs. H₁: μ > 50 anos (unilateral à direita).
                    - **Estimativa amostral (x̄):** {fmt_br(sam_idade['mean'])} anos (s = {fmt_br(sam_idade['std'])}) | **Parâmetro real (μ):** {fmt_br(pop_idade['mean'])} anos.
                    - **IC 95%:** [{fmt_br(ci_m1[0])}; {fmt_br(ci_m1[1])}] anos. Temos 95% de confiança de que o nosso intervalo contém (abraça) a verdadeira média populacional μ (o valor real de 56,70 anos está contido no intervalo).
                    - **Teste t (gl = 381):** t = {fmt_br(t_stat_m1, 4)} | valor-p = {pval_m1:.2e} (α = 0,05).
                    - **Conclusão:** Como valor-p < 0,05, **rejeita-se H₀**. Há evidência estatística de que a idade média no diagnóstico supera 50 anos, confirmando a média populacional (56,70 anos).
                    """
                )
                render_chart(
                    build_ci_comparison_figure(
                        pop_idade["mean"],
                        sam_idade["mean"],
                        ci_m1[0],
                        ci_m1[1],
                        50.0,
                        title="M1: Intervalo de Confiança (95%) vs. Parâmetro Real (μ)",
                        x_label="Idade no diagnóstico",
                        unit=" anos",
                        is_proportion=False,
                    ),
                    key="ci_chart_m1",
                )

        with col_m2:
            with st.container(border=True):
                st.markdown("##### Hipótese de Média 2 (M2): Tempo Médio até o Tratamento")
                st.markdown(
                    f"""
                    - **Variável:** `tempo_tratamento_dias` (dias até o 1º tratamento).
                    - **Hipóteses:** H₀: μ ≤ 60 dias vs. H₁: μ > 60 dias (unilateral à direita).
                    - **Estimativa amostral (x̄):** {fmt_br(sam_tempo['mean'])} dias (s = {fmt_br(sam_tempo['std'])}) | **Parâmetro real (μ):** {fmt_br(pop_tempo['mean'])} dias.
                    - **IC 95%:** [{fmt_br(ci_m2[0])}; {fmt_br(ci_m2[1])}] dias. Temos 95% de confiança de que o nosso intervalo contém (abraça) a verdadeira média populacional μ (o valor real de 111,40 dias está contido no intervalo).
                    - **Teste t (gl = 381):** t = {fmt_br(t_stat_m2, 4)} | valor-p = {pval_m2:.2e} (α = 0,05).
                    - **Conclusão:** Como valor-p < 0,05, **rejeita-se H₀**. Há evidência estatística de que o tempo médio até o tratamento ultrapassa o limite legal de 60 dias, em acordo com a população (111,40 dias).
                    """
                )
                render_chart(
                    build_ci_comparison_figure(
                        pop_tempo["mean"],
                        sam_tempo["mean"],
                        ci_m2[0],
                        ci_m2[1],
                        60.0,
                        title="M2: Intervalo de Confiança (95%) vs. Parâmetro Real (μ)",
                        x_label="Tempo até o 1º tratamento",
                        unit=" dias",
                        is_proportion=False,
                    ),
                    key="ci_chart_m2",
                )

        st.markdown("#### C) Resumo Comparativo: População vs. Amostra")
        df_confronto = pd.DataFrame(
            [
                {
                    "Análise": "Proporção 1 (P1): Tratamento ≤ 60 dias",
                    "População (N=48.866)": "p = 37,31%",
                    "Amostra (n=382)": "p̂ = 32,98%",
                    "IC 95%": "[28,27%; 37,70%]",
                    "IC Abraça Parâmetro?": "Sim",
                    "Hipóteses": "H₀: p ≥ 0,50 vs. H₁: p < 0,50",
                    "Estatística / valor-p": f"Z = {fmt_br(z_stat_p1, 2)} (p < 0,0001)",
                    "Decisão (α = 5%)": "Rejeita-se H₀",
                },
                {
                    "Análise": "Proporção 2 (P2): Faixa 50 a 69 anos",
                    "População (N=48.866)": "p = 50,12%",
                    "Amostra (n=382)": "p̂ = 48,43%",
                    "IC 95%": "[43,42%; 53,44%]",
                    "IC Abraça Parâmetro?": "Sim",
                    "Hipóteses": "H₀: p = 0,50 vs. H₁: p ≠ 0,50",
                    "Estatística / valor-p": f"Z = {fmt_br(z_stat_p2, 2)} (p = 0,5392)",
                    "Decisão (α = 5%)": "Não se rejeita H₀",
                },
                {
                    "Análise": "Média 1 (M1): Idade no diagnóstico",
                    "População (N=48.866)": "μ = 56,70 anos",
                    "Amostra (n=382)": "x̄ = 57,46 anos",
                    "IC 95%": "[56,07; 58,86] anos",
                    "IC Abraça Parâmetro?": "Sim",
                    "Hipóteses": "H₀: μ ≤ 50 vs. H₁: μ > 50",
                    "Estatística / valor-p": f"t = {fmt_br(t_stat_m1, 2)} (p < 0,0001)",
                    "Decisão (α = 5%)": "Rejeita-se H₀",
                },
                {
                    "Análise": "Média 2 (M2): Tempo até 1º tratamento",
                    "População (N=48.866)": "μ = 111,40 dias",
                    "Amostra (n=382)": "x̄ = 116,54 dias",
                    "IC 95%": "[105,85; 127,23] dias",
                    "IC Abraça Parâmetro?": "Sim",
                    "Hipóteses": "H₀: μ ≤ 60 vs. H₁: μ > 60",
                    "Estatística / valor-p": f"t = {fmt_br(t_stat_m2, 2)} (p < 0,0001)",
                    "Decisão (α = 5%)": "Rejeita-se H₀",
                },
            ]
        )
        render_table(df_confronto, key="tbl_confronto_p3")

        with st.container(border=True):
            st.markdown(
                """
                **Variabilidade Amostral e Limitações da Análise:**
                - **Efeito da variabilidade amostral:** As estimativas pontuais da amostra apresentam pequenas diferenças em relação aos parâmetros da população devido ao sorteio aleatório, mas os quatro intervalos de confiança de 95% abraçaram os parâmetros populacionais reais e levaram às mesmas conclusões dos dados populacionais.
                - **Limitações da base:** (1) Os registros abrangem exclusivamente o SUS, não representando atendimentos realizados na rede privada; (2) O tempo até o tratamento considera o primeiro procedimento administrativo registrado no sistema (SIA/SIH), excluindo casos sem informação de tratamento ou com preenchimento inconsistente.
                """
            )

    # =========================================================================
    # ABA 5: REFERÊNCIAS E EXTRAÇÃO DE DADOS (APENAS ITENS 1 E 2)
    # =========================================================================
    with tab5:
        st.subheader("Referências e Extração de Dados")

        st.markdown("#### 1. Identificação Oficial da Fonte de Dados")
        df_fontes = pd.DataFrame(
            [
                {
                    "Item": "Fonte Oficial",
                    "Descrição / Link de Acesso": "Ministério da Saúde / DATASUS - PAINEL-Oncologia Brasil",
                },
                {
                    "Item": "Página de Acesso aos Dados",
                    "Descrição / Link de Acesso": "https://datasus.saude.gov.br/transferencia-de-arquivos/ (Fonte: PAINEL-ONCOLOGIA, Modalidade: Dados, Tipo: PO)",
                },
                {
                    "Item": "Link Direto da Base Original (2024)",
                    "Descrição / Link de Acesso": "ftp://ftp.datasus.gov.br/dissemin/publicos/PAINEL_ONCOLOGIA/Dados/POBR2024.dbc",
                },
                {
                    "Item": "Dicionário de Variáveis",
                    "Descrição / Link de Acesso": "ftp://ftp.datasus.gov.br/dissemin/publicos/PAINEL_ONCOLOGIA/Doc/Dicionario_Painel_Oncologia.pdf",
                },
                {
                    "Item": "Nota Técnica Oficial",
                    "Descrição / Link de Acesso": "http://tabnet.datasus.gov.br/cgi/painel_onco/doc/painel_oncologia.pdf",
                },
            ]
        )
        render_table(df_fontes, key="tbl_fontes")

        st.markdown("#### 2. Etapas de Filtragem e Seleção dos Dados")
        df_funil = pd.DataFrame(
            [
                {
                    "Etapa": "1. Base bruta nacional do PAINEL-Oncologia 2024 (POBR2024.dbc)",
                    "Critério Aplicado": "Total de neoplasias registradas no SUS em 2024",
                    "Quantidade de Registros": "644.320",
                },
                {
                    "Etapa": "2. Recorte de câncer de mama (cancer_mama_2024.csv)",
                    "Critério Aplicado": "Filtro DIAG_DETH == 'C50' (Neoplasia maligna da mama)",
                    "Quantidade de Registros": "64.776",
                },
                {
                    "Etapa": "3. População de interesse (populacao_mama_2024.csv)",
                    "Critério Aplicado": "Sexo feminino, idade entre 18 e 115 anos e tempo_tratamento_dias >= 0",
                    "Quantidade de Registros": "48.866",
                },
                {
                    "Etapa": "4. Amostra Aleatória Simples (AAS)",
                    "Critério Aplicado": "Sorteio aleatório simples sem reposição (n = 382, random_state = 42)",
                    "Quantidade de Registros": "382",
                },
            ]
        )
        render_table(df_funil, key="tbl_funil")


if __name__ == "__main__":
    main()
