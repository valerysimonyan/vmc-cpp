#!/usr/bin/env python3
"""Live dashboard for VMC training progress.

Reads training.csv, written by descent() in lib/descent.cpp, and polls it
on an interval to refresh the plots. Does not touch or depend on any C++ source.

Usage:
    pip install dash plotly pandas
    python plot_training.py
    open http://127.0.0.1:8050

A row is classified as an SR step if lambda > 0 (ADAM rows leave the SRStepLog
default-constructed, so lambda/cg_iters/etc. are exactly 0) -- this tracks
N_gd automatically instead of hardcoding it here.
"""

import math
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training.csv")
REFRESH_MS = 250
BETA_MIN = 0.1  # must match beta_min in lib/constants.h -- envelope decay = beta_min + exp(alpha)

BG = "#111111"
CARD_BG = "#1a1a1a"
BORDER = "#333333"
TEXT_DIM = "#888888"
TEXT = "#eeeeee"
COLOR_ENERGY = "#4fc3f7"
COLOR_VAR = "#ba68c8"
COLOR_ACCEPT = "#81c784"
COLOR_SPIN_ACCEPT = "#4db6ac"
COLOR_TAU_ACCEPT = "#f06292"
COLOR_SR = "#4fc3f7"
COLOR_ADAM = "#ffb74d"
COLOR_RRMS = "#aed581"
COLOR_ALPHA = "#ffd54f"
COLOR_L2 = "#7986cb"
COLOR_METRO_MS = "#4fc3f7"
COLOR_LOCAL_E_MS = "#ba68c8"
COLOR_SR_MS = "#ffb74d"
COLOR_MS_ITER = "#e57373"
COLOR_WARN = "#e57373"
COLOR_OK = "#81c784"

app = Dash(__name__)
app.title = "VMC Training Monitor"

app.layout = html.Div(
    style={
        "fontFamily": "Menlo, Consolas, monospace",
        "backgroundColor": BG,
        "color": TEXT,
        "padding": "24px",
        "minHeight": "100vh",
    },
    children=[
        html.H2("VMC Training Monitor", style={"marginTop": 0}),
        html.Div(id="stat-cards", style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "marginBottom": "24px"}),
        dcc.Graph(id="energy-plot"),
        dcc.Graph(id="var-plot"),
        dcc.Graph(id="accept-plot"),
        dcc.Graph(id="rrms-plot"),
        dcc.Graph(id="l2-plot"),
        dcc.Graph(id="alpha-plot"),
        dcc.Graph(id="timing-plot"),
        dcc.Interval(id="tick", interval=REFRESH_MS, n_intervals=0),
    ],
)


def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        # File mid-write (partial last line) -- just wait for the next tick.
        return None
    if df.empty:
        return None
    return df


def stat_card(label, value, color=TEXT):
    return html.Div(
        style={
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "padding": "10px 16px",
            "minWidth": "140px",
            "backgroundColor": CARD_BG,
        },
        children=[
            html.Div(label, style={"fontSize": "12px", "color": TEXT_DIM}),
            html.Div(value, style={"fontSize": "20px", "color": color}),
        ],
    )


def empty_figure(title):
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, title=title)
    return fig


def add_phase_transition(fig, transition_step):
    if transition_step is not None:
        fig.add_vline(x=transition_step, line_dash="dash", line_color="#888888")


@app.callback(
    [
        Output("stat-cards", "children"),
        Output("energy-plot", "figure"),
        Output("var-plot", "figure"),
        Output("accept-plot", "figure"),
        Output("rrms-plot", "figure"),
        Output("l2-plot", "figure"),
        Output("alpha-plot", "figure"),
        Output("timing-plot", "figure"),
    ],
    [Input("tick", "n_intervals")],
)
def update(_n):
    df = load_data()
    if df is None:
        waiting = empty_figure("waiting for training.csv ...")
        return [stat_card("Status", "waiting for training.csv ...")], waiting, waiting, waiting, waiting, waiting, waiting, waiting

    last = df.iloc[-1]
    is_sr = last["lambda"] > 0
    phase = "SR" if is_sr else "ADAM"
    phase_color = COLOR_SR if is_sr else COLOR_ADAM

    cards = [
        stat_card("Step", f"{int(last['step'])}"),
        stat_card("Phase", phase, color=phase_color),
        stat_card("Energy", f"{last['E_exp']:.5f} ± {last['E_err']:.5f}"),
        stat_card("Variance", f"{last['var']:.5g}"),
        stat_card("Acceptance", f"{last['acceptance']:.3f}"),
        stat_card("Spin Acceptance", f"{last['spin_acceptance']:.3f}", color=COLOR_SPIN_ACCEPT),
        stat_card("Tau Acceptance", f"{last['tau_acceptance']:.3f}", color=COLOR_TAU_ACCEPT),
        stat_card("r_rms", f"{last['r_rms']:.4f} fm", color=COLOR_RRMS),
        stat_card("L²", f"{last['L2']:.4f}", color=COLOR_L2),
        stat_card("Node Hits", f"{int(last['node_hits'])}", color=COLOR_WARN if last["node_hits"] > 0 else TEXT),
        stat_card("β (env. decay)", f"{BETA_MIN + math.exp(last['alpha']):.5g} fm⁻¹", color=COLOR_ALPHA),
        stat_card("Grad Alpha", f"{last['grad_alpha']:.4g}"),
    ]
    if is_sr:
        cards += [
            stat_card("Lambda", f"{last['lambda']:.4g}"),
            stat_card("RMS Damp (mean)", f"{last['rms_damp_mean']:.4g}"),
            stat_card("CG Iters", f"{int(last['cg_iters'])}"),
            stat_card("CG Residual", f"{last['cg_residual']:.3g}"),
            stat_card("Delta Norm", f"{last['delta_norm']:.4g}"),
            stat_card("Metric Norm²", f"{last['sq_metric_norm']:.4g}"),
            stat_card("Delta Alpha", f"{last['delta_alpha']:.4g}"),
            stat_card("Norm Capped", "YES" if last["norm_capped"] else "no",
                      color=COLOR_WARN if last["norm_capped"] else COLOR_OK),
        ]

    sr_mask = df["lambda"] > 0
    transition_step = df.loc[sr_mask, "step"].min() if sr_mask.any() else None

    # Energy plot, shaded +/- E_err band around E_exp
    upper = df["E_exp"] + df["E_err"]
    lower = df["E_exp"] - df["E_err"]
    energy_fig = go.Figure()
    energy_fig.add_trace(go.Scatter(x=df["step"], y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    energy_fig.add_trace(go.Scatter(
        x=df["step"], y=lower, line=dict(width=0), fill="tonexty",
        fillcolor="rgba(79,195,247,0.2)", showlegend=False, hoverinfo="skip",
    ))
    energy_fig.add_trace(go.Scatter(
        x=df["step"], y=df["E_exp"], mode="lines+markers", name="E_exp",
        line=dict(color=COLOR_ENERGY),
    ))
    add_phase_transition(energy_fig, transition_step)
    energy_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="Energy", xaxis_title="step", yaxis_title="E_exp",
        margin=dict(t=40, b=30),
    )

    var_fig = go.Figure()
    var_fig.add_trace(go.Scatter(
        x=df["step"], y=df["var"], mode="lines+markers", name="var", line=dict(color=COLOR_VAR),
    ))
    add_phase_transition(var_fig, transition_step)
    var_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="Variance", xaxis_title="step", yaxis_title="var", yaxis_type="log",
        margin=dict(t=40, b=30),
    )

    # Acceptance plot: all three exchange rates together. The 0.45/0.55
    # reference band is calibrated to the position-step tuning target in
    # parallel_run() specifically -- it doesn't literally apply to
    # spin/tau acceptance (those are untuned discrete swaps), shown here
    # only as a rough eyeball reference.
    accept_fig = go.Figure()
    accept_fig.add_trace(go.Scatter(
        x=df["step"], y=df["acceptance"], mode="lines+markers", name="acceptance", line=dict(color=COLOR_ACCEPT),
    ))
    accept_fig.add_trace(go.Scatter(
        x=df["step"], y=df["spin_acceptance"], mode="lines+markers", name="spin_acceptance", line=dict(color=COLOR_SPIN_ACCEPT),
    ))
    accept_fig.add_trace(go.Scatter(
        x=df["step"], y=df["tau_acceptance"], mode="lines+markers", name="tau_acceptance", line=dict(color=COLOR_TAU_ACCEPT),
    ))
    accept_fig.add_hline(y=0.45, line_dash="dot", line_color="#666666")
    accept_fig.add_hline(y=0.55, line_dash="dot", line_color="#666666")
    add_phase_transition(accept_fig, transition_step)
    accept_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="Acceptance rates", xaxis_title="step", yaxis_title="acceptance",
        yaxis_range=[0, 1], margin=dict(t=40, b=30),
    )

    rrms_fig = go.Figure()
    rrms_fig.add_trace(go.Scatter(
        x=df["step"], y=df["r_rms"], mode="lines+markers", name="r_rms", line=dict(color=COLOR_RRMS),
    ))
    add_phase_transition(rrms_fig, transition_step)
    rrms_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="r_rms (point matter radius)", xaxis_title="step", yaxis_title="r_rms (fm)",
        margin=dict(t=40, b=30),
    )

    l2_fig = go.Figure()
    l2_fig.add_trace(go.Scatter(
        x=df["step"], y=df["L2"], mode="lines+markers", name="L2", line=dict(color=COLOR_L2),
    ))
    l2_fig.add_hline(y=0, line_dash="dot", line_color="#666666")   # L=0 (S-wave)
    l2_fig.add_hline(y=2, line_dash="dot", line_color="#666666")   # L=1 (P-wave), l(l+1)=2
    add_phase_transition(l2_fig, transition_step)
    l2_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="⟨L²⟩ (ℏ²=1 units) -- dotted: l(l+1) for L=0, L=1", xaxis_title="step", yaxis_title="L2",
        margin=dict(t=40, b=30),
    )

    alpha_fig = go.Figure()
    beta = BETA_MIN + df["alpha"].apply(math.exp)
    alpha_fig.add_trace(go.Scatter(
        x=df["step"], y=beta, mode="lines+markers", name="beta", line=dict(color=COLOR_ALPHA),
    ))
    add_phase_transition(alpha_fig, transition_step)
    alpha_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="Envelope decay rate  β = beta_min + eᵅ", xaxis_title="step", yaxis_title="β (fm⁻¹)",
        margin=dict(t=40, b=30),
    )

    timing_fig = go.Figure()
    timing_fig.add_trace(go.Scatter(x=df["step"], y=df["metro_ms"], mode="lines", name="metro_ms", line=dict(color=COLOR_METRO_MS)))
    timing_fig.add_trace(go.Scatter(x=df["step"], y=df["local_E_ms"], mode="lines", name="local_E_ms", line=dict(color=COLOR_LOCAL_E_MS)))
    timing_fig.add_trace(go.Scatter(x=df["step"], y=df["sr_ms"], mode="lines", name="sr_ms", line=dict(color=COLOR_SR_MS)))
    timing_fig.add_trace(go.Scatter(x=df["step"], y=df["ms_iter"], mode="lines", name="ms_iter (total)", line=dict(color=COLOR_MS_ITER, width=2, dash="dot")))
    add_phase_transition(timing_fig, transition_step)
    timing_fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="Timing breakdown", xaxis_title="step", yaxis_title="ms",
        margin=dict(t=40, b=30),
    )

    return cards, energy_fig, var_fig, accept_fig, rrms_fig, l2_fig, alpha_fig, timing_fig


if __name__ == "__main__":
    app.run(debug=False, port=8050)