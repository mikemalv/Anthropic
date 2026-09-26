"""Plotly figure builders. Pure functions: DataFrame in, Figure out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Validated categorical slots (dataviz reference palette); color follows the entity.
COLORS = {"hike": "#2a78d6", "cut": "#eb6834", "hold": "#1baf7a"}
# Decision-sample groups are different entities from hike/cut, so they get their own hues.
SAMPLE_COLORS = {"all": "#4a3aa7", "stress": "#4a3aa7", "routine": "#8a8983"}
REGIME_COLORS = {"tightening": COLORS["hike"], "easing": COLORS["cut"], "on hold": COLORS["hold"]}
NEUTRAL = "#8a8983"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e8e7e3"
SURFACE = "#fcfcfb"


def _style(fig: go.Figure, title: str, source: str, height: int = 460) -> go.Figure:
    fig.update_layout(
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 17, "color": TEXT},
        },
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Inter, -apple-system, Segoe UI, sans-serif", "size": 12, "color": TEXT},
        margin={"l": 64, "r": 24, "t": 96, "b": 72},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "xanchor": "left"},
        hoverlabel={"bgcolor": "white", "font_color": TEXT},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont_color=TEXT_MUTED)
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=NEUTRAL, linecolor=GRID, tickfont_color=TEXT_MUTED
    )
    fig.add_annotation(
        text=f"Source: FRED, {source}",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.16,
        xanchor="left",
        showarrow=False,
        font={"size": 11, "color": TEXT_MUTED},
    )
    return fig


def _band(fig: go.Figure, x, low, high, color: str, name: str, **kw) -> None:
    fig.add_trace(
        go.Scatter(
            x=np.r_[x, x[::-1]],
            y=np.r_[high, low[::-1]],
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            name=name,
        ),
        **kw,
    )


def policy_and_vix(target: pd.Series, vix: pd.Series, events: pd.DataFrame) -> go.Figure:
    """Two stacked panels on a shared time axis: policy target with moves marked, and VIX."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.4, 0.6]
    )
    fig.add_trace(
        go.Scatter(
            x=target.index,
            y=target,
            line={"color": TEXT_MUTED, "width": 2, "shape": "hv"},
            name="Fed target (upper bound)",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Target %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for direction in ["hike", "cut"]:
        e = events[events["direction"] == direction]
        fig.add_trace(
            go.Scatter(
                x=e.index,
                y=e["target_after_pct"],
                mode="markers",
                name=f"Rate {direction}",
                marker={
                    "color": COLORS[direction],
                    "size": 8,
                    "line": {"color": SURFACE, "width": 2},
                },
                customdata=e["change_bp"],
                hovertemplate="%{x|%Y-%m-%d}<br>%{customdata:+.0f} bp → %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=vix.index,
            y=vix,
            line={"color": TEXT_MUTED, "width": 1.2},
            name="VIX",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>VIX %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Fed target (%)", row=1, col=1)
    fig.update_yaxes(title_text="VIX (index)", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return _style(
        fig,
        "Fed policy moves and market volatility, 1990–today",
        "DFEDTAR, DFEDTARU, VIXCLS",
        height=560,
    )


def event_paths(summaries: dict[str, pd.DataFrame], ylabel: str, source: str) -> go.Figure:
    """Average path around policy events with 95% bands, one line per group (hike/cut)."""
    fig = go.Figure()
    for name, s in summaries.items():
        color = COLORS.get(name, NEUTRAL)
        x = s.index.to_numpy()
        _band(fig, x, s["ci_low"].to_numpy(), s["ci_high"].to_numpy(), color, name)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=s["mean"],
                name=f"{name.title()}s (n={int(s['n'].iloc[0])})",
                line={"color": color, "width": 2},
                hovertemplate="Day %{x}<br>%{y:+.2f}<extra>" + name + "</extra>",
            )
        )
    fig.add_vline(x=0, line={"color": NEUTRAL, "dash": "dot", "width": 1})
    fig.update_xaxes(title_text="Trading days relative to the decision (0 = decision day)")
    fig.update_yaxes(title_text=ylabel, zeroline=True)
    fig.update_layout(hovermode="x unified")
    return _style(fig, "VIX around Fed decisions: average change from the day before", source)


def placebo_histogram(
    null_draws: pd.DataFrame, observed: dict[str, float], horizon: int
) -> go.Figure:
    """Placebo distributions (random days at matched VIX levels) vs. the observed event mean."""
    fig = go.Figure()
    for name, obs in observed.items():
        color = COLORS[name]
        fig.add_trace(
            go.Histogram(
                x=null_draws[name],
                name=f"Placebo days matched to {name}s",
                marker_color=color,
                opacity=0.35,
                nbinsx=50,
                hovertemplate="Mean change %{x:.2f}<br>%{y} draws<extra></extra>",
            )
        )
        fig.add_vline(
            x=obs,
            line={"color": color, "width": 2},
            annotation_text=f"Actual {name}s: {obs:+.2f}",
            annotation_font_color=TEXT,
            annotation_position="top left" if obs < 0 else "top right",
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text=f"Mean VIX change, day −1 to day +{horizon} (points)")
    fig.update_yaxes(title_text="Placebo draws")
    return _style(
        fig,
        f"Are Fed decision days special? Day +{horizon} vs. 2,000 placebo samples",
        "VIXCLS, DFEDTAR, DFEDTARU",
    )


def regime_boxes(frame: pd.DataFrame, value_col: str, ylabel: str, source: str) -> go.Figure:
    """Distribution of a volatility measure by policy regime."""
    fig = go.Figure()
    for regime in ["tightening", "on hold", "easing"]:
        v = frame.loc[frame["regime"] == regime, value_col]
        fig.add_trace(
            go.Box(
                y=v,
                name=f"{regime.title()} ({len(v):,} days)",
                marker_color=REGIME_COLORS[regime],
                boxpoints=False,
                line={"width": 1.5},
            )
        )
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(showlegend=False)
    return _style(fig, f"{ylabel} by Fed policy regime (6-month change in target)", source)


def surprise_scatter(events: pd.DataFrame, fits: dict[str, tuple[float, float]]) -> go.Figure:
    """Decision-day 2-year yield surprise vs. same-day VIX change, with fitted lines."""
    fig = go.Figure()
    for crisis, label, color in [
        (False, "Routine decisions", SAMPLE_COLORS["routine"]),
        (True, "Stress-period decisions (VIX ≥ 25 the day before)", SAMPLE_COLORS["stress"]),
    ]:
        e = events[events["crisis"] == crisis]
        fig.add_trace(
            go.Scatter(
                x=e["surprise_bp"],
                y=e["dvix0"],
                mode="markers",
                name=label,
                marker={"color": color, "size": 9, "line": {"color": SURFACE, "width": 2}},
                customdata=np.c_[
                    e.index.strftime("%Y-%m-%d"), e["change_bp"].map("{:+.0f}".format)
                ],
                hovertemplate="%{customdata[0]} (%{customdata[1]} bp)<br>"
                "2y surprise %{x:+.0f} bp<br>VIX %{y:+.2f}<extra></extra>",
            )
        )
    xs = np.linspace(events["surprise_bp"].min(), events["surprise_bp"].max(), 50)
    for (name, (a, b)), dash in zip(fits.items(), ["solid", "dash"], strict=False):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=a + b * xs,
                mode="lines",
                name=name,
                line={"color": TEXT, "width": 2, "dash": dash},
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(
        title_text="Policy surprise: 2-year Treasury yield change on decision day (bp)"
    )
    fig.update_yaxes(title_text="VIX change on decision day (points)")
    return _style(fig, "Hawkish surprises vs. same-day VIX moves", "DGS2, VIXCLS", height=520)


def surprise_response(lp: pd.DataFrame, lp_routine: pd.DataFrame) -> go.Figure:
    """Estimated VIX response to a 25 bp hawkish surprise, by horizon, with 95% bands."""
    fig = go.Figure()
    for name, frame, color in [
        ("All decisions", lp, SAMPLE_COLORS["all"]),
        ("Routine decisions only (VIX < 25)", lp_routine, SAMPLE_COLORS["routine"]),
    ]:
        x = frame.index.to_numpy()
        _band(fig, x, frame["ci_low"].to_numpy(), frame["ci_high"].to_numpy(), color, name)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=frame["effect"],
                name=name,
                line={"color": color, "width": 2},
                hovertemplate="Day %{x}<br>%{y:+.2f} VIX pts<extra>" + name + "</extra>",
            )
        )
    fig.add_hline(y=0, line={"color": NEUTRAL, "width": 1})
    fig.update_xaxes(title_text="Trading days after the decision")
    fig.update_yaxes(title_text="VIX change per +25 bp surprise (points)")
    fig.update_layout(hovermode="x unified")
    return _style(fig, "How long does a policy surprise move VIX?", "DGS2, VIXCLS")
