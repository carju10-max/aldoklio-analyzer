"""
app.py — Aldo&Klio Analyzer
Interfaz web construida con Streamlit.
Ejecutar: streamlit run app.py
"""

import os
import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Aldo&Klio Analyzer",
    page_icon="🎚️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────────────────── */
.stApp { background-color: #080808; }
[data-testid="stSidebar"] { background: #0d0d0d; border-right: 1px solid #1c1c1c; }

/* ── Top nav bar ────────────────────────────────────────────────────────────── */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #080808;
    border-bottom: 1px solid #1a1a1a;
    padding: 0 28px;
    height: 58px;
    margin: -1rem -1rem 1.6rem -1rem;
}
.top-nav-logo {
    font-size: 1.45rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.top-nav-logo span {
    background: linear-gradient(135deg, #888 0%, #666 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 300;
}
.top-nav-logo .crumb {
    background: none;
    -webkit-text-fill-color: #2a2a2a;
    font-size: 0.85rem;
    font-weight: 400;
    letter-spacing: 0;
}
.top-nav-caps {
    display: flex;
    gap: 6px;
    align-items: center;
}
.cap-pill {
    background: transparent;
    border: 1px solid #222;
    color: #5a5a5a;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.68rem;
    letter-spacing: 0.3px;
    white-space: nowrap;
}
.top-nav-status {
    font-size: 0.72rem;
    color: #333;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Upload area ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: #0d0d0d !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #444 !important;
}

/* ── KPI cards ──────────────────────────────────────────────────────────── */
.kpi-card {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.kpi-label { color: #3a3a3a; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px; }
.kpi-value { color: #ffffff; font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
.kpi-sub   { color: #4a4a4a; font-size: 0.72rem; }

/* ── Chord badges ───────────────────────────────────────────────────────── */
.chord-badge {
    display: inline-block;
    background: #111111;
    border: 1px solid #2a2a2a;
    color: #cccccc;
    padding: 5px 13px;
    margin: 3px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    cursor: default;
}

/* ── Segment bar ────────────────────────────────────────────────────────── */
.seg-bar {
    border-left: 3px solid #2a2a2a;
    background: #0d0d0d;
    padding: 9px 14px;
    margin: 4px 0;
    border-radius: 0 8px 8px 0;
}

/* ── Stem card ──────────────────────────────────────────────────────────── */
.stem-card {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-radius: 12px;
    padding: 12px 16px 10px;
    margin-bottom: 10px;
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr { border-color: #1a1a1a; }

/* ── File list row ──────────────────────────────────────────────────────── */
.file-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #1e1e1e;
    background: #0d0d0d;
    margin-bottom: 2px;
    transition: border-color 0.15s;
}
.file-row.open  { border-color: #333333; background: #111111; }
.file-row-name  { flex:1; color:#cccccc; font-size:.88rem; font-weight:500;
                  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.file-row-meta  { color:#3a3a3a; font-size:.73rem; white-space:nowrap; }
.file-row-badge { color:#555555; font-size:.71rem; white-space:nowrap; font-variant-numeric:tabular-nums; }
.file-row-badge.ready { color:#707070; }

/* ── Instrument cards (Moises style) ─────────────────────────────────────── */
.sep-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: border-color .15s, background .15s;
    margin-bottom: 6px;
    min-height: 44px;
}
.sep-card.selected {
    background: #1c1c1c;
    border-color: #e0d0ff;
    box-shadow: 0 0 0 1px rgba(168,85,247,.15);
}
.sep-card.locked { opacity: .45; cursor: default; }
.sep-card-icon { font-size: 1.4rem; min-width: 28px; text-align: center; }
.sep-card-body { flex: 1; }
.sep-card-title { color: #e0e0e0; font-size: .93rem; font-weight: 600; }
.sep-card.selected .sep-card-title { color: #ffffff; }
.sep-card-sub   { color: #444; font-size: .73rem; margin-top: 2px; }
.sep-card-end   { color: #444; font-size: .85rem; }
.sep-card.selected .sep-card-end { color: #c084fc; }

/* Basic preset cards */
.preset-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 13px 16px;
    transition: all .15s;
    text-align: left;
    min-height: 58px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.preset-card.active {
    background: #1e1a2e;
    border-color: #7c3aed;
    box-shadow: 0 0 0 1px rgba(124,58,237,.3);
}
.preset-card-title { color: #888; font-size: .87rem; font-weight: 600; line-height: 1.2; }
.preset-card.active .preset-card-title { color: #e9d5ff; }
.preset-card-sub { color: #333; font-size: .7rem; margin-top: 3px; }
.preset-card.active .preset-card-sub { color: #7c3aed; }

/* Section labels */
.sep-section-label {
    color: #444;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 22px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sep-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1a1a;
}

/* ── Streamlit element tweaks ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: #0d0d0d; border-bottom: 1px solid #1e1e1e; }
.stTabs [data-baseweb="tab"] { color: #4a4a4a; }
.stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 2px solid #ffffff; }
button[data-testid="baseButton-primary"] {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
    height: 44px !important;
    max-height: 44px !important;
    font-size: .88rem !important;
}
button[data-testid="baseButton-primary"]:hover {
    background: #dddddd !important;
}
button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    color: #888888 !important;
    height: 28px !important;
    min-height: 0 !important;
    padding: 0 10px !important;
    font-size: .72rem !important;
    line-height: 1 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Utilidades de color ───────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Funciones de renderizado (análisis) ──────────────────────────────────────

def kpi(label: str, value: str, sub: str = ""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_waveform(results: dict):
    wf  = results['waveform']
    dur = results['duration']
    segs = results['structure']['segments']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=wf['times'], y=wf['amplitudes'],
        mode='lines',
        line=dict(color='#4fc3f7', width=0.7),
        fill='tozeroy',
        fillcolor='rgba(79,195,247,0.12)',
        name='Amplitud',
    ))

    for seg in segs:
        color = seg.get('color', '#9e9e9e')
        mid   = (seg['start'] + seg['end']) / 2
        amp   = max(abs(v) for v in wf['amplitudes']) * 0.88 if wf['amplitudes'] else 0.5

        fig.add_vline(x=seg['start'], line=dict(color=color, width=1.2, dash='dash'))
        fig.add_annotation(
            x=mid, y=amp,
            text=f"<b>{seg['label']}</b>",
            showarrow=False,
            font=dict(color=color, size=11),
            bgcolor='rgba(13,17,23,0.6)',
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )

    # Solo marcar cada 16 beats (evita cientos de vlines que hacen lento Plotly)
    beats = results['tempo']['beat_times']
    if beats:
        for bt in beats[::16]:
            fig.add_vline(x=bt, line=dict(color='rgba(255,255,255,0.05)', width=0.5))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=300,
        margin=dict(t=30, b=40, l=50, r=20),
        xaxis=dict(title='Tiempo (s)', range=[0, dur], gridcolor='#1a2233'),
        yaxis=dict(title='Amplitud', gridcolor='#1a2233'),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_spectrogram(results: dict):
    spec   = results['spectrogram']
    chroma = results['chromagram']
    dur    = results['duration']

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Espectrograma Mel (dB)', 'Cromograma'),
        vertical_spacing=0.14,
        row_heights=[0.55, 0.45],
    )

    spec_arr   = np.array(spec['spectrogram'])
    chroma_arr = np.array(chroma['chroma'])

    fig.add_trace(go.Heatmap(
        z=spec_arr, x=spec['times'],
        colorscale='Magma', showscale=True,
        colorbar=dict(x=1.01, len=0.48, y=0.78, title='dB'),
        name='Mel Spec',
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=chroma_arr, x=chroma['times'],
        y=chroma['notes'],
        colorscale='Viridis', showscale=True,
        colorbar=dict(x=1.01, len=0.42, y=0.22, title='Energía'),
        name='Chroma',
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=560,
        margin=dict(t=50, b=40, l=60, r=80),
    )
    fig.update_xaxes(title_text='Tiempo (s)', row=2, col=1, gridcolor='#1a2233')
    fig.update_yaxes(title_text='Frecuencia (Hz)', row=1, col=1)
    fig.update_yaxes(title_text='Nota', row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_chords(results: dict):
    chords   = results['chords']
    key_info = results['key']

    col_bar, col_prog = st.columns([1, 1.6])

    with col_bar:
        st.subheader("📊 Acordes más frecuentes")
        if chords['top_chords']:
            names  = [c[0] for c in chords['top_chords'][:10]]
            counts = [c[1] for c in chords['top_chords'][:10]]
            fig = go.Figure(go.Bar(
                x=counts, y=names,
                orientation='h',
                marker=dict(color=counts, colorscale='Blues', reversescale=True),
                text=counts, textposition='outside',
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                height=380,
                margin=dict(t=10, b=40, l=90, r=40),
                xaxis=dict(title='Frecuencia', gridcolor='#1a2233'),
                yaxis=dict(gridcolor='#1a2233'),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_prog:
        st.subheader("🎸 Progresión detectada")
        changes = chords['chord_changes'][:24]

        if changes:
            import streamlit.components.v1 as components

            badges_html = ""
            for i, ch in enumerate(changes):
                sep = '<div style="width:100%;height:6px;"></div>' if (i > 0 and i % 6 == 0) else ""
                badges_html += sep + f"""
                <div title="En {ch['time_fmt']}" style="
                    background:#1e3a5f;border:1px solid #2979b8;border-radius:8px;
                    padding:7px 12px;text-align:center;min-width:62px;cursor:default;">
                  <div style="color:#607d8b;font-size:10px;">{ch['time_fmt']}</div>
                  <div style="color:#90caf9;font-size:16px;font-weight:700;line-height:1.3;">{ch['chord']}</div>
                </div>"""

            full_html = f"""
            <div style="
                display:flex;flex-wrap:wrap;gap:7px;padding:8px 2px;
                font-family:sans-serif;background:transparent;">
              {badges_html}
            </div>"""

            n_rows  = max(1, (len(changes) + 5) // 6)
            height  = n_rows * 72 + 30
            components.html(full_html, height=height, scrolling=False)

    st.subheader("🎹 Distribución de notas en la tonalidad detectada")

    root   = key_info['root']
    mode   = key_info['mode']
    ivs    = [0, 2, 4, 5, 7, 9, 11] if mode == 'major' else [0, 2, 3, 5, 7, 8, 10]
    scale_notes = {(root + i) % 12 for i in ivs}

    from audio_analysis import NOTE_NAMES_EN
    chroma_mean = key_info.get('chroma_mean', [0.0] * 12)
    colors_bar = [
        '#f9a825' if i == root
        else '#42a5f5' if i in scale_notes
        else '#37474f'
        for i in range(12)
    ]

    fig2 = go.Figure(go.Bar(
        x=NOTE_NAMES_EN, y=chroma_mean,
        marker=dict(color=colors_bar),
        text=[f"{'★' if i == root else ''}" for i in range(12)],
        textposition='outside',
    ))
    fig2.add_annotation(
        x=0.5, y=1.06, xref='paper', yref='paper', showarrow=False,
        text='🟡 Tónica &nbsp;&nbsp; 🔵 Notas de escala &nbsp;&nbsp; ⬛ Fuera de escala',
        font=dict(size=11, color='#90a4ae'),
    )
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=270,
        margin=dict(t=40, b=40, l=50, r=20),
        xaxis=dict(title='Nota', gridcolor='#1a2233'),
        yaxis=dict(title='Energía promedio', gridcolor='#1a2233'),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_structure(results: dict):
    segs = results['structure']['segments']
    dur  = results['duration']

    if not segs:
        st.warning("No se pudo segmentar la estructura. Prueba con un archivo más largo.")
        return

    fig = go.Figure()
    for seg in segs:
        color = seg.get('color', '#9e9e9e')
        fig.add_shape(
            type='rect',
            x0=seg['start'], x1=seg['end'], y0=0, y1=1,
            fillcolor=color, opacity=0.75,
            line=dict(color='#0d1117', width=2),
        )
        fig.add_annotation(
            x=(seg['start'] + seg['end']) / 2, y=0.5,
            text=f"<b>{seg['label']}</b><br><span style='font-size:10px'>{seg['start_fmt']}–{seg['end_fmt']}</span>",
            showarrow=False,
            font=dict(color='white', size=11),
        )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=140,
        margin=dict(t=20, b=50, l=20, r=20),
        xaxis=dict(title='Tiempo (segundos)', range=[0, dur], gridcolor='#1a2233'),
        yaxis=dict(showticklabels=False, range=[0, 1]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    energies = [s['energy'] for s in segs]
    names    = [f"{s['label']}\n{s['start_fmt']}" for s in segs]
    colors_e = [s.get('color', '#9e9e9e') for s in segs]

    fig2 = go.Figure(go.Bar(
        x=names, y=energies,
        marker=dict(color=colors_e),
        text=[f"{e:.0%}" for e in energies],
        textposition='outside',
    ))
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=280,
        margin=dict(t=20, b=60, l=60, r=20),
        xaxis=dict(title='Segmento', gridcolor='#1a2233'),
        yaxis=dict(title='Energía relativa', range=[0, 1.15], gridcolor='#1a2233'),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Detalle de segmentos")
    for seg in segs:
        color  = seg.get('color', '#9e9e9e')
        filled = int(seg['energy'] * 20)
        bar    = '█' * filled + '░' * (20 - filled)
        st.markdown(f"""
        <div class="seg-bar" style="border-color:{color};">
            <span style="color:{color};font-weight:700;font-size:15px;">{seg['label']}</span>
            &nbsp;&nbsp;
            <span style="color:#90a4ae;font-size:13px;">
                {seg['start_fmt']} → {seg['end_fmt']} &nbsp;|&nbsp; {seg['duration']:.1f} s
            </span>
            <br>
            <span style="color:#37474f;font-family:monospace;font-size:11px;">
                Energía: <span style="color:{color};">{bar}</span> {seg['energy']:.0%}
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_processing_log(log: list):
    if not log:
        st.markdown(
            "<div style='color:#2a2a2a;font-size:.85rem;padding:2rem;text-align:center'>"
            "Sin log de procesamiento (archivo procesado antes de esta versión)</div>",
            unsafe_allow_html=True,
        )
        return

    COLORS = {'ok': '#4ade80', 'info': '#818cf8', 'warn': '#fb923c', 'error': '#f87171'}
    ICONS  = {'ok': '✓', 'info': '·', 'warn': '!', 'error': '✕'}

    # Cabecera con tiempo total
    last = log[-1]
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:14px'>"
        f"<span style='color:#666;font-size:.72rem;text-transform:uppercase;"
        f"letter-spacing:1px'>Log de procesamiento</span>"
        f"<span style='color:#4ade80;font-size:.78rem;font-family:monospace'>"
        f"Tiempo total: {last['elapsed']} s</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    rows_html = ""
    for entry in log:
        c     = COLORS.get(entry['level'], '#888')
        icon  = ICONS.get(entry['level'], '·')
        msg   = entry['msg'].replace('<', '&lt;').replace('>', '&gt;')
        rows_html += (
            f"<tr>"
            f"<td style='color:#333;font-size:.72rem;white-space:nowrap;padding:5px 10px 5px 0'>"
            f"{entry['ts']}</td>"
            f"<td style='color:#2a2a2a;font-size:.72rem;white-space:nowrap;padding:5px 10px 5px 0;"
            f"font-variant-numeric:tabular-nums'>+{entry['elapsed']}s</td>"
            f"<td style='padding:5px 10px 5px 0;text-align:center'>"
            f"<span style='color:{c};font-weight:700;font-size:.8rem'>{icon}</span></td>"
            f"<td style='color:#555;font-size:.78rem;white-space:nowrap;padding:5px 12px 5px 0;"
            f"font-weight:600'>{entry['step']}</td>"
            f"<td style='color:#888;font-size:.76rem;padding:5px 0;font-family:monospace;"
            f"line-height:1.5'>{msg}</td>"
            f"</tr>"
        )

    st.markdown(
        f"<div style='background:#060606;border:1px solid #141414;border-radius:10px;"
        f"padding:14px 18px;overflow-x:auto'>"
        f"<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr>"
        f"<th style='color:#222;font-size:.65rem;text-transform:uppercase;letter-spacing:.8px;"
        f"text-align:left;padding:0 10px 8px 0;border-bottom:1px solid #0f0f0f'>Hora</th>"
        f"<th style='color:#222;font-size:.65rem;text-transform:uppercase;letter-spacing:.8px;"
        f"text-align:left;padding:0 10px 8px 0;border-bottom:1px solid #0f0f0f'>Elapsed</th>"
        f"<th style='color:#222;font-size:.65rem;text-transform:uppercase;letter-spacing:.8px;"
        f"text-align:center;padding:0 10px 8px 0;border-bottom:1px solid #0f0f0f'>Nv</th>"
        f"<th style='color:#222;font-size:.65rem;text-transform:uppercase;letter-spacing:.8px;"
        f"text-align:left;padding:0 10px 8px 0;border-bottom:1px solid #0f0f0f'>Paso</th>"
        f"<th style='color:#222;font-size:.65rem;text-transform:uppercase;letter-spacing:.8px;"
        f"text-align:left;padding:0 0 8px 0;border-bottom:1px solid #0f0f0f'>Detalle</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


def render_summary(results: dict):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎵 Información musical")
        items = {
            "Tonalidad":          results['key']['key'],
            "Tonalidad (inglés)": results['key']['key_en'],
            "Confianza":          f"{results['key']['confidence']} %",
            "Tempo":              f"{results['tempo']['bpm']} BPM",
            "Compás":             results['time_signature']['time_sig'],
            "Tipo de compás":     results['time_signature']['description'],
            "Escala":             results['scale']['scale_name'],
            "Modo":               results['scale']['mode_name'],
            "Fórmula de escala":  results['scale']['formula'],
        }
        for k, v in items.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        st.subheader("🎼 Notas de la escala")
        notes_html = ' &nbsp;—&nbsp; '.join(
            f"<span style='color:#4fc3f7'><b>{n}</b></span>"
            for n in results['scale']['notes']
        )
        st.markdown(notes_html, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Estadísticas del audio")
        dur = results['duration']
        items2 = {
            "Duración":               f"{int(dur//60)}:{int(dur%60):02d} min",
            "Frecuencia de muestreo": f"{results['sample_rate']} Hz",
            "Nº de segmentos":        str(results['structure']['n_segments']),
            "Cambios de acorde":      str(len(results['chords']['chord_changes'])),
        }
        if results['chords']['top_chords']:
            best = results['chords']['top_chords'][0]
            items2["Acorde más frecuente"] = f"{best[0]}  ({best[1]} veces)"

        for k, v in items2.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        st.subheader("🏗️ Estructura de la canción")
        icons = {
            'Intro':'🟠','Verso':'🔵','Coro':'🟢',
            'Puente':'🟡','Solo':'🩷','Outro':'🟣','Cuerpo':'🔵',
        }
        for seg in results['structure']['segments']:
            icon = icons.get(seg['label'], '⚫')
            st.markdown(
                f"{icon} **{seg['label']}** &nbsp; {seg['start_fmt']} → {seg['end_fmt']}"
                f" &nbsp; _{seg['duration']:.0f} s_"
            )


# ── Reproductor integrado (chord_player eliminado — ya incluido en integrated_player) ─

def _compress_to_mp3(wav_bytes: bytes, bitrate: str = '96k') -> tuple[bytes, str]:
    """Comprime WAV → MP3 para reducir tamaño. Devuelve (data, mime_type)."""
    try:
        import io as _io
        from pydub import AudioSegment
        from audio_analysis import MusicAnalyzer
        ffbin = MusicAnalyzer._find_ffmpeg()
        if ffbin:
            import pydub as _pd
            _pd.AudioSegment.converter = ffbin
            _pd.AudioSegment.ffmpeg    = ffbin
            seg = AudioSegment.from_wav(_io.BytesIO(wav_bytes))
            out = _io.BytesIO()
            seg.export(out, format='mp3', bitrate=bitrate)
            return out.getvalue(), 'audio/mpeg'
    except Exception:
        pass
    return wav_bytes, 'audio/wav'


@st.cache_data(show_spinner=False)
def _mp3_for_download(wav_bytes: bytes, bitrate: str = '192k') -> tuple[bytes, str]:
    return _compress_to_mp3(wav_bytes, bitrate)


# ── Reproductor integrado: pistas + acordes + piano ──────────────────────────

def render_integrated_player(
    results: dict,
    audio_bytes: bytes,
    stems: dict | None = None,
):
    """
    UN solo componente Web Audio API con:
      • Mezcla de pistas (M/S/Volumen) si stems disponible
      • Metrónomo inteligente (0.5×/1×/2×)
      • Timeline de acordes (clic para saltar)
      • Piano iluminado con las notas del acorde actual
    """
    import base64, json
    import streamlit.components.v1 as components

    # ── Datos musicales ───────────────────────────────────────────────────
    chords   = results['chords']['chord_changes']
    duration = float(results['duration'])
    bpm         = float(results['tempo']['bpm'])
    beat_offset = float(results['tempo'].get('beat_offset', 0.0))
    key_name    = results['key']['key']
    time_sig    = int(results['time_signature']['beats_per_measure'])
    pps      = min(80, max(35, 2400 / max(duration, 1)))
    total_w  = int(duration * pps) + 60

    # ── Preparar audio ────────────────────────────────────────────────────
    has_stems = bool(stems)
    stems_list: list[dict] = []

    if has_stems:
        for key, stem in stems.items():
            mono = stem['audio_mono']
            step = max(1, len(mono) // 1600)
            wf   = [round(float(v), 4) for v in mono[::step]]
            data, mime = _compress_to_mp3(stem['wav_bytes'])
            stems_list.append({
                'key': key, 'name': stem['name_es'],
                'icon': stem['icon'], 'color': stem['color'],
                'wf': wf, 'b64': base64.b64encode(data).decode(), 'mime': mime,
            })
        full_b64 = full_mime = ''
    else:
        data, mime = _compress_to_mp3(audio_bytes)
        full_b64, full_mime = base64.b64encode(data).decode(), mime

    stems_json  = json.dumps(stems_list)
    chords_json = json.dumps(chords)

    n_rows = (len(stems_list) + 1) if has_stems else 0   # +1 metrónomo
    comp_h = 54 + n_rows * 90 + 10 + 68 + 158 + 16

    # ══════════════════════════════════════════════════════════════════════
    # CSS
    # ══════════════════════════════════════════════════════════════════════
    css = r"""
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0a;color:#c0c0c0;font-family:'Segoe UI',system-ui,sans-serif;
     user-select:none;overflow:hidden}
/* ── Transport ─────────────────────────────────────────────────────────────── */
#tp{display:flex;align-items:center;gap:8px;background:#0d0d0d;
    border-bottom:1px solid #1c1c1c;padding:8px 14px;height:54px}
.tb{background:#161616;border:1px solid #252525;color:#888;border-radius:6px;
    width:34px;height:34px;font-size:14px;cursor:pointer;transition:background .15s}
.tb:hover{background:#222}.tb:disabled{opacity:.3;cursor:default}
.tb.playing{background:#fff;border-color:#fff;color:#000}
#sk{flex:1;height:4px;background:#1c1c1c;border-radius:2px;cursor:pointer;position:relative}
#sf{height:100%;background:#fff;border-radius:2px;pointer-events:none;width:0%}
#sd{position:absolute;top:-6px;width:16px;height:16px;border-radius:50%;background:#fff;
    box-shadow:0 0 6px rgba(255,255,255,.4);margin-left:-8px;pointer-events:none;left:0%}
.tt{font-size:.78rem;color:#444;font-variant-numeric:tabular-nums;white-space:nowrap}
#spd{background:#111;border:1px solid #222;color:#666;border-radius:5px;
     padding:3px 8px;font-size:.75rem;cursor:pointer}
#stlbl{font-size:.68rem;color:#333;min-width:55px}
#klbl{font-size:.78rem;color:#888;font-weight:600;letter-spacing:.5px}
/* ── Mixer ──────────────────────────────────────────────────────────────────── */
#bd{display:flex}
#cc{width:190px;min-width:190px;border-right:1px solid #181818}
.ctl{height:90px;border-bottom:1px solid #151515;padding:7px 10px;background:#0a0a0a;
     display:flex;flex-direction:column;justify-content:space-between}
.ct{display:flex;align-items:center;gap:5px}
.bm,.bs{width:21px;height:21px;font-size:.6rem;font-weight:700;border-radius:4px;
        border:1px solid #222;cursor:pointer;background:transparent;
        color:#333;transition:all .12s;line-height:1}
.bm.on{background:#fff;border-color:#fff;color:#000}
.bs.on{background:#888;border-color:#888;color:#000}
.ci{font-size:.8rem;font-weight:600;color:#444;min-width:18px;text-align:center}
.cn{font-size:.75rem;font-weight:600;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vr-row{display:flex;align-items:center;gap:5px}
input[type=range].vr{-webkit-appearance:none;flex:1;height:2px;border-radius:1px;
   outline:none;cursor:pointer;background:#1e1e1e}
input[type=range].vr::-webkit-slider-thumb{-webkit-appearance:none;
   width:12px;height:12px;border-radius:50%;cursor:pointer;background:var(--tc,#888)}
.vp{font-size:.58rem;color:#333;min-width:26px;text-align:right}
#wc{flex:1;position:relative;overflow:hidden}
#ph{position:absolute;top:0;bottom:0;width:1px;background:#fff;z-index:30;
    pointer-events:none;left:0;box-shadow:0 0 6px rgba(255,255,255,.5);display:none}
.wr{height:90px;border-bottom:1px solid #131313;cursor:crosshair;
    position:relative;overflow:hidden;background:#060606}
.wr canvas{position:absolute;top:0;left:0;width:100%;height:100%}
.wr.muted canvas{opacity:.1;filter:grayscale(1)}
.mpills{position:absolute;top:5px;left:6px;display:flex;gap:4px;z-index:10}
.mpill{background:#111;border:1px solid #1e1e1e;color:#444;border-radius:10px;
       padding:2px 7px;font-size:.6rem;cursor:pointer}
.mpill.sel{background:#1e1e1e;border-color:#333;color:#aaa}
/* ── Piano section ──────────────────────────────────────────────────────────── */
#ps{background:#0a0812;border-top:1px solid #1e1830;
    padding:10px 16px 8px;display:flex;align-items:flex-start;gap:18px}
#ci{display:flex;flex-direction:column;justify-content:center;min-width:160px;gap:3px}
#cm{font-size:3rem;font-weight:900;color:#2a1f3d;
    transition:color .25s,text-shadow .25s;line-height:1;letter-spacing:-1px}
#cm.active{color:#c084fc;text-shadow:0 0 30px rgba(192,132,252,.4)}
.ci-lbl{font-size:.53rem;color:#2a1f3d;text-transform:uppercase;
        letter-spacing:1.2px;margin-top:6px}
#cn{font-size:.95rem;font-weight:700;color:#3d2a5a}
#nn{font-size:.78rem;font-weight:600;color:#7c5fa0;min-height:17px;margin-top:2px}
#nn .en{color:#4a3060;font-size:.68rem}
#pw{flex:1;overflow-x:auto}
canvas#pv{display:block}
/* ── Chord timeline ─────────────────────────────────────────────────────────── */
#cs{background:#07060e;border-top:1px solid #1a1528;padding:3px 0 0}
#to{overflow-x:auto;overflow-y:hidden;padding:4px 6px 20px;height:68px}
#ti{position:relative;height:44px}
.cb{position:absolute;top:1px;height:42px;border-radius:6px;display:flex;
    flex-direction:column;justify-content:center;align-items:center;
    overflow:hidden;border:1px solid rgba(168,85,247,.08);
    cursor:pointer;transition:border-color .1s,background .1s}
.cb:hover{border-color:rgba(168,85,247,.3)!important;background:rgba(168,85,247,.05)!important}
.cb.active{border:1px solid rgba(192,132,252,.7)!important;
           background:rgba(124,58,237,.15)!important;
           box-shadow:0 0 12px rgba(168,85,247,.2);z-index:6}
.cb-n{font-size:11px;font-weight:800;white-space:nowrap;
      overflow:hidden;max-width:94%;text-overflow:ellipsis;color:#b48dde}
.cb.active .cb-n{color:#e9d5ff}
.cb-t{font-size:8px;color:rgba(168,85,247,.25);margin-top:1px}
#tlph{position:absolute;top:0;bottom:0;width:1px;background:#a855f7;
      z-index:10;pointer-events:none;left:0;box-shadow:0 0 6px #a855f7}
"""

    # ══════════════════════════════════════════════════════════════════════
    # JavaScript — datos Python → JS
    # ══════════════════════════════════════════════════════════════════════
    js_data = (
        f"const STEMS={stems_json};\n"
        f"const CHORDS={chords_json};\n"
        f"const HAS_STEMS={'true' if has_stems else 'false'};\n"
        f"const FULL_B64='{full_b64}';\n"
        f"const FULL_MIME='{full_mime}';\n"
        f"const BPM={bpm:.3f};\n"
        f"const BEAT_OFFSET={beat_offset:.3f};\n"
        f"const DURATION={duration:.3f};\n"
        f"const TIMESIG={time_sig};\n"
        f"const KEY_NAME='{key_name}';\n"
        f"const PPS={pps:.2f};\n"
        f"const TOTAL_W={total_w};\n"
    )

    # ══════════════════════════════════════════════════════════════════════
    # JavaScript — lógica (sin variables Python, sin f-string)
    # ══════════════════════════════════════════════════════════════════════
    js_logic = r"""
// ── Audio engine ──────────────────────────────────────────────────────────────
const AC=new (window.AudioContext||window.webkitAudioContext)();
const DEST=AC.createGain();DEST.connect(AC.destination);
const gainNodes={},buffers={},volumes={};
const muted=new Set(),soloed=new Set();
let metroGain=null,metroBuf=null,metroMult=1,singleBuf=null;
let isPlaying=false,seekPos=0,ctxStart=0,playRate=1.0;
let activeSrcs=[],metroSrcs=[],rafId=null;

function fmt(s){return Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0')}
function getCurTime(){return isPlaying?Math.min(seekPos+(AC.currentTime-ctxStart)*playRate,DURATION):seekPos}
function resumeCtx(){if(AC.state==='suspended')AC.resume()}

// ── Gains ─────────────────────────────────────────────────────────────────────
function buildGains(){
    if(HAS_STEMS){
        STEMS.forEach(st=>{const g=AC.createGain();g.connect(DEST);gainNodes[st.key]=g;volumes[st.key]=1});
        metroGain=AC.createGain();metroGain.connect(DEST);volumes['metro']=0.7;
    }
}
function applyGains(){
    if(!HAS_STEMS)return;
    const hs=soloed.size>0;
    STEMS.forEach(st=>{const h=!muted.has(st.key)&&(!hs||soloed.has(st.key));gainNodes[st.key].gain.value=h?(volumes[st.key]||1):0});
    metroGain.gain.value=!muted.has('metro')&&(!hs||soloed.has('metro'))?(volumes['metro']||0.7):0;
}
function toggleMute(k){
    muted.has(k)?muted.delete(k):muted.add(k);
    document.getElementById('m-'+k)?.classList.toggle('on',muted.has(k));
    document.getElementById('wr-'+k)?.classList.toggle('muted',muted.has(k));
    applyGains();
}
function toggleSolo(k){
    soloed.has(k)?soloed.delete(k):soloed.add(k);
    document.getElementById('s-'+k)?.classList.toggle('on',soloed.has(k));
    applyGains();
}
function setVolume(k,v){volumes[k]=parseFloat(v)/100;document.getElementById('vl-'+k).textContent=v+'%';applyGains()}
function setMetroMult(m){
    metroMult=m;metroSrcs.forEach(s=>{try{s.stop()}catch(e){}});metroSrcs=[];
    if(isPlaying)scheduleMetro(getCurTime());
    document.querySelectorAll('.mpill').forEach(p=>p.classList.toggle('sel',parseFloat(p.dataset.m)===m));
}
function setRate(r){playRate=parseFloat(r);if(isPlaying){pauseAll();startAll()}}

// ── Load audio ────────────────────────────────────────────────────────────────
async function loadAll(){
    setSt('Cargando audio…');
    async function dec(b64){const bin=atob(b64),u8=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u8[i]=bin.charCodeAt(i);return AC.decodeAudioData(u8.buffer)}
    if(HAS_STEMS){
        await Promise.all(STEMS.map(async st=>{try{buffers[st.key]=await dec(st.b64)}catch(e){console.error('dec',st.key,e)}}));
    }else if(FULL_B64){
        try{singleBuf=await dec(FULL_B64)}catch(e){console.error('dec single',e)}
    }
    setSt('✅ Listo');
    document.getElementById('btn-play').disabled=false;
    setTimeout(()=>{drawAll();buildTL()},180);
}

// ── Metronome ─────────────────────────────────────────────────────────────────
function makeMetroBuf(){
    const sr=AC.sampleRate,buf=AC.createBuffer(1,Math.ceil(sr*.055),sr),ch=buf.getChannelData(0);
    for(let i=0;i<ch.length;i++){const t=i/sr;ch[i]=Math.sin(2*Math.PI*1100*t)*Math.exp(-t*55)}
    return buf;
}
function scheduleMetro(from){
    if(!metroBuf)metroBuf=makeMetroBuf();
    const iv=60/(BPM*metroMult);
    // El primer beat real de la canción está en BEAT_OFFSET, no en t=0.
    // Calculamos qué beat cae justo después de `from`.
    const firstBeat=BEAT_OFFSET%iv;  // offset dentro del primer compás
    const relFrom=from-firstBeat;
    const sb=Math.ceil(relFrom/iv);
    const eb=Math.ceil((DURATION-firstBeat)/iv);
    for(let b=sb;b<eb;b++){
        const bt=firstBeat+b*iv;
        if(bt<0||bt>DURATION)continue;
        const st=AC.currentTime+(bt-from)/playRate;
        if(st<AC.currentTime+.01)continue;
        const src=AC.createBufferSource();src.buffer=metroBuf;src.playbackRate.value=playRate;
        const env=AC.createGain();
        // Acento en el tiempo 1 de cada compás (relativo al primer beat real)
        const beatInBar=Math.round((bt-BEAT_OFFSET)/iv)%Math.round(TIMESIG*metroMult);
        const isA=beatInBar===0;
        const vol=(volumes['metro']||.7)*(isA?1:.5);
        env.gain.setValueAtTime(vol,st);env.gain.exponentialRampToValueAtTime(.001,st+.07);
        src.connect(env);env.connect(metroGain);src.start(st);metroSrcs.push(src);
    }
}

// ── Playback ──────────────────────────────────────────────────────────────────
function startAll(){
    resumeCtx();ctxStart=AC.currentTime;
    if(HAS_STEMS){
        STEMS.forEach(st=>{if(!buffers[st.key])return;const src=AC.createBufferSource();src.buffer=buffers[st.key];src.playbackRate.value=playRate;src.connect(gainNodes[st.key]);src.start(0,seekPos);activeSrcs.push(src)});
        scheduleMetro(seekPos);
    }else{
        if(!singleBuf)return;const src=AC.createBufferSource();src.buffer=singleBuf;src.playbackRate.value=playRate;src.connect(DEST);src.start(0,seekPos);activeSrcs.push(src);
    }
    isPlaying=true;updTp();rafId=requestAnimationFrame(tick);
}
function pauseAll(){
    activeSrcs.forEach(s=>{try{s.stop()}catch(e){}});metroSrcs.forEach(s=>{try{s.stop()}catch(e){}});
    activeSrcs=[];metroSrcs=[];seekPos=Math.min(getCurTime(),DURATION);
    isPlaying=false;cancelAnimationFrame(rafId);updTp();
}
function stopAll(){pauseAll();seekPos=0;updPH(0);updTime(0)}
function togglePlay(){isPlaying?pauseAll():startAll()}
function seekTo(t){const was=isPlaying;pauseAll();seekPos=Math.max(0,Math.min(t,DURATION));was?startAll():(updPH(seekPos/DURATION),updTime(seekPos))}

// ── Chord helpers ─────────────────────────────────────────────────────────────
const ES_MAP={Sol:7,Si:11,Fa:5,Re:2,Mi:4,La:9,Do:0};
const ES_KEYS=['Sol','Si','Fa','Re','Mi','La','Do'];
const IVS={'M7':[0,4,7,11],'m7':[0,3,7,10],'dim':[0,3,6],'aug':[0,4,8],'sus4':[0,5,7],'sus2':[0,2,7],'7':[0,4,7,10],'m':[0,3,7],'':[0,4,7]};
const ESN=['Do','Do#','Re','Re#','Mi','Fa','Fa#','Sol','Sol#','La','La#','Si'];
const ENN=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function c2n(name){
    if(!name||name==='N/C'||name==='—')return[];
    let root=-1,rest=name;
    for(const n of ES_KEYS){if(name.startsWith(n)){root=ES_MAP[n];rest=name.slice(n.length);break}}
    if(root<0)return[];if(rest[0]==='#'){root=(root+1)%12;rest=rest.slice(1)}
    return(IVS[rest]||IVS['']).map(iv=>(root+iv)%12);
}
function cCol(n){
    // Mono palette: all chord types use similar dark-grey tones, only slight variation
    if(n.includes('dim'))return{bg:'rgba(255,255,255,.04)',txt:'#666',brd:'rgba(255,255,255,.08)'};
    if(n.includes('aug'))return{bg:'rgba(255,255,255,.06)',txt:'#777',brd:'rgba(255,255,255,.1)'};
    if(n.includes('sus'))return{bg:'rgba(255,255,255,.05)',txt:'#666',brd:'rgba(255,255,255,.09)'};
    if(n.includes('m7')||n.includes('M7'))return{bg:'rgba(255,255,255,.07)',txt:'#888',brd:'rgba(255,255,255,.12)'};
    if(n.endsWith('7'))return{bg:'rgba(255,255,255,.06)',txt:'#777',brd:'rgba(255,255,255,.1)'};
    if(n.includes('m'))return{bg:'rgba(255,255,255,.05)',txt:'#666',brd:'rgba(255,255,255,.09)'};
    return{bg:'rgba(255,255,255,.08)',txt:'#999',brd:'rgba(255,255,255,.14)'};
}

// ── Chord timeline ────────────────────────────────────────────────────────────
function buildTL(){
    const ti=document.getElementById('ti');if(!ti)return;
    ti.style.width=TOTAL_W+'px';
    CHORDS.forEach((ch,i)=>{
        const nxt=i+1<CHORDS.length?CHORDS[i+1].time:DURATION;
        const x=ch.time*PPS,w=Math.max(4,(nxt-ch.time)*PPS-2),c=cCol(ch.chord);
        const d=document.createElement('div');d.className='cb';d.dataset.idx=i;
        d.style.cssText='left:'+x+'px;width:'+w+'px;background:'+c.bg+';border-color:'+c.brd;
        if(w>22)d.innerHTML='<span class="cb-n" style="color:'+c.txt+'">'+ch.chord+'</span><span class="cb-t">'+ch.time_fmt+'</span>';
        d.addEventListener('click',()=>seekTo(ch.time));ti.appendChild(d);
    });
    const step=DURATION<=60?5:DURATION<=180?10:30;
    for(let t=0;t<=DURATION;t+=step){
        const tk=document.createElement('div');
        tk.style.cssText='position:absolute;bottom:-16px;left:'+(t*PPS)+'px;transform:translateX(-50%)';
        tk.innerHTML='<div style="width:1px;height:4px;background:rgba(255,255,255,.12);margin:0 auto"></div><div style="font-size:7px;color:#37474f;text-align:center;white-space:nowrap">'+fmt(t)+'</div>';
        ti.appendChild(tk);
    }
}

// ── Piano ─────────────────────────────────────────────────────────────────────
const KM=[{b:false,w:0},{b:true,w:0},{b:false,w:1},{b:true,w:1},{b:false,w:2},{b:false,w:3},{b:true,w:3},{b:false,w:4},{b:true,w:4},{b:false,w:5},{b:true,w:5},{b:false,w:6}];
const OCT=2,WW=32,WH=86,BW=19,BH=54;
function drawPiano(hi){
    const cv=document.getElementById('pv');if(!cv)return;
    const ctx=cv.getContext('2d');cv.width=7*OCT*WW+1;cv.height=WH+18;
    function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.arcTo(x+w,y,x+w,y+r,r);ctx.lineTo(x+w,y+h-r);ctx.arcTo(x+w,y+h,x+w-r,y+h,r);ctx.lineTo(x+r,y+h);ctx.arcTo(x,y+h,x,y+h-r,r);ctx.lineTo(x,y+r);ctx.arcTo(x,y,x+r,y,r);ctx.closePath()}
    // White keys
    for(let o=0;o<OCT;o++)for(let s=0;s<12;s++){
        const k=KM[s];if(k.b)continue;const x=(o*7+k.w)*WW,on=hi.includes(s);
        ctx.fillStyle=on?'#e8e8e8':'#1e1e1e';ctx.strokeStyle=on?'#aaa':'#2a2a2a';ctx.lineWidth=1;
        rr(x+.5,.5,WW-1,WH-1,3);ctx.fill();ctx.stroke();
        ctx.fillStyle=on?'#000':'#3a3a3a';ctx.font='8px system-ui';ctx.textAlign='center';
        ctx.fillText(ENN[s],x+WW/2,WH-6);
        if(on){ctx.fillStyle='#000';ctx.font='bold 8px system-ui';ctx.fillText(ESN[s],x+WW/2,WH-16)}
    }
    // Black keys
    for(let o=0;o<OCT;o++)for(let s=0;s<12;s++){
        const k=KM[s];if(!k.b)continue;const x=(o*7+k.w)*WW+WW-BW/2,on=hi.includes(s);
        ctx.fillStyle=on?'#cccccc':'#0a0a0a';ctx.strokeStyle=on?'#888':'#1a1a1a';ctx.lineWidth=1;
        rr(x+.5,.5,BW-1,BH-1,3);ctx.fill();ctx.stroke();
        if(on){ctx.fillStyle='#000';ctx.font='bold 8px system-ui';ctx.textAlign='center';ctx.fillText(ENN[s],x+BW/2,BH-8)}
    }
    // Octave separator
    ctx.strokeStyle='rgba(255,255,255,.06)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(7*WW,0);ctx.lineTo(7*WW,WH);ctx.stroke();
}

// ── Waveforms ─────────────────────────────────────────────────────────────────
function drawWF(id,wf,color){
    const cv=document.getElementById(id);if(!cv)return;
    const dpr=window.devicePixelRatio||1;cv.width=cv.offsetWidth*dpr;cv.height=cv.offsetHeight*dpr;
    const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
    const W=cv.offsetWidth,H=cv.offsetHeight,mid=H/2;
    ctx.fillStyle='#060606';ctx.fillRect(0,0,W,H);
    const n=wf.length,bw=Math.max(1,W/n),mx=Math.max(...wf.map(Math.abs))||1;
    const hx=color.replace('#',''),r=parseInt(hx.slice(0,2),16),g=parseInt(hx.slice(2,4),16),b=parseInt(hx.slice(4,6),16);
    ctx.fillStyle=`rgba(${r},${g},${b},0.75)`;
    for(let i=0;i<n;i++){const a=(Math.abs(wf[i])/mx)*mid*.82;ctx.fillRect(i*bw,mid-a,bw*.7,a*2)}
}
function drawMetroWF(id){
    const cv=document.getElementById(id);if(!cv)return;
    const dpr=window.devicePixelRatio||1;cv.width=cv.offsetWidth*dpr;cv.height=cv.offsetHeight*dpr;
    const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
    const W=cv.offsetWidth,H=cv.offsetHeight;ctx.fillStyle='#060606';ctx.fillRect(0,0,W,H);
    const iv=60/BPM,tot=Math.floor(DURATION/iv);
    for(let b=0;b<tot;b++){const x=(b*iv/DURATION)*W,isA=b%TIMESIG===0,bH=isA?H*.75:H*.38;ctx.fillStyle=isA?'rgba(255,255,255,.55)':'rgba(255,255,255,.18)';ctx.fillRect(x,(H-bH)/2,Math.max(2,W/tot*.45),bH)}
}
function drawAll(){
    if(HAS_STEMS){STEMS.forEach(st=>drawWF('wf-'+st.key,st.wf,st.color));drawMetroWF('wf-metro')}
    drawPiano([]);
}

// ── UI updaters ───────────────────────────────────────────────────────────────
function updTp(){const btn=document.getElementById('btn-play');if(btn){btn.textContent=isPlaying?'⏸':'▶';btn.classList.toggle('playing',isPlaying)}}
function setSt(m){const e=document.getElementById('stlbl');if(e)e.textContent=m}
function updTime(t){const e=document.getElementById('tc');if(e)e.textContent=fmt(t)}
function updPH(frac){
    // Stem playhead
    const ph=document.getElementById('ph'),wc=document.getElementById('wc');
    if(ph&&wc){ph.style.left=(frac*wc.clientWidth)+'px';ph.style.display='block'}
    // Seek bar
    const sf=document.getElementById('sf'),sd=document.getElementById('sd');
    if(sf)sf.style.width=(frac*100)+'%';if(sd)sd.style.left=(frac*100)+'%';
    // Chord timeline playhead
    const tp=document.getElementById('tlph'),ti=document.getElementById('ti');
    if(tp&&ti){tp.style.left=(frac*TOTAL_W)+'px'}
    // Auto-scroll chord timeline
    const to=document.getElementById('to');
    if(to){const x=frac*TOTAL_W;to.scrollLeft=Math.max(0,x-to.clientWidth/2)}
}
let lastCI=-1;
function updChord(t){
    let idx=-1;for(let i=CHORDS.length-1;i>=0;i--){if(t>=CHORDS[i].time){idx=i;break}}
    if(idx===lastCI)return;lastCI=idx;
    const cmEl=document.getElementById('cm');
    const cnEl=document.getElementById('cn');
    const nnEl=document.getElementById('nn');
    if(idx>=0){
        const ch=CHORDS[idx].chord;
        if(cmEl){cmEl.textContent=ch;cmEl.classList.toggle('active',ch!=='N/C'&&ch!=='—')}
        const nx=idx+1<CHORDS.length?CHORDS[idx+1].chord:'—';
        if(cnEl){cnEl.textContent=nx}
        const notes=c2n(ch);drawPiano(notes);
        if(nnEl){if(notes.length){const es=notes.map(n=>ESN[n]),en=notes.map(n=>ENN[n]);nnEl.innerHTML=es.join(' · ')+' <span class="en">('+en.join(' - ')+')</span>'}else nnEl.textContent=''}
        document.querySelectorAll('.cb').forEach((b,i)=>b.classList.toggle('active',i===idx));
    } else {
        if(cmEl){cmEl.textContent='—';cmEl.classList.remove('active')}
        if(cnEl)cnEl.textContent='—';
        if(nnEl)nnEl.textContent='';
        drawPiano([]);
    }
}

// ── Tick ──────────────────────────────────────────────────────────────────────
function tick(){
    const t=getCurTime();if(t>=DURATION){stopAll();return}
    updPH(t/DURATION);updTime(t);updChord(t);
    rafId=requestAnimationFrame(tick);
}

// ── Events ────────────────────────────────────────────────────────────────────
document.getElementById('sk').addEventListener('click',e=>{
    const r=e.currentTarget.getBoundingClientRect();seekTo(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*DURATION)
});
const wc=document.getElementById('wc');
if(wc)wc.addEventListener('click',e=>{const r=wc.getBoundingClientRect();seekTo(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*DURATION)});
const to=document.getElementById('to');
if(to)to.addEventListener('click',e=>{const ti=document.getElementById('ti');if(!ti)return;const r=ti.getBoundingClientRect();seekTo(Math.max(0,Math.min(1,(e.clientX-r.left)/TOTAL_W))*DURATION)});

// ── Init ──────────────────────────────────────────────────────────────────────
buildGains();loadAll();
document.getElementById('tt').textContent=fmt(DURATION);
const kl=document.getElementById('klbl');if(kl)kl.textContent=KEY_NAME;
window.addEventListener('resize',()=>setTimeout(drawAll,100));
"""

    # ══════════════════════════════════════════════════════════════════════
    # HTML
    # ══════════════════════════════════════════════════════════════════════
    def ctl_row(key, icon, name, color, vol=100):
        return (
            f"<div class='ctl' style='border-left:4px solid {color}'>"
            f"<div class='ct'>"
            f"<button class='bm' id='m-{key}' onclick=\"toggleMute('{key}')\">M</button>"
            f"<button class='bs' id='s-{key}' onclick=\"toggleSolo('{key}')\">S</button>"
            f"<span class='ci'>{icon}</span>"
            f"<span class='cn' style='color:{color}'>{name}</span></div>"
            f"<div class='vr-row'>"
            f"<input type='range' class='vr' id='vol-{key}' style='--tc:{color}' "
            f"min='0' max='200' value='{vol}' oninput=\"setVolume('{key}',this.value)\">"
            f"<span class='vp' id='vl-{key}'>{vol}%</span></div></div>"
        )

    def wf_row(key, extra=""):
        return f"<div class='wr' id='wr-{key}'><canvas id='wf-{key}'></canvas>{extra}</div>"

    mixer_html = ""
    if has_stems:
        ctrl_rows  = "".join(ctl_row(st['key'], st['icon'], st['name'], st['color']) for st in stems_list)
        ctrl_rows += ctl_row('metro', 'M', 'Metrónomo', '#666666', 70)
        wave_rows  = "".join(wf_row(st['key']) for st in stems_list)
        mpills = ("<div class='mpills'>"
                  "<span class='mpill' data-m='0.5' onclick='setMetroMult(0.5)'>0.5×</span>"
                  "<span class='mpill sel' data-m='1' onclick='setMetroMult(1)'>1×</span>"
                  "<span class='mpill' data-m='2' onclick='setMetroMult(2)'>2×</span></div>")
        wave_rows += wf_row('metro', mpills)
        mixer_html = (
            f"<div id='bd'>"
            f"<div id='cc'>{ctrl_rows}</div>"
            f"<div id='wc'><div id='ph'></div>{wave_rows}</div>"
            f"</div>"
        )

    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{css}"
        f"input[type=range].vr::-webkit-slider-thumb{{background:var(--tc,#888)}}"
        f"</style></head><body>"
        f"<div id='tp'>"
        f"<button class='tb' id='btn-play' onclick='togglePlay()' disabled>▶</button>"
        f"<button class='tb' onclick='stopAll()'>⏹</button>"
        f"<span class='tt' id='tc'>0:00</span>"
        f"<div id='sk'><div id='sf'></div><div id='sd'></div></div>"
        f"<span class='tt' id='tt'>--:--</span>"
        f"<select id='spd' onchange='setRate(this.value)'>"
        f"<option value='0.5'>0.5×</option><option value='1' selected>1×</option>"
        f"<option value='1.5'>1.5×</option><option value='2'>2×</option></select>"
        f"<span id='klbl'></span>"
        f"<span id='stlbl'>Cargando…</span>"
        f"</div>"
        f"{mixer_html}"
        f"<div id='cs'><div id='to'><div id='ti'><div id='tlph'></div></div></div></div>"
        f"<div id='ps'>"
        f"<div id='ci'>"
        f"<div id='cm'>—</div>"
        f"<div class='ci-lbl'>SIGUIENTE</div><div id='cn'>—</div>"
        f"<div class='ci-lbl' style='margin-top:4px'>NOTAS</div><div id='nn'></div>"
        f"</div>"
        f"<div id='pw'><canvas id='pv'></canvas></div>"
        f"</div>"
        f"<script>{js_data}{js_logic}</script>"
        f"</body></html>"
    )

    components.html(html, height=comp_h, scrolling=False)


def render_stem_mixer(stems: dict, bpm: float = 120.0,
                      duration: float = 0.0, time_sig: int = 4):
    """Eliminado — reemplazado por render_integrated_player()."""
    st.info("Usar render_integrated_player() en su lugar.")
    return


def _render_stem_mixer_LEGACY_DO_NOT_USE(stems: dict, bpm: float = 120.0,
                      duration: float = 0.0, time_sig: int = 4):
    """LEGACY — no se llama. Dejar hasta que se confirme que no hace falta."""
    import base64, json
    import streamlit.components.v1 as components

    # ── Preparar duración si no viene ────────────────────────────────────
    if duration <= 0:
        first = next(iter(stems.values()))
        duration = float(len(first['audio_mono'])) / float(first['sr'])

    # ── Serializar pistas ─────────────────────────────────────────────────
    stems_list: list[dict] = []
    for key, stem in stems.items():
        mono  = stem['audio_mono']
        n_pts = 1600
        step  = max(1, len(mono) // n_pts)
        wf    = [round(float(v), 4) for v in mono[::step]]

        data, mime = _compress_to_mp3(stem['wav_bytes'])
        b64 = base64.b64encode(data).decode()

        stems_list.append({
            'key':  key,
            'name': stem['name_es'],
            'icon': stem['icon'],
            'color': stem['color'],
            'wf':   wf,
            'b64':  b64,
            'mime': mime,
        })

    stems_json = json.dumps(stems_list)
    n_tracks   = len(stems_list) + 1      # +1 metrónomo
    comp_h     = 60 + n_tracks * 96 + 20  # transport + filas + padding

    # ══════════════════════════════════════════════════════════════════════
    # CSS
    # ══════════════════════════════════════════════════════════════════════
    css = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#cfd8e3;font-family:'Segoe UI',sans-serif;
     user-select:none;overflow:hidden}

/* ── Transport ── */
#transport{display:flex;align-items:center;gap:8px;
  background:#111827;border-bottom:1px solid #1f2d45;
  padding:8px 14px;height:54px}
.tb{background:#1a2540;border:1px solid #2a3a5a;color:#90caf9;
    border-radius:7px;width:34px;height:34px;font-size:14px;
    cursor:pointer;transition:background .15s}
.tb:hover{background:#243050}
.tb:disabled{opacity:.4;cursor:default}
.tb.playing{background:#1565c0;border-color:#1976d2;color:#fff}
#seek-bar{flex:1;height:6px;background:#1f2d45;border-radius:3px;
  cursor:pointer;position:relative}
#seek-fill{height:100%;background:#4fc3f7;border-radius:3px;
  pointer-events:none;width:0%}
#seek-dot{position:absolute;top:-5px;width:16px;height:16px;
  border-radius:50%;background:#fff;box-shadow:0 0 7px #4fc3f7;
  margin-left:-8px;pointer-events:none;left:0%}
.t-time{font-size:.82rem;color:#78909c;font-variant-numeric:tabular-nums;
  white-space:nowrap}
#speed-sel{background:#1a2540;border:1px solid #2a3a5a;color:#90caf9;
  border-radius:7px;padding:4px 8px;font-size:.78rem;cursor:pointer}
#status-lbl{font-size:.7rem;color:#546e7a;min-width:55px}

/* ── Body layout ── */
#body{display:flex}

/* ── Controls column ── */
#ctrl-col{width:195px;min-width:195px;border-right:1px solid #1f2d45}
.ctl{height:96px;border-bottom:1px solid #1a2233;padding:7px 10px;
  background:#0f1520;display:flex;flex-direction:column;
  justify-content:space-between}
.ctl-top{display:flex;align-items:center;gap:5px}
.bm,.bs{width:22px;height:22px;font-size:.62rem;font-weight:700;
  border-radius:4px;border:1px solid #37474f;cursor:pointer;
  background:transparent;color:#546e7a;transition:all .12s;line-height:1}
.bm.on{background:#bf360c;border-color:#bf360c;color:#fff}
.bs.on{background:#f57f17;border-color:#f57f17;color:#000}
.c-icon{font-size:.95rem}
.c-name{font-size:.78rem;font-weight:700;flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vol-row{display:flex;align-items:center;gap:5px}
input[type=range].vr{-webkit-appearance:none;flex:1;height:3px;
  border-radius:2px;outline:none;cursor:pointer;background:#1f2d45}
input[type=range].vr::-webkit-slider-thumb{-webkit-appearance:none;
  width:13px;height:13px;border-radius:50%;cursor:pointer}
.vpct{font-size:.6rem;color:#546e7a;min-width:26px;text-align:right}

/* ── Waveform column ── */
#wave-col{flex:1;position:relative;overflow:hidden}
#playhead{position:absolute;top:0;bottom:0;width:2px;
  background:#ff1744;z-index:30;pointer-events:none;left:0;
  box-shadow:0 0 8px rgba(255,23,68,.7);display:none}
.wrow{height:96px;border-bottom:1px solid #1a2233;
  cursor:crosshair;position:relative;overflow:hidden;
  background:#090d14;transition:background .1s}
.wrow canvas{position:absolute;top:0;left:0;width:100%;height:100%}
.wrow.muted canvas{opacity:.18;filter:grayscale(.6)}
.wrow:hover{background:#0c111e}
.metro-pills{position:absolute;top:6px;left:8px;
  display:flex;gap:4px;z-index:10}
.mpill{background:#1a2540;border:1px solid #2a3a5a;color:#90caf9;
  border-radius:10px;padding:2px 8px;font-size:.65rem;cursor:pointer;
  transition:all .12s}
.mpill.sel{background:#1b5e20;border-color:#2e7d32;color:#a5d6a7}
"""

    # ══════════════════════════════════════════════════════════════════════
    # JavaScript
    # ══════════════════════════════════════════════════════════════════════
    js = f"""
// ── Constants ────────────────────────────────────────────────────────────────
const STEMS    = {stems_json};
const BPM      = {bpm:.3f};
const DURATION = {duration:.3f};
const TIMESIG  = {time_sig};

// ── Audio engine ──────────────────────────────────────────────────────────────
const AC   = new (window.AudioContext || window.webkitAudioContext)();
const DEST = AC.createGain();  // master
DEST.connect(AC.destination);

const gainNodes = {{}};   // key → GainNode
const buffers   = {{}};   // key → AudioBuffer
const volumes   = {{}};   // key → 0..2
const muted     = new Set();
const soloed    = new Set();
let   metroBuf  = null;
let   metroGain = null;
let   metroMult = 1;    // metronome subdivision: 0.5 / 1 / 2

let isPlaying    = false;
let seekPos      = 0;
let ctxStart     = 0;
let playRate     = 1.0;
let rafId        = null;
let activeSrcs   = [];
let metroSrcs    = [];

// ── Utils ─────────────────────────────────────────────────────────────────────
function fmt(s) {{
    return Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0');
}}
function getCurTime() {{
    if (!isPlaying) return seekPos;
    return Math.min(seekPos + (AC.currentTime - ctxStart) * playRate, DURATION);
}}
function resumeCtx() {{
    if (AC.state === 'suspended') AC.resume();
}}

// ── Build gain nodes ──────────────────────────────────────────────────────────
function buildGains() {{
    STEMS.forEach(st => {{
        const g = AC.createGain(); g.connect(DEST);
        gainNodes[st.key] = g;
        volumes[st.key]   = 1.0;
    }});
    metroGain = AC.createGain(); metroGain.connect(DEST);
    volumes['metro'] = 0.7;
}}

// ── Load all audio ────────────────────────────────────────────────────────────
async function loadAll() {{
    setStatus('Cargando audio…');
    await Promise.all(STEMS.map(async st => {{
        try {{
            const bin = atob(st.b64);
            const u8  = new Uint8Array(bin.length);
            for (let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i);
            buffers[st.key] = await AC.decodeAudioData(u8.buffer);
        }} catch(e) {{ console.error('decode',st.key,e); }}
    }}));
    setStatus('✅ Listo');
    document.getElementById('btn-play').disabled = false;
    setTimeout(drawAll, 150);
}}

// ── Metronome click buffer ────────────────────────────────────────────────────
function makeMetroBuf() {{
    const sr  = AC.sampleRate;
    const dur = 0.055;
    const buf = AC.createBuffer(1, Math.ceil(sr*dur), sr);
    const ch  = buf.getChannelData(0);
    for (let i=0;i<ch.length;i++) {{
        const t = i/sr;
        ch[i] = Math.sin(2*Math.PI*1100*t) * Math.exp(-t*55);
    }}
    return buf;
}}

function scheduleMetro(fromTime) {{
    if (!metroBuf) metroBuf = makeMetroBuf();
    const bpmEff  = BPM * metroMult;
    const interval = 60 / bpmEff;
    const startBeat = Math.ceil(fromTime / interval);
    const endBeat   = Math.ceil(DURATION / interval);
    for (let b=startBeat; b<endBeat; b++) {{
        const beatTime  = b * interval;
        const schedTime = AC.currentTime + (beatTime - fromTime) / playRate;
        if (schedTime < AC.currentTime + 0.01) continue;
        const src = AC.createBufferSource();
        src.buffer = metroBuf;
        src.playbackRate.value = playRate;
        const env = AC.createGain();
        const isAccent = b % Math.round(TIMESIG * metroMult) === 0;
        const vol = (volumes['metro']||0.7) * (isAccent ? 1.0 : 0.55);
        env.gain.setValueAtTime(vol, schedTime);
        env.gain.exponentialRampToValueAtTime(0.001, schedTime + 0.07);
        src.connect(env); env.connect(metroGain);
        src.start(schedTime);
        metroSrcs.push(src);
    }}
}}

// ── Playback ──────────────────────────────────────────────────────────────────
function startAll() {{
    resumeCtx();
    ctxStart = AC.currentTime;
    STEMS.forEach(st => {{
        if (!buffers[st.key]) return;
        const src = AC.createBufferSource();
        src.buffer = buffers[st.key];
        src.playbackRate.value = playRate;
        src.connect(gainNodes[st.key]);
        src.start(0, seekPos);
        src.onended = () => {{ if (getCurTime() >= DURATION-0.2) stopAll(); }};
        activeSrcs.push(src);
    }});
    scheduleMetro(seekPos);
    isPlaying = true;
    updateTransport();
    rafId = requestAnimationFrame(tick);
}}

function pauseAll() {{
    activeSrcs.forEach(s=>{{try{{s.stop();}}catch(e){{}}}});
    metroSrcs.forEach(s=>{{try{{s.stop();}}catch(e){{}}}});
    activeSrcs=[]; metroSrcs=[];
    seekPos = Math.min(getCurTime(), DURATION);
    isPlaying = false;
    cancelAnimationFrame(rafId);
    updateTransport();
}}

function stopAll() {{
    pauseAll();
    seekPos = 0;
    updatePlayhead(0);
    document.getElementById('time-cur').textContent = '0:00';
    document.getElementById('seek-fill').style.width = '0%';
    document.getElementById('seek-dot').style.left   = '0%';
}}

function togglePlay() {{
    if (isPlaying) pauseAll(); else startAll();
}}

function seekTo(t) {{
    const was = isPlaying;
    pauseAll();
    seekPos = Math.max(0, Math.min(t, DURATION));
    if (was) startAll();
    else {{ updatePlayhead(seekPos/DURATION); updateTimeDisplay(seekPos); }}
}}

// ── Gain management ───────────────────────────────────────────────────────────
function applyGains() {{
    const hasSolo = soloed.size > 0;
    STEMS.forEach(st => {{
        const hear = !muted.has(st.key) && (!hasSolo || soloed.has(st.key));
        gainNodes[st.key].gain.value = hear ? (volumes[st.key]||1) : 0;
    }});
    const mM = muted.has('metro');
    const hM = !mM && (!hasSolo || soloed.has('metro'));
    metroGain.gain.value = hM ? (volumes['metro']||0.7) : 0;
}}

function toggleMute(key) {{
    muted.has(key) ? muted.delete(key) : muted.add(key);
    const btn = document.getElementById('m-'+key);
    if(btn) btn.classList.toggle('on', muted.has(key));
    const row = document.getElementById('wr-'+key);
    if(row) row.classList.toggle('muted', muted.has(key));
    applyGains();
}}

function toggleSolo(key) {{
    soloed.has(key) ? soloed.delete(key) : soloed.add(key);
    const btn = document.getElementById('s-'+key);
    if(btn) btn.classList.toggle('on', soloed.has(key));
    applyGains();
}}

function setVolume(key, val) {{
    volumes[key] = parseFloat(val)/100;
    const lbl = document.getElementById('vl-'+key);
    if(lbl) lbl.textContent = val+'%';
    applyGains();
}}

function setMetroMult(m) {{
    metroMult = m;
    // Reschedule if playing
    metroSrcs.forEach(s=>{{try{{s.stop();}}catch(e){{}}}});
    metroSrcs=[];
    if(isPlaying) scheduleMetro(getCurTime());
    document.querySelectorAll('.mpill').forEach(p=>{{
        p.classList.toggle('sel', parseFloat(p.dataset.m)===m);
    }});
}}

// ── Transport & UI ────────────────────────────────────────────────────────────
function updateTransport() {{
    const btn = document.getElementById('btn-play');
    btn.textContent = isPlaying ? '⏸' : '▶';
    btn.classList.toggle('playing', isPlaying);
}}

function updatePlayhead(frac) {{
    const ph = document.getElementById('playhead');
    const wc = document.getElementById('wave-col');
    if(!ph||!wc) return;
    const x = frac * wc.clientWidth;
    ph.style.left    = x+'px';
    ph.style.display = 'block';
    const sf = document.getElementById('seek-fill');
    const sd = document.getElementById('seek-dot');
    if(sf) sf.style.width = (frac*100)+'%';
    if(sd) sd.style.left  = (frac*100)+'%';
}}

function updateTimeDisplay(t) {{
    const el = document.getElementById('time-cur');
    if(el) el.textContent = fmt(t);
}}

function setStatus(msg) {{
    const el = document.getElementById('status-lbl');
    if(el) el.textContent = msg;
}}

function tick() {{
    const t = getCurTime();
    if(t >= DURATION) {{ stopAll(); return; }}
    updatePlayhead(t/DURATION);
    updateTimeDisplay(t);
    rafId = requestAnimationFrame(tick);
}}

// ── Seek bar ──────────────────────────────────────────────────────────────────
function handleSeekBar(e) {{
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1,(e.clientX-rect.left)/rect.width));
    seekTo(frac*DURATION);
}}

// ── Click on waveform row ─────────────────────────────────────────────────────
document.getElementById('wave-col').addEventListener('click', e => {{
    const wc   = document.getElementById('wave-col');
    const rect = wc.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1,(e.clientX-rect.left)/rect.width));
    seekTo(frac*DURATION);
}});

// ── Speed ─────────────────────────────────────────────────────────────────────
function setRate(r) {{
    playRate = parseFloat(r);
    if(isPlaying) {{ pauseAll(); startAll(); }}
}}

// ── Draw waveforms ────────────────────────────────────────────────────────────
function drawWF(id, wf, color) {{
    const cv = document.getElementById(id);
    if(!cv) return;
    const dpr = window.devicePixelRatio||1;
    cv.width  = cv.offsetWidth*dpr;
    cv.height = cv.offsetHeight*dpr;
    const ctx = cv.getContext('2d');
    ctx.scale(dpr,dpr);
    const W=cv.offsetWidth, H=cv.offsetHeight, mid=H/2;
    ctx.fillStyle='#090d14'; ctx.fillRect(0,0,W,H);
    const n   = wf.length;
    const bw  = Math.max(1, W/n);
    const maxV= Math.max(...wf.map(Math.abs))||1;
    const hex = color.replace('#','');
    const r=parseInt(hex.slice(0,2),16), g=parseInt(hex.slice(2,4),16), b=parseInt(hex.slice(4,6),16);
    ctx.fillStyle = `rgba(${{r}},${{g}},${{b}},0.85)`;
    for(let i=0;i<n;i++) {{
        const amp = (Math.abs(wf[i])/maxV)*mid*0.86;
        ctx.fillRect(i*bw, mid-amp, bw*0.72, amp*2);
    }}
}}

function drawMetro(id) {{
    const cv = document.getElementById(id);
    if(!cv) return;
    const dpr = window.devicePixelRatio||1;
    cv.width  = cv.offsetWidth*dpr;
    cv.height = cv.offsetHeight*dpr;
    const ctx = cv.getContext('2d');
    ctx.scale(dpr,dpr);
    const W=cv.offsetWidth, H=cv.offsetHeight;
    ctx.fillStyle='#090d14'; ctx.fillRect(0,0,W,H);
    const interval  = 60/BPM;
    const totalBeats= Math.floor(DURATION/interval);
    for(let b=0;b<totalBeats;b++) {{
        const x = (b*interval/DURATION)*W;
        const isA = b%TIMESIG===0;
        const bH  = isA ? H*0.75 : H*0.38;
        ctx.fillStyle = isA ? 'rgba(102,187,106,0.9)' : 'rgba(102,187,106,0.35)';
        ctx.fillRect(x,(H-bH)/2, Math.max(2,W/totalBeats*0.45), bH);
    }}
}}

function drawAll() {{
    STEMS.forEach(st => drawWF('wf-'+st.key, st.wf, st.color));
    drawMetro('wf-metro');
}}

// ── Init ──────────────────────────────────────────────────────────────────────
buildGains();
loadAll();
document.getElementById('time-tot').textContent = fmt(DURATION);
window.addEventListener('resize', () => {{ setTimeout(drawAll,100); }});
"""

    # ══════════════════════════════════════════════════════════════════════
    # HTML
    # ══════════════════════════════════════════════════════════════════════
    def ctl_row(key, icon, name, color, init_vol=100):
        return (
            f"<div class='ctl' style='border-left:4px solid {color}'>"
            f"  <div class='ctl-top'>"
            f"    <button class='bm' id='m-{key}' onclick=\"toggleMute('{key}')\">M</button>"
            f"    <button class='bs' id='s-{key}' onclick=\"toggleSolo('{key}')\">S</button>"
            f"    <span class='c-icon'>{icon}</span>"
            f"    <span class='c-name' style='color:{color}'>{name}</span>"
            f"  </div>"
            f"  <div class='vol-row'>"
            f"    <input type='range' class='vr' id='vol-{key}' "
            f"      style='--tc:{color}' min='0' max='200' value='{init_vol}'"
            f"      oninput=\"setVolume('{key}',this.value)\">"
            f"    <span class='vpct' id='vl-{key}'>{init_vol}%</span>"
            f"  </div>"
            f"</div>"
        )

    def wf_row(key, extra_content=""):
        return (
            f"<div class='wrow' id='wr-{key}'>"
            f"  <canvas id='wf-{key}'></canvas>"
            f"  {extra_content}"
            f"</div>"
        )

    ctrl_rows = "".join(
        ctl_row(st['key'], st['icon'], st['name'], st['color'])
        for st in stems_list
    )
    ctrl_rows += ctl_row('metro', '🥁', 'Metrónomo', '#66bb6a', init_vol=70)

    wave_rows = "".join(wf_row(st['key']) for st in stems_list)
    metro_pills = (
        "<div class='metro-pills'>"
        "<span class='mpill' data-m='0.5' onclick='setMetroMult(0.5)'>0.5×</span>"
        "<span class='mpill sel' data-m='1' onclick='setMetroMult(1)'>1×</span>"
        "<span class='mpill' data-m='2' onclick='setMetroMult(2)'>2×</span>"
        "</div>"
    )
    wave_rows += wf_row('metro', metro_pills)

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
{css}
input[type=range].vr::-webkit-slider-thumb {{background:var(--tc,#4fc3f7)}}
</style></head><body>
<div id='transport'>
  <button class='tb' id='btn-play' onclick='togglePlay()' disabled>▶</button>
  <button class='tb' onclick='stopAll()'>⏹</button>
  <span class='t-time' id='time-cur'>0:00</span>
  <div id='seek-bar' onclick='handleSeekBar(event)'>
    <div id='seek-fill'></div>
    <div id='seek-dot'></div>
  </div>
  <span class='t-time' id='time-tot'>--:--</span>
  <select id='speed-sel' onchange='setRate(this.value)'>
    <option value='0.5'>0.5×</option>
    <option value='1' selected>1×</option>
    <option value='1.5'>1.5×</option>
    <option value='2'>2×</option>
  </select>
  <span id='status-lbl'>Cargando…</span>
</div>
<div id='body'>
  <div id='ctrl-col'>{ctrl_rows}</div>
  <div id='wave-col'>
    <div id='playhead'></div>
    {wave_rows}
  </div>
</div>
<script>{js}</script>
</body></html>"""

    components.html(html, height=comp_h, scrolling=False)






# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY PERSISTENCE  (metadata + audio + stems en disco)
# ─────────────────────────────────────────────────────────────────────────────

LIBRARY_DIR  = Path(__file__).parent / "library"
LIBRARY_JSON = LIBRARY_DIR / "index.json"

# Info de pistas para reconstruir stems desde disco
STEMS_INFO_DISK = {
    'vocals': {'name_es': 'Voces',    'icon': 'V', 'color': '#c8d4e0'},
    'drums':  {'name_es': 'Bateria',  'icon': 'D', 'color': '#a0b0c0'},
    'bass':   {'name_es': 'Bajo',     'icon': 'B', 'color': '#8090a0'},
    'other':  {'name_es': 'Otro',     'icon': 'O', 'color': '#687888'},
    'piano':  {'name_es': 'Piano',    'icon': 'P', 'color': '#90a8b8'},
    'guitar': {'name_es': 'Guitarra', 'icon': 'G', 'color': '#b0c0cc'},
}


def _to_serializable(obj):
    """Convierte numpy arrays y tipos numpy a Python nativo para JSON."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _save_item(item: dict):
    """
    Guarda en disco:
      library/{id}/meta.json   ← resultados de analisis + metadatos
      library/{id}/audio.wav   ← audio completo
      library/{id}/stems/*.wav ← una pista por instrumento
    """
    import soundfile as sf

    item_dir = LIBRARY_DIR / item['id']
    item_dir.mkdir(parents=True, exist_ok=True)

    # Metadatos + resultados de analisis (sin bytes)
    meta = {k: v for k, v in item.items()
            if k not in ('results', 'stems', 'audio_bytes')}
    if item.get('results'):
        meta['results'] = _to_serializable(item['results'])
    (item_dir / 'meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # Audio completo
    if item.get('audio_bytes'):
        (item_dir / 'audio.wav').write_bytes(item['audio_bytes'])

    # Pistas individuales
    if item.get('stems'):
        stems_dir = item_dir / 'stems'
        stems_dir.mkdir(exist_ok=True)
        for key, stem in item['stems'].items():
            wav_path = stems_dir / f"{key}.wav"
            if not wav_path.exists():
                wav_path.write_bytes(stem['wav_bytes'])


def _load_item_from_disk(item_id: str) -> dict | None:
    """Carga todos los datos de un item desde disco."""
    item_dir = LIBRARY_DIR / item_id
    meta_path = item_dir / 'meta.json'

    if not meta_path.exists():
        return None

    data = json.loads(meta_path.read_text(encoding='utf-8'))

    # Audio WAV
    audio_path = item_dir / 'audio.wav'
    if audio_path.exists():
        data['audio_bytes'] = audio_path.read_bytes()
    else:
        data['audio_bytes'] = None

    # Stems
    stems_dir = item_dir / 'stems'
    if stems_dir.exists():
        try:
            import soundfile as sf
            stems = {}
            for wav_file in sorted(stems_dir.glob('*.wav')):
                key = wav_file.stem
                info = STEMS_INFO_DISK.get(key)
                if not info:
                    continue
                audio_np, sr = sf.read(str(wav_file), always_2d=True)
                mono = audio_np.mean(axis=1)
                stems[key] = {
                    'name_es':    info['name_es'],
                    'icon':       info['icon'],
                    'color':      info['color'],
                    'audio_mono': mono,
                    'sr':         sr,
                    'wav_bytes':  wav_file.read_bytes(),
                }
            data['stems'] = stems if stems else None
        except Exception:
            data['stems'] = None
    else:
        data['stems'] = None

    return data


def _load_library() -> list:
    """
    Carga la lista de items con SOLO metadatos (sin audio/stems).
    El audio completo se carga on-demand en show_detail().
    """
    LIBRARY_DIR.mkdir(exist_ok=True)
    if not LIBRARY_JSON.exists():
        return []
    try:
        entries = json.loads(LIBRARY_JSON.read_text(encoding='utf-8'))
        # Asegurar que cada entrada tenga campos nulos para no romper accesos
        for e in entries:
            e.setdefault('results', None)
            e.setdefault('stems', None)
            e.setdefault('audio_bytes', None)
        return entries
    except Exception:
        return []


def _save_library_index(items: list):
    """Actualiza el index JSON con los metadatos."""
    LIBRARY_DIR.mkdir(exist_ok=True)
    safe_keys = ('id', 'filename', 'size_mb', 'ext', 'date_added',
                 'bpm', 'key', 'key_en', 'duration', 'stems_plan', 'model_name')
    index = [{k: v for k, v in it.items() if k in safe_keys} for it in items]
    LIBRARY_JSON.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8'
    )


def _delete_item_from_disk(item_id: str):
    """Elimina la carpeta de un item del disco."""
    import shutil as _shutil
    item_dir = LIBRARY_DIR / item_id
    if item_dir.exists():
        _shutil.rmtree(item_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# INSTRUMENT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

FREE_INSTRUMENTS = {
    'vocals': {'label': 'Voces',    'abbr': 'V', 'sub': '4 o 6 pistas'},
    'drums':  {'label': 'Bateria',  'abbr': 'D', 'sub': '4 o 6 pistas'},
    'bass':   {'label': 'Bajo',     'abbr': 'B', 'sub': '4 o 6 pistas'},
    'other':  {'label': 'Otro',     'abbr': 'O', 'sub': '4 o 6 pistas'},
    'piano':  {'label': 'Piano',    'abbr': 'P', 'sub': 'Solo modelo 6 pistas'},
    'guitar': {'label': 'Guitarra', 'abbr': 'G', 'sub': 'Solo modelo 6 pistas'},
}

PREMIUM_INSTRUMENTS = [
    {'label': 'Teclado',          'abbr': 'K'},
    {'label': 'Vientos',          'abbr': 'W'},
    {'label': 'Cuerdas',          'abbr': 'S'},
    {'label': 'Efectos',          'abbr': 'FX'},
    {'label': 'Coros / Backing',  'abbr': 'BK'},
    {'label': 'Sintetizador',     'abbr': 'SY'},
]

FREE_LIMIT = 5   # max instrumentos en version gratuita

DEFAULT_SELECTION = {'vocals', 'drums', 'bass', 'other'}   # seleccion por defecto


# ─────────────────────────────────────────────────────────────────────────────
# PANTALLA 1 — BIBLIOTECA
# ─────────────────────────────────────────────────────────────────────────────

def show_library():
    lib = st.session_state.get('library', [])

    col_h, col_add = st.columns([8, 1])
    with col_h:
        n = len(lib)
        st.markdown(
            f"<h2 style='margin:0;color:#fff;font-size:1.2rem;font-weight:600'>"
            f"Separacion de pistas</h2>"
            f"<p style='color:#252525;font-size:.74rem;margin:3px 0 0'>"
            f"{n} archivo{'s' if n != 1 else ''}</p>",
            unsafe_allow_html=True,
        )
    with col_add:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("+ Agregar", type="primary", use_container_width=True, key="lib_add"):
            st.session_state['page'] = 'new'
            st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if not lib:
        st.markdown(
            "<div style='text-align:center;padding:5rem 2rem;"
            "border:1px dashed #141414;border-radius:14px'>"
            "<div style='font-size:2rem;color:#1e1e1e;margin-bottom:14px'>&#9836;</div>"
            "<p style='color:#2e2e2e;font-size:.92rem;margin:0'>Aun no tienes ningun archivo</p>"
            "<p style='color:#1e1e1e;font-size:.78rem;margin:6px 0 0'>"
            "Haz clic en <strong style=\"color:#444\">+ Agregar</strong> para empezar"
            "</p></div>",
            unsafe_allow_html=True,
        )
        return

    # Cabeceras de tabla
    st.markdown(
        "<div style='display:grid;"
        "grid-template-columns:minmax(0,3fr) 110px 60px 90px 80px 80px;"
        "padding:5px 12px;border-bottom:1px solid #141414;"
        "color:#1e1e1e;font-size:.65rem;text-transform:uppercase;letter-spacing:.9px'>"
        "<span>Titulo</span><span>Fecha</span><span>BPM</span>"
        "<span>Tono</span><span>Duracion</span><span></span></div>",
        unsafe_allow_html=True,
    )

    for item in lib:
        dur  = item.get('duration', 0)
        ds   = f"{int(dur//60)}:{int(dur%60):02d}" if dur else '—'
        ext  = (item.get('ext', '') or '').upper()
        is_v = ext in ('MP4', 'MKV', 'MOV', 'AVI', 'WEBM')
        name = item.get('filename', 'Archivo')

        c1, c2, c3, c4, c5, c6 = st.columns([3, 1.1, 0.6, 0.9, 0.8, 0.8])
        with c1:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;padding:8px 0'>"
                f"<span style='color:#252525'>{'&#9654;' if is_v else '&#9836;'}</span>"
                f"<div><div style='color:#aaa;font-size:.83rem;font-weight:500;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{name}</div>"
                f"<div style='color:#252525;font-size:.68rem'>"
                f"{ext} &middot; {item.get('size_mb',0):.1f} MB</div></div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div style='color:#303030;font-size:.76rem;padding:12px 0'>"
                f"{(item.get('date_added','') or '')[:10]}</div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div style='color:#404040;font-size:.79rem;padding:12px 0'>"
                f"{item.get('bpm','—')}</div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"<div style='color:#404040;font-size:.79rem;padding:12px 0'>"
                f"{item.get('key_en', item.get('key','—'))}</div>",
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f"<div style='color:#404040;font-size:.79rem;padding:12px 0'>{ds}</div>",
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Abrir", key=f"open_{item['id']}", use_container_width=True):
                st.session_state['page']      = 'detail'
                st.session_state['detail_id'] = item['id']
                st.rerun()

        st.markdown(
            "<hr style='margin:0;border:none;border-top:1px solid #0d0d0d'>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PANTALLA 2a — SUBIR ARCHIVO
# ─────────────────────────────────────────────────────────────────────────────

def show_new_file():
    col_back, col_title = st.columns([1, 9])
    with col_back:
        if st.button("← Volver", key="back_new"):
            st.session_state['page'] = 'library'
            st.rerun()
    with col_title:
        st.markdown(
            "<h2 style='margin:0;color:#fff;font-size:1.12rem;font-weight:600'>"
            "Agregar archivo</h2>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    AUDIO_EXTS = ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aiff', 'wma']
    VIDEO_EXTS = ['mp4', 'mkv', 'avi', 'mov', 'webm', 'wmv', 'flv', 'm4v', '3gp']
    ALL_EXTS   = AUDIO_EXTS + VIDEO_EXTS

    uploaded = st.file_uploader(
        "upload",
        type=ALL_EXTS,
        label_visibility="collapsed",
        key="new_file_uploader",
        help="MP3, WAV, FLAC, M4A, OGG, MP4, MKV, MOV y mas",
    )

    if uploaded is None:
        st.markdown(
            "<div style='border:1px dashed #181818;border-radius:12px;"
            "padding:3rem 2rem;text-align:center;color:#242424;font-size:.9rem;"
            "margin-top:.5rem'>"
            "Arrastra un archivo aqui o haz clic para seleccionar<br>"
            "<span style='font-size:.74rem;color:#181818'>"
            "MP3 &middot; WAV &middot; FLAC &middot; M4A &middot; "
            "MP4 &middot; MKV &middot; MOV &middot; y mas</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    ext      = uploaded.name.rsplit('.', 1)[-1].lower()
    size_mb  = uploaded.size / 1_048_576
    is_video = ext in VIDEO_EXTS

    # Fila compacta del archivo seleccionado
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"background:#111;border:1px solid #222;border-radius:8px;"
        f"padding:11px 16px;margin:10px 0'>"
        f"<span style='color:#3a3a3a;font-size:.95rem'>"
        f"{'&#9654;' if is_video else '&#9836;'}</span>"
        f"<span style='color:#c0c0c0;font-size:.87rem;font-weight:500;flex:1'>"
        f"{uploaded.name}</span>"
        f"<span style='color:#2e2e2e;font-size:.72rem'>"
        f"{size_mb:.1f} MB &middot; {ext.upper()}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.button("Continuar  →", type="primary", use_container_width=True, key="new_continue"):
        # Guardar bytes en session_state para usarlos en la pantalla de instrumentos
        uploaded.seek(0)
        st.session_state['pending_file'] = {
            'name':    uploaded.name,
            'data':    uploaded.read(),
            'ext':     ext,
            'size_mb': round(size_mb, 2),
        }
        # Seleccion por defecto al entrar a pantalla de instrumentos
        st.session_state['instr_sel'] = set(DEFAULT_SELECTION)
        st.session_state['sep_mode']  = 'basic_4'
        st.session_state['page']      = 'instruments'
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PANTALLA 2b — SELECCION DE INSTRUMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def _sep_row(icon: str, title: str, subtitle: str, is_sel: bool, is_locked: bool,
             btn_key: str, sel: set, stem_key: str):
    """Renderiza una fila de instrumento compacta estilo Moises."""
    col_card, col_btn = st.columns([11, 1], gap="small")

    cls = "sep-card" + (" selected" if is_sel else "") + (" locked" if is_locked else "")
    end = "&#128274;" if is_locked else ("&#10003;" if is_sel else "")

    with col_card:
        st.markdown(
            f"<div class='{cls}'>"
            f"<div class='sep-card-icon' style='font-family:monospace;font-weight:800;"
            f"font-size:1rem;color:{'#9b72cf' if is_sel else '#2e2e2e'}'>{icon}</div>"
            f"<div class='sep-card-body'>"
            f"<div class='sep-card-title'>{title}</div>"
            f"{'<div class=\"sep-card-sub\">' + subtitle + '</div>' if subtitle else ''}"
            f"</div>"
            f"<div class='sep-card-end'>{end}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_btn:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if not is_locked:
            lbl = "✓" if is_sel else "+"
            if st.button(lbl, key=btn_key, use_container_width=True):
                updated = set(sel)
                if is_sel:
                    updated.discard(stem_key)
                else:
                    updated.add(stem_key)
                st.session_state['instr_sel'] = updated
                st.session_state['sep_mode']  = 'custom'
                st.rerun()


def show_select_instruments():
    from stem_separation import check_demucs
    demucs_ok, _ = check_demucs()

    pending = st.session_state.get('pending_file', {})
    if not pending:
        st.session_state['page'] = 'new'
        st.rerun()
        return

    # ── Volver ────────────────────────────────────────────────────────────
    if st.button("← Volver", key="back_instr"):
        st.session_state['page'] = 'new'
        st.rerun()

    # ── Título ────────────────────────────────────────────────────────────
    name    = pending.get('name', '')
    size_mb = pending.get('size_mb', 0)
    ext     = pending.get('ext', '')
    is_v    = ext in ('mp4', 'mkv', 'avi', 'mov', 'webm')

    st.markdown(
        "<h1 style='font-size:1.7rem;font-weight:800;color:#fff;"
        "margin:12px 0 4px;letter-spacing:-.5px'>Separar pistas</h1>"
        "<p style='color:#444;font-size:.88rem;margin:0 0 18px'>"
        "Selecciona las pistas para separarlas</p>",
        unsafe_allow_html=True,
    )

    # ── Archivo ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"background:#0e0e0e;border:1px solid #1a1a1a;border-radius:10px;"
        f"padding:10px 16px;margin-bottom:6px'>"
        f"<span style='color:#3a3a3a;font-size:.9rem'>"
        f"{'&#9654;' if is_v else '&#9836;'}</span>"
        f"<span style='color:#aaa;font-size:.84rem;font-weight:500;flex:1'>{name}</span>"
        f"<span style='color:#282828;font-size:.7rem'>"
        f"{size_mb:.1f} MB &middot; {ext.upper()}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not demucs_ok:
        st.warning("demucs no instalado — solo se hará análisis musical.\n\n`pip install demucs`")
        sel_stems  = []
        model_name = 'htdemucs_ft'

    else:
        sep_mode = st.session_state.get('sep_mode', 'basic_4')
        sel      = st.session_state.get('instr_sel', set(DEFAULT_SELECTION))

        # ── Separación básica ──────────────────────────────────────────────
        st.markdown(
            "<div class='sep-section-label'>Separación básica</div>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            act4 = sep_mode == 'basic_4'
            st.markdown(
                f"<div class='preset-card {'active' if act4 else ''}'>"
                f"<div style='font-size:1rem;margin-bottom:8px;color:#555'>&#8646;</div>"
                f"<div class='preset-card-title'>Voz, Batería, Bajo, Otros</div>"
                f"<div class='preset-card-sub'>4 pistas</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("✓ Seleccionado" if act4 else "Seleccionar",
                         key="preset_4"):
                st.session_state['sep_mode']  = 'basic_4'
                st.session_state['instr_sel'] = {'vocals', 'drums', 'bass', 'other'}
                st.rerun()

        with col2:
            act2 = sep_mode == 'basic_2'
            st.markdown(
                f"<div class='preset-card {'active' if act2 else ''}'>"
                f"<div style='font-size:1rem;margin-bottom:8px;color:#555'>&#8644;</div>"
                f"<div class='preset-card-title'>Voz, Instrumental</div>"
                f"<div class='preset-card-sub'>2 pistas</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("✓ Seleccionado" if act2 else "Seleccionar",
                         key="preset_2"):
                st.session_state['sep_mode']  = 'basic_2'
                st.session_state['instr_sel'] = {'vocals', 'other'}
                st.rerun()

        # ── Separación personalizada ───────────────────────────────────────
        st.markdown(
            "<div class='sep-section-label'>Separación personalizada</div>",
            unsafe_allow_html=True,
        )

        CUSTOM_INSTRUMENTS = [
            ('vocals', 'V',  'Voces',    '',             False),
            ('guitar', 'G',  'Guitarra', 'Modelo 6 pistas', False),
            ('bass',   'B',  'Bajo',     '',             False),
            ('drums',  'D',  'Batería',  '',             False),
            ('piano',  'P',  'Piano',    'Modelo 6 pistas', False),
            ('other',  'O',  'Otros',    '',             False),
        ]
        PREMIUM_CUSTOM = [
            ('prem_k', 'K', 'Teclados',              '', True),
            ('prem_w', 'W', 'Viento',                '', True),
            ('prem_s', 'S', 'Instrumentos de cuerda','', True),
            ('prem_b', 'BK','Coros / Backing',       '', True),
        ]

        all_items = CUSTOM_INSTRUMENTS + PREMIUM_CUSTOM
        for i in range(0, len(all_items), 2):
            c1, c2 = st.columns(2)
            for col, item in zip([c1, c2], all_items[i:i+2]):
                key, icon, label, sub, locked = item
                with col:
                    _sep_row(
                        icon, label, sub,
                        is_sel=key in sel,
                        is_locked=locked,
                        btn_key=f"sep_{key}",
                        sel=sel,
                        stem_key=key,
                    )

        # ── Pistas multimedia ──────────────────────────────────────────────
        st.markdown(
            "<div class='sep-section-label'>Pistas multimedia</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#2a2a2a;font-size:.72rem;margin:-6px 0 8px'>"
            "Para cine, televisión, podcasts y más</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sep-card locked'>"
            "<div class='sep-card-icon' style='font-size:1.1rem;color:#2a2a2a'>FX</div>"
            "<div class='sep-card-body'>"
            "<div class='sep-card-title' style='color:#333'>"
            "Diálogos, banda sonora y efectos</div>"
            "<div class='sep-card-sub'>3 pistas</div>"
            "</div>"
            "<div class='sep-card-end'>&#128274;</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        sel      = st.session_state.get('instr_sel', set(DEFAULT_SELECTION))
        sel_stems  = [s for s in sel if not s.startswith('prem_')]
        model_name = 'htdemucs_6s' if any(s in sel_stems for s in ('piano', 'guitar')) \
                     else 'htdemucs_ft'

    # ── Botón principal ────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1px solid #111;margin:0 0 14px'>",
        unsafe_allow_html=True,
    )

    label_btn = "Empezar separación" if demucs_ok else "Solo analizar"
    if st.button(label_btn, type="primary", use_container_width=True, key="instr_start"):
        _run_analysis(pending, sel_stems, model_name, demucs_ok)


def _run_analysis(pending: dict, sel_stems: list, model_name: str, demucs_ok: bool):
    """Ejecuta analisis + separacion y navega al detalle."""
    import time as _time

    tmp_path = None
    log: list[dict] = []
    t0 = _time.monotonic()

    def _log(level: str, step: str, msg: str):
        elapsed = round(_time.monotonic() - t0, 2)
        log.append({
            'ts':      datetime.now().strftime('%H:%M:%S'),
            'elapsed': elapsed,
            'level':   level,   # ok | info | warn | error
            'step':    step,
            'msg':     msg,
        })

    try:
        _log('info', 'Inicio', f"Archivo: {pending['name']}  |  {pending['size_mb']:.1f} MB  |  .{pending['ext'].upper()}")

        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=f".{pending['ext']}") as tmp:
            tmp.write(pending['data'])
            tmp_path = tmp.name
        _log('info', 'Temp', f"Archivo temporal creado: {tmp_path}")

        from audio_analysis import MusicAnalyzer
        progress    = st.progress(0, text="Iniciando análisis...")
        status      = st.empty()
        live_log    = st.empty()
        ratio       = 0.60 if (demucs_ok and sel_stems) else 1.0

        def _render_live(entries: list):
            COLORS = {'ok': '#4ade80', 'info': '#818cf8', 'warn': '#fb923c', 'error': '#f87171'}
            ICONS  = {'ok': '✓', 'info': '·', 'warn': '!', 'error': '✕'}
            rows = ""
            for e in entries:
                lvl   = e['level']
                col   = COLORS.get(lvl, '#888')
                icon  = ICONS.get(lvl, '·')
                rows += (
                    f"<tr>"
                    f"<td style='color:#333;font-size:.7rem;padding:3px 8px 3px 0;"
                    f"white-space:nowrap'>+{e['elapsed']}s</td>"
                    f"<td style='text-align:center;padding:3px 6px 3px 0'>"
                    f"<span style='color:{col};font-weight:700;font-size:.8rem'>{icon}</span></td>"
                    f"<td style='color:#555;font-size:.75rem;white-space:nowrap;"
                    f"padding:3px 10px 3px 0;font-weight:600'>{e['step']}</td>"
                    f"<td style='color:#888;font-size:.72rem;font-family:monospace'>{e['msg']}</td>"
                    f"</tr>"
                )
            live_log.markdown(
                f"<div style='background:#060606;border:1px solid #141414;border-radius:8px;"
                f"padding:10px 14px;margin-top:8px'>"
                f"<table style='width:100%;border-collapse:collapse'>{rows}</table></div>",
                unsafe_allow_html=True,
            )

        def _log_and_render(level: str, step: str, msg: str):
            _log(level, step, msg)
            _render_live(log)

        def _ap(pct: int, msg: str):
            progress.progress(min(int(pct * ratio), 100), text=msg)
            status.info(f"⏳ {msg}")
            _log_and_render('info', 'Análisis', msg)

        analyzer = MusicAnalyzer(tmp_path)

        t_load = _time.monotonic()
        try:
            analyzer.load_audio()
            _log_and_render('ok', 'Carga de audio',
                 f"SR={analyzer.sr} Hz  |  dur={analyzer.y.shape[0]/analyzer.sr:.1f}s"
                 f"  |  tardó {_time.monotonic()-t_load:.2f}s")
        except EnvironmentError:
            _log_and_render('error', 'Carga de audio', 'ffmpeg no encontrado')
            progress.empty(); status.empty()
            st.error("ffmpeg no encontrado — instalar: `winget install --id Gyan.FFmpeg -e`")
            return
        except Exception as e:
            _log_and_render('error', 'Carga de audio', str(e))
            progress.empty(); status.empty()
            st.error(f"No se pudo cargar el audio: {e}")
            return

        t_anal = _time.monotonic()
        try:
            results = analyzer.analyze(progress_cb=_ap)
            _log_and_render('ok', 'Análisis musical',
                 f"Tonalidad: {results.get('key',{}).get('key_en','?')}"
                 f"  |  BPM: {results.get('tempo',{}).get('bpm','?')}"
                 f"  |  Segmentos: {results.get('structure',{}).get('n_segments','?')}"
                 f"  |  Acordes: {len(results.get('chords',{}).get('chord_changes',[]))}"
                 f"  |  tardó {_time.monotonic()-t_anal:.2f}s")
        except Exception as e:
            _log_and_render('error', 'Análisis musical', str(e))
            progress.empty(); status.empty()
            st.error(f"Error en el análisis: {e}")
            return

        # Audio WAV en memoria
        audio_bytes = None
        try:
            import soundfile as sf, io as _io
            _buf = _io.BytesIO()
            sf.write(_buf, analyzer.y, analyzer.sr, format='WAV', subtype='PCM_16')
            audio_bytes = _buf.getvalue()
            _log_and_render('ok', 'WAV', f"{len(audio_bytes)/1_048_576:.2f} MB en memoria")
        except Exception as e:
            _log_and_render('warn', 'WAV', f"No se pudo generar WAV: {e}")

        # Separacion de pistas
        stems = None
        if demucs_ok and sel_stems:
            progress.progress(60, text="Separando pistas con IA...")
            status.info("⏳ Separando pistas con IA...")
            from stem_separation import separate_stems, _MODEL_CACHE

            cached = any(
                f"{model_name}_{d}" in _MODEL_CACHE for d in ('cpu', 'cuda')
            )
            _log_and_render('info', 'Demucs',
                 f"Modelo: {model_name}  |  pistas: {sel_stems}"
                 f"  |  caché: {'SÍ ⚡' if cached else 'NO (primera carga ~30s)'}")

            def _sp(pct: int, msg: str):
                progress.progress(min(60 + int(pct * 0.40), 100), text=msg)
                status.info(f"⏳ {msg}")
                _log_and_render('info', 'Demucs', msg)

            t_stem = _time.monotonic()
            try:
                all_s = separate_stems(tmp_path, model_name=model_name, progress_cb=_sp)
                stems = {k: v for k, v in all_s.items() if k in sel_stems}
                total_stem_mb = sum(len(v['wav_bytes']) for v in stems.values()) / 1_048_576
                _log_and_render('ok', 'Separación',
                     f"Pistas: {list(stems.keys())}"
                     f"  |  {total_stem_mb:.1f} MB"
                     f"  |  tardó {_time.monotonic()-t_stem:.2f}s")
            except Exception as e:
                _log_and_render('error', 'Separación', str(e))
                st.warning(f"Error al separar pistas: {e}")

        progress.empty()
        status.empty()
        live_log.empty()

        total_elapsed = round(_time.monotonic() - t0, 2)
        _log('ok', 'Completado', f"Proceso total: {total_elapsed}s")

        # Construir item y guardar en disco
        item_id = str(uuid.uuid4())
        item = {
            'id':             item_id,
            'filename':       pending['name'],
            'size_mb':        pending['size_mb'],
            'ext':            pending['ext'],
            'date_added':     datetime.now().strftime('%Y-%m-%d %H:%M'),
            'bpm':            results.get('tempo', {}).get('bpm', 0),
            'key':            results.get('key', {}).get('key', '—'),
            'key_en':         results.get('key', {}).get('key_en', '—'),
            'duration':       results.get('duration', 0),
            'stems_plan':     sel_stems,
            'model_name':     model_name,
            'processing_log': log,
            'results':        results,
            'stems':          stems,
            'audio_bytes':    audio_bytes,
        }

        _save_item(item)

        lib = st.session_state.get('library', [])
        lib.append(item)
        st.session_state['library'] = lib
        _save_library_index(lib)

        st.session_state.pop('pending_file', None)
        st.session_state.pop('instr_sel', None)

        st.session_state['page']      = 'detail'
        st.session_state['detail_id'] = item_id
        st.rerun()

    except Exception as e:
        _log('error', 'Excepción inesperada', str(e))
        st.error(f"Error inesperado: {e}")
        st.exception(e)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# PANTALLA 3 — DETALLE DEL ARCHIVO ANALIZADO
# ─────────────────────────────────────────────────────────────────────────────

def show_detail():
    item_id = st.session_state.get('detail_id')
    lib     = st.session_state.get('library', [])
    item    = next((x for x in lib if x['id'] == item_id), None)

    # ── Lazy-load: cargar audio + stems + results desde disco si no están en RAM ──
    if item is not None and item.get('results') is None:
        with st.spinner("Cargando datos del archivo..."):
            full = _load_item_from_disk(item_id)
            if full:
                for i, x in enumerate(lib):
                    if x['id'] == item_id:
                        lib[i] = full
                        break
                st.session_state['library'] = lib
                item = full

    if item is None:
        st.warning("Archivo no encontrado.")
        if st.button("← Volver a la biblioteca"):
            st.session_state['page'] = 'library'
            st.rerun()
        return

    col_back, col_info, col_del = st.columns([1, 8, 1])
    with col_back:
        if st.button("← Volver", key="back_detail"):
            st.session_state['page'] = 'library'
            st.rerun()
    with col_info:
        dur = item.get('duration', 0)
        ds  = f"{int(dur//60)}:{int(dur%60):02d}" if dur else '—'
        st.markdown(
            f"<div style='padding:3px 0'>"
            f"<span style='color:#ccc;font-size:.9rem;font-weight:600'>"
            f"{item['filename']}</span>"
            f"<span style='color:#2a2a2a;font-size:.73rem;margin-left:10px'>"
            f"{item.get('key','—')} &middot; {item.get('bpm','—')} BPM"
            f" &middot; {ds}</span></div>",
            unsafe_allow_html=True,
        )
    with col_del:
        if st.button("\U0001f5d1", key="del_item", help="Eliminar de la biblioteca"):
            _delete_item_from_disk(item_id)
            st.session_state['library'] = [x for x in lib if x['id'] != item_id]
            _save_library_index(st.session_state['library'])
            st.session_state['page'] = 'library'
            st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    results     = item.get('results')
    stems       = item.get('stems')
    audio_bytes = item.get('audio_bytes')

    if results is None:
        st.markdown(
            "<div style='background:#0a0a0a;border:1px solid #181818;border-radius:10px;"
            "padding:2rem;text-align:center;margin-top:1rem'>"
            "<p style='color:#444;margin:0'>Los datos de analisis no estan en memoria.</p>"
            "<p style='color:#252525;font-size:.8rem;margin:6px 0 0'>"
            "Esto no deberia ocurrir — los datos se guardan en disco automaticamente.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Reproductor integrado
    if audio_bytes:
        render_integrated_player(results, audio_bytes, stems)
    else:
        st.warning("No se pudo generar el reproductor.")

    # Descargas de pistas
    if stems:
        st.markdown(
            "<p style='color:#2e2e2e;font-size:.74rem;margin:12px 0 6px'>"
            "Descargar pistas</p>",
            unsafe_allow_html=True,
        )
        dl_cols = st.columns(len(stems))
        for col, (key, stem) in zip(dl_cols, stems.items()):
            with col:
                dl_data, dl_mime = _mp3_for_download(stem['wav_bytes'])
                dl_ext = 'mp3' if dl_mime == 'audio/mpeg' else 'wav'
                st.download_button(
                    label=stem['name_es'],
                    data=dl_data,
                    file_name=f"{key}.{dl_ext}",
                    mime=dl_mime,
                    key=f"dl_{key}_{item_id}",
                    use_container_width=True,
                )

    # KPIs — al final, compactos
    st.markdown(
        "<div style='height:18px'></div>"
        "<div style='border-top:1px solid #111;padding-top:14px;"
        "display:flex;gap:24px;flex-wrap:wrap;align-items:center'>",
        unsafe_allow_html=True,
    )
    d = results['duration']
    meta_items = [
        ("Tonalidad", results['key']['key_en']),
        ("Tempo",     f"{results['tempo']['bpm']} BPM"),
        ("Compás",    results['time_signature']['time_sig']),
        ("Duración",  f"{int(d//60)}:{int(d%60):02d}"),
    ]
    cols_meta = st.columns(len(meta_items))
    for col, (label, val) in zip(cols_meta, meta_items):
        with col:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div style='color:#252525;font-size:.62rem;text-transform:uppercase;"
                f"letter-spacing:1px'>{label}</div>"
                f"<div style='color:#555;font-size:.88rem;font-weight:600'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import shutil
    from stem_separation import check_demucs

    if 'page' not in st.session_state:
        st.session_state['page'] = 'library'
    if 'library' not in st.session_state:
        st.session_state['library'] = _load_library()

    demucs_ok, _ = check_demucs()
    ffmpeg_ok    = bool(shutil.which("ffmpeg"))

    page = st.session_state['page']

    # Breadcrumb para el nav bar
    if page == 'new':
        crumb = "<span style='color:#242424'> / Agregar</span>"
    elif page == 'instruments':
        pf  = st.session_state.get('pending_file', {})
        nm  = pf.get('name', '')
        nm  = (nm[:26] + '…') if len(nm) > 26 else nm
        crumb = f"<span style='color:#242424'> / {nm}</span>"
    elif page == 'detail':
        lib  = st.session_state.get('library', [])
        did  = st.session_state.get('detail_id')
        item = next((x for x in lib if x['id'] == did), None)
        nm   = (item.get('filename','') or '') if item else ''
        nm   = (nm[:26] + '…') if len(nm) > 26 else nm
        crumb = f"<span style='color:#242424'> / {nm}</span>"
    else:
        crumb = ""

    ff_dot = "\U0001f7e2" if ffmpeg_ok else "\U0001f534"
    dm_dot = "\U0001f7e2" if demucs_ok else "\U0001f534"

    model_ready = False
    if demucs_ok:
        try:
            from stem_separation import _MODEL_CACHE
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_ready = f'htdemucs_ft_{device}' in _MODEL_CACHE
        except Exception:
            pass
    model_dot  = "\U0001f7e2" if model_ready else "\U0001f7e1"
    model_lbl  = "modelo listo ⚡" if model_ready else "modelo cargando…"

    st.markdown(
        f"<div class='top-nav'>"
        f"<div class='top-nav-logo'>Aldo&amp;Klio"
        f"<span> Analyzer</span>"
        f"<span class='crumb'>{crumb}</span>"
        f"</div>"
        f"<div class='top-nav-status'>"
        f"<span>{ff_dot} ffmpeg</span>"
        f"<span>{dm_dot} AI stems</span>"
        f"<span>{model_dot} {model_lbl}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("**Aldo&Klio Analyzer**")
        st.caption("Analisis musical con IA")
        st.divider()
        if ffmpeg_ok: st.success("ffmpeg activo")
        else:         st.warning("ffmpeg no encontrado\n[Descargar](https://www.gyan.dev/ffmpeg/builds/)")
        if demucs_ok: st.success("AI stems activo")
        else:         st.warning("AI stems no instalado\n`pip install demucs`")
        st.divider()
        st.caption("Librosa · Demucs · Streamlit · Plotly")

    # Enrutamiento
    if page == 'library':
        show_library()
    elif page == 'new':
        show_new_file()
    elif page == 'instruments':
        show_select_instruments()
    elif page == 'detail':
        show_detail()
    else:
        st.session_state['page'] = 'library'
        st.rerun()


if __name__ == "__main__":
    main()
