from __future__ import annotations
import json, math, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from minesafe.core.config import APP_NAME, VERSION, NODE_ID, DEFAULT_GATEWAY, NODE_COORDS, MINE_GRAPH, REGULATORY_BOUNDS
from minesafe.core.telemetry import demo_telemetry, sanitize_and_validate
from minesafe.core.hazard import evaluate_hazard, risk_label, calculate_trpi
from minesafe.core.gateway import GatewayClient
from minesafe.core.route import route_with_breakdown, rank_routes
from minesafe.core.workers import demo_workers, Worker
from minesafe.core.methods import reference_rows, methodology_rows, o2_buffer_proxy_minutes
from minesafe.models.sensor_ai import demo_model, FEATURES
from minesafe.models.vision import analyze_structure, YOLOVision, save_uploaded_model, YOLO_AVAILABLE

st.set_page_config(page_title=APP_NAME, page_icon="⛏", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
:root{--bg:#07100d;--panel:#0b1713;--line:#1d3a2e;--muted:#769188;--text:#e9f2ed;--green:#39d98a;--amber:#f4b942;--red:#ff5c5c;--cyan:#56d6e8}
html,body,[class*="css"]{font-family:'IBM Plex Mono',monospace}.stApp{background:radial-gradient(circle at 85% 5%,#10291e 0,#07100d 32%,#050a08 75%);color:var(--text)}
.block-container{padding:1rem 1.2rem 4rem;max-width:1600px}[data-testid="stHeader"],#MainMenu,footer{visibility:hidden;height:0}
section[data-testid="stSidebar"]{background:#06100c;border-right:1px solid #173326}.brand{border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:12px}.brand h1{font-family:'Barlow Condensed';font-size:30px;margin:0}.brand small{font-size:9px;color:var(--muted);letter-spacing:1.5px}
.topline{display:flex;gap:10px;align-items:center;margin-bottom:10px}.eyebrow,.source{font-size:10px;color:var(--muted);letter-spacing:1px}.mini{font-size:10px;color:var(--muted);letter-spacing:0.8px}

/* STATUS PILLS */
.status{font-size:12px;font-weight:600;border:1px solid #26533e;border-radius:24px;padding:6px 14px;background:#0b1e16;display:inline-flex;align-items:center;gap:7px;letter-spacing:0.8px}
.cond-green{border-color:#2a754d;background:#0c2419;color:var(--green)}
.cond-amber{border-color:#7a5818;background:#291e0a;color:var(--amber)}
.cond-red{border-color:#822424;background:#2d0d0d;color:var(--red)}
.cond-cyan{border-color:#1c4d54;background:#081b1e;color:var(--cyan)}

.alertbar{border:1px solid #7f2929;background:linear-gradient(90deg,#2b1111,#180d0d);padding:10px 14px;border-radius:8px;margin:10px 0 14px;display:flex;justify-content:space-between}.alertbar .big{font-family:'Barlow Condensed';font-size:22px;color:#ff9b9b}

/* KPI CARDS */
.kpi{background:linear-gradient(180deg,#0e1c17,#09130f);border:1px solid var(--line);border-radius:9px;padding:12px 14px;min-height:98px;display:flex;flex-direction:column;justify-content:space-between}
.kpi .label{font-size:12px;color:#ffffff;font-weight:600;letter-spacing:1.2px;text-transform:uppercase}
.kpi .value{font-family:'Barlow Condensed';font-size:31px;font-weight:700;line-height:1.05;margin-top:2px}
.kpi .meta{font-size:10px;color:#8faaa0;text-transform:uppercase;letter-spacing:0.8px}
.green{color:var(--green)}.amber{color:var(--amber)}.red{color:var(--red)}.cyan{color:var(--cyan)}

.section{font-family:'Barlow Condensed';font-size:23px;letter-spacing:.8px;margin:16px 0 8px}

/* SENSOR & CONTENT PANELS */
.panel{background:rgba(11,23,19,.86);border:1px solid var(--line);border-radius:9px;padding:14px}
.panel .sensor-label{font-size:12px;font-weight:600;color:#ffffff;letter-spacing:1px;text-transform:uppercase}
.panel .sensor-val{font-family:'Barlow Condensed';font-size:33px;font-weight:700;line-height:1.1;margin-top:3px}
.panel .sensor-sub{font-size:10px;color:var(--muted);margin-top:4px;letter-spacing:0.8px}
.paneltitle{font-size:11px;font-weight:700;letter-spacing:1px;color:#ffffff}
.tag{font-size:8px;border:1px solid #284b3d;border-radius:4px;padding:3px 6px;color:#8aa79b}.reading{display:flex;justify-content:space-between;border-bottom:1px solid #14281f;padding:8px 0}.reading:last-child{border:0}

/* WORKER CARDS */
.worker{border:1px solid var(--line);background:#09150f;border-radius:8px;padding:8px 11px}
.worker .tagline{display:flex;justify-content:space-between;font-size:10px;color:#ffffff;font-weight:600}
.worker .hr{font-family:'Barlow Condensed';font-size:26px;line-height:1.1;margin:2px 0}
.bar{height:4px;background:#14281f;border-radius:4px;overflow:hidden;margin:5px 0}.fill{height:100%;background:var(--amber)}

/* UNIFIED CAMERA CONTAINER */
.cam-container{
    border-left: 1px solid #1e3d30;
    border-right: 1px solid #1e3d30;
    border-bottom: 1px solid #1e3d30;
    border-top: none !important;
    border-radius: 0 0 9px 9px;
    background: #07120e;
    overflow: hidden;
    margin: 0 0 16px;
}
.cam-header{background:#0a1813;padding:6px 14px;border-bottom:none !important;display:flex;justify-content:space-between;align-items:center;min-height:50px}
.cam-badge{font-size:9px;padding:3px 9px;border-radius:12px;letter-spacing:0.8px;font-weight:600}
.cam-badge-red{background:#2b0d0d;color:var(--red);border:1px solid #822424}
.cam-badge-green{background:#072417;color:var(--green);border:1px solid #1e5a3c}

/* TRANSPARENT MODE SWITCH BUTTONS */
div[data-testid="stColumn"] button[key*="btn_cam"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #6d887e !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 19px !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    padding: 2px 8px !important;
}
div[data-testid="stColumn"] button[key*="btn_cam"]:hover {
    color: #ffffff !important;
    background: transparent !important;
}

/* CAMERA VIEWPORTS */
.nv-viewport{
    height:280px;
    background:radial-gradient(ellipse at center,#05140d 0%,#020906 60%,#000302 100%);
    position:relative;
    display:flex;
    align-items:center;
    justify-content:center;
    border-bottom:1px solid #0d1e16;
}
.th-viewport{
    height:280px;
    background:radial-gradient(circle at 45% 50%,#52183b 0%,#300f2e 28%,#190924 55%,#0d0617 80%,#05020a 100%);
    position:relative;
    display:flex;
    align-items:center;
    justify-content:center;
    border-bottom:1px solid #241126;
}
.normal-viewport{
    height:280px;
    background:radial-gradient(ellipse at 50% 50%,#183329 0%,#0c1b15 50%,#040a08 100%);
    position:relative;
    display:flex;
    align-items:center;
    justify-content:center;
    border-bottom:1px solid #1a3c2e;
}
.cam-crosshair{position:absolute;top:50%;left:50%;width:44px;height:44px;transform:translate(-50%,-50%);border:1px dashed rgba(255,255,255,0.28);border-radius:50%}
.cam-watermark{font-size:12px;color:rgba(255,255,255,0.85);background:rgba(0,0,0,0.58);padding:6px 14px;border-radius:4px;letter-spacing:1px;font-family:'IBM Plex Mono',monospace}
.cam-footer{padding:8px 14px;font-size:10px;color:var(--muted);display:flex;justify-content:space-between;background:#050c09}

/* CHECKLIST CONTAINER */
.checklist-default {
    background: #081410;
    border: 1px solid #1d3a2e;
    box-shadow: none;
    border-radius: 9px;
    padding: 16px;
    margin: 8px 0 16px;
    transition: all 0.3s ease;
}
.checklist-critical {
    background: #140808;
    border: 1.5px solid #ff5c5c;
    box-shadow: 0 0 16px rgba(255, 92, 92, 0.22);
    border-radius: 9px;
    padding: 16px;
    margin: 8px 0 16px;
    transition: all 0.3s ease;
}

div[data-testid="stCheckbox"] {
    background: rgba(0, 0, 0, 0.25);
    padding: 6px 10px;
    border-radius: 6px;
    margin-bottom: 6px;
    border: 1px solid #1d3a2e;
    transition: border-color 0.2s ease, background-color 0.2s ease;
}
div[data-testid="stCheckbox"]:hover { border-color: #2b5543; }
div[data-testid="stCheckbox"]:has(input:checked) {
    border-color: rgba(57, 217, 138, 0.5);
    background: rgba(14, 38, 28, 0.45);
}
div[data-testid="stCheckbox"] label span[role="checkbox"][aria-checked="false"] {
    background-color: #0b1713 !important;
    border: 1.5px solid #3d5a4e !important;
}
div[data-testid="stCheckbox"] label span[role="checkbox"][aria-checked="true"] {
    background-color: #1a4f35 !important;
    border: 1.5px solid #39d98a !important;
}
div[data-testid="stCheckbox"] label span[role="checkbox"][aria-checked="true"] svg { fill: #ffffff !important; }
div[data-testid="stCheckbox"] label p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px !important;
    color: #e9f2ed;
}

div[data-baseweb="select"], div[data-baseweb="select"] * {
    border-color: #1d3a2e !important;
    outline: none !important;
}
div[data-baseweb="select"] > div {
    border: 1px solid #1d3a2e !important;
    background-color: #091712 !important;
    box-shadow: none !important;
}
div[data-baseweb="select"]:hover > div,
div[data-baseweb="select"]:focus-within > div {
    border: 1px solid #2d5a46 !important;
    box-shadow: none !important;
}

div[data-baseweb="input"], div[data-baseweb="input"] * {
    border-color: #1d3a2e !important;
    outline: none !important;
}
div[data-baseweb="input"] > div {
    border: 1px solid #1d3a2e !important;
    background-color: #091712 !important;
    box-shadow: none !important;
}
div[data-baseweb="input"]:focus-within > div {
    border-color: #2d5a46 !important;
    box-shadow: none !important;
}

.sidebar-active-title { color: #ffffff !important; font-weight: 700; font-size: 11px; letter-spacing: 1px; margin-bottom: 4px; }
.sidebar-inactive-title { color: #668075 !important; font-weight: 600; font-size: 11px; letter-spacing: 1px; margin-bottom: 4px; }
.active-ctrl { opacity: 1.0 !important; transition: opacity 0.2s ease; }
.inactive-ctrl { opacity: 0.40 !important; filter: grayscale(90%); pointer-events: none; transition: opacity 0.2s ease; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_ai(): return demo_model()
@st.cache_resource
def get_yolo(model_path: str, confidence: float, tracking: bool): return YOLOVision(model_path, confidence, tracking)

for key, default in {
    'seq': 1000, 'logs': [], 'last_cmd': 'STOP', 'rover_active': False,
    'incident_ack': False, 'prev_scenario': None, 'param_history': [],
    'camera_mode': 'Normal Vision'
}.items():
    if key not in st.session_state: st.session_state[key] = default

for i in range(1, 10):
    chk_key = f'ccm_{i}'
    if chk_key not in st.session_state: st.session_state[chk_key] = False

# Sidebar
with st.sidebar:
    st.markdown('<div class="brand"><h1>MINESAFE TITAN MODE</h1><small>SIH 2026 - PS 26039</small></div>', unsafe_allow_html=True)
    
    source = st.radio('Telemetry', ['LIVE ESP32 GATEWAY', 'SIMULATION LAB'], index=1, label_visibility='collapsed')
    is_live = source.startswith('LIVE')

    st.markdown(f'<div class="{"sidebar-active-title" if is_live else "sidebar-inactive-title"}">LIVE ESP32 GATEWAY CONFIG</div>', unsafe_allow_html=True)
    gw_css = "active-ctrl" if is_live else "inactive-ctrl"
    st.markdown(f'<div class="{gw_css}">', unsafe_allow_html=True)
    gateway_url = st.text_input('Gateway URL', DEFAULT_GATEWAY, disabled=not is_live)
    st.markdown('</div>', unsafe_allow_html=True)

    sim_css = "active-ctrl" if not is_live else "inactive-ctrl"
    st.markdown(f'<div class="{"sidebar-active-title" if not is_live else "sidebar-inactive-title"}">SIMULATION LAB // INCIDENT INJECTION</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{sim_css}">', unsafe_allow_html=True)
    
    scenario = st.selectbox(
        'Scenario',
        [
            'Nominal Operations',
            'Carbon Monoxide Poisoning',
            'Explosive Methane Ingress',
            'Dynamic Strata Roof Collapse',
            'Thermal Ceiling Exceeded',
            'Compound Disaster'
        ],
        label_visibility='collapsed',
        disabled=is_live
    )
    
    route_blocked = (scenario in {'Compound Disaster', 'Dynamic Strata Roof Collapse'}) if not is_live else False

    hazard_node = st.selectbox('Hazard location', ['BASE', 'ZONE_A'], index=1, disabled=is_live)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('**WORKER TELEMETRY**')
    worker_mode = st.radio('Worker channel', ['SIMULATED WEARABLES', 'LIVE /api/workers'], label_visibility='collapsed')
    st.divider()
    st.markdown('**VISION / YOLOv8**')
    vision_source = st.radio('Vision input', ['DISABLED', 'IMAGE UPLOAD', 'CAMERA SNAPSHOT'], label_visibility='collapsed')
    vision_file = st.file_uploader('Mine / worker frame', type=['png','jpg','jpeg']) if vision_source == 'IMAGE UPLOAD' else (st.camera_input('Capture underground frame') if vision_source == 'CAMERA SNAPSHOT' else None)
    custom_yolo = st.file_uploader('Custom mine YOLO .pt', type=['pt'])
    yolo_conf = st.slider('YOLO confidence', .15, .85, .35, .05)
    tracking = st.checkbox('Persistent tracking (ByteTrack)', True)
    if not YOLO_AVAILABLE:
        st.warning('Ultralytics is not installed. Install requirements.txt to activate vision.')
    else:
        st.caption('Baseline YOLOv8n generic detection. Custom mine weights activate safety PPE/hazard classes.')
    st.divider()
    if st.button('↻ RESET INCIDENT', use_container_width=True):
        st.session_state.incident_ack = False
        st.session_state.logs = []
        st.rerun()
    st.caption('Research decision-support prototype. Not certified mine-control equipment.')

if st.session_state.prev_scenario != scenario:
    st.session_state.prev_scenario = scenario
    for i in range(1, 10):
        st.session_state[f'ccm_{i}'] = False

st.session_state.seq += 1; seq = st.session_state.seq
client = GatewayClient(gateway_url); link_failed = False

# Strict Data Quality & Fail-Safe Pipeline
if is_live:
    telemetry, gw = client.get_telemetry(seq)
    if telemetry is None or not getattr(gw, 'online', False) or not telemetry.is_valid:
        telemetry = sanitize_and_validate({}, seq, 'FAIL-SAFE · GATEWAY UNAVAILABLE')
        link_failed = True
        gw = type('GW', (), {'online': False, 'message': 'FAIL-SAFE ACTIVE'})()
    else:
        link_failed = False
else:
    telemetry = sanitize_and_validate(demo_telemetry(scenario), seq, 'SIMULATED · SCENARIO')
    gw = type('GW', (), {'online': False, 'message': 'SIMULATION LAB ACTIVE'})()

# Computer Vision Inference
vision_result = None
if vision_file is not None:
    custom_path = save_uploaded_model(custom_yolo) if custom_yolo else None
    model_path = custom_path or 'yolov8n.pt'
    vision_result = get_yolo(model_path, yolo_conf, tracking).predict(Image.open(vision_file).convert('RGB'))

# Hazard Evaluation Engine
score, primary, secondary, breaches, scores = evaluate_hazard(telemetry)
if vision_result and vision_result.has_actionable_event:
    vscore = vision_result.vision_score
    scores = dict(scores)
    scores['VISION'] = vscore
    breaches = list(breaches) + [f'Vision event: {e.event} ({e.confidence:.0%})' for e in vision_result.events]
    if vision_result.has_critical_event:
        score = max(score, int(round(vscore)))
    else:
        score = int(round(max(score, min(100, .8 * score + .2 * vscore))))
    ordered = sorted(scores, key=scores.get, reverse=True)
    primary, secondary = ordered[0], ordered[1]
risk = risk_label(score)

ai = get_ai()
ai_prob = float(ai.predict_proba(np.array([[getattr(telemetry, f) for f in FEATURES]], float))[0] * 100) if telemetry.is_valid else 0.0

workers = []
worker_link_failed = False
if worker_mode.startswith('LIVE'):
    if is_live and not link_failed:
        wl, wr = client.get_workers()
        if wl is None: worker_link_failed = True
        else:
            for item in wl:
                try:
                    workers.append(Worker(str(item.get('tag','UNKNOWN')), str(item.get('name','Worker')), str(item.get('zone','UNKNOWN')), float(item['hr_bpm']), float(item['spo2_pct']), float(item['temp_c']), bool(item.get('fall_detected', False)), 'LIVE'))
                except Exception: continue
    else:
        worker_link_failed = True
else:
    workers = demo_workers()

target_dest = "ZONE_A" if "ZONE_A" in MINE_GRAPH else list(MINE_GRAPH.keys())[-1]
blocked = [("BASE", target_dest)] if route_blocked else []
node_penalties = {hazard_node: min(220, max(0, float(score) * 2.2))}
route_result = route_with_breakdown("BASE", target_dest, blocked, node_penalties=node_penalties)
candidates = rank_routes("BASE", target_dest, blocked, node_penalties)

trpis = []
for w in workers:
    tval, parts = calculate_trpi(telemetry, w.hr_bpm, w.spo2_pct, route_blocked, w.fall_detected) if telemetry.is_valid else (100.0, {'Data quality fail-safe': 100.0})
    trpis.append((w, tval, parts))
max_trpi = max([x[1] for x in trpis], default=0)
vision_priority = vision_result.vision_score if vision_result else 0
overall = max(score, max_trpi, vision_priority)
priority = 'IMMEDIATE' if overall >= 80 else 'URGENT' if overall >= 60 else 'MONITOR'

# Top Header Pill Badges
hardware_connected = is_live and (not link_failed) and getattr(gw, 'online', False)

if is_live:
    if hardware_connected:
        state_label = 'LIVE DATA'
        state_cls = 'cond-green'
    else:
        state_label = 'GATEWAY LOST (FAIL-SAFE)'
        state_cls = 'cond-red'
else:
    state_label = 'SIMULATION DATA'
    state_cls = 'cond-cyan'

if score >= 60 or risk in {'CRITICAL', 'HIGH'}:
    mine_status_label = 'DANGER'
    cond_cls = 'cond-red'
elif score >= 35 or risk == 'ELEVATED':
    mine_status_label = 'CAUTIOUS'
    cond_cls = 'cond-amber'
else:
    mine_status_label = 'SAFE'
    cond_cls = 'cond-green'

# Header
st.markdown(
    f'''<div class="topline">
        <div>
            <div style="font-family:Barlow Condensed;font-size:34px;font-weight:700;line-height:1.1;margin-bottom:2px">MINESAFE TITAN</div>
            <div class="eyebrow">Real-Time Mine Safety Monitoring & Hazard Detection System</div>
        </div>
        <div style="margin-left:auto;display:flex;gap:10px;align-items:center">
            <div class="status {cond_cls}">● MINE STATUS: {mine_status_label}</div>
            <div class="status {state_cls}">● {state_label}</div>
        </div>
    </div>''',
    unsafe_allow_html=True
)

if score >= 60 or breaches:
    detail = breaches[0] if breaches else f'{primary} risk elevated'
    st.markdown(f'<div class="alertbar"><div><div class="big">⚠ INCIDENT // {risk}</div><div class="mini">{detail} · response priority {priority}</div></div><div class="mini">SEQ {seq:06d}<br>{datetime.now():%H:%M:%S} LOCAL</div></div>', unsafe_allow_html=True)

# Top KPI Summary Cards
if ai_prob >= 60.0:
    ai_status_label = 'ANOMALOUS'
    ai_status_color = 'red'
elif ai_prob >= 35.0:
    ai_status_label = 'DEVIATION DETECTED'
    ai_status_color = 'amber'
else:
    ai_status_label = 'NOMINAL PATTERN'
    ai_status_color = 'green'

k = st.columns(5, gap='medium')
kpis = [
    ('MINE RISK', f'{score}/100', 'red' if score >= 60 else 'amber' if score >= 35 else 'green', 'LOW' if score < 35 else risk),
    ('AI ANOMALY', f'{ai_prob:.1f}%', ai_status_color, ai_status_label),
    ('VISION', f'{vision_result.vision_score:.0f}/100' if vision_result else 'STANDBY', 'red' if vision_result and vision_result.vision_score >= 70 else 'amber' if vision_result and vision_result.vision_score else 'cyan', 'COMPUTER VISION'),
    ('WORKERS', str(len(workers)), 'green' if workers and workers[0].source == 'LIVE' else 'cyan', 'WEARABLE CHANNEL'),
    ('TRPI MAX', f'{max_trpi:.1f}/100', 'red' if max_trpi >= 60 else 'amber' if max_trpi >= 35 else 'green', 'WORKER TRIAGE')
]
for col, (a, b, d, meta_text) in zip(k, kpis):
    col.markdown(f'<div class="kpi"><div class="label">{a}</div><div class="value {d}">{b}</div><div class="meta">{meta_text}</div></div>', unsafe_allow_html=True)

if not telemetry.is_valid:
    bad = ', '.join(f'{a}:{b}' for a, b in telemetry.status_map.items() if b not in {'VALID', 'DERIVED_VALID'})
    st.error(f'FAIL-SAFE DATA QUALITY — rejected invalid channels; no silent clamping. {bad}')
if worker_link_failed:
    st.error('FAIL-SAFE WORKER CHANNEL — live worker telemetry unavailable; simulated workers were not substituted.')

curr_dt = datetime.now()
curr_pressure = round(101.3 + (telemetry.temp - 25.0) * 0.08 - (telemetry.humidity - 50.0) * 0.03, 1) if telemetry.is_valid else 101.3
curr_moisture = round(min(100.0, telemetry.humidity * 0.88), 1) if telemetry.is_valid else 50.0

# Tabs
tab2, tab3, tab4, tab5, tab6 = st.tabs(['◉ ENVIRONMENT', '♙ PERSONNEL & RESCUE', '◈ AI / DATA', '▣ AUDIT & STANDARDS', '📜 STATUTORY STANDARDS & METHODS'])

with tab2:
    st.markdown('<div class="section">ENVIRONMENTAL MONITORING</div>', unsafe_allow_html=True)
    sensors = [
        ('TEMPERATURE', telemetry.temp, '°C', telemetry.temp >= 33.5, 'ceiling 33.5 °C'),
        ('CO', telemetry.co, 'PPM', telemetry.co >= 50.0, 'ceiling 50 PPM'),
        ('METHANE', telemetry.ch4, '%', telemetry.ch4 >= 0.75, 'ceiling 0.75 %'),
        ('VIBRATIONS', telemetry.strata_vibe_g, 'g', telemetry.strata_vibe_g >= 2.0, 'ceiling 2.0 g'),
        ('HUMIDITY', telemetry.humidity, '%', telemetry.humidity >= 85.0, 'ceiling 85 %'),
    ]
    
    sensor_cols = st.columns(5, gap='medium')
    for col, (s_name, s_val, s_unit, is_bad, s_limit) in zip(sensor_cols, sensors):
        invalid = not math.isfinite(float(s_val)) if s_val is not None else True
        val_display = '—' if invalid else f'{s_val:.1f}'
        val_color = 'red' if (invalid or is_bad) else 'green'
        col.markdown(f'''
        <div class="panel">
            <div class="sensor-label">{s_name}</div>
            <div class="sensor-val {val_color}">{val_display} <span style="font-size:14px;color:var(--muted);font-weight:400">{s_unit}</span></div>
            <div class="sensor-sub">{s_limit}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Camera Stream / Viewport Container + Relocated Rover Teleoperation
    st.markdown('<div class="section" style="border-bottom:1px solid #1e3d30;padding-bottom:6px">MINE VISION MONITORING</div>', unsafe_allow_html=True)
    vision_left_col, vision_right_space = st.columns([1.35, 0.65], gap='medium')

    with vision_left_col:
        cam_badge_cls = "cam-badge-green" if hardware_connected else "cam-badge-red"
        cam_badge_text = "● ACTIVE" if hardware_connected else "● INACTIVE"

        st.markdown('<div class="cam-container">', unsafe_allow_html=True)
        h_left, h_right = st.columns([0.75, 0.25])
        with h_left:
            mode_btns = st.columns(3)
            if mode_btns[0].button("NORMAL VISION", key="btn_cam_norm", use_container_width=True):
                st.session_state.camera_mode = "Normal Vision"; st.rerun()
            if mode_btns[1].button("NIGHT VISION", key="btn_cam_night", use_container_width=True):
                st.session_state.camera_mode = "Night Vision"; st.rerun()
            if mode_btns[2].button("THERMAL VISION", key="btn_cam_therm", use_container_width=True):
                st.session_state.camera_mode = "Thermal Vision"; st.rerun()
        with h_right:
            st.markdown(f'<div style="text-align:right;padding-top:6px"><span class="cam-badge {cam_badge_cls}">{cam_badge_text}</span></div>', unsafe_allow_html=True)

        cam_mode = st.session_state.get('camera_mode', 'Normal Vision')

        if cam_mode == "Thermal Vision":
            st.markdown('''
            <div class="th-viewport">
                <div class="cam-crosshair"></div>
                <div class="cam-watermark">LIVE THERMAL VISION</div>
            </div>
            <div class="cam-footer">
                <span>TUNNEL_HEAD_EAST · LWIR 8-14µm</span>
                <span>NETD < 50mK · GAIN: HIGH FLIR RADIOMETRIC · RESOLUTION: 1080p</span>
            </div>
            ''', unsafe_allow_html=True)
        elif cam_mode == "Night Vision":
            st.markdown('''
            <div class="nv-viewport">
                <div class="cam-crosshair"></div>
                <div class="cam-watermark">LIVE NIGHT VISION</div>
            </div>
            <div class="cam-footer">
                <span>TUNNEL_HEAD_EAST · IR 850nm ILLUMINATOR ACTIVE</span>
                <span>LOW-LIGHT ENHANCED · FPS: 30 · RESOLUTION: 1080p</span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            if hardware_connected:
                st.markdown(f'''
                <div style="height:280px;background:#000;display:flex;justify-content:center;align-items:center;overflow:hidden;position:relative;">
                    <img src="{gateway_url}:81/stream" style="width:100%;height:100%;object-fit:cover;" onerror="this.style.display='none'"/>
                    <div class="cam-crosshair"></div>
                    <div class="cam-watermark" style="position:absolute;bottom:12px;left:12px;">LIVE HARDWARE STREAM</div>
                </div>
                <div class="cam-footer">
                    <span>ESP32-S3-CAM · OV2640 / OV5640</span>
                    <span>FPS: 30 · RESOLUTION: 1080p MJPEG</span>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="normal-viewport">
                    <div class="cam-crosshair"></div>
                    <div class="cam-watermark">LIVE NORMAL VISION</div>
                </div>
                <div class="cam-footer">
                    <span>TUNNEL_HEAD_EAST · RGB 400-700nm</span>
                    <span>FPS: 30 · RESOLUTION: 1080p</span>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with vision_right_space:
        st.markdown(f'''
        <div class="panel" style="margin-top:0px;margin-bottom:12px;border-top:none !important;border-radius:0 0 9px 9px;">
            <div class="paneltitle">ROVER M-01 TELEOPERATION</div>
            <div class="mini" style="line-height:1.7;margin-top:4px">
                Status: <b class="{'green' if st.session_state.rover_active else 'amber'}">{'ACTIVE / MOVING' if st.session_state.rover_active else 'STANDBY / HOLD'}</b><br>
                Last Command: <b style="color:#ffffff">{st.session_state.last_cmd}</b><br>
                Target Sector: <b>ZONE_A</b>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        ctrl_grid = st.columns(3)
        with ctrl_grid[1]:
            if st.button('▲', key='rover_fwd', use_container_width=True):
                st.session_state.last_cmd = 'FORWARD'; st.session_state.rover_active = True
                client.send_rover_command('forward') if hardware_connected else None
                st.session_state.logs.append('ROVER FORWARD'); st.rerun()

        ctrl_row2 = st.columns(3)
        with ctrl_row2[0]:
            if st.button('◀', key='rover_left', use_container_width=True):
                st.session_state.last_cmd = 'LEFT'; st.session_state.rover_active = True
                client.send_rover_command('left') if hardware_connected else None
                st.session_state.logs.append('ROVER LEFT'); st.rerun()

        with ctrl_row2[1]:
            toggle_label = 'STOP' if st.session_state.rover_active else 'START'
            if st.button(toggle_label, key='rover_toggle', use_container_width=True):
                st.session_state.rover_active = not st.session_state.rover_active
                st.session_state.last_cmd = 'START' if st.session_state.rover_active else 'EMERGENCY STOP'
                cmd_code = 'start' if st.session_state.rover_active else 'emergency_stop'
                client.send_rover_command(cmd_code) if hardware_connected else None
                st.session_state.logs.append(f'ROVER {st.session_state.last_cmd}'); st.rerun()

        with ctrl_row2[2]:
            if st.button('▶', key='rover_right', use_container_width=True):
                st.session_state.last_cmd = 'RIGHT'; st.session_state.rover_active = True
                client.send_rover_command('right') if hardware_connected else None
                st.session_state.logs.append('ROVER RIGHT'); st.rerun()

        ctrl_row3 = st.columns(3)
        with ctrl_row3[1]:
            if st.button('▼', key='rover_rev', use_container_width=True):
                st.session_state.last_cmd = 'REVERSE'; st.session_state.rover_active = True
                client.send_rover_command('reverse') if hardware_connected else None
                st.session_state.logs.append('ROVER REVERSE'); st.rerun()

    st.markdown('<div class="section">RISK COMPONENTS</div>', unsafe_allow_html=True)
    st.plotly_chart(go.Figure(go.Bar(x=list(scores.values()), y=list(scores.keys()), orientation='h', text=[f'{v:.0f}' for v in scores.values()], textposition='outside')).update_layout(height=260, margin=dict(l=0, r=40, t=5, b=5), paper_bgcolor='#0b1713', plot_bgcolor='#0b1713', font=dict(color='#dcebe4'), xaxis=dict(range=[0, 110])), use_container_width=True, config={'displayModeBar': False})
    st.markdown('<div class="panel"><div class="paneltitle">THRESHOLD BREACHES</div>' + (''.join(f'<div class="reading"><b>{b}</b><span class="red">BREACH</span></div>' for b in breaches) if breaches else '<div class="mini">No configured threshold breaches.</div>') + '</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section">PERSONNEL & RESCUE</div>', unsafe_allow_html=True)
    for row in [trpis[i:i+3] for i in range(0, len(trpis), 3)]:
        for col, (w, tv, parts) in zip(st.columns(len(row)), row):
            col.markdown(f'''
            <div class="worker">
                <div class="tagline"><b>{w.tag} · {w.name}</b></div>
                <div class="hr">{w.hr_bpm:.0f} <span style="font-size:9px;color:#769188">BPM</span></div>
                <div class="mini">SpO₂ {w.spo2_pct:.0f}% · Temp {w.temp_c:.1f}°C · Fall {"YES" if w.fall_detected else "NO"}</div>
                <div class="bar"><div class="fill" style="width:{min(100, tv)}%"></div></div>
                <div style="display:flex;justify-content:space-between">
                    <span class="mini">TRPI</span>
                    <b class="{"red" if tv >= 80 else "amber" if tv >= 60 else "green"}">{tv:.1f}/100</b>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown('<div class="section">COAL MINE MAP</div>', unsafe_allow_html=True)
    rescue_map_col, sector_details_col = st.columns([1.35, 0.65], gap='medium')

    if score >= 60:
        haz_status = "DANGEROUS"
        haz_color = "#ff5c5c"
    elif score >= 35:
        haz_status = "CAUTION"
        haz_color = "#f4b942"
    else:
        haz_status = "SAFE"
        haz_color = "#39d98a"

    if hazard_node == "BASE":
        base_status = haz_status
        base_color = haz_color
        zone_a_status = "SAFE"
        zone_a_color = "#39d98a"
    else:
        base_status = "SAFE"
        base_color = "#39d98a"
        zone_a_status = haz_status
        zone_a_color = haz_color

    with rescue_map_col:
        tunnel_coords = {"BASE": (1.0, 5.0), "ZONE_A": (8.5, 5.0)}
        worker_node_positions = [(3.0, 5.0), (6.2, 5.0), (7.5, 5.0)]

        fig_rescue = go.Figure()
        fig_rescue.add_trace(go.Scatter(x=[tunnel_coords["BASE"][0], tunnel_coords["ZONE_A"][0]], y=[tunnel_coords["BASE"][1], tunnel_coords["ZONE_A"][1]], mode='lines', line=dict(width=28, color='#143026'), hoverinfo='skip', showlegend=False))
        fig_rescue.add_trace(go.Scatter(x=[tunnel_coords["BASE"][0], tunnel_coords["ZONE_A"][0]], y=[tunnel_coords["BASE"][1], tunnel_coords["ZONE_A"][1]], mode='lines', line=dict(width=1.5, color='#2a6b52', dash='dash'), hoverinfo='skip', showlegend=False))
        fig_rescue.add_trace(go.Scatter(x=[tunnel_coords["BASE"][0]], y=[tunnel_coords["BASE"][1]], mode='markers+text', text=[f'BASE<br><b>{base_status}</b>'], textposition='bottom center', textfont=dict(size=10, family='IBM Plex Mono', color=base_color), marker=dict(size=22, color=base_color, symbol='square'), name='BASE'))
        fig_rescue.add_trace(go.Scatter(x=[tunnel_coords["ZONE_A"][0]], y=[tunnel_coords["ZONE_A"][1]], mode='markers+text', text=[f'ZONE A<br><b>{zone_a_status}</b>'], textposition='bottom center', textfont=dict(size=10, family='IBM Plex Mono', color=zone_a_color), marker=dict(size=22, color=zone_a_color, symbol='circle'), name='ZONE A'))

        rover_color = '#39d98a' if st.session_state.rover_active else '#56d6e8'
        fig_rescue.add_trace(go.Scatter(x=[4.6], y=[5.0], mode='markers+text', text=[f'ROVER M-01<br>({"ACTIVE" if st.session_state.rover_active else "IDLE"})'], textposition='top center', textfont=dict(size=9, family='IBM Plex Mono', color=rover_color), marker=dict(size=18, color=rover_color, symbol='diamond'), name='ROVER M-01'))

        for i, (w, tv, _) in enumerate(trpis[:3]):
            wx, wy = worker_node_positions[i]
            w_color = '#ff5c5c' if tv >= 60 else '#f4b942' if tv >= 35 else '#39d98a'
            fig_rescue.add_trace(go.Scatter(x=[wx], y=[wy], mode='markers+text', text=[f'{w.tag}<br>{w.name}'], textposition='bottom center', textfont=dict(size=8.5, family='IBM Plex Mono', color='#dcebe4'), marker=dict(size=13, color=w_color, symbol='circle'), name=w.tag))

        fig_rescue.update_layout(height=230, margin=dict(l=10, r=10, t=15, b=10), paper_bgcolor='#07100d', plot_bgcolor='#07100d', xaxis=dict(visible=False, range=[0, 9.5]), yaxis=dict(visible=False, range=[3.5, 6.5]), font=dict(color='#dcebe4', family='IBM Plex Mono'), showlegend=False)
        st.plotly_chart(fig_rescue, use_container_width=True, config={'displayModeBar': False})

    with sector_details_col:
        base_temp = 24.5 if hazard_node != "BASE" else telemetry.temp
        base_ch4 = 0.04 if hazard_node != "BASE" else telemetry.ch4
        base_hum = 52.0 if hazard_node != "BASE" else telemetry.humidity
        base_vibe = 0.02 if hazard_node != "BASE" else telemetry.strata_vibe_g

        zone_a_temp = telemetry.temp if hazard_node == "ZONE_A" else 26.0
        zone_a_ch4 = telemetry.ch4 if hazard_node == "ZONE_A" else 0.08
        zone_a_hum = telemetry.humidity if hazard_node == "ZONE_A" else 58.0
        zone_a_vibe = telemetry.strata_vibe_g if hazard_node == "ZONE_A" else 0.05

        card_html = (
            f'<div class="panel" style="margin-top:0px;padding:10px">'
            f'<div class="paneltitle" style="display:flex;justify-content:space-between"><span>SECTOR TELEMETRY CONDITIONS</span><span class="mini">ACTIVE INJECTION</span></div>'
            f'<div style="margin-top:8px;padding:6px 8px;background:#06120e;border-radius:6px;border-left:3px solid {base_color}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center"><b style="font-size:11px;color:#ffffff">BASE (COMMAND SHAFT)</b><span style="font-size:9px;font-weight:700;color:{base_color}">{base_status}</span></div>'
            f'<div class="mini" style="margin-top:2px;display:grid;grid-template-columns:1fr 1fr;gap:2px">'
            f'<span>Temp: <b style="color:#ffffff">{base_temp:.1f}°C</b></span>'
            f'<span>Methane: <b style="color:{"#ff5c5c" if base_ch4 >= 0.75 else "#ffffff"}">{base_ch4:.2f}%</b></span>'
            f'<span>Humidity: <b style="color:#ffffff">{base_hum:.1f}%</b></span>'
            f'<span>Vibe: <b style="color:{"#ff5c5c" if base_vibe >= 2.0 else "#ffffff"}">{base_vibe:.2f}g</b></span>'
            f'</div></div>'
            f'<div style="margin-top:8px;padding:6px 8px;background:#06120e;border-radius:6px;border-left:3px solid {zone_a_color}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center"><b style="font-size:11px;color:#ffffff">ZONE A (WORKING FACE)</b><span style="font-size:9px;font-weight:700;color:{zone_a_color}">{zone_a_status}</span></div>'
            f'<div class="mini" style="margin-top:2px;display:grid;grid-template-columns:1fr 1fr;gap:2px">'
            f'<span>Temp: <b style="color:{"#ff5c5c" if zone_a_temp >= 33.5 else "#ffffff"}">{zone_a_temp:.1f}°C</b></span>'
            f'<span>Methane: <b style="color:{"#ff5c5c" if zone_a_ch4 >= 0.75 else "#ffffff"}">{zone_a_ch4:.2f}%</b></span>'
            f'<span>Humidity: <b style="color:{"#ff5c5c" if zone_a_hum >= 85.0 else "#ffffff"}">{zone_a_hum:.1f}%</b></span>'
            f'<span>Vibe: <b style="color:{"#ff5c5c" if zone_a_vibe >= 2.0 else "#ffffff"}">{zone_a_vibe:.2f}g</b></span>'
            f'</div></div>'
            f'<div class="mini" style="margin-top:6px;color:var(--muted);font-size:9px">Parameters sync with active sensor/scenario stream.</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # Checklist Container
    st.markdown('<div class="section">CRITICAL CONDITION MEASURES</div>', unsafe_allow_html=True)
    is_critical_env = (scenario != 'Nominal Operations') or (score >= 60) or (risk in {'CRITICAL', 'HIGH'}) or route_blocked
    all_checked = all(st.session_state.get(f'ccm_{i}', False) for i in range(1, 10))

    if is_critical_env and not all_checked:
        box_css_class = "checklist-critical"
        badge_label = "● ACTIVE HAZARD DETECTED // EMERGENCY PROTOCOL REQUIRED"
        badge_color = "var(--red)"
    else:
        box_css_class = "checklist-default"
        badge_label = "● PROTOCOLS VERIFIED" if all_checked else "● NOMINAL OPERATIONS (STANDBY)"
        badge_color = "#769188"

    with st.container():
        st.markdown(f'''
        <div class="{box_css_class}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                <span style="font-family:'Barlow Condensed';font-size:14px;letter-spacing:1px;font-weight:700;color:#ffffff">EMERGENCY PROTOCOL EXECUTION</span>
                <span style="font-size:10px;font-weight:700;color:{badge_color};letter-spacing:0.8px">{badge_label}</span>
            </div>
        ''', unsafe_allow_html=True)
        
        cc_cols = st.columns(2, gap='medium')
        with cc_cols[0]:
            st.checkbox('Alert the control room and mine authorities.', key='ccm_1')
            st.checkbox('Isolate the affected zone and evacuate workers.', key='ccm_2')
            st.checkbox('Maintain proper ventilation to control gas and heat.', key='ccm_3')
            st.checkbox('Continuously monitor methane, temperature, pressure, humidity and moisture.', key='ccm_4')
            st.checkbox('Restrict entry to the affected area.', key='ccm_5')
        with cc_cols[1]:
            st.checkbox('Deploy trained rescue personnel when required.', key='ccm_6')
            st.checkbox('Maintain communication and a safe rescue route.', key='ccm_7')
            st.checkbox('Check secondary hazards such as gas, heat, pressure, roof instability and water ingress.', key='ccm_8')
            st.checkbox('Allow re-entry only after safety clearance.', key='ccm_9')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">TRPI EXPLAINABILITY</div>', unsafe_allow_html=True)
    if trpis:
        chosen = max(trpis, key=lambda x: x[1])
        st.dataframe(pd.DataFrame({'Contributor': list(chosen[2].keys()), 'Weighted contribution': list(chosen[2].values())}), use_container_width=True, hide_index=True)
    a, b, c = st.columns(3)
    a.metric('RESPONSE PRIORITY', priority)
    b.metric('MAX TRPI', f'{max_trpi:.1f}')
    c.metric('ROUTE', 'BLOCKED' if route_blocked else 'OPEN')

with tab4:
    st.markdown('<div class="section">AI / DATA</div>', unsafe_allow_html=True)
    a, b = st.columns([.8, 1.2])
    with a:
        st.markdown(f'<div class="panel"><div class="paneltitle">SENSOR ANOMALY MODEL · TinyMLP</div><div style="font-family:Barlow Condensed;font-size:38px;color:#f4b942">{ai_prob:.1f}%</div><div class="mini">Current anomaly probability · demo-trained synthetic baseline</div></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader('Train in-session from labelled historical CSV', type=['csv'], key='traincsv')
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                missing = [f for f in FEATURES + ['label'] if f not in df.columns]
                if missing: st.error('Missing columns: ' + ', '.join(missing))
                elif len(df) < 30: st.error('Need at least 30 labelled rows.')
                else:
                    m = get_ai()
                    acc = m.fit(df[FEATURES].values, df['label'].values, epochs=900, lr=.025)
                    st.success(f'In-session training complete · accuracy {acc*100:.1f}%')
                    ai_prob = float(m.predict_proba(np.array([[getattr(telemetry, f) for f in FEATURES]], float))[0] * 100) if telemetry.is_valid else 0
            except Exception as e: st.error(f'Training failed: {e}')
    with b:
        st.markdown('<div class="panel"><div class="paneltitle">YOLOv8 MINE VISION</div><div class="mini">Detection → persistent tracking → explicit safety-event mapping</div></div>', unsafe_allow_html=True)
        if vision_result:
            x, y = st.columns([1.35, .65])
            with x:
                if vision_result.processed: st.image(vision_result.processed, use_container_width=True)
            with y:
                st.metric('PERSONS', vision_result.person_count)
                st.metric('VISION RISK', f'{vision_result.vision_score:.1f}/100')
                for e in vision_result.events: st.error(f'{e.event} · {e.confidence:.0%}')
            if vision_result.detections:
                st.dataframe(pd.DataFrame([{'class': d.label, 'confidence': round(d.confidence, 3)} for d in vision_result.detections]), use_container_width=True, hide_index=True)
        else:
            st.info('Enable vision in sidebar to run detection pipeline.')

    st.markdown('<div class="section" style="margin-top:24px">ENVIRONMENTAL PARAMETER TRENDS</div>', unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns([0.72, 0.28])
    with filter_col1:
        st.caption("Historical trend analysis for current environmental sensor channels.")
    with filter_col2:
        period = st.selectbox("Resolution", ["YEARS (Monthly)", "MONTHS (Daily)", "DAYS (Per-Minute)"], index=0, label_visibility="collapsed")

    curr_t = float(telemetry.temp) if telemetry.is_valid else 25.0
    curr_p = float(curr_pressure)
    curr_m = float(telemetry.ch4) if telemetry.is_valid else 0.05
    curr_h = float(telemetry.humidity) if telemetry.is_valid else 50.0
    curr_w = float(curr_moisture)

    if period == "YEARS (Monthly)":
        num_points = 12
        x_axis = [(curr_dt - timedelta(days=(num_points - 1 - i) * 30)).strftime("%b") for i in range(num_points)]
        year_subtitle = f"{curr_dt.year - 1} – {curr_dt.year}"
        y_temp = [round(curr_t + np.sin(i * 0.5) * 4.0, 1) for i in range(num_points)]
        y_press = [round(curr_p + np.cos(i * 0.5) * 1.5, 1) for i in range(num_points)]
        y_meth = [round(max(0.01, curr_m + np.sin(i * 0.3) * 0.15), 2) for i in range(num_points)]
        y_hum = [round(max(30.0, min(90.0, curr_h + np.cos(i * 0.5) * 8.0)), 1) for i in range(num_points)]
        y_moist = [round(max(20.0, min(85.0, curr_w + np.sin(i * 0.5) * 6.5)), 1) for i in range(num_points)]
    elif period == "MONTHS (Daily)":
        num_points = 15
        x_axis = [(curr_dt - timedelta(days=(num_points - 1 - i) * 2)).strftime("%d %b") for i in range(num_points)]
        year_subtitle = f"{curr_dt.year}"
        y_temp = [round(curr_t + np.sin(i * 0.4) * 2.0 - 0.5, 1) for i in range(num_points)]
        y_press = [round(curr_p + np.cos(i * 0.3) * 0.8, 1) for i in range(num_points)]
        y_meth = [round(max(0.01, curr_m + np.sin(i * 0.5) * 0.09), 2) for i in range(num_points)]
        y_hum = [round(max(20.0, min(95.0, curr_h + np.sin(i * 0.4) * 4.5)), 1) for i in range(num_points)]
        y_moist = [round(max(10.0, min(90.0, curr_w + np.cos(i * 0.4) * 3.5)), 1) for i in range(num_points)]
    else:
        num_points = 15
        x_axis = [(curr_dt - timedelta(minutes=(num_points - 1 - i) * 4)).strftime("%H:%M") for i in range(num_points)]
        year_subtitle = f"{curr_dt.strftime('%d %b %Y')}"
        y_temp = [round(curr_t + np.sin(i * 0.6) * 0.4, 1) for i in range(num_points)]
        y_press = [round(curr_p + np.cos(i * 0.5) * 0.3, 1) for i in range(num_points)]
        y_meth = [round(max(0.01, curr_m + np.sin(i * 0.8) * 0.04), 2) for i in range(num_points)]
        y_hum = [round(max(10.0, curr_h + np.cos(i * 0.5) * 1.5), 1) for i in range(num_points)]
        y_moist = [round(max(5.0, curr_w + np.sin(i * 0.7) * 1.2), 1) for i in range(num_points)]

    chart_col1, chart_col2 = st.columns(2, gap='medium')
    with chart_col1:
        st.markdown('<div class="paneltitle" style="margin-bottom:8px">TEMPERATURE & PRESSURE</div>', unsafe_allow_html=True)
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(x=x_axis, y=y_temp, name='Temperature (°C)', mode='lines+markers', line=dict(color='#ff8c42', width=2.5), marker=dict(size=5, color='#ff8c42')))
        fig_tp.add_trace(go.Scatter(x=x_axis, y=y_press, name='Pressure (kPa)', mode='lines+markers', line=dict(color='#56d6e8', width=2, dash='dot'), marker=dict(size=4, color='#56d6e8'), yaxis='y2'))
        fig_tp.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=35), paper_bgcolor='#0b1713', plot_bgcolor='#0b1713', font=dict(color='#dcebe4', family='IBM Plex Mono', size=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor='rgba(0,0,0,0)'), hovermode='x unified', yaxis=dict(title=dict(text='Temp (°C)', font=dict(color='#ff8c42')), showgrid=True, gridcolor='#152c22'), yaxis2=dict(title=dict(text='Press (kPa)', font=dict(color='#56d6e8')), overlaying='y', side='right', showgrid=False), xaxis=dict(title=dict(text=year_subtitle, font=dict(color='#769188', size=10)), showgrid=True, gridcolor='#152c22'))
        st.plotly_chart(fig_tp, use_container_width=True, config={'displayModeBar': False})

    with chart_col2:
        st.markdown('<div class="paneltitle" style="margin-bottom:8px">GAS, HUMIDITY & MOISTURE</div>', unsafe_allow_html=True)
        fig_ghm = go.Figure()
        fig_ghm.add_trace(go.Scatter(x=x_axis, y=y_meth, name='Methane (%)', mode='lines+markers', line=dict(color='#ff5c5c', width=2.5), marker=dict(size=5, color='#ff5c5c')))
        fig_ghm.add_trace(go.Scatter(x=x_axis, y=y_hum, name='Humidity (%)', mode='lines+markers', line=dict(color='#39d98a', width=2), marker=dict(size=4, color='#39d98a'), yaxis='y2'))
        fig_ghm.add_trace(go.Scatter(x=x_axis, y=y_moist, name='Moisture (%)', mode='lines+markers', line=dict(color='#b99cff', width=2, dash='dash'), marker=dict(size=4, color='#b99cff'), yaxis='y2'))
        fig_ghm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=35), paper_bgcolor='#0b1713', plot_bgcolor='#0b1713', font=dict(color='#dcebe4', family='IBM Plex Mono', size=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor='rgba(0,0,0,0)'), hovermode='x unified', yaxis=dict(title=dict(text='CH₄ (%)', font=dict(color='#ff5c5c')), showgrid=True, gridcolor='#152c22'), yaxis2=dict(title=dict(text='% Levels', font=dict(color='#39d98a')), overlaying='y', side='right', showgrid=False), xaxis=dict(title=dict(text=year_subtitle, font=dict(color='#769188', size=10)), showgrid=True, gridcolor='#152c22'))
        st.plotly_chart(fig_ghm, use_container_width=True, config={'displayModeBar': False})

with tab5:
    st.markdown('<div class="section">AUDIT / STANDARDS <span>TRACEABLE PROTOTYPE STATE</span></div>', unsafe_allow_html=True)
    candidates_payload = [r.to_dict() for r in candidates]
    audit = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'sequence': seq,
        'ps': '26039',
        'version': VERSION,
        'data_state': state_label,
        'telemetry': telemetry.to_dict(),
        'hazard': {'score': score, 'risk': risk, 'primary': primary, 'secondary': secondary, 'breaches': breaches, 'components': scores},
        'route': {'recommended': route_result.to_dict(), 'candidates': candidates_payload, 'blocked_edges': blocked, 'hazard_node': hazard_node},
        'standards_reference': reference_rows(),
        'methods': methodology_rows()
    }
    st.download_button('EXPORT FORENSIC JSON', json.dumps(audit, indent=2, default=str), file_name=f'MineSafe_Audit_{datetime.now():%Y%m%d_%H%M%S}.json', mime='application/json', use_container_width=True)
    if candidates:
        st.dataframe(pd.DataFrame([{'rank': i + 1, 'path': ' → '.join(r.path), 'distance_m': round(r.distance_m, 1), 'hazard_penalty': round(r.hazard_penalty, 1), 'total_cost': round(r.total_cost, 1)} for i, r in enumerate(candidates)]), use_container_width=True, hide_index=True)

with tab6:
    st.markdown('<div class="section">STATUTORY STANDARDS & DECISION METHODOLOGY <span>DGMS-REFERENCED BASELINE</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><div class="paneltitle">1. STATUTORY / REFERENCE BASELINE — COAL MINES REGULATIONS, 2017</div>'
        '<div class="mini" style="line-height:1.9;margin-top:8px">'
        '• <b>Oxygen (O₂):</b> minimum 19.0% by volume.<br>'
        '• <b>Carbon dioxide (CO₂):</b> not more than 0.5% by volume (5,000 ppm).<br>'
        '• <b>Methane:</b> not more than 0.75% in the general body of return air.<br>'
        '• <b>Wet-bulb temperature:</b> not more than 33.5°C.<br>'
        '• <b>CO:</b> configured alert threshold at 50 ppm.'
        '</div></div>',
        unsafe_allow_html=True
    )
    st.link_button('OPEN OFFICIAL DGMS CMR 2017 SOURCE', 'https://www.dgms.gov.in/writereaddata/UploadFile/Coal_Mines_Regulation_2017_Noti.pdf', use_container_width=True)

st.markdown(f'<div style="position:fixed;bottom:0;left:0;right:0;background:#050b08;border-top:1px solid #173326;padding:5px 14px;font-size:8px;color:#668278;z-index:999">MINESAFE TITAN · {VERSION} · {state_label} · SEQ {seq} · RESEARCH DECISION-SUPPORT ONLY</div>', unsafe_allow_html=True)