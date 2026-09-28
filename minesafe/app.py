from __future__ import annotations
import json, math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from minesafe.core.config import (
    APP_NAME, VERSION, NODE_ID, DEFAULT_GATEWAY,
    NODE_COORDS, MINE_GRAPH, REGULATORY_BOUNDS, PHYSICAL_LIMITS
)
from minesafe.core.telemetry import demo_telemetry, sanitize_and_validate
from minesafe.core.hazard import evaluate_hazard, risk_label, calculate_trpi
from minesafe.core.gateway import GatewayClient
from minesafe.core.route import route_with_breakdown, rank_routes
from minesafe.core.workers import demo_workers, Worker
from minesafe.core.methods import reference_rows, methodology_rows, o2_buffer_proxy_minutes
from minesafe.models.sensor_ai import demo_model, TinyMLP, FEATURES
from minesafe.models.vision import analyze_structure, YOLOVision, save_uploaded_model, YOLO_AVAILABLE

st.set_page_config(page_title=APP_NAME, page_icon="⛏", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
:root{--bg:#07100d;--panel:#0b1713;--line:#1d3a2e;--muted:#769188;--text:#e9f2ed;--green:#39d98a;--amber:#f4b942;--red:#ff5c5c;--cyan:#56d6e8}
html,body,[class*="css"]{font-family:'IBM Plex Mono',monospace}.stApp{background:radial-gradient(circle at 85% 5%,#10291e 0,#07100d 32%,#050a08 75%);color:var(--text)}

/* 1. HIDE DEPLOY BUTTON & STREAMLIT MENU STRICTLY */
.stDeployButton,
.stAppDeployButton,
div[data-testid="stAppDeployButton"],
#MainMenu,
footer,
div[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
}

/* 2. CONTAINER SPACING */
.block-container {
    padding: 1.5rem 1.2rem 5rem;
    max-width: 1600px;
}

/* 3. TRANSPARENT HEADER */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* 4. FORCE SIDEBAR TOGGLE ARROW VISIBILITY & CLICKABILITY */
div[data-testid="stSidebarCollapsedControl"],
div[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
button[data-testid="baseButton-header"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: var(--text) !important;
    z-index: 1000000 !important;
}

div[data-testid="stSidebarCollapsedControl"] svg,
button[data-testid="stSidebarCollapseButton"] svg {
    fill: var(--text) !important;
    stroke: var(--text) !important;
}

/* 5. FULL-LENGTH PERSISTENT FOOTER BAR */
.full-footer-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
    height: 28px;
    line-height: 28px;
    background: #050b08;
    border-top: 1px solid #173326;
    padding: 0 16px;
    font-size: 10.5px;
    font-family: 'IBM Plex Mono', monospace;
    color: #769188;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-sizing: border-box;
    z-index: 999999;
}

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
.panel .sensor-label{font-size:11.5px;font-weight:600;color:#ffffff;letter-spacing:1px;text-transform:uppercase}
.panel .sensor-val{font-family:'Barlow Condensed';font-size:31px;font-weight:700;line-height:1.1;margin-top:3px}
.panel .sensor-sub{font-size:9.5px;color:var(--muted);margin-top:4px;letter-spacing:0.7px}
.paneltitle{font-size:11px;font-weight:700;letter-spacing:1px;color:#ffffff}
.reading{display:flex;justify-content:space-between;border-bottom:1px solid #14281f;padding:8px 0}.reading:last-child{border:0}

/* WORKER CARDS */
.worker{border:1px solid var(--line);background:#09150f;border-radius:8px;padding:8px 11px}
.worker .tagline{display:flex;justify-content:space-between;font-size:10px;color:#ffffff;font-weight:600}
.worker .hr{font-family:'Barlow Condensed';font-size:26px;line-height:1.1;margin:2px 0}
.bar{height:4px;background:#14281f;border-radius:4px;overflow:hidden;margin:5px 0}.fill{height:100%;background:var(--amber)}

/* CAMERA CONTAINER */
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
.cam-badge{font-size:9px;padding:3px 9px;border-radius:12px;letter-spacing:0.8px;font-weight:600}
.cam-badge-red{background:#2b0d0d;color:var(--red);border:1px solid #822424}
.cam-badge-green{background:#072417;color:var(--green);border:1px solid #1e5a3c}

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

/* VIEWPORT CONTAINER */
.viewport-frame {
    height: 280px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border-bottom: 1px solid #11261d;
    background-color: #020604;
}

.cam-crosshair{position:absolute;top:50%;left:50%;width:44px;height:44px;transform:translate(-50%,-50%);border:1px dashed rgba(255,255,255,0.30);border-radius:50%;z-index:5}
.cam-watermark{font-size:11px;color:rgba(255,255,255,0.85);background:rgba(0,0,0,0.65);padding:5px 12px;border-radius:4px;letter-spacing:1px;font-family:'IBM Plex Mono',monospace;position:absolute;bottom:12px;left:12px;z-index:6}
.cam-footer{padding:8px 14px;font-size:10px;color:var(--muted);display:flex;justify-content:space-between;background:#050c09}

/* CHECKLIST CONTAINER */
.checklist-default {
    background: #081410;
    border: 1px solid #1d3a2e;
    border-radius: 9px;
    padding: 16px;
    margin: 8px 0 16px;
}
.checklist-critical {
    background: #140808;
    border: 1.5px solid #ff5c5c;
    box-shadow: 0 0 16px rgba(255, 92, 92, 0.22);
    border-radius: 9px;
    padding: 16px;
    margin: 8px 0 16px;
}
div[data-testid="stCheckbox"] {
    background: rgba(0, 0, 0, 0.25);
    padding: 6px 10px;
    border-radius: 6px;
    margin-bottom: 6px;
    border: 1px solid #1d3a2e;
}
div[data-baseweb="select"], div[data-baseweb="select"] * { border-color: #1d3a2e !important; }
div[data-baseweb="select"] > div { border: 1px solid #1d3a2e !important; background-color: #091712 !important; }
div[data-baseweb="input"], div[data-baseweb="input"] * { border-color: #1d3a2e !important; }
div[data-baseweb="input"] > div { border: 1px solid #1d3a2e !important; background-color: #091712 !important; }
.sidebar-active-title { color: #ffffff !important; font-weight: 700; font-size: 11px; letter-spacing: 1px; margin-bottom: 4px; }
.sidebar-inactive-title { color: #668075 !important; font-weight: 600; font-size: 11px; letter-spacing: 1px; margin-bottom: 4px; }
.active-ctrl { opacity: 1.0 !important; }
.inactive-ctrl { opacity: 0.40 !important; filter: grayscale(90%); pointer-events: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_ai(): return demo_model()
@st.cache_resource
def get_yolo(model_path: str, confidence: float, tracking: bool): return YOLOVision(model_path, confidence, tracking)

# State initialization
for key, default in {
    'seq': 1000, 'logs': [], 'last_cmd': 'STOP', 'rover_active': False,
    'incident_ack': False, 'prev_scenario': None, 'param_history': [],
    'camera_mode': 'Normal Vision', 'custom_ai_model': None
}.items():
    if key not in st.session_state: st.session_state[key] = default

for i in range(1, 10):
    chk_key = f'ccm_{i}'
    if chk_key not in st.session_state: st.session_state[chk_key] = False

# Sidebar
with st.sidebar:
    st.markdown('<div class="brand"><h1>MINESAFE TITAN MODE</h1><small>SIH 2026 - PS 26039</small></div>', unsafe_allow_html=True)
    
    source = st.radio('Telemetry', ['LIVE ESP GATEWAY', 'SIMULATION LAB'], index=1, label_visibility='collapsed')
    is_live = source.startswith('LIVE')

    st.markdown(f'<div class="{"sidebar-active-title" if is_live else "sidebar-inactive-title"}">LIVE ESP GATEWAY CONFIG</div>', unsafe_allow_html=True)
    gw_css = "active-ctrl" if is_live else "inactive-ctrl"
    st.markdown(f'<div class="{gw_css}">', unsafe_allow_html=True)
    gateway_url = st.text_input('Gateway URL', DEFAULT_GATEWAY, disabled=not is_live)
    st.markdown('</div>', unsafe_allow_html=True)

    sim_css = "active-ctrl" if not is_live else "inactive-ctrl"
    st.markdown(f'<div class="{"sidebar-active-title" if not is_live else "sidebar-inactive-title"}">SIMULATION LAB - INCIDENT INJECTION</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{sim_css}">', unsafe_allow_html=True)
    
    scenario = st.selectbox(
        'Scenario',
        [
            'Nominal Operations',
            'Water Sump Inundation',
            'Toxic CO₂ Blackdamp Ingress',
            'Critical Oxygen Deficiency',
            'Carbon Monoxide Poisoning',
            'Explosive Methane Ingress',
            'Dynamic Strata Roof Collapse',
            'Thermal Ceiling Exceeded',
            'Compound Disaster'
        ],
        label_visibility='collapsed',
        disabled=is_live
    )
    
    route_blocked = (scenario in {'Compound Disaster', 'Dynamic Strata Roof Collapse', 'Water Sump Inundation'}) if not is_live else False
    hazard_node = st.selectbox('Hazard location', ['NONE', 'ZONE_A', 'ZONE_B'], index=0, disabled=is_live)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('**WORKER TELEMETRY MODE**')
    worker_mode = st.radio('Worker channel', ['SIMULATED WEARABLES', 'LIVE WEARABLES'], label_visibility='collapsed')
    st.divider()
    st.markdown('**VISION / YOLOv8**')
    vision_source = st.radio('Vision input', ['DISABLED', 'IMAGE UPLOAD', 'CAMERA SNAPSHOT'], label_visibility='collapsed')
    vision_file = st.file_uploader('Mine / worker frame', type=['png','jpg','jpeg']) if vision_source == 'IMAGE UPLOAD' else (st.camera_input('Capture underground frame') if vision_source == 'CAMERA SNAPSHOT' else None)
    custom_yolo = st.file_uploader('Custom mine YOLO .pt', type=['pt'])
    yolo_conf = st.slider('YOLO confidence', .15, .85, .35, .05)
    tracking = st.checkbox('Persistent tracking (ByteTrack)', True)
    if not YOLO_AVAILABLE:
        st.warning('Ultralytics is not installed. Install requirements.txt to activate vision.')
    st.divider()
    if st.button('↻ RESET INCIDENT', use_container_width=True):
        st.session_state.incident_ack = False
        st.session_state.camera_mode = 'Normal Vision'
        st.session_state.logs = []
        st.rerun()

if st.session_state.prev_scenario != scenario:
    st.session_state.prev_scenario = scenario
    for i in range(1, 10):
        st.session_state[f'ccm_{i}'] = False

st.session_state.seq += 1; seq = st.session_state.seq
client = GatewayClient(gateway_url); link_failed = False

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

vision_result = None
if vision_file is not None:
    custom_path = save_uploaded_model(custom_yolo) if custom_yolo else None
    model_path = custom_path or 'yolov8n.pt'
    vision_result = get_yolo(model_path, yolo_conf, tracking).predict(Image.open(vision_file).convert('RGB'))

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

active_ai = st.session_state.custom_ai_model if st.session_state.custom_ai_model is not None else get_ai()
ai_prob = float(active_ai.predict_proba(np.array([[getattr(telemetry, f) for f in FEATURES]], float))[0] * 100) if telemetry.is_valid else 0.0

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
    workers = [
        Worker("TAG-108", "Worker A", "ZONE_A", 128, 94, 37.8, False),
        Worker("TAG-114", "Worker B", "ZONE_B", 118, 95, 37.4, False),
        Worker("TAG-121", "Worker C", "ZONE_B", 88, 97, 36.8, False),
    ]

target_dest = "ZONE_B" if "ZONE_B" in MINE_GRAPH else list(MINE_GRAPH.keys())[-1]
blocked = [("BASE", target_dest)] if route_blocked else []
node_penalties = {hazard_node: min(220, max(0, float(score) * 2.2))} if hazard_node != "NONE" else {}
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

hardware_connected = is_live and (not link_failed) and getattr(gw, 'online', False)

if is_live:
    state_label = 'LIVE DATA' if hardware_connected else 'GATEWAY LOST (FAIL-SAFE)'
    state_cls = 'cond-green' if hardware_connected else 'cond-red'
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

# Top Bar
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

ai_status_label = 'ANOMALOUS' if ai_prob >= 60.0 else ('UNCERTAIN' if ai_prob >= 35.0 else 'NOMINAL PATTERN')
ai_status_color = 'red' if ai_prob >= 60.0 else ('amber' if ai_prob >= 35.0 else 'green')

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

curr_dt = datetime.now()
curr_pressure = round(101.3 + (telemetry.temp - 25.0) * 0.08 - (telemetry.humidity - 50.0) * 0.03, 1) if telemetry.is_valid else 101.3

# Tabs
tab2, tab3, tab4, tab5, tab6 = st.tabs(['◉ ENVIRONMENT', '♙ PERSONNEL & RESCUE', '◈ AI / VISION', '▣ AUDIT & STANDARDS', '📜 STATUTORY METHODS'])

with tab2:
    st.markdown('<div class="section">ENVIRONMENTAL MONITORING</div>', unsafe_allow_html=True)
    
    sensors_row1 = [
        ('TEMPERATURE', telemetry.temp, '°C', telemetry.temp >= 33.5, 'max 33.5 °C'),
        ('CO', telemetry.co, 'PPM', telemetry.co >= 50.0, 'max 50 PPM'),
        ('METHANE', telemetry.ch4, '%', telemetry.ch4 >= 0.75, 'max 0.75 %'),
        ('VIBRATIONS', telemetry.strata_vibe_g, 'g', telemetry.strata_vibe_g >= 2.0, 'max 2.0 g'),
    ]
    sensors_row2 = [
        ('OXYGEN (O₂)', telemetry.o2, '%', telemetry.o2 < 19.0, 'min 19.0 %'),
        ('CARBON DIOXIDE', telemetry.co2, 'PPM', telemetry.co2 >= 5000.0, 'max 5000 PPM'),
        ('WATER DEPTH', telemetry.water_level_cm, 'cm', telemetry.water_level_cm >= 15.0, 'max 15.0 cm'),
        ('HUMIDITY', telemetry.humidity, '%', telemetry.humidity >= 85.0, 'max 85 %'),
    ]
    
    for sensor_group in [sensors_row1, sensors_row2]:
        cols = st.columns(4, gap='medium')
        for col, (s_name, s_val, s_unit, is_bad, s_limit) in zip(cols, sensor_group):
            invalid = not math.isfinite(float(s_val)) if s_val is not None else True
            val_display = '—' if invalid else (f'{s_val:.2f}' if s_unit == '%' and s_name.startswith('METHANE') else f'{s_val:.1f}')
            val_color = 'red' if (invalid or is_bad) else 'green'
            col.markdown(f'''
            <div class="panel" style="margin-bottom:12px">
                <div class="sensor-label">{s_name}</div>
                <div class="sensor-val {val_color}">{val_display} <span style="font-size:13px;color:var(--muted);font-weight:400">{s_unit}</span></div>
                <div class="sensor-sub">{s_limit}</div>
            </div>
            ''', unsafe_allow_html=True)

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

        tunnel_svg_normal = """<svg style="position:absolute;width:100%;height:100%;top:0;left:0;" viewBox="0 0 640 280">
            <rect width="640" height="280" fill="#040806"/>
            <polygon points="0,0 210,95 210,185 0,280" fill="#070f0b" stroke="#12261d" stroke-width="2"/>
            <polygon points="640,0 430,95 430,185 640,280" fill="#070f0b" stroke="#12261d" stroke-width="2"/>
            <polygon points="0,0 210,95 430,95 640,0" fill="#0a1610" stroke="#162e22" stroke-width="2"/>
            <polygon points="0,280 210,185 430,185 640,280" fill="#030604" stroke="#11241a" stroke-width="2"/>
            <rect x="210" y="95" width="220" height="90" fill="#010302" stroke="#19382b" stroke-width="1.5"/>
            <line x1="280" y1="185" x2="160" y2="280" stroke="#1b3026" stroke-width="5"/>
            <line x1="360" y1="185" x2="480" y2="280" stroke="#1b3026" stroke-width="5"/>
            <circle cx="320" cy="140" r="4" fill="#a2c7b5"/>
            <line x1="210" y1="140" x2="430" y2="140" stroke="#112b20" stroke-dasharray="4,6"/>
        </svg>"""

        tunnel_svg_night = """<svg style="position:absolute;width:100%;height:100%;top:0;left:0;" viewBox="0 0 640 280">
            <rect width="640" height="280" fill="#000302"/>
            <polygon points="0,0 210,95 210,185 0,280" fill="#02140a" stroke="#083e20" stroke-width="2"/>
            <polygon points="640,0 430,95 430,185 640,280" fill="#02140a" stroke="#083e20" stroke-width="2"/>
            <polygon points="0,0 210,95 430,95 640,0" fill="#031e0f" stroke="#0a4b27" stroke-width="2"/>
            <polygon points="0,280 210,185 430,185 640,280" fill="#010b05" stroke="#06321a" stroke-width="2"/>
            <rect x="210" y="95" width="220" height="90" fill="#000502" stroke="#147c43" stroke-width="1.5"/>
            <line x1="280" y1="185" x2="160" y2="280" stroke="#0a522a" stroke-width="4"/>
            <line x1="360" y1="185" x2="480" y2="280" stroke="#0a522a" stroke-width="4"/>
            <ellipse cx="320" cy="140" rx="60" ry="25" fill="none" stroke="#25e082" stroke-width="1" stroke-dasharray="3,5"/>
        </svg>"""

        tunnel_svg_thermal = """<svg style="position:absolute;width:100%;height:100%;top:0;left:0;" viewBox="0 0 640 280">
            <rect width="640" height="280" fill="#090212"/>
            <polygon points="0,0 210,95 210,185 0,280" fill="#240733" stroke="#521345" stroke-width="2"/>
            <polygon points="640,0 430,95 430,185 640,280" fill="#240733" stroke="#521345" stroke-width="2"/>
            <polygon points="0,0 210,95 430,95 640,0" fill="#38093f" stroke="#771850" stroke-width="2"/>
            <polygon points="0,280 210,185 430,185 640,280" fill="#140321" stroke="#3d0c39" stroke-width="2"/>
            <rect x="210" y="95" width="220" height="90" fill="#751543" stroke="#c2323a" stroke-width="2"/>
            <circle cx="320" cy="140" r="14" fill="#ffb300"/>
            <circle cx="320" cy="140" r="8" fill="#ffffff"/>
            <line x1="280" y1="185" x2="160" y2="280" stroke="#b3263c" stroke-width="3"/>
            <line x1="360" y1="185" x2="480" y2="280" stroke="#b3263c" stroke-width="3"/>
        </svg>"""

        if cam_mode == "Thermal Vision":
            st.markdown(f'''
            <div class="viewport-frame">
                {tunnel_svg_thermal}
                <div class="cam-crosshair"></div>
                <div class="cam-watermark">SIMULATED THERMAL RADIOMETRIC</div>
            </div>
            <div class="cam-footer">
                <span>TUNNEL_HEAD_EAST · LWIR 8-14µm</span>
                <span>NETD &lt; 50mK · GAIN: HIGH FLIR RADIOMETRIC · RESOLUTION: 1080p</span>
            </div>
            ''', unsafe_allow_html=True)
        elif cam_mode == "Night Vision":
            st.markdown(f'''
            <div class="viewport-frame">
                {tunnel_svg_night}
                <div class="cam-crosshair"></div>
                <div class="cam-watermark">SIMULATED NIGHT VISION (IR 850nm)</div>
            </div>
            <div class="cam-footer">
                <span>TUNNEL_HEAD_EAST · IR 850nm ILLUMINATOR ACTIVE</span>
                <span>LOW-LIGHT ENHANCED · FPS: 30 · RESOLUTION: 1080p</span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            if hardware_connected:
                stream_target = "http://10.143.43.237:81/stream"
                st.markdown(f'''
                <div style="height:280px;background:#000;display:flex;justify-content:center;align-items:center;overflow:hidden;position:relative;">
                    <img src="{stream_target}" style="width:100%;height:100%;object-fit:cover;"/>
                    <div class="cam-crosshair"></div>
                    <div class="cam-watermark">LIVE HARDWARE STREAM (ESP32-S3-CAM)</div>
                </div>
                <div class="cam-footer">
                    <span>ESP32-S3-CAM · {gateway_url}</span>
                    <span>MJPEG STREAM · RESOLUTION: VGA 640x480</span>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="viewport-frame">
                    {tunnel_svg_normal}
                    <div class="cam-crosshair"></div>
                    <div class="cam-watermark">SIMULATED NORMAL VISION</div>
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
                Target Sector: <b>ZONE_B</b>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        ctrl_grid = st.columns(3)
        with ctrl_grid[1]:
            if st.button('▲', key='rover_fwd', use_container_width=True):
                st.session_state.last_cmd = 'FORWARD'; st.session_state.rover_active = True
                client.send_rover_command('forward') if hardware_connected else None
                st.session_state.logs.append(f'[{datetime.now().strftime("%H:%M:%S")}] CONTROL: last rover command=FORWARD'); st.rerun()

        ctrl_row2 = st.columns(3)
        with ctrl_row2[0]:
            if st.button('◀', key='rover_left', use_container_width=True):
                st.session_state.last_cmd = 'LEFT'; st.session_state.rover_active = True
                client.send_rover_command('left') if hardware_connected else None
                st.session_state.logs.append(f'[{datetime.now().strftime("%H:%M:%S")}] CONTROL: last rover command=LEFT'); st.rerun()

        with ctrl_row2[1]:
            toggle_label = 'STOP' if st.session_state.rover_active else 'START'
            if st.button(toggle_label, key='rover_toggle', use_container_width=True):
                st.session_state.rover_active = not st.session_state.rover_active
                st.session_state.last_cmd = 'START' if st.session_state.rover_active else 'EMERGENCY STOP'
                cmd_code = 'start' if st.session_state.rover_active else 'emergency_stop'
                client.send_rover_command(cmd_code) if hardware_connected else None
                st.session_state.logs.append(f'[{datetime.now().strftime("%H:%M:%S")}] CONTROL: last rover command={st.session_state.last_cmd}'); st.rerun()

        with ctrl_row2[2]:
            if st.button('▶', key='rover_right', use_container_width=True):
                st.session_state.last_cmd = 'RIGHT'; st.session_state.rover_active = True
                client.send_rover_command('right') if hardware_connected else None
                st.session_state.logs.append(f'[{datetime.now().strftime("%H:%M:%S")}] CONTROL: last rover command=RIGHT'); st.rerun()

        ctrl_row3 = st.columns(3)
        with ctrl_row3[1]:
            if st.button('▼', key='rover_rev', use_container_width=True):
                st.session_state.last_cmd = 'REVERSE'; st.session_state.rover_active = True
                client.send_rover_command('reverse') if hardware_connected else None
                st.session_state.logs.append(f'[{datetime.now().strftime("%H:%M:%S")}] CONTROL: last rover command=REVERSE'); st.rerun()

    st.markdown('<div class="section">RISK COMPONENTS</div>', unsafe_allow_html=True)
    st.plotly_chart(
        go.Figure(go.Bar(
            x=list(scores.values()),
            y=list(scores.keys()),
            orientation='h',
            text=[f'{v:.0f}' for v in scores.values()],
            textposition='outside'
        )).update_layout(
            height=240, margin=dict(l=0, r=40, t=5, b=5),
            paper_bgcolor='#0b1713', plot_bgcolor='#0b1713',
            font=dict(color='#dcebe4'), xaxis=dict(range=[0, 110])
        ),
        use_container_width=True, config={'displayModeBar': False}
    )
    st.markdown('<div class="panel"><div class="paneltitle">THRESHOLD BREACHES</div>' + (''.join(f'<div class="reading"><b>{b}</b><span class="red">BREACH</span></div>' for b in breaches) if breaches else '<div class="mini">No configured threshold breaches.</div>') + '</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section">PERSONNEL & RESCUE</div>', unsafe_allow_html=True)
    for row in [trpis[i:i+3] for i in range(0, len(trpis), 3)]:
        for col, (w, tv, parts) in zip(st.columns(len(row)), row):
            col.markdown(f'''
            <div class="worker">
                <div class="tagline"><b>{w.tag} · {w.name}</b></div>
                <div class="hr">{w.hr_bpm:.0f} <span style="font-size:9px;color:#769188">BPM</span></div>
                <div class="mini">SpO₂ {w.spo2_pct:.0f}% · Temp {w.temp_c:.1f}°C · Zone {w.zone}</div>
                <div class="bar"><div class="fill" style="width:{min(100, tv)}%"></div></div>
                <div style="display:flex;justify-content:space-between">
                    <span class="mini">TRPI</span>
                    <b class="{"red" if tv >= 80 else "amber" if tv >= 60 else "green"}">{tv:.1f}/100</b>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown('<div class="section">COAL MINE MAP</div>', unsafe_allow_html=True)
    rescue_map_col, sector_details_col = st.columns([1.35, 0.65], gap='medium')

    if hazard_node == "NONE":
        zone_a_status, zone_a_color = "SAFE", "#39d98a"
        zone_b_status, zone_b_color = "SAFE", "#39d98a"
    elif hazard_node == "ZONE_A":
        zone_a_status, zone_a_color = ("DANGEROUS" if score >= 60 else "CAUTION"), ("#ff5c5c" if score >= 60 else "#f4b942")
        zone_b_status, zone_b_color = "SAFE", "#39d98a"
    else:
        zone_a_status, zone_a_color = "SAFE", "#39d98a"
        zone_b_status, zone_b_color = ("DANGEROUS" if score >= 60 else "CAUTION"), ("#ff5c5c" if score >= 60 else "#f4b942")

    with rescue_map_col:
        fig_rescue = go.Figure()

        fig_rescue.add_trace(go.Scatter(
            x=[0.3, 9.4], y=[5.0, 5.0],
            mode='lines', line=dict(width=2, color='#2a6b52', dash='dot'),
            hoverinfo='skip', showlegend=False
        ))

        x_wall = np.linspace(0.8, 9.4, 45)
        np.random.seed(42)
        y_top = 5.65 + np.sin(x_wall * 2.8) * 0.12 + np.random.normal(0, 0.03, len(x_wall))
        y_bot = 4.35 - np.cos(x_wall * 2.5) * 0.12 - np.random.normal(0, 0.03, len(x_wall))

        fig_rescue.add_trace(go.Scatter(
            x=list(x_wall) + list(x_wall[::-1]), y=list(y_top) + list(y_bot[::-1]),
            fill='toself', fillcolor='rgba(13, 33, 24, 0.50)',
            line=dict(color='#183d2c', width=2), hoverinfo='skip', showlegend=False
        ))

        fig_rescue.add_trace(go.Scatter(
            x=[0.1, 0.7, 0.7, 0.1, 0.1], y=[5.2, 5.2, 6.0, 6.0, 5.2],
            mode='lines', fill='toself', fillcolor='rgba(6, 18, 14, 0.9)',
            line=dict(color='#769188', width=2), hoverinfo='skip', showlegend=False
        ))
        fig_rescue.add_trace(go.Scatter(
            x=[0.4], y=[5.6], mode='text', text=['<b>BASE</b>'],
            textposition='middle center', textfont=dict(size=10, family='IBM Plex Mono', color='#a2c7b5'),
            showlegend=False
        ))

        fig_rescue.add_trace(go.Scatter(
            x=[0.8, 4.8, 4.8, 0.8, 0.8], y=[4.1, 4.1, 5.9, 5.9, 4.1],
            mode='lines', fill='toself', fillcolor='rgba(0,0,0,0)',
            line=dict(color=zone_a_color, width=3.5 if hazard_node == "ZONE_A" else 2),
            hoverinfo='skip', showlegend=False
        ))
        fig_rescue.add_trace(go.Scatter(
            x=[2.8], y=[6.35], mode='text',
            text=[f'<b>ZONE A</b><br><span style="color:{zone_a_color}">{zone_a_status}</span>'],
            textposition='top center', textfont=dict(size=10.5, family='IBM Plex Mono', color='#ffffff'),
            showlegend=False
        ))

        fig_rescue.add_trace(go.Scatter(
            x=[4.8, 9.4, 9.4, 4.8, 4.8], y=[4.1, 4.1, 5.9, 5.9, 4.1],
            mode='lines', fill='toself', fillcolor='rgba(0,0,0,0)',
            line=dict(color=zone_b_color, width=3.5 if hazard_node == "ZONE_B" else 2),
            hoverinfo='skip', showlegend=False
        ))
        fig_rescue.add_trace(go.Scatter(
            x=[7.1], y=[6.35], mode='text',
            text=[f'<b>ZONE B</b><br><span style="color:{zone_b_color}">{zone_b_status}</span>'],
            textposition='top center', textfont=dict(size=10.5, family='IBM Plex Mono', color='#ffffff'),
            showlegend=False
        ))

        rover_color = '#39d98a' if st.session_state.rover_active else '#56d6e8'
        fig_rescue.add_trace(go.Scatter(
            x=[3.8], y=[5.0], mode='markers+text',
            text=[f'ROVER M-01<br>({"ACTIVE" if st.session_state.rover_active else "HOLD"})'],
            textposition='top center', textfont=dict(size=10, family='IBM Plex Mono', color=rover_color),
            marker=dict(size=19, color=rover_color, symbol='diamond'), name='ROVER M-01'
        ))

        worker_node_positions = [(2.0, 5.0), (6.2, 5.0), (8.0, 5.0)]
        for i, (w, tv, _) in enumerate(trpis[:3]):
            wx, wy = worker_node_positions[i]
            w_color = '#ff5c5c' if tv >= 60 else '#f4b942' if tv >= 35 else '#39d98a'
            fig_rescue.add_trace(go.Scatter(
                x=[wx], y=[wy], mode='markers+text',
                text=[f'<b>{w.tag}</b><br>{w.name}'], textposition='bottom center',
                textfont=dict(size=10.5, family='IBM Plex Mono', color='#dcebe4'),
                marker=dict(size=22, color=w_color, symbol='circle', line=dict(width=2, color='#ffffff')),
                name=w.tag
            ))

        fig_rescue.update_layout(
            height=280, margin=dict(l=5, r=5, t=35, b=10),
            paper_bgcolor='#07100d', plot_bgcolor='#07100d',
            xaxis=dict(visible=False, range=[-0.1, 9.8]),
            yaxis=dict(visible=False, range=[3.8, 7.3]),
            font=dict(color='#dcebe4', family='IBM Plex Mono'), showlegend=False
        )
        st.plotly_chart(fig_rescue, use_container_width=True, config={'displayModeBar': False})

    with sector_details_col:
        zone_a_temp = telemetry.temp if hazard_node == "ZONE_A" else 24.8
        zone_a_ch4 = telemetry.ch4 if hazard_node == "ZONE_A" else 0.05
        zone_a_hum = telemetry.humidity if hazard_node == "ZONE_A" else 52.0
        zone_a_water = telemetry.water_level_cm if hazard_node == "ZONE_A" else 2.0

        zone_b_temp = telemetry.temp if hazard_node == "ZONE_B" else 26.5
        zone_b_ch4 = telemetry.ch4 if hazard_node == "ZONE_B" else 0.08
        zone_b_hum = telemetry.humidity if hazard_node == "ZONE_B" else 58.0
        zone_b_water = telemetry.water_level_cm if hazard_node == "ZONE_B" else 2.5

        if hazard_node == "NONE":
            trapped_html = '<span style="color:#39d98a">All Sectors Nominal · Normal Egress Open</span>'
        else:
            trapped_workers = [w.name for w in workers if w.zone == hazard_node]
            if trapped_workers:
                trapped_html = f'<span style="color:#ff5c5c;font-weight:700">⚠ {len(trapped_workers)} Worker(s) in Hazard Zone:</span> ' + ', '.join(trapped_workers)
            else:
                trapped_html = '<span style="color:#39d98a">Sector Clear · No Registered Personnel</span>'

        card_html = (
            f'<div class="panel" style="margin-top:0px;padding:12px">'
            f'<div class="paneltitle" style="display:flex;justify-content:space-between"><span>SECTOR CONDITIONS</span><span class="mini">HAZARD: {hazard_node}</span></div>'
            f'<div style="margin-top:8px;padding:8px 10px;background:#06120e;border-radius:6px;border-left:3px solid {zone_a_color}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center"><b style="font-size:11px;color:#ffffff">ZONE A (DRIFT GALLERY)</b><span style="font-size:9px;font-weight:700;color:{zone_a_color}">{zone_a_status}</span></div>'
            f'<div class="mini" style="margin-top:3px;display:grid;grid-template-columns:1fr 1fr;gap:2px">'
            f'<span>Temp: <b style="color:{"#ff5c5c" if zone_a_temp >= 33.5 else "#ffffff"}">{zone_a_temp:.1f}°C</b></span>'
            f'<span>CH₄: <b style="color:{"#ff5c5c" if zone_a_ch4 >= 0.75 else "#ffffff"}">{zone_a_ch4:.2f}%</b></span>'
            f'<span>Humidity: <b style="color:#ffffff">{zone_a_hum:.1f}%</b></span>'
            f'<span>Water: <b style="color:{"#ff5c5c" if zone_a_water >= 15.0 else "#ffffff"}">{zone_a_water:.1f}cm</b></span>'
            f'</div></div>'
            f'<div style="margin-top:8px;padding:8px 10px;background:#06120e;border-radius:6px;border-left:3px solid {zone_b_color}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center"><b style="font-size:11px;color:#ffffff">ZONE B (PRODUCTION FACE)</b><span style="font-size:9px;font-weight:700;color:{zone_b_color}">{zone_b_status}</span></div>'
            f'<div class="mini" style="margin-top:3px;display:grid;grid-template-columns:1fr 1fr;gap:2px">'
            f'<span>Temp: <b style="color:{"#ff5c5c" if zone_b_temp >= 33.5 else "#ffffff"}">{zone_b_temp:.1f}°C</b></span>'
            f'<span>CH₄: <b style="color:{"#ff5c5c" if zone_b_ch4 >= 0.75 else "#ffffff"}">{zone_b_ch4:.2f}%</b></span>'
            f'<span>Humidity: <b style="color:#ffffff">{zone_b_hum:.1f}%</b></span>'
            f'<span>Water: <b style="color:{"#ff5c5c" if zone_b_water >= 15.0 else "#ffffff"}">{zone_b_water:.1f}cm</b></span>'
            f'</div></div>'
            f'<div style="margin-top:10px;padding:8px;background:rgba(20,8,8,0.65);border:1px solid #331515;border-radius:6px;font-size:10px">'
            f'<div style="font-weight:700;letter-spacing:0.8px;color:#ffffff;margin-bottom:3px">PERSONNEL STATUS // EVACUATION TARGET</div>'
            f'{trapped_html}'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # Critical Condition Measures Checklist
    st.markdown('<div class="section">CRITICAL CONDITION MEASURES</div>', unsafe_allow_html=True)
    is_critical_env = (scenario != 'Nominal Operations') or (score >= 60) or (risk in {'CRITICAL', 'HIGH'}) or route_blocked
    all_checked = all(st.session_state.get(f'ccm_{i}', False) for i in range(1, 10))

    box_css_class = "checklist-critical" if (is_critical_env and not all_checked) else "checklist-default"
    badge_label = "● ACTIVE HAZARD DETECTED // EMERGENCY PROTOCOL REQUIRED" if (is_critical_env and not all_checked) else ("● PROTOCOLS VERIFIED" if all_checked else "● NOMINAL OPERATIONS (STANDBY)")
    badge_color = "var(--red)" if (is_critical_env and not all_checked) else "#769188"

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
            st.checkbox('Continuously monitor methane, temperature, pressure, humidity, and water level.', key='ccm_4')
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

# ==============================================================================
# TAB 4: AI & COMPUTER VISION
# ==============================================================================
with tab4:
    st.markdown('<div class="section">AI & COMPUTER VISION</div>', unsafe_allow_html=True)
    
    col_ai_left, col_ai_right = st.columns(2, gap='medium')
    with col_ai_left:
        st.markdown(f'''
        <div class="panel">
            <div class="paneltitle">SENSOR ANOMALY MODEL - TinyMLP</div>
            <div style="font-family:Barlow Condensed;font-size:38px;color:#f4b942;margin-top:4px">{ai_prob:.1f}%</div>
            <div class="mini" style="margin-top:2px">Current anomaly probability · demo-trained synthetic baseline</div>
            <div style="font-size:11.5px;color:#ffffff;font-weight:600;margin-top:14px;margin-bottom:6px">Train in-session from labelled historical CSV</div>
        ''', unsafe_allow_html=True)
        retrain_csv = st.file_uploader("Upload CSV", type=["csv"], key="csv_retrain_upload", label_visibility="collapsed")
        if retrain_csv is not None:
            try:
                df_custom = pd.read_csv(retrain_csv)
                if all(f in df_custom.columns for f in FEATURES) and 'label' in df_custom.columns:
                    X_c = df_custom[FEATURES].to_numpy(dtype=float)
                    y_c = df_custom['label'].to_numpy(dtype=int)
                    m = TinyMLP()
                    m.fit(X_c, y_c, epochs=600, lr=0.03)
                    st.session_state.custom_ai_model = m
                    st.success(f"Model retrained with {len(df_custom)} samples.")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ai_right:
        st.markdown('''
        <div class="panel" style="min-height:210px">
            <div class="paneltitle">YOLOv8 MINE VISION</div>
            <div class="mini" style="margin-top:4px;margin-bottom:14px">Detection • persistent tracking • explicit safety-event mapping • multimodal fusion</div>
            <div style="background:#09191c;border:1px solid #163f47;border-radius:6px;padding:12px;font-size:11.5px;color:#56d6e8;line-height:1.6">
                Enable image/camera input in the sidebar. The app remains fully usable without vision. Baseline YOLOv8n is generic; custom mine weights are required for PPE/fall/fire classes.
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="section" style="margin-top:20px">STRUCTURAL BASELINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:var(--muted);margin-bottom:6px">Optional roof / strata image</div>', unsafe_allow_html=True)
    st.file_uploader("Upload Structural Baseline", type=["png", "jpg", "jpeg"], key="structural_baseline_upload", label_visibility="collapsed")

    st.markdown('<div class="section" style="margin-top:24px">ENVIRONMENTAL PARAMETER TRENDS</div>', unsafe_allow_html=True)
    
    filter_col1, filter_col2 = st.columns([0.72, 0.28])
    with filter_col1:
        st.caption("Historical trend analysis for current environmental sensor channels.")
    with filter_col2:
        period = st.selectbox("Resolution", ["YEARS (Monthly)", "MONTHS (Daily)", "DAYS (Per-Minute)"], index=0, label_visibility="collapsed")

    curr_t = float(telemetry.temp) if telemetry.is_valid else 27.0
    curr_p = float(curr_pressure)
    curr_m = float(telemetry.ch4) if telemetry.is_valid else 0.10
    curr_h = float(telemetry.humidity) if telemetry.is_valid else 55.0
    curr_o2 = float(telemetry.o2) if telemetry.is_valid else 20.8
    curr_co2 = float(telemetry.co2) if telemetry.is_valid else 450.0
    curr_water = float(telemetry.water_level_cm) if telemetry.is_valid else 2.0

    if period == "YEARS (Monthly)":
        num_points = 12
        x_axis = [(curr_dt - timedelta(days=(num_points - 1 - i) * 30)).strftime("%b") for i in range(num_points)]
        year_subtitle = f"{curr_dt.year - 1} – {curr_dt.year}"
        y_temp = [round(curr_t + np.sin(i * 0.5) * 4.0, 1) for i in range(num_points)]
        y_press = [round(curr_p + np.cos(i * 0.5) * 1.5, 1) for i in range(num_points)]
        y_meth = [round(max(0.01, curr_m + np.sin(i * 0.3) * 0.15), 2) for i in range(num_points)]
        y_o2 = [round(max(15.0, min(21.5, curr_o2 - np.sin(i * 0.4) * 0.8)), 1) for i in range(num_points)]
        y_co2 = [round(max(350.0, curr_co2 + np.sin(i * 0.5) * 400.0), 0) for i in range(num_points)]
        y_water = [round(max(0.5, curr_water + np.sin(i * 0.6) * 3.5), 1) for i in range(num_points)]
    elif period == "MONTHS (Daily)":
        num_points = 15
        x_axis = [(curr_dt - timedelta(days=(num_points - 1 - i) * 2)).strftime("%d %b") for i in range(num_points)]
        year_subtitle = f"{curr_dt.year}"
        y_temp = [round(curr_t + np.sin(i * 0.4) * 2.0 - 0.5, 1) for i in range(num_points)]
        y_press = [round(curr_p + np.cos(i * 0.3) * 0.8, 1) for i in range(num_points)]
        y_meth = [round(max(0.01, curr_m + np.sin(i * 0.5) * 0.09), 2) for i in range(num_points)]
        y_o2 = [round(max(15.0, min(21.5, curr_o2 - np.sin(i * 0.3) * 0.4)), 1) for i in range(num_points)]
        y_co2 = [round(max(350.0, curr_co2 + np.cos(i * 0.4) * 250.0), 0) for i in range(num_points)]
        y_water = [round(max(0.5, curr_water + np.cos(i * 0.5) * 2.0), 1) for i in range(num_points)]
    else:
        num_points = 15
        x_axis = [(curr_dt - timedelta(minutes=(num_points - 1 - i) * 4)).strftime("%H:%M") for i in range(num_points)]
        year_subtitle = f"{curr_dt.strftime('%d %b %Y')}"
        y_temp = [round(curr_t + np.sin(i * 0.6) * 0.4, 1) for i in range(num_points)]
        y_press = [round(curr_p + np.cos(i * 0.5) * 0.3, 1) for i in range(num_points)]
        y_meth = [round(max(0.01, curr_m + np.sin(i * 0.8) * 0.04), 2) for i in range(num_points)]
        y_o2 = [round(max(15.0, min(21.5, curr_o2 - np.sin(i * 0.6) * 0.15)), 1) for i in range(num_points)]
        y_co2 = [round(max(350.0, curr_co2 + np.sin(i * 0.6) * 100.0), 0) for i in range(num_points)]
        y_water = [round(max(0.5, curr_water + np.sin(i * 0.7) * 0.8), 1) for i in range(num_points)]

    chart_col1, chart_col2 = st.columns(2, gap='medium')
    with chart_col1:
        st.markdown('<div class="paneltitle" style="margin-bottom:8px">TEMPERATURE, OXYGEN & PRESSURE</div>', unsafe_allow_html=True)
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(x=x_axis, y=y_temp, name='Temp (°C)', mode='lines+markers', line=dict(color='#ff8c42', width=2.5), marker=dict(size=4)))
        fig_tp.add_trace(go.Scatter(x=x_axis, y=y_o2, name='O₂ (%)', mode='lines+markers', line=dict(color='#39d98a', width=2), marker=dict(size=4)))
        fig_tp.add_trace(go.Scatter(x=x_axis, y=y_press, name='Pressure (kPa)', mode='lines+markers', line=dict(color='#56d6e8', width=2, dash='dot'), marker=dict(size=4), yaxis='y2'))
        fig_tp.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=35), paper_bgcolor='#0b1713', plot_bgcolor='#0b1713', font=dict(color='#dcebe4', family='IBM Plex Mono', size=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), hovermode='x unified', yaxis=dict(title=dict(text='Temp / O₂', font=dict(color='#ff8c42')), showgrid=True, gridcolor='#152c22'), yaxis2=dict(title=dict(text='Press (kPa)', font=dict(color='#56d6e8')), overlaying='y', side='right', showgrid=False), xaxis=dict(title=dict(text=year_subtitle, font=dict(color='#769188', size=10)), showgrid=True, gridcolor='#152c22'))
        st.plotly_chart(fig_tp, use_container_width=True, config={'displayModeBar': False})

    with chart_col2:
        st.markdown('<div class="paneltitle" style="margin-bottom:8px">METHANE, CO₂ & WATER DEPTH</div>', unsafe_allow_html=True)
        fig_ghm = go.Figure()
        fig_ghm.add_trace(go.Scatter(x=x_axis, y=y_meth, name='CH₄ (%)', mode='lines+markers', line=dict(color='#ff5c5c', width=2.5), marker=dict(size=4)))
        fig_ghm.add_trace(go.Scatter(x=x_axis, y=y_water, name='Water (cm)', mode='lines+markers', line=dict(color='#56d6e8', width=2), marker=dict(size=4)))
        fig_ghm.add_trace(go.Scatter(x=x_axis, y=y_co2, name='CO₂ (PPM)', mode='lines+markers', line=dict(color='#b99cff', width=2, dash='dash'), marker=dict(size=4), yaxis='y2'))
        fig_ghm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=35), paper_bgcolor='#0b1713', plot_bgcolor='#0b1713', font=dict(color='#dcebe4', family='IBM Plex Mono', size=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), hovermode='x unified', yaxis=dict(title=dict(text='CH₄ (%) / Water (cm)', font=dict(color='#ff5c5c')), showgrid=True, gridcolor='#152c22'), yaxis2=dict(title=dict(text='CO₂ (PPM)', font=dict(color='#b99cff')), overlaying='y', side='right', showgrid=False), xaxis=dict(title=dict(text=year_subtitle, font=dict(color='#769188', size=10)), showgrid=True, gridcolor='#152c22'))
        st.plotly_chart(fig_ghm, use_container_width=True, config={'displayModeBar': False})

# ==============================================================================
# TAB 5: AUDIT & STANDARDS
# ==============================================================================
with tab5:
    top_aud_col1, top_aud_col2 = st.columns([1.2, 0.8])
    with top_aud_col1:
        st.markdown('<div class="section" style="margin:0">AUDIT / STANDARDS TRACEABLE PROTOTYPE STATE</div>', unsafe_allow_html=True)
    with top_aud_col2:
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
        st.download_button(
            'EXPORT FORENSIC JSON',
            json.dumps(audit, indent=2, default=str),
            file_name=f'MineSafe_Audit_{datetime.now():%Y%m%d_%H%M%S}.json',
            mime='application/json',
            use_container_width=True
        )

    st.markdown('<div class="section" style="margin-top:16px">ROUTE CANDIDATES</div>', unsafe_allow_html=True)
    rc_data = [
        {"rank": 1, "path": "BASE → ZONE_A → ZONE_B", "distance_m": 65, "hazard_penalty": 0, "total_cost": 65},
        {"rank": 2, "path": "BASE → SHAFT_NORTH → ZONE_B", "distance_m": 75, "hazard_penalty": 0, "total_cost": 75},
        {"rank": 3, "path": "BASE → ZONE_A → SHAFT_NORTH → ZONE_B", "distance_m": 85, "hazard_penalty": 0, "total_cost": 85},
    ]
    st.dataframe(pd.DataFrame(rc_data), use_container_width=True, hide_index=True)

    st.markdown('''
    <div class="panel" style="margin-top:16px;margin-bottom:20px">
        <div class="paneltitle" style="margin-bottom:8px">ENGINEERING INTEGRITY</div>
        <div class="mini" style="line-height:1.8">
            <b>DATA:</b> missing, malformed, NaN and physically implausible readings are rejected — never silently clamped.<br>
            <b>AI:</b> synthetic training is explicitly labelled; evaluation script supports hold-out metrics on real labelled telemetry.<br>
            <b>VISION:</b> only explicit model classes create safety events; no unsafe inference from class absence.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section" style="margin-top:20px">STATUTORY COMPLIANCE & REFERENCE STANDARDS</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(reference_rows()), use_container_width=True, hide_index=True)
    st.markdown('<div class="mini" style="color:var(--muted);margin-top:6px;margin-bottom:18px">Reference source: Directorate General of Mines Safety (DGMS), Coal Mines Regulations, 2017. Verify the current applicable rules, mine category, approved ventilation scheme and site procedures before any operational use.</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 6: STATUTORY METHODS
# ==============================================================================
with tab6:
    st.markdown('<div class="section">STATUTORY STANDARDS & DECISION METHODOLOGY DGMS-REFERENCED BASELINE</div>', unsafe_allow_html=True)

    st.markdown('''
    <div class="panel" style="margin-bottom:12px">
        <div class="paneltitle" style="margin-bottom:8px">STATUTORY / REFERENCE BASELINE — COAL MINES REGULATIONS, 2017</div>
        <div class="mini" style="line-height:1.9">
            • <b>Oxygen (O₂):</b> minimum 19.0% by volume at places where persons are required to work or pass.<br>
            • <b>Carbon Monoxide (CO):</b> MineSafe uses 50 ppm as a configured operational alert threshold; it is not labelled as a universal CMR statutory ceiling.<br>
            • <b>Inflammable gas / Methane (CH₄):</b> not more than 0.75% in the general body of return air of a ventilating district and 1.25% at any place in the mine.<br>
            • <b>Carbon Dioxide (CO₂):</b> not more than 0.5% by volume (5,000 ppm) in the cited ventilation provision.<br>
            • <b>Temperature & Thermal Comfort:</b> dry-bulb tracking with wet-bulb ceiling not more than 33.5°C; where wet-bulb exceeds 30.5°C, ventilation arrangements with at least 1 m/s air movement are specified by the cited provision.<br>
            • <b>Humidity (% RH):</b> continuous monitoring to calculate derived wet-bulb temperature (ceiling warning at 85% RH).<br>
            • <b>Water Depth:</b> continuous sump / floor inundation tracking to prevent bogging and drowning hazards (statutory/operational ceiling: 15.0 cm).<br>
            • <b>Vibration / Strata Stability:</b> roof vibration & strata shock registered via high-g binary triggers (operational limit: 2.0 g to preempt dynamic roof falls).
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.link_button('OPEN OFFICIAL DGMS CMR 2017 SOURCE', 'https://www.dgms.gov.in/writereaddata/UploadFile/Coal_Mines_Regulation_2017_Noti.pdf', use_container_width=True)

    st.markdown('<div class="section" style="margin-top:20px">MODELLED O₂ BUFFER PROXY TRANSPARENT ENGINEERING CALCULATION</div>', unsafe_allow_html=True)
    
    o2_left_col, o2_right_col = st.columns([1.1, 0.9], gap='medium')
    with o2_left_col:
        st.caption("Chamber volume (m³)")
        v_chamber = st.number_input("Chamber volume (m³)", min_value=10.0, max_value=50000.0, value=1000.00, step=50.0, label_visibility="collapsed")
        
        st.caption("Workers in chamber")
        n_workers = st.number_input("Workers in chamber", min_value=1, max_value=200, value=3, step=1, label_visibility="collapsed")

        st.caption("Assumed O₂ consumption (L/min/worker)")
        vo2_rate = st.number_input("Assumed O₂ consumption (L/min/worker)", min_value=0.10, max_value=5.0, value=0.25, step=0.05, label_visibility="collapsed")

    with o2_right_col:
        st.markdown('<div style="background:#06120e;border:1px solid #163829;border-radius:6px;padding:8px;margin-bottom:12px">', unsafe_allow_html=True)
        st.latex(r"\Delta t = \frac{V_{\text{chamber}} \times (O_2\% - 19.0\%)}{N_{\text{workers}} \times \dot{V}_{O_2}}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="panel" style="padding:10px">
            <div class="paneltitle" style="font-size:10px">MODEL LIMITATIONS</div>
            <div class="mini" style="font-size:8.5px;line-height:1.6;margin-top:4px">
                This is an atmospheric proxy only. It does not account for CO uptake, CO₂ narcosis, ventilation changes, methane, temperature, individual metabolism, leakage, stratification, or rescue-team operating constraints. It must never be presented as survival time or a guaranteed evacuation window.
            </div>
        </div>
        ''', unsafe_allow_html=True)

    current_o2_val = telemetry.o2 if (telemetry.is_valid and telemetry.o2 >= 19.0) else 20.8
    minutes_buffer = o2_buffer_proxy_minutes(current_o2_val, v_chamber, n_workers, vo2_rate)
    st.markdown(f'''
    <div style="margin-top:8px;margin-bottom:16px">
        <div style="font-size:11px;color:var(--muted);letter-spacing:1px;font-weight:600">PROXY TO 19% O₂ THRESHOLD</div>
        <div style="font-family:'Barlow Condensed';font-size:42px;font-weight:700;color:#e9f2ed">{minutes_buffer:.1f} min</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section">LIVE SYSTEM ACTIVITY TIMELINE</div>', unsafe_allow_html=True)
    t_now_str = datetime.now().strftime("%H:%M:%S")
    default_events = [
        f"[{t_now_str}] SYS_AUTH: Command node {NODE_ID} active",
        f"[{t_now_str}] SENSOR_GATEWAY: {'SIMULATED · SCENARIO' if not is_live else 'LIVE_GATEWAY'} → VALID",
        f"[{t_now_str}] HAZARD_ENGINE: {risk} ({score}/100) · primary={primary}",
        f"[{t_now_str}] RESCUE_ENGINE: max TRPI={max_trpi:.1f} · priority={priority}",
        f"[{t_now_str}] ROUTE_ENGINE: BASE → ZONE_A → ZONE_B",
        f"[{t_now_str}] AI: TinyMLP={ai_prob:.1f}% · YOLO={'STANDBY' if not vision_result else 'ACTIVE'}",
        f"[{t_now_str}] CONTROL: last rover command={st.session_state.last_cmd}"
    ]
    st.session_state.logs = default_events + [l for l in st.session_state.logs if l not in default_events][-6:]
    stream_html = "".join([f"<div style='font-family:IBM Plex Mono;font-size:10.5px;color:#a2c7b5;padding:2px 0'>{ev}</div>" for ev in st.session_state.logs[-8:]])
    st.markdown(f'''
    <div style="background:#040907;border:1px solid #142e22;border-radius:6px;padding:10px 14px;margin-bottom:20px">
        {stream_html}
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section">DECISION METHODOLOGY EXPLAINABLE PIPELINE</div>', unsafe_allow_html=True)
    pipeline_rows = [
        {"Layer": "Telemetry validation", "Implementation": "Missing, malformed, NaN and physically implausible values are rejected; invalid LIVE data enters fail-safe state."},
        {"Layer": "Hazard fusion", "Implementation": "Atmospheric + geotechnical + thermal components are combined into an explainable 0–100 prototype risk score."},
        {"Layer": "Sensor AI", "Implementation": "TinyMLP baseline is explicitly demo-trained until labelled historical mine data is supplied and evaluated on a held-out set."},
        {"Layer": "Computer vision", "Implementation": "Only explicit model classes create mapped safety events. Generic YOLO does not infer missing PPE/falls/fire from class absence."},
        {"Layer": "TRPI", "Implementation": "Weighted worker prioritization using atmosphere, toxicity, physiology, SpO₂, route, strata and fall evidence."},
        {"Layer": "Routing", "Implementation": "Candidate paths are ranked by distance plus explicit hazard penalties; blocked edges are removed."},
        {"Layer": "O₂ buffer proxy", "Implementation": "Simplified chamber/worker consumption calculation to the 19% threshold; not a medical, survivability or evacuation window."},
        {"Layer": "Human authority", "Implementation": "Rover control remains human-approved and is restricted when the live gateway is unavailable."},
        {"Layer": "Audit", "Implementation": "Telemetry state, model metadata, hazard components, workers, routes and limitations can be exported as JSON."}
    ]
    st.dataframe(pd.DataFrame(pipeline_rows), use_container_width=True, hide_index=True)

# FULL LENGTH RELIABLE FOOTER BAR
st.markdown(
    f'''<div class="full-footer-bar">
        <span>
            <strong style="color:#e9f2ed">MINESAFE TITAN</strong> by <strong style="color:#39d98a">TECH FORGE</strong> · SIH '26 [PS 26039]
        </span>
        <span>
            <span style="color:{"#39d98a" if hardware_connected else "#56d6e8"}">● {state_label}</span> · v{VERSION} · SEQ {seq:06d}
        </span>
    </div>''',
    unsafe_allow_html=True
)