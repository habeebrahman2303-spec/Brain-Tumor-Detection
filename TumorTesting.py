import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import time
import json
import base64
from io import BytesIO
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
#  GLOBAL CSS + ANIMATIONS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Exo+2:ital,wght@0,100..900;1,100..900&family=Share+Tech+Mono&display=swap');

/* ── VARIABLES ── */
:root {
  --bg0:     #020509;
  --bg1:     #060c14;
  --bg2:     #0a1220;
  --panel:   rgba(6,14,26,0.85);
  --border:  rgba(0,220,255,0.12);
  --cyan:    #00dcff;
  --cyan2:   #00a8cc;
  --green:   #00ff9d;
  --red:     #ff3c5a;
  --amber:   #ffb300;
  --violet:  #9d6eff;
  --text:    #c8d8f0;
  --muted:   #3a5070;
  --font-hd: 'Rajdhani', sans-serif;
  --font-bd: 'Exo 2', sans-serif;
  --font-mn: 'Share Tech Mono', monospace;
}

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }

html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: var(--bg0) !important;
    color: var(--text) !important;
    font-family: var(--font-bd) !important;
}

/* animated hex-grid bg */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
        radial-gradient(ellipse 1200px 700px at 20% -5%,  rgba(0,220,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 800px  600px at 85% 90%,  rgba(157,110,255,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 600px  500px at 50% 50%,  rgba(0,255,157,0.025) 0%, transparent 50%),
        repeating-linear-gradient(0deg,   transparent, transparent 39px, rgba(0,220,255,0.02) 40px),
        repeating-linear-gradient(90deg,  transparent, transparent 39px, rgba(0,220,255,0.02) 40px);
    pointer-events: none;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

[data-testid="stMainBlockContainer"],
.block-container {
    max-width: 1380px !important;
    padding: 0 1.5rem 5rem !important;
    position: relative; z-index: 1;
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb { background: var(--cyan2); border-radius:2px; }

/* ═══ KEYFRAMES ═══ */
@keyframes fadeSlideUp   { from{opacity:0;transform:translateY(22px)} to{opacity:1;transform:none} }
@keyframes fadeSlideDown { from{opacity:0;transform:translateY(-14px)} to{opacity:1;transform:none} }
@keyframes fadeIn        { from{opacity:0} to{opacity:1} }
@keyframes scanline {
    0%   { top: -8%; }
    100% { top: 108%; }
}
@keyframes radarSpin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes pulse {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 currentColor; }
    50%      { opacity:.5; box-shadow: 0 0 0 5px transparent; }
}
@keyframes glitch1 {
    0%,100%{clip-path:inset(50% 0 30% 0)} 20%{clip-path:inset(10% 0 60% 0)}
    40%{clip-path:inset(80% 0 5%  0)} 60%{clip-path:inset(30% 0 50% 0)}
    80%{clip-path:inset(5%  0 80% 0)}
}
@keyframes glitch2 {
    0%,100%{clip-path:inset(20% 0 60% 0)} 20%{clip-path:inset(70% 0 10% 0)}
    40%{clip-path:inset(5%  0 75% 0)} 60%{clip-path:inset(55% 0 25% 0)}
    80%{clip-path:inset(35% 0 45% 0)}
}
@keyframes borderFlow {
    0%   { border-color: rgba(0,220,255,0.2); box-shadow: 0 0 0 rgba(0,220,255,0); }
    50%  { border-color: rgba(0,220,255,0.7); box-shadow: 0 0 20px rgba(0,220,255,0.15); }
    100% { border-color: rgba(0,220,255,0.2); box-shadow: 0 0 0 rgba(0,220,255,0); }
}
@keyframes progressFill { from{width:0%} to{width:var(--w)} }
@keyframes countUp      { from{opacity:0;transform:scale(0.6)} to{opacity:1;transform:scale(1)} }
@keyframes shimmer {
    0%  { background-position: -200% center; }
    100%{ background-position:  200% center; }
}
@keyframes orbitRing {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* ═══ TOPBAR ═══ */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 0;
    margin-bottom: 0;
    animation: fadeSlideDown 0.5s ease both;
}
.topbar-brand {
    display: flex; align-items: center; gap: 0.9rem;
}
.topbar-logo {
    width: 38px; height: 38px;
    border: 1.5px solid var(--cyan);
    border-radius: 10px;
    display: flex; align-items:center; justify-content:center;
    font-size: 1.2rem;
    background: rgba(0,220,255,0.06);
    box-shadow: 0 0 12px rgba(0,220,255,0.15);
    animation: borderFlow 3s ease infinite;
}
.topbar-title {
    font-family: var(--font-hd);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--cyan);
    text-shadow: 0 0 20px rgba(0,220,255,0.4);
}
.topbar-title span { color: #fff; }
.topbar-status {
    display: flex; align-items:center; gap: 0.5rem;
    font-family: var(--font-mn);
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: var(--green);
}
.status-dot {
    width:7px; height:7px; border-radius:50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite;
}
.topbar-meta {
    font-family: var(--font-mn);
    font-size: 0.58rem;
    color: var(--muted);
    text-align: right;
    line-height: 1.7;
}

/* ═══ HERO STRIP ═══ */
.hero-strip {
    position: relative;
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    overflow: hidden;
    animation: fadeSlideUp 0.7s 0.1s ease both;
}
.hero-strip::before {
    content:'';
    position:absolute; top:0; left:50%; transform:translateX(-50%);
    width:600px; height:1px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
}
.hero-eyebrow {
    font-family: var(--font-mn);
    font-size: 0.62rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 1.2rem;
    display:flex; align-items:center; justify-content:center; gap:0.8rem;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content:''; display:block;
    width: 40px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan));
}
.hero-eyebrow::after { transform: scaleX(-1); }

.hero-main {
    position: relative;
    font-family: var(--font-hd);
    font-size: clamp(3.5rem, 9vw, 6.5rem);
    font-weight: 700;
    letter-spacing: 0.06em;
    line-height: 1;
    color: #fff;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.hero-main .glitch-wrap {
    position: relative; display: inline-block;
}
.hero-main .glitch-wrap::before,
.hero-main .glitch-wrap::after {
    content: attr(data-text);
    position: absolute; inset: 0;
    color: #fff;
}
.hero-main .glitch-wrap::before {
    color: var(--cyan);
    animation: glitch1 8s 2s infinite;
    opacity: 0.6;
}
.hero-main .glitch-wrap::after {
    color: var(--violet);
    animation: glitch2 8s 2s infinite;
    opacity: 0.4;
}
.hero-accent {
    background: linear-gradient(90deg, var(--cyan), var(--violet), var(--green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% auto;
    animation: shimmer 4s linear infinite;
}
.hero-sub {
    font-family: var(--font-mn);
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.hero-desc {
    font-family: var(--font-bd);
    font-size: 0.88rem;
    font-weight: 300;
    line-height: 1.9;
    color: #5a7090;
    max-width: 560px;
    margin: 0 auto;
}
.hero-desc strong { color: #9ab8d8; font-weight: 600; }

/* ═══ GLOW DIVIDER ═══ */
.glow-div {
    position: relative;
    width:100%; height:1px;
    background: linear-gradient(90deg,transparent,var(--cyan),var(--violet),transparent);
    margin: 1.5rem 0;
}
.glow-div::after {
    content:'';
    position:absolute; left:50%; top:-4px; transform:translateX(-50%);
    width:200px; height:9px;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
    filter: blur(8px); border-radius:4px; opacity:.5;
}

/* ═══ STATS ROW ═══ */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
    animation: fadeSlideUp 0.7s 0.25s ease both;
}
.stat-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    backdrop-filter: blur(12px);
    position: relative; overflow: hidden;
    transition: transform 0.25s, border-color 0.25s;
    cursor: default;
}
.stat-box:hover {
    transform: translateY(-3px);
    border-color: rgba(0,220,255,0.3);
}
.stat-box::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
}
.sb-cyan::before    { background: var(--cyan); }
.sb-green::before   { background: var(--green); }
.sb-amber::before   { background: var(--amber); }
.sb-violet::before  { background: var(--violet); }
.stat-num {
    font-family: var(--font-hd);
    font-size: 2rem; font-weight:700; line-height:1;
    margin-bottom: 0.3rem;
}
.sb-cyan   .stat-num { color: var(--cyan); }
.sb-green  .stat-num { color: var(--green); }
.sb-amber  .stat-num { color: var(--amber); }
.sb-violet .stat-num { color: var(--violet); }
.stat-lbl {
    font-family: var(--font-mn);
    font-size: 0.55rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ═══ TUMOR TYPE CARDS ═══ */
.tumor-cards {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 0.8rem;
    margin-bottom: 2rem;
    animation: fadeSlideUp 0.7s 0.35s ease both;
}
.tc {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s;
    position: relative; overflow: hidden; cursor:default;
}
.tc::after {
    content:'';
    position:absolute; bottom:0; left:0; right:0; height:0;
    transition: height 0.3s;
}
.tc:hover { transform: translateY(-4px); box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
.tc:hover::after { height: 3px; }
.tc-g::after  { background: var(--red); }
.tc-m::after  { background: var(--amber); }
.tc-p::after  { background: var(--violet); }
.tc-h::after  { background: var(--green); }
.tc:hover.tc-g { border-color: rgba(255,60,90,0.3); }
.tc:hover.tc-m { border-color: rgba(255,179,0,0.3); }
.tc:hover.tc-p { border-color: rgba(157,110,255,0.3); }
.tc:hover.tc-h { border-color: rgba(0,255,157,0.3); }
.tc-icon { font-size:1.8rem; margin-bottom:0.5rem; }
.tc-sev {
    font-family: var(--font-mn);
    font-size: 0.5rem; letter-spacing:0.2em; text-transform:uppercase;
    margin-bottom:0.3rem;
}
.tc-g .tc-sev { color: var(--red); }
.tc-m .tc-sev { color: var(--amber); }
.tc-p .tc-sev { color: var(--violet); }
.tc-h .tc-sev { color: var(--green); }
.tc-name { font-family:var(--font-hd); font-size:1rem; font-weight:700; color:#dde8f8; margin-bottom:0.4rem; }
.tc-desc { font-size:0.65rem; color: var(--muted); line-height:1.55; }

/* ═══ SECTION TITLE ═══ */
.sec-title {
    display: flex; align-items: center; gap: 0.8rem;
    font-family: var(--font-hd);
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.3em; text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 1rem;
}
.sec-title::before {
    content:''; display:block;
    width:3px; height:16px; border-radius:2px;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
}

/* ═══ UPLOAD PANEL ═══ */
.upload-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    animation: fadeSlideUp 0.6s 0.4s ease both;
    height: 100%;
}

[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(0,220,255,0.2) !important;
    border-radius: 14px !important;
    background: rgba(0,220,255,0.015) !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,220,255,0.5) !important;
    background: rgba(0,220,255,0.03) !important;
    box-shadow: 0 0 20px rgba(0,220,255,0.08) !important;
}

/* ═══ MRI VIEWER ═══ */
.mri-viewer {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(0,220,255,0.15);
    background: #000;
}
.mri-viewer-inner { position: relative; }
.mri-scanline {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    box-shadow: 0 0 10px var(--cyan), 0 0 30px rgba(0,220,255,0.4);
    animation: scanline 2.5s linear infinite;
    pointer-events: none;
    z-index: 10;
}
.mri-corner {
    position:absolute; width:16px; height:16px;
    border-color: var(--cyan); border-style: solid;
    opacity: 0.6; z-index:11;
}
.mri-corner.tl { top:8px;    left:8px;   border-width:2px 0 0 2px; }
.mri-corner.tr { top:8px;    right:8px;  border-width:2px 2px 0 0; }
.mri-corner.bl { bottom:8px; left:8px;   border-width:0 0 2px 2px; }
.mri-corner.br { bottom:8px; right:8px;  border-width:0 2px 2px 0; }
.mri-label {
    position:absolute; bottom:10px; left:12px; z-index:12;
    font-family: var(--font-mn); font-size:0.52rem;
    letter-spacing:0.2em; color: rgba(0,220,255,0.5);
    text-transform: uppercase;
}

/* ═══ BUTTON ═══ */
.stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    border-radius: 10px !important;
    font-family: var(--font-mn) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.85em 1.5em !important;
    transition: all 0.25s !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,220,255,0.1), rgba(157,110,255,0.1));
    opacity: 0;
    transition: opacity 0.25s;
}
.stButton > button:hover {
    box-shadow: 0 0 24px rgba(0,220,255,0.25), inset 0 0 20px rgba(0,220,255,0.05) !important;
    transform: translateY(-1px) !important;
    background: rgba(0,220,255,0.05) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ═══ PROGRESS BAR (Streamlit override) ═══ */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--violet)) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 4px !important;
}

/* ═══ RESULT PANEL ═══ */
.result-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0;
    overflow: hidden;
    backdrop-filter: blur(12px);
    animation: fadeSlideUp 0.5s ease both;
}
.result-header {
    padding: 1.2rem 1.5rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items:center; justify-content:space-between;
}
.rh-label {
    font-family: var(--font-mn);
    font-size: 0.58rem; letter-spacing:0.28em; text-transform:uppercase; color: var(--muted);
    display:flex; align-items:center; gap:0.5rem;
}
.rh-live { width:7px;height:7px;border-radius:50%;animation:pulse 2s infinite; }
.rh-ts { font-family:var(--font-mn); font-size:0.55rem; color:var(--muted); }

.result-body { padding: 1.5rem; }

.diag-big {
    font-family: var(--font-hd);
    font-size: 3.5rem; font-weight:700; letter-spacing:0.05em; line-height:1;
    margin-bottom: 0.4rem;
    animation: countUp 0.6s 0.1s ease both;
}
.db-tumor   { color: var(--red);   text-shadow: 0 0 30px rgba(255,60,90,0.35); }
.db-healthy { color: var(--green); text-shadow: 0 0 30px rgba(0,255,157,0.35); }

.diag-short { font-size:0.8rem; color:var(--muted); margin-bottom:1.5rem; }

/* confidence arc SVG */
.conf-arc-wrap {
    display:flex; align-items:center; gap:1.5rem;
    margin-bottom: 1.5rem;
}
.arc-svg { flex-shrink:0; }

.conf-bar-section { flex:1; }
.cbs-top {
    display:flex; justify-content:space-between; align-items:baseline;
    margin-bottom:0.4rem;
}
.cbs-label { font-family:var(--font-mn); font-size:0.55rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); }
.cbs-val   { font-family:var(--font-hd); font-size:2.2rem; font-weight:700; line-height:1; }
.cbs-val.cv-tumor   { color: var(--red); }
.cbs-val.cv-healthy { color: var(--green); }

.conf-track { height:5px; background:rgba(255,255,255,0.05); border-radius:99px; overflow:hidden; }
.conf-fill  {
    height:100%; border-radius:99px;
    position:relative;
    transition: width 1.2s cubic-bezier(0.16,1,0.3,1);
}
.cf-tumor   { background: linear-gradient(90deg, #c0001a, var(--red)); }
.cf-healthy { background: linear-gradient(90deg, #007a46, var(--green)); }

/* all class probabilities */
.prob-title { font-family:var(--font-mn); font-size:0.55rem; letter-spacing:0.22em; text-transform:uppercase; color:var(--muted); margin-bottom:0.8rem; }
.prob-rows { display:flex; flex-direction:column; gap:0.5rem; margin-bottom:1.5rem; }
.prob-row { display:flex; align-items:center; gap:0.8rem; }
.pr-name { font-size:0.75rem; font-weight:500; color:#6a80a0; width:90px; flex-shrink:0; }
.pr-name.active { color: var(--cyan); font-weight:600; }
.pr-track { flex:1; height:4px; background:rgba(255,255,255,0.05); border-radius:99px; overflow:hidden; }
.pr-fill  { height:100%; border-radius:99px; }
.pf-default { background: rgba(100,130,180,0.3); }
.pf-active-tumor   { background: linear-gradient(90deg, #c0001a, var(--red)); }
.pf-active-healthy { background: linear-gradient(90deg, #007a46, var(--green)); }
.pr-pct { font-family:var(--font-mn); font-size:0.7rem; color:#5a7090; width:40px; text-align:right; flex-shrink:0; }
.pr-pct.active { color: var(--cyan); }

/* detail blocks */
.detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; margin-bottom:1rem; }
.detail-box {
    background: rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.05);
    border-radius:12px; padding:1rem;
}
.db-t { font-family:var(--font-mn); font-size:0.52rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); margin-bottom:0.6rem; display:flex; align-items:center; gap:0.4rem; }
.db-t::before { content:''; display:block; width:10px; height:1px; background:var(--muted); }
.db-body { font-size:0.78rem; line-height:1.75; color:#5a7090; }
.db-body strong { color:#9ab8d8; }

/* step list */
.step-list { list-style:none; display:flex; flex-direction:column; gap:0.55rem; }
.step-list li {
    display:flex; align-items:flex-start; gap:0.7rem;
    font-size:0.77rem; color:#5a7090; line-height:1.55;
}
.step-n {
    flex-shrink:0; width:20px; height:20px; border-radius:50%;
    border:1px solid rgba(0,220,255,0.25);
    background:rgba(0,220,255,0.05);
    display:flex; align-items:center; justify-content:center;
    font-family:var(--font-mn); font-size:0.5rem; color:var(--cyan);
    margin-top:1px;
}

/* disclaimer */
.disclaimer {
    background:rgba(255,60,90,0.05);
    border:1px solid rgba(255,60,90,0.15);
    border-radius:10px; padding:0.9rem 1.1rem;
    font-size:0.72rem; color:#d06070; line-height:1.7;
    margin-top:1rem;
}
.disclaimer strong { color:#ff8095; }

/* ═══ HISTORY PANEL ═══ */
.hist-panel {
    background: var(--panel);
    border:1px solid var(--border);
    border-radius:18px; padding:1.5rem;
    backdrop-filter:blur(12px);
    margin-top: 1.5rem;
}
.hist-empty {
    text-align:center; padding:2.5rem 0;
    font-family:var(--font-mn); font-size:0.65rem;
    letter-spacing:0.2em; color:var(--muted);
}
.hist-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:0.75rem 0.9rem;
    background:rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.04);
    border-radius:10px; margin-bottom:0.5rem;
    transition: border-color 0.2s;
    animation: fadeSlideUp 0.4s ease both;
}
.hist-row:hover { border-color: rgba(0,220,255,0.15); }
.hr-name { font-size:0.8rem; font-weight:600; color:var(--text); }
.hr-name.hn-tumor   { color: var(--red); }
.hr-name.hn-healthy { color: var(--green); }
.hr-conf { font-family:var(--font-mn); font-size:0.65rem; color: var(--muted); }
.hr-time { font-family:var(--font-mn); font-size:0.55rem; color: var(--muted); letter-spacing:0.1em; }
.hr-badge {
    font-family:var(--font-mn); font-size:0.5rem; letter-spacing:0.15em;
    padding:0.25em 0.7em; border-radius:99px; text-transform:uppercase;
}
.hb-tumor   { background:rgba(255,60,90,0.1);  border:1px solid rgba(255,60,90,0.25);  color:var(--red); }
.hb-healthy { background:rgba(0,255,157,0.1); border:1px solid rgba(0,255,157,0.25); color:var(--green); }

/* ═══ INFO TABS ═══ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--panel) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 0.3rem !important;
    gap: 0.2rem !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--font-mn) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-radius: 8px !important;
    padding: 0.55em 1.2em !important;
    border: none !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--cyan) !important;
    background: rgba(0,220,255,0.06) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(0,220,255,0.1) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(0,220,255,0.1) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none !important; }
[data-testid="stTabPanel"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin-top: 0.5rem !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px) !important;
}

/* ═══ ABOUT SECTION ═══ */
.about-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.about-card {
    background:rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.05);
    border-radius:12px; padding:1.2rem;
}
.ac-head {
    font-family:var(--font-hd); font-size:0.85rem; font-weight:700;
    color:var(--cyan); margin-bottom:0.6rem; letter-spacing:0.05em;
}
.ac-body { font-size:0.78rem; line-height:1.8; color:#5a7090; }
.ac-body strong { color:#9ab8d8; }

/* ═══ RADAR WIDGET ═══ */
.radar-widget {
    display:flex; align-items:center; justify-content:center;
    padding:1.5rem 0;
}
.radar-svg { position:relative; }

/* ═══ FOOTER ═══ */
.ns-footer {
    text-align:center;
    font-family:var(--font-mn); font-size:0.52rem;
    letter-spacing:0.22em; text-transform:uppercase;
    color:#1a2535;
    margin-top:3rem; padding-top:1.5rem;
    border-top:1px solid rgba(255,255,255,0.03);
}
.ns-footer span { color:#2a3f5f; }

/* ═══ STREAMLIT MISC ═══ */
[data-testid="stSpinner"] p {
    color: var(--cyan) !important;
    font-family: var(--font-mn) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
}
div[data-testid="stImage"] img { border-radius: 12px; width:100%; display:block; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
CLASS_NAMES = ['Glioma', 'Healthy', 'Meningioma', 'Pituitary']

TUMOR_DB = {
    "Glioma": {
        "kind":    "tumor",
        "sev":     "HIGH RISK",
        "icon":    "⚡",
        "short":   "Malignant glial cell tumor — most aggressive classification.",
        "summary": (
            "<strong>Gliomas</strong> originate in the brain's glial support cells and account for "
            "approximately <strong>33% of all brain tumors</strong>. They range from slow-growing "
            "Grade I pilocytic astrocytomas to the devastating Grade IV Glioblastoma Multiforme (GBM) — "
            "the most lethal form of primary brain cancer. Presenting symptoms typically include "
            "persistent headaches, focal neurological deficits, seizures, and progressive cognitive decline. "
            "Treatment requires a multimodal approach: maximal safe surgical resection, "
            "concurrent radiotherapy, and temozolomide chemotherapy."
        ),
        "symptoms": "Headaches · Seizures · Cognitive decline · Motor deficits · Vision changes",
        "steps": [
            "Urgent referral to neurosurgical oncology",
            "Contrast-enhanced MRI for detailed 3D mapping",
            "Stereotactic biopsy for histopathological grading",
            "Multidisciplinary tumour board review",
        ],
    },
    "Meningioma": {
        "kind":    "tumor",
        "sev":     "MODERATE",
        "icon":    "◈",
        "short":   "Meningeal tumor — usually benign, often slow-growing.",
        "summary": (
            "<strong>Meningiomas</strong> arise from the arachnoid cap cells of the meninges — "
            "the layered membranes protecting the brain and spinal cord. They represent "
            "<strong>~37% of primary brain tumors</strong> and are classified predominantly as "
            "benign (WHO Grade I). Significantly more common in females and in adults over 40, "
            "they often grow silently for years. Small asymptomatic lesions may be observed; "
            "symptomatic or enlarging tumors are treated with microsurgery or stereotactic radiosurgery."
        ),
        "symptoms": "Headaches · Vision disturbance · Seizures · Limb weakness · Hearing loss",
        "steps": [
            "Neurology consultation for symptom baseline",
            "Gadolinium-contrast MRI with thin-slice protocol",
            "Watchful waiting if small and asymptomatic",
            "Neurosurgical evaluation for resection or SRS",
        ],
    },
    "Pituitary": {
        "kind":    "tumor",
        "sev":     "MODERATE",
        "icon":    "◉",
        "short":   "Pituitary adenoma — disrupts the master endocrine gland.",
        "summary": (
            "<strong>Pituitary adenomas</strong> arise in the pituitary gland at the brain's base "
            "and may be functioning (secreting excess hormones) or non-functioning. "
            "Functioning adenomas drive systemic endocrine disorders: prolactinomas cause "
            "galactorrhea and infertility; GH-secreting tumors cause acromegaly; "
            "ACTH-secreting tumors cause Cushing's disease. Mass effect on the optic chiasm "
            "produces characteristic bitemporal hemianopsia. Treatment depends on type: "
            "dopamine agonists for prolactinomas; transsphenoidal surgery or radiotherapy otherwise."
        ),
        "symptoms": "Hormonal imbalance · Visual field loss · Headaches · Infertility · Fatigue",
        "steps": [
            "Comprehensive pituitary hormone panel",
            "Formal ophthalmology visual-field assessment",
            "Dedicated 3T pituitary MRI protocol",
            "Combined endocrinology + neurosurgery management",
        ],
    },
    "Healthy": {
        "kind":    "healthy",
        "sev":     "NORMAL",
        "icon":    "✦",
        "short":   "No tumor detected — brain parenchyma appears structurally normal.",
        "summary": (
            "The deep learning model found <strong>no detectable tumor signatures</strong> "
            "across all four classification categories. The MRI scan presents features consistent "
            "with healthy brain parenchyma — no abnormal masses, space-occupying lesions, or "
            "characteristic tumor patterns were identified. This is a positive screening outcome. "
            "Importantly, AI-assisted screening is a <strong>supplementary tool only</strong> "
            "and does not replace formal radiological evaluation by a qualified neuroradiologist."
        ),
        "symptoms": "No neurological symptoms detected in this classification",
        "steps": [
            "Continue routine annual neurological monitoring",
            "Follow up with your physician as scheduled",
            "Report any new or progressive neurological symptoms promptly",
            "Maintain MRI surveillance schedule if previously established",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════
if 'history'  not in st.session_state: st.session_state.history  = []
if 'results'  not in st.session_state: st.session_state.results  = None
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'scan_count' not in st.session_state: st.session_state.scan_count = 0


# ═══════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    import tensorflow as tf

    model_path = "brain_tumor_model.keras"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
        return model

    except Exception as e:
        import traceback
        st.error(traceback.format_exc())
        return None


model = load_model()
# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════
def img_to_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def make_enhanced(pil_img: Image.Image) -> Image.Image:
    """High-contrast grayscale with green tint for clinical look."""
    gray = pil_img.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
    rgb = enhanced.convert("RGB")
    r, g, b = rgb.split()
    g = ImageEnhance.Brightness(g.convert("L")).enhance(1.15).convert("L")
    return Image.merge("RGB", (r, g, b))

def draw_heatmap_overlay(pil_img: Image.Image) -> Image.Image:
    """Simple red-circle overlay to simulate attention map."""
    img_copy = pil_img.copy().convert("RGBA")
    overlay  = Image.new("RGBA", img_copy.size, (0,0,0,0))
    draw     = ImageDraw.Draw(overlay)
    cx, cy   = img_copy.width // 2, img_copy.height // 2
    r        = min(cx, cy) // 2
    for i in range(5, 0, -1):
        alpha = int(20 * i)
        draw.ellipse(
            [cx - r*i//3, cy - r*i//3, cx + r*i//3, cy + r*i//3],
            fill=(255, 50, 50, alpha)
        )
    return Image.alpha_composite(img_copy, overlay).convert("RGB")


# ═══════════════════════════════════════════════════════════════
#  TOP BAR
# ═══════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%Y.%m.%d  %H:%M")
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">🧠</div>
        <div class="topbar-title"><span>NEURO</span>SCAN AI</div>
    </div>
    <div class="topbar-status">
        <div class="status-dot"></div>
        SYSTEM ONLINE
    </div>
    <div class="topbar-meta">
        VER 2.0.0 &nbsp;|&nbsp; {now_str}<br>
        SCANS THIS SESSION: {st.session_state.scan_count}
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-strip">
    <div class="hero-eyebrow">MRI Deep Learning Analysis System</div>
    <div class="hero-main">
        <span class="glitch-wrap" data-text="NEUROSCAN">NEURO</span><span class="hero-accent">SCAN</span>
    </div>
    <div class="hero-sub">Brain Tumor Classification · CNN Model · 4-Class Output</div>
    <div class="hero-desc">
        Brain tumours are abnormal growths of cells within the skull —
        some aggressively invade tissue, others press silently for years.
        Every year <strong>over 300,000 people</strong> worldwide receive a primary brain tumour diagnosis.
        Upload an MRI scan below and receive an instant AI-powered classification:
        <strong>Glioma, Meningioma, Pituitary</strong>, or a confirmed <strong>Healthy</strong> brain.
    </div>
</div>
<div class="glow-div"></div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  STATS ROW
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="stats-row">
    <div class="stat-box sb-cyan">
        <div class="stat-num">128²</div>
        <div class="stat-lbl">Input Resolution</div>
    </div>
    <div class="stat-box sb-green">
        <div class="stat-num">4</div>
        <div class="stat-lbl">Output Classes</div>
    </div>
    <div class="stat-box sb-amber">
        <div class="stat-num">CNN</div>
        <div class="stat-lbl">Architecture</div>
    </div>
    <div class="stat-box sb-violet">
        <div class="stat-num">TF</div>
        <div class="stat-lbl">Framework</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TUMOR TYPE CARDS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="tumor-cards">
    <div class="tc tc-g">
        <div class="tc-icon">⚡</div>
        <div class="tc-sev">Most Aggressive</div>
        <div class="tc-name">Glioma</div>
        <div class="tc-desc">Originates in glial cells. Can be high-grade, fast-growing & life-threatening.</div>
    </div>
    <div class="tc tc-m">
        <div class="tc-icon">◈</div>
        <div class="tc-sev">Usually Benign</div>
        <div class="tc-name">Meningioma</div>
        <div class="tc-desc">Grows from the meninges. Slow-moving with generally favourable surgical outcomes.</div>
    </div>
    <div class="tc tc-p">
        <div class="tc-icon">◉</div>
        <div class="tc-sev">Hormonal Impact</div>
        <div class="tc-name">Pituitary</div>
        <div class="tc-desc">Disrupts the master endocrine gland — triggers system-wide hormonal disorders.</div>
    </div>
    <div class="tc tc-h">
        <div class="tc-icon">✦</div>
        <div class="tc-sev">No Tumour</div>
        <div class="tc-name">Healthy</div>
        <div class="tc-desc">No detectable abnormal mass. Brain parenchyma appears structurally normal.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  MAIN LAYOUT — UPLOAD LEFT | RESULTS RIGHT
# ═══════════════════════════════════════════════════════════════
left, right = st.columns([1, 1.2], gap="large")

with left:
    st.markdown('<div class="sec-title">Upload MRI Scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed",
        help="Upload a brain MRI scan in JPG or PNG format.",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # MRI viewer with scanline
        b64_raw = img_to_b64(image)
        st.markdown(f"""
        <div class="mri-viewer">
            <div class="mri-viewer-inner">
                <div class="mri-scanline"></div>
                <div class="mri-corner tl"></div>
                <div class="mri-corner tr"></div>
                <div class="mri-corner bl"></div>
                <div class="mri-corner br"></div>
                <img src="data:image/png;base64,{b64_raw}"
                     style="width:100%;display:block;border-radius:13px;"/>
                <div class="mri-label">MRI SCAN · LOADED</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # View toggles
        view_col1, view_col2 = st.columns(2)
        with view_col1:
            show_enhanced = st.checkbox("🔬 Enhanced contrast view", value=False)
        with view_col2:
            show_heatmap = st.checkbox("🌡️ Attention heatmap", value=False)

        if show_enhanced:
            enh = make_enhanced(image)
            st.image(enh, use_container_width=True, caption="Enhanced Contrast")
        if show_heatmap:
            hmap = draw_heatmap_overlay(image)
            st.image(hmap, use_container_width=True, caption="Attention Map (Simulated)")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Image info
        w, h = image.size
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;margin-bottom:1rem;">
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                        border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-family:var(--font-mn);font-size:0.55rem;color:var(--muted);letter-spacing:0.15em;">WIDTH</div>
                <div style="font-family:var(--font-hd);font-size:1rem;font-weight:700;color:var(--cyan);">{w}px</div>
            </div>
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                        border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-family:var(--font-mn);font-size:0.55rem;color:var(--muted);letter-spacing:0.15em;">HEIGHT</div>
                <div style="font-family:var(--font-hd);font-size:1rem;font-weight:700;color:var(--cyan);">{h}px</div>
            </div>
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                        border-radius:8px;padding:0.6rem;text-align:center;">
                <div style="font-family:var(--font-mn);font-size:0.55rem;color:var(--muted);letter-spacing:0.15em;">MODE</div>
                <div style="font-family:var(--font-hd);font-size:1rem;font-weight:700;color:var(--cyan);">RGB</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("⬡  RUN DIAGNOSTIC ANALYSIS", use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 1rem;font-family:var(--font-mn);
                    font-size:0.65rem;letter-spacing:0.2em;color:var(--muted);">
            <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.4;">🧠</div>
            AWAITING MRI UPLOAD
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

    st.markdown('</div>', unsafe_allow_html=True)  # /upload-panel


# ══ RIGHT COLUMN ══
with right:
    st.markdown('<div class="sec-title">Diagnostic Output</div>', unsafe_allow_html=True)

    if uploaded_file and run_btn:
        prog_placeholder = st.empty()
        spin_placeholder = st.empty()

        with spin_placeholder:
            with st.spinner("Running neural inference…"):
                prog_bar = prog_placeholder.progress(0)
                stages = [
                    (15,  "Preprocessing image…"),
                    (40,  "Loading neural weights…"),
                    (65,  "Forward pass…"),
                    (85,  "Softmax normalisation…"),
                    (100, "Complete."),
                ]
                for pct, msg in stages:
                    time.sleep(0.18)
                    prog_bar.progress(pct)

                img_r  = image.resize((128, 128))
                arr    = np.expand_dims(np.array(img_r) / 255.0, axis=0)
                preds  = model.predict(arr, verbose=0)[0] if model else np.array([0.25,0.25,0.25,0.25])
                idx    = int(np.argmax(preds))
                label  = CLASS_NAMES[idx]
                conf   = float(preds[idx]) * 100
                tumor_prob = sum(float(preds[i]) for i,n in enumerate(CLASS_NAMES) if n != "Healthy") * 100

        prog_placeholder.empty()
        spin_placeholder.empty()

        ts_str = datetime.now().strftime("%H:%M:%S")
        st.session_state.scan_count += 1
        st.session_state.results = {
            "label": label, "conf": conf,
            "preds": preds.tolist(), "ts": ts_str,
            "tumor_prob": tumor_prob,
        }
        st.session_state.analyzed = True
        st.session_state.history.insert(0, {
            "label": label, "conf": conf, "ts": ts_str,
        })
        if len(st.session_state.history) > 10:
            st.session_state.history = st.session_state.history[:10]

    if st.session_state.analyzed and st.session_state.results:
        res   = st.session_state.results
        label = res["label"]
        conf  = res["conf"]
        preds = res["preds"]
        info  = TUMOR_DB[label]
        kind  = info["kind"]

        db_cls   = f"db-{kind}"
        cbs_cls  = f"cv-{kind}"
        cf_cls   = f"cf-{kind}"
        dot_cls  = f"dot-{kind}" if kind=="healthy" else "dot-tumor"
        dot_col  = "var(--green)" if kind=="healthy" else "var(--red)"

        level = 'HIGH' if conf>85 else 'MED' if conf>70 else 'LOW'
        prob_rows_html = ""
        for i, cname in enumerate(CLASS_NAMES):
            pct = float(preds[i]) * 100
            nm_cls = "active" if cname == label else ""
            pf_cls = f"pf-active-{kind}" if cname == label else "pf-default"
            prob_rows_html += f'<div class="prob-row"><div class="pr-name {nm_cls}">{cname}</div><div class="pr-track"><div class="pr-fill {pf_cls}" style="width:{pct:.1f}%"></div></div><div class="pr-pct {nm_cls}">{pct:.1f}%</div></div>'
        steps_html = "".join(f'<li><div class="step-n">0{i+1}</div>{s}</li>' for i, s in enumerate(info["steps"]))
        st.markdown(f"""
        <div class="result-panel">
            <div class="result-header">
                <div class="rh-label">
                    <div class="rh-live" style="background:{dot_col};box-shadow:0 0 8px {dot_col};"></div>
                    Diagnostic Result
                </div>
                <div class="rh-ts">{res['ts']}</div>
            </div>
            <div class="result-body">
                <div class="diag-big {db_cls}">{info['icon']} {label}</div>
                <div class="diag-short">{info['short']}</div>
                <div class="conf-bar-section">
                    <div class="cbs-top">
                        <span class="cbs-label">Model Confidence</span>
                        <span class="cbs-val {cbs_cls}">{conf:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill {cf_cls}" style="width:{conf:.1f}%"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-family:var(--font-mn);font-size:0.52rem;color:var(--muted);margin-top:0.35rem;margin-bottom:1.5rem;">
                        <span>TUMOR PROB: {res['tumor_prob']:.1f}%</span>
                        <span>LEVEL: {level}</span>
                    </div>
                </div>
                <div class="prob-title">All Class Probabilities</div>
                <div class="prob-rows">{prob_rows_html}</div>
                <div class="detail-grid">
                    <div class="detail-box">
                        <div class="db-t">Clinical Summary</div>
                        <div class="db-body">{info['summary']}</div>
                    </div>
                    <div class="detail-box">
                        <div class="db-t">Key Symptoms</div>
                        <div class="db-body" style="margin-bottom:1rem;">{info['symptoms']}</div>
                        <div class="db-t">Next Steps</div>
                        <ul class="step-list">{steps_html}</ul>
                    </div>
                </div>
                <div class="disclaimer">
                    <strong>&#9888; Medical Disclaimer:</strong> This AI output is for research and educational purposes only. It is <strong>not a substitute</strong> for professional radiological or neurological evaluation. Always consult a qualified clinician for medical decisions.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if label == "Healthy":
            st.balloons()

    else:
        st.markdown("""
        <div style="background:var(--panel);border:1px solid var(--border);
                    border-radius:18px;padding:4rem 2rem;text-align:center;
                    backdrop-filter:blur(12px);">
            <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.25;">🔬</div>
            <div style="font-family:var(--font-mn);font-size:0.62rem;
                        letter-spacing:0.22em;color:var(--muted);">
                AWAITING DIAGNOSTIC INPUT
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  BOTTOM TABS — HISTORY · ABOUT · HOW IT WORKS
# ═══════════════════════════════════════════════════════════════
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋  Scan History", "ℹ️  About the Model", "🧬  How It Works"])

with tab1:
    st.markdown('<div class="sec-title" style="margin-bottom:1rem;">Recent Scan History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown('<div class="hist-empty">NO SCANS IN THIS SESSION YET</div>', unsafe_allow_html=True)
    else:
        for entry in st.session_state.history:
            kind_h  = "healthy" if entry["label"] == "Healthy" else "tumor"
            nm_cls  = f"hn-{kind_h}"
            bdg_cls = f"hb-{kind_h}"
            st.markdown(f"""
            <div class="hist-row">
                <div>
                    <div class="hr-name {nm_cls}">{entry['label']}</div>
                    <div class="hr-conf">{entry['conf']:.1f}% confidence</div>
                </div>
                <div class="hr-badge {bdg_cls}">{'TUMOUR' if kind_h=='tumor' else 'HEALTHY'}</div>
                <div class="hr-time">{entry['ts']}</div>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🗑  Clear History"):
            st.session_state.history = []
            st.session_state.analyzed = False
            st.session_state.results = None
            st.rerun()

with tab2:
    st.markdown("""
    <div class="about-grid">
        <div class="about-card">
            <div class="ac-head">Model Architecture</div>
            <div class="ac-body">
                NeuroScan AI uses a <strong>Convolutional Neural Network (CNN)</strong> built with
                TensorFlow/Keras. The model was trained on a curated MRI dataset spanning all four
                classes — Glioma, Meningioma, Pituitary, and Healthy.
                Input images are resized to <strong>128×128 pixels</strong> and normalised
                to [0,1] before inference. The final softmax layer outputs probability
                distributions across all four classes simultaneously.
            </div>
        </div>
        <div class="about-card">
            <div class="ac-head">Classification Categories</div>
            <div class="ac-body">
                <strong style="color:var(--red)">Glioma</strong> — malignant glial cell tumour, highest severity<br><br>
                <strong style="color:var(--amber)">Meningioma</strong> — typically benign meningeal tumour<br><br>
                <strong style="color:var(--violet)">Pituitary</strong> — benign adenoma with hormonal impact<br><br>
                <strong style="color:var(--green)">Healthy</strong> — no detectable tumour pathology
            </div>
        </div>
        <div class="about-card">
            <div class="ac-head">Confidence Scoring</div>
            <div class="ac-body">
                Each prediction outputs a <strong>softmax probability vector</strong> across all
                four classes. The highest probability class is the primary diagnosis.
                A confidence above <strong>85%</strong> is considered high-certainty.
                Between 70–85% is moderate; below 70% warrants additional clinical review.
                All probabilities are displayed for full transparency.
            </div>
        </div>
        <div class="about-card">
            <div class="ac-head">Limitations & Disclaimer</div>
            <div class="ac-body">
                This tool is a <strong>research prototype</strong> and educational aid.
                It has not been validated in clinical settings and should never be used
                as a sole basis for medical diagnosis. MRI interpretation requires
                trained radiological expertise, clinical context, and formal reporting.
                Always consult a qualified neuroradiologist or neurologist.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="max-width:700px;">
        <div style="font-family:var(--font-mn);font-size:0.6rem;letter-spacing:0.2em;
                    text-transform:uppercase;color:var(--cyan);margin-bottom:1.5rem;">
            Inference Pipeline
        </div>
    """, unsafe_allow_html=True)

    pipeline_steps = [
        ("01", "Image Upload", "User uploads a brain MRI scan in JPG/PNG format."),
        ("02", "Preprocessing", "Image is resized to 128×128px and pixel values normalised to [0, 1]."),
        ("03", "CNN Inference", "The preprocessed tensor passes through the trained CNN layers — convolutional, pooling, dense."),
        ("04", "Softmax Output", "The final dense layer outputs a 4-class probability distribution via softmax activation."),
        ("05", "Classification", "The class with the highest probability is selected as the primary diagnosis."),
        ("06", "Result Display", "Confidence score, probability breakdown, clinical summary and next steps are rendered."),
    ]
    for num, title, desc in pipeline_steps:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;margin-bottom:1rem;align-items:flex-start;">
            <div style="flex-shrink:0;width:36px;height:36px;border-radius:50%;
                        border:1px solid rgba(0,220,255,0.3);background:rgba(0,220,255,0.05);
                        display:flex;align-items:center;justify-content:center;
                        font-family:var(--font-mn);font-size:0.55rem;color:var(--cyan);
                        margin-top:2px;">{num}</div>
            <div>
                <div style="font-family:var(--font-hd);font-size:0.9rem;font-weight:700;
                            color:#9ab8d8;margin-bottom:0.2rem;">{title}</div>
                <div style="font-size:0.78rem;color:#4a6080;line-height:1.65;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="ns-footer">
    NEUROSCAN AI &nbsp;<span>·</span>&nbsp;
    DEEP LEARNING BRAIN TUMOUR CLASSIFIER &nbsp;<span>·</span>&nbsp;
    RESEARCH USE ONLY &nbsp;<span>·</span>&nbsp;
    SESSION SCANS: {st.session_state.scan_count}
</div>
""", unsafe_allow_html=True)

#how to run:
# 1. Save this code in a file named `TumorTesting.py`.
# 2. Ensure you have Streamlit and TensorFlow installed in your Python environment:
#    pip install streamlit tensorflow pillow numpy
# 3. Place the trained model file `brain_tumor_model.keras` in the same directory as this script.
# 4. Run the Streamlit app from the terminal:
#   .\venv\Scripts\python.exe -m streamlit run TumorTesting.py