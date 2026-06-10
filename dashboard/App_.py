import sys
import datetime as dt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import plotly.graph_objects as go
import streamlit as st

import datalib as dl

st.set_page_config(page_title="Water waste detection",
                   layout="wide", initial_sidebar_state="expanded")

# accente comune ambelor teme
ACCENT, AMBER, CORAL = "#14B8A6", "#E08A2C", "#DC5A3D"
TYPE_PALETTE = ["#14B8A6", "#1FB6A6", "#3FA7B5", "#17968B", "#5DCAA5",
                "#8AC6CE", "#9FE1CB", "#6E8A92", "#46606A"]

DARK = dict(page="#0C1416", panel="#121E22", panel2="#16252A", border="#213138",
            text="#D6E2E5", muted="#7C9098", grid="#1C2B31",
            step1="#142A2A", step2="#2E2A1A", step3="#33201A")
LIGHT = dict(page="#F4F7F8", panel="#FFFFFF", panel2="#EDF2F3", border="#D8E2E3",
             text="#0B2027", muted="#51646C", grid="#E4ECED",
             step1="#D7F0E9", step2="#FBE3C7", step3="#F3C9B0")


def inject_css(p):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');
:root {{
  --page:{p['page']}; --panel:{p['panel']}; --panel2:{p['panel2']};
  --border:{p['border']}; --text:{p['text']}; --muted:{p['muted']};
  --accent:{ACCENT}; --amber:{AMBER}; --coral:{CORAL};
}}
.stApp {{ background: var(--page); }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; color:var(--text); }}
#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding-top:.8rem; max-width:1520px; }}

/* sidebar + widgets native */
section[data-testid="stSidebar"] {{ background:var(--panel); border-right:1px solid var(--border); }}
section[data-testid="stSidebar"] * {{ color:var(--text); }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
[data-testid="stDateInput"] input {{ background:var(--panel2) !important;
  border-color:var(--border) !important; color:var(--text) !important; }}
[data-baseweb="popover"], [role="listbox"], [data-baseweb="menu"] {{
  background:var(--panel2) !important; color:var(--text) !important; }}
[data-baseweb="tag"] {{ background:var(--accent) !important; color:#06201D !important; }}
.stButton button {{ background:var(--panel2); color:var(--text); border:1px solid var(--border); }}
.stButton button:hover {{ border-color:var(--accent); color:var(--accent); }}
.stDownloadButton button {{ background:var(--panel2); color:var(--text); border:1px solid var(--border); }}

/* lizibilitate text in ambele teme */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{ color:var(--text); }}
[data-testid="stMarkdownContainer"] strong {{ color:var(--text); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color:var(--muted) !important; }}
code, [data-testid="stMarkdownContainer"] code {{ background:var(--panel2) !important;
  color:var(--accent) !important; border:1px solid var(--border); padding:1px 6px;
  border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:.78rem; }}
/* accent teal pentru radio/toggle in loc de rosu */
[role="switch"][aria-checked="true"] {{ background-color:var(--accent) !important; }}
[data-testid="stRadio"] [aria-checked="true"] svg,
[data-baseweb="radio"] [aria-checked="true"] svg {{ fill:var(--accent) !important; }}
[data-baseweb="radio"] div[aria-checked="true"] {{ border-color:var(--accent) !important; }}

/* top toolbar */
.bar {{ display:flex; justify-content:space-between; align-items:center;
  border:1px solid var(--border); border-radius:6px; background:var(--panel); padding:14px 18px; }}
.bar h1 {{ font-size:1.15rem; font-weight:600; margin:0; letter-spacing:.3px; color:var(--text); }}
.bar .meta {{ font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--muted); margin-top:3px; }}
.status {{ font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--muted);
  display:flex; align-items:center; gap:14px; }}
.dot {{ width:8px; height:8px; border-radius:50%; background:var(--accent);
  box-shadow:0 0 8px var(--accent); display:inline-block; margin-right:6px; }}
.status b {{ color:var(--text); }}

.eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:.7rem; text-transform:uppercase;
  letter-spacing:1.5px; color:var(--muted); margin:14px 0 6px;
  border-left:2px solid var(--accent); padding-left:8px; }}

.kpi {{ border:1px solid var(--border); border-radius:6px; background:var(--panel);
  padding:13px 15px; height:100%; }}
.kpi .lab {{ font-family:'JetBrains Mono',monospace; font-size:.66rem; text-transform:uppercase;
  letter-spacing:1px; color:var(--muted); }}
.kpi .val {{ font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.9rem;
  line-height:1.1; margin-top:6px; color:var(--text); }}
.kpi .sub {{ font-size:.72rem; color:var(--muted); margin-top:3px; }}
.kpi .dlt {{ font-family:'JetBrains Mono',monospace; font-size:.7rem; margin-top:5px; }}
.up {{ color:var(--accent); }} .down {{ color:var(--amber); }} .flat {{ color:var(--muted); }}
.accent-aqua {{ border-top:2px solid var(--accent); }}
.accent-amber {{ border-top:2px solid var(--amber); }}
.spk {{ width:100%; height:30px; display:block; margin-top:10px; }}

.scard {{ border:1px solid var(--border); border-radius:6px; background:var(--panel); padding:16px 18px; }}
.scard .big {{ font-family:'JetBrains Mono',monospace; font-size:1.55rem; font-weight:700; color:var(--accent); }}
.scard .sm {{ font-family:'JetBrains Mono',monospace; font-size:.68rem; color:var(--muted);
  text-transform:uppercase; letter-spacing:1px; margin-top:2px; }}

.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ font-family:'JetBrains Mono',monospace; font-size:.74rem;
  font-weight:500; text-transform:uppercase; letter-spacing:1px; color:var(--muted);
  padding:9px 20px; border:1px solid transparent; border-radius:6px 6px 0 0; margin-bottom:-1px; }}
.stTabs [data-baseweb="tab"]:hover {{ color:var(--text); background:var(--panel2); }}
.stTabs [aria-selected="true"] {{ color:var(--text); background:var(--panel);
  border-color:var(--border); border-bottom-color:var(--panel); border-top:2px solid var(--accent); }}
.stTabs [data-baseweb="tab-highlight"] {{ display:none; }}
.stTabs [data-baseweb="tab-border"] {{ background-color:transparent; }}

/* tabel propriu de evenimente */
.tbl {{ width:100%; border-collapse:collapse; font-size:.8rem; }}
.tbl th {{ font-family:'JetBrains Mono',monospace; font-size:.64rem; text-transform:uppercase;
  letter-spacing:1px; color:var(--muted); text-align:left; padding:8px 10px;
  border-bottom:1px solid var(--border); }}
.tbl td {{ padding:7px 10px; border-bottom:1px solid var(--border); color:var(--text); }}
.tbl tr:hover td {{ background:var(--panel2); }}
.mono {{ font-family:'JetBrains Mono',monospace; }}
.bdg {{ font-family:'JetBrains Mono',monospace; font-size:.66rem; font-weight:700;
  padding:2px 7px; border-radius:4px; }}
.b-col {{ background:rgba(20,184,166,.16); color:var(--accent); }}
.b-alm {{ background:rgba(224,138,44,.16); color:var(--amber); }}
.b-stp {{ background:var(--panel2); color:var(--muted); }}
.cbar {{ height:7px; border-radius:4px; background:var(--panel2); position:relative; min-width:70px; }}
.cbar > span {{ position:absolute; left:0; top:0; bottom:0; border-radius:4px; background:var(--accent); }}
</style>
""", unsafe_allow_html=True)


PERIODS = {"Azi": 1, "Ultimele 7 zile": 7, "Ultimele 30 zile": 30}


def style_chart(fig, p, height=300, legend=False):
    fig.update_layout(height=height, margin=dict(t=10, b=10, l=10, r=14),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=p["text"], size=12), showlegend=legend,
        xaxis=dict(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["border"]),
        yaxis=dict(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["border"]))
    return fig


def sparkline(values, color):
    """Mini-grafic inline SVG (tendinta pe 7 zile) pentru cardurile KPI."""
    if not values:
        return ""
    w, h, pad = 150, 30, 3
    mx, mn = max(values), min(values)
    rng = (mx - mn) or 1
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = pad + (i * (w - 2 * pad) / (n - 1) if n > 1 else 0)
        y = h - pad - (v - mn) / rng * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    line = " ".join(pts)
    area = f"{pad},{h - pad} {line} {w - pad},{h - pad}"
    lx, ly = pts[-1].split(",")
    return (f'<svg class="spk" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polygon points="{area}" fill="{color}" fill-opacity="0.10"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" stroke-linejoin="round"/>'
            f'<circle cx="{lx}" cy="{ly}" r="2.2" fill="{color}"/></svg>')


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("**SETARI**")
    dark = st.toggle("Mod intunecat", value=False)
    source = st.radio("Sursa de date", ["Demo", "Live (local)", "AWS DynamoDB"], index=2)
    period_label = st.selectbox("Perioada", list(PERIODS.keys()), index=1)
    end_day = st.date_input("Pana in ziua", value=dt.date.today())
    st.caption("Variatiile (+/- %) sunt fata de perioada anterioara, egala.")
    if st.button("Reincarca", use_container_width=True):
        st.cache_data.clear()

P = DARK if dark else LIGHT
inject_css(P)
days = PERIODS[period_label]

# Credentiale AWS portabile: daca exista .streamlit/secrets.toml cu o sectiune
# [aws], le punem in mediu, iar boto3 (din dynamodb_reader) le foloseste automat.
# Asa dashboard-ul merge pe orice laptop care are fisierul secrets.toml, fara
# `aws configure` local si fara chei scrise in cod.
import os as _os
try:
    if "aws" in st.secrets:
        _aws = st.secrets["aws"]
        _os.environ.setdefault("AWS_ACCESS_KEY_ID", _aws["access_key_id"])
        _os.environ.setdefault("AWS_SECRET_ACCESS_KEY", _aws["secret_access_key"])
        _os.environ.setdefault("AWS_DEFAULT_REGION", _aws.get("region", "eu-north-1"))
except Exception:  # noqa: BLE001 — niciun secrets.toml prezent
    pass


# --------------------------------------------------------------------------- #
# Date
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=15, show_spinner=False)
def load_demo():
    return dl.generate_demo_data()

@st.cache_data(ttl=5, show_spinner=False)
def load_live():
    return dl.load_live_events(str(PROJECT_ROOT / "dashboard" / "live_events.jsonl"))

@st.cache_data(ttl=15, show_spinner=False)
def load_dynamodb():
    from cloud.dynamodb_reader import get_all_events
    return get_all_events()

try:
    if source == "Demo":
        events = load_demo()
    elif source == "Live (local)":
        events = load_live()
        if not events:
            st.info("Inca nu exista evenimente live. Porneste live_camera.py "
                    "si arata un deseu camerei, apoi apasa Reincarca.")
    else:
        events = load_dynamodb()
except Exception as exc:  # noqa: BLE001
    events = []
    st.error(f"Nu am putut citi din DynamoDB: {exc}")
    st.caption("Verifica `aws configure` si tabela PlasticDetectorEvents (eu-north-1). "
               "Pana atunci foloseste sursa Demo.")

df = dl.events_to_dataframe(events)
cur, prev = dl.kpis_with_delta(df, end_day, days)
sl = dl.period_slice(df, end_day, days)


# --------------------------------------------------------------------------- #
# Toolbar
# --------------------------------------------------------------------------- #
st.markdown(f"""
<div class="bar">
  <div>
    <h1>Water waste detection</h1>
    <div class="meta">detectie AI (SAHI + YOLO) // colectare automata // AWS IoT + DynamoDB</div>
  </div>
  <div class="status">
    <span><span class="dot"></span>{source.upper()}</span>
    <span>PERIOADA <b>{period_label}</b></span>
    <span>SYNC <b>{dt.datetime.now():%H:%M:%S}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)


def delta_html(now, before, good_up=True):
    pc = dl.pct_change(now, before)
    if pc is None:
        return '<div class="dlt flat">-- fara referinta</div>'
    if pc == 0:
        return '<div class="dlt flat">= 0% vs anterior</div>'
    up = pc > 0
    cls = "up" if (up == good_up) else "down"
    return f'<div class="dlt {cls}">{"+" if up else "-"}{abs(int(pc))}% vs anterior</div>'


def kpi(col, label, value, sub, accent="", delta="", spark=""):
    col.markdown(f'<div class="kpi {accent}"><div class="lab">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div>'
                 f'{delta}{spark}</div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">indicatori cheie</div>', unsafe_allow_html=True)
SPK_DAYS = 7
c = st.columns(5)
kpi(c[0], "Evenimente", cur["events"], "trimise in cloud", "",
    delta_html(cur["events"], prev["events"]),
    sparkline(dl.daily_series(df, end_day, SPK_DAYS, "events"), ACCENT))
kpi(c[1], "Colectate / servo", cur["collected"], f'rata {cur["collection_rate"]}%',
    "accent-aqua", delta_html(cur["collected"], prev["collected"]),
    sparkline(dl.daily_series(df, end_day, SPK_DAYS, "collected"), ACCENT))
kpi(c[2], "Alarme / LED+buzzer", cur["alarms"], f'incredere {cur["avg_confidence"]}',
    "accent-amber", delta_html(cur["alarms"], prev["alarms"], good_up=False),
    sparkline(dl.daily_series(df, end_day, SPK_DAYS, "alarms"), AMBER))
kpi(c[3], "Deseu prea mare", cur["alarms_large"], "necolectabil", "accent-amber",
    delta_html(cur["alarms_large"], prev["alarms_large"], good_up=False),
    sparkline(dl.daily_series(df, end_day, SPK_DAYS, "alarms_large"), AMBER))
kpi(c[4], "Obiecte / frame", cur["avg_load"], f'varf {cur["peak_load"]}', "",
    delta_html(cur["avg_load"], prev["avg_load"], good_up=False),
    sparkline(dl.daily_series(df, end_day, SPK_DAYS, "avg_load"), CORAL))


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_over, tab_trend, tab_events, tab_sys = st.tabs(
    ["Ansamblu", "Tendinte", "Evenimente", "Sistem"])


def empty_note():
    st.caption("Nicio detectie in perioada selectata. "
               "Pe Demo, datele acopera ultimele ~5 saptamani.")


with tab_over:
    g1, g2, g3 = st.columns([0.9, 1.1, 1])
    with g1:
        st.markdown('<div class="eyebrow">indice de incarcare</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=cur["avg_load"],
            number={"suffix": " obj/frame",
                    "font": {"size": 24, "color": P["text"], "family": "JetBrains Mono"}},
            gauge={"axis": {"range": [0, 26], "tickcolor": P["muted"]},
                   "bgcolor": P["panel"], "borderwidth": 0,
                   "bar": {"color": ACCENT, "thickness": 0.28},
                   "steps": [{"range": [0, 5], "color": P["step1"]},
                             {"range": [5, 15], "color": P["step2"]},
                             {"range": [15, 26], "color": P["step3"]}],
                   "threshold": {"line": {"color": CORAL, "width": 4},
                                 "thickness": 0.8, "value": dl.HIGH_POLLUTION_THRESHOLD}}))
        gauge.update_layout(height=300, margin=dict(t=14, b=10, l=22, r=22),
                            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=P["text"]))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption(f"Prag alarma de poluare: {dl.HIGH_POLLUTION_THRESHOLD} obiecte/frame.")

    with g2:
        st.markdown('<div class="eyebrow">deseuri pe tip</div>', unsafe_allow_html=True)
        bt = dl.waste_by_type(sl)
        if bt.empty:
            empty_note()
        else:
            donut = go.Figure(go.Pie(
                labels=bt["selected_label"], values=bt["count"], hole=.62,
                marker=dict(colors=TYPE_PALETTE[:len(bt)], line=dict(color=P["page"], width=1.5)),
                textinfo="none", sort=False))
            donut.update_layout(height=300, margin=dict(t=10, b=44, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=P["text"]),
                legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center",
                            font=dict(size=10.5, color=P["muted"])),
                annotations=[dict(text=f'<b>{int(bt["count"].sum())}</b><br>obiecte',
                                  showarrow=False,
                                  font=dict(size=16, color=P["text"], family="JetBrains Mono"))])
            st.plotly_chart(donut, use_container_width=True)

    with g3:
        st.markdown('<div class="eyebrow">distributie actiuni</div>', unsafe_allow_html=True)
        stop = max(0, cur["events"] - cur["collected"] - cur["alarms"])
        bar = go.Figure(go.Bar(
            x=[cur["collected"], cur["alarms"], stop],
            y=["Colectate", "Alarme", "Apa curata (STOP)"],
            orientation="h", marker_color=[ACCENT, AMBER, P["muted"]],
            text=[cur["collected"], cur["alarms"], stop],
            textposition="outside", textfont=dict(color=P["text"])))
        style_chart(bar, P)
        bar.update_layout(yaxis=dict(showgrid=False, linecolor=P["border"]))
        st.plotly_chart(bar, use_container_width=True)
        r = dl.alarm_reasons(sl)
        st.caption(f"Din alarme: {r['Deseu prea mare']} deseu prea mare // "
                   f"{r['Poluare ridicata']} poluare ridicata.")


with tab_trend:
    st.markdown('<div class="eyebrow">evolutie zilnica: colectate vs alarme</div>',
                unsafe_allow_html=True)
    hist = dl.daily_history(sl)
    if hist.empty:
        empty_note()
    else:
        hist["date"] = hist["date"].astype(str)
        line = go.Figure()
        line.add_scatter(x=hist["date"], y=hist["Colectate"], name="Colectate",
                         mode="lines+markers", line=dict(color=ACCENT, width=2.5),
                         fill="tozeroy", fillcolor="rgba(20,184,166,.10)")
        line.add_scatter(x=hist["date"], y=hist["Alarme"], name="Alarme",
                         mode="lines+markers", line=dict(color=AMBER, width=2.5))
        style_chart(line, P, legend=True)
        line.update_layout(legend=dict(orientation="h", y=1.14, x=0, font=dict(color=P["muted"])),
                           xaxis=dict(showgrid=False, linecolor=P["border"], gridcolor=P["grid"]))
        st.plotly_chart(line, use_container_width=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="eyebrow">activitate pe ore</div>', unsafe_allow_html=True)
        ha = dl.hourly_activity(sl)
        if ha["count"].sum() == 0:
            empty_note()
        else:
            hb = go.Figure(go.Bar(x=ha["hour"], y=ha["count"], marker_color=ACCENT))
            style_chart(hb, P, height=270)
            hb.update_layout(xaxis=dict(title="ora", showgrid=False, dtick=3,
                                        linecolor=P["border"], gridcolor=P["grid"]))
            st.plotly_chart(hb, use_container_width=True)
    with t2:
        st.markdown('<div class="eyebrow">distributia increderii modelului</div>',
                    unsafe_allow_html=True)
        cv = dl.confidence_values(sl)
        if not cv:
            empty_note()
        else:
            hf = go.Figure(go.Histogram(x=cv, nbinsx=12, marker_color="#1FB6A6"))
            style_chart(hf, P, height=270)
            hf.update_layout(xaxis=dict(title="confidence", showgrid=False,
                                        range=[0.5, 1.0], linecolor=P["border"], gridcolor=P["grid"]))
            st.plotly_chart(hf, use_container_width=True)


with tab_events:
    f1, f2 = st.columns(2)
    actions = f1.multiselect("Filtreaza dupa actiune",
                             ["COLLECT", "ALARM", "STOP"], default=["COLLECT", "ALARM"])
    types = sorted(sl["selected_label"].dropna().unique().tolist())
    chosen = f2.multiselect("Filtreaza dupa tip", types, default=types)

    view = sl[sl["action"].isin(actions)]
    if chosen:
        view = view[view["selected_label"].isin(chosen) | view["selected_label"].isna()]
    view = view.sort_values("timestamp", ascending=False)
    st.caption(f"{len(view)} evenimente in perioada selectata (afisate primele 60).")

    badge = {"COLLECT": "b-col", "ALARM": "b-alm", "STOP": "b-stp"}
    rows = ""
    for _, e in view.head(60).iterrows():
        conf = e["selected_confidence"]
        cbar = (f'<div class="cbar"><span style="width:{conf*100:.0f}%"></span></div>'
                if conf == conf else "—")  # conf==conf -> not NaN
        rows += (f'<tr><td class="mono">{e["timestamp"]:%d %b %H:%M:%S}</td>'
                 f'<td><span class="bdg {badge.get(e["action"], "b-stp")}">{e["action"]}</span></td>'
                 f'<td>{e["reason"] or "—"}</td><td class="mono">{e["selected_label"] or "—"}</td>'
                 f'<td>{cbar}</td><td class="mono">{e["detections_count"]}</td>'
                 f'<td class="mono">{e["device_id"]}</td></tr>')
    table = (f'<table class="tbl"><thead><tr><th>Data/ora</th><th>Actiune</th>'
             f'<th>Motiv</th><th>Tip</th><th>Incredere</th><th>Obiecte</th>'
             f'<th>Dispozitiv</th></tr></thead><tbody>{rows}</tbody></table>')
    if view.empty:
        empty_note()
    else:
        st.markdown(table, unsafe_allow_html=True)

    csv = view[["timestamp", "action", "reason", "selected_label",
                "selected_confidence", "detections_count", "device_id"]]
    st.download_button("Descarca CSV", data=csv.to_csv(index=False).encode("utf-8"),
                       file_name=f"water_waste_{end_day}.csv", mime="text/csv")


with tab_sys:
    s = dl.system_stats(df)

    def scard(col, big, sm):
        col.markdown(f'<div class="scard"><div class="big">{big}</div>'
                     f'<div class="sm">{sm}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">totaluri (tot istoricul)</div>', unsafe_allow_html=True)
    sc = st.columns(4)
    scard(sc[0], s["total_events"], "evenimente total")
    scard(sc[1], s["collected_all"], "colectate total")
    scard(sc[2], s["alarms_all"], "alarme total")
    scard(sc[3], s["active_days"], "zile cu activitate")

    st.markdown('<div class="eyebrow">stare dispozitive</div>', unsafe_allow_html=True)
    st.markdown(f"Ultimul eveniment primit: **{s['last_seen']}**")
    if s["devices"]:
        for dev in s["devices"]:
            ddf = df[df["device_id"] == dev]
            last = ddf["timestamp"].max()
            online = (dt.datetime.now() - last.to_pydatetime()).total_seconds() < 86400
            state = "ACTIV (24h)" if online else "INACTIV"
            st.markdown(f"- `{dev}` — {state} // {len(ddf)} evenimente // ultimul {last:%d %b %H:%M}")
    src_txt = {"Demo": "date demo locale",
               "Live (local)": "camera locala (live_events.jsonl)"}.get(
                   source, "AWS DynamoDB / PlasticDetectorEvents (eu-north-1)")
    st.caption("Sursa: " + src_txt)

st.caption("Proof-of-concept // Edge-AI Plastic Detector // AWS IoT Core + DynamoDB")