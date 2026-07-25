# -*- coding: utf-8 -*-
"""
CallAI Analytics - Streamlit App (SaaS Edition)
================================================
Fetches CDR call data from the CRM, filters out abandoned (VDCL) calls,
lets you pick a call-type bucket, runs Silero VAD for talk-time/silence
metrics, transcribes calls with Groq Whisper and scores sentiment with a
Groq LLM, and produces downloadable Excel reports plus emailed
performance/defaulter alerts for mentors.

Requires: streamlit >= 1.32 (uses st.dialog for the Email Portal).
"""

import os
import re
import io
import time
import shutil
import smtplib
import subprocess
import tempfile
import threading
import schedule
from datetime import date, timedelta, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from urllib.parse import urljoin
from pathlib import Path
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import librosa
from bs4 import BeautifulSoup
import streamlit as st
import torch
import torchaudio
import groq
from supabase import create_client, Client

# Try to import silero_vad, fallback to torch.hub if not available
try:
    from silero_vad import load_silero_vad, VADIterator
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    print("silero_vad package not found, using torch.hub fallback")

# ----------------------------------------------------------------------
# Supabase Configuration
# ----------------------------------------------------------------------
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    # Fallback for local development
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

def get_supabase_client() -> Optional[Client]:
    """Get Supabase client instance."""
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

supabase = get_supabase_client()

# ----------------------------------------------------------------------
# Database helper functions
# ----------------------------------------------------------------------
def load_clients_from_db():
    """Load all clients from Supabase."""
    if not supabase:
        return {}
    try:
        response = supabase.table("clients").select("*").execute()
        clients = {}
        for client in response.data:
            clients[client["name"]] = {
                "company_id": str(client["company_id"]),
                "short_max": client.get("short_max", 120),
                "medium_min": client.get("medium_min", 120),
                "medium_max": client.get("medium_max", 300),
                "large_min": client.get("large_min", 300),
                "extra_large_enabled": client.get("extra_large_enabled", False),
                "extra_large_min": client.get("extra_large_min", 480),
            }
        return clients
    except Exception as e:
        print(f"Error loading clients: {e}")
        return {}

def save_client_to_db(name: str, company_id: str, config: Dict = None):
    """Save or update client in Supabase."""
    if not supabase:
        return False
    try:
        data = {
            "name": name,
            "company_id": company_id,
        }
        if config:
            data.update(config)
        
        # Check if client exists
        existing = supabase.table("clients").select("id").eq("name", name).execute()
        if existing.data:
            supabase.table("clients").update(data).eq("name", name).execute()
        else:
            supabase.table("clients").insert(data).execute()
        return True
    except Exception as e:
        print(f"Error saving client: {e}")
        return False

def delete_client_from_db(name: str):
    """Delete client from Supabase."""
    if not supabase:
        return False
    try:
        supabase.table("clients").delete().eq("name", name).execute()
        return True
    except Exception as e:
        print(f"Error deleting client: {e}")
        return False

def load_mentor_emails_from_db():
    """Load mentor emails from Supabase."""
    if not supabase:
        return []
    try:
        response = supabase.table("mentor_emails").select("email").execute()
        return [row["email"] for row in response.data]
    except Exception as e:
        print(f"Error loading mentor emails: {e}")
        return []

def save_mentor_emails_to_db(emails: List[str]):
    """Save mentor emails to Supabase (sync)."""
    if not supabase:
        return False
    try:
        # Get current emails
        current = supabase.table("mentor_emails").select("email").execute()
        current_set = {row["email"] for row in current.data}
        new_set = set(emails)
        
        # Add new emails
        for email in new_set - current_set:
            supabase.table("mentor_emails").insert({"email": email}).execute()
        
        # Remove old emails
        for email in current_set - new_set:
            supabase.table("mentor_emails").delete().eq("email", email).execute()
        
        return True
    except Exception as e:
        print(f"Error saving mentor emails: {e}")
        return False

def load_smtp_config_from_db():
    """Load SMTP config from Supabase."""
    if not supabase:
        return None
    try:
        response = supabase.table("smtp_config").select("*").limit(1).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error loading SMTP config: {e}")
        return None

def save_smtp_config_to_db(config: Dict):
    """Save SMTP config to Supabase."""
    if not supabase:
        return False
    try:
        existing = supabase.table("smtp_config").select("id").limit(1).execute()
        if existing.data:
            supabase.table("smtp_config").update(config).eq("id", existing.data[0]["id"]).execute()
        else:
            supabase.table("smtp_config").insert(config).execute()
        return True
    except Exception as e:
        print(f"Error saving SMTP config: {e}")
        return False

def load_scheduler_config_from_db():
    """Load scheduler config from Supabase."""
    if not supabase:
        return None
    try:
        response = supabase.table("scheduler_config").select("*").limit(1).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error loading scheduler config: {e}")
        return None

def save_scheduler_config_to_db(config: Dict):
    """Save scheduler config to Supabase."""
    if not supabase:
        return False
    try:
        existing = supabase.table("scheduler_config").select("id").limit(1).execute()
        if existing.data:
            supabase.table("scheduler_config").update(config).eq("id", existing.data[0]["id"]).execute()
        else:
            supabase.table("scheduler_config").insert(config).execute()
        return True
    except Exception as e:
        print(f"Error saving scheduler config: {e}")
        return False

def load_defaulters_config_from_db():
    """Load defaulters config from Supabase."""
    if not supabase:
        return None
    try:
        response = supabase.table("defaulters_config").select("*").limit(1).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error loading defaulters config: {e}")
        return None

def save_defaulters_config_to_db(config: Dict):
    """Save defaulters config to Supabase."""
    if not supabase:
        return False
    try:
        existing = supabase.table("defaulters_config").select("id").limit(1).execute()
        if existing.data:
            supabase.table("defaulters_config").update(config).eq("id", existing.data[0]["id"]).execute()
        else:
            supabase.table("defaulters_config").insert(config).execute()
        return True
    except Exception as e:
        print(f"Error saving defaulters config: {e}")
        return False

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CallAI · Talk-Time + Sentiment",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better UI
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Inter, Roboto, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .callai-hero {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .callai-hero h1 {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: white;
    }
    .callai-hero p {
        font-size: 15px;
        margin: 0;
        opacity: 0.9;
    }
    .step-card {
        background: #FFFFFF;
        border: 1px solid #ECECF4;
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(20, 20, 43, 0.04);
    }
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #4F46E5;
        color: white;
        font-weight: 700;
        font-size: 14px;
        margin-right: 10px;
    }
    .step-title {
        font-size: 17px;
        font-weight: 700;
        color: #14142B;
        display: inline-flex;
        align-items: center;
        margin-bottom: 6px;
    }
    .step-subtitle {
        color: #6E7191;
        font-size: 13.5px;
        margin: 0 0 16px 40px;
    }
    .metric-pill {
        background: #F5F4FF;
        border: 1px solid #E4E1FF;
        border-radius: 14px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-pill .value {
        font-size: 24px;
        font-weight: 800;
        color: #4F46E5;
    }
    .metric-pill .label {
        font-size: 12.5px;
        color: #6E7191;
        margin-top: 2px;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        border: none;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
    }
    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        background: linear-gradient(135deg, #16A34A 0%, #22C55E 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.4rem;
    }
    .status-banner-ok {
        background: #ECFDF5;
        border: 1px solid #6EE7B7;
        color: #065F46;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 13.5px;
    }
    .status-banner-danger {
        background: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 13.5px;
    }
    .agent-card {
        background: #F8F7FF;
        border-left: 4px solid #4F46E5;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .agent-card .agent-name {
        font-weight: 700;
        color: #14142B;
        font-size: 15px;
    }
    .agent-card .agent-stats {
        color: #6E7191;
        font-size: 13px;
        margin-top: 4px;
    }
    .config-box {
        background: #F8F7FF;
        border: 2px solid #E4E1FF;
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
    }
    .email-chip {
        display: inline-block;
        background: #E5E7EB;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px 5px;
        font-size: 13px;
    }
    .email-list-box {
        background: #F8F7FF;
        border: 1px solid #E4E1FF;
        border-radius: 10px;
        padding: 12px 16px;
        min-height: 50px;
        margin-top: 8px;
    }
    .buffer-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 20px;
        background: #F5F4FF;
        border: 1px solid #E4E1FF;
        border-radius: 12px;
        margin: 10px 0;
    }
    .buffer-spinner {
        width: 20px;
        height: 20px;
        border: 3px solid #E4E1FF;
        border-top-color: #4F46E5;
        border-radius: 50%;
        animation: buffer-spin 0.8s linear infinite;
        flex-shrink: 0;
    }
    @keyframes buffer-spin { to { transform: rotate(360deg); } }
    .buffer-label {
        font-weight: 600;
        color: #4F46E5;
        font-size: 14px;
    }
    .success-box {
        background: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 12px 16px;
        color: #065F46;
        font-weight: 600;
        margin: 8px 0;
    }
    .warning-box {
        background: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 12px 16px;
        color: #92400E;
        font-weight: 600;
        margin: 8px 0;
    }
    .info-box {
        background: #DBEAFE;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 12px 16px;
        color: #1E40AF;
        font-weight: 600;
        margin: 8px 0;
    }
    .tab-button {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 14px;
    }
    .tab-button.active {
        background: #4F46E5;
        color: white;
    }
    .tab-button.inactive {
        background: #F3F4F6;
        color: #6B7280;
    }
    .tab-button.inactive:hover {
        background: #E5E7EB;
    }
    .tabs-container {
        display: flex;
        gap: 4px;
        margin-bottom: 20px;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0;
    }
    .client-card {
        background: #F8F7FF;
        border: 1px solid #E4E1FF;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .client-card .client-name {
        font-weight: 700;
        font-size: 16px;
        color: #14142B;
    }
    .client-card .client-details {
        color: #6E7191;
        font-size: 13px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def render_buffer(container, message="Working..."):
    """Show a small animated 'buffering' indicator."""
    container.markdown(
        f'<div class="buffer-box"><div class="buffer-spinner"></div>'
        f'<div class="buffer-label">{message}</div></div>',
        unsafe_allow_html=True,
    )

def fmt_hms(total_seconds):
    if total_seconds is None or (isinstance(total_seconds, float) and np.isnan(total_seconds)):
        return "-"
    total_seconds = int(round(total_seconds))
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def bump_ui_version():
    """Force every version-scoped widget to re-initialize from the latest
    session_state / DB values on the next rerun. Call this right before
    st.rerun() after ANY save/apply/reset action so the UI never shows
    stale values."""
    st.session_state["ui_version"] = st.session_state.get("ui_version", 0) + 1

def vkey(name: str) -> str:
    """Build a widget key that changes whenever bump_ui_version() has been
    called. Streamlit ignores a widget's `value`/`default` param once a
    key already has state - suffixing the key with the current ui_version
    forces the widget to rebuild from the fresh value instead of showing
    a stale cached one."""
    return f"{name}__v{st.session_state.get('ui_version', 0)}"

# ----------------------------------------------------------------------
# CRM + Groq configuration
# ----------------------------------------------------------------------
try:
    CRM_EMAIL = st.secrets["CRM_EMAIL"]
    CRM_PASSWORD = st.secrets["CRM_PASSWORD"]
except Exception:
    CRM_EMAIL = "ispark@dialdesk.in"
    CRM_PASSWORD = "1234"

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = ""

CRM_BASE = "https://crmapi.dialdesk.in"
LOGIN_URL = f"{CRM_BASE}/auth/login"
CDR_URL = f"{CRM_BASE}/report/cdr_report"

# ----------------------------------------------------------------------
# Load configuration from Supabase
# ----------------------------------------------------------------------
def load_all_configs():
    """Load all configurations from Supabase into session state."""
    # Load clients
    db_clients = load_clients_from_db()
    if db_clients:
        st.session_state["clients"] = db_clients
        st.session_state["client_names"] = list(db_clients.keys())
    else:
        # Fallback to default clients if Supabase is empty
        default_clients = {
            "Weebo": {"company_id": "687", "short_max": 90, "medium_min": 90, "medium_max": 240, "large_min": 240, "extra_large_enabled": False, "extra_large_min": 480},
            "Hari Om Pvt Ltd": {"company_id": "689", "short_max": 150, "medium_min": 150, "medium_max": 360, "large_min": 360, "extra_large_enabled": False, "extra_large_min": 480},
            "F1 INFO SOLUTION": {"company_id": "609", "short_max": 120, "medium_min": 120, "medium_max": 300, "large_min": 300, "extra_large_enabled": False, "extra_large_min": 480},
            "Saatvik": {"company_id": "663", "short_max": 120, "medium_min": 120, "medium_max": 300, "large_min": 300, "extra_large_enabled": False, "extra_large_min": 480},
            "Fortum Charge": {"company_id": "395", "short_max": 120, "medium_min": 120, "medium_max": 300, "large_min": 300, "extra_large_enabled": False, "extra_large_min": 480},
            "Alphanso": {"company_id": "629", "short_max": 120, "medium_min": 120, "medium_max": 300, "large_min": 300, "extra_large_enabled": False, "extra_large_min": 480},
        }
        st.session_state["clients"] = default_clients
        st.session_state["client_names"] = list(default_clients.keys())
    
    # Load mentor emails
    db_emails = load_mentor_emails_from_db()
    if db_emails:
        st.session_state["mentor_emails"] = db_emails
    else:
        st.session_state["mentor_emails"] = ["urvi.wadhwa@teammas.in"]
    
    # Load SMTP config
    db_smtp = load_smtp_config_from_db()
    if db_smtp:
        st.session_state["smtp_config"] = db_smtp
    else:
        st.session_state["smtp_config"] = {
            "host": "mail.dialdesk.net",
            "port": 587,
            "username": "tickets@dialdesk.net",
            "password": "DiaL@#433#212",
            "from_email": "tickets@dialdesk.net",
            "use_tls": True,
        }
    
    # Load scheduler config
    db_scheduler = load_scheduler_config_from_db()
    if db_scheduler:
        st.session_state["scheduler_time"] = db_scheduler.get("scheduler_time", "23:00")
        st.session_state["scheduler_enabled"] = db_scheduler.get("scheduler_enabled", True)
        st.session_state["auto_schedule_clients"] = db_scheduler.get("auto_schedule_clients", ["Weebo", "Hari Om Pvt Ltd"])
    else:
        st.session_state["scheduler_time"] = "23:00"
        st.session_state["scheduler_enabled"] = True
        st.session_state["auto_schedule_clients"] = ["Weebo", "Hari Om Pvt Ltd"]
    
    # Load defaulters config
    db_defaulters = load_defaulters_config_from_db()
    if db_defaulters:
        st.session_state["silence_threshold"] = db_defaulters.get("silence_threshold", 30)
        st.session_state["min_calls_per_agent"] = db_defaulters.get("min_calls_per_agent", 3)
    else:
        st.session_state["silence_threshold"] = 30
        st.session_state["min_calls_per_agent"] = 3

# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
defaults = {
    "token": None,
    "cdr_df": None,
    "cdr_client": None,
    "final_df": None,
    "agent_analytics_df": None,
    "vdcl_removed": 0,
    "duration_overrides": {},
    "complete_analysis_df": None,
    "agent_silence_by_category_df": None,
    "defaulter_agents_df": None,
    "mentor_emails": [],
    "last_email_time": None,
    "scheduler_running": False,
    "email_large_threshold_min": None,
    "smtp_config": {},
    "scheduler_time": "23:00",
    "scheduler_enabled": True,
    "clients": {},
    "client_names": [],
    "silence_threshold": 30,
    "min_calls_per_agent": 3,
    "current_tab": "Dashboard",
    "auto_schedule_clients": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load configs from Supabase
load_all_configs()

# ----------------------------------------------------------------------
# CRM functions
# ----------------------------------------------------------------------
def do_login():
    resp = requests.post(
        LOGIN_URL,
        json={"email": CRM_EMAIL, "password": CRM_PASSWORD},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30,
        proxies=None,
    )
    resp.raise_for_status()
    data = resp.json()
    token = (
        data.get("token")
        or data.get("access_token")
        or (data.get("data", {}) or {}).get("token")
    )
    if not token:
        raise RuntimeError(f"Login response had no token field: {data}")
    st.session_state["token"] = token
    return token

def get_valid_token():
    if not st.session_state.get("token"):
        with st.spinner("Signing in..."):
            do_login()
    return st.session_state["token"]

def fetch_cdr(payload, retry_on_401=True):
    token = get_valid_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    resp = requests.post(CDR_URL, json=payload, headers=headers, timeout=120, proxies=None)
    if resp.status_code == 401 and retry_on_401:
        do_login()
        return fetch_cdr(payload, retry_on_401=False)
    return resp

# ----------------------------------------------------------------------
# Column mapping + duration parsing
# ----------------------------------------------------------------------
COLUMN_CANDIDATES = {
    "date": ["call_date", "CallDate", "Date"],
    "time": ["start_time", "Time", "StartTime"],
    "agent_name": ["full_name", "AgentName", "agent", "Agent Name", "agent_name"],
    "call_from": ["phone_number", "PhoneNumber", "Call From"],
    "recording": ["Recording", "RecordingUrl", "RecordingURL", "recording_url"],
}

def find_column(df, keys):
    lower_map = {c.lower(): c for c in df.columns}
    for key in keys:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None

def parse_duration_series_to_seconds(series):
    s = series.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    needs_time_parse = numeric.isna() & s.str.contains(":", na=False)
    if needs_time_parse.any():
        def to_seconds(val):
            parts = val.split(":")
            try:
                parts = [float(p) for p in parts]
            except ValueError:
                return np.nan
            if len(parts) == 3:
                h, m, sec = parts
                return h * 3600 + m * 60 + sec
            elif len(parts) == 2:
                m, sec = parts
                return m * 60 + sec
            return np.nan
        numeric.loc[needs_time_parse] = s.loc[needs_time_parse].apply(to_seconds)
    return numeric

def resolve_duration_column(df):
    candidates_in_order = [
        ("call_duration", "sec"),
        ("call_duration1", "sec"),
        ("CallDurationSecond", "sec"),
        ("Talkduration", "sec"),
        ("CallDurationMinute", "min"),
    ]
    best_col, best_seconds, best_score = None, None, -1
    for name, unit in candidates_in_order:
        col = find_column(df, [name])
        if not col:
            continue
        seconds = parse_duration_series_to_seconds(df[col])
        if unit == "min":
            seconds = seconds * 60
        non_null = seconds.notna().sum()
        non_zero = (seconds.fillna(0) > 0).sum()
        score = non_zero
        if non_null > 0 and score > best_score:
            best_col, best_seconds, best_score = col, seconds.fillna(0), score
    return best_col, best_seconds

def filter_out_vdcl_calls(df):
    if df is None or len(df) == 0:
        return df, 0
    agent_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'agent name' in col_lower or 'agentname' in col_lower or 'full_name' in col_lower or 'agent' in col_lower:
            agent_col = col
            break
    if agent_col is None:
        return df, 0
    agent_values = df[agent_col].fillna('').astype(str)
    mask = agent_values.str.contains('VDCL', case=False, na=False, regex=False)
    removed_count = int(mask.sum())
    filtered_df = df[~mask].copy()
    return filtered_df, removed_count

# ----------------------------------------------------------------------
# Get duration config from clients table
# ----------------------------------------------------------------------
def get_duration_config(client_name):
    base = {
        "short_max": 120,
        "medium_min": 120,
        "medium_max": 300,
        "large_min": 300,
        "extra_large_enabled": False,
        "extra_large_min": 480,
    }
    clients = st.session_state.get("clients", {})
    if client_name in clients:
        data = clients[client_name]
        if isinstance(data, dict):
            base = {
                "short_max": data.get("short_max", base["short_max"]),
                "medium_min": data.get("medium_min", base["medium_min"]),
                "medium_max": data.get("medium_max", base["medium_max"]),
                "large_min": data.get("large_min", base["large_min"]),
                "extra_large_enabled": data.get("extra_large_enabled", base["extra_large_enabled"]),
                "extra_large_min": data.get("extra_large_min", base["extra_large_min"]),
            }
    # IMPORTANT: session-only overrides set via the "✅ Apply" button in
    # Step 2 of the dashboard must win over the saved DB config for the
    # current browser session. Previously this dict was written to but
    # never read anywhere, so "Apply" appeared to work but silently had
    # zero effect on buckets, labels, agent analytics, or defaulter
    # detection. This merge is what makes Apply/Reset actually flexible.
    overrides = st.session_state.get("duration_overrides", {})
    if client_name in overrides:
        base.update(overrides[client_name])
    return base

# ----------------------------------------------------------------------
# Agent analytics
# ----------------------------------------------------------------------
def generate_agent_analytics(df, duration_col='_duration_sec', duration_config=None):
    if df is None or len(df) == 0:
        return None
    if duration_config is None:
        duration_config = get_duration_config(st.session_state.get("cdr_client", ""))
    agent_col = None
    for col in ['full_name', 'agent', 'Agent Name', 'AgentName', 'agent_name']:
        if col in df.columns:
            agent_col = col
            break
    if agent_col is None:
        return None
    df_copy = df.copy()
    if duration_col not in df_copy.columns:
        return None
    has_xl = duration_config.get('extra_large_enabled', False)
    def categorize_call(duration):
        if duration < duration_config['short_max']:
            return 'Short'
        elif duration <= duration_config['medium_max']:
            return 'Medium'
        elif has_xl and duration > duration_config['extra_large_min']:
            return 'Extra Large'
        else:
            return 'Large'
    df_copy['Call_Category'] = df_copy[duration_col].apply(categorize_call)
    agent_stats = df_copy.groupby(agent_col).agg({
        duration_col: ['count', 'mean', 'sum'],
        'Call_Category': lambda x: x.value_counts().to_dict()
    }).reset_index()
    agent_stats.columns = ['Agent', 'Total_Calls', 'Avg_Duration', 'Total_Duration', 'Category_Counts']
    def extract_category_counts(category_dict, category):
        return category_dict.get(category, 0)
    agent_stats['Short_Calls'] = agent_stats['Category_Counts'].apply(lambda x: extract_category_counts(x, 'Short'))
    agent_stats['Medium_Calls'] = agent_stats['Category_Counts'].apply(lambda x: extract_category_counts(x, 'Medium'))
    agent_stats['Large_Calls'] = agent_stats['Category_Counts'].apply(lambda x: extract_category_counts(x, 'Large'))
    if has_xl:
        agent_stats['Extra_Large_Calls'] = agent_stats['Category_Counts'].apply(
            lambda x: extract_category_counts(x, 'Extra Large')
        )
    agent_stats['Short_%'] = (agent_stats['Short_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    agent_stats['Medium_%'] = (agent_stats['Medium_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    agent_stats['Large_%'] = (agent_stats['Large_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    if has_xl:
        agent_stats['Extra_Large_%'] = (agent_stats['Extra_Large_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    agent_stats['Avg_Duration_Formatted'] = agent_stats['Avg_Duration'].apply(fmt_hms)
    agent_stats['Total_Duration_Formatted'] = agent_stats['Total_Duration'].apply(fmt_hms)
    agent_stats = agent_stats.drop('Category_Counts', axis=1)
    agent_stats = agent_stats.sort_values('Total_Calls', ascending=False)
    agent_stats['Rank'] = range(1, len(agent_stats) + 1)
    return agent_stats

# ----------------------------------------------------------------------
# VAD and Audio Processing Functions
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_silero_vad_model():
    """Load Silero VAD model using torch.hub with caching."""
    box = st.empty()
    render_buffer(box, "Loading voice-detection model (first run only)...")
    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    try:
        os.makedirs(hub_dir, exist_ok=True)
    except Exception:
        hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub")
        os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        get_speech_timestamps = utils[0]
        box.empty()
        return model, get_speech_timestamps
    except Exception as e:
        print(f"Error loading VAD from torch.hub: {e}")
        box.empty()
        return None, None

def compute_speech_energy_based(audio, sr, threshold=0.01, min_speech_duration=0.1):
    """Simple energy-based speech detection as fallback."""
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    window_size = int(sr * 0.025)
    hop_size = int(sr * 0.010)
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        energy.append(np.sqrt(np.mean(window**2)))
    energy = np.array(energy)
    is_speech = energy > threshold
    min_frames = int(min_speech_duration * sr / hop_size)
    speech_intervals = []
    start = None
    for i, speech in enumerate(is_speech):
        if speech and start is None:
            start = i * hop_size / sr
        elif not speech and start is not None:
            end = i * hop_size / sr
            if end - start >= min_speech_duration:
                speech_intervals.append((start, end))
            start = None
    if start is not None:
        end = len(audio) / sr
        if end - start >= min_speech_duration:
            speech_intervals.append((start, end))
    return speech_intervals

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val * 0.95
    return audio

def process_audio_file(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        data = normalize_audio(data)
        return data, sr
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None

def compute_vad_metrics(audio: np.ndarray, sr: int, threshold: float = 0.3, 
                        dead_air_secs: float = 5.0) -> Dict:
    total_duration = len(audio) / sr
    try:
        model, get_speech_timestamps = load_silero_vad_model()
        if model is not None and get_speech_timestamps is not None:
            audio_tensor = torch.from_numpy(audio).float()
            vad_kwargs = {
                'sampling_rate': sr,
                'threshold': threshold,
                'min_speech_duration_ms': 250,
                'min_silence_duration_ms': 200,
                'speech_pad_ms': 400,
                'window_size_samples': 512,
            }
            speech_timestamps = get_speech_timestamps(audio_tensor, model, **vad_kwargs)
            speech_intervals = []
            for ts in speech_timestamps:
                start = ts['start'] / sr
                end = ts['end'] / sr
                speech_intervals.append((start, end))
            return calculate_metrics_from_intervals(speech_intervals, total_duration, dead_air_secs)
    except Exception as e:
        print(f"Silero VAD failed, using fallback: {e}")
    speech_intervals = compute_speech_energy_based(audio, sr, threshold=0.02)
    return calculate_metrics_from_intervals(speech_intervals, total_duration, dead_air_secs)

def calculate_metrics_from_intervals(speech_intervals: List[Tuple[float, float]], 
                                    total_duration: float, 
                                    dead_air_secs: float) -> Dict:
    if not speech_intervals:
        return {
            "talk_time": 0.0,
            "silence_time": round(total_duration, 2),
            "dead_air": round(total_duration, 2) if total_duration > dead_air_secs else 0.0,
            "longest_silence": round(total_duration, 2),
            "duration": round(total_duration, 2),
            "speech_segments": []
        }
    speech_intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in speech_intervals:
        if not merged or start > merged[-1][1] + 0.1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    speech_time = 0.0
    longest_silence = 0.0
    dead_air = 0.0
    prev_end = 0.0
    for start, end in merged:
        speech_time += (end - start)
        silence = max(0.0, start - prev_end)
        longest_silence = max(longest_silence, silence)
        if silence > dead_air_secs:
            dead_air += silence
        prev_end = end
    ending_silence = max(0.0, total_duration - prev_end)
    longest_silence = max(longest_silence, ending_silence)
    if ending_silence > dead_air_secs:
        dead_air += ending_silence
    silence_time = max(0.0, total_duration - speech_time)
    return {
        "talk_time": round(speech_time, 2),
        "silence_time": round(silence_time, 2),
        "dead_air": round(dead_air, 2),
        "longest_silence": round(longest_silence, 2),
        "duration": round(total_duration, 2),
        "speech_segments": [(round(s, 2), round(e, 2)) for s, e in merged]
    }

def transcribe_audio_chunk(client: groq.Groq, audio: np.ndarray, sr: int) -> str:
    try:
        audio = normalize_audio(audio)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        data = buf.getvalue()
        buf = io.BytesIO(data)
        prompt = '''
This conversation is between an agent of a company called Weebo and one of their customers. Weebo is an internet service provider. The call will be about internet services, internet problems, billing, etc.

The call will be code mixed Hindi and english. Bahut sare words Hindi mein honge.

Some frequent words : namaste, hello, namaskar, sir, ma'am, complain, internet, raha, rahi, deta, deti
'''
        result = client.audio.transcriptions.create(
            file=('audio.wav', buf),
            model='whisper-large-v3-turbo',
            prompt=prompt,
            response_format='text',
            language='hi',
        )
        return str(result)
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def resolve_audio_url(recording_url):
    if not isinstance(recording_url, str) or not recording_url.strip():
        return None
    recording_url = recording_url.strip()
    if recording_url.lower().endswith((".mp3", ".wav", ".m4a", ".mp4")):
        return recording_url
    return html_recording_to_direct_url(recording_url)

def html_recording_to_direct_url(webform_url, retries=3):
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    audio_exts = (".mp3", ".wav", ".m4a", ".mp4")
    for attempt in range(retries):
        try:
            resp = session.get(webform_url, headers=headers, timeout=30, stream=True)
            resp.raise_for_status()
            content_type = resp.headers.get('content-type', '').lower()
            if 'audio' in content_type or 'video' in content_type:
                return resp.url
            if resp.url.lower().endswith(audio_exts):
                return resp.url
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(["audio", "video"]):
                src = tag.get("src")
                if src and any(ext in src.lower() for ext in audio_exts):
                    return urljoin(webform_url, src)
            for tag in soup.find_all("source"):
                src = tag.get("src")
                if src and any(ext in src.lower() for ext in audio_exts):
                    return urljoin(webform_url, src)
            for div in soup.find_all(attrs={"data-recording": True}):
                for attr in ["data-recording", "data-url", "data-src", "data-file"]:
                    url = div.get(attr)
                    if url:
                        return urljoin(webform_url, url)
            patterns = [
                r'https?://[^\s"\']+\.(?:mp3|wav|m4a)',
                r'//[^\s"\']+\.(?:mp3|wav|m4a)',
                r'/[^\s"\']+\.(?:mp3|wav|m4a)',
            ]
            for pattern in patterns:
                m = re.search(pattern, html, re.IGNORECASE)
                if m:
                    match = m.group()
                    if match.startswith("//"):
                        return "https:" + match
                    if match.startswith("/"):
                        return urljoin(webform_url, match)
                    return match
            js_patterns = [
                r'recordingUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'audioUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'fileUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'src\s*[:=]\s*["\']([^"\']+\.(?:mp3|wav|m4a))["\']',
            ]
            for pattern in js_patterns:
                m = re.search(pattern, html, re.IGNORECASE)
                if m:
                    return urljoin(webform_url, m.group(1))
            iframe = soup.find("iframe")
            if iframe and iframe.get("src"):
                iframe_src = urljoin(webform_url, iframe.get("src"))
                return html_recording_to_direct_url(iframe_src, retries=retries - 1)
            meta_refresh = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
            if meta_refresh and meta_refresh.get("content"):
                m = re.search(r"url=([^;]+)", meta_refresh.get("content"), re.IGNORECASE)
                if m:
                    return html_recording_to_direct_url(urljoin(webform_url, m.group(1)), retries=retries - 1)
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if any(ext in href.lower() for ext in audio_exts):
                    return urljoin(webform_url, href)
            return None
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1)
    return None

def process_recording_with_transcription(rec_url, vad_threshold, dead_air_secs, tmpdir, tag):
    metrics = {"talk_time": 0.0, "silence_time": 0.0, "dead_air": 0.0, 
               "longest_silence": 0.0, "duration": 0.0}
    debug_status = "OK"
    transcript = ""
    sentiment = "Neutral"
    actual_mp3 = None
    mp3_path = os.path.join(tmpdir, f"{tag}.mp3")
    wav_path = os.path.join(tmpdir, f"{tag}.wav")
    try:
        if not rec_url:
            return metrics, "No recording URL in this row", None, "", "Neutral"
        actual_mp3 = resolve_audio_url(rec_url)
        if not actual_mp3:
            return metrics, "Could not resolve audio URL", None, "", "Neutral"
        r = requests.get(actual_mp3, timeout=120, stream=True)
        if r.status_code != 200:
            return metrics, f"Download failed: HTTP {r.status_code}", actual_mp3, "", "Neutral"
        with open(mp3_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            return metrics, "Downloaded file is empty", actual_mp3, "", "Neutral"
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        ff = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "16000", 
             "-ac", "1", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if ff.returncode != 0 or not os.path.exists(wav_path):
            err_tail = ff.stderr.decode(errors="ignore")[-300:]
            return metrics, f"FFmpeg conversion failed: {err_tail.strip()}", actual_mp3, "", "Neutral"
        audio_data, sr = process_audio_file(wav_path)
        if audio_data is None:
            return metrics, "Failed to load audio", actual_mp3, "", "Neutral"
        vad_result = compute_vad_metrics(audio_data, sr, vad_threshold, dead_air_secs)
        metrics = vad_result
        if GROQ_API_KEY and vad_result.get("talk_time", 0) > 1.0:
            try:
                speech_segments = vad_result.get("speech_segments", [])
                if speech_segments:
                    client = groq.Groq(api_key=GROQ_API_KEY)
                    transcript_parts = []
                    for start, end in speech_segments[:5]:
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        segment = audio_data[start_sample:end_sample]
                        if len(segment) > sr * 0.5:
                            text = transcribe_audio_chunk(client, segment, sr)
                            if text:
                                transcript_parts.append(text)
                    transcript = "\n\n".join(transcript_parts)
                    if transcript:
                        sentiment = analyze_sentiment(transcript)
                else:
                    if len(audio_data) > sr * 1.0:
                        client = groq.Groq(api_key=GROQ_API_KEY)
                        transcript = transcribe_audio_chunk(client, audio_data, sr)
                        if transcript:
                            sentiment = analyze_sentiment(transcript)
                debug_status = "OK"
            except Exception as e:
                debug_status = f"Transcription error: {str(e)[:100]}"
        else:
            if not GROQ_API_KEY:
                debug_status = "No API Key"
            else:
                debug_status = "No speech detected"
        return metrics, debug_status, actual_mp3, transcript, sentiment
    except requests.exceptions.RequestException as e:
        return metrics, f"Network error: {str(e)[:100]}", actual_mp3, "", "Neutral"
    except Exception as e:
        return metrics, f"Processing error: {str(e)[:100]}", actual_mp3, "", "Neutral"
    finally:
        for path in [mp3_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

def analyze_sentiment(text: str) -> str:
    if not text or not text.strip():
        return "Neutral"
    if not GROQ_API_KEY:
        return "No API Key"
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the sentiment of the given call transcript as exactly one "
                        "word: Positive, Negative, or Neutral. Reply with only that one word, "
                        "nothing else."
                    ),
                },
                {"role": "user", "content": text[:3000]},
            ],
            temperature=0,
            max_tokens=5,
        )
        raw = resp.choices[0].message.content.strip()
        label = raw.split()[0].strip(".,!").capitalize() if raw else ""
        if label in ("Positive", "Negative", "Neutral"):
            return label
        return "Neutral"
    except Exception:
        return "Neutral"

def process_recording_metrics_only(rec_url, vad_threshold, dead_air_secs, tmpdir, tag):
    metrics = {"talk_time": 0.0, "silence_time": 0.0, "dead_air": 0.0, 
               "longest_silence": 0.0, "duration": 0.0}
    debug_status = "OK"
    actual_mp3 = None
    mp3_path = os.path.join(tmpdir, f"{tag}.mp3")
    wav_path = os.path.join(tmpdir, f"{tag}.wav")
    try:
        if not rec_url:
            return metrics, "No recording URL in this row", None
        actual_mp3 = resolve_audio_url(rec_url)
        if not actual_mp3:
            return metrics, "Could not resolve audio URL", None
        r = requests.get(actual_mp3, timeout=120, stream=True)
        if r.status_code != 200:
            return metrics, f"Download failed: HTTP {r.status_code}", actual_mp3
        with open(mp3_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            return metrics, "Downloaded file is empty", actual_mp3
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        ff = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "16000", 
             "-ac", "1", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if ff.returncode != 0 or not os.path.exists(wav_path):
            err_tail = ff.stderr.decode(errors="ignore")[-300:]
            return metrics, f"FFmpeg conversion failed: {err_tail.strip()}", actual_mp3
        audio_data, sr = process_audio_file(wav_path)
        if audio_data is None:
            return metrics, "Failed to load audio", actual_mp3
        metrics = compute_vad_metrics(audio_data, sr, vad_threshold, dead_air_secs)
        return metrics, "OK", actual_mp3
    except requests.exceptions.RequestException as e:
        return metrics, f"Network error: {str(e)[:100]}", actual_mp3
    except Exception as e:
        return metrics, f"Processing error: {str(e)[:100]}", actual_mp3
    finally:
        for path in [mp3_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

def run_complete_call_analysis(all_calls_df, col_recording, col_agent, col_date, col_time, col_phone,
                                vad_threshold, dead_air_secs, duration_config,
                                silence_threshold=30, min_calls=3):
    progress = st.progress(0)
    buffer_box = st.empty()
    render_buffer(buffer_box, "Analyzing calls...")
    total = len(all_calls_df)
    silence_rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (_, row) in enumerate(all_calls_df.iterrows()):
            metrics, debug_status, actual_mp3 = process_recording_metrics_only(
                row.get(col_recording), vad_threshold, dead_air_secs, tmpdir, f"complete_{i}",
            )
            dur = metrics.get("duration")
            sil = metrics.get("silence_time")
            silence_pct = round((sil / dur * 100), 1) if dur and dur > 0 and sil is not None else None
            date_val = row.get(col_date) if col_date else None
            time_val = row.get(col_time) if col_time else None
            if date_val and pd.notna(date_val):
                try:
                    date_val = pd.to_datetime(date_val).strftime("%d/%m/%Y")
                except Exception:
                    pass
            silence_rows.append({
                "Date": date_val,
                "Time": time_val,
                "Agent Name": row.get(col_agent) if col_agent else None,
                "Call From": row.get(col_phone) if col_phone else None,
                "Duration (sec)": dur,
                "Silence Time (sec)": sil,
                "Silence %": silence_pct,
                "Longest Silence (sec)": metrics.get("longest_silence"),
                "Duration_CRM": row.get("_duration_sec"),
                "Status": debug_status,
                "Actual MP3": actual_mp3,
            })
            progress.progress((i + 1) / total)
    buffer_box.empty()
    progress.empty()
    silence_df = pd.DataFrame(silence_rows)
    perf_df = analyze_agent_silence_by_category(silence_df, duration_config)
    defaulters_df = detect_defaulters_simple(silence_df, duration_config, silence_threshold, min_calls)
    return silence_df, perf_df, defaulters_df

def detect_defaulters_simple(silence_df, duration_config, silence_threshold=30, min_calls=3):
    if silence_df is None or len(silence_df) == 0:
        return None
    short_max = duration_config.get('short_max', 120)
    medium_max = duration_config.get('medium_max', 300)
    def categorize_by_duration(duration):
        if duration is None or pd.isna(duration):
            return 'Unknown'
        if duration < short_max:
            return 'Short'
        elif duration <= medium_max:
            return 'Medium'
        else:
            return 'Large'
    silence_df_copy = silence_df.copy()
    silence_df_copy['Duration (sec)'] = pd.to_numeric(silence_df_copy['Duration (sec)'], errors='coerce')
    silence_df_copy['Category'] = silence_df_copy['Duration (sec)'].apply(categorize_by_duration)
    silence_df_copy = silence_df_copy[silence_df_copy['Category'] != 'Unknown']
    if len(silence_df_copy) == 0:
        return None
    agent_data = []
    for agent in silence_df_copy['Agent Name'].unique():
        agent_calls = silence_df_copy[silence_df_copy['Agent Name'] == agent]
        total_calls = len(agent_calls)
        if total_calls < min_calls:
            continue
        overall_avg_silence = agent_calls['Silence %'].mean()
        short_calls = agent_calls[agent_calls['Category'] == 'Short']
        medium_calls = agent_calls[agent_calls['Category'] == 'Medium']
        large_calls = agent_calls[agent_calls['Category'] == 'Large']
        short_avg_silence = short_calls['Silence %'].mean() if len(short_calls) > 0 else 0
        medium_avg_silence = medium_calls['Silence %'].mean() if len(medium_calls) > 0 else 0
        large_avg_silence = large_calls['Silence %'].mean() if len(large_calls) > 0 else 0
        is_defaulter = (overall_avg_silence > silence_threshold) or (short_avg_silence > silence_threshold)
        if is_defaulter:
            agent_data.append({
                'Agent': agent,
                'Total_Calls': total_calls,
                'Short_Calls': len(short_calls),
                'Short_Silence_%': round(short_avg_silence, 1),
                'Medium_Calls': len(medium_calls),
                'Medium_Silence_%': round(medium_avg_silence, 1),
                'Large_Calls': len(large_calls),
                'Large_Silence_%': round(large_avg_silence, 1),
                'Overall_Silence_%': round(overall_avg_silence, 1),
            })
    if len(agent_data) == 0:
        return None
    df_defaulters = pd.DataFrame(agent_data)
    df_defaulters = df_defaulters.sort_values('Overall_Silence_%', ascending=False)
    df_defaulters['Rank'] = range(1, len(df_defaulters) + 1)
    return df_defaulters

def analyze_agent_silence_by_category(silence_df, duration_config):
    if silence_df is None or len(silence_df) == 0:
        return None
    short_max = duration_config.get('short_max', 120)
    medium_max = duration_config.get('medium_max', 300)
    def categorize_by_duration(duration):
        if duration is None or pd.isna(duration):
            return 'Unknown'
        if duration < short_max:
            return 'Short'
        elif duration <= medium_max:
            return 'Medium'
        else:
            return 'Large'
    silence_df_copy = silence_df.copy()
    silence_df_copy['Duration (sec)'] = pd.to_numeric(silence_df_copy['Duration (sec)'], errors='coerce')
    silence_df_copy['Category'] = silence_df_copy['Duration (sec)'].apply(categorize_by_duration)
    agent_category_silence = silence_df_copy.groupby(['Agent Name', 'Category']).agg({
        'Silence %': ['mean', 'count'],
    }).reset_index()
    agent_category_silence.columns = ['Agent', 'Category', 'Avg_Silence_%', 'Call_Count']
    pivot_df = agent_category_silence.pivot_table(
        index='Agent', columns='Category', values=['Avg_Silence_%', 'Call_Count'], fill_value=0
    ).reset_index()
    pivot_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_df.columns.values]
    pivot_df = pivot_df.rename(columns={'Agent_': 'Agent'})
    for cat in ['Short', 'Medium', 'Large']:
        if f'Avg_Silence_%_{cat}' not in pivot_df.columns:
            pivot_df[f'Avg_Silence_%_{cat}'] = 0
        if f'Call_Count_{cat}' not in pivot_df.columns:
            pivot_df[f'Call_Count_{cat}'] = 0
    agent_overall = silence_df_copy.groupby('Agent Name').agg({
        'Silence %': 'mean', 'Silence Time (sec)': 'sum', 'Duration (sec)': 'sum'
    }).reset_index()
    agent_overall.columns = ['Agent', 'Overall_Avg_Silence_%', 'Total_Silence_Time', 'Total_Duration']
    pivot_df = pivot_df.merge(agent_overall, on='Agent', how='left')
    pivot_df['Total_Calls'] = pivot_df['Call_Count_Short'] + pivot_df['Call_Count_Medium'] + pivot_df['Call_Count_Large']
    pivot_df = pivot_df.sort_values('Overall_Avg_Silence_%', ascending=False).reset_index(drop=True)
    pivot_df['Rank'] = range(1, len(pivot_df) + 1)
    return pivot_df

def categorize_duration_4(duration, short_max, medium_max, large_threshold_sec):
    if duration is None or pd.isna(duration):
        return "Unknown"
    if duration < short_max:
        return "Short"
    if duration <= medium_max:
        return "Medium"
    if duration <= large_threshold_sec:
        return "Long"
    return "Large"

def build_agent_performance_table(silence_df, short_max, medium_max, large_threshold_sec):
    if silence_df is None or len(silence_df) == 0:
        return None
    df = silence_df.copy()
    df["Duration (sec)"] = pd.to_numeric(df["Duration (sec)"], errors="coerce")
    df["Category"] = df["Duration (sec)"].apply(
        lambda d: categorize_duration_4(d, short_max, medium_max, large_threshold_sec)
    )
    df = df[df["Category"] != "Unknown"]
    if len(df) == 0:
        return None
    rows = []
    for agent, g in df.groupby("Agent Name"):
        row = {"Agent": agent, "Total_Calls": len(g), "Overall_Silence": round(g["Silence %"].mean(), 1)}
        for cat in ["Short", "Medium", "Long", "Large"]:
            sub = g[g["Category"] == cat]
            row[f"{cat}_Calls"] = len(sub)
            row[f"{cat}_Silence"] = round(sub["Silence %"].mean(), 1) if len(sub) else 0.0
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("Overall_Silence", ascending=False).reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    return out

def default_large_threshold_min(duration_config):
    return round(duration_config["medium_max"] / 60 + 3, 1)

# ----------------------------------------------------------------------
# Email functions
# ----------------------------------------------------------------------
def get_smtp_config():
    return st.session_state.get("smtp_config", {})

def test_smtp_connection(smtp_config=None):
    if smtp_config is None:
        smtp_config = get_smtp_config()
    try:
        if not smtp_config.get("password"):
            return False, "SMTP password not configured."
        host = smtp_config.get("host", "mail.dialdesk.net")
        port = int(smtp_config.get("port", 587))
        username = smtp_config.get("username", "")
        password = smtp_config.get("password", "")
        use_tls = smtp_config.get("use_tls", True)
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()
            server.login(username, password)
        return True, f"✅ SMTP connection successful to {host}:{port}"
    except smtplib.SMTPAuthenticationError as e:
        return False, f"❌ Authentication failed: {str(e)}"
    except Exception as e:
        return False, f"❌ SMTP Error: {str(e)}"

def _send_mime(msg, mentor_emails, smtp_config=None):
    if smtp_config is None:
        smtp_config = get_smtp_config()
    host = smtp_config.get("host", "mail.dialdesk.net")
    port = int(smtp_config.get("port", 587))
    username = smtp_config.get("username", "")
    password = smtp_config.get("password", "")
    use_tls = smtp_config.get("use_tls", True)
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo()
        if use_tls:
            server.starttls()
            server.ehlo()
        server.login(username, password)
        server.sendmail(msg["From"], mentor_emails, msg.as_string())

def send_performance_report_email(perf_table, silence_df, client_name, mentor_emails, large_threshold_min, smtp_config=None):
    if smtp_config is None:
        smtp_config = get_smtp_config()
    if perf_table is None or len(perf_table) == 0:
        return False, "No performance data available"
    mentor_emails = [e.strip() for e in mentor_emails if e and '@' in e]
    if not mentor_emails:
        return False, "No valid mentor emails found!"
    smtp_ok, smtp_msg = test_smtp_connection(smtp_config)
    if not smtp_ok:
        return False, f"SMTP Error: {smtp_msg}"
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_config.get("from_email") or smtp_config["username"]
        msg["To"] = ", ".join(mentor_emails)
        msg["Subject"] = f"📊 Agent Performance Report — {client_name} ({date.today().strftime('%d/%m/%Y')})"
        agent_rows = ""
        for _, row in perf_table.iterrows():
            overall = row.get('Overall_Silence', 0)
            color = '#DC2626' if overall > 40 else ('#D97706' if overall > 30 else '#16A34A')
            agent_rows += f"""
            <tr>
                <td style="padding:10px;font-weight:bold;">{row.get('Agent', 'Unknown')}</td>
                <td style="padding:10px;text-align:center;">{row.get('Total_Calls', 0)}</td>
                <td style="padding:10px;text-align:center;">{row.get('Short_Calls', 0)}<br><small style="color:#6B7280;">({row.get('Short_Silence', 0)}%)</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Medium_Calls', 0)}<br><small style="color:#6B7280;">({row.get('Medium_Silence', 0)}%)</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Long_Calls', 0)}<br><small style="color:#6B7280;">({row.get('Long_Silence', 0)}%)</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Large_Calls', 0)}<br><small style="color:#6B7280;">({row.get('Large_Silence', 0)}%)</small></td>
                <td style="padding:10px;text-align:center;font-weight:bold;color:{color};">{overall}%</td>
            </tr>
            """
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: #4F46E5; padding: 20px; border-radius: 8px; color: white; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #1F2937; color: white; padding: 10px; text-align: center; }}
                td {{ padding: 8px; text-align: center; border-bottom: 1px solid #E5E7EB; }}
                .footer {{ margin-top: 20px; color: #6B7280; font-size: 12px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📊 Agent Performance Report</h2>
                <p><strong>Client:</strong> {client_name} | <strong>Date:</strong> {date.today().strftime('%d/%m/%Y')}</p>
            </div>

            <h3>📋 Agent Performance Table</h3>
            <p style="color:#6B7280;font-size:13px;">
                "Large" = calls longer than {large_threshold_min:.1f} min · counts shown with average silence % in brackets.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Total Calls</th>
                        <th>Short<br><small>(Silence %)</small></th>
                        <th>Medium<br><small>(Silence %)</small></th>
                        <th>Long<br><small>(Silence %)</small></th>
                        <th>Large<br><small>(Silence %)</small></th>
                        <th>Overall %</th>
                    </tr>
                </thead>
                <tbody>
                    {agent_rows}
                </tbody>
            </table>

            <div class="footer">
                <p>— CallAI Analytics | Automated Performance Report</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                perf_table.to_excel(writer, index=False, sheet_name="Performance Report")
                if silence_df is not None and len(silence_df) > 0:
                    silence_df.to_excel(writer, index=False, sheet_name="All Calls Data")
            buf.seek(0)
            excel_attachment = MIMEApplication(buf.read(), _subtype="xlsx")
            excel_attachment.add_header(
                'Content-Disposition', 'attachment',
                filename=f'Performance_Report_{client_name}_{date.today().strftime("%Y%m%d")}.xlsx'
            )
            msg.attach(excel_attachment)
        except Exception as e:
            print(f"Excel attachment error: {e}")

        try:
            _send_mime(msg, mentor_emails, smtp_config)
            return True, f"✅ Performance report sent to {len(mentor_emails)} mentors!"
        except Exception as e:
            return False, f"❌ Failed to send email: {str(e)}"

    except Exception as e:
        return False, f"❌ Error preparing email: {str(e)}"

def send_defaulter_alert_email(defaulters_df, silence_df, client_name, mentor_emails, smtp_config=None):
    if smtp_config is None:
        smtp_config = get_smtp_config()
    if defaulters_df is None or len(defaulters_df) == 0:
        return False, "No defaulters to report"
    mentor_emails = [e.strip() for e in mentor_emails if e and '@' in e]
    if not mentor_emails:
        return False, "No valid mentor emails found!"
    smtp_ok, smtp_msg = test_smtp_connection(smtp_config)
    if not smtp_ok:
        return False, f"SMTP Error: {smtp_msg}"
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_config.get("from_email") or smtp_config["username"]
        msg["To"] = ", ".join(mentor_emails)
        msg["Subject"] = f"🚨 Defaulter Alert — {len(defaulters_df)} agent(s) flagged ({client_name})"
        agent_rows = ""
        for _, row in defaulters_df.iterrows():
            overall = row.get('Overall_Silence_%', 0)
            if overall > 40:
                color, badge = '#DC2626', '🔴 HIGH'
            elif overall > 30:
                color, badge = '#D97706', '🟠 MEDIUM'
            else:
                color, badge = '#2563EB', '🔵 LOW'
            agent_rows += f"""
            <tr>
                <td style="padding:10px;font-weight:bold;color:{color};">{row.get('Agent', 'Unknown')}</td>
                <td style="padding:10px;text-align:center;">{row.get('Total_Calls', 0)}</td>
                <td style="padding:10px;text-align:center;">{row.get('Short_Calls', 0)}<br><small>{row.get('Short_Silence_%', 0)}%</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Medium_Calls', 0)}<br><small>{row.get('Medium_Silence_%', 0)}%</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Large_Calls', 0)}<br><small>{row.get('Large_Silence_%', 0)}%</small></td>
                <td style="padding:10px;text-align:center;font-weight:bold;color:{color};">{overall}%</td>
                <td style="padding:10px;text-align:center;"><span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:12px;">{badge}</span></td>
            </tr>
            """
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: #FEE2E2; padding: 20px; border-radius: 8px; border-left: 6px solid #DC2626; margin-bottom: 20px; }}
                .summary {{ background: #F3F4F6; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #1F2937; color: white; padding: 10px; text-align: center; }}
                td {{ padding: 8px; text-align: center; border-bottom: 1px solid #E5E7EB; }}
                .footer {{ margin-top: 20px; color: #6B7280; font-size: 12px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 Defaulter Agents Detected</h2>
                <p><strong>Client:</strong> {client_name} | <strong>Date:</strong> {date.today().strftime('%d/%m/%Y')}</p>
                <p><strong>Total Defaulters:</strong> {len(defaulters_df)}</p>
            </div>
            <div class="summary">
                <p>Agents with <strong>Overall Silence &gt; 30%</strong> OR <strong>Short Call Silence &gt; 30%</strong> are flagged.</p>
                <p><span style="color:#DC2626;">🔴 High</span> = &gt;40% |
                   <span style="color:#D97706;">🟠 Medium</span> = 30-40% |
                   <span style="color:#2563EB;">🔵 Low</span> = &lt;30%</p>
            </div>
            <h3>📊 Defaulter Agents</h3>
            <table>
                <thead>
                    <tr>
                        <th>Agent</th><th>Total Calls</th>
                        <th>Short<br><small style="font-weight:normal;">(Silence)</small></th>
                        <th>Medium<br><small style="font-weight:normal;">(Silence)</small></th>
                        <th>Large<br><small style="font-weight:normal;">(Silence)</small></th>
                        <th>Overall %</th><th>Severity</th>
                    </tr>
                </thead>
                <tbody>{agent_rows}</tbody>
            </table>
            <div class="footer">
                <p>— CallAI Analytics | Automated Defaulter Detection</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                defaulters_df.to_excel(writer, index=False, sheet_name="Defaulters")
                if silence_df is not None and len(silence_df) > 0:
                    silence_df.to_excel(writer, index=False, sheet_name="All Calls Data")
            buf.seek(0)
            excel_attachment = MIMEApplication(buf.read(), _subtype="xlsx")
            excel_attachment.add_header(
                'Content-Disposition', 'attachment',
                filename=f'Defaulter_Alert_{client_name}_{date.today().strftime("%Y%m%d")}.xlsx'
            )
            msg.attach(excel_attachment)
        except Exception as e:
            print(f"Excel attachment error: {e}")

        try:
            _send_mime(msg, mentor_emails, smtp_config)
            return True, f"✅ Defaulter alert sent to {len(mentor_emails)} mentors!"
        except Exception as e:
            return False, f"❌ Failed to send email: {str(e)}"

    except Exception as e:
        return False, f"❌ Error preparing email: {str(e)}"

# ----------------------------------------------------------------------
# Scheduler functions
# ----------------------------------------------------------------------
def run_scheduled_job():
    """Run the scheduled job for all auto-schedule clients."""
    try:
        smtp_config = get_smtp_config()
        do_login()
        yesterday = date.today() - timedelta(days=1)
        auto_clients = st.session_state.get("auto_schedule_clients", ["Weebo", "Hari Om Pvt Ltd"])
        mentor_emails = st.session_state.get("mentor_emails", [])
        clients = st.session_state.get("clients", {})

        for client_name in auto_clients:
            if client_name not in clients:
                continue
            client_data = clients[client_name]
            company_id = client_data["company_id"] if isinstance(client_data, dict) else client_data

            payload = {
                "from_date": yesterday.strftime("%Y-%m-%d"),
                "to_date": yesterday.strftime("%Y-%m-%d"),
                "company_id": str(company_id),
            }

            resp = fetch_cdr(payload)
            if resp.status_code != 200:
                continue

            data = resp.json()
            records = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(records, dict):
                for v in records.values():
                    if isinstance(v, list):
                        records = v
                        break
            cdr_df = pd.DataFrame(records)
            cdr_df, _ = filter_out_vdcl_calls(cdr_df)
            _, duration_seconds = resolve_duration_column(cdr_df)
            cdr_df["_duration_sec"] = duration_seconds

            if len(cdr_df) == 0:
                continue

            col_recording = find_column(cdr_df, COLUMN_CANDIDATES["recording"])
            col_agent = find_column(cdr_df, COLUMN_CANDIDATES["agent_name"])
            col_date = find_column(cdr_df, COLUMN_CANDIDATES["date"])
            col_time = find_column(cdr_df, COLUMN_CANDIDATES["time"])
            col_phone = find_column(cdr_df, COLUMN_CANDIDATES["call_from"])

            if not col_recording:
                continue

            duration_config = get_duration_config(client_name)
            silence_threshold = st.session_state.get("silence_threshold", 30)
            min_calls = st.session_state.get("min_calls_per_agent", 3)
            
            silence_df, _, defaulters_df = run_complete_call_analysis(
                cdr_df, col_recording, col_agent, col_date, col_time, col_phone,
                vad_threshold=0.3, dead_air_secs=5, duration_config=duration_config,
                silence_threshold=silence_threshold, min_calls=min_calls
            )

            large_threshold_sec = default_large_threshold_min(duration_config) * 60
            perf_table = build_agent_performance_table(
                silence_df, duration_config["short_max"], duration_config["medium_max"], large_threshold_sec
            )

            if perf_table is not None and len(perf_table) > 0:
                send_performance_report_email(
                    perf_table, silence_df, client_name, mentor_emails,
                    default_large_threshold_min(duration_config), smtp_config
                )

            if defaulters_df is not None and len(defaulters_df) > 0:
                send_defaulter_alert_email(defaulters_df, silence_df, client_name, mentor_emails, smtp_config)

        st.session_state["last_email_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        print(f"[{datetime.now()}] Scheduled job error: {str(e)}")

def start_scheduler():
    """Start the scheduler with configured time."""
    if st.session_state.get("scheduler_running", False):
        return
    
    scheduler_time = st.session_state.get("scheduler_time", "23:00")
    scheduler_enabled = st.session_state.get("scheduler_enabled", True)
    
    if not scheduler_enabled:
        return

    def scheduler_loop():
        schedule.clear()
        schedule.every().day.at(scheduler_time).do(run_scheduled_job)
        while True:
            schedule.run_pending()
            time.sleep(60)

    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    st.session_state["scheduler_running"] = True

def restart_scheduler():
    st.session_state["scheduler_running"] = False
    time.sleep(1)
    start_scheduler()

# ----------------------------------------------------------------------
# Dashboard / Call Analytics Section
# ----------------------------------------------------------------------
def render_dashboard():
    """Render the main dashboard with call analytics."""
    
    # Check if we have data
    have_data = (
        st.session_state["cdr_df"] is not None
        and len(st.session_state["cdr_df"]) > 0
    )

    # Step 1: Client + date range
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">1</span>Choose Client & Date Range</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Only calls belonging to the selected client will be fetched.</p>', unsafe_allow_html=True)

    clients = st.session_state.get("clients", {})
    client_names = list(clients.keys()) if clients else []

    if not client_names:
        st.warning("No clients configured. Please add clients in the Settings tab.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        client_name = st.selectbox("Client", options=sorted(client_names))
        client_data = clients.get(client_name, {})
        company_id = client_data["company_id"] if isinstance(client_data, dict) else client_data
    with c2:
        dc1, dc2 = st.columns(2)
        with dc1:
            from_date = st.date_input("Start Date", value=date.today() - timedelta(days=1), max_value=date.today())
        with dc2:
            to_date = st.date_input("End Date", value=date.today(), max_value=date.today())
        if from_date > to_date:
            st.error("End Date must be on or after Start Date.")

    if st.session_state.get("cdr_client") is not None and st.session_state["cdr_client"] != client_name:
        st.session_state["cdr_df"] = None
        st.session_state["cdr_client"] = None
        st.session_state["final_df"] = None
        st.session_state["agent_analytics_df"] = None
        st.session_state["complete_analysis_df"] = None
        st.session_state["agent_silence_by_category_df"] = None
        st.session_state["defaulter_agents_df"] = None

    fetch_clicked = st.button("📥  Fetch Calls", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if fetch_clicked:
        if from_date > to_date:
            st.error("Please correct the date range before fetching.")
        else:
            try:
                payload = {
                    "from_date": from_date.strftime("%Y-%m-%d"),
                    "to_date": to_date.strftime("%Y-%m-%d"),
                    "company_id": str(company_id),
                }
                with st.spinner(f"Fetching calls for {client_name}..."):
                    resp = fetch_cdr(payload)
                if resp.status_code == 200:
                    data = resp.json()
                    records = data.get("data", data) if isinstance(data, dict) else data
                    if isinstance(records, dict):
                        for v in records.values():
                            if isinstance(v, list):
                                records = v
                                break
                    cdr_df = pd.DataFrame(records)

                    cdr_df, removed_count = filter_out_vdcl_calls(cdr_df)
                    st.session_state["vdcl_removed"] = removed_count

                    dur_source_col, duration_seconds = resolve_duration_column(cdr_df)
                    cdr_df["_duration_sec"] = duration_seconds
                    st.session_state["cdr_df"] = cdr_df
                    st.session_state["cdr_client"] = client_name
                    st.session_state["final_df"] = None
                    st.session_state["agent_analytics_df"] = None
                    st.session_state["complete_analysis_df"] = None
                    st.session_state["agent_silence_by_category_df"] = None
                    st.session_state["defaulter_agents_df"] = None

                    if len(cdr_df) == 0:
                        st.warning(f"No valid calls found for **{client_name}** in this date range.")
                    else:
                        st.markdown(
                            f'<span class="status-banner-ok">✅ Fetched {len(cdr_df)} valid calls for {client_name}</span>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.error(f"Fetch failed: HTTP {resp.status_code} — {resp.text[:300]}")
            except Exception as e:
                st.error(f"Fetch error: {e}")

    # ---- Step 2 onwards: only once we have data ----
    have_data = (
        st.session_state["cdr_df"] is not None
        and len(st.session_state["cdr_df"]) > 0
    )

    if not have_data:
        st.info("👆 Pick a client and date range above, then click **Fetch Calls** to get started.")
        return

    cdr_df = st.session_state["cdr_df"].copy()
    col_date = find_column(cdr_df, COLUMN_CANDIDATES["date"])
    col_time = find_column(cdr_df, COLUMN_CANDIDATES["time"])
    col_agent = find_column(cdr_df, COLUMN_CANDIDATES["agent_name"])
    col_phone = find_column(cdr_df, COLUMN_CANDIDATES["call_from"])
    col_recording = find_column(cdr_df, COLUMN_CANDIDATES["recording"])
    dur_source_col, duration_seconds = resolve_duration_column(cdr_df)
    cdr_df, _ = filter_out_vdcl_calls(cdr_df)
    if dur_source_col is None:
        st.error("Could not find a usable call-duration column.")
        st.stop()
    cdr_df["_duration_sec"] = duration_seconds

    # ---- Step 2: filter & display ----
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">2</span>Pick Call Type & Count</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Choose which calls you want in the report.</p>', unsafe_allow_html=True)
    duration_config = get_duration_config(client_name)

    with st.expander(f"📊 Duration Configuration for {client_name}", expanded=False):
        has_xl = duration_config.get('extra_large_enabled', False)
        large_line = (
            f"• Large calls: {fmt_hms(duration_config['large_min'])} – {fmt_hms(duration_config['extra_large_min'])}<br>"
            f"• Extra Large calls: &gt; {fmt_hms(duration_config['extra_large_min'])}<br>"
            if has_xl else
            f"• Large calls: &gt; {fmt_hms(duration_config['large_min'])}<br>"
        )
        st.markdown(f"""
        <div class="config-box">
            <strong>Current Duration Settings:</strong><br>
            • Short calls: &lt; {fmt_hms(duration_config['short_max'])}<br>
            • Medium calls: {fmt_hms(duration_config['medium_min'])} – {fmt_hms(duration_config['medium_max'])}<br>
            {large_line}
        </div>
        """, unsafe_allow_html=True)

        st.caption("Change the cutoffs below for this client (this browser session only), then click Apply.")
        oc1, oc2, oc3 = st.columns([1, 1, 1])
        with oc1:
            short_cutoff_min = st.number_input(
                "Short ends at (minutes)", min_value=0.5, max_value=15.0,
                value=round(duration_config['short_max'] / 60, 2), step=0.5,
                key=vkey(f"short_cutoff_{client_name}"),
            )
        with oc2:
            large_cutoff_min = st.number_input(
                "Large starts after (minutes)", min_value=short_cutoff_min, max_value=30.0,
                value=max(round(duration_config['large_min'] / 60, 2), short_cutoff_min), step=0.5,
                key=vkey(f"large_cutoff_{client_name}"),
            )
        with oc3:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                apply_clicked = st.button("✅ Apply", key="apply_overrides", use_container_width=True)
            with bcol2:
                if st.button("↩️ Reset", key="reset_overrides", use_container_width=True):
                    st.session_state["duration_overrides"].pop(client_name, None)
                    bump_ui_version()
                    st.success("↩️ Reset to default.")
                    st.rerun()

        st.markdown("---")
        enable_xl = st.checkbox(
            "➕ Also split out an 'Extra Large' bucket (e.g. calls over 8 min)",
            value=duration_config.get('extra_large_enabled', False), key=vkey(f"enable_xl_{client_name}"),
        )
        xl_cutoff_min = None
        if enable_xl:
            default_xl = duration_config.get('extra_large_min', duration_config['large_min'] + 180)
            xl_cutoff_min = st.number_input(
                "Extra Large starts after (minutes)", min_value=large_cutoff_min, max_value=60.0,
                value=max(round(default_xl / 60, 2), large_cutoff_min + 0.5), step=0.5,
                key=vkey(f"xl_cutoff_{client_name}"),
            )
        if apply_clicked:
            override = {
                "short_max": int(round(short_cutoff_min * 60)),
                "medium_min": int(round(short_cutoff_min * 60)),
                "medium_max": int(round(large_cutoff_min * 60)),
                "large_min": int(round(large_cutoff_min * 60)),
                "extra_large_enabled": enable_xl,
            }
            if enable_xl and xl_cutoff_min:
                override["extra_large_min"] = int(round(xl_cutoff_min * 60))
            st.session_state["duration_overrides"][client_name] = override
            bump_ui_version()
            st.success("✅ Applied for this session!")
            st.rerun()

    st.markdown("#### 🔍 Search")
    scol1, scol2 = st.columns([1.6, 1.2])
    with scol1:
        agent_options = sorted(cdr_df[col_agent].dropna().astype(str).unique().tolist()) if col_agent else []
        selected_agents = st.multiselect("Agent Name (leave empty for all agents)", options=agent_options, default=[])
    with scol2:
        phone_search = st.text_input("Phone Number contains", value="", placeholder="e.g. 98765")

    if selected_agents and col_agent:
        cdr_df = cdr_df[cdr_df[col_agent].astype(str).isin(selected_agents)]
    if phone_search.strip() and col_phone:
        cdr_df = cdr_df[cdr_df[col_phone].astype(str).str.contains(phone_search.strip(), case=False, na=False)]

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(f'<div class="metric-pill"><div class="value">{len(cdr_df)}</div><div class="label">Total valid calls</div></div>', unsafe_allow_html=True)
    with p2:
        avg_dur = cdr_df["_duration_sec"].mean() if len(cdr_df) else 0
        st.markdown(f'<div class="metric-pill"><div class="value">{fmt_hms(avg_dur)}</div><div class="label">Average call duration</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="metric-pill"><div class="value">{client_name}</div><div class="label">Client</div></div>', unsafe_allow_html=True)
    with p4:
        st.markdown(f'<div class="metric-pill"><div class="value">{st.session_state.get("vdcl_removed", 0)}</div><div class="label">Abandoned (VDCL) removed</div></div>', unsafe_allow_html=True)

    bcol, ccol = st.columns([2, 1.2])
    with bcol:
        has_xl = duration_config.get('extra_large_enabled', False)
        short_label = f"Short (< {fmt_hms(duration_config['short_max'])})"
        medium_label = f"Medium ({fmt_hms(duration_config['medium_min'])} – {fmt_hms(duration_config['medium_max'])})"
        if has_xl:
            large_label = f"Large ({fmt_hms(duration_config['large_min'])} – {fmt_hms(duration_config['extra_large_min'])})"
            extra_large_label = f"Extra Large (> {fmt_hms(duration_config['extra_large_min'])})"
        else:
            large_label = f"Large (> {fmt_hms(duration_config['large_min'])})"
            extra_large_label = None
        bucket_options = ["All calls", short_label, medium_label, large_label]
        if has_xl:
            bucket_options.append(extra_large_label)
        bucket_options.append("Custom Filter")
        bucket = st.radio("Call type", bucket_options, horizontal=True)

        custom_filter_expr = ""
        if bucket == "Custom Filter":
            st.caption("📝 Pick calls by duration — no code needed.")
            filter_kind = st.radio(
                "Show calls where duration is...",
                ["Less than", "Greater than", "Between", "Exactly 0 (zero-duration)"],
                horizontal=True, key="custom_filter_kind",
            )
            if filter_kind == "Less than":
                cf_max = st.number_input("...this many minutes", min_value=0.0, value=2.0, step=0.5, key="cf_lt")
                custom_filter_expr = f"duration < {cf_max * 60}"
            elif filter_kind == "Greater than":
                cf_min = st.number_input("...this many minutes", min_value=0.0, value=8.0, step=0.5, key="cf_gt")
                custom_filter_expr = f"duration > {cf_min * 60}"
            elif filter_kind == "Between":
                cfb1, cfb2 = st.columns(2)
                with cfb1:
                    cf_from = st.number_input("From (minutes)", min_value=0.0, value=5.0, step=0.5, key="cf_from")
                with cfb2:
                    cf_to = st.number_input("To (minutes)", min_value=cf_from, value=10.0, step=0.5, key="cf_to")
                custom_filter_expr = f"duration >= {cf_from * 60} and duration <= {cf_to * 60}"
            else:
                custom_filter_expr = "duration == 0"
            st.caption("Filter applied: calls between the values you chose above.")

    with ccol:
        count_mode = st.radio("How many calls?", ["All matching", "Manual number"], horizontal=True)

    sort_order = st.radio(
        "Sort by Duration", ["Descending (longest first)", "Ascending (shortest first)"], horizontal=True, index=0,
    )
    ascending_sort = sort_order.startswith("Ascending")

    if bucket == short_label:
        matched = cdr_df[cdr_df["_duration_sec"] < duration_config['short_max']]
    elif bucket == medium_label:
        matched = cdr_df[(cdr_df["_duration_sec"] >= duration_config['medium_min']) & (cdr_df["_duration_sec"] <= duration_config['medium_max'])]
    elif bucket == large_label:
        if has_xl:
            matched = cdr_df[(cdr_df["_duration_sec"] > duration_config['large_min']) & (cdr_df["_duration_sec"] <= duration_config['extra_large_min'])]
        else:
            matched = cdr_df[cdr_df["_duration_sec"] > duration_config['large_min']]
    elif has_xl and bucket == extra_large_label:
        matched = cdr_df[cdr_df["_duration_sec"] > duration_config['extra_large_min']]
    elif bucket == "Custom Filter":
        if custom_filter_expr:
            expr = custom_filter_expr.replace('duration', 'cdr_df["_duration_sec"]')
            try:
                matched = cdr_df[eval(expr)]
            except Exception as e:
                st.error(f"⚠️ Error in filter expression: {e}")
                matched = cdr_df
        else:
            matched = cdr_df
    else:
        matched = cdr_df

    available = len(matched)
    if count_mode == "Manual number":
        manual_n = st.number_input("Number of calls", min_value=1, value=min(50, available) if available else 1, step=1)
        selected_df = matched.head(int(manual_n))
        if available < manual_n:
            st.warning(f"Only {available} call(s) match – showing all.")
        else:
            st.info(f"Showing {int(manual_n)} of {available} matching calls.")
    else:
        selected_df = matched
        st.info(f"Showing all {available} matching calls for **{bucket}**.")

    display_cols = [
        "campaign_id", "agent", "full_name", "leadid", "phone_number",
        "call_date", "start_time", "end_time", "call_duration1", "Recording", "_duration_sec"
    ]
    available_display_cols = [c for c in display_cols if c in selected_df.columns]
    table_df = selected_df[available_display_cols].copy()
    table_df.sort_values("_duration_sec", ascending=ascending_sort, inplace=True)
    table_df.insert(1, "S.No", range(1, len(table_df) + 1))
    final_display_cols = ["_duration_sec", "S.No"] + [c for c in available_display_cols if c != "_duration_sec"]
    table_df = table_df[final_display_cols]
    st.markdown(f"### Filtered Data – Sorted by Duration ({'ascending' if ascending_sort else 'descending'})")
    st.dataframe(table_df, use_container_width=True, height=350)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Step 3: VAD + sentiment on filtered calls ----
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">3</span>Run Talk-Time & Sentiment Analysis (Filtered Calls)</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Downloads recordings, measures speech/silence, transcribes with Groq Whisper (VAD-based), and performs sentiment analysis via Groq LLM.</p>', unsafe_allow_html=True)

    with st.expander("⚙️ Fine-tune detection accuracy (optional)"):
        st.caption(
            "If Talk Time is coming out too low / Silence too high, move the slider "
            "towards 'Detect more speech'. Default is fine for most calls."
        )
        sensitivity = st.slider("Detection sensitivity", min_value=1, max_value=9, value=5,
                                 help="Lower = detect more speech. Higher = stricter, only counts confident speech.")
        vad_threshold = round(0.1 + (sensitivity - 1) * (0.35 - 0.1) / 8, 3)
        dead_air_secs = st.number_input("Count a pause as 'Dead Air' only if longer than (sec)", min_value=1, value=5, step=1)

    run_vad_clicked = st.button("▶️  Run Analysis & Build Report (Filtered)", type="primary")

    if run_vad_clicked:
        if not col_recording:
            st.error("No recording-URL column found in CDR data — cannot fetch recordings.")
        elif len(selected_df) == 0:
            st.warning("No calls selected — nothing to process.")
        else:
            results = []
            progress = st.progress(0)
            buffer_box = st.empty()
            render_buffer(buffer_box, "Analyzing calls, transcribing & scoring sentiment...")
            total_rows = len(selected_df)

            with tempfile.TemporaryDirectory() as tmpdir:
                for i, (_, row) in enumerate(selected_df.iterrows()):
                    rec_url = row.get(col_recording)
                    
                    metrics, debug_status, actual_mp3, transcript, sentiment = process_recording_with_transcription(
                        rec_url, vad_threshold, dead_air_secs, tmpdir, f"filtered_{i}"
                    )
                    
                    if debug_status != "OK" and "API" not in debug_status:
                        st.warning(f"Row {i+1} ({row.get(col_agent) if col_agent else ''}): {debug_status}")

                    crm_duration = row.get("_duration_sec")
                    date_val = row.get(col_date) if col_date else None
                    time_val = row.get(col_time) if col_time else None
                    if date_val and pd.notna(date_val):
                        try:
                            date_val = pd.to_datetime(date_val).strftime("%d/%m/%Y")
                        except Exception:
                            pass

                    results.append({
                        "Date": date_val, "Time": time_val,
                        "Agent Name": row.get(col_agent) if col_agent else None,
                        "Call From": row.get(col_phone) if col_phone else None,
                        "Actual MP3": actual_mp3,
                        "Audio Duration(sec)": metrics.get("duration"),
                        "Audio Call Duration": crm_duration,
                        "AI Tools Talk time": metrics.get("talk_time"),
                        "Silence Time": metrics.get("silence_time"),
                        "Dead Air(included in Silence time)": metrics.get("dead_air"),
                        "Longest Silence": metrics.get("longest_silence"),
                        "Transcript": transcript[:500] + "..." if len(transcript) > 500 else transcript,
                        "Sentiment": sentiment,
                        "_debug_status": debug_status,
                    })
                    progress.progress((i + 1) / total_rows)

            buffer_box.empty()
            progress.empty()

            final_df = pd.DataFrame(results)
            final_df.sort_values("Audio Call Duration", ascending=False, inplace=True)
            REORDERED_COLUMNS = [
                "Audio Call Duration", "Date", "Time", "Agent Name", "Call From",
                "AI Tools Talk time", "Silence Time", "Dead Air(included in Silence time)",
                "Longest Silence", "Transcript", "Sentiment", "Audio Duration(sec)", "Actual MP3",
            ]
            final_df = final_df[REORDERED_COLUMNS + ["_debug_status"]]
            agent_analytics_df = generate_agent_analytics(selected_df, '_duration_sec', duration_config)

            st.session_state["final_df"] = final_df
            st.session_state["agent_analytics_df"] = agent_analytics_df

            failed_count = (final_df["_debug_status"] != "OK").sum()
            if failed_count > 0:
                st.error(f"⚠️ {failed_count} of {len(final_df)} call(s) failed to process.")
            else:
                st.success("✅ All calls processed successfully.")
            st.dataframe(final_df.drop(columns=["_debug_status"]), use_container_width=True, height=380)

            if agent_analytics_df is not None and len(agent_analytics_df) > 0:
                st.markdown("### 📊 Agent-Wise Analytics")
                st.info("Comprehensive breakdown of call performance by agent.")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🏆 Top Agents by Large Calls**")
                    top_large = agent_analytics_df.nlargest(3, 'Large_Calls')[['Agent', 'Large_Calls', 'Large_%']]
                    for _, row in top_large.iterrows():
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-name">{row['Agent']}</div>
                            <div class="agent-stats">{row['Large_Calls']} large calls ({row['Large_%']:.1f}% of total)</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("**📉 Agents with Most Short Calls**")
                    top_short = agent_analytics_df.nlargest(3, 'Short_Calls')[['Agent', 'Short_Calls', 'Short_%']]
                    for _, row in top_short.iterrows():
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-name">{row['Agent']}</div>
                            <div class="agent-stats">{row['Short_Calls']} short calls ({row['Short_%']:.1f}% of total)</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("**📈 Performance Summary**")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Agents", len(agent_analytics_df))
                with m2:
                    st.metric("Avg Calls per Agent", f"{agent_analytics_df['Total_Calls'].mean():.1f}")
                with m3:
                    st.metric("Most Active Agent", agent_analytics_df.iloc[0]['Agent'] if len(agent_analytics_df) > 0 else "N/A")
                with m4:
                    best_large = agent_analytics_df.nlargest(1, 'Large_Calls')['Agent'].iloc[0] if len(agent_analytics_df) > 0 else "N/A"
                    st.metric("Most Large Calls", best_large)

                detail_cols = ['Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_%', 'Medium_Calls', 'Medium_%', 'Large_Calls', 'Large_%']
                if 'Extra_Large_Calls' in agent_analytics_df.columns:
                    detail_cols += ['Extra_Large_Calls', 'Extra_Large_%']
                detail_cols += ['Avg_Duration_Formatted', 'Total_Duration_Formatted']
                st.dataframe(agent_analytics_df[detail_cols], use_container_width=True, height=300)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Step 4: download filtered report ----
    if st.session_state.get("final_df") is not None:
        EXPORT_COLUMNS = [
            "Audio Call Duration", "Date", "Time", "Agent Name", "Call From",
            "AI Tools Talk time", "Silence Time", "Dead Air(included in Silence time)",
            "Longest Silence", "Transcript", "Sentiment", "Audio Duration(sec)", "Actual MP3",
        ]
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title"><span class="step-badge">4</span>Download Report</div>', unsafe_allow_html=True)
        st.markdown('<p class="step-subtitle">Excel file with two sheets: Detailed Call Report (with Transcript & Sentiment) and Agent-Wise Analytics.</p>', unsafe_allow_html=True)

        export_df = st.session_state["final_df"][EXPORT_COLUMNS]
        agent_export_df = st.session_state.get("agent_analytics_df")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Call Report")
            if agent_export_df is not None and len(agent_export_df) > 0:
                has_xl_col = 'Extra_Large_Calls' in agent_export_df.columns
                export_cols = ['Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_%', 'Medium_Calls', 'Medium_%', 'Large_Calls', 'Large_%']
                export_headers = ['Rank', 'Agent', 'Total Calls', 'Short Calls', 'Short %', 'Medium Calls', 'Medium %', 'Large Calls', 'Large %']
                if has_xl_col:
                    export_cols += ['Extra_Large_Calls', 'Extra_Large_%']
                    export_headers += ['Extra Large Calls', 'Extra Large %']
                export_cols += ['Avg_Duration_Formatted', 'Total_Duration_Formatted']
                export_headers += ['Avg Duration', 'Total Duration']
                agent_sheet = agent_export_df[export_cols].copy()
                agent_sheet.columns = export_headers
                agent_sheet.to_excel(writer, index=False, sheet_name="Agent Analytics")
        buf.seek(0)
        st.download_button(
            "⬇️  Download Excel Report", data=buf,
            file_name=f"CallAI_Talk_Time_Report_{client_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Step 5: Complete Call Analysis (all calls) ----
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">5</span>Complete Call Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Run VAD on ALL calls to build the agent performance dashboard and detect defaulters.</p>', unsafe_allow_html=True)

    total_calls = len(cdr_df)
    st.info(f"📞 **{total_calls}** calls available for complete analysis")

    st.markdown("**⚙️ Analysis Configuration**")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        silence_threshold = st.number_input(
            "Silence threshold (%)", min_value=20, max_value=50, value=st.session_state.get("silence_threshold", 30), step=5,
            help="Agents with overall OR short-call silence above this are flagged.",
            key=vkey("complete_silence_threshold")
        )
    with cfg_col2:
        min_calls_per_agent = st.number_input(
            "Minimum calls per agent", min_value=1, max_value=10, value=st.session_state.get("min_calls_per_agent", 3), step=1,
            help="Agents with fewer calls are skipped.", key=vkey("complete_min_calls")
        )
    with cfg_col3:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**🎙️ Voice Detection Settings**")
    vad_col1, vad_col2 = st.columns(2)
    with vad_col1:
        complete_sensitivity = st.slider("VAD Sensitivity", 1, 9, 5, key="complete_sensitivity")
        complete_vad_threshold = round(0.1 + (complete_sensitivity - 1) * (0.35 - 0.1) / 8, 3)
    with vad_col2:
        complete_dead_air = st.number_input("Dead-air threshold (sec)", min_value=1, value=5, step=1, key="complete_dead_air")

    run_complete_clicked = st.button("▶️  Run Analysis", type="primary", key="run_complete_analysis_btn")

    if run_complete_clicked:
        if not col_recording:
            st.error("No recording-URL column found — cannot run analysis.")
        elif len(cdr_df) == 0:
            st.warning("No calls to analyze.")
        else:
            dur_config = get_duration_config(client_name)
            silence_df, perf_df, defaulters_df = run_complete_call_analysis(
                cdr_df, col_recording, col_agent, col_date, col_time, col_phone,
                complete_vad_threshold, complete_dead_air, dur_config,
                silence_threshold, min_calls_per_agent
            )
            st.session_state["complete_analysis_df"] = silence_df
            st.session_state["agent_silence_by_category_df"] = perf_df
            st.session_state["defaulter_agents_df"] = defaulters_df

            st.markdown("### 📞 Call-Level Silence Analysis")
            st.info(f"Analyzed {len(silence_df)} calls across all categories")
            display_cols = ['Date', 'Time', 'Agent Name', 'Duration (sec)', 'Silence %', 'Longest Silence (sec)']
            display_cols = [c for c in display_cols if c in silence_df.columns]
            st.dataframe(silence_df[display_cols].head(20), use_container_width=True, height=300)

            if perf_df is not None and len(perf_df) > 0:
                st.markdown("### 📊 Agent Performance Dashboard")
                perf_display = perf_df[[
                    'Rank', 'Agent', 'Overall_Avg_Silence_%',
                    'Avg_Silence_%_Short', 'Avg_Silence_%_Medium', 'Avg_Silence_%_Large', 'Total_Calls'
                ]].copy()
                perf_display.columns = ['Rank', 'Agent', 'Overall Silence %', 'Short %', 'Medium %', 'Large %', 'Total Calls']
                st.dataframe(perf_display, use_container_width=True, height=300)

            if defaulters_df is not None and len(defaulters_df) > 0:
                st.markdown(
                    f'<div class="status-banner-danger">🚨 {len(defaulters_df)} defaulter agent(s) detected!</div>',
                    unsafe_allow_html=True
                )
                def_display = defaulters_df[[
                    'Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_Silence_%',
                    'Medium_Calls', 'Medium_Silence_%', 'Large_Calls', 'Large_Silence_%', 'Overall_Silence_%'
                ]].copy()
                def_display.columns = [
                    'Rank', 'Agent', 'Total Calls', 'Short Calls', 'Short Silence %',
                    'Medium Calls', 'Medium Silence %', 'Large Calls', 'Large Silence %', 'Overall Silence %'
                ]
                st.dataframe(def_display, use_container_width=True, height=300)

    st.markdown("### ⬇️ Download Complete Analysis Reports")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        if st.session_state.get("complete_analysis_df") is not None:
            silence_df = st.session_state["complete_analysis_df"]
            buf_s = io.BytesIO()
            with pd.ExcelWriter(buf_s, engine="openpyxl") as writer:
                silence_df.to_excel(writer, index=False, sheet_name="Call Data")
            buf_s.seek(0)
            st.download_button(
                "⬇️ Download Call Data", data=buf_s,
                file_name=f"Complete_Call_Data_{client_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_complete_data",
            )
    with dcol2:
        if st.session_state.get("agent_silence_by_category_df") is not None:
            perf_df = st.session_state["agent_silence_by_category_df"]
            buf_p = io.BytesIO()
            with pd.ExcelWriter(buf_p, engine="openpyxl") as writer:
                perf_df.to_excel(writer, index=False, sheet_name="Performance Dashboard")
            buf_p.seek(0)
            st.download_button(
                "⬇️ Download Performance Report", data=buf_p,
                file_name=f"Performance_Dashboard_{client_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_complete_perf",
            )
    with dcol3:
        if st.session_state.get("defaulter_agents_df") is not None and len(st.session_state["defaulter_agents_df"]) > 0:
            defaulters_df = st.session_state["defaulter_agents_df"]
            buf_d = io.BytesIO()
            with pd.ExcelWriter(buf_d, engine="openpyxl") as writer:
                defaulters_df.to_excel(writer, index=False, sheet_name="Defaulter Agents")
            buf_d.seek(0)
            st.download_button(
                "⬇️ Download Defaulter Report", data=buf_d,
                file_name=f"Defaulter_Report_{client_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_complete_def",
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Settings Section
# ----------------------------------------------------------------------
def render_settings():
    """Render the settings page with tabs."""
    
    st.markdown("""
    <div class="callai-hero">
        <h1>⚙️ Settings</h1>
        <p>Manage clients, email recipients, and system configuration.</p>
    </div>
    """, unsafe_allow_html=True)

    top_l, top_r = st.columns([4, 1])
    with top_r:
        if st.button("🔄 Refresh from DB", key="settings_refresh_all", use_container_width=True):
            load_all_configs()
            bump_ui_version()
            st.success("Synced with database.")
            st.rerun()
    
    # Create tabs for settings
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Client Configuration", 
        "📧 Email Settings", 
        "⚙️ System Config",
        "📊 Defaulters Config"
    ])
    
    # ---- Tab 1: Client Configuration ----
    with tab1:
        st.markdown("### 🏢 Client Management")
        st.caption("Add, remove, or update client configurations. Changes sync across all users.")
        
        # Display existing clients
        clients = st.session_state.get("clients", {})
        
        if clients:
            st.markdown("#### Current Clients")
            for name, data in clients.items():
                if isinstance(data, dict):
                    company_id = data.get("company_id", "")
                    short_max = data.get("short_max", 120)
                    medium_max = data.get("medium_max", 300)
                    large_min = data.get("large_min", 300)
                else:
                    company_id = data
                    short_max = 120
                    medium_max = 300
                    large_min = 300
                
                st.markdown(f"""
                <div class="client-card">
                    <div class="client-name">{name}</div>
                    <div class="client-details">
                        Company ID: {company_id} · 
                        Short: &lt;{fmt_hms(short_max)} · 
                        Medium: {fmt_hms(short_max)}–{fmt_hms(medium_max)} · 
                        Large: &gt;{fmt_hms(large_min)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Add new client
        st.markdown("#### ➕ Add New Client")
        col1, col2 = st.columns(2)
        with col1:
            new_client_name = st.text_input("Client Name", placeholder="e.g., New Client", key="new_client_name")
            new_company_id = st.text_input("Company ID", placeholder="e.g., 123", key="new_company_id")
        with col2:
            new_short = st.number_input("Short ends at (seconds)", min_value=30, value=120, step=10, key="new_short")
            new_medium = st.number_input("Medium ends at (seconds)", min_value=new_short, value=300, step=10, key="new_medium")
            new_large = st.number_input("Large starts at (seconds)", min_value=new_medium, value=300, step=10, key="new_large")
        
        if st.button("➕ Add Client", type="primary", key="add_client_btn"):
            if new_client_name and new_company_id:
                config = {
                    "short_max": int(new_short),
                    "medium_min": int(new_short),
                    "medium_max": int(new_medium),
                    "large_min": int(new_large),
                    "extra_large_enabled": False,
                    "extra_large_min": int(new_medium + 180),
                }
                if save_client_to_db(new_client_name, new_company_id, config):
                    st.success(f"✅ Client '{new_client_name}' added successfully!")
                    load_all_configs()
                    bump_ui_version()
                    st.rerun()
                else:
                    st.error("❌ Failed to add client. Check Supabase connection.")
            else:
                st.warning("Please enter both Client Name and Company ID.")
        
        # Remove client
        st.markdown("#### 🗑️ Remove Client")
        if clients:
            remove_client = st.selectbox("Select client to remove", [""] + list(clients.keys()), key="remove_client_select")
            if remove_client and st.button("🗑️ Remove Client", type="secondary", key="remove_client_btn"):
                if delete_client_from_db(remove_client):
                    st.success(f"✅ Client '{remove_client}' removed successfully!")
                    load_all_configs()
                    bump_ui_version()
                    st.rerun()
                else:
                    st.error("❌ Failed to remove client.")
        else:
            st.info("No clients to remove.")
    
    # ---- Tab 2: Email Settings ----
    with tab2:
        st.markdown("### 📧 Email Settings")
        st.caption("Manage SMTP configuration and report recipients.")
        
        # SMTP Configuration
        st.markdown("#### SMTP Configuration")
        smtp_config = st.session_state.get("smtp_config", {})
        
        col1, col2 = st.columns(2)
        with col1:
            smtp_host = st.text_input("SMTP Server", value=smtp_config.get("host", "mail.dialdesk.net"), key=vkey("settings_smtp_host"))
            smtp_port = st.number_input("Port", value=int(smtp_config.get("port", 587)), step=1, key=vkey("settings_smtp_port"))
            smtp_username = st.text_input("Username", value=smtp_config.get("username", ""), key=vkey("settings_smtp_username"))
        with col2:
            smtp_password = st.text_input("Password", value=smtp_config.get("password", ""), type="password", key=vkey("settings_smtp_password"))
            smtp_from = st.text_input("From Email", value=smtp_config.get("from_email", ""), key=vkey("settings_smtp_from"))
            smtp_tls = st.checkbox("Use TLS", value=smtp_config.get("use_tls", True), key=vkey("settings_smtp_tls"))
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 Save SMTP Settings", key="save_smtp_settings"):
                updated_config = {
                    "host": smtp_host,
                    "port": int(smtp_port),
                    "username": smtp_username,
                    "password": smtp_password,
                    "from_email": smtp_from or smtp_username,
                    "use_tls": smtp_tls,
                }
                if save_smtp_config_to_db(updated_config):
                    st.success("✅ SMTP settings saved!")
                    load_all_configs()
                    bump_ui_version()
                    st.rerun()
                else:
                    st.error("❌ Failed to save SMTP settings.")
        
        with col_btn2:
            if st.button("🔬 Test Connection", key="test_smtp_settings"):
                ok, msg = test_smtp_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        
        st.markdown("---")
        
        # Mentor Emails
        st.markdown("#### 👥 Report Recipients")
        st.caption("Emails that will receive performance reports and defaulter alerts.")
        
        current_emails = st.session_state.get("mentor_emails", [])
        
        if current_emails:
            chips = "".join(f'<span class="email-chip">{e}</span>' for e in current_emails)
            st.markdown(f'<div class="email-list-box">{chips}</div>', unsafe_allow_html=True)
        else:
            st.warning("No recipients added yet.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            new_email = st.text_input("Add recipient email", placeholder="mentor@example.com", key="settings_new_email")
        with col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("➕ Add", use_container_width=True, key="settings_add_email"):
                if new_email and "@" in new_email and "." in new_email:
                    emails = current_emails + [new_email.strip()]
                    if save_mentor_emails_to_db(emails):
                        st.success(f"✅ Added {new_email.strip()}")
                        load_all_configs()
                        bump_ui_version()
                        st.rerun()
                    else:
                        st.error("❌ Failed to save email.")
                else:
                    st.warning("Please enter a valid email address.")
        
        if current_emails:
            col1, col2 = st.columns([2, 1])
            with col1:
                remove_email = st.selectbox("Remove recipient", [""] + current_emails, key=vkey("settings_remove_select"))
            with col2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Remove", use_container_width=True, key="settings_remove_email"):
                    if remove_email:
                        emails = [e for e in current_emails if e != remove_email]
                        if save_mentor_emails_to_db(emails):
                            st.success(f"✅ Removed {remove_email}")
                            load_all_configs()
                            bump_ui_version()
                            st.rerun()
                        else:
                            st.error("❌ Failed to remove email.")
        
        # Auto-schedule clients
        st.markdown("#### 🤖 Auto-Schedule Clients")
        st.caption("Clients for which daily reports will be automatically generated.")
        
        auto_clients = st.session_state.get("auto_schedule_clients", ["Weebo", "Hari Om Pvt Ltd"])
        all_clients = list(st.session_state.get("clients", {}).keys())
        
        # Only show the multiselect if there are clients
        if all_clients:
            # Filter auto_clients to only include those that exist in all_clients
            valid_auto_clients = [c for c in auto_clients if c in all_clients]
            
            selected_auto = st.multiselect(
                "Select clients for auto-reporting",
                options=all_clients,
                default=valid_auto_clients if valid_auto_clients else [],
                key=vkey("settings_auto_clients")
            )
            
            if st.button("💾 Save Auto-Schedule Clients", key="save_auto_clients"):
                # Merge with the existing scheduler_time / scheduler_enabled so we
                # never overwrite them, and use the same upsert-safe helper used
                # everywhere else (no hard-coded row id, so it always saves
                # correctly whether or not a scheduler_config row exists yet).
                config = {
                    "scheduler_time": st.session_state.get("scheduler_time", "23:00"),
                    "scheduler_enabled": st.session_state.get("scheduler_enabled", True),
                    "auto_schedule_clients": selected_auto,
                }
                if save_scheduler_config_to_db(config):
                    st.session_state["auto_schedule_clients"] = selected_auto
                    st.success("✅ Auto-schedule clients updated!")
                    load_all_configs()
                    bump_ui_version()
                    st.rerun()
                else:
                    st.error("❌ Failed to update. Check Supabase connection.")
        else:
            st.info("Add clients first to configure auto-scheduling.")
    
    # ---- Tab 3: System Config ----
    with tab3:
        st.markdown("### ⚙️ System Configuration")
        
        # Scheduler Configuration
        st.markdown("#### ⏰ Scheduler Settings")
        
        current_time = st.session_state.get("scheduler_time", "23:00")
        current_enabled = st.session_state.get("scheduler_enabled", True)
        
        col1, col2 = st.columns(2)
        with col1:
            new_scheduler_time = st.text_input("Report Time (HH:MM)", value=current_time, key=vkey("settings_scheduler_time"))
        with col2:
            scheduler_enabled = st.checkbox("Enable Auto-Scheduler", value=current_enabled, key=vkey("settings_scheduler_enabled"))
        
        if st.button("💾 Save Scheduler Settings", key="save_scheduler_settings"):
            try:
                datetime.strptime(new_scheduler_time, "%H:%M")
                config = {
                    "scheduler_time": new_scheduler_time,
                    "scheduler_enabled": scheduler_enabled,
                    "auto_schedule_clients": st.session_state.get("auto_schedule_clients", []),
                }
                if save_scheduler_config_to_db(config):
                    st.session_state["scheduler_time"] = new_scheduler_time
                    st.session_state["scheduler_enabled"] = scheduler_enabled
                    restart_scheduler()
                    bump_ui_version()
                    st.success(f"✅ Scheduler updated! Reports will run daily at {new_scheduler_time}")
                    st.rerun()
                else:
                    st.error("❌ Failed to save scheduler settings.")
            except ValueError:
                st.error("❌ Invalid time format. Please use HH:MM (e.g., 23:00)")
        
        # Scheduler status
        if st.session_state.get("scheduler_running", False):
            st.markdown(f'<div class="success-box">🟢 Scheduler is running. Next report at {st.session_state.get("scheduler_time", "23:00")}</div>', unsafe_allow_html=True)
        else:
            status_text = "enabled" if st.session_state.get("scheduler_enabled", True) else "disabled"
            st.markdown(f'<div class="warning-box">🟡 Scheduler is {status_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Last email status
        if st.session_state.get("last_email_time"):
            st.caption(f"📨 Last report sent: {st.session_state['last_email_time']}")
        
        # Supabase Status
        st.markdown("#### 🔗 Database Status")
        if supabase:
            st.markdown('<div class="success-box">✅ Connected to Supabase</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ Not connected to Supabase. Changes won\'t be saved across sessions.</div>', unsafe_allow_html=True)
    
    # ---- Tab 4: Defaulters Config ----
    with tab4:
        st.markdown("### 📊 Defaulter Detection Configuration")
        st.caption("Configure thresholds for detecting defaulter agents.")
        
        current_silence = st.session_state.get("silence_threshold", 30)
        current_min_calls = st.session_state.get("min_calls_per_agent", 3)
        
        col1, col2 = st.columns(2)
        with col1:
            new_silence_threshold = st.number_input(
                "Silence Threshold (%)", 
                min_value=20, max_value=50, 
                value=current_silence, step=5,
                help="Agents with overall or short-call silence above this are flagged.",
                key=vkey("settings_silence_threshold")
            )
        with col2:
            new_min_calls = st.number_input(
                "Minimum Calls per Agent",
                min_value=1, max_value=10,
                value=current_min_calls, step=1,
                help="Agents with fewer calls are skipped from defaulter detection.",
                key=vkey("settings_min_calls")
            )
        
        if st.button("💾 Save Defaulter Settings", key="save_defaulter_settings"):
            config = {
                "silence_threshold": int(new_silence_threshold),
                "min_calls_per_agent": int(new_min_calls),
            }
            if save_defaulters_config_to_db(config):
                st.session_state["silence_threshold"] = new_silence_threshold
                st.session_state["min_calls_per_agent"] = new_min_calls
                bump_ui_version()
                st.success("✅ Defaulter settings saved!")
                st.rerun()
            else:
                st.error("❌ Failed to save defaulter settings.")
        
        st.markdown("---")
        st.markdown("#### 📋 How Defaulters are Detected")
        st.info("""
        An agent is flagged as a **defaulter** if:
        
        1. **Overall Silence > {silence_threshold}%** across all their calls
        2. **OR Short Call Silence > {silence_threshold}%** (calls shorter than the "Short" cutoff)
        
        Agents with fewer than **{min_calls} calls** are skipped from detection.
        
        **Severity Levels:**
        - 🔴 **High**: Overall Silence > 40%
        - 🟠 **Medium**: Overall Silence 30-40%
        - 🔵 **Low**: Overall Silence < 30% (but still above threshold)
        """.format(
            silence_threshold=st.session_state.get("silence_threshold", 30),
            min_calls=st.session_state.get("min_calls_per_agent", 3)
        ))

# ----------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------
# Start scheduler on app load
if not st.session_state.get("scheduler_running", False):
    start_scheduler()

# Header
st.markdown("""
<div class="callai-hero">
    <h1>📞 CallAI · Talk-Time + Sentiment</h1>
    <p>Analyze call recordings, measure talk-time/silence, transcribe, score sentiment, and generate reports.</p>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_dashboard, tab_settings = st.tabs(["📊 Dashboard / Call Analytics", "⚙️ Settings"])

with tab_dashboard:
    render_dashboard()

with tab_settings:
    render_settings()
