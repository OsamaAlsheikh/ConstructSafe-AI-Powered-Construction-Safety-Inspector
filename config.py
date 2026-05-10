# config.py  [ROOT FOLDER]
# Central configuration — ALL other files import from here.
# Change a model name or prompt in ONE place; everything updates.

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / 'violations.db'
TEST_VIDEOS_DIR = BASE_DIR / 'test_videos'


import os
from pathlib import Path

VLLM_VISION_URL = os.getenv('VLLM_VISION_URL', 'http://localhost:8000')
VLLM_REPORT_URL = os.getenv('VLLM_REPORT_URL', 'http://localhost:8001')
VISION_MODEL_NAME = 'Qwen/Qwen2.5-VL-7B-Instruct'
REPORT_MODEL_NAME = os.getenv('REPORT_MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')
VISION_MAX_TOKENS  = 512
REPORT_MAX_TOKENS  = 1024
VISION_TEMPERATURE = 0.1
REPORT_TEMPERATURE = 0.2

# ── Models ────────────────────────────────────────────────
VISION_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
REPORT_MODEL_ID = os.getenv('REPORT_MODEL_ID', 'Qwen/Qwen2.5-7B-Instruct')
USE_4BIT = True   # Set False if you have enough VRAM (>40GB)

# ── Video settings ────────────────────────────────────────
FRAME_SAMPLE_INTERVAL_SECONDS = 2  # Sample one frame every 2 seconds
MAX_FRAMES = 8                     # Cap to control cost + speed

# ── OSHA violation taxonomy ───────────────────────────────
VIOLATION_TYPES = [
    {'code':'PPE-001',  'name':'Missing hard hat',            'regulation':'29 CFR 1926.100(a)',   'severity':'Serious'},
    {'code':'PPE-002',  'name':'Missing high-visibility vest','regulation':'29 CFR 1926.201(a)',   'severity':'Serious'},
    {'code':'FALL-001', 'name':'No fall protection at height','regulation':'29 CFR 1926.502(d)',   'severity':'Willful'},
    {'code':'FALL-002', 'name':'Unsafe scaffold',             'regulation':'29 CFR 1926.451(g)(1)','severity':'Serious'},
    {'code':'ELEC-001', 'name':'Exposed electrical cable',    'regulation':'29 CFR 1926.416(a)(1)','severity':'Serious'},
    {'code':'HOUSE-001','name':'Blocked emergency exit',      'regulation':'29 CFR 1926.34(a)',    'severity':'Serious'},
]
VIOLATION_LOOKUP = {v['code']: v for v in VIOLATION_TYPES}

# ── Vision system prompt (sent to Qwen2.5-VL with every frame) ──
VISION_SYSTEM_PROMPT = """You are an expert OSHA construction safety inspector.
Analyse the image and return ONLY valid JSON — no preamble, no explanation:
{
  'violations': [{'code':'PPE-001','confidence':0.92,
                  'location_description':'...','observation':'...'}],
  'worker_count': 3,
  'overall_risk_level': 'HIGH',
  'scene_description': '...'
}
Violation codes: PPE-001 (hard hat), PPE-002 (hi-vis vest),
FALL-001 (fall protection), FALL-002 (scaffold),
ELEC-001 (electrical), HOUSE-001 (emergency exit).
Do NOT invent violations. Confidence 0.0-1.0."""

VISION_USER_PROMPT = 'Inspect this construction site. Return JSON only.'

# ── Report system prompt (sent to fine-tuned LLM) ────────
REPORT_SYSTEM_PROMPT = """You are an OSHA Compliance Officer (29 CFR 1926).
Write a formal safety inspection report. Include:
  - Executive Summary
  - Each violation: regulation, severity, observation, corrective action, deadline, penalty
  - Compliance Summary
  - Inspector Notes
Use precise OSHA language. Reference exact CFR citations."""


# config.py — complete file
# config.py  [ROOT FOLDER]
# Central configuration — ALL other files import from here.
# Change a model name or prompt in ONE place; everything updates.

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / 'violations.db'
TEST_VIDEOS_DIR = BASE_DIR / 'test_videos'

# ── Models ────────────────────────────────────────────────
VISION_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
REPORT_MODEL_ID = os.getenv('REPORT_MODEL_ID', 'Qwen/Qwen2.5-7B-Instruct')
USE_4BIT = True   # Set False if you have enough VRAM (>40GB)

# ── Video settings ────────────────────────────────────────
FRAME_SAMPLE_INTERVAL_SECONDS = 2  # Sample one frame every 2 seconds
MAX_FRAMES = 8                     # Cap to control cost + speed

# ── OSHA violation taxonomy ───────────────────────────────
VIOLATION_TYPES = [
    {'code':'PPE-001',  'name':'Missing hard hat',            'regulation':'29 CFR 1926.100(a)',   'severity':'Serious'},
    {'code':'PPE-002',  'name':'Missing high-visibility vest','regulation':'29 CFR 1926.201(a)',   'severity':'Serious'},
    {'code':'FALL-001', 'name':'No fall protection at height','regulation':'29 CFR 1926.502(d)',   'severity':'Willful'},
    {'code':'FALL-002', 'name':'Unsafe scaffold',             'regulation':'29 CFR 1926.451(g)(1)','severity':'Serious'},
    {'code':'ELEC-001', 'name':'Exposed electrical cable',    'regulation':'29 CFR 1926.416(a)(1)','severity':'Serious'},
    {'code':'HOUSE-001','name':'Blocked emergency exit',      'regulation':'29 CFR 1926.34(a)',    'severity':'Serious'},
]
VIOLATION_LOOKUP = {v['code']: v for v in VIOLATION_TYPES}

# ── Vision system prompt (sent to Qwen2.5-VL with every frame) ──
VISION_SYSTEM_PROMPT = """You are an expert OSHA construction safety inspector.
Analyse the image and return ONLY valid JSON — no preamble, no explanation:
{
  'violations': [{'code':'PPE-001','confidence':0.92,
                  'location_description':'...','observation':'...'}],
  'worker_count': 3,
  'overall_risk_level': 'HIGH',
  'scene_description': '...'
}
Violation codes: PPE-001 (hard hat), PPE-002 (hi-vis vest),
FALL-001 (fall protection), FALL-002 (scaffold),
ELEC-001 (electrical), HOUSE-001 (emergency exit).
Do NOT invent violations. Confidence 0.0-1.0."""

VISION_USER_PROMPT = 'Inspect this construction site. Return JSON only.'

# ── Report system prompt (sent to fine-tuned LLM) ────────
REPORT_SYSTEM_PROMPT = """You are an OSHA Compliance Officer (29 CFR 1926).
Write a formal safety inspection report. Include:
  - Executive Summary
  - Each violation: regulation, severity, observation, corrective action, deadline, penalty
  - Compliance Summary
  - Inspector Notes
Use precise OSHA language. Reference exact CFR citations."""

# ── Email templates ───────────────────────────────────────
EMAIL_SUBJECT_TEMPLATE = 'SAFETY VIOLATION ALERT -- {site_name} -- {date} -- {violation_count} violation(s)'
EMAIL_BODY_TEMPLATE = '''Dear {supervisor_name},
ConstructSafe detected {violation_count} safety violation(s) at {site_name} on {date}.
Risk level: {risk_level}. Workers at risk: {workers_at_risk}.
{urgent_warning}
REQUIRED ACTIONS:
{action_items}
Full OSHA report attached. Confirm receipt within 24 hours.'''
URGENT_WARNING = 'IMMEDIATE ACTION REQUIRED. Work must halt until corrections are verified.'

# ── Database settings ─────────────────────────────────────
DB_VERSION = 1
REPEAT_VIOLATION_THRESHOLD = 3     # Inspections before 'repeat' flag fires
REPEAT_VIOLATION_WINDOW_DAYS = 30  # Rolling window for repeat detection
