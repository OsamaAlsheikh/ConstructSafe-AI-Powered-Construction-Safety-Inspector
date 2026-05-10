# mock_mode.py  [root/]
# Patches vllm_client so the pipeline runs without AMD Cloud.
# Import this at the top of app.py during local testing.
# Remove the import before deploying to HF Space.

import json
from PIL import Image

def mock_call_vision(system_prompt, user_text, image):
    """Returns a fake but realistic violation JSON string."""
    fake = {
        "violations": [
            {
                "code": "PPE-001",
                "confidence": 0.91,
                "location_description": "Worker in foreground, left of frame",
                "observation": "Male worker operating power drill without hard hat in active zone"
            },
            {
                "code": "FALL-001",
                "confidence": 0.87,
                "location_description": "Worker on elevated platform, right side",
                "observation": "Worker at approximately 8ft elevation without harness or guardrail"
            }
        ],
        "worker_count": 4,
        "overall_risk_level": "HIGH",
        "scene_description": "Outdoor construction site, concrete structure, multiple workers present"
    }
    return json.dumps(fake)

def mock_call_report(system_prompt, user_prompt):
    """Returns a fake but realistic OSHA report string."""
    return """CONSTRUCTION SITE SAFETY INSPECTION REPORT
============================================
Site: [extracted from prompt]
Inspection Date: Mock Inspection
Inspector: ConstructSafe AI System (MOCK MODE)
Overall Risk Level: HIGH

EXECUTIVE SUMMARY
-----------------
This inspection identified 2 safety violations requiring immediate attention.
One Willful violation (fall protection) requires immediate work stoppage.

VIOLATIONS DETECTED
-------------------
Violation #1: Missing hard hat
  Regulation: 29 CFR 1926.100(a)
  Severity: Serious
  Observation: Worker operating drill without head protection
  Corrective Action: Provide ANSI-compliant hard hats immediately
  Deadline: Immediate
  Potential Penalty: $1,190 - $15,625

Violation #2: No fall protection at height
  Regulation: 29 CFR 1926.502(d)
  Severity: Willful
  Observation: Worker at 8ft elevation without harness or guardrail
  Corrective Action: HALT work at elevation. Install guardrails or issue PFAS.
  Deadline: IMMEDIATE - work halt required
  Potential Penalty: $15,625 - $156,259

COMPLIANCE SUMMARY
------------------
Violations found: 2
Workers at risk: 4
Immediate action required: YES"""

def mock_check_vllm_health():
    """Tells the status bar the server is online (fake)."""
    return {
        "healthy": True,
        "vision_loaded": True,
        "report_loaded": True,
        "models": ["Qwen/Qwen2.5-VL-7B-Instruct", "mock/constructsafe-osha-7b"],
        "vision_url": "http://localhost:MOCK",
        "report_url": "http://localhost:MOCK",
        "error": None,
    }

# ── Patch the real vllm_client functions with mock versions ──
import vllm_client
vllm_client.call_vision  = mock_call_vision
vllm_client.call_report  = mock_call_report
vllm_client.check_vllm_health = mock_check_vllm_health

print("MOCK MODE ACTIVE — AMD Cloud calls replaced with fake responses")