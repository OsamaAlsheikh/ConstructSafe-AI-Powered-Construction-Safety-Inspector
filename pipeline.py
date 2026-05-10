# pipeline.py  [ROOT FOLDER]
# The orchestrator. The ONLY file app.py needs to call.
# Wires vision_agent -> report_agent -> notification_agent
# in sequence, with error handling and timing at each step.

import logging, time
from pathlib import Path
from typing import Union
from vision_agent       import analyse_media
from report_agent       import generate_report
from notification_agent import draft_notification
from database           import get_repeat_violation_count

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def run_inspection(media_path, site_name: str,
                   supervisor_name: str='Site Supervisor',
                   supervisor_email: str='') -> dict:
    """
    Full inspection pipeline: vision -> report -> notification.
    Returns a result dict with keys:
      success, error, timing, vision_results, report_text,
      notification, sample_frames
    """
    t0  = time.time()
    res = {'success':False,'error':None,'timing':{},
           'vision_results':{},'report_text':'',
           'notification':{},'sample_frames':[]}

    # ── Step 1: Vision ─────────────────────────────────────
    t1 = time.time()
    try:
        vr = analyse_media(media_path)
        res['vision_results'] = vr
        res['sample_frames']  = vr.get('annotated_frames',[])[:4]
    except Exception as e:
        res['error'] = f'Vision analysis failed: {e}'; return res
    res['timing']['vision_seconds'] = round(time.time()-t1, 2)

    # ── Step 2: Report ─────────────────────────────────────
    is_repeat = get_repeat_violation_count(site_name) >= 3
    t2 = time.time()
    try:
        rt = generate_report(vr, site_name, is_repeat)
        res['report_text'] = rt
    except Exception as e:
        res['error'] = f'Report generation failed: {e}'; return res
    res['timing']['report_seconds'] = round(time.time()-t2, 2)

    # ── Step 3: Notification ───────────────────────────────
    t3 = time.time()
    try:
        notif = draft_notification(vr, rt, site_name, supervisor_name, supervisor_email)
        res['notification'] = notif
    except Exception as e:
        res['error'] = f'Notification failed: {e}'; return res
    res['timing']['notification_seconds'] = round(time.time()-t3, 2)
    res['timing']['total_seconds']        = round(time.time()-t0, 2)
    res['success'] = True
    logger.info(f"[PIPELINE] Done in {res['timing']['total_seconds']}s. Violations: {len(vr['violations'])}")
    return res
