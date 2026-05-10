# # notification_agent.py  [ROOT FOLDER]
# # Track 1: AI Agents & Agentic Workflows
# # Takes the completed report and:
# #   1. Logs inspection to SQLite database
# #   2. Checks for repeat offender pattern
# #   3. Determines escalation level (MEDIUM / HIGH / CRITICAL)
# #   4. Drafts the supervisor notification email

# import logging
# from datetime import datetime
# from config import (EMAIL_SUBJECT_TEMPLATE, EMAIL_BODY_TEMPLATE,
#                     URGENT_WARNING, REPEAT_VIOLATION_THRESHOLD)
# from database import log_inspection, get_site_history, get_repeat_violation_count

# logger = logging.getLogger(__name__)

# def _determine_escalation(violations: list, repeat_count: int) -> tuple:
#     """
#     Returns (urgency, action_items, requires_immediate).
#     Logic: Willful OR (repeat + serious) => CRITICAL
#            Serious only => HIGH
#            Otherwise   => MEDIUM
#     """
#     has_willful = any(v.get('severity')=='Willful' for v in violations)
#     has_serious = any(v.get('severity')=='Serious' for v in violations)
#     is_repeat   = repeat_count >= REPEAT_VIOLATION_THRESHOLD

#     if has_willful or (is_repeat and has_serious):
#         return ('CRITICAL',
#                 '1. HALT work in affected areas immediately\n'
#                 '2. Do not resume until violations are corrected\n'
#                 '3. Contact OSHA compliance officer within 2 hours\n'
#                 '4. Document corrections with photographs\n'
#                 '5. Provide corrective action plan within 24 hours',
#                 True)
#     elif has_serious:
#         return ('HIGH',
#                 '1. Correct serious violations before continuing\n'
#                 '2. Provide required PPE immediately\n'
#                 '3. Document corrections and report within 4 hours',
#                 True)
#     return ('MEDIUM',
#             '1. Correct all violations within 24 hours\n'
#             '2. Brief site supervisor on findings\n'
#             '3. Update site safety checklist',
#             False)

# def draft_notification(vision_results: dict, report_text: str,
#                        site_name: str, supervisor_name: str,
#                        supervisor_email: str) -> dict:
#     """
#     MAIN ENTRY POINT. Returns:
#     { email_subject, email_body, urgency, is_repeat_site,
#       repeat_count, inspection_id, site_history }
#     """
#     violations    = vision_results.get('violations', [])
#     risk_level    = vision_results.get('overall_risk_level', 'UNKNOWN')
#     workers       = vision_results.get('total_workers_seen', 0)
#     date_str      = datetime.now().strftime('%Y-%m-%d %H:%M')

#     # Step 1: Log to database
#     inspection_id = log_inspection(site_name, violations, risk_level,
#                                    report_text, supervisor_email)

#     # Step 2: Check repeat pattern
#     repeat_count = get_repeat_violation_count(site_name)
#     is_repeat    = repeat_count >= REPEAT_VIOLATION_THRESHOLD

#     # Step 3: Escalation level
#     urgency, action_items, needs_immediate = _determine_escalation(violations, repeat_count)

#     # Step 4: Draft email
#     subject = EMAIL_SUBJECT_TEMPLATE.format(
#         site_name=site_name,
#         date=datetime.now().strftime('%Y-%m-%d'),
#         violation_count=len(violations))
#     body = EMAIL_BODY_TEMPLATE.format(
#         supervisor_name=supervisor_name or 'Site Supervisor',
#         site_name=site_name, date=date_str,
#         violation_count=len(violations), risk_level=risk_level,
#         workers_at_risk=workers,
#         urgent_warning=URGENT_WARNING if needs_immediate else '',
#         action_items=action_items)

#     return {
#         'email_subject': subject, 'email_body': body,
#         'urgency': urgency, 'is_repeat_site': is_repeat,
#         'repeat_count': repeat_count, 'inspection_id': inspection_id,
#         'site_history': get_site_history(site_name, limit=10),
#     }



# notification_agent.py
import logging
from datetime import datetime
from config import (
    EMAIL_SUBJECT_TEMPLATE, EMAIL_BODY_TEMPLATE,
    URGENT_WARNING, REPEAT_VIOLATION_THRESHOLD,
)
from database import log_inspection, get_site_history, get_repeat_violation_count

logger = logging.getLogger(__name__)


def _determine_escalation(violations: list, repeat_count: int) -> tuple:
    has_willful = any(v.get('severity') == 'Willful' for v in violations)
    has_serious = any(v.get('severity') == 'Serious' for v in violations)
    is_repeat   = repeat_count >= REPEAT_VIOLATION_THRESHOLD

    if has_willful or (is_repeat and has_serious):
        return (
            'CRITICAL',
            '1. HALT work in affected areas immediately\n'
            '2. Do not resume until violations are corrected and verified\n'
            '3. Contact OSHA compliance officer within 2 hours\n'
            '4. Document all corrective actions with photographs\n'
            '5. Provide written corrective action plan within 24 hours',
            True,
        )
    elif has_serious:
        return (
            'HIGH',
            '1. Correct all serious violations before work continues\n'
            '2. Provide required PPE to all workers immediately\n'
            '3. Document corrections and report to site manager within 4 hours',
            True,
        )
    return (
        'MEDIUM',
        '1. Correct all violations within 24 hours\n'
        '2. Brief site supervisor on findings\n'
        '3. Update site safety checklist accordingly',
        False,
    )


def draft_notification(
    vision_results: dict,
    report_text: str,
    site_name: str,
    supervisor_name: str,
    supervisor_email: str,
) -> dict:
    violations  = vision_results.get('violations', [])
    risk_level  = vision_results.get('overall_risk_level', 'UNKNOWN')
    workers     = vision_results.get('total_workers_seen', 0)
    date_str    = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Step 1: Log to database
    inspection_id = log_inspection(
        site_name, violations, risk_level, report_text, supervisor_email
    )
    logger.info(f'Inspection logged as ID {inspection_id}')

    # Step 2: Check repeat history
    repeat_count = get_repeat_violation_count(site_name)
    is_repeat    = repeat_count >= REPEAT_VIOLATION_THRESHOLD
    logger.info(f'Site repeat count: {repeat_count}. Repeat flag: {is_repeat}')

    # Step 3: Escalation
    urgency, action_items, needs_immediate = _determine_escalation(
        violations, repeat_count
    )

    # Step 4: Draft email
    subject = EMAIL_SUBJECT_TEMPLATE.format(
        site_name=site_name,
        date=datetime.now().strftime('%Y-%m-%d'),
        violation_count=len(violations),
    )
    body = EMAIL_BODY_TEMPLATE.format(
        supervisor_name=supervisor_name or 'Site Supervisor',
        site_name=site_name,
        date=date_str,
        violation_count=len(violations),
        risk_level=risk_level,
        workers_at_risk=workers,
        urgent_warning=URGENT_WARNING if needs_immediate else '',
        action_items=action_items,
    )

    return {
        'email_subject':  subject,
        'email_body':     body,
        'urgency':        urgency,
        'is_repeat_site': is_repeat,
        'repeat_count':   repeat_count,
        'inspection_id':  inspection_id,
        'site_history':   get_site_history(site_name, limit=10),
    }