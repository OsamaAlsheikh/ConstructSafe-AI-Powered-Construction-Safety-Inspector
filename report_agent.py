# # Track 2: Fine-Tuned OSHA Report Generator
# # Loads Qwen2.5-7B (fine-tuned on OSHA data via QLoRA) and generates
# # a formal, citation-accurate safety inspection report.

# import torch, logging
# from datetime import datetime
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from config import REPORT_MODEL_ID, USE_4BIT, REPORT_SYSTEM_PROMPT

# logger = logging.getLogger(__name__)
# _report_model = None     # Singleton — loaded once
# _report_tokenizer = None

# def _load_report_model():
#     global _report_model, _report_tokenizer
#     if _report_model is not None:
#         return _report_model, _report_tokenizer
#     qcfg = BitsAndBytesConfig(load_in_4bit=True,
#                                bnb_4bit_compute_dtype=torch.bfloat16,
#                                bnb_4bit_use_double_quant=True,
#                                bnb_4bit_quant_type='nf4') if USE_4BIT else None
#     _report_model = AutoModelForCausalLM.from_pretrained(
#         REPORT_MODEL_ID, torch_dtype=torch.bfloat16,
#         quantization_config=qcfg, device_map='auto').eval()
#     _report_tokenizer = AutoTokenizer.from_pretrained(REPORT_MODEL_ID)
#     return _report_model, _report_tokenizer

# def _format_violations_for_prompt(violations: list) -> str:
#     """Convert violation list to readable text for the report prompt."""
#     if not violations: return 'No violations detected. Site appears compliant.'
#     lines = []
#     for i, v in enumerate(violations, 1):
#         lines.append(
#             f'Violation {i}:\n'
#             f"  Code: {v.get('code','N/A')}\n"
#             f"  Type: {v.get('name', v.get('code',''))}\n"
#             f"  Regulation: {v.get('regulation','N/A')}\n"
#             f"  Severity: {v.get('severity','Serious')}\n"
#             f"  Observation: {v.get('observation','N/A')}\n"
#             f"  Location: {v.get('location_description','N/A')}\n"
#             f"  Confidence: {v.get('confidence',0):.0%}"
#         )
#     return '\n\n'.join(lines)

# def generate_report(vision_results: dict, site_name: str,
#                     is_repeat_offender: bool = False) -> str:
#     """
#     MAIN ENTRY POINT. Takes vision_agent output and returns a
#     full OSHA-formatted inspection report as a plain-text string.
#     """
#     model, tokenizer = _load_report_model()
#     violations = vision_results.get('violations', [])
#     risk  = vision_results.get('overall_risk_level', 'UNKNOWN')
#     workers = vision_results.get('total_workers_seen', 0)
#     scene = vision_results.get('scene_description', 'Construction site')
#     date  = datetime.now().strftime('%B %d, %Y at %I:%M %p')

#     user_prompt = (
#         f'Generate a formal OSHA inspection report.\n'
#         f'SITE: {site_name}\nDATE: {date}\nSCENE: {scene}\n'
#         f'WORKERS: {workers}\nRISK: {risk}\n'
#         f"REPEAT OFFENDER: {'YES' if is_repeat_offender else 'No'}\n"
#         f'VIOLATIONS:\n{_format_violations_for_prompt(violations)}'
#     )
#     messages = [{'role':'system','content':REPORT_SYSTEM_PROMPT},
#                 {'role':'user',  'content':user_prompt}]
#     text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(text, return_tensors='pt').to(model.device)
#     with torch.no_grad():
#         out = model.generate(**inputs, max_new_tokens=1024,
#                              temperature=0.2, do_sample=True,
#                              repetition_penalty=1.1)
#     report = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#     # Safety fallback: add header if model skipped it
#     if 'CONSTRUCTION SITE SAFETY' not in report[:100]:
#         report = f'CONSTRUCTION SITE SAFETY INSPECTION REPORT\nSite: {site_name}\nDate: {date}\n\n' + report
#     return report


# report_agent.py
import logging
from datetime import datetime
from vllm_client import call_report
from config import REPORT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _format_violations_for_prompt(violations: list) -> str:
    if not violations:
        return 'No violations detected. Site appears compliant.'
    lines = []
    for i, v in enumerate(violations, 1):
        lines.append(
            f'Violation {i}:\n'
            f"  Code: {v.get('code', 'N/A')}\n"
            f"  Type: {v.get('name', v.get('code', ''))}\n"
            f"  Regulation: {v.get('regulation', 'N/A')}\n"
            f"  Severity: {v.get('severity', 'Serious')}\n"
            f"  Observation: {v.get('observation', 'N/A')}\n"
            f"  Location: {v.get('location_description', 'N/A')}\n"
            f"  Confidence: {v.get('confidence', 0):.0%}"
        )
    return '\n\n'.join(lines)


def generate_report(vision_results: dict, site_name: str,
                    is_repeat_offender: bool = False) -> str:
    """
    Main entry point. Calls AMD Cloud vLLM via one HTTP request.
    Returns complete OSHA report as plain-text string.
    """
    violations = vision_results.get('violations', [])
    risk       = vision_results.get('overall_risk_level', 'UNKNOWN')
    workers    = vision_results.get('total_workers_seen', 0)
    scene      = vision_results.get('scene_description', 'Construction site')
    date_str   = datetime.now().strftime('%B %d, %Y at %I:%M %p')

    user_prompt = (
        f'Generate a formal OSHA construction safety inspection report.\n'
        f'SITE: {site_name}\n'
        f'DATE: {date_str}\n'
        f'SCENE: {scene}\n'
        f'WORKERS OBSERVED: {workers}\n'
        f'OVERALL RISK LEVEL: {risk}\n'
        f"REPEAT OFFENDER: {'YES' if is_repeat_offender else 'No'}\n"
        f'\nDETECTED VIOLATIONS:\n'
        f'{_format_violations_for_prompt(violations)}'
    )

    logger.info('Calling AMD Cloud vLLM for report generation...')
    report = call_report(REPORT_SYSTEM_PROMPT, user_prompt)

    if 'CONSTRUCTION SITE SAFETY' not in report[:100]:
        report = (
            f'CONSTRUCTION SITE SAFETY INSPECTION REPORT\n'
            f'{"="*44}\n'
            f'Site: {site_name}\nDate: {date_str}\n\n'
        ) + report
    return report