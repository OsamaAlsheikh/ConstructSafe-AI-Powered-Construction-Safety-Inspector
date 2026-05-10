# """ # Track 3: Vision & Multimodal AI
# # Loads Qwen2.5-VL-7B-Instruct, extracts frames from video,
# # sends each frame to the model, returns structured violation JSON.
# """


# import json, re, cv2, torch, logging
# from pathlib import Path
# from PIL import Image
# from typing import Union
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from config import (VISION_MODEL_ID, USE_4BIT, FRAME_SAMPLE_INTERVAL_SECONDS,
#                     MAX_FRAMES, VISION_SYSTEM_PROMPT, VISION_USER_PROMPT, VIOLATION_LOOKUP)

# logger = logging.getLogger(__name__)
# _vision_model = None        # Singleton — loaded once on first call
# _vision_processor = None

# def _load_vision_model():
#     """Load Qwen2.5-VL with optional 4-bit quantisation. Cached after first load."""
#     global _vision_model, _vision_processor
#     if _vision_model is not None:
#         return _vision_model, _vision_processor
#     if USE_4BIT:
#         from transformers import BitsAndBytesConfig
#         qcfg = BitsAndBytesConfig(load_in_4bit=True,
#                                    bnb_4bit_compute_dtype=torch.bfloat16,
#                                    bnb_4bit_use_double_quant=True,
#                                    bnb_4bit_quant_type='nf4')
#     else:
#         qcfg = None
#     _vision_model = Qwen2VLForConditionalGeneration.from_pretrained(
#         VISION_MODEL_ID, torch_dtype=torch.bfloat16,
#         quantization_config=qcfg, device_map='auto').eval()
#     _vision_processor = AutoProcessor.from_pretrained(
#         VISION_MODEL_ID,
#         min_pixels=256*28*28,   # Min visual tokens
#         max_pixels=1280*28*28)  # Max visual tokens (controls VRAM)
#     return _vision_model, _vision_processor

# def extract_frames_from_video(video_path: str) -> list:
#     """Extract one PIL Image every FRAME_SAMPLE_INTERVAL_SECONDS, capped at MAX_FRAMES."""
#     cap = cv2.VideoCapture(str(video_path))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     interval = int(fps * FRAME_SAMPLE_INTERVAL_SECONDS)
#     frames, idx = [], 0
#     while len(frames) < MAX_FRAMES:
#         ret, frame = cap.read()
#         if not ret: break
#         if idx % interval == 0:
#             frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#         idx += 1
#     cap.release()
#     return frames

# def _analyse_single_frame(model, processor, image, frame_number) -> dict:
#     """Send one PIL Image to Qwen2.5-VL. Returns parsed JSON dict or empty violations on failure."""
#     messages = [
#         {'role': 'system', 'content': VISION_SYSTEM_PROMPT},
#         {'role': 'user',   'content': [{'type':'image','image':image},
#                                         {'type':'text','text':VISION_USER_PROMPT}]},
#     ]
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     img_inputs, vid_inputs = process_vision_info(messages)
#     inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
#                        padding=True, return_tensors='pt').to(model.device)
#     with torch.no_grad():
#         ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
#     out = processor.batch_decode(
#         [ids[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]
#     out = re.sub(r'^```json\s*', '', out.strip())
#     out = re.sub(r'\s*```$', '', out)
#     try:
#         return json.loads(out)
#     except json.JSONDecodeError:
#         logger.warning(f'Frame {frame_number}: JSON parse failed')
#         return {'violations':[], 'worker_count':0, 'overall_risk_level':'UNKNOWN', 'scene_description':'parse error'}

# def _deduplicate(violations: list) -> list:
#     """Keep highest-confidence detection per violation code across all frames."""
#     best = {}
#     for v in violations:
#         code = v.get('code','UNKNOWN')
#         if code not in best or v.get('confidence',0) > best[code].get('confidence',0):
#             best[code] = v
#     return list(best.values())

# def _enrich(violations: list) -> list:
#     """Add regulation, severity, violation_name from VIOLATION_LOOKUP to each detection."""
#     return [{**v, **VIOLATION_LOOKUP.get(v.get('code',''), {})} for v in violations]

# def analyse_media(media_path) -> dict:
#     """
#     MAIN ENTRY POINT. Accepts video or image, returns:
#     { violations, total_workers_seen, overall_risk_level,
#       scene_description, frames_analysed, annotated_frames }
#     """
#     path = Path(media_path)
#     model, processor = _load_vision_model()
#     if path.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:
#         frames = [Image.open(path).convert('RGB')]
#     else:
#         frames = extract_frames_from_video(path)
#     all_v, risks, descs, workers = [], [], [], 0
#     for i, frame in enumerate(frames):
#         r = _analyse_single_frame(model, processor, frame, i)
#         all_v.extend(r.get('violations', []))
#         risks.append(r.get('overall_risk_level','LOW'))
#         descs.append(r.get('scene_description',''))
#         workers = max(workers, r.get('worker_count',0))
#     order = {'LOW':0,'MEDIUM':1,'HIGH':2,'CRITICAL':3,'UNKNOWN':1}
#     return {
#         'violations': _enrich(_deduplicate(all_v)),
#         'total_workers_seen': workers,
#         'overall_risk_level': max(risks, key=lambda r: order.get(r,1)) if risks else 'LOW',
#         'scene_description': descs[0] if descs else 'Construction site',
#         'frames_analysed': len(frames),
#         'annotated_frames': frames,
#     }

# vision_agent.py
import json
import re
import cv2
import logging
from pathlib import Path
from PIL import Image
from typing import Union
from vllm_client import call_vision
from config import (
    FRAME_SAMPLE_INTERVAL_SECONDS, MAX_FRAMES,
    VISION_SYSTEM_PROMPT, VISION_USER_PROMPT,
    VIOLATION_LOOKUP,
)

logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path: str) -> list:
    """Extract one PIL Image every FRAME_SAMPLE_INTERVAL_SECONDS. CPU only."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video_path}')
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = int(fps * FRAME_SAMPLE_INTERVAL_SECONDS)
    frames, idx = [], 0
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    logger.info(f'Extracted {len(frames)} frames (FPS={fps:.1f})')
    return frames


def _analyse_single_frame(image: Image.Image, frame_num: int) -> dict:
    """Send one frame to AMD Cloud vLLM via HTTP. Parse JSON response."""
    raw = call_vision(VISION_SYSTEM_PROMPT, VISION_USER_PROMPT, image)
    clean = re.sub(r'^```json\s*', '', raw.strip())
    clean = re.sub(r'\s*```$', '', clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(f'Frame {frame_num}: JSON parse failed. Raw: {raw[:200]}')
        return {
            'violations': [], 'worker_count': 0,
            'overall_risk_level': 'UNKNOWN',
            'scene_description': 'Frame parse error',
        }


def _deduplicate(violations: list) -> list:
    """Keep highest-confidence detection per violation code."""
    best = {}
    for v in violations:
        code = v.get('code', 'UNKNOWN')
        if code not in best or v.get('confidence', 0) > best[code].get('confidence', 0):
            best[code] = v
    return list(best.values())


def _enrich(violations: list) -> list:
    """Add OSHA metadata from VIOLATION_LOOKUP to each detection."""
    return [{**v, **VIOLATION_LOOKUP.get(v.get('code', ''), {})}
            for v in violations]


def analyse_media(media_path: Union[str, Path]) -> dict:
    """
    Main entry point. Accepts video or image file.
    Calls AMD Cloud vLLM for each frame via HTTP.
    Returns: {violations, total_workers_seen, overall_risk_level,
              scene_description, frames_analysed, annotated_frames}
    """
    path = Path(media_path)
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    if path.suffix.lower() in image_exts:
        frames = [Image.open(path).convert('RGB')]
    else:
        frames = extract_frames_from_video(path)

    all_violations, risks, descs, workers = [], [], [], 0
    for i, frame in enumerate(frames):
        logger.info(f'Analysing frame {i+1}/{len(frames)} via AMD Cloud...')
        result = _analyse_single_frame(frame, i)
        all_violations.extend(result.get('violations', []))
        risks.append(result.get('overall_risk_level', 'LOW'))
        descs.append(result.get('scene_description', ''))
        workers = max(workers, result.get('worker_count', 0))

    order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3, 'UNKNOWN': 1}
    return {
        'violations':         _enrich(_deduplicate(all_violations)),
        'total_workers_seen': workers,
        'overall_risk_level': max(risks, key=lambda r: order.get(r, 1)) if risks else 'LOW',
        'scene_description':  descs[0] if descs else 'Construction site',
        'frames_analysed':    len(frames),
        'annotated_frames':   frames,
    }