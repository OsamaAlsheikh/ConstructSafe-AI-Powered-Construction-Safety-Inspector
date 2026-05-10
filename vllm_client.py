# vllm_client.py  [root/]
# The bridge between the HF Space and AMD Cloud.
# During local testing with mock_mode.py, these functions
# are replaced before they ever make a real HTTP call.

import base64
import io
import logging
from PIL import Image
from openai import OpenAI
from config import (
    VLLM_VISION_URL, VLLM_REPORT_URL,
    VISION_MODEL_NAME, REPORT_MODEL_NAME,
    VISION_MAX_TOKENS, REPORT_MAX_TOKENS,
    VISION_TEMPERATURE, REPORT_TEMPERATURE,
)

logger = logging.getLogger(__name__)

# Two OpenAI clients pointing at AMD Cloud vLLM ports
vision_client = OpenAI(
    base_url=f'{VLLM_VISION_URL}/v1',
    api_key='not-needed',
)
report_client = OpenAI(
    base_url=f'{VLLM_REPORT_URL}/v1',
    api_key='not-needed',
)


def _pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def call_vision(system_prompt: str, user_text: str,
                image: Image.Image) -> str:
    image_b64 = _pil_to_base64(image)
    response = vision_client.chat.completions.create(
        model=VISION_MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': [
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                {'type': 'text', 'text': user_text},
            ]},
        ],
        max_tokens=VISION_MAX_TOKENS,
        temperature=VISION_TEMPERATURE,
    )
    return response.choices[0].message.content


def call_report(system_prompt: str, user_prompt: str) -> str:
    response = report_client.chat.completions.create(
        model=REPORT_MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt},
        ],
        max_tokens=REPORT_MAX_TOKENS,
        temperature=REPORT_TEMPERATURE,
    )
    return response.choices[0].message.content


def check_vllm_health() -> dict:
    result = {
        'healthy': False, 'vision_loaded': False, 'report_loaded': False,
        'models': [], 'vision_url': VLLM_VISION_URL,
        'report_url': VLLM_REPORT_URL, 'error': None,
    }
    try:
        vision_models = [m.id for m in vision_client.models.list().data]
        report_models = [m.id for m in report_client.models.list().data]
        result['models']        = vision_models + report_models
        result['vision_loaded'] = VISION_MODEL_NAME in vision_models
        result['report_loaded'] = REPORT_MODEL_NAME in report_models
        result['healthy']       = result['vision_loaded'] and result['report_loaded']
    except Exception as e:
        result['error'] = str(e)
    return result