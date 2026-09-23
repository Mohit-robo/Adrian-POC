"""
Solar Site Inspection Pipeline — Solar System
=================================================
Sends all images from Solar_panel_provider and Solar_elec_system 
to the local Qwen3-VL-2B model and extracts solar module and inverter details.

Usage:
    ~/trt-export/bin/python inspect_site_solar.py
"""

import sys
import os
import json
import base64
import re
import glob
import imageio.v3 as iio
import numpy as np
import cv2
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VLM_BASE_URL   = "http://192.168.128.141:10800/v1"
VLM_MODEL      = "/models/Qwen3-VL-2B-Instruct"
VLM_API_KEY    = "EMPTY"
MAX_IMAGES     = 10
JPEG_QUALITY   = 70
MAX_LONG_SIDE  = 1280

# ─── SCHEMA ───────────────────────────────────────────────────────────────────
SCHEMA_INFO = {
    "description": "solar modules and electrical inverter system",
    "system_instruction": (
        "You are an expert solar site inspector analysing site photos. "
        "You will be shown photos of the existing solar panels and the electrical/inverter system.\n\n"
        "IMPORTANT RULES:\n"
        "1. For each field, return ONLY the single chosen value.\n"
        "2. solar_module_make_model: Look for the nameplate or sticker on the back of the solar panels.\n"
        "3. inverter_location: Look at where the inverter is mounted — 'Exterior' if outside, 'Interior' if inside.\n"
        "4. inverter_make_model: Look for the nameplate or sticker on the inverter.\n"
        "5. inverter_breaker_location: Identify any breaker labels or AC disconnects associated with the solar/inverter system.\n"
        "6. Return ONLY valid JSON, no markdown fences."
    ),
    "response_schema": {
        "solar_module_make_model": "string  or 'Unknown'",
        "inverter_location": "ENUM: Exterior or Interior or Unknown",
        "inverter_make_model": "string or 'Unknown'",
        "inverter_breaker_location": "string or 'Unknown'"
    }
}

def load_and_encode_image(path: str) -> str | None:
    try:
        img = iio.imread(path)
    except Exception as e:
        print(f"  ⚠ Could not load {path}: {e}")
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    h, w = img.shape[:2]
    if max(h, w) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not success:
        return None

    return base64.b64encode(buf.tobytes()).decode("utf-8")

def collect_images(directories: list[str]) -> list[str]:
    exts = ["*.HEIC", "*.heic", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    paths = []
    for d in directories:
        for ext in exts:
            paths.extend(glob.glob(os.path.join(d, ext)))
    return sorted(set(paths))

def extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None

def run():
    dirs = ["sorted_data/Solar_panel_provider", "sorted_data/Solar_elec_system"]
    
    print(f"\n{'='*60}")
    print(f"  Solar Site Inspector — Solar System")
    print(f"  Directories : {', '.join(dirs)}")
    print(f"{'='*60}\n")

    image_paths = collect_images(dirs)
    if not image_paths:
        print(f"❌ No images found in specified directories.")
        sys.exit(1)
    
    print(f"✅ Found {len(image_paths)} image(s)")

    if len(image_paths) > MAX_IMAGES:
        step = len(image_paths) / MAX_IMAGES
        image_paths = [image_paths[int(i * step)] for i in range(MAX_IMAGES)]
        print(f"   ↳ Sampled {MAX_IMAGES} images (evenly spaced) to fit context window")

    print("📷 Loading and encoding images...")
    encoded_images = []
    for path in image_paths:
        b64 = load_and_encode_image(path)
        if b64:
            encoded_images.append(b64)
            print(f"   ✓ {os.path.basename(path)}  ({len(b64)//1024} KB encoded)")

    if not encoded_images:
        print("❌ All images failed to load.")
        sys.exit(1)

    system_prompt = SCHEMA_INFO["system_instruction"]
    user_text = (
        f"Analyse these {len(encoded_images)} photos of a residential {SCHEMA_INFO['description']}.\n\n"
        f"Return a single JSON object matching this exact schema structure:\n"
        f"{json.dumps(SCHEMA_INFO['response_schema'], indent=2)}\n\n"
        f"Return ONLY the raw JSON output without markdown fences."
    )

    content = []
    for b64 in encoded_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": content}
    ]

    print(f"\n🤖 Calling Qwen3-VL-2B ({len(encoded_images)} images)...")
    client = OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY)
    
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    
    raw_response = response.choices[0].message.content
    result = extract_json(raw_response)
    
    output_json = "solar_system_inspection.json"
    
    if result:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Extraction successful! Output saved to:\n   {output_json}")
        print("\n📋 Extracted data:")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ JSON parsing failed.")
        print("\n--- RAW RESPONSE ---")
        print(raw_response)

if __name__ == "__main__":
    run()
