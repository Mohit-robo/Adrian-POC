"""
Solar Site Inspection Pipeline — MSP (Main Service Panel)
==========================================================
Sends each MSP image individually to the local Qwen3-VL-2B model.
Extracts meter info, panel specs, and breaker inventory.

Usage:
    ~/trt-export/bin/python inspect_site_msp.py sorted_data/MSP
"""

import sys
import os
import json
import base64
import re
import glob
import imageio.v3 as iio
import cv2
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VLM_BASE_URL   = "http://192.168.128.178:10801/v1"
VLM_MODEL      = "qwen3-vl-2b"
VLM_API_KEY    = "EMPTY"
MAX_IMAGES     = 10
JPEG_QUALITY   = 85
MAX_LONG_SIDE  = 1280

# ─── SCHEMA ───────────────────────────────────────────────────────────────────
SCHEMA_INFO = {
    "description": "Main Service Panel (MSP) or Meter Disconnect",
    "system_instruction": (
        "You are an expert electrical inspector analysing site photos for a solar "
        "installation company. You will be shown multiple photos of the same "
        "Main Service Panel (MSP) or Meter Disconnect from different angles.\n\n"
        "IMPORTANT RULES:\n"
        "1. For each field, return ONLY the single chosen value.\n"
        "2. meter_number: Look for the meter serial number on the utility meter label.\n"
        "3. utility_name: Look for the utility logo or text on the meter or panel.\n"
        "4. meter_location: 'Exterior' if the meter is outside the building, 'Interior' otherwise.\n"
        "5. msp_location: 'Exterior' if the main panel is outside, 'Interior' otherwise.\n"
        "6. main_panel_type: Classify the panel — 'Meter_disconnect' if it is a combined meter+disconnect "
        "with no separate load breakers, 'Main_breaker_panel' if it contains a main breaker and load circuits, "
        "'Meter_only' if only the meter is visible.\n"
        "7. main_breaker_rating_amps: The amperage on the main breaker or service disconnect (e.g. '100', '200'). "
        "Set to 'NA' if no main breaker is present.\n"
        "8. bus_bar_rating_amps: Look for a rating label inside the panel on the bus bar (e.g. '125', '200'). "
        "Set to 'Unknown' if not visible.\n"
        "9. max_branch_breakers: The maximum number of branch breakers or spaces the panel supports. "
        "Look for a label like 'MAX 40 CIRCUITS'. Set to 'Unknown' if not visible.\n"
        "10. available_breaker_slots: Count empty/blank positions in the breaker directory. Set to 0 if full.\n"
        "11. make_model: Brand and model of the panel. Set to 'Unknown' if not visible.\n"
        "12. Breaker inventory:\n"
        "    - 1-pole: single breaker occupying ONE slot (120V).\n"
        "    - 2-pole: two breakers occupying TWO adjacent slots (240V), sharing one label.\n"
        "    - Do NOT list the main service disconnect as a load breaker.\n"
        "    - Record label and amperage for each breaker.\n"
        "13. Return ONLY valid JSON, no markdown fences."
    ),
    "response_schema": {
        "component": "Always 'MSP'",
        "meter_number": "string or 'Unknown'",
        "utility_name": "string or 'Unknown'",
        "meter_location": "ENUM: Exterior or Interior or Unknown",
        "msp_location": "ENUM: Exterior or Interior or Unknown",
        "main_panel_type": "ENUM: Meter_disconnect or Main_breaker_panel or Meter_only or Unknown",
        "main_breaker_rating_amps": "Number only e.g. 100, or 'NA', or 'Unknown'",
        "bus_bar_rating_amps": "Number only e.g. 200, or 'NA', or 'Unknown'",
        "max_branch_breakers": "Number or 'Unknown'",
        "available_breaker_slots": "Integer or 'Unknown'",
        "make_model": "string or 'Unknown'",
        "breakers_1_pole": {
            "count": "total number of 1-pole breakers",
            "breakers": [{"label": "string", "rating_amps": "number"}]
        },
        "breakers_2_pole": {
            "count": "total number of 2-pole breakers",
            "breakers": [{"label": "string", "rating_amps": "number"}]
        },
        "condition_notes": "Brief free text or 'None'",
        "confidence": "ENUM: high, medium, low"
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

def extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

def collect_images(directory: str) -> list[str]:
    exts = ["*.HEIC", "*.heic", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(set(paths))

def run(directory: str):
    print(f"\n{'='*60}")
    print(f"  Solar Site Inspector — MSP")
    print(f"  Directory : {directory}")
    print(f"{'='*60}\n")

    image_paths = collect_images(directory)
    if not image_paths:
        print(f"❌ No images found in {directory}")
        sys.exit(1)

    print(f"✅ Found {len(image_paths)} image(s)")

    if len(image_paths) > MAX_IMAGES:
        step = len(image_paths) / MAX_IMAGES
        image_paths = [image_paths[int(i * step)] for i in range(MAX_IMAGES)]
        print(f"   ↳ Sampled {MAX_IMAGES} images to fit context window")

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
        max_tokens=2048,
    )

    raw_response = response.choices[0].message.content
    result = extract_json(raw_response)

    dir_name = os.path.basename(directory.rstrip("/"))
    output_json = os.path.join(directory, f"{dir_name}.json")

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
    if len(sys.argv) < 2:
        print("Usage: python inspect_site_msp.py <directory>")
        sys.exit(1)
    run(sys.argv[1])
