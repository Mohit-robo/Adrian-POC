"""
Garage Measurement Extractor — Groq API Version
=================================================
Uses qwen/qwen3.8-27b on Groq, batching 2 images at a time (7000 ITPM limit).
Each image is sent individually for maximum focus.
Results are consolidated into one final JSON using a text-only call.

Usage:
    ~/trt-export/bin/python inspect_garage_measurements_groq.py
"""

import os
import json
import base64
import re
import glob
import time
import imageio.v3 as iio
import cv2
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_MODEL       = "groq:qwen/qwen3.8-27b"
GROQ_API_KEY     = ""
MAX_IMAGES_BATCH = 2
JPEG_QUALITY     = 85
MAX_LONG_SIDE    = 1600
SLEEP_TIME       = 65   # seconds — resets 7000 ITPM limit

IMAGE_DIR = "sorted_data/garage_measurement_images"

SYSTEM_PROMPT = (
    "You are an expert at reading measuring tapes in site photos. "
    "Your task is to identify and extract every measurement visible in the image.\n\n"
    "CRITICAL RULES:\n"
    "1. Identify WHAT is being measured — describe the physical edge, surface, or span "
    "in plain words (e.g. 'rafter/truss depth', 'rafter spacing', 'return wall height left side', "
    "'return wall height right side', 'garage door opening width').\n"
    "2. Read the tape measure value exactly where the tape terminates — read inch and fraction markings carefully.\n"
    "3. Report ALL measurements in INCHES only.\n"
    "4. If the tape shows feet+inches, convert: e.g. 7'2\" = 86 inches.\n"
    "5. If multiple tapes are visible, extract ALL of them.\n"
    "6. Be precise — read to the nearest 1/4 inch if visible.\n"
    "7. Return ONLY valid JSON with no markdown fences."
)

RESPONSE_SCHEMA = {
    "measurements": [
        {
            "description": "What is being measured (e.g. 'rafter depth', 'return wall height left')",
            "tape_reading": "Raw value as seen on tape (e.g. '2 ft 3 in' or '39')",
            "value_inches": "Numeric total in inches (e.g. 27 or 39)"
        }
    ]
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
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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

def call_groq(model, images_b64: list[str], fname_hint: str) -> dict | None:
    user_text = (
        f"Examine this photo carefully ({fname_hint}). It shows measuring tapes placed against "
        f"surfaces in a garage. Read EVERY tape measure value visible.\n\n"
        f"For each measurement:\n"
        f"- Describe precisely WHAT is being measured.\n"
        f"- Read the tape value exactly as shown.\n"
        f"- Convert to total inches.\n\n"
        f"Return a single JSON object:\n"
        f"{json.dumps(RESPONSE_SCHEMA, indent=2)}\n\n"
        f"Return ONLY the raw JSON. No markdown."
    )
    content = []
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": user_text})

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content)
    ]
    response = model.invoke(messages)
    return extract_json(response.content)

def run():
    print(f"\n{'='*60}")
    print(f"  Garage Measurement Extractor — Groq API")
    print(f"  Directory : {IMAGE_DIR}")
    print(f"{'='*60}\n")

    image_paths = collect_images(IMAGE_DIR)
    if not image_paths:
        print(f"❌ No images found in {IMAGE_DIR}")
        return

    print(f"✅ Found {len(image_paths)} image(s). Processing individually (2 images max per API call)...\n")

    model = init_chat_model(GROQ_MODEL, api_key=GROQ_API_KEY)
    all_results = []

    for i, path in enumerate(image_paths):
        fname = os.path.basename(path)
        print(f"📷 [{i+1}/{len(image_paths)}] Processing: {fname}")

        b64 = load_and_encode_image(path)
        if not b64:
            print(f"   ❌ Encoding failed. Skipping.\n")
            continue
        print(f"   ✓ Encoded ({len(b64)//1024} KB)")

        try:
            result = call_groq(model, [b64], fname)
            if result and "measurements" in result:
                print(f"   ✅ Found {len(result['measurements'])} measurement(s):")
                for m in result["measurements"]:
                    print(f"      → {m.get('description','?')}: {m.get('tape_reading','?')} = {m.get('value_inches','?')} inches")
                all_results.append({"image": fname, "measurements": result["measurements"]})
            else:
                print(f"   ❌ JSON parsing failed.")
                all_results.append({"image": fname, "measurements": []})
        except Exception as e:
            print(f"   ❌ Groq API error: {e}")
            all_results.append({"image": fname, "measurements": []})

        print()

        # Sleep between calls to respect 7000 ITPM rate limit (except last image)
        if i < len(image_paths) - 1:
            print(f"   ⏳ Sleeping {SLEEP_TIME}s to reset rate limit...\n")
            time.sleep(SLEEP_TIME)

    # Save
    output_path = os.path.join(IMAGE_DIR, "garage_measurements_groq.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*60}")
    print("\n📋 FULL EXTRACTION SUMMARY:")
    for entry in all_results:
        print(f"\n📷 {entry['image']}:")
        if entry["measurements"]:
            for m in entry["measurements"]:
                print(f"   • {m.get('description','?')}")
                print(f"     Tape: {m.get('tape_reading','?')}  →  {m.get('value_inches','?')} inches")
        else:
            print("   (no measurements extracted)")

if __name__ == "__main__":
    run()
