"""
Solar Site Inspection Pipeline — Groq API Version
=================================================
Sends images to a Groq model using LangChain's init_chat_model.

Usage:
    ~/trt-export/bin/python inspect_site_groq.py sorted_data/Sub_panel_2
"""

import sys
import os
import json
import glob
import re
import base64
import imageio.v3 as iio
import numpy as np
import cv2
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_MODEL = "groq:qwen/qwen3.8-27b"
GROQ_API_KEY = ""
MAX_IMAGES = 2
JPEG_QUALITY = 70
MAX_LONG_SIDE = 1280

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
SCHEMAS = {
    "sub_panel": {
        "description": "electrical SUB PANEL",
        "system_instruction": (
            "You are an expert electrical inspector analysing site photos for a solar "
            "installation company. You will be shown multiple photos of the same electrical "
            "component taken from different angles.\n\n"
            "IMPORTANT RULES:\n"
            "1. For each field, return ONLY the single chosen value.\n"
            "2. panel_type: look at where the main cable conduit or service entry enters the enclosure — top = Top_fed, bottom = Bottom_fed.\n"
            "3. main_breaker_rating_amps: ONLY set this to a number if there is a dedicated MAIN BREAKER with a 'MAIN' label "
            "whose sole purpose is to shut off this entire panel. "
            "Most sub panels do NOT have one — they are main-lug panels. "
            "If all breakers have circuit labels (Dishwasher, Range, Dryer, etc.), they are load circuits — set this to 'NA'.\n"
            "4. feeder_breaker: look for any labels, wire tags, stickers, or handwriting near the incoming cables "
            "that identify the upstream source panel (e.g. 'From Sub Panel 1', 'House Sub', 'MSP'). "
            "Set source_panel to whatever is identified; if nothing is visible, set source_panel to 'Unknown'. "
            "Set rating_amps to the amperage shown on or near the feeder wire, otherwise 'Unknown'.\n"
            "5. Breaker inventory — this panel has a breaker directory with numbered positions and labels. Follow these steps:\n"
            "   STEP A: Read EVERY label in the panel directory on the door — there will be two columns (left and right) "
            "with numbered positions (1, 2, 3, 4... up to 16 or more). Read ALL of them carefully.\n"
            "   STEP B: For each numbered position that has a label AND a breaker installed, record it. "
            "A label at position N on the left column corresponds to the breaker in the left-side slot N. "
            "A label at position N on the right column corresponds to the breaker in the right-side slot N.\n"
            "   STEP C: Classify each breaker:\n"
            "   - 1-pole: single breaker body occupying ONE slot (120V). Typical labels: Dishwasher, Disposal, "
            "Kitchen plugs, Lights, Office, Washer, Garage outlet, Furnace.\n"
            "   - 2-pole: two breaker bodies occupying TWO adjacent slots (240V), both the same amperage, "
            "sharing one label. Typical labels: Range, Dryer, A/C, HVAC, Solar, EV Charger. "
            "In Bryant panels, 2-pole breakers often have red/green or colored indicators on both bodies.\n"
            "   STEP D: Every breaker with a label must appear in either breakers_1_pole or breakers_2_pole. "
            "Do NOT assign any circuit-labelled breaker to main_breaker_rating_amps.\n"
            "   Set count = number of entries in each list.\n"
            "6. available_breaker_slots: count positions in the directory that are empty (no breaker installed). "
            "If all slots are filled, set to 0."
        ),
        "response_schema": {
            "component": "Always 'sub_panel'",
            "location": "ENUM: Exterior or Interior or Unknown",
            "panel_type": "ENUM: Top_fed or Bottom_fed or Unknown",
            "bus_bar_rating_amps": "Number only e.g. 125, or 'NA', or 'Unknown'",
            "main_breaker_rating_amps": "Number only e.g. 100, or 'NA', or 'Unknown'",
            "feeder_breaker": {
                "source_panel": "e.g. 'Sub_panel_1' or 'MSP' or 'Unknown'",
                "rating_amps": "e.g. '100' or 'Unknown'"
            },
            "available_breaker_slots": "Integer count or 'Unknown'",
            "make_model": "Brand and model, e.g. 'Bryant' or 'Unknown'",
            "breakers_1_pole": {
                "count": "total number of 1-pole breakers",
                "breakers": [{"label": "string", "rating_amps": "number"}]
            },
            "breakers_2_pole": {
                "count": "total number of 2-pole breakers",
                "breakers": [{"label": "string", "rating_amps": "number"}]
            },
            "condition_notes": "Brief free text describing visible condition, or 'None'",
            "confidence": "ENUM: high, medium, low"
        }
    }
}

def detect_component_type(directory: str) -> str | None:
    name = os.path.basename(directory).lower()
    if "sub_panel" in name or "subpanel" in name:
        return "sub_panel"
    return None

def collect_images(directory: str) -> list[str]:
    exts = ["*.HEIC", "*.heic", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(set(paths))

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
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Strip <think> blocks if present from models like Qwen
    if "<think>" in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
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

def run(directory: str):
    print(f"\n{'='*60}")
    print(f"  Solar Site Inspector — Groq Version")
    print(f"  Directory : {directory}")
    print(f"{'='*60}\n")

    component_type = detect_component_type(directory)
    if component_type is None:
        print(f"❌ Unknown directory type: '{os.path.basename(directory)}'")
        sys.exit(1)
    
    schema_info = SCHEMAS[component_type]
    print(f"✅ Component type detected: {component_type}")

    image_paths = collect_images(directory)
    if not image_paths:
        print(f"❌ No images found in {directory}")
        sys.exit(1)
    print(f"✅ Found {len(image_paths)} image(s)")

    if len(image_paths) > MAX_IMAGES:
        step = len(image_paths) / MAX_IMAGES
        image_paths = [image_paths[int(i * step)] for i in range(MAX_IMAGES)]
        print(f"   ↳ Sampled {MAX_IMAGES} images (evenly spaced) to fit context window")

    print("📷 Loading and encoding images...")
    
    # Using LangChain's HumanMessage content array for images
    human_content = []
    
    for path in image_paths:
        b64 = load_and_encode_image(path)
        if b64:
            human_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            print(f"   ✓ {os.path.basename(path)}  ({len(b64)//1024} KB encoded)")
        else:
            print(f"   ✗ {os.path.basename(path)}  (skipped)")

    if not human_content:
        print("❌ All images failed to load.")
        sys.exit(1)

    prompt_text = (
        f"Analyse these {len(image_paths)} photos of a residential {schema_info['description']}.\n\n"
        f"Return a single JSON object matching this exact schema structure:\n"
        f"{json.dumps(schema_info['response_schema'], indent=2)}\n\n"
        f"Return ONLY the raw JSON output without markdown fences."
    )
    human_content.append({"type": "text", "text": prompt_text})

    print(f"\n🤖 Calling {GROQ_MODEL} via LangChain...")
    try:
        model = init_chat_model(GROQ_MODEL, api_key=GROQ_API_KEY)
        
        messages = [
            SystemMessage(content=schema_info["system_instruction"]),
            HumanMessage(content=human_content)
        ]
        
        response = model.invoke(messages)
        raw_text = response.content
        
        result = extract_json(raw_text)
        
        dir_name = os.path.basename(directory.rstrip("/"))
        output_json = os.path.join(directory, f"{dir_name}_groq.json")
        
        if result:
            with open(output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n✅ Extraction successful! Output saved to:\n   {output_json}")
            print("\n📋 Extracted data:")
            print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ JSON parsing failed.")
            print("\n--- RAW RESPONSE ---")
            print(raw_text)
            
    except Exception as e:
        print(f"❌ Error calling Groq API: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_site_groq.py <directory>")
        sys.exit(1)
    run(sys.argv[1])
