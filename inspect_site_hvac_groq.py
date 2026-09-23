"""
Solar Site Inspection Pipeline — HVAC Groq API Version
=======================================================
Sends images of HVAC equipment to a Groq model using LangChain's init_chat_model.

Usage:
    ~/trt-export/bin/python inspect_site_hvac_groq.py sorted_data/HVAC
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
GROQ_MODEL    = "groq:qwen/qwen3.8-27b"
GROQ_API_KEY  = ""
MAX_IMAGES    = 4          # HVAC units may need multiple angles
JPEG_QUALITY  = 70
MAX_LONG_SIDE = 1280

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
SCHEMAS = {
    "hvac": {
        "description": "HVAC (heating, ventilation, and air conditioning) unit",
        "system_instruction": (
            "You are an expert HVAC inspector analysing site photos for a solar "
            "installation company. You will be shown one or more photos of the same "
            "HVAC unit taken from different angles.\n\n"
            "IMPORTANT RULES:\n"
            "1. For each field, return ONLY the single chosen value.\n"
            "2. unit_type: identify whether the unit is a heat pump, central AC, "
            "mini-split, packaged unit, or other. Look for labels on the cabinet.\n"
            "3. Read the manufacturer nameplate/data plate carefully — it is a silver "
            "or white sticker on the side or back panel of the unit. "
            "The plate has clearly labelled rows. Extract the following fields:\n"
            "   - make: brand name. Carrier units have model numbers starting with '24'. "
            "Other brands include Trane, Lennox, Rheem, Goodman, York.\n"
            "   - model: the full MODEL number as printed.\n"
            "   - serial_number: labelled 'SERIAL' on the plate .\n"
            "   - refrigerant_type: labelled 'DEVICE' or 'REFRIGERANT'.\n"
            "   - factory_charge_lbs: the refrigerant factory charge in LBS as printed."
            "Return numeric value only.\n"
            "   - voltage_phase: look for 'POWER SUPPLY' row — format is 'VOLTS AC / PH / HZ' "
            "   - permissible_voltage_min: the MIN voltage at unit.\n"
            "   - permissible_voltage_max: the MAX voltage at unit.\n"
            "   - compressor_rla: look for 'COMPRESSOR' section, 'RLA' value "
            "(Running Load Amps). Return numeric value only.\n"
            "   - fan_motor_fla: look for 'FAN MOTOR' section, 'FLA' value "
            "(Full Load Amps). Return numeric value only.\n"
            "   - minimum_circuit_ampacity: labelled 'MINIMUM CIRCUIT AMPS' or 'MCA'."
            " Return numeric value only.\n"
            "   - year_manufactured: decode from serial number — for Carrier, the first "
            "two digits of the serial are the year (e.g. serial '2418...' → year 2024). "
            "Otherwise use any explicit year label.\n"
            "4. location: where the unit is physically installed "
            "(e.g. 'Exterior/Backyard', 'Rooftop', 'Garage', 'Interior/Closet').\n"
            "5. condition_notes: assess visible physical condition — look for rust, "
            "bent or damaged fins, debris buildup, refrigerant oil stains, damaged wiring, "
            "or missing access panels.\n"
            "6. disconnect_box_present: look for a small metal box mounted near the unit "
            "with a pull-out fuse or breaker handle.\n"
            "7. Set confidence based on nameplate legibility: high if all fields are "
            "clearly readable, medium if partially obscured, low if nameplate is missing "
            "or unreadable."
        ),
        "response_schema": {
            "component": "Always 'hvac'",
            "unit_type": "ENUM: heat_pump | central_ac | mini_split | packaged_unit | other | Unknown",
            "location": "e.g. 'Exterior/Backyard' or 'Rooftop' or 'Unknown'",
            "make": "Brand name e.g. 'Carrier' or 'Unknown'",
            "model": "Full model number or 'Unknown'",
            "serial_number": "As printed or 'Unknown'",
            "refrigerant_type": "Printed Code or 'Unknown'",
            "factory_charge_lbs": "Numeric value only or 'Unknown'",
            "voltage_phase": "e.g. '208-230V/1Ph/60Hz' or 'Unknown'",
            "permissible_voltage_min": "Numeric value only or 'Unknown'",
            "permissible_voltage_max": "Numeric value only or 'Unknown'",
            "compressor_rla": "Numeric value only or 'Unknown'",
            "fan_motor_fla": "Numeric value only or 'Unknown'",
            "minimum_circuit_ampacity": "Numeric value only or 'Unknown'",
            "year_manufactured": "4-digit year or 'Unknown'",
            "disconnect_box_present": "ENUM: yes | no | Unknown",
            "condition_notes": "Brief free text describing visible physical condition",
            "confidence": "ENUM: high | medium | low"
        }
    }
}


def detect_component_type(directory: str) -> str | None:
    name = os.path.basename(directory).lower()
    if "hvac" in name or "ac" in name or "heat_pump" in name or "aircon" in name:
        return "hvac"
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
    print(f"  HVAC Site Inspector — Groq Version")
    print(f"  Directory : {directory}")
    print(f"{'='*60}\n")

    component_type = detect_component_type(directory)
    if component_type is None:
        print(f"❌ Unknown directory type: '{os.path.basename(directory)}'")
        print("   Expected a directory named 'HVAC', 'AC', 'heat_pump', or similar.")
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
        f"Analyse these {len(image_paths)} photo(s) of a residential {schema_info['description']}.\n\n"
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
        output_json = os.path.join("outputs", f"{dir_name}_groq.json")
        os.makedirs("outputs", exist_ok=True)

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
        # Default to the known HVAC directory when run without args
        default_dir = "sorted_data/HVAC"
        print(f"No directory specified — defaulting to: {default_dir}")
        run(default_dir)
    else:
        run(sys.argv[1])
