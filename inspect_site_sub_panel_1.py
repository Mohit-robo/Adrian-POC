"""
Solar Site Inspection Pipeline
================================
Sends all images in a sorted_data/<component> directory to Qwen3-VL-2B
and extracts structured JSON per the hardcoded schema for that component type.

Usage:
    ~/trt-export/bin/python inspect_site.py <directory>

Example:
    ~/trt-export/bin/python inspect_site.py sorted_data/Sub_panel_1
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
VLM_API_KEY    = "EMPTY"          # vLLM doesn't need a real key
MAX_IMAGES     = 10               # cap to avoid context overflow (16k tokens)
JPEG_QUALITY   = 70               # lower = smaller payload
MAX_LONG_SIDE  = 1280             # resize large images before encoding

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
SCHEMAS = {
    "sub_panel": {
        "description": "electrical SUB PANEL",
        "template": {
            "component":                     "sub_panel",
            "location":                      "ENUM: Exterior or Interior or Unknown",
            "panel_type":                    "ENUM: Top_fed or Bottom_fed or Unknown — Top_fed means cables enter from the top of the panel; Bottom_fed means cables enter from the bottom",
            "bus_bar_rating_amps":           "number only, e.g. 200 — or the string NA if not applicable — or Unknown",
            "main_breaker_rating_amps":      "number only e.g. 100 — or the string NA if there is no main breaker — or Unknown",
            "feeder_breaker": {
                "source_panel":              "label of the panel that feeds this sub panel, e.g. MSP — or Unknown",
                "rating_amps":               "number only, e.g. 100 — or Unknown"
            },
            "available_breaker_slots":       "integer count of physically empty/unused breaker slots — or Unknown",
            "make_model":                    "brand name and model number from the nameplate, e.g. Eaton BR2040B200 — or Unknown",
            "breakers_1_pole": {
                "count": "integer — total number of 1-pole breakers, or 0",
                "breakers": [
                    {"label": "text label on or beside the breaker", "rating_amps": "number only"}
                ]
            },
            "breakers_2_pole": {
                "count": "integer — total number of 2-pole breakers, or 0",
                "breakers": [
                    {"label": "text label on or beside the breaker", "rating_amps": "number only"}
                ]
            },
            "condition_notes":               "brief free text describing visible condition — rust, damage, overcrowding, loose wires, etc. Write None if no issues.",
            "confidence":                    "ENUM: high or medium or low"
        },
        "extra_instructions": (
            "IMPORTANT RULES:\n"
            "1. For each field, return ONLY the single chosen value — do NOT include 'or', '|', or multiple options.\n"
            "2. panel_type: look at where the main cable conduit or service entry enters the enclosure — top = Top_fed, bottom = Bottom_fed.\n"
            "3. main_breaker_rating_amps: ONLY set this to a number if there is a dedicated breaker physically mounted INSIDE this sub panel "
            "whose sole purpose is to shut off this entire panel. This is rare in sub panels — most sub panels do NOT have one. "
            "If you do not see such a breaker, set this to 'NA'. "
            "IMPORTANT: A large 100A breaker labelled 'House Sub' or any other load label is a CIRCUIT breaker feeding a downstream load — "
            "it is NOT the main breaker. It must be included in the breaker inventory list, not treated as the main breaker.\n"
            "4. feeder_breaker: the breaker in the upstream MSP panel that supplies power to this sub panel. "
            "You CANNOT see the MSP in these photos. Set feeder_breaker.source_panel to 'MSP' by default for a sub panel, "
            "and set feeder_breaker.rating_amps to Unknown unless a label or wire tag in these photos explicitly states the upstream feeder size.\n"
            "5. Breaker inventory — follow these steps carefully:\n"
            "   STEP A: List every piece of text you can see written anywhere inside the panel enclosure "
            "(on the door face, on stickers, on wire tags, or on the panel interior). "
            "Examples: 'House Sub', 'Solar', 'HVAC', 'Dryer', 'MSP'.\n"
            "   STEP B: For each label found in Step A, locate the breaker(s) adjacent to or below that label. "
            "A label written on the panel door face or interior ALWAYS identifies the breaker(s) physically next to it — "
            "it is a CIRCUIT BREAKER label, NOT a panel name or decoration.\n"
            "   STEP C: Classify each labeled breaker:\n"
            "   - 1-pole: single breaker body in one slot (120V circuits).\n"
            "   - 2-pole: two breaker bodies in the same column or ganged side-by-side, both the same amperage (240V circuits). "
            "Any circuit rated 30A+ or named 'Solar', 'HVAC', 'EV', 'Dryer', or 'Sub' is almost certainly 2-pole.\n"
            "   STEP D: Every labeled breaker found in Steps A-C must appear in either breakers_1_pole or breakers_2_pole. "
            "Do NOT put any breaker in main_breaker_rating_amps unless it has a dedicated 'MAIN' label and no circuit name.\n"
            "   Set count = number of entries in each breakers list.\n"
            "6. available_breaker_slots: count empty knockout positions row by row across the full panel interior. "
            "Each physical slot that has no breaker installed counts as one available slot."
        )
    }
}

def detect_component_type(directory: str) -> str | None:
    """Map directory name to a known schema key."""
    name = os.path.basename(directory).lower()
    if "sub_panel" in name or "subpanel" in name:
        return "sub_panel"
    return None


def load_and_encode_image(path: str) -> str | None:
    """Load an image (including HEIC), resize, and return base64-encoded JPEG."""
    try:
        img = iio.imread(path)
    except Exception as e:
        print(f"  ⚠ Could not load {path}: {e}")
        return None

    # Handle RGBA or grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    # Resize if too large
    h, w = img.shape[:2]
    if max(h, w) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Encode to JPEG bytes
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not success:
        return None

    return base64.b64encode(buf.tobytes()).decode("utf-8")


def collect_images(directory: str) -> list[str]:
    """Return all image paths in a directory, sorted."""
    exts = ["*.HEIC", "*.heic", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(set(paths))


def build_messages(schema_info: dict, image_b64_list: list[str]) -> list[dict]:
    """Build the OpenAI chat messages payload with embedded images."""
    system_prompt = (
        "You are an expert electrical inspector analysing site photos for a solar "
        "installation company. You will be shown multiple photos of the same electrical "
        "component taken from different angles. "
        "Extract the requested fields and return ONLY valid JSON — no markdown fences, "
        "no explanation, no extra text. "
        "If a field cannot be determined from the images, use the string \"Unknown\" "
        "or \"NA\" as indicated in the schema. "
        "For breaker inventories, read every visible label and amperage rating. "
        "For available slots, count visually empty breaker positions carefully."
    )

    schema_json = json.dumps(schema_info["template"], indent=2)
    extra_instructions = schema_info.get("extra_instructions", "")
    if callable(extra_instructions):
        extra_instructions = extra_instructions()
    user_text = (
        f"Analyse these {len(image_b64_list)} photos of a residential "
        f"{schema_info['description']}.\n\n"
        f"{extra_instructions}\n\n"
        "Return a single JSON object matching this exact schema structure "
        "(replace the description strings with actual extracted values):\n"
        f"{schema_json}\n\n"
        "Rules:\n"
        "- For ENUM fields, return exactly one of the listed options as a plain string.\n"
        "- For number fields, return a plain integer or float — no units, no extra text.\n"
        "- For list fields (breakers_1_pole, breakers_2_pole), include one entry "
        "per breaker you can identify. Use an empty list [] if none are visible.\n"
        "- Set confidence to high if most fields are clearly readable, "
        "medium if some are inferred, low if photos are unclear.\n"
        "Return ONLY the raw JSON object — no markdown, no explanation."
    )

    # Build the content array: interleave images then text
    content = []
    for b64 in image_b64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": content}
    ]


def extract_json(text: str) -> dict | None:
    """Try to parse JSON from VLM response, stripping markdown fences if present."""
    # Strip ```json ... ``` fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def call_vlm(client: OpenAI, messages: list[dict], retry: bool = False) -> str:
    """Call the vLLM server and return the raw response text."""
    extra = " Return ONLY the raw JSON object, no markdown." if retry else ""
    if retry and messages[-1]["role"] == "user":
        # Append a stricter instruction to the last user message
        last_content = messages[-1]["content"]
        if isinstance(last_content, list):
            last_content.append({"type": "text", "text": extra})
        else:
            messages[-1]["content"] += extra

    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def run(directory: str):
    print(f"\n{'='*60}")
    print(f"  Solar Site Inspector")
    print(f"  Directory : {directory}")
    print(f"{'='*60}\n")

    # 1. Detect component type
    component_type = detect_component_type(directory)
    if component_type is None:
        print(f"❌ Unknown directory type: '{os.path.basename(directory)}'")
        print("   Supported types: sub_panel")
        sys.exit(1)
    schema_info = SCHEMAS[component_type]
    print(f"✅ Component type detected: {component_type}")

    # 2. Collect images
    image_paths = collect_images(directory)
    if not image_paths:
        print(f"❌ No images found in {directory}")
        sys.exit(1)
    print(f"✅ Found {len(image_paths)} image(s)")

    # Cap to MAX_IMAGES (take evenly spaced selection to cover the full set)
    if len(image_paths) > MAX_IMAGES:
        step = len(image_paths) / MAX_IMAGES
        image_paths = [image_paths[int(i * step)] for i in range(MAX_IMAGES)]
        print(f"   ↳ Sampled {MAX_IMAGES} images (evenly spaced) to fit context window")

    # 3. Load and encode images
    print("📷 Loading and encoding images...")
    encoded_images = []
    for path in image_paths:
        b64 = load_and_encode_image(path)
        if b64:
            encoded_images.append(b64)
            print(f"   ✓ {os.path.basename(path)}  ({len(b64)//1024} KB encoded)")
        else:
            print(f"   ✗ {os.path.basename(path)}  (skipped)")

    if not encoded_images:
        print("❌ All images failed to load.")
        sys.exit(1)

    # 4. Call VLM
    client = OpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY)
    messages = build_messages(schema_info, encoded_images)

    print(f"\n🤖 Calling Qwen3-VL-2B ({len(encoded_images)} images)...")
    raw_response = call_vlm(client, messages)

    # 5. Parse JSON
    result = extract_json(raw_response)
    if result is None:
        print("⚠  JSON parse failed on first attempt — retrying with stricter prompt...")
        raw_response = call_vlm(client, messages, retry=True)
        result = extract_json(raw_response)

    # 6. Write output
    dir_name = os.path.basename(directory.rstrip("/"))
    output_json = os.path.join(directory, f"{dir_name}.json")
    raw_output  = os.path.join(directory, f"{dir_name}_raw.txt")

    if result:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Extraction successful! Output saved to:\n   {output_json}")
        print("\n📋 Extracted data:")
        print(json.dumps(result, indent=2))
    else:
        with open(raw_output, "w") as f:
            f.write(raw_response)
        print(f"\n❌ JSON parsing failed after retry.")
        print(f"   Raw VLM output saved to: {raw_output}")
        print("\n--- RAW RESPONSE ---")
        print(raw_response)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_site.py <directory>")
        print("Example: python inspect_site.py sorted_data/Sub_panel_1")
        sys.exit(1)
    run(sys.argv[1])
