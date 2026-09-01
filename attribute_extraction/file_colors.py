"""
FISAD Tops Color / Bottom Color Extraction
============================================
Fills in the two currently-empty colour columns by having a current
vision model (Claude Haiku 4.5) look at each image directly, using the
already-correct Tops/Bottom garment-type labels as context.

USAGE:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 fill_colors.py

Resumable: progress is saved incrementally to FISAD_data_with_colors.csv.
If interrupted, re-running will skip images already processed.
"""

import os
import json
import base64
import time
import concurrent.futures
import pandas as pd
import requests
from anthropic import Anthropic

INPUT_CSV = os.environ.get("FISAD_INPUT_CSV", "FISAD_data.csv")
OUTPUT_CSV = os.environ.get("FISAD_OUTPUT_CSV", "FISAD_data_with_colors.csv")
MODEL = "claude-haiku-4-5-20251001"
MAX_WORKERS = 8
RETRY_LIMIT = 3
TIME_BUDGET_SECONDS = 100  # stop before hitting tool timeout, checkpoint, exit cleanly

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

COLOR_OPTIONS = "Blue, Black, Red, White, Grey, Green, Pink, Brown, Yellow, Purple, Orange, Multi-Color, Other, Not visible"

PROMPT_TEMPLATE = """You are labeling a fashion/apparel image for a research dataset. This image has already been labeled with: Tops = "{tops}", Bottom = "{bottom}".

Look at the image and identify the COLOUR of the top garment and the COLOUR of the bottom garment, using ONLY these categories: {colors}

Use "Not visible" if that garment isn't visible/applicable in the image. Respond with ONLY a valid JSON object, no other text, in exactly this format:
{{"TopsColor": "...", "BottomColor": "..."}}
"""


def fetch_image_b64(url, timeout=15):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    media_type = "image/png" if url.lower().endswith(".png") else "image/jpeg"
    return base64.standard_b64encode(resp.content).decode("utf-8"), media_type


def label_one(row):
    img_id, url = row["id"], row["path"]
    tops = row.get("Tops", "unknown")
    bottom = row.get("Bottom", "unknown")
    prompt = PROMPT_TEMPLATE.format(tops=tops, bottom=bottom, colors=COLOR_OPTIONS)
    for attempt in range(RETRY_LIMIT):
        try:
            img_b64, media_type = fetch_image_b64(url)
            message = client.messages.create(
                model=MODEL,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = message.content[0].text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            parsed = json.loads(text)
            parsed["id"] = img_id
            parsed["_status"] = "ok"
            return parsed
        except Exception as e:
            if attempt == RETRY_LIMIT - 1:
                return {"id": img_id, "TopsColor": None, "BottomColor": None, "_status": f"error: {e}"}
            time.sleep(2 ** attempt)


def main():
    df = pd.read_csv(INPUT_CSV)

    done_ids = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        done_ids = set(existing["id"].tolist())
        print(f"Resuming: {len(done_ids)} images already processed.")

    todo = df[~df["id"].isin(done_ids)]
    print(f"Remaining: {len(todo)} images.")

    write_header = not os.path.exists(OUTPUT_CSV)
    start_time = time.time()
    results = []
    processed_count = 0

    rows_iter = todo.to_dict("records")
    i = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while i < len(rows_iter) and (time.time() - start_time) < TIME_BUDGET_SECONDS:
            batch = rows_iter[i:i + 40]
            futures = [executor.submit(label_one, row) for row in batch]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                processed_count += 1
            i += len(batch)
            pd.DataFrame(results).to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
            write_header = False
            print(f"  ...{processed_count}/{len(todo)} in this run, elapsed {time.time()-start_time:.0f}s")
            results = []

    print(f"\nDone this run. Elapsed {time.time()-start_time:.0f}s")
    if os.path.exists(OUTPUT_CSV):
        final = pd.read_csv(OUTPUT_CSV)
        print(f"Total processed so far: {len(final)}/{len(df)}")


if __name__ == "__main__":
    main()
