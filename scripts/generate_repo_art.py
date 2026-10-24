#!/usr/bin/env python3
"""Generate the Futurist / Art Deco artistic cover image for qubit-deja-vu
using the Gemini image API (gemini-3.1-flash-image), following the setup in
~/poznamky/CZ/project_2100/generate_czech_images.py.

  python scripts/generate_repo_art.py
"""
import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "gemini-3.1-flash-image"
REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO / "notes"

NO_TEXT_CONSTRAINT = (
    "Pure image only: no text, no letters, no words, no numbers, "
    "no equations, no logos, no watermarks, no signage anywhere in the frame."
)

PROMPTS = [
    {
        "name": "qubit_deja_vu_art",
        "text": (
            "Wide 16:9 landscape illustration in a fusion of Italian Futurism (Umberto Boccioni, "
            "Giacomo Balla) and 1930s Art Deco scientific poster art. "
            "Subject: 'Qubit Deja Vu': testing whether a quantum measurement remembers the previous outcome. "
            "At the left, a monumental stepped Art Deco cryogenic chandelier of polished copper and brass "
            "condenses downward into a superconducting quantum chip. Across the frame sweeps a dynamic diagonal "
            "train of rhythmic microwave pulses and translucent prism-arches representing sixty-six rapid "
            "mid-circuit readouts without reset. A luminous equatorial sphere travels through the succession "
            "of measurement arches; trailing behind it like Boccioni motion-echoes are ghosted copper and amber "
            "afterimages (readout memory, deja vu from one measurement to the next), while a sharp crimson "
            "trajectory escaping upward into a leaked third level is intercepted by a pair of geometric Art Deco "
            "caliper-blades. Sharp faceted planes, copper hatching over warm cream paper, deep Prussian steel-blue, "
            "charcoal shadows, radiant amber gold, and crimson accents. "
            + NO_TEXT_CONSTRAINT
        ),
    },
    {
        "name": "qubit_deja_vu_art_v2",
        "text": (
            "Wide 16:9 landscape artwork combining 1930s Art Deco architectural symmetry and Italian Futurist "
            "dynamism, gouache and linocut texture on warm cream ground. "
            "Subject: sequential quantum measurement and the memory of a detector ('Qubit Deja Vu'). "
            "In the center stands a stepped Art Deco brass and copper superconducting resonator tower radiating "
            "concentric geometric phase rings. Through a corridor of repeating angular measurement frames, "
            "a single quantum sphere is observed again and again: on the cool steel-blue side, clean memoryless "
            "flips alternate in crisp geometric balance; on the warm amber-and-crimson side, ghostly overlapping "
            "echo-silhouettes of the previous state linger across frames like a visual deja vu. "
            "Bold black outlines, faceted constructivist planes, copper gold, steel blue, charcoal, cream, "
            "and a crimson note. "
            + NO_TEXT_CONSTRAINT
        ),
    },
    {
        "name": "qubit_deja_vu_art_v3",
        "text": (
            "Wide 16:9 landscape illustration in the manner of 1930s Art Deco machine-age poster art "
            "with Italian Futurist motion lines, engraved copper hatching on warm cream paper. "
            "Subject: the flipping qubit itself, shown close and monumental. "
            "A single luminous equatorial sphere rests on a stepped brass pedestal at the center, caught "
            "between two orientations: its cool steel-blue hemisphere tilts one way while a warm amber "
            "afterimage of the same sphere tilts the other way, so the flip reads as one object occupying "
            "two balanced states. Boccioni-style motion echoes fan out behind it in translucent copper arcs. "
            "Above, a small cryogenic chandelier of polished copper cylinders descends toward the sphere; "
            "below, concentric geometric rings suggest a clock face without numerals. "
            "Faceted constructivist planes, charcoal shadows, Prussian blue, radiant gold, and one crimson "
            "accent. "
            + NO_TEXT_CONSTRAINT
        ),
    },
    {
        "name": "qubit_deja_vu_art_v4",
        "text": (
            "Wide 16:9 landscape artwork fusing Art Deco architectural rhythm with Futurist dynamism, "
            "gouache and linocut texture on warm cream ground. "
            "Subject: a long reset-free chain of rapid quantum measurements, drawn as a receding colonnade "
            "of sixty-six angular prism arches in alternating brass and steel-blue. "
            "A glowing amber sphere travels the full length of the corridor, passing through every arch, "
            "and behind it a graded series of ghosted copper afterimages records each previous measurement. "
            "The perspective is low and oblique, so the arches converge toward a distant stepped deco tower; "
            "thin crimson trajectory lines thread the colonnade like a single continuous pulse train. "
            "Deep charcoal shadows, faceted planes, cream highlights, and restrained gold. "
            + NO_TEXT_CONSTRAINT
        ),
    },
    {
        "name": "qubit_deja_vu_art_v5",
        "text": (
            "Wide 16:9 landscape illustration in a fusion of Art Deco scientific instrument art and "
            "Italian Futurism, copper and brass against warm cream, polished machine surfaces. "
            "Subject: a qubit leaking into a third level and being caught. "
            "At center, three stacked geometric energy terraces rise like stepped deco platforms; "
            "a luminous amber sphere sits on the middle terrace while a thin crimson arc escapes upward "
            "from it toward the highest terrace. Two monumental faceted caliper-blades, part precision "
            "instrument and part deco ornament, close around the crimson arc without touching the sphere. "
            "Beneath the terraces, a superconducting chip is rendered as an engraved geometric mandala; "
            "above, microwave pulses appear as rhythmic chevrons. Charcoal shadows, Prussian blue, "
            "radiant gold, and a single crimson accent. "
            + NO_TEXT_CONSTRAINT
        ),
    },
    {
        "name": "qubit_deja_vu_art_v6",
        "text": (
            "Wide 16:9 landscape artwork in 1930s Art Deco poster style crossed with Futurist motion "
            "painting, bold black outlines and faceted constructivist planes on warm cream paper. "
            "Subject: two readings of the same flipping qubit, side by side. "
            "On the left, a cool steel-blue hall of clean alternating frames shows a single sphere flipping "
            "crisply between two orientations, each frame independent, with no residue between them. "
            "On the right, the same hall glows amber and crimson, and each sphere trails overlapping "
            "echo-silhouettes of the state before it, so the sequence reads as visual deja vu. "
            "A narrow central tower of stepped brass and copper separates the two halls and rises into "
            "concentric phase rings; its base grips a small superconducting chip. "
            "Sharply raked light, deep charcoal shadows, geometric clouds, and measured gold highlights. "
            + NO_TEXT_CONSTRAINT
        ),
    },
]


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    env_candidates = [
        REPO / ".env",
        Path.home() / "poznamky" / "CZ" / "project_2100" / ".env",
    ]
    for env_path in env_candidates:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.removeprefix("export ").strip()
                if k == "GEMINI_API_KEY":
                    return v.strip().strip("'").strip('"')
    raise RuntimeError("GEMINI_API_KEY not set in environment or .env")


def generate_image(prompt: str, output_path: Path, model: str, seed=None) -> bool:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={load_api_key()}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"imageConfig": {"aspectRatio": "16:9"}},
    }
    if seed is not None:
        payload["generationConfig"]["seed"] = seed

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                parts = body["candidates"][0]["content"]["parts"]
                image_part = next((p for p in parts if "inlineData" in p), None)
                if not image_part:
                    print("Error: model returned no inlineData")
                    return False
                img_bytes = base64.b64decode(image_part["inlineData"]["data"])
                output_path.write_bytes(img_bytes)
                print(f"Saved {output_path} ({len(img_bytes)} bytes)")
                return True
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 503) and attempt < 3:
                wait = 10 * (attempt + 1)
                print(f"HTTP {e.code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            # Fallback if imageConfig is not supported on this endpoint
            if e.code == 400 and "imageConfig" in payload.get("generationConfig", {}):
                payload.pop("generationConfig", None)
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"}, method="POST"
                )
                continue
            print(f"HTTP Error {e.code}: {err[:300]}")
            return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate Futurist / Art Deco artwork for qubit-deja-vu")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(PROMPTS):
        out_path = out_dir / f"{item['name']}.png"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing: {out_path}")
            continue
        generate_image(item["text"], out_path, args.model)
        if i < len(PROMPTS) - 1:
            time.sleep(3)


if __name__ == "__main__":
    main()
