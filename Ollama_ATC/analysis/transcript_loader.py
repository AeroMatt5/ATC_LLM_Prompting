"""
Utilities for loading and normalising ATC transcript JSON files.

Both the ground truth and LLM-generated transcripts share the same
schema: each turn has a ``prompt`` (pilot) and ``response`` (ATC) field.
Legacy files that use ``atc_ground_truth`` instead of ``response`` are
detected automatically.
"""

import json
from pathlib import Path


def _normalise(text: str) -> str:
    """Collapse internal newlines to spaces and strip whitespace."""
    return " ".join(text.split())


def _extract_response(turn: dict) -> str:
    """Return the ATC text from a turn regardless of field name."""
    raw = turn.get("response") or turn.get("atc_ground_truth") or ""
    return _normalise(raw)


def load_transcript(filepath):
    """Load any transcript JSON and return {turn_id: response_text}.

    Works for both ground truth and LLM-generated files.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    return {
        t["turn_id"]: _extract_response(t)
        for t in data.get("turns", [])
    }


# Keep explicit aliases for clarity in calling code.
load_ground_truth = load_transcript
load_llm_transcript = load_transcript


def load_transcript_metadata(filepath):
    """Return top-level metadata (everything except turns)."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if k != "turns"}


def load_all_transcripts(directory, loader_fn=None):
    """Load every .json file in *directory*.

    Returns a list of dicts: [{"filepath": str, "turns": {id: text}}].
    """
    if loader_fn is None:
        loader_fn = load_transcript
    directory = Path(directory)
    results = []
    for fp in sorted(directory.glob("*.json")):
        results.append({"filepath": str(fp), "turns": loader_fn(fp)})
    return results


def common_turn_ids(gt_turns: dict, llm_turns: dict) -> list:
    """Sorted list of turn IDs present in both transcripts."""
    return sorted(set(gt_turns) & set(llm_turns))
