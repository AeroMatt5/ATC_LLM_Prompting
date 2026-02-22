import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_API_PATH = "/api/chat"


def load_prompts_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    seed: int,
    temperature: float,
    top_p: float,
    num_predict: int,
    timeout_s: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + DEFAULT_API_PATH
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def new_chat_skeleton(model: str, attempt_index: int, run_id: str, condition: str, system_prompt: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "condition": condition,
        "model": model,
        "attempt_index": attempt_index,  # 1..N
        "system_prompt": system_prompt,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "turns": [],
    }


def save_chat(chat_path: str, chat_obj: Dict[str, Any]) -> None:
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(chat_obj, f, indent=2, ensure_ascii=False)


def load_or_init_chat(chat_path: str, init_obj: Dict[str, Any]) -> Dict[str, Any]:
    if os.path.exists(chat_path):
        with open(chat_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return init_obj


def build_messages(system_prompt: str, prior_turns: List[Dict[str, Any]], new_user_prompt: str, keep_context: bool) -> List[Dict[str, str]]:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    if keep_context:
        for t in prior_turns:
            messages.append({"role": "user", "content": t["prompt"]})
            messages.append({"role": "assistant", "content": t["response"]})

    messages.append({"role": "user", "content": new_user_prompt})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--condition", default="A0")
    parser.add_argument("--outdir", default="runs")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--system", required=True)
    parser.add_argument("--keep-context", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-predict", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts = load_prompts_txt(args.prompts)
    if not prompts:
        raise SystemExit("No prompts found in prompts file.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"{args.condition}_{run_id}")
    chats_dir = os.path.join(run_dir, "chats")
    ensure_dir(chats_dir)

    chat_paths = []
    chats = []
    for i in range(1, args.n + 1):
        path = os.path.join(chats_dir, f"attempt_{i:02d}.json")
        chat_paths.append(path)
        init_obj = new_chat_skeleton(args.model, i, run_id, args.condition, args.system)
        chats.append(load_or_init_chat(path, init_obj))

    base_seed = args.seed if args.seed != 0 else int(time.time())

    for turn_id, user_prompt in enumerate(prompts, start=1):
        print(f"Turn {turn_id}/{len(prompts)}...")

        for attempt_idx in range(1, args.n + 1):
            chat_obj = chats[attempt_idx - 1]
            messages = build_messages(args.system, chat_obj["turns"], user_prompt, args.keep_context)

            seed = base_seed + (attempt_idx * 10_000) + turn_id

            data = ollama_chat(
                base_url=args.base_url,
                model=args.model,
                messages=messages,
                seed=seed,
                temperature=args.temperature,
                top_p=args.top_p,
                num_predict=args.num_predict,
                timeout_s=args.timeout,
            )

            response_text = (data.get("message", {}) or {}).get("content", "").strip()

            chat_obj["turns"].append({
                "turn_id": turn_id,
                "prompt": user_prompt,
                "response": response_text,
                "seed": seed,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })

            save_chat(chat_paths[attempt_idx - 1], chat_obj)

    index = {
        "run_id": run_id,
        "condition": args.condition,
        "model": args.model,
        "n_attempts": args.n,
        "keep_context": args.keep_context,
        "system_prompt": args.system,
        "prompts_file": os.path.abspath(args.prompts),
        "output_dir": os.path.abspath(run_dir),
        "attempt_files": [os.path.abspath(p) for p in chat_paths],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sampling": {
            "base_seed": base_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": args.num_predict,
        }
    }
    with open(os.path.join(run_dir, "run_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Output folder:\n{run_dir}")


if __name__ == "__main__":
    main()

