#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.environ.get(
    "VLLM_MODEL",
    "/home/mnnh-ruzen/Documents/projects/qwen3.5-27b",
)


def post_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_completion_tokens: int,
    enable_thinking: bool,
):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Failed to connect to {base_url}. Is vLLM running?"
        ) from exc


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def build_image_part(image: str) -> dict:
    if is_url(image):
        return {
            "type": "image_url",
            "image_url": {"url": image},
        }

    path = os.path.expanduser(image)
    if not os.path.isfile(path):
        raise SystemExit(f"Image file not found: {image}")

    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "application/octet-stream"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
        },
    }


def build_user_message(prompt: str, images: list[str]) -> dict:
    if not images:
        return {"role": "user", "content": prompt}

    content = [{"type": "text", "text": prompt}]
    content.extend(build_image_part(image) for image in images)
    return {"role": "user", "content": content}


def extract_text(response: dict, show_reasoning: bool) -> str:
    choice = response["choices"][0]["message"]
    reasoning = choice.get("reasoning")
    content = choice.get("content")

    parts = []
    if show_reasoning and reasoning:
        parts.append("[reasoning]")
        parts.append(reasoning.strip())
    if content:
        if show_reasoning and reasoning:
            parts.append("")
            parts.append("[answer]")
        parts.append(content.strip())

    if not parts:
        return json.dumps(response, ensure_ascii=False, indent=2)
    return "\n".join(parts)


def run_once(args: argparse.Namespace) -> int:
    prompt = resolve_prompt(args)
    if not prompt:
        raise SystemExit("Prompt is empty.")

    response = post_chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=[build_user_message(prompt, args.image)],
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        enable_thinking=args.thinking,
    )
    print(extract_text(response, args.show_reasoning))
    return 0


def run_interactive(args: argparse.Namespace) -> int:
    messages: list[dict] = []
    print("Interactive mode. Type /quit to exit, /reset to clear history.")

    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not user_input:
            continue
        if user_input in {"/quit", "/exit"}:
            return 0
        if user_input == "/reset":
            messages.clear()
            print("history cleared")
            continue

        messages.append(build_user_message(user_input, args.image))
        response = post_chat_completion(
            base_url=args.base_url,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            enable_thinking=args.thinking,
        )
        assistant_message = response["choices"][0]["message"]
        answer = extract_text(response, args.show_reasoning)
        print(f"bot> {answer}")
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content") or "",
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send chat requests to a local vLLM OpenAI-compatible server."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. Omit to enter interactive chat mode.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read prompt text from stdin instead of shell arguments.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Attach an image by local path or http(s) URL. Can be used multiple times.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen thinking output.",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Print reasoning text when the model returns it.",
    )
    return parser


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return " ".join(args.prompt).strip()
    if args.stdin or not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.prompt or args.stdin or not sys.stdin.isatty():
        return run_once(args)
    return run_interactive(args)


if __name__ == "__main__":
    sys.exit(main())
