from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from safety_lab.cloud_safety import LocalCloudSafetyClient, SafetyThresholds
from safety_lab.factory import build_bot
from safety_lab.reporting import run_attack_suite, run_observability_demo
from safety_lab.scenarios import ATTACK_PROMPTS, BENIGN_PROMPTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Production guardrails and observability hands-on lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Run one chatbot turn.")
    chat_parser.add_argument("message", help="User message to send.")
    chat_parser.add_argument("--base", action="store_true", help="Disable guardrails to demonstrate the vulnerable baseline.")

    subparsers.add_parser("scenario", help="Run the built-in attack and benign scenarios.")
    subparsers.add_parser("report", help="Generate attack/defense and observability reports.")

    cloud_parser = subparsers.add_parser("cloud-test", help="Run local cloud-threshold experiment.")
    cloud_parser.add_argument(
        "--prompt",
        default="How do I treat a severe bleeding arm wound?",
        help="Prompt to analyze.",
    )
    cloud_parser.add_argument("--threshold", type=int, default=4, help="Severity threshold to block at.")

    args = parser.parse_args()
    if args.command == "chat":
        bot = build_bot(force_base=args.base)
        result = bot.chat(args.message)
        print(json.dumps(_chat_payload(result), indent=2))
        return

    if args.command == "scenario":
        for prompt in ATTACK_PROMPTS + BENIGN_PROMPTS:
            bot = build_bot()
            result = bot.chat(prompt)
            print(json.dumps(_chat_payload(result), indent=2))
        return

    if args.command == "cloud-test":
        client = LocalCloudSafetyClient(SafetyThresholds.uniform(args.threshold))
        result = client.analyze(args.prompt, source="input")
        print(json.dumps(asdict(result), indent=2))
        return

    if args.command == "report":
        attack = run_attack_suite(Path("reports/attack-defense-results.json"))
        observability = run_observability_demo(Path("reports/observability-demo.json"))
        print(
            json.dumps(
                {
                    "attack_report": "reports/attack-defense-results.json",
                    "observability_report": "reports/observability-demo.json",
                    "summary": attack["summary"],
                    "intervention_span": observability["intervention_span"],
                },
                indent=2,
            )
        )


def _chat_payload(result) -> dict:
    return {
        "user_input": result.user_input,
        "raw_model_output": result.raw_model_output,
        "final_output": result.final_output,
        "blocked": result.blocked,
        "total_latency_ms": result.total_latency_ms,
        "guardrail_events": [asdict(event) for event in result.guardrail_events],
        "cloud_events": [asdict(event) for event in result.cloud_events],
        "spans": [asdict(span) for span in result.spans],
    }


if __name__ == "__main__":
    main()

