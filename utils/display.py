from __future__ import annotations


def print_header(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def print_summary(summary: str, length: str) -> None:
    print_header(f"Refined Summary ({length})")
    print(summary)
    print("=" * 72)
