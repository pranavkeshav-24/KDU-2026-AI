from __future__ import annotations

from config import VALID_LENGTHS
from pipeline import load_components, run_pipeline
from utils.display import print_summary


def get_text_from_user() -> str:
    print("Paste your source text. Submit an empty line to finish.\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            if lines:
                break
            print("Text cannot be empty. Please paste at least one line.")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def get_length_from_user() -> str:
    while True:
        choice = input("Summary length [short / medium / long]: ").strip().lower()
        if choice in VALID_LENGTHS:
            return choice
        print("Invalid choice. Use short, medium, or long.")


def run_qa_loop(state: dict, qa_model) -> None:
    print("\nAsk questions about the text. Type 'exit' to quit.\n")
    refined_context = state["refined_summary"]
    raw_context = state.get("raw_summary", "").strip()

    while True:
        question = input("Q: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye.")
            break

        result = qa_model.answer(question, refined_context)
        used_fallback = False

        # Retry on richer, pre-refinement context when confidence is low.
        if result["low_confidence"] and raw_context and raw_context != refined_context:
            fallback_result = qa_model.answer(question, raw_context)
            if fallback_result["score"] > result["score"]:
                result = fallback_result
                used_fallback = True

        if result["low_confidence"]:
            print(f"Warning: low confidence ({result['score']:.2f}).")
        if used_fallback:
            print("Note: retried answer against raw merged summary context.")

        print(f"A: {result['answer']}\n")
        state["qa_history"].append(
            {"question": question, "answer": result["answer"], "score": result["score"]}
        )


def main() -> None:
    print("Loading models... this can take time on first run.")
    components = load_components()

    raw_text = get_text_from_user()
    length = get_length_from_user()

    state = run_pipeline(raw_text=raw_text, length=length, components=components)
    print_summary(state["refined_summary"], length=length)

    run_qa_loop(state, components.qa_model)


if __name__ == "__main__":
    main()
