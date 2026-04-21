from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.runnables import RunnableLambda

from models.qa import QAModel
from models.refiner import Refiner
from models.summarizer import BartSummarizer
from preprocessing.preprocessor import Preprocessor

AppState = Dict[str, Any]


@dataclass
class AppComponents:
    preprocessor: Preprocessor
    summarizer: BartSummarizer
    refiner: Refiner
    qa_model: QAModel


def load_components() -> AppComponents:
    return AppComponents(
        preprocessor=Preprocessor(),
        summarizer=BartSummarizer(),
        refiner=Refiner(),
        qa_model=QAModel(),
    )


def preprocess_stage(state: AppState, components: AppComponents) -> AppState:
    state["chunks"] = components.preprocessor.chunk(state["raw_text"])
    return state


def summarize_stage(state: AppState, components: AppComponents) -> AppState:
    partial_summaries, raw_summary = components.summarizer.summarize_chunks(state["chunks"])
    state["partial_summaries"] = partial_summaries
    state["raw_summary"] = raw_summary
    return state


def refine_stage(state: AppState, components: AppComponents) -> AppState:
    state["refined_summary"] = components.refiner.refine(state["raw_summary"], state["length"])
    return state


def build_pipeline(components: AppComponents):
    return (
        RunnableLambda(lambda state: preprocess_stage(state, components))
        | RunnableLambda(lambda state: summarize_stage(state, components))
        | RunnableLambda(lambda state: refine_stage(state, components))
    )


def run_pipeline(raw_text: str, length: str, components: AppComponents) -> AppState:
    chain = build_pipeline(components)
    initial_state: AppState = {
        "raw_text": raw_text,
        "length": length,
        "chunks": [],
        "partial_summaries": [],
        "raw_summary": "",
        "refined_summary": "",
        "qa_history": [],
    }
    return chain.invoke(initial_state)
