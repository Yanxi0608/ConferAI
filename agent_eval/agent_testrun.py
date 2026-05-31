from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agent_develop"
sys.path.append(str(AGENT_DIR))

from agent_structure import build_summarize_subgraph

# 输入整理好的文本证据
def build_aligned_units_from_sources(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    transcription = inputs.get("source_transcription", "") or ""
    ocr_text = inputs.get("source_ocr", "") or ""

    return [
        {
            "unit_id": inputs.get("video_filename", "case") + "_gold_unit_1",
            "t_start": 0.0,
            "t_end": 300.0,
            "speech_text": transcription,
            "slide_text": ocr_text,
        }
    ]

# 只跑 Writer-Critic 循环，输入是整理好的文本证据，输出是最终的总结文本和相关评审信息
def writer_critic_target(inputs: dict[str, Any]) -> dict[str, Any]:
    aligned_units = build_aligned_units_from_sources(inputs)

    state = {
        "video_path": inputs.get("video_path", inputs.get("video_filename", "")),
        "aligned_units": aligned_units,
        "rag_snippets": [],
        "iteration": 0,
        "max_iterations": 3,
        "score_threshold": 8.5,
        "critic_feedback": "",
        "draft_summary": "",
    }

    app = build_summarize_subgraph()
    final_state = app.invoke(state)

    return {
        "summary": final_state.get("draft_summary", ""),
        "aligned_units": final_state.get("aligned_units", []),
        "critic_score": final_state.get("critic_score"),
        "critic_feedback": final_state.get("critic_feedback", ""),
        "iteration": final_state.get("iteration", 0),
    }