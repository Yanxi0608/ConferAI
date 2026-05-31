from __future__ import annotations

import re
from typing import Any


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _keyword_hit(point: str, summary: str) -> bool:
    point_norm = _normalize(point)
    summary_norm = _normalize(summary)

    tokens = [
        token
        for token in re.split(r"[\s,，。；;:：()（）/\\\-]+", point_norm)
        if len(token) >= 2
    ]

    if not tokens:
        return False

    matched = sum(1 for token in tokens if token in summary_norm)
    return matched / len(tokens) >= 0.4


def knowledge_recall(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    summary = outputs.get("summary", "") or ""
    required_points = reference_outputs.get("must_have_points", []) or []

    if not required_points:
        return {
            "key": "knowledge_recall",
            "score": 0.0,
            "comment": "No must_have_points found in reference_outputs.",
        }

    missing = [
        point for point in required_points
        if not _keyword_hit(point, summary)
    ]

    score = (len(required_points) - len(missing)) / len(required_points)

    return {
        "key": "knowledge_recall",
        "score": round(score, 3),
        "comment": "Missing points: " + " | ".join(missing[:5]),
    }


def faithfulness(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    summary = outputs.get("summary", "") or ""
    aligned_units = outputs.get("aligned_units", []) or []

    evidence_text = " ".join(
        str(unit.get("speech_text", "")) + " " + str(unit.get("slide_text", ""))
        for unit in aligned_units
    )

    summary_norm = _normalize(summary)
    evidence_norm = _normalize(evidence_text)

    if not summary_norm:
        return {
            "key": "faithfulness",
            "score": 0.0,
            "comment": "Empty summary.",
        }

    if not evidence_norm:
        return {
            "key": "faithfulness",
            "score": 0.0,
            "comment": "No aligned_units evidence returned by target.",
        }

    suspicious_markers = [
        "state-of-the-art",
        "sota",
        "实验表明显著优于",
        "生产部署",
        "用户研究",
        "大规模上线",
    ]

    unsupported = [
        marker for marker in suspicious_markers
        if marker.lower() in summary_norm and marker.lower() not in evidence_norm
    ]

    score = 1.0 if not unsupported else max(0.0, 1.0 - 0.2 * len(unsupported))

    return {
        "key": "faithfulness",
        "score": round(score, 3),
        "comment": "Unsupported markers: " + " | ".join(unsupported),
    }