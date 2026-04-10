# agents/threat_agent.py
"""
Agentic Threat Evaluation Layer — with LangChain + LLM integration.

Takes raw YOLO detection output (class, confidence, bbox) and produces:
  - Threat Level: LOW | MEDIUM | HIGH | CRITICAL
  - Threat Summary: Human-readable description
  - Evasion Vector: Suggested action for the AUV / operator

TWO operating modes:
  1. Rule-based (offline, default) — fast deterministic rules, no internet needed
  2. LLM-based  (optional)        — chains detection data through a language model
     for richer, context-aware reasoning

LLM backends supported (set via AQUATHREAT_LLM_BACKEND env var):
  - "ollama"  → local Ollama instance (e.g. llama3, mistral) — fully offline
  - "openai"  → OpenAI API (requires OPENAI_API_KEY env var)
  - "none"    → rule-based only (default)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums & Data Classes ─────────────────────────────────────────────────────

class ThreatLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class MineClass(int, Enum):
    BOTTOM_MINE   = 0
    MOORED_MINE   = 1
    DRIFTING_MINE = 2
    ARTILLERY_UXO = 3


CLASS_NAMES = {
    0: "Bottom Mine",
    1: "Moored Mine",
    2: "Drifting Mine",
    3: "Artillery / UXO",
}

CLASS_DANGER_WEIGHT = {
    MineClass.BOTTOM_MINE:   0.7,
    MineClass.MOORED_MINE:   0.8,
    MineClass.DRIFTING_MINE: 0.95,
    MineClass.ARTILLERY_UXO: 0.85,
}


@dataclass
class Detection:
    class_id:   int
    confidence: float
    bbox:       list[float]   # [x_center, y_center, width, height] normalised

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, "Unknown")

    @property
    def danger_weight(self) -> float:
        return CLASS_DANGER_WEIGHT.get(MineClass(self.class_id), 0.5)

    @property
    def threat_score(self) -> float:
        return self.confidence * self.danger_weight


@dataclass
class ThreatAssessment:
    level:          ThreatLevel
    score:          float
    summary:        str
    evasion_vector: str
    llm_reasoning:  str = ""                           # filled only in LLM mode
    detections:     list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "threat_level":    self.level.value,
            "threat_score":    round(self.score, 3),
            "summary":         self.summary,
            "evasion_vector":  self.evasion_vector,
            "detections": [
                {
                    "class":      det.class_name,
                    "confidence": round(det.confidence, 3),
                    "bbox":       det.bbox,
                    "score":      round(det.threat_score, 3),
                }
                for det in self.detections
            ],
        }
        if self.llm_reasoning:
            d["llm_reasoning"] = self.llm_reasoning
        return d

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── LangChain LLM Chain Builder ──────────────────────────────────────────────

def _build_llm_chain(backend: str):
    """
    Build and return a LangChain chain for threat reasoning.

    Returns None if the backend is unavailable or not configured,
    so the caller can gracefully fall back to rule-based mode.

    Args:
        backend: "ollama" | "openai" | "none"
    """
    if backend == "none":
        return None

    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
    except ImportError:
        print("[ThreatAgent] langchain not installed — falling back to rule-based mode.")
        print("              Run: pip install langchain langchain-community langchain-openai")
        return None

    # ── System prompt ────────────────────────────────────────────────────────
    # The prompt gives the LLM its role, the input format, and the exact
    # JSON schema it must return — essential for reliable structured output.
    template = """You are an autonomous underwater threat assessment AI for a naval AUV.

You have received the following detection data from the onboard computer-vision system:

Detections:
{detection_json}

Simulated depth: {depth_m} metres
Rule-based preliminary threat level: {rule_level}

Your task:
1. Analyse the detection data considering class type, confidence scores, and depth.
2. Assign a final threat level: LOW, MEDIUM, HIGH, or CRITICAL.
3. Write a concise threat summary (1-2 sentences).
4. Suggest a specific evasion vector for the AUV operator.

Respond ONLY with a valid JSON object in exactly this format:
{{
  "threat_level": "HIGH",
  "summary": "...",
  "evasion_vector": "...",
  "reasoning": "..."
}}"""

    prompt = PromptTemplate(
        input_variables=["detection_json", "depth_m", "rule_level"],
        template=template,
    )

    # ── Backend selection ────────────────────────────────────────────────────
    if backend == "ollama":
        try:
            from langchain_community.llms import Ollama
            # Default model: llama3. Change to "mistral", "gemma", etc. as needed.
            # Make sure Ollama is running: `ollama serve` and `ollama pull llama3`
            llm = Ollama(model=os.getenv("OLLAMA_MODEL", "llama3"), temperature=0.1)
            print(f"[ThreatAgent] LLM backend: Ollama ({os.getenv('OLLAMA_MODEL','llama3')})")
        except ImportError:
            print("[ThreatAgent] langchain-community not installed.")
            print("              Run: pip install langchain-community")
            return None

    elif backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("[ThreatAgent] OPENAI_API_KEY not set — falling back to rule-based.")
                return None
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.1,
                api_key=api_key,
            )
            print(f"[ThreatAgent] LLM backend: OpenAI ({os.getenv('OPENAI_MODEL','gpt-4o-mini')})")
        except ImportError:
            print("[ThreatAgent] langchain-openai not installed.")
            print("              Run: pip install langchain-openai")
            return None
    else:
        print(f"[ThreatAgent] Unknown backend '{backend}' — falling back to rule-based.")
        return None

    return LLMChain(llm=llm, prompt=prompt)


# ── Main Threat Agent ─────────────────────────────────────────────────────────

class ThreatAgent:
    """
    Agentic threat evaluation layer.

    By default runs in fast rule-based mode. Set the LLM_BACKEND parameter
    (or AQUATHREAT_LLM_BACKEND env var) to "ollama" or "openai" to enable
    LangChain-powered reasoning.

    Rule-based decision priority:
      1. Drifting mine conf > 0.5              → CRITICAL
      2. ≥3 simultaneous threats, mean > 0.5  → CRITICAL
      3. max_score > 0.60                      → HIGH
      4. max_score 0.40–0.60                   → MEDIUM
      5. max_score < 0.40 or no detections     → LOW
    """

    SCORE_THRESHOLDS = {
        ThreatLevel.CRITICAL: 0.85,
        ThreatLevel.HIGH:     0.60,
        ThreatLevel.MEDIUM:   0.40,
        ThreatLevel.LOW:      0.0,
    }

    def __init__(self, llm_backend: Optional[str] = None):
        """
        Args:
            llm_backend: "ollama" | "openai" | "none" (default).
                         Falls back to rule-based if LLM is unavailable.
        """
        backend = llm_backend or os.getenv("AQUATHREAT_LLM_BACKEND", "none")
        self._chain = _build_llm_chain(backend)
        self._use_llm = self._chain is not None

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        detections: list[Detection],
        simulated_depth_m: Optional[float] = None,
    ) -> ThreatAssessment:
        """
        Evaluate detections → ThreatAssessment.

        Always runs rule-based first to get a quick preliminary level.
        If LLM mode is active, passes the data to the LangChain chain
        for richer reasoning and overwrites the summary/evasion fields.
        """
        # Step 1: Always compute rule-based assessment
        rule_assessment = self._rule_based(detections, simulated_depth_m)

        # Step 2: If LLM is configured, enrich with LLM reasoning
        if self._use_llm and detections:
            rule_assessment = self._llm_enrich(rule_assessment, detections,
                                                simulated_depth_m)

        return rule_assessment

    # ── Rule-based core ───────────────────────────────────────────────────────

    def _rule_based(
        self,
        detections: list[Detection],
        simulated_depth_m: Optional[float],
    ) -> ThreatAssessment:
        if not detections:
            return ThreatAssessment(
                level=ThreatLevel.LOW, score=0.0,
                summary="No threats detected in the current frame.",
                evasion_vector="Continue on current heading. Maintain sonar sweep.",
                detections=[],
            )

        depth_modifier = 1.0
        if simulated_depth_m is not None:
            depth_modifier = 1.0 + min(0.15, simulated_depth_m / 200.0)

        scores     = [d.threat_score * depth_modifier for d in detections]
        max_score  = max(scores)
        mean_score = sum(scores) / len(scores)
        n_det      = len(detections)

        drifting = [d for d in detections if d.class_id == MineClass.DRIFTING_MINE]
        if drifting and max(d.confidence for d in drifting) > 0.5:
            return self._make(
                ThreatLevel.CRITICAL, max_score, detections,
                f"DRIFTING MINE detected (conf={drifting[0].confidence:.2f}). "
                "Unpredictable trajectory — immediate evasion required.",
                "EMERGENCY STOP. Reverse to safe distance (>50m). "
                "Surface and await operator command.",
            )

        if n_det >= 3 and mean_score > 0.5:
            return self._make(
                ThreatLevel.CRITICAL, max_score, detections,
                f"{n_det} simultaneous threats detected. Possible minefield.",
                "Do NOT advance. Plot reverse course. Alert command immediately.",
            )

        level = ThreatLevel.LOW
        for lvl in [ThreatLevel.HIGH, ThreatLevel.MEDIUM, ThreatLevel.LOW]:
            if max_score >= self.SCORE_THRESHOLDS[lvl]:
                level = lvl
                break

        summary, evasion = self._narrative(level, detections, max_score)
        return self._make(level, max_score, detections, summary, evasion)

    # ── LLM enrichment ────────────────────────────────────────────────────────

    def _llm_enrich(
        self,
        rule_result: ThreatAssessment,
        detections: list[Detection],
        simulated_depth_m: Optional[float],
    ) -> ThreatAssessment:
        """
        Pass detection data through the LangChain chain.
        On any LLM error, return the original rule-based result unchanged.
        """
        try:
            det_json = json.dumps(
                [{"class": d.class_name, "confidence": round(d.confidence, 3),
                  "threat_score": round(d.threat_score, 3)} for d in detections],
                indent=2
            )
            response = self._chain.run(
                detection_json = det_json,
                depth_m        = str(simulated_depth_m or "unknown"),
                rule_level     = rule_result.level.value,
            )

            # Parse the JSON response from the LLM
            clean    = response.strip().lstrip("```json").rstrip("```").strip()
            parsed   = json.loads(clean)

            return ThreatAssessment(
                level          = ThreatLevel(parsed.get("threat_level",
                                                         rule_result.level.value)),
                score          = rule_result.score,
                summary        = parsed.get("summary",        rule_result.summary),
                evasion_vector = parsed.get("evasion_vector", rule_result.evasion_vector),
                llm_reasoning  = parsed.get("reasoning", ""),
                detections     = detections,
            )

        except Exception as e:
            print(f"[ThreatAgent] LLM call failed ({e}) — using rule-based result.")
            return rule_result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make(self, level, score, detections, summary, evasion) -> ThreatAssessment:
        return ThreatAssessment(
            level=level, score=round(score, 3),
            summary=summary, evasion_vector=evasion,
            detections=detections,
        )

    def _narrative(self, level, detections, max_score) -> tuple[str, str]:
        top = max(detections, key=lambda d: d.threat_score)
        return {
            ThreatLevel.HIGH: (
                f"HIGH threat: {top.class_name} detected "
                f"(conf={top.confidence:.2f}, score={max_score:.2f}). "
                "Immediate evasive action advised.",
                "Reduce speed. Alter heading 45° starboard. Increase sonar sweep rate.",
            ),
            ThreatLevel.MEDIUM: (
                f"MEDIUM threat: {top.class_name} detected (conf={top.confidence:.2f}). "
                "Proceed with caution.",
                "Slow to minimum speed. Maintain safe 20m lateral clearance. Log position.",
            ),
            ThreatLevel.LOW: (
                f"LOW threat: {top.class_name} possibly detected "
                f"(conf={top.confidence:.2f}). Low confidence — continue monitoring.",
                "Continue on heading. Flag position for follow-up sonar sweep.",
            ),
        }.get(level, ("Unknown threat level.", "Hold position."))


# ── Quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Rule-based mode (default) ===")
    agent = ThreatAgent(llm_backend="none")
    dets  = [
        Detection(class_id=2, confidence=0.82, bbox=[0.5, 0.4, 0.1, 0.12]),
        Detection(class_id=0, confidence=0.65, bbox=[0.3, 0.7, 0.08, 0.09]),
    ]
    assessment = agent.evaluate(dets, simulated_depth_m=35.0)
    print(assessment)

    print("\n=== LLM mode (Ollama) — set AQUATHREAT_LLM_BACKEND=ollama to activate ===")
    print("    Make sure Ollama is running: ollama serve && ollama pull llama3")
