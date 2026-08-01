# backend/moderation_ml.py
from __future__ import annotations

from typing import Dict, Tuple
import logging
import os
import re
import warnings


logger = logging.getLogger(__name__)

# Suppress transformers/BERT load warnings (position_ids UNEXPECTED is harmless
# when loading a fine-tuned classification checkpoint from a base architecture).
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._token").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Non-LLM model: RoBERTa fine-tuned on Jigsaw (Detoxify).
_DETOXIFY_IMPORT_ERROR: Exception | None = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from detoxify import Detoxify
    except Exception as exc:  # pragma: no cover - depends on local ML stack
        Detoxify = None
        _DETOXIFY_IMPORT_ERROR = exc


SAFE_MESSAGES: Dict[str, str] = {
    "self_harm": (
        "I'm really sorry you're feeling this way, but I can't help with that. "
        "If you're in immediate danger or considering self-harm, please contact your local emergency number "
        "or speak to a qualified professional right now."
    ),
    "sexual_minors": "I can't assist with sexual content involving minors.",
    "violence_illegal": "I can't assist with requests that could harm you or others or are illegal.",
    "medical_harm": "I can't assist with dangerous medical instructions or overdose guidance.",
    "toxicity": "I'm not able to continue with abusive or threatening language.",
    "nsfw": "I can't assist with sexually explicit content.",
    "general": "Sorry, I can't help with this request.",
}


class ModerationEnsemble:
    """
    Non-LLM moderation gate: Detoxify (RoBERTa on Jigsaw) + high-precision rules -> allow/block.
    Runs BEFORE any retrieval or PubMed calls.
    """

    _CLINICAL_CONTEXT_PATTERN = re.compile(
        r"\b(therapeutic index|prescrib|clinical management|drug interaction|"
        r"formulary|pharmacokinetic|pharmacodynamic|safe prescribing|"
        r"adverse effect management|dose adjustment|monitoring parameter)\b",
        re.IGNORECASE,
    )

    _CLINICAL_ROLE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
        "doctor": {"medical_harm": 2.0},
        "nurse": {"medical_harm": 1.5},
        "midwife": {"medical_harm": 1.5},
        "physiotherapist": {"medical_harm": 1.3},
        "patient": {},
        "caregiver": {},
    }

    def __init__(
        self,
        detoxify_variant: str = "original",
        thresholds: Dict[str, float] | None = None,
    ):
        self.detox = None
        self.detox_backend = "rules_only"
        self.detox_error = ""

        if Detoxify is None:
            self.detox_error = self._format_backend_error(
                _DETOXIFY_IMPORT_ERROR or ImportError("Detoxify import failed")
            )
            logger.warning(
                "Detoxify is unavailable; moderation will use rules only. %s",
                self.detox_error,
            )
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.detox = Detoxify(detoxify_variant)
                self.detox_backend = "detoxify+rules"
            except Exception as exc:  # pragma: no cover - depends on local ML stack
                self.detox = None
                self.detox_error = self._format_backend_error(exc)
                logger.warning(
                    "Detoxify initialization failed; moderation will use rules only. %s",
                    self.detox_error,
                )

        self.t = thresholds or {
            "toxicity": 0.60,
            "severe_toxicity": 0.40,
            "threat": 0.35,
            "identity_attack": 0.50,
            "insult": 0.65,
            "obscene": 0.65,
            "sexual_explicit": 0.40,
        }

        self.re_self_harm = re.compile(
            r"(kill myself|suicide|end my life|self[-\s]?harm|hurt myself|cutting|self[-\s]?mutilation)",
            re.IGNORECASE,
        )
        self.re_med_harm = re.compile(
            r"(overdose|lethal dose|how many (?:pills|tablets) (?:to|would) (?:kill|harm)|"
            r"(?:is it safe to )?mix .* (?:to overdose))",
            re.IGNORECASE,
        )
        self.re_violence_illegal = re.compile(
            r"(kill (?:someone|people)|how to (?:make|build) (?:a )?(?:bomb|explosive|molotov|napalm)|"
            r"buy (?:unregistered|illegal) (?:gun|weapon)|how to (?:hack|scam|forge|cook meth))",
            re.IGNORECASE,
        )
        self.re_age_minor = re.compile(
            r"(\b(?:minor|underage|child|children|12|13|14|15|16|17)\b|(\d{1,2})\s?(?:yo|yrs?|years? old))",
            re.IGNORECASE,
        )
        self.re_sexual = re.compile(r"(sex|nude|naked|porn|explicit|xxx|erotic)", re.IGNORECASE)

    def decide(self, text: str, role_key: str = "patient") -> Tuple[bool, str, str, Dict]:
        """
        Decide if text should be blocked.

        Args:
            text: The user's question text.
            role_key: Canonical clinical role (patient, doctor, nurse, midwife, physiotherapist, caregiver).

        Returns:
            blocked (bool), category (str), safe_message (str), details (dict with model scores and rule hits).
        """
        text = (text or "").strip()
        if not text:
            return False, "allow", "", {"reason": "empty"}

        det_scores = self._detox_scores(text)
        rules = self._rule_hits(text)
        has_clinical_context = bool(self._CLINICAL_CONTEXT_PATTERN.search(text))
        is_clinical_role = role_key in ("doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional")

        if rules["sexual_minors"]:
            return True, "sexual_minors", SAFE_MESSAGES["sexual_minors"], self._pack(det_scores, rules)
        if rules["self_harm"]:
            return True, "self_harm", SAFE_MESSAGES["self_harm"], self._pack(det_scores, rules)
        if rules["violence_illegal"]:
            return True, "violence_illegal", SAFE_MESSAGES["violence_illegal"], self._pack(det_scores, rules)
        if rules["medical_harm"]:
            if not (is_clinical_role and has_clinical_context):
                return True, "medical_harm", SAFE_MESSAGES["medical_harm"], self._pack(det_scores, rules)

        if det_scores.get("sexual_explicit", 0.0) >= self.t["sexual_explicit"]:
            return True, "nsfw", SAFE_MESSAGES["nsfw"], self._pack(det_scores, rules)
        if det_scores.get("threat", 0.0) >= self.t["threat"]:
            return True, "violence_illegal", SAFE_MESSAGES["violence_illegal"], self._pack(det_scores, rules)
        if det_scores.get("severe_toxicity", 0.0) >= self.t["severe_toxicity"]:
            return True, "toxicity", SAFE_MESSAGES["toxicity"], self._pack(det_scores, rules)

        if (
            det_scores.get("toxicity", 0.0) >= self.t["toxicity"]
            or det_scores.get("obscene", 0.0) >= self.t["obscene"]
            or det_scores.get("insult", 0.0) >= self.t["insult"]
            or det_scores.get("identity_attack", 0.0) >= self.t["identity_attack"]
        ):
            return True, "toxicity", SAFE_MESSAGES["toxicity"], self._pack(det_scores, rules)

        return False, "allow", "", self._pack(det_scores, rules)

    def _detox_scores(self, text: str) -> Dict[str, float]:
        if self.detox is None:
            return self._empty_detox_scores()

        try:
            out = self.detox.predict(text)
            return {key: float(value) for key, value in out.items()}
        except Exception as exc:  # pragma: no cover - depends on local ML stack
            self.detox = None
            self.detox_backend = "rules_only"
            self.detox_error = self._format_backend_error(exc)
            logger.warning(
                "Detoxify prediction failed; switching to rules-only moderation. %s",
                self.detox_error,
            )
            return self._empty_detox_scores()

    def _rule_hits(self, text: str) -> Dict[str, bool]:
        sexual_minors = bool(self.re_age_minor.search(text) and self.re_sexual.search(text))
        return {
            "self_harm": bool(self.re_self_harm.search(text)),
            "medical_harm": bool(self.re_med_harm.search(text)),
            "violence_illegal": bool(self.re_violence_illegal.search(text)),
            "sexual_minors": sexual_minors,
        }

    def _pack(self, det_scores: Dict[str, float], rules: Dict[str, bool]) -> Dict:
        return {
            "detoxify": det_scores,
            "rules": rules,
            "moderation_backend": self.detox_backend,
            "moderation_backend_error": self.detox_error,
        }

    @staticmethod
    def _empty_detox_scores() -> Dict[str, float]:
        return {
            "toxicity": 0.0,
            "severe_toxicity": 0.0,
            "threat": 0.0,
            "identity_attack": 0.0,
            "insult": 0.0,
            "obscene": 0.0,
            "sexual_explicit": 0.0,
        }

    @staticmethod
    def _format_backend_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"
