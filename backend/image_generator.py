"""
Medical illustration generator using GPT image generation.
Detects when a question would genuinely benefit from a visual and generates a
safe, clinical-style illustration inline in the chat.
"""
from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


_VISUAL_REQUEST_PATTERN = re.compile(
    r"\b(show|illustration|diagram|picture|image|visual|draw|sketch|"
    r"picture form|image form|with pictures|with diagrams|"
    r"what does .{0,30} look like|can you show)\b",
    re.IGNORECASE,
)

_PROCEDURAL_REQUEST_PATTERN = re.compile(
    r"\b(how to do|how to perform|how to apply|how do i|how should i|"
    r"step[-\s]?by[-\s]?step|proper technique|correct form|demonstrate)\b",
    re.IGNORECASE,
)

_EXERCISE_SUBJECT_PATTERN = re.compile(
    r"\b(exercises?|stretche?s?|poses?|postures?|positions?|movements?|techniques?|"
    r"planks?|squats?|lunges?|deadlifts?|push.?ups?|sit.?ups?|crunches?|bridges?|"
    r"yoga|pilates|physiotherapy|rehab)\b",
    re.IGNORECASE,
)

_ANATOMY_SUBJECT_PATTERN = re.compile(
    r"\b(anatomy|anatomical|cross.section|structure of|location of|"
    r"nerve|muscle|tendon|ligament|bone|joint|organ|heart|lung|kidney|"
    r"liver|spine|vertebra|pelvis|shoulder|knee|hip|ankle|wrist|elbow)\b",
    re.IGNORECASE,
)

_PROCEDURAL_SUBJECT_PATTERN = re.compile(
    r"\b(first aid|cpr|heimlich|recovery position|bandages?|splints?|slings?|"
    r"bandaging|dressing|injection|epipen|inhaler technique|blood pressure cuff|"
    r"peak flow|rice)\b",
    re.IGNORECASE,
)

_ANATOMY_LOOKUP_PATTERN = re.compile(
    r"\b(diagram of|where is|location of|structure of|anatomy of|look like|located)\b",
    re.IGNORECASE,
)

_NON_VISUAL_INFO_PATTERN = re.compile(
    r"\b(symptoms?|causes?|diagnosis|diagnostic|treatment|treatments|medication|"
    r"medications|dose|doses|side effects?|summary|summarize|explain|"
    r"urgent|emergency|when should|should i)\b",
    re.IGNORECASE,
)

_UNSAFE_TOPICS = re.compile(
    r"\b(blood|wound|injury|trauma|gore|violent|nude|naked|explicit)\b",
    re.IGNORECASE,
)


@dataclass
class IllustrationResult:
    image_url: str
    prompt_used: str
    caption: str
    generated: bool = True
    image_bytes: Optional[bytes] = field(default=None, repr=False)


class ImageGenerator:
    """
    Generates clinical-style illustrations via OpenAI image generation.
    Call detect_illustration_need() first, then generate_illustration() if True.
    """

    IMAGE_MODEL = "gpt-image-1"
    IMAGE_SIZE = "1024x1024"
    IMAGE_QUALITY = "medium"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        from openai import OpenAI

        from backend.clinical_llm_gateway import guard_clinical_client
        self.client = guard_clinical_client(OpenAI(api_key=api_key), purpose="patient-education-image")

    @staticmethod
    def detect_illustration_need(question: str) -> bool:
        """
        Returns True only when the user explicitly asks for a visual.
        """
        text = (question or "").strip()
        if not text or _UNSAFE_TOPICS.search(text):
            return False

        return bool(_VISUAL_REQUEST_PATTERN.search(text))

    def generate_illustration(
        self,
        question: str,
        context_answer: str = "",
    ) -> Optional[IllustrationResult]:
        """
        Generates a clinical-style image for the given question.
        Returns None if generation fails or content is deemed unsafe.
        """
        question = (question or "").strip()
        if not question:
            return None

        prompt = self._build_prompt(question, context_answer)
        if not prompt:
            return None

        try:
            response = self.client.images.generate(
                model=self.IMAGE_MODEL,
                prompt=prompt,
                size=self.IMAGE_SIZE,
                quality=self.IMAGE_QUALITY,
                n=1,
            )
            item = response.data[0]
            caption = self._build_caption(question)

            if item.url:
                return IllustrationResult(
                    image_url=item.url,
                    prompt_used=prompt,
                    caption=caption,
                )
            if item.b64_json:
                return IllustrationResult(
                    image_url="",
                    prompt_used=prompt,
                    caption=caption,
                    image_bytes=base64.b64decode(item.b64_json),
                )
            print("ImageGenerator: response contained neither url nor b64_json.")
            return None
        except Exception as exc:
            print(f"ImageGenerator: generation failed - {exc}")
            return None

    def _build_prompt(self, question: str, answer_context: str) -> str:
        """
        Constructs a safe, clinically appropriate image prompt.
        """
        del answer_context
        topic = self._extract_topic(question)
        if not topic:
            return ""

        style = (
            "Clean medical education illustration, vector art style, "
            "white background, anatomically accurate, no blood or wounds, "
            "suitable for a healthcare information platform. "
            "Professional diagram style, labelled if appropriate."
        )

        q_lower = question.lower()

        if any(
            word in q_lower
            for word in (
                "exercise",
                "stretch",
                "pose",
                "plank",
                "squat",
                "yoga",
                "pilates",
                "movement",
                "technique",
                "rehab",
            )
        ):
            return (
                f"Clear step-by-step exercise illustration showing {topic}. "
                f"Show correct body positioning and form with a simple human figure. "
                f"{style}"
            )

        if any(
            word in q_lower
            for word in (
                "anatomy",
                "anatomical",
                "diagram",
                "structure",
                "muscle",
                "bone",
                "joint",
                "nerve",
                "organ",
            )
        ):
            return (
                f"Medical anatomy diagram of {topic}. "
                f"Educational illustration with clear labels. "
                f"{style}"
            )

        if any(
            word in q_lower
            for word in (
                "first aid",
                "cpr",
                "recovery position",
                "heimlich",
                "bandage",
                "dressing",
                "injection",
                "sling",
                "splint",
            )
        ):
            return (
                f"Medical first aid illustration showing {topic}. "
                f"Step-by-step instructional diagram. "
                f"{style}"
            )

        if any(word in q_lower for word in ("posture", "position", "rice")):
            return (
                f"Medical illustration showing correct {topic}. "
                f"Clear instructional diagram with positioning guide. "
                f"{style}"
            )

        return f"Medical education illustration showing {topic}. {style}"

    @staticmethod
    def _extract_topic(question: str) -> str:
        """Extract the core illustration subject from the question."""
        cleaned = re.sub(
            r"^(show me|can you show|demonstrate|draw|illustrate|"
            r"what does|how to do|how to perform|how to apply|"
            r"diagram of|picture of|image of|visual of)",
            "",
            question.strip(),
            flags=re.IGNORECASE,
        ).strip()

        cleaned = re.sub(r"\?+$", "", cleaned).strip()
        cleaned = re.sub(r"\b(please|for me|to me|me|a|an|the)\b", "", cleaned).strip()
        cleaned = " ".join(cleaned.split())

        if _UNSAFE_TOPICS.search(cleaned):
            return ""

        return cleaned[:200]

    @staticmethod
    def _build_caption(question: str) -> str:
        """Builds a short descriptive caption for the generated image."""
        cleaned = re.sub(
            r"^(show me|can you|please|how to|what does)",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"\?+$", "", cleaned).strip()
        cleaned = cleaned[:80].capitalize()
        return f"Illustration: {cleaned}" if cleaned else "Generated illustration"
