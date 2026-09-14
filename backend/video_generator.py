"""
Medical video generator using OpenAI Sora-2.
Detects when a question would benefit from a short video demonstration and
generates a clinical-style clip (max 8 seconds, max 1 per user per hour).
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Rate limiting ──────────────────────────────────────────────────────────────
VIDEO_COOLDOWN_MINUTES = 60
VIDEO_MAX_DURATION_SECONDS = 8    # kept under the 10-second cap

# ── Video trigger keywords ─────────────────────────────────────────────────────
_VIDEO_PATTERNS = [
    re.compile(
        r"\b(video|animate|animation|clip|motion|moving|"
        r"show (?:me )?(?:a )?video|video of|video demonstration|"
        r"demonstrate (?:with )?(?:a )?video|video guide|video tutorial)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(how (?:does|do) .{0,40} (?:look|move|work) in (?:motion|action|video))\b",
        re.IGNORECASE,
    ),
]

# Safety: topics that must not appear in generated videos
_UNSAFE_TOPICS = re.compile(
    r"\b(blood|wound|injury|gore|violent|nude|naked|explicit|surgery|autopsy)\b",
    re.IGNORECASE,
)


@dataclass
class VideoResult:
    video_url: str
    prompt_used: str
    caption: str
    duration_seconds: int


@dataclass
class VideoRateLimitResult:
    allowed: bool
    wait_minutes: int = 0
    message: str = ""


class VideoGenerator:
    """
    Generates short clinical demonstration videos via OpenAI Sora-2.
    Enforces a hard 1-video-per-hour rate limit per user.
    """

    VIDEO_MODEL = "sora-2"
    VIDEO_RESOLUTION = "480p"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        from openai import OpenAI
        from backend.clinical_llm_gateway import guard_clinical_client
        self.client = guard_clinical_client(OpenAI(api_key=api_key), purpose="patient-education-video")

    # ── Detection ──────────────────────────────────────────────────────────────

    @staticmethod
    def detect_video_request(question: str) -> bool:
        """Returns True only if the user explicitly requests a video or animation."""
        text = (question or "").strip()
        return any(p.search(text) for p in _VIDEO_PATTERNS)

    # ── Rate limiting ──────────────────────────────────────────────────────────

    @staticmethod
    def check_rate_limit(last_video_at: Optional[str]) -> VideoRateLimitResult:
        """
        Returns whether the user is allowed to generate a video now.
        last_video_at: ISO-8601 UTC timestamp string, or None/empty if never generated.
        """
        if not last_video_at:
            return VideoRateLimitResult(allowed=True)

        try:
            last_dt = datetime.fromisoformat(last_video_at)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return VideoRateLimitResult(allowed=True)

        elapsed = datetime.now(timezone.utc) - last_dt
        cooldown = timedelta(minutes=VIDEO_COOLDOWN_MINUTES)

        if elapsed >= cooldown:
            return VideoRateLimitResult(allowed=True)

        remaining = cooldown - elapsed
        wait_minutes = max(1, int(remaining.total_seconds() / 60) + 1)
        return VideoRateLimitResult(
            allowed=False,
            wait_minutes=wait_minutes,
            message=(
                f"Video generation is limited to once per hour. "
                f"You can generate your next video in approximately **{wait_minutes} minute(s)**."
            ),
        )

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate_video(
        self,
        question: str,
        context_answer: str = "",
    ) -> Optional[VideoResult]:
        """
        Generates a short clinical demonstration video using Sora-2.
        Returns None if generation fails or content is unsafe.
        """
        question = (question or "").strip()
        if not question:
            return None

        prompt = self._build_prompt(question, context_answer)
        if not prompt:
            return None

        try:
            response = self.client.video.generations.create(
                model=self.VIDEO_MODEL,
                prompt=prompt,
                duration=VIDEO_MAX_DURATION_SECONDS,
                resolution=self.VIDEO_RESOLUTION,
                n=1,
            )
            video_url = response.data[0].url
            caption = self._build_caption(question)
            return VideoResult(
                video_url=video_url,
                prompt_used=prompt,
                caption=caption,
                duration_seconds=VIDEO_MAX_DURATION_SECONDS,
            )
        except Exception as exc:
            print(f"VideoGenerator: Sora-2 generation failed -- {exc}")
            return None

    # ── Prompt building ────────────────────────────────────────────────────────

    def _build_prompt(self, question: str, answer_context: str) -> str:
        """Constructs a safe, clinically appropriate Sora-2 video prompt."""
        topic = self._extract_topic(question)
        if not topic:
            return ""

        style = (
            "Short clinical education video, clean white background, "
            "professional medical illustration style, no blood or wounds, "
            "no text overlays, smooth slow motion, suitable for a healthcare platform."
        )

        q_lower = question.lower()

        if any(w in q_lower for w in ("exercise", "exercises", "stretch", "stretches",
                                       "rehab", "physio", "movement", "technique",
                                       "plank", "squat", "yoga", "pilates",
                                       "mobilisation", "mobilization", "back pain",
                                       "neck pain", "shoulder pain", "knee pain")):
            return (
                f"Short demonstration video showing correct technique for {topic}. "
                f"Human figure demonstrating the movement from start to finish. "
                f"{style}"
            )

        if any(w in q_lower for w in ("first aid", "cpr", "recovery position",
                                       "heimlich", "bandage", "sling")):
            return (
                f"Short instructional video demonstrating {topic} first aid technique. "
                f"Step-by-step professional demonstration. "
                f"{style}"
            )

        if any(w in q_lower for w in ("anatomy", "joint", "muscle", "movement",
                                       "range of motion", "rom")):
            return (
                f"Short anatomical demonstration video showing {topic}. "
                f"Medical education animation style. "
                f"{style}"
            )

        return f"Short clinical demonstration video of {topic}. {style}"

    @staticmethod
    def _extract_topic(question: str) -> str:
        """Extract the core video subject from the question."""
        cleaned = re.sub(
            r"^(show me|can you show|generate|create|make|animate|"
            r"a video of|video of|video showing|demonstrate|"
            r"video tutorial|video guide|how does|how do)",
            "",
            question.strip(),
            flags=re.IGNORECASE,
        ).strip()

        cleaned = re.sub(r"\?+$", "", cleaned).strip()
        cleaned = re.sub(r"\b(please|for me|to me|me|a|an|the|look|in (?:motion|video|action))\b",
                         "", cleaned).strip()
        cleaned = " ".join(cleaned.split())

        if _UNSAFE_TOPICS.search(cleaned):
            return ""

        return cleaned[:200]

    @staticmethod
    def _build_caption(question: str) -> str:
        cleaned = re.sub(r"^(show me|can you|please|how to|video of|animate)", "",
                         question, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\?+$", "", cleaned).strip()
        return f"Video: {cleaned[:80].capitalize()}" if cleaned else "Generated video"
