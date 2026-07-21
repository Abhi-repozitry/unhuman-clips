import json
import re
from backend.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL
from backend.providers.llm import call_llm_sync
from typing import Callable, Optional


def _call_llm(messages: list, progress_cb: Optional[Callable[[str, float], None]] = None) -> str:
    """Call NVIDIA LLM with primary model, retry with fallback on failure."""
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Skipping LLM commentary and using local fallback."
        )

    try:
        return call_llm_sync(
            messages=messages,
            model=NVIDIA_MODEL,
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            temperature=0.1,
            max_tokens=4000,
        )
    except Exception as e:
        raise RuntimeError(f"All NVIDIA models failed. Last error: {e}") from e


def _clip_transcript_text(transcript: list, clip: dict) -> str:
    if not transcript:
        return ""
    start = clip.get("start", 0)
    end = clip.get("end", 0)
    parts = []
    for entry in transcript:
        if entry["end"] < start or entry["start"] > end:
            continue
        parts.append(entry["text"])
    return " ".join(parts)


def _extract_json_array(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```json"):
        raw = raw[len("```json") :].strip()
    if raw.startswith("```"):
        raw = raw[len("```") :].strip()
    if raw.endswith("```"):
        raw = raw[: -len("```")].strip()
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        raise ValueError("No JSON array found in commentary response.")
    return match.group(0)


def _fallback_commentary(clip_windows: list) -> list[dict]:
    hook_templates = [
        "The hidden layer is the real story.",
        "This is where the answer gets strange.",
        "The important part is not on the surface.",
        "This changes how the whole system feels.",
        "The weird part happens before the answer.",
    ]
    insight_templates = [
        "The output is just the visible tip of the process.",
        "That hidden work is what makes the moment matter.",
        "The takeaway is about process, not just words.",
        "This is why simple answers can hide complex reasoning.",
        "The real signal is what happens underneath.",
    ]
    ai_hooks = [
        "AI's hidden layer is the real story.",
        "The answer is not where reasoning starts.",
        "Claude's visible words are only the surface.",
        "This is where AI starts feeling strange.",
        "The model is doing more than talking.",
    ]
    ai_insights = [
        "The answer is polished, but the reasoning is buried underneath.",
        "The useful question is what happened before the words appeared.",
        "That gap between process and output is the whole point.",
        "This is why model behavior can look simple but feel deep.",
        "The real signal is inside the computation, not the sentence.",
    ]
    human_hooks = [
        "Your mind is hiding most of the work.",
        "The surface thought is not the whole mind.",
        "Most thinking happens where you cannot see it.",
    ]
    human_insights = [
        "The surface thought is only the part you can describe.",
        "That hidden processing explains why the analogy works.",
        "The invisible work gives the visible thought its shape.",
    ]
    result = []
    for i, window in enumerate(clip_windows):
        topic = (window.get("topic") or "").lower()
        hook = hook_templates[i % len(hook_templates)]
        insight = insight_templates[i % len(insight_templates)]
        if "ai" in topic or "model" in topic or "claude" in topic:
            hook = ai_hooks[i % len(ai_hooks)]
            insight = ai_insights[i % len(ai_insights)]
        elif "human" in topic or "mind" in topic or "thinking" in topic:
            hook = human_hooks[i % len(human_hooks)]
            insight = human_insights[i % len(human_insights)]
        elif "question" in topic:
            hook = "This question makes the whole clip matter."
            insight = "The answer depends on what is happening below the surface."
        result.append({
            "clip_index": i,
            "hook_text": hook,
            "insight_text": insight,
        })
    return result


def write_commentary(clip_windows: list, video_title: str, transcript: Optional[list] = None, progress_cb: Optional[Callable[[str, float], None]] = None) -> list[dict]:
    if progress_cb:
        progress_cb("Preparing clip descriptions for commentary...", 10)

    clips_text = ""
    for i, window in enumerate(clip_windows):
        clip_text = _clip_transcript_text(transcript or [], window)
        clips_text += (
            f"Clip {i}:\n"
            f"Topic: {window.get('topic', '')}\n"
            f"Why it hooks: {window.get('why_it_hooks', '')}\n"
            f"Payoff: {window.get('payoff', '')}\n"
            f"Reason: {window.get('reason', '')}\n"
            f"Transcript inside clip: {clip_text[:900]}\n\n"
        )

    if progress_cb:
        progress_cb("Generating commentary script with LLM...", 40)

    try:
        raw_content = _call_llm([
            {
                "role": "user",
                "content": f"""Video title: {video_title}

Selected clips with topic analysis and transcript context:
{clips_text}

For each clip, write two voiceover lines:
1. hook_text: under 10 words. It must grab attention in the first 3 seconds before the clip body.
2. insight_text: under 14 words. It must add context, opinion, stakes, or curiosity after the clip.

Rules:
- Do NOT repeat or paraphrase the transcript.
- Do NOT narrate what is visibly happening.
- Do NOT use generic phrases like "watch this", "let's see", "this is interesting", or "here's why".
- Make each line specific to the video topic and selected moment.
- Use curiosity, contrast, stakes, or interpretation.

Respond with ONLY a raw JSON array, no markdown fences:
[
  {{"clip_index": int, "hook_text": string, "insight_text": string}}
]

The clip_index must match the order of clip_windows (0, 1, 2, ...)."""
            }
        ], progress_cb)
    except Exception as e:
        print(f"[WARN] LLM commentary unavailable: {e}")
        result = _fallback_commentary(clip_windows)
        if progress_cb:
            progress_cb(f"Generated {len(result)} fallback hook/insight pairs", 100)
        return result

    if progress_cb:
        progress_cb("Parsing commentary response...", 80)

    try:
        commentary_data = json.loads(_extract_json_array(raw_content))
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Failed to parse JSON response: {e}\nRaw response: {raw_content}") from e

    if not isinstance(commentary_data, list):
        raise RuntimeError(f"Expected JSON array, got {type(commentary_data)}\nRaw: {raw_content}")

    result = []
    for item in commentary_data:
        if not isinstance(item, dict):
            raise RuntimeError(f"Expected object in array, got {type(item)}\nRaw: {raw_content}")

        clip_index = item.get("clip_index")
        hook_text = str(item.get("hook_text", "")).strip()
        insight_text = str(item.get("insight_text", "")).strip()

        if not isinstance(clip_index, int):
            raise RuntimeError(f"clip_index must be an integer\nRaw: {raw_content}")
        if not hook_text or not insight_text:
            fallback = _fallback_commentary([clip_windows[clip_index]])[0]
            hook_text = hook_text or fallback["hook_text"]
            insight_text = insight_text or fallback["insight_text"]

        result.append({
            "clip_index": clip_index,
            "hook_text": hook_text,
            "insight_text": insight_text,
        })

    result.sort(key=lambda x: x["clip_index"])

    if progress_cb:
        progress_cb(f"Generated {len(result)} hook/insight pairs", 100)

    return result
