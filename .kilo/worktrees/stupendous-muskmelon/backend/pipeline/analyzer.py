import json
import openai
import re
from backend.config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    NVIDIA_MODEL_FALLBACK,
    CLIP_COUNT_MIN,
    CLIP_COUNT_MAX,
    CLIP_DURATION_SOFT_MIN,
    CLIP_DURATION_SOFT_MAX,
    HOOK_SECONDS,
    INSIGHT_SECONDS_MAX,
    MAX_OUTPUT_DURATION,
)
from typing import Callable, Optional


def _call_llm(messages: list, progress_cb: Optional[Callable[[str, float], None]] = None) -> str:
    """Call NVIDIA LLM with primary model, retry with fallback on failure."""
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Skipping LLM analysis and using local fallback."
        )

    client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    models_to_try = [NVIDIA_MODEL]
    if NVIDIA_MODEL_FALLBACK:
        models_to_try.append(NVIDIA_MODEL_FALLBACK)

    last_error = None
    for model in models_to_try:
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1200,
            }
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass
            response = client.chat.completions.create(**kwargs)
            raw_content = response.choices[0].message.content
            if raw_content is None:
                finish_reason = response.choices[0].finish_reason
                refusal = getattr(response.choices[0].message, 'refusal', None)
                raise RuntimeError(
                    f"NVIDIA API returned empty content. "
                    f"Finish reason: {finish_reason}. "
                    f"Refusal: {refusal}. "
                    f"Full response: {response}"
                )
            return raw_content.strip()
        except Exception as e:
            last_error = e
            if progress_cb:
                progress_cb(f"Model {model} failed, trying fallback...", 30)
            print(f"[WARN] LLM call failed with model {model}: {e}")
            continue

    raise RuntimeError(f"All NVIDIA models failed. Last error: {last_error}")


def _fallback_clip_selection(transcript: list) -> list[dict]:
    """Fallback: group transcript into coherent 10-15s windows."""
    if not transcript:
        return []

    def _window_score(text: str) -> int:
        lowered = text.lower()
        score = 0
        strong_terms = (
            "but", "why", "how", "because", "question", "answer", "surprising",
            "hidden", "under the hood", "inside", "difference", "changes",
            "important", "problem", "actually", "realizing", "wondered", "could",
        )
        for term in strong_terms:
            if term in lowered:
                score += 2
        if "?" in text:
            score += 3
        if any(word in lowered for word in ("ai", "model", "neural", "brain", "claude")):
            score += 3
        return score

    candidates = []
    for start_seg in range(len(transcript)):
        end_seg = start_seg
        start = transcript[start_seg]["start"]
        while end_seg + 1 < len(transcript) and transcript[end_seg]["end"] - start < CLIP_DURATION_SOFT_MIN:
            end_seg += 1
        while end_seg + 1 < len(transcript) and transcript[end_seg + 1]["end"] - start <= CLIP_DURATION_SOFT_MAX:
            end_seg += 1
        duration = transcript[end_seg]["end"] - start

        duration = transcript[end_seg]["end"] - start
        if not (CLIP_DURATION_SOFT_MIN <= duration <= CLIP_DURATION_SOFT_MAX):
            continue
        text = " ".join(entry["text"] for entry in transcript[start_seg:end_seg + 1])
        candidates.append({
            "start_segment": start_seg,
            "end_segment": end_seg,
            "text": text,
            "score": _window_score(text),
            "duration": duration,
        })

    clips = []
    total_planned = 0.0
    used_ranges = []
    for candidate in sorted(candidates, key=lambda c: (-c["score"], c["start_segment"])):
        if len(clips) >= CLIP_COUNT_MAX:
            break
        if any(not (candidate["end_segment"] < start or candidate["start_segment"] > end) for start, end in used_ranges):
            continue

        planned = candidate["duration"] + HOOK_SECONDS + INSIGHT_SECONDS_MAX
        if total_planned + planned > MAX_OUTPUT_DURATION:
            continue

        text = candidate["text"]
        clips.append({
            "start_segment": candidate["start_segment"],
            "end_segment": candidate["end_segment"],
            "topic": _summarize_fallback_topic(text),
            "why_it_hooks": "This moment contains a high-signal explanation or question.",
            "payoff": _summarize_fallback_payoff(text),
            "reason": f"Fallback scored window: {text[:80]}",
        })
        used_ranges.append((candidate["start_segment"], candidate["end_segment"]))
        total_planned += planned

    clips.sort(key=lambda c: c["start_segment"])

    print(f"[INFO] Fallback selected {len(clips)} clips, planned duration ~{total_planned:.1f}s")
    return clips


def _summarize_fallback_topic(text: str) -> str:
    lowered = text.lower()
    if "ai" in lowered or "model" in lowered or "neural" in lowered or "claude" in lowered:
        return "AI reasoning has hidden layers beneath the visible answer"
    if "brain" in lowered or "mind" in lowered or "conscious" in lowered:
        return "Human thinking has surface thoughts and hidden processing"
    if "question" in lowered or "wondered" in lowered or "could" in lowered:
        return "The clip raises the central question viewers need answered"
    return "A key idea that changes how the topic makes sense"


def _summarize_fallback_payoff(text: str) -> str:
    lowered = text.lower()
    if "under the hood" in lowered or "inside" in lowered:
        return "The important work happens inside the system, not just in the output"
    if "unconscious" in lowered or "surface" in lowered:
        return "The visible thought is only a small part of the process"
    if "ai" in lowered or "model" in lowered:
        return "The same hidden/visible split may apply to AI behavior"
    return "The moment gives viewers a useful frame for the rest of the video"


def _normalize_clip_range(transcript: list, start_seg: int, end_seg: int) -> tuple[int, int]:
    max_idx = len(transcript) - 1
    start_seg = min(max(start_seg, 0), max_idx)
    end_seg = min(max(end_seg, start_seg), max_idx)

    while transcript[end_seg]["end"] - transcript[start_seg]["start"] < CLIP_DURATION_SOFT_MIN:
        can_expand_after = end_seg < max_idx
        can_expand_before = start_seg > 0
        if can_expand_after:
            end_seg += 1
        elif can_expand_before:
            start_seg -= 1
        else:
            break

    while transcript[end_seg]["end"] - transcript[start_seg]["start"] > CLIP_DURATION_SOFT_MAX and end_seg > start_seg:
        remove_after_duration = transcript[end_seg - 1]["end"] - transcript[start_seg]["start"]
        remove_before_duration = transcript[end_seg]["end"] - transcript[start_seg + 1]["start"]
        if abs(remove_after_duration - CLIP_DURATION_SOFT_MAX) <= abs(remove_before_duration - CLIP_DURATION_SOFT_MAX):
            end_seg -= 1
        else:
            start_seg += 1

    return start_seg, end_seg


def select_clips(transcript: list, video_title: str, video_description: str, progress_cb: Optional[Callable[[str, float], None]] = None) -> list[dict]:
    if progress_cb:
        progress_cb("Preparing transcript for analysis...", 10)

    if not transcript:
        raise RuntimeError("Transcript is empty; cannot select clips.")

    transcript_text = ""
    for i, entry in enumerate(transcript):
        duration = entry['end'] - entry['start']
        if duration > 0.5:
            transcript_text += f"Seg {i} [{entry['start']:.1f}-{entry['end']:.1f}s]: {entry['text']}\n"

    description = (video_description or "")[:500]

    if progress_cb:
        progress_cb("Sending transcript to LLM for clip selection...", 30)

    # Force strict JSON output from the model.
    prompt = f"""You are a JSON-only output machine. You MUST output ONLY valid JSON. No explanations, no thinking, no text before or after.

Given the video title: {video_title}
And description: {description[:500]}

Transcript (segment index: [start-end] text):
{transcript_text}

Select {CLIP_COUNT_MIN}-{CLIP_COUNT_MAX} standalone short-form moments. Each selected range should be {CLIP_DURATION_SOFT_MIN:.0f}-{CLIP_DURATION_SOFT_MAX:.0f} seconds when possible.

Choose moments that will make sense after a 3-second hook, then the original clip, then a short topical insight. Prefer moments with clear stakes, contrast, surprising claims, useful explanations, or a payoff. Do NOT pick random fragments or single sentences without context.

The final edit has a hard cap of {MAX_OUTPUT_DURATION} seconds. Estimate each moment as clip duration + {HOOK_SECONDS:.0f}s hook + up to {INSIGHT_SECONDS_MAX:.0f}s insight.

Output this EXACT format with NO other text:
[
  {{"start_segment": 0, "end_segment": 2, "topic": "specific topic", "why_it_hooks": "why viewers care", "payoff": "what the viewer learns", "reason": "brief reason"}},
  {{"start_segment": 5, "end_segment": 7, "topic": "specific topic", "why_it_hooks": "why viewers care", "payoff": "what the viewer learns", "reason": "brief reason"}}
]"""

    try:
        raw_content = _call_llm(
            [
                {"role": "system", "content": "You must respond with ONLY valid JSON. No explanations, no thinking, no text before or after the JSON."},
                {"role": "user", "content": prompt}
            ],
            progress_cb,
        )
    except Exception as e:
        print(f"[WARN] LLM clip selection unavailable: {e}")
        print("[INFO] Using fallback clip selection based on transcript analysis")
        if progress_cb:
            progress_cb("Using fallback clip selection...", 50)
        raw_content = "[]"
        clip_data = _fallback_clip_selection(transcript)
    else:
        clip_data = None

    def _extract_json_array(text: str) -> str:
        t = text.strip()
        # Strip common markdown wrappers
        if t.startswith("```json"):
            t = t[len("```json") :].strip()
        if t.startswith("```"):
            t = t[len("```") :].strip()
        if t.endswith("```"):
            t = t[: -len("```")].strip()

        # If it's not pure JSON, recover the first JSON array in the text.
        m = re.search(r"\[[\s\S]*\]", t)
        if not m:
            raise ValueError("No JSON array found in LLM response.")
        return m.group(0).strip()

    if progress_cb:
        progress_cb("Parsing clip selections...", 80)

    try:
        if clip_data is None:
            raw_json_array = _extract_json_array(raw_content)
            clip_data = json.loads(raw_json_array)
    except Exception as e:
        # Retry once with an even stricter instruction.
        if progress_cb:
            progress_cb("LLM returned non-JSON. Retrying with stricter output constraints...", 40)

        raw_content_retry = _call_llm(
            [
                {
                    "role": "user",
                    "content": prompt + "\n\nCRITICAL: Output ONLY the JSON array. Nothing else."
                }
            ],
            progress_cb,
        )

        try:
            raw_json_array = _extract_json_array(raw_content_retry)
            clip_data = json.loads(raw_json_array)
        except Exception as e2:
            print(f"[WARN] LLM failed to produce JSON after retry: {e2}")
            print("[INFO] Using fallback clip selection based on transcript analysis")
            clip_data = _fallback_clip_selection(transcript)
            if progress_cb:
                progress_cb("Using fallback clip selection...", 50)

    if not isinstance(clip_data, list):
        raise RuntimeError(f"Expected JSON array, got {type(clip_data)}\nRaw: {raw_content}")

    clips = []
    for item in clip_data:
        if not isinstance(item, dict):
            raise RuntimeError(f"Expected object in array, got {type(item)}\nRaw: {raw_content}")

        start_seg = item.get("start_segment")
        end_seg = item.get("end_segment")

        if not isinstance(start_seg, int) or not isinstance(end_seg, int):
            raise RuntimeError(f"start_segment and end_segment must be integers\nRaw: {raw_content}")
        start_seg, end_seg = _normalize_clip_range(transcript, start_seg, end_seg)

        actual_start = transcript[start_seg]["start"]
        actual_end = transcript[end_seg]["end"]
        duration = actual_end - actual_start
        planned_duration = duration + HOOK_SECONDS + INSIGHT_SECONDS_MAX
        if sum((c["end"] - c["start"]) + HOOK_SECONDS + INSIGHT_SECONDS_MAX for c in clips) + planned_duration > MAX_OUTPUT_DURATION:
            print(f"[INFO] Skipping clip {start_seg}-{end_seg}; planned output would exceed {MAX_OUTPUT_DURATION}s")
            continue

        if duration < CLIP_DURATION_SOFT_MIN or duration > CLIP_DURATION_SOFT_MAX:
            print(f"[WARNING] Clip {start_seg}-{end_seg} duration {duration:.1f}s outside soft range [{CLIP_DURATION_SOFT_MIN}-{CLIP_DURATION_SOFT_MAX}]")

        clips.append({
            "start": actual_start,
            "end": actual_end,
            "start_segment": start_seg,
            "end_segment": end_seg,
            "topic": item.get("topic", ""),
            "why_it_hooks": item.get("why_it_hooks", ""),
            "payoff": item.get("payoff", ""),
            "reason": item.get("reason", ""),
        })

    if progress_cb:
        progress_cb(f"Selected {len(clips)} clips", 100)

    return clips
