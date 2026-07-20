import json
import openai
import re
import time
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
from backend.models import ReelPlan
from typing import Callable, Optional


MIN_COMMENTARY_WINDOW_SECONDS = 2.5


def _summarize_transcript_for_llm(transcript: list, max_total_chars: int = 3500) -> str:
    """
    Aggressively summarize a long transcript for LLM consumption.
    Keeps only the most informative segments within the char limit.
    """
    if not transcript:
        return ""

    full_text = ""
    for i, entry in enumerate(transcript):
        duration = entry['end'] - entry['start']
        if duration > 0.5:
            full_text += f"Seg {i} [{entry['start']:.1f}-{entry['end']:.1f}s]: {entry['text']}\n"

    if len(full_text) <= max_total_chars:
        return full_text

    print(f"[INFO] Transcript too long ({len(full_text)} chars), aggressive summarization...")

    # Score segments by informativeness
    scored = []
    filler_words = {"okay", "yeah", "right", "uh", "um", "oh", "so", "well", "huh", "like", "just", "wow", "ah", "hmm"}
    for i, entry in enumerate(transcript):
        duration = entry['end'] - entry['start']
        text = entry['text'].strip()
        words = text.split()
        word_count = len(words)

        # Score: longer segments with more words = more informative
        score = word_count
        if duration < 0.8:
            score -= 5
        if word_count <= 2 and text.lower() in filler_words:
            score -= 10
        if "?" in text:
            score += 3
        if any(w in text.lower() for w in ("but", "because", "actually", "important", "problem", "why", "how")):
            score += 3

        scored.append((score, i, entry["start"], entry["end"], text))

    # Sort by score descending, take top segments
    scored.sort(key=lambda x: -x[0])

    result_lines = []
    total_chars = 0
    for score, i, start, end, text in scored:
        line = f"Seg {i} [{start:.1f}-{end:.1f}s]: {text}\n"
        if total_chars + len(line) > max_total_chars:
            break
        result_lines.append((i, line))
        total_chars += len(line)

    # Re-sort by segment index to maintain chronological order
    result_lines.sort(key=lambda x: x[0])
    compressed = "".join(line for _, line in result_lines)

    print(f"[INFO] Transcript compressed from {len(full_text)} to {len(compressed)} chars "
          f"({len(result_lines)} of {len(transcript)} segments kept)")
    return compressed


def _call_llm(messages: list, progress_cb: Optional[Callable[[str, float], None]] = None) -> str:
    """Call NVIDIA LLM with primary model, retry with fallback on failure.
    Uses exponential backoff between retries."""
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Skipping LLM analysis and using local fallback."
        )

    client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, max_retries=0)
    models_to_try = [NVIDIA_MODEL]
    if NVIDIA_MODEL_FALLBACK:
        models_to_try.append(NVIDIA_MODEL_FALLBACK)
    print(f"[DEBUG] Resolved models_to_try at runtime: {models_to_try}")

    last_error = None
    backoff_delays = [3, 8, 15]

    for model in models_to_try:
        for attempt in range(2):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 32768,
                    "timeout": 240.0,
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
                        f"Finish reason: {finish_reason}. Refusal: {refusal}."
                    )
                finish_reason = response.choices[0].finish_reason
                print(f"[DEBUG] LLM finish_reason: {finish_reason}")
                truncated = raw_content[:300] + "..." if len(raw_content) > 300 else raw_content
                print(f"[DEBUG] LLM response preview: {truncated}")

                # If truncated by length, try to close the JSON
                if finish_reason == "length":
                    print(f"[WARN] LLM response was truncated (finish_reason=length, {len(raw_content)} chars)")
                    # Try to find a valid JSON prefix by balancing braces
                    raw_content = _try_repair_truncated_json(raw_content)
                    if raw_content:
                        print(f"[INFO] Repaired truncated JSON ({len(raw_content)} chars)")
                        return raw_content

                return raw_content.strip()
            except Exception as e:
                last_error = e
                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                print(f"[WARN] LLM call failed with model {model} (attempt {attempt+1}/2): {e}")
                if attempt == 0:
                    if progress_cb:
                        progress_cb(f"Model {model} failed, retrying in {delay}s...", 30)
                    print(f"[INFO] Waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    if progress_cb:
                        progress_cb(f"Model {model} failed twice, trying next model...", 30)
                    break

    raise RuntimeError(f"All NVIDIA models failed after retries. Last error: {last_error}")


def _try_repair_truncated_json(text: str) -> str:
    """Try to repair a truncated JSON by balancing braces and brackets."""
    if not text:
        return ""

    # Count opening/closing braces and brackets
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")

    # Add missing closing braces/brackets
    repaired = text.rstrip().rstrip(",")
    repaired += "}" * (open_braces - close_braces)
    repaired += "]" * (open_brackets - close_brackets)

    # Try to parse
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    # If still broken, try to find the last complete JSON object
    try:
        # Find the outermost object
        start = repaired.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(repaired)):
                if repaired[i] == "{":
                    depth += 1
                elif repaired[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = repaired[start:i+1]
                        json.loads(candidate)
                        return candidate
    except (json.JSONDecodeError, IndexError):
        pass

    return ""


def _build_reel_plan_prompt(video_title: str, video_description: str, transcript_text: str) -> str:
    """Build the full LLM prompt for reel_plan generation with persona."""
    return f"""You are a viral content strategist. Turn this video into scroll-stopping reels (<=90s each).

TITLE: {video_title}
DESC: {video_description[:300]}
TRANSCRIPT:
{transcript_text}

RULES:
1. HOOK (0-3s): Specific scroll-stopper. Formula: [specific moment] + [why it matters]. NEVER "This is amazing!" or "You won't believe this!"
2. ARC: SETUP (context) -> TENSION (stakes) -> PAYOFF (no narration over payoff). First clip gives context. Last clip resolves.
3. PRIORITIZE: emotional impact > surprise > skill > stakes > spectacle. Exclude filler.
4. COMMENTARY: Insider explaining what/why. NOT hype. ✓ "Notice his hands shaking." ✗ "This is crazy!"
5. NARRATION: ~1 per 15-20s. Min 2 (hook + 1). Max 3-5 for 45-90s. Most clips play on original audio. Silence between events. 2-3 sentences max.
6. Output complete valid JSON. Do NOT truncate.

OUTPUT ONLY JSON:
{{
  "reel_groups": [
    {{
      "group_index": 0,
      "group_reasoning": "Why this is a distinct compelling story unit",
      "estimated_duration_seconds": 45.0,
      "reel_summary": {{
        "title": "Scroll-stopping title",
        "source_understanding": "What the video is about",
        "narrative_angle": "Framing for this reel",
        "key_moment": "The payoff moment"
      }},
      "source_clips": [
        {{"source_start": 12.3, "source_end": 18.7, "reason": "Why this clip is essential"}},
        {{"source_start": 35.1, "source_end": 42.0, "reason": "What tension this adds"}}
      ],
      "narration_events": [
        {{"event_type": "hook", "reel_start": 0.0, "reel_end": 3.0, "text": "Scroll-stopping hook anchored to real moment", "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 8.7, "reel_end": 11.2, "text": "Insider observation explaining why this matters", "voice_id": null}}
      ]
    }}
  ]
}}"""


def _fallback_reel_plan(transcript: list, video_title: str) -> dict:
    """
    Fallback: generate 3-5 reel groups from transcript segments.
    Each group gets 3-4 clips forming a mini-story arc.
    """
    if not transcript:
        return {"reel_groups": []}

    # Score segments by informativeness
    scored_segments = []
    for i, entry in enumerate(transcript):
        duration = entry["end"] - entry["start"]
        text = entry["text"].strip()
        words = text.split()
        score = len(words)
        if duration >= 3.0:
            score += 2
        if "?" in text:
            score += 3
        if any(w in text.lower() for w in ("but", "because", "actually", "important", "problem", "why", "how", "win", "lose", "beat", "amazing", "incredible")):
            score += 2
        scored_segments.append((score, i, entry["start"], entry["end"], text, duration))

    scored_segments.sort(key=lambda x: -x[0])

    # Pick top segments, spread across the video timeline
    total_duration = transcript[-1]["end"] - transcript[0]["start"]
    groups = []
    clips_per_group = 3
    num_groups = min(5, max(3, len(transcript) // 20))

    for g in range(num_groups):
        group_clips = []
        group_duration = 0.0
        target_zone_start = (g / num_groups) * total_duration
        target_zone_end = ((g + 1) / num_groups) * total_duration

        # Find best segments in this time zone
        zone_segments = [s for s in scored_segments if target_zone_start <= s[2] <= target_zone_end]
        zone_segments.sort(key=lambda x: -x[0])

        for score, i, start, end, text, duration in zone_segments:
            if len(group_clips) >= clips_per_group:
                break
            if group_duration + duration > 25.0:
                continue
            group_clips.append({
                "source_start": start,
                "source_end": end,
                "reason": f"Key moment: {text[:80]}"
            })
            group_duration += duration

        # If zone had no good segments, grab from anywhere
        if not group_clips:
            for score, i, start, end, text, duration in scored_segments:
                if len(group_clips) >= clips_per_group:
                    break
                if group_duration + duration > 25.0:
                    continue
                group_clips.append({
                    "source_start": start,
                    "source_end": end,
                    "reason": f"Key moment: {text[:80]}"
                })
                group_duration += duration

        if not group_clips:
            continue

        hook_duration = 3.0
        commentary_duration = len(group_clips) * 3.0
        estimated = group_duration + hook_duration + commentary_duration

        group_title = f"{video_title[:50]} - Part {g+1}" if video_title else f"Part {g+1}"

        groups.append({
            "group_index": g,
            "group_reasoning": f"Fallback group {g+1}: {len(group_clips)} clips from video segment",
            "estimated_duration_seconds": min(estimated, 90.0),
            "reel_summary": {
                "title": group_title,
                "source_understanding": f"Key moments from {video_title[:60] if video_title else 'the video'}",
                "narrative_angle": f"Compelling moments from the video - Part {g+1}",
                "key_moment": group_clips[-1]["reason"][:60] if group_clips else "The climax"
            },
            "source_clips": group_clips,
            "narration_events": [
                {
                    "event_type": "hook",
                    "reel_start": 0.0,
                    "reel_end": hook_duration,
                    "text": f"Watch what happens in this {group_duration:.0f}-second moment — it changes everything.",
                    "voice_id": None
                }
            ] + [
                {
                    "event_type": "commentary",
                    "reel_start": hook_duration + sum(
                        (c["source_end"] - c["source_start"]) for c in group_clips[:j]
                    ) + j * 2.5,
                    "reel_end": hook_duration + sum(
                        (c["source_end"] - c["source_start"]) for c in group_clips[:j+1]
                    ) + j * 2.5 + 2.5,
                    "text": f"Notice what happens here — {c['reason'][:50]}",
                    "voice_id": None
                }
                for j, c in enumerate(group_clips)
            ]
        })

    if not groups:
        # Absolute fallback: one group with first few segments
        source_clips = []
        total_dur = 0.0
        for entry in transcript[:5]:
            dur = entry["end"] - entry["start"]
            if total_dur + dur > 30.0:
                break
            source_clips.append({
                "source_start": entry["start"],
                "source_end": entry["end"],
                "reason": f"Opening segment: {entry['text'][:60]}"
            })
            total_dur += dur

        groups.append({
            "group_index": 0,
            "group_reasoning": "Fallback: opening segments",
            "estimated_duration_seconds": min(total_dur + 10.0, 90.0),
            "reel_summary": {
                "title": video_title[:80] if video_title else "Untitled",
                "source_understanding": "Opening moments from the video",
                "narrative_angle": "Key moments from the video",
                "key_moment": source_clips[-1]["reason"][:60] if source_clips else "Opening"
            },
            "source_clips": source_clips,
            "narration_events": [
                {"event_type": "hook", "reel_start": 0.0, "reel_end": 3.0, "text": "What you're about to see changes how you understand this.", "voice_id": None}
            ] + [
                {"event_type": "commentary", "reel_start": 3.0 + j * 2.5 + sum((c["source_end"] - c["source_start"]) for c in source_clips[:j]), "reel_end": 3.0 + j * 2.5 + sum((c["source_end"] - c["source_start"]) for c in source_clips[:j+1]) + 2.5, "text": f"Notice what happens here — {c['reason'][:50]}", "voice_id": None}
                for j, c in enumerate(source_clips)
            ]
        })

    print(f"[INFO] Fallback generated {len(groups)} group(s) with {sum(len(g['source_clips']) for g in groups)} total clips")
    return {"reel_groups": groups}


def _extract_json_object(text: str) -> str:
    """Extract first JSON object from text, stripping markdown fences."""
    t = text.strip()
    if t.startswith("```json"):
        t = t[len("```json"):].strip()
    if t.startswith("```"):
        t = t[len("```"):].strip()
    if t.endswith("```"):
        t = t[:-len("```")].strip()

    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("No JSON object found in LLM response.")
    return m.group(0).strip()


def select_reel_plan(
    transcript: list,
    video_title: str,
    video_description: str,
    progress_cb: Optional[Callable[[str, float], None]] = None
) -> dict:
    """Main entry: returns reel_plan dict with reel_groups array."""
    if progress_cb:
        progress_cb("Preparing transcript for reel analysis...", 10)

    if not transcript:
        raise RuntimeError("Transcript is empty; cannot build reel plan.")

    # Aggressive transcript summarization
    transcript_text = _summarize_transcript_for_llm(transcript, max_total_chars=3500)

    description = (video_description or "")[:300]

    if progress_cb:
        progress_cb("Sending transcript to LLM for reel planning...", 30)

    prompt = _build_reel_plan_prompt(video_title, description, transcript_text)

    try:
        raw_content = _call_llm(
            [
                {"role": "system", "content": "You must respond with ONLY valid JSON. No explanations, no thinking, no text before or after the JSON object."},
                {"role": "user", "content": prompt}
            ],
            progress_cb,
        )
    except Exception as e:
        print(f"[WARN] LLM reel planning unavailable: {e}")
        print("[INFO] Using fallback reel plan (multi-group)")
        if progress_cb:
            progress_cb("Using fallback reel plan...", 50)
        return ReelPlan(**_fallback_reel_plan(transcript, video_title), is_fallback=True)

    if progress_cb:
        progress_cb("Parsing reel plan...", 80)

    try:
        raw_json = _extract_json_object(raw_content)
        reel_plan = json.loads(raw_json)
        truncated = raw_json[:500].encode('ascii', 'replace').decode()
        print(f"[DEBUG] Raw LLM reel_plan: {truncated}...")
    except Exception as e:
        print(f"[WARN] First LLM response failed to parse: {e}")
        raw_preview = raw_content[:2000].encode('ascii','replace').decode()
        print(f"[DEBUG] Raw content (first attempt): {raw_preview}")
        if progress_cb:
            progress_cb("LLM returned non-JSON. Retrying with stricter constraints...", 40)

        raw_content_retry = _call_llm(
            [
                {
                    "role": "user",
                    "content": prompt + "\n\nCRITICAL: Output ONLY the JSON object. No additional text."
                }
            ],
            progress_cb,
        )

        try:
            raw_json = _extract_json_object(raw_content_retry)
            reel_plan = json.loads(raw_json)
            retry_truncated = raw_json[:500].encode('ascii', 'replace').decode()
            print(f"[DEBUG] Raw LLM reel_plan (retry): {retry_truncated}...")
        except Exception as e2:
            retry_preview = raw_content_retry[:2000].encode('ascii','replace').decode()
            print(f"[DEBUG] Raw content (retry attempt): {retry_preview}")
            print(f"[WARN] LLM failed to produce JSON after retry: {e2}")
            print("[INFO] Using fallback reel plan")
            return ReelPlan(**_fallback_reel_plan(transcript, video_title), is_fallback=True)

    if not isinstance(reel_plan, dict) or "reel_groups" not in reel_plan:
        print(f"[WARN] LLM returned object without 'reel_groups' key: {reel_plan}")
        print("[INFO] Using fallback reel plan")
        return ReelPlan(**_fallback_reel_plan(transcript, video_title), is_fallback=True)

    groups = reel_plan["reel_groups"]
    if not isinstance(groups, list) or len(groups) == 0:
        print(f"[WARN] 'reel_groups' must be a non-empty array, got {type(groups)}. Using fallback.")
        return ReelPlan(**_fallback_reel_plan(transcript, video_title), is_fallback=True)

    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise RuntimeError(f"Group {i} must be an object")

        if "source_clips" not in group or not isinstance(group["source_clips"], list) or len(group["source_clips"]) == 0:
            print(f"[WARN] Group {i} missing valid 'source_clips'. Using fallback.")
            return ReelPlan(**_fallback_reel_plan(transcript, video_title), is_fallback=True)

        if "narration_events" not in group or not isinstance(group["narration_events"], list):
            print(f"[WARN] Group {i} missing 'narration_events'. Using fallback.")
            return ReelPlan(**_fallback_reel_plan(transcript, video_title), is_fallback=True)

        if group.get("estimated_duration_seconds", 0) > 90:
            print(f"[WARN] Group {i} estimated duration {group['estimated_duration_seconds']}s exceeds 90s cap")

        for event in group["narration_events"]:
            if event.get("event_type") == "hook" and event.get("reel_start", 1) != 0.0:
                print(f"[WARN] Group {i} hook must start at reel_start=0.0, got {event.get('reel_start')}")

    if progress_cb:
        progress_cb(f"Built reel plan with {len(groups)} group(s)", 100)

    return ReelPlan(**reel_plan)


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
    """Legacy flat clip selection (kept for compatibility)."""
    if progress_cb:
        progress_cb("Preparing transcript for analysis...", 10)

    if not transcript:
        raise RuntimeError("Transcript is empty; cannot select clips.")

    transcript_text = _summarize_transcript_for_llm(transcript, max_total_chars=3500)

    description = (video_description or "")[:300]

    if progress_cb:
        progress_cb("Sending transcript to LLM for clip selection...", 30)

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
        if t.startswith("```json"):
            t = t[len("```json") :].strip()
        if t.startswith("```"):
            t = t[len("```") :].strip()
        if t.endswith("```"):
            t = t[: -len("```")].strip()

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
            retry_preview = raw_content_retry[:2000].encode('ascii','replace').decode()
            print(f"[DEBUG] Raw content (retry): {retry_preview}")
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