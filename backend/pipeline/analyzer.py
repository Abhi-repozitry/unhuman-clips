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


def _call_llm(messages: list, progress_cb: Optional[Callable[[str, float], None]] = None) -> str:
    """Call NVIDIA LLM with primary model, retry with fallback on failure."""
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Skipping LLM analysis and using local fallback."
        )

    client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, max_retries=0)
    models_to_try = [NVIDIA_MODEL]
    if NVIDIA_MODEL_FALLBACK:
        models_to_try.append(NVIDIA_MODEL_FALLBACK)

    last_error = None
    for model in models_to_try:
        for attempt in range(2):  # one retry per model before falling through
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 8000,
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
                print(f"[DEBUG] LLM finish_reason: {response.choices[0].finish_reason}")
                return raw_content.strip()
            except Exception as e:
                last_error = e
                print(f"[WARN] LLM call failed with model {model} (attempt {attempt+1}/2): {e}")
                if attempt == 0:
                    if progress_cb:
                        progress_cb(f"Model {model} failed, retrying...", 30)
                    time.sleep(3)
                    continue
                else:
                    if progress_cb:
                        progress_cb(f"Model {model} failed twice, trying next model...", 30)
                    break

    raise RuntimeError(f"All NVIDIA models failed after retries. Last error: {last_error}")


def _build_reel_plan_prompt(video_title: str, video_description: str, transcript_text: str) -> str:
    """Build the full LLM prompt for reel_plan generation with persona."""
    return f"""You are a sharp insider analyst. Analyze this video and produce a reel_plan.

VIDEO TITLE: {video_title}
VIDEO DESCRIPTION: {video_description[:500]}

TRANSCRIPT (segment index: [start-end] text):
{transcript_text}

TASK:
1. Understand what this video is actually about — the core story, argument, or moments.
2. Decide: does it tell ONE cohesive story (one group), or contain MULTIPLE distinct, self-contained moments (multiple groups)?
3. For each group, output a reel_plan object.

CONSTRAINTS:
- Each group's final output MUST be <= 90 seconds.
- Each group gets its own 0-based timeline (hook starts at reel_start=0.0 for THAT group).
- MIN_COMMENTARY_WINDOW_SECONDS = 2.5 — commentary blocks must have room to breathe.
- Hook per group: ~3s. Commentary per clip: ~2-4s. Clip duration from source.
- Narration events are TIME WINDOWS on the continuous video, not separate visual segments.
- Source clips play back-to-back continuously; narration audio/captions overlay on top.

NARRATIVE ARC — every group MUST have a clear purpose, not just related clips:
- The group's source_clips, IN ORDER, must form a complete arc: SETUP (establish
  what's happening / why we're watching) -> DEVELOPMENT (tension, stakes, or
  progression) -> PAYOFF (the resolution — the reason this reel exists).
- The FIRST clip must give the viewer enough context to understand what's going
  on. Do not open on a clip that starts mid-sentence or mid-action with zero
  setup — the viewer has no context from the rest of the source video.
- The LAST clip must deliver an actual resolution or payoff — not just
  whichever clip happened to fit before the 90s cap. If the real payoff moment
  doesn't fit within 90s alongside enough setup, trim the setup/middle instead
  of cutting off before the payoff.
- reel_summary.key_moment must describe a genuine resolution — an answer, an
  outcome, a reveal — not "the middle of an ongoing scene."
- Prefer clip boundaries that align to natural sentence/thought breaks in the
  transcript over arbitrary timestamps, so clips don't start or end mid-word.
- If the available footage genuinely has no clear beginning-middle-end, don't
  force a group out of it — leave it out rather than produce a reel that goes
  nowhere.

PERSONA GUIDE - VOICE: sharp insider who explains what's happening and why it
matters — not a wall-to-wall commentary track.

HOOK (event_type="hook", one per group, reel_start=0.0):
A direct "check this out" cold open — confident, pulls the viewer in, no
vague hype claims.
✓ GOOD: "Check this out.", "Watch what happens at the media wall.", "This
almost didn't happen."
✗ BAD (hype with no content): "This is amazing!", "You won't believe what
happens next.", "The truth will shock you."

COMMENTARY (event_type="commentary"):
✓ GOOD — mixes brief context/explanation (what's happening, why) with insider
observation:
- "He's been turned away twice already — this is the last shot before the gates close."
- "Notice the politician's hands — they're shaking before the answer, then perfectly still after."
- "The code comment says 'todo' from 2018. It's still there."
✗ BAD — generic hype, nothing anchored to the footage:
- "This is amazing!"
- "You won't believe what happens next."
- "This changes everything."

NARRATION BUDGET — do not narrate over everything:
- Total narration events per group (hook + commentary): roughly 1 per 15-20s
  of reel runtime. Minimum 2 (hook + 1 commentary). Typically 3-5 for a
  45-90s reel. Most clips should play on their original audio alone — do NOT
  add a commentary event for every clip.
- NEVER place a narration event over the group's own key_moment / payoff (the
  moment reel_summary.key_moment describes). That moment plays on original
  audio only — no hook, no commentary. Let it land on its own.
- Leave real silent gaps between narration events — not back-to-back.

RULES:
1. Every line MUST be anchored to a specific visible moment or exact transcript quote.
2. If you can't point to WHAT the viewer should see or hear, DELETE the line.
3. Structure: short context/observation sentence, then a closer explaining WHY it matters.
4. Voice: insider explaining the story as it unfolds — not a hype person, not silent narration either.
5. Length: each narration block = 2-3 sentences max, never more than 4.
6. The budget and silence rules above are hard constraints, not suggestions.

OUTPUT FORMAT — JSON OBJECT ONLY, NO MARKDOWN, NO EXPLANATION:
{{
  "reel_groups": [
    {{
      "group_index": 0,
      "group_reasoning": "Why this is a distinct story unit",
      "estimated_duration_seconds": 45.0,
      "reel_summary": {{
        "title": "Hook-worthy title",
        "source_understanding": "What the source video is about",
        "narrative_angle": "Insider framing for this reel",
        "key_moment": "The payoff moment"
      }},
      "source_clips": [
        {{"source_start": 12.3, "source_end": 18.7, "reason": "Specific reason this clip matters"}},
        {{"source_start": 35.1, "source_end": 42.0, "reason": "Another reason"}}
      ],
      "narration_events": [
        {{
          "event_type": "hook",
          "reel_start": 0.0,
          "reel_end": 3.0,
          "text": "Sharp insider hook text anchored to visible moment",
          "voice_id": null
        }},
        {{
          "event_type": "commentary",
          "reel_start": 8.7,
          "reel_end": 11.2,
          "text": "Short observation. Closer explaining why it matters.",
          "voice_id": null
        }}
      ]
    }}
  ]
}}"""


def _fallback_reel_plan(transcript: list, video_title: str) -> dict:
    """Fallback: single group covering first viable clips when LLM unavailable."""
    if not transcript:
        return {"reel_groups": []}

    source_clips = []
    total_duration = 0.0
    max_total = 30.0

    for i, entry in enumerate(transcript):
        if len(source_clips) >= 4:
            break
        duration = entry["end"] - entry["start"]
        if duration >= 2.0 and total_duration + duration <= max_total:
            source_clips.append({
                "source_start": entry["start"],
                "source_end": entry["end"],
                "reason": f"Fallback segment {i}: {entry['text'][:60]}"
            })
            total_duration += duration

    if not source_clips and transcript:
        entry = transcript[0]
        source_clips.append({
            "source_start": entry["start"],
            "source_end": min(entry["end"], entry["start"] + 8.0),
            "reason": "Fallback: first segment"
        })
        total_duration = source_clips[0]["source_end"] - source_clips[0]["source_start"]

    hook_duration = 3.0
    commentary_duration = len(source_clips) * 3.0
    estimated = total_duration + hook_duration + commentary_duration

    return {
        "reel_groups": [{
            "group_index": 0,
            "group_reasoning": "Fallback: single cohesive unit (LLM unavailable)",
            "estimated_duration_seconds": min(estimated, 90.0),
            "reel_summary": {
                "title": video_title[:80] if video_title else "Untitled",
                "source_understanding": "Auto-selected segments from transcript",
                "narrative_angle": "Key moments from the video",
                "key_moment": source_clips[0]["reason"] if source_clips else "Opening"
            },
            "source_clips": source_clips,
            "narration_events": [
                {
                    "event_type": "hook",
                    "reel_start": 0.0,
                    "reel_end": hook_duration,
                    "text": "What you're about to see changes how you understand this.",
                    "voice_id": None
                }
            ] + [
                {
                    "event_type": "commentary",
                    "reel_start": hook_duration + sum(
                        (c["source_end"] - c["source_start"]) for c in source_clips[:j]
                    ) + j * 2.5,
                    "reel_end": hook_duration + sum(
                        (c["source_end"] - c["source_start"]) for c in source_clips[:j+1]
                    ) + j * 2.5 + 2.5,
                    "text": f"Notice what happens here — {c['reason'][:50]}",
                    "voice_id": None
                }
                for j, c in enumerate(source_clips)
            ]
        }]
    }


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

    transcript_text = ""
    for i, entry in enumerate(transcript):
        duration = entry['end'] - entry['start']
        if duration > 0.5:
            transcript_text += f"Seg {i} [{entry['start']:.1f}-{entry['end']:.1f}s]: {entry['text']}\n"

    description = (video_description or "")[:500]

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
        print("[INFO] Using fallback reel plan (single group)")
        if progress_cb:
            progress_cb("Using fallback reel plan...", 50)
        return ReelPlan(**_fallback_reel_plan(transcript, video_title))

    if progress_cb:
        progress_cb("Parsing reel plan...", 80)

    try:
        raw_json = _extract_json_object(raw_content)
        reel_plan = json.loads(raw_json)
        print(f"[DEBUG] Raw LLM reel_plan: {raw_json[:500].encode('ascii', 'replace').decode()}...")
    except Exception as e:
        print(f"[WARN] First LLM response failed to parse: {e}")
        print(f"[DEBUG] Raw content (first attempt): {raw_content[:2000].encode('ascii','replace').decode()}")
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
            print(f"[DEBUG] Raw LLM reel_plan (retry): {raw_json[:500].encode('ascii', 'replace').decode()}...")
        except Exception as e2:
            print(f"[DEBUG] Raw content (retry attempt): {raw_content_retry[:2000].encode('ascii','replace').decode()}")
            print(f"[WARN] LLM failed to produce JSON after retry: {e2}")
            print("[INFO] Using fallback reel plan")
            return ReelPlan(**_fallback_reel_plan(transcript, video_title))

    if not isinstance(reel_plan, dict) or "reel_groups" not in reel_plan:
        raise RuntimeError(f"Expected object with 'reel_groups' array, got: {reel_plan}")

    groups = reel_plan["reel_groups"]
    if not isinstance(groups, list):
        raise RuntimeError(f"'reel_groups' must be an array, got {type(groups)}")

    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise RuntimeError(f"Group {i} must be an object")

        if "source_clips" not in group or not isinstance(group["source_clips"], list):
            raise RuntimeError(f"Group {i} missing 'source_clips' array")

        if "narration_events" not in group or not isinstance(group["narration_events"], list):
            raise RuntimeError(f"Group {i} missing 'narration_events' array")

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

    transcript_text = ""
    for i, entry in enumerate(transcript):
        duration = entry['end'] - entry['start']
        if duration > 0.5:
            transcript_text += f"Seg {i} [{entry['start']:.1f}-{entry['end']:.1f}s]: {entry['text']}\n"

    description = (video_description or "")[:500]

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
