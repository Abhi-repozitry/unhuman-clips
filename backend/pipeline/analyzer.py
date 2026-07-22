import json
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
    MIN_OUTPUT_DURATION,
    MAX_OUTPUT_DURATION,
)
from backend.models import ReelPlan, LLMInteraction
from backend.providers.llm import call_llm_sync
from typing import Any, Callable, Optional, List


def _format_full_transcript(transcript: list) -> str:
    """Pass the 100% full, un-chunked transcript directly to the LLM.
    The primary LLM (stepfun-ai/step-3.7-flash) has a 256k token context window; full transcripts fit with ease."""
    if not transcript:
        return ""
    lines = []
    for i, entry in enumerate(transcript):
        start = entry.get("start", 0.0)
        end = entry.get("end", 0.0)
        text = entry.get("text", "").strip()
        if text:
            lines.append(f"Seg {i} [{start:.1f}-{end:.1f}s]: {text}")
    full_text = "\n".join(lines)
    print(f"[INFO] Passing 100% FULL transcript to LLM ({len(lines)} segments, {len(full_text)} chars — NO CHUNKING)")
    return full_text


def _summarize_transcript_for_llm(transcript: list, max_total_chars: int = 1000000) -> str:
    """Backward-compatible alias: passes full un-chunked transcript."""
    return _format_full_transcript(transcript)


def _call_llm(messages: list, progress_cb: Optional[Callable[[str, float], None]] = None,
              reporter: Optional[Any] = None,
              interactions: Optional[List[LLMInteraction]] = None,
              stage_name: str = "reel_plan") -> str:
    """Call NVIDIA LLM with primary model, retry with fallback on failure.
    Uses exponential backoff between retries. Now collects LLMInteraction records
    for rich UI display and logs via reporter."""
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Skipping LLM analysis and using local fallback."
        )

    models_to_try = [NVIDIA_MODEL]
    if NVIDIA_MODEL_FALLBACK and NVIDIA_MODEL_FALLBACK != NVIDIA_MODEL:
        models_to_try.append(NVIDIA_MODEL_FALLBACK)

    print(f"[DEBUG] Resolved models_to_try at runtime: {models_to_try}")

    last_error = None
    for attempt, model in enumerate(models_to_try):
        try:
            print(f"[INFO] Calling LLM with model: {model}")
            raw_content = call_llm_sync(
                messages=messages,
                model=model,
                api_key=NVIDIA_API_KEY,
                base_url=NVIDIA_BASE_URL,
                temperature=0.0,
                max_tokens=131072,
                timeout=480.0,
                reporter=reporter,
                interactions=interactions,
                stage_name=stage_name,
            )
            truncated = raw_content[:300] + "..." if len(raw_content) > 300 else raw_content
            print(f"[DEBUG] LLM response preview (model {model}): {truncated}")

            # Broadcast updated interactions after each successful call
            if reporter and interactions is not None:
                reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in interactions])

            # Log full raw content to a debug file
            try:
                from backend.config import WORKING_DIR
                debug_path = WORKING_DIR / f"llm_debug_{int(time.time())}.txt"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(raw_content)
                print(f"[DEBUG] Full LLM raw output saved to {debug_path}")
            except Exception as log_e:
                print(f"[WARN] Failed to write LLM debug log: {log_e}")

            return raw_content.strip()
        except Exception as e:
            print(f"[WARN] LLM call failed with model {model}: {e}")
            last_error = e

    raise RuntimeError(f"All NVIDIA models failed after retries. Last error: {last_error}") from last_error


def _try_repair_truncated_json(text: str) -> str:
    """Try to repair a truncated JSON by balancing braces and brackets,
    fixing trailing commas, and closing unclosed string quotes."""
    if not text:
        return ""

    # 1. Fix trailing commas before closing brackets/braces first
    repaired = re.sub(r',\s*([}\]])', r'\1', text.strip())

    # If truncated inside a string literal, close the quote
    # Count unescaped double quotes
    unescaped_quotes = len(re.findall(r'(?<!\\)"', repaired))
    if unescaped_quotes % 2 != 0:
        repaired += '"'

    # Count opening/closing braces and brackets
    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")

    # Add missing closing braces/brackets
    repaired += "}" * max(0, open_braces - close_braces)
    repaired += "]" * max(0, open_brackets - close_brackets)

    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    # 2. Try to find the last complete JSON object by scanning backwards
    try:
        for start_pos in [repaired.find("{"), repaired.find("[")]:
            if start_pos < 0:
                continue
            depth = 0
            in_string = False
            escape = False
            for i in range(start_pos, len(repaired)):
                ch = repaired[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in ('{', '['):
                    depth += 1
                elif ch in ('}', ']'):
                    depth -= 1
                    if depth == 0:
                        candidate = repaired[start_pos:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            continue
    except (json.JSONDecodeError, IndexError):
        pass

    # 3. Last resort: try to find any valid JSON substring
    try:
        for start_pos in range(len(repaired)):
            if repaired[start_pos] in ('{', '['):
                for end_pos in range(len(repaired), start_pos, -1):
                    candidate = repaired[start_pos:end_pos]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
    except (json.JSONDecodeError, IndexError):
        pass

    return ""


def _compute_group_count_target(source_duration_seconds: float) -> tuple[int, int]:
    """
    Group count target for 5-40 minute long-form videos.

    min_groups scales with video length to ensure long videos produce multiple reels.
    max_groups is a ceiling the LLM can reach up to.

      - <5 min     → (1, 4)
      - 5-10 min   → (3, 6)
      - 10-20 min  → (4, 8)
      - 20-40+ min → (5, 12)
    """
    if source_duration_seconds <= 300:
        return (1, 4)
    if source_duration_seconds <= 600:
        return (3, 6)
    if source_duration_seconds <= 1200:
        return (4, 8)
    return (5, 12)


def _build_reel_plan_prompt(video_title: str, video_description: str, transcript_text: str,
                           min_groups: int = 1, max_groups: int = 2,
                           clips_per_group: str = "5-10",
                           narration_per_group: str = "3-6",
                           reel_duration_target: str = "90-150",
                           source_duration: float = 0.0) -> str:
    """Build the full LLM prompt for reel_plan generation.

    IMPROVED: Smart clip selection with quality hierarchy, VAD-aware narration placement,
    and strict duration enforcement. The LLM must pick HIGH-IMPACT moments only.

    Deterministic by design: temperature=0.0, concrete specs.
    """
    reuse_note = ""
    if source_duration <= 120:
        reuse_note = ("\nREUSE RULE: This is a short source video. Clips SHOULD be reused across groups "
                      "with different narrative angles and commentary. What makes each group distinct is "
                      "the story framing and narration, not unique footage.")

    # Parse reel_duration_target to extract min/max
    dur_parts = reel_duration_target.split("-")
    try:
        dur_min = int(dur_parts[0])
        dur_max = int(dur_parts[1])
    except (IndexError, ValueError):
        dur_min, dur_max = 90, 150

    # Build clip count targets that force enough content to fill the duration
    recommended_clip_count = max(6, round(dur_min / 12))
    max_recommended_clips = min(20, round(dur_max / 8) + 2)

    # Compute timeline coverage bins for spread enforcement
    if source_duration > 0:
        early_end = source_duration * 0.25
        mid_start = source_duration * 0.25
        mid_end = source_duration * 0.75
        late_start = source_duration * 0.75
    else:
        early_end = mid_start = mid_end = late_start = 0

    return f"""You are an elite short-form content strategist. Your SOLE job is to find the 5-8 highest-impact, most viral-worthy moments in this video and assemble them into a {dur_min}-{dur_max} second vertical reel that MAXIMIZES viewer retention.

You are NOT a summarizer. You are a HUNTER — scanning for PEAK MOMENTS only. Every second must earn its place.

===== CORE MISSION: FIND PEAK MOMENTS =====
Scan the ENTIRE transcript for moments that make viewers STOP SCROLLING. These are:

TIER 1 — HIGHEST VALUE (always include if present):
• Action climaxes: physical feats, reveals, demonstrations, transformations
• Emotional peaks: shock, triumph, breakdown, laughter, tears, rage
• Stakes moments: "if this fails...", ultimatums, gambles, high-consequence decisions
• Viral hooks: outrageous claims, absurd situations, "did that just happen?" moments

TIER 2 — HIGH VALUE (include 2-3 per group):
• Key payoffs: answers to built-up questions, before/after reveals, results
• Surprising twists: plot turns, unexpected outcomes, contrarian takes
• Humor peaks: the biggest laugh, funniest exchange, most absurd moment
• Expert insights: specific numbers, data points, professional techniques

TIER 3 — SUPPORTING (use to fill gaps or bridge TIER 1-2 moments):
• Setup context: necessary background that makes TIER 1-2 moments land
• Transitional energy: moments that maintain momentum between peaks
• Reactions: genuine audience/participant reactions to high moments

EXCLUDE entirely: filler, greetings, repetitive explanations, low-energy passages, generic statements, transitions without substance.

===== DURATION & STRUCTURE =====
Each reel group: {dur_min}-{dur_max} seconds (HARD MINIMUM {dur_min}s).
Total estimated duration = sum(clip durations) + sum(narration durations) + 2s padding.
You MUST hit {dur_min}s minimum. If your first selection is short, ADD MORE HIGH-VALUE CLIPS.

Clips per group: {clips_per_group} (8-25 seconds each)
- SWEET SPOT: 12-18s for dialogue/exposition
- Extended: 18-25s for demonstrations, stories, transformations
- Quick cuts: 8-12s for reactions, punchlines, rapid-fire moments
- NEVER select clips under 6s unless they are absolute gold-tier punchlines

===== SOURCE VIDEO =====
Title: {video_title}
Description: {video_description[:10000]}
Duration: {source_duration:.1f} seconds

Transcript (segment index [timestamp]):
{transcript_text}

===== MANDATORY OUTPUT =====
- Output {min_groups}-{max_groups} reel_groups. Each group tells a DIFFERENT story arc.
- Groups MUST be spread across the FULL video timeline — NOT clustered in the first few minutes.
- Timeline coverage is MANDATORY:
  * Early zone: 0.0s - {early_end:.0f}s (at least 1-2 clips from here)
  * Middle zone: {mid_start:.0f}s - {mid_end:.0f}s (at least 2-3 clips from here)
  * Late zone: {late_start:.0f}s - {source_duration:.0f}s (at least 1-2 clips from here)
- Every group needs a HOOK (opening 3s), BUILD (middle tension), and PAYOFF (final 5-8s).

===== CLIP SELECTION RULES =====
1. ONLY select moments from TIER 1 or TIER 2. Use TIER 3 sparingly.
2. Each clip must have a CLEAR reason tied to the group's narrative arc.
3. No filler. No "overview" clips. Every clip must deliver a specific emotional or informational payload.
4. Prefer LONGER clips (12-25s) that let moments breathe over rapid-fire cuts.
5. The final clip of each group must be the STRONGEST moment — the payoff.
6. Do NOT include clips that merely mention the topic — include clips that DEMONSTRATE it.

===== NARRATION RULES =====
Hook (event_type: "hook"):
- reel_start: 0.0 always. reel_end: 2.5-4.0 seconds.
- 6-10 words. Specific to this video's content. Creates immediate curiosity.
- BANNED: "Watch what happens", "You won't believe", "This is insane", "Wait for it"

Commentary (event_type: "commentary"):
- 8-14 words each. Adds SPECIFIC context the viewer cannot get from footage alone.
- Use numbers, names, or concrete details. Never vague.
- Distribute across the reel: one at 25-40%, one at 50-65%, one at 70-85%.
- BANNED: "As you can see", "Notice how", "Check this out", "Pretty cool"

The audio system uses AI-powered Voice Activity Detection for ducking — it only ducks when TTS narration is actually speaking. This means narration can be placed over dialogue; the ducking will only activate during actual TTS speech, leaving surrounding dialogue intact.

===== CRITICAL: NARRATION PLACEMENT =====
- Leave at least 0.8s clear gap between narration events.
- NEVER place narration over the group's key_moment (the main payoff/climax).
- The last 5-8 seconds of the reel should be FREE of narration — let the payoff land.
- Total narration duration should be no more than 25% of total reel duration.
- Narration events must NOT overlap each other.

===== TEXT RULES =====
Allowed: letters, numbers, . , ! ? ' - — " : ;
BANNED: / \\ | * # _ < > [ ] {{ }}
Use contractions. Be conversational. Be specific.

===== SELF-VERIFICATION (MANDATORY) =====
Before outputting, you MUST verify:
1. total_clip_duration = sum of (source_end - source_start) for all clips
2. total_narration_duration = sum of (reel_end - reel_start) for all narration events
3. estimated_duration = total_clip_duration + total_narration_duration + 2.0
4. estimated_duration >= {dur_min}? If NO: add more clips until YES.
5. Clips span early/middle/late zones? If NO: replace clips to fix coverage.
6. estimated_duration_seconds = estimated_duration from step 3.

===== OUTPUT (STRICT JSON ONLY) =====
Output ONLY valid JSON. No markdown. No explanation.
Use "source_start" and "source_end" for clip timestamps.

{{
  "reel_groups": [
    {{
      "group_index": 0,
      "group_reasoning": "Why these specific moments form a compelling arc. Include duration breakdown.",
      "estimated_duration_seconds": {dur_min}.0,
      "reel_summary": {{
        "title": "Scroll-stopping title (max 60 chars)",
        "short_description": "One-sentence hook (max 150 chars)",
        "source_understanding": "What this covers",
        "narrative_angle": "Emotional framing",
        "key_moment": "The single strongest moment in this group"
      }},
      "source_clips": [
        {{"source_start": 0.0, "source_end": 15.0, "reason": "HOOK: Open with highest-impact visual/verbal moment"}},
        {{"source_start": 45.0, "source_end": 60.0, "reason": "BUILD: Escalate tension with key dialogue"}},
        {{"source_start": 120.0, "source_end": 138.0, "reason": "PAYOFF: The climactic reveal/transformation"}}
      ],
      "narration_events": [
        {{"event_type": "hook", "reel_start": 0.0, "reel_end": 3.0, "text": "Specific hook tied to this video...", "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 25.0, "reel_end": 28.0, "text": "Specific context with numbers or details.", "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 55.0, "reel_end": 58.0, "text": "Expert insight the footage alone doesn't convey.", "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 80.0, "reel_end": 83.0, "text": "Payoff line that ties the story together.", "voice_id": null}}
      ]
    }}
  ]
}}"""




def _extract_json_object(text: str) -> str:
    """Extract first JSON object from text, stripping markdown fences and outer conversational text."""
    t = text.strip()
    # Check for markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", t, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

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
    progress_cb: Optional[Callable[[str, float], None]] = None,
    reporter: Optional[Any] = None,
    interactions: Optional[List[LLMInteraction]] = None,
) -> dict:
    """Main entry: returns reel_plan dict with reel_groups array.
    If reporter and interactions are provided, collects structured LLMInteraction
    records during processing and broadcasts them live via set_stage_data_key."""
    if progress_cb:
        progress_cb("Preparing transcript for reel analysis...", 10)

    if not transcript:
        raise RuntimeError("Transcript is empty; cannot build reel plan.")

    # Pass 100% full transcript without any chunking, compression, or truncation.
    # step-3.7-flash has a 256k token context window (~1M chars), easily fitting full long-form transcripts.
    transcript_text = _format_full_transcript(transcript)

    description = (video_description or "")[:10000]

    if progress_cb:
        progress_cb("Sending transcript to LLM for reel planning...", 30)

    source_duration = transcript[-1]["end"] if transcript else 0.0
    min_groups, max_groups = _compute_group_count_target(source_duration)

    # Compute per-group specs proportional to source duration
    import math
    
    # AGGRESSIVE DURATION TARGETS: aim for 90-150s per group as the primary target
    # For very short source videos (< 150s), target 60-90% of source duration
    # For longer sources (>= 150s), target 90-150s
    if source_duration < 120:
        # Short source: target 60-90% but min 45s
        reel_dur_min = max(45, int(source_duration * 0.6))
        reel_dur_max = min(int(source_duration * 0.9), 120)
    elif source_duration < 300:
        # Medium source: target 90-150s
        reel_dur_min = max(60, min(90, int(source_duration * 0.5)))
        reel_dur_max = min(150, int(source_duration * 0.8))
    else:
        # Long source (5+ min): target 90-150s hardcore
        reel_dur_min = 90
        reel_dur_max = 150
    
    # Ensure at least 30s spread between min/max
    if reel_dur_max - reel_dur_min < 30:
        reel_dur_max = reel_dur_min + 30
    # Hard cap at config max
    reel_dur_max = min(reel_dur_max, int(MAX_OUTPUT_DURATION))
    reel_duration_target = f"{reel_dur_min}-{reel_dur_max}"
    
    # More clips: need 6-12 clips to fill 90-150s with 10-18s average clip length
    # Compute clips range based on duration target
    clips_min = max(3, math.floor(reel_dur_min / 18))  # fewest clips if all 18s
    clips_max = max(clips_min + 2, math.ceil(reel_dur_max / 8))  # most clips if all 8s
    # Clamp to reasonable range
    clips_min = min(clips_min, clips_max - 1)
    clips_min = max(3, min(clips_min, 8))
    clips_max = max(clips_min + 2, min(clips_max, 16))
    clips_per_group = f"{clips_min}-{clips_max}"
    
    # More narration: 3-6 events to fill narrative arc
    narr_min = max(3, math.ceil(reel_dur_min / 30))
    narr_max = max(narr_min + 1, math.ceil(reel_dur_max / 25))
    narr_min = min(narr_min, 4)
    narr_max = max(4, min(narr_max, 8))
    narration_per_group = f"{narr_min}-{narr_max}"

    print(f"[INFO] AGGRESSIVE TARGETS for {source_duration:.1f}s video: "
          f"{min_groups}-{max_groups} groups, clips: {clips_per_group}, narration: {narration_per_group}, duration: {reel_duration_target}s (min {reel_dur_min}s per group)")

    prompt = _build_reel_plan_prompt(
        video_title, description, transcript_text,
        min_groups=min_groups, max_groups=max_groups,
        clips_per_group=clips_per_group,
        narration_per_group=narration_per_group,
        reel_duration_target=reel_duration_target,
        source_duration=source_duration,
    )
    print(f"[DEBUG] Prompt length: {len(prompt)} chars, full transcript: {len(transcript_text)} chars")

    try:
        raw_content = _call_llm(
            [
                {"role": "system", "content": "You must respond with ONLY valid JSON. No explanations, no thinking, no text before or after the JSON object."},
                {"role": "user", "content": prompt}
            ],
            progress_cb,
            reporter=reporter,
            interactions=interactions,
            stage_name="reel_plan",
        )
    except Exception as e:
        raise RuntimeError(f"LLM failed: {e}") from e

    if progress_cb:
        progress_cb("Parsing reel plan...", 80)

    try:
        raw_json = _extract_json_object(raw_content)
        # Try to parse — if it fails, attempt repair
        try:
            reel_plan = json.loads(raw_json)
        except json.JSONDecodeError:
            repaired = _try_repair_truncated_json(raw_json)
            if repaired:
                print(f"[INFO] Repaired malformed JSON from LLM")
                reel_plan = json.loads(repaired)
            else:
                raise
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
                    "content": prompt + "\n\nCRITICAL: Output ONLY the complete JSON object. No additional text. Do not truncate."
                }
            ],
            progress_cb,
            reporter=reporter,
            interactions=interactions,
            stage_name="reel_plan_retry",
        )

        try:
            raw_json = _extract_json_object(raw_content_retry)
            try:
                reel_plan = json.loads(raw_json)
            except json.JSONDecodeError:
                repaired = _try_repair_truncated_json(raw_content_retry)
                if repaired:
                    print(f"[INFO] Repaired malformed JSON from LLM retry")
                    reel_plan = json.loads(repaired)
                else:
                    raise
            retry_truncated = raw_json[:500].encode('ascii', 'replace').decode()
            print(f"[DEBUG] Raw LLM reel_plan (retry): {retry_truncated}...")
        except Exception as e2:
            retry_preview = raw_content_retry[:2000].encode('ascii','replace').decode()
            print(f"[DEBUG] Raw content (retry attempt): {retry_preview}")
            raise RuntimeError(f"LLM failed: LLM output could not be parsed as valid JSON ({e2})") from e2

    if not isinstance(reel_plan, dict) or "reel_groups" not in reel_plan:
        raise RuntimeError(f"LLM failed: LLM output missing 'reel_groups' key: {reel_plan}")

    groups = reel_plan["reel_groups"]
    if not isinstance(groups, list) or len(groups) == 0:
        raise RuntimeError(f"LLM failed: 'reel_groups' must be a non-empty array, got {type(groups)}")

    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise RuntimeError(f"LLM failed: Group {i} must be an object")

        if "source_clips" not in group or not isinstance(group["source_clips"], list) or len(group["source_clips"]) == 0:
            raise RuntimeError(f"LLM failed: Group {i} missing valid 'source_clips'")

        if "narration_events" not in group or not isinstance(group["narration_events"], list):
            raise RuntimeError(f"LLM failed: Group {i} missing valid 'narration_events'")

        if group.get("estimated_duration_seconds", 0) > int(MAX_OUTPUT_DURATION):
            print(f"[WARN] Group {i} estimated duration {group['estimated_duration_seconds']}s exceeds {MAX_OUTPUT_DURATION}s cap")

        print(f"\n[INFO] Group {i} Narration Events:")
        usable_count = 0
        hook_seen = False
        for j, event in enumerate(group["narration_events"]):
            ev_type = str(event.get("event_type", "unknown")).strip().lower()
            if ev_type == "hook":
                if hook_seen:
                    print(f"[INFO] Group {i} has duplicate hook event {j}; converting to 'commentary'")
                    event["event_type"] = "commentary"
                    ev_type = "commentary"
                else:
                    hook_seen = True

            text = event.get("text", "")
            r_start = event.get("reel_start", 0.0)
            r_end = event.get("reel_end", 0.0)
            print(f"  {j+1}. [{ev_type.upper()}] {r_start:.1f}s - {r_end:.1f}s: \"{text[:60]}...\"")

            if ev_type not in ("hook", "commentary"):
                print(f"[WARN] Group {i} narration event {j} has unrecognized event_type "
                      f"'{ev_type}' — this will be SILENTLY DROPPED before TTS "
                      f"(only 'hook'/'commentary' are voiced).")
            else:
                usable_count += 1

            if ev_type == "hook" and r_start != 0.0:
                print(f"[WARN] Group {i} hook must start at reel_start=0.0, got {r_start}")
                event["reel_start"] = 0.0

            if r_end > group.get("estimated_duration_seconds", 130):
                print(f"[WARN] Group {i} event ends at {r_end}s, which exceeds estimated duration {group.get('estimated_duration_seconds')}s")

        if usable_count == 0:
            print(f"[WARN] Group {i} has ZERO usable narration events after filtering — "
                  f"this group's final video will have NO narration audio.")

    # Soft log if group count differs from target — don't waste an LLM call
    # retrying, since the target is content-proportional and the LLM may have
    # legitimately decided the content warrants fewer/more groups.
    if len(groups) < min_groups:
        print(f"[INFO] LLM returned {len(groups)} groups (target was {min_groups}-{max_groups}). "
              f"Accepting — content may not support more distinct angles.")
    elif len(groups) > max_groups:
        print(f"[INFO] LLM returned {len(groups)} groups (target was {min_groups}-{max_groups}). "
              f"Accepting — more content available than expected.")

    # Code-level Quality & Distinctness Filter:
    # Filter out empty/invalid groups or exact duplicates to maintain robust quality
    filtered_groups = []
    seen_clip_fingerprints = set()

    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        if not clips:
            print(f"[WARN] Pruning Group {i}: No source clips provided.")
            continue

        # Fingerprint clip ranges to prevent exact duplicate groups
        fingerprint = tuple(sorted((round(c.get("source_start", 0.0), 1), round(c.get("source_end", 0.0), 1)) for c in clips))
        if fingerprint in seen_clip_fingerprints and len(groups) > 1:
            print(f"[WARN] Pruning Group {i}: Exact duplicate clip selection of a previous group.")
            continue

        seen_clip_fingerprints.add(fingerprint)
        filtered_groups.append(group)

    if filtered_groups:
        groups = filtered_groups
        reel_plan["reel_groups"] = groups
    else:
        print("[WARN] All groups filtered out! Keeping original primary group.")
        groups = [groups[0]]
        reel_plan["reel_groups"] = groups

    # Clamp clip timestamps to actual source video bounds [0, source_duration] & enforce clip floor (>= 3.0s)
    clamped_count = 0
    for i, group in enumerate(groups):
        # Sanitize narration text
        for event in group.get("narration_events", []):
            if "text" in event and isinstance(event["text"], str):
                cleaned_text = re.sub(r'[\*\#\_\[\]\{\}\/\\<>"]', '', event["text"]).strip()
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                event["text"] = cleaned_text

        for clip in group.get("source_clips", []):
            s = clip.get("source_start", 0.0)
            e = clip.get("source_end", 0.0)
            new_s = max(0.0, min(s, source_duration))
            new_e = max(0.0, min(e, source_duration))
            # Enforce minimum clip duration floor of 3.0 seconds to prevent encoder issues
            if new_e - new_s < 3.0 and source_duration >= 3.0:
                new_e = min(source_duration, new_s + 3.0)
                if new_e - new_s < 3.0:
                    new_s = max(0.0, new_e - 3.0)

            if new_s != s or new_e != e:
                clamped_count += 1
                clip["source_start"] = new_s
                clip["source_end"] = new_e
            # Ensure start < end
            if clip["source_start"] >= clip["source_end"]:
                clip["source_end"] = min(clip["source_start"] + 3.0, source_duration)
    if clamped_count > 0:
        print(f"[INFO] Adjusted/Clamped {clamped_count} clip timestamps (bounds: [0, {source_duration:.1f}s], min_duration: 3.0s)")

    if progress_cb:
        progress_cb(f"Built reel plan with {len(groups)} group(s)", 100)

    # Log key stats
    total_clips = sum(len(g.get("source_clips", [])) for g in groups)
    total_narrations = sum(len(g.get("narration_events", [])) for g in groups)
    avg_duration = sum(g.get("estimated_duration_seconds", 0) for g in groups) / max(len(groups), 1)
    print(f"[INFO] REEL PLAN STATS: {len(groups)} groups, {total_clips} total clips, "
          f"{total_narrations} total narrations, avg duration {avg_duration:.1f}s")

    # ===== POST-PROCESSING DURATION ENFORCEMENT =====
    # Ensure each group's total clip duration + narration + padding hits the minimum target
    # If a group is too short, warn and log the shortfall (the compositor will have to pad)
    min_reel_dur = reel_dur_min  # from above computation
    for i, group in enumerate(groups):
        clips = group.get("source_clips", [])
        clips_total = sum(c.get("source_end", 0) - c.get("source_start", 0) for c in clips)
        nar_events = group.get("narration_events", [])
        nar_total = sum(e.get("reel_end", 0) - e.get("reel_start", 0) for e in nar_events)
        actual_estimated = clips_total + nar_total + 2.0

        # Re-set estimated_duration_seconds to the computed actual if LLM estimate is too far off
        llm_estimate = group.get("estimated_duration_seconds", 0)
        if llm_estimate < min_reel_dur and source_duration >= min_reel_dur:
            print(f"[WARN] Group {i}: LLM estimated_duration_seconds={llm_estimate:.1f}s is below target {min_reel_dur}s. "
                  f"Actual computed: {actual_estimated:.1f}s (clips={clips_total:.1f}s, narration={nar_total:.1f}s).")

        # Bump estimated_duration_seconds to at least the computed actual
        if actual_estimated > llm_estimate:
            print(f"[INFO] Group {i}: Raising estimated_duration_seconds from {llm_estimate:.1f}s to {actual_estimated:.1f}s (computed from {len(clips)} clips + {len(nar_events)} events + 2s pad)")
            group["estimated_duration_seconds"] = round(actual_estimated, 1)

        # Log detailed per-group report
        print(f"[INFO] Group {i} DURATION REPORT: clips={clips_total:.1f}s ({len(clips)} clips), "
              f"narration={nar_total:.1f}s ({len(nar_events)} events), "
              f"estimated={group['estimated_duration_seconds']:.1f}s, "
              f"target_range={reel_duration_target}s")

    # Broadcast final interactions state
    if reporter and interactions is not None:
        reporter.set_stage_data_key("llm_interactions", [i.model_dump() for i in interactions])

    return ReelPlan(**reel_plan)


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

    transcript_text = _format_full_transcript(transcript)

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
        raise RuntimeError(f"LLM failed: {e}") from e

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
            raise RuntimeError(f"LLM failed: Unable to parse valid JSON array from LLM response ({e2})") from e2

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