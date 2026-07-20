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


def _summarize_transcript_for_llm(transcript: list, max_total_chars: int = 15000) -> str:
    """
    Summarize a long transcript for LLM consumption, targeting 13000-16000 chars.
    Prioritizes segments with high emotional impact, questions, surprises,
    skill displays, contrasts, strong action, and high-stakes statements.
    Preserves context around key moments — does not over-merge or cut aggressively.
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

    print(f"[INFO] Transcript too long ({len(full_text)} chars), smart summarization targeting ~{max_total_chars} chars...")

    # Score segments by engagement/informativeness signals
    filler_words = {"okay", "yeah", "right", "uh", "um", "oh", "so", "well", "huh", "like", "just", "wow", "ah", "hmm"}

    # High-signal keywords that indicate engaging content
    strong_terms = {
        "but", "because", "actually", "important", "problem", "why", "how",
        "wait", "what", "really", "never", "always", "must", "need",
        "secret", "hidden", "truth", "reveal", "shock", "insane", "crazy",
        "best", "worst", "most", "ever", "first", "finally", "finally",
        "impossible", "possible", "change", "different", "compare",
        "win", "lose", "beat", "amazing", "incredible", "unbelievable",
        "watch", "look", "listen", "notice", "see this", "check",
        "breakthrough", "discover", "found", "invent", "create",
        "struggle", "fail", "success", "achieve", "master", "expert",
        "versus", "vs", "vs.", "verses", "battle", "fight", "contest",
        "lesson", "learn", "teach", "explain", "understand", "realize",
        "surprising", "curious", "fascinating", "interesting",
        "guarantee", "prove", "evidence", "research", "study", "data",
        "before", "after", "result", "outcome", "impact", "effect",
        "challenge", "difficult", "hard", "easy", "simple", "complex",
        "controversial", "debate", "argument", "opinion", "fact",
        "dangerous", "risky", "safe", "protect", "save", "avoid",
        "guaranteed", "certain", "uncertain", "maybe", "perhaps",
        "dream", "goal", "ambition", "vision", "future", "potential",
        "shocking", "stunning", "remarkable", "extraordinary", "unusual",
        "humanity", "versus", "verses", "literally", "death", "alive",
        "save", "destroy", "end", "beginning", "history", "future",
    }

    scored = []
    for i, entry in enumerate(transcript):
        duration = entry['end'] - entry['start']
        text = entry['text'].strip()
        words = text.split()
        word_count = len(words)
        lowered = text.lower()

        # Base score from word count (more words = more substance)
        score = word_count

        # Penalize very short filler segments
        if duration < 0.8:
            score -= 5
        if word_count <= 2 and lowered.strip() in filler_words:
            score -= 10

        # --- POSITIVE SIGNALS ---

        # Questions create engagement
        if "?" in text:
            score += 6

        # Exclamations / strong statements
        if text.endswith("!"):
            score += 5

        # Strong/emotional keywords
        for term in strong_terms:
            if term in lowered:
                score += 3

        # Contrast indicators (but, however, although, yet, still)
        if any(w in lowered.split() for w in ("but", "however", "although", "yet", "still", "despite", "though")):
            score += 6

        # Surprise indicators
        if any(w in lowered.split() for w in ("wow", "whoa", "oh", "ah", "ha", "surprise", "unexpected", "suddenly")):
            score += 6

        # Skill / expertise displays (numbers, data, specific terminology)
        if re.search(r'\d+', text):  # numbers
            score += 4
        if any(len(w) > 12 for w in words):  # technical/long words
            score += 3

        # Emotional impact signals
        emotion_words = {"love", "hate", "fear", "scared", "excited", "amazing", "terrible",
                         "beautiful", "ugly", "happy", "sad", "angry", "frustrated",
                         "proud", "embarrassed", "shocked", "disgust", "hope", "despair",
                         "thrilled", "devastated", "stunning", "horrifying", "delight",
                         "heartbreaking", "breathtaking", "hilarious", "intense"}
        if any(ew in lowered.split() for ew in emotion_words):
            score += 5

        # Self-corrections, hesitations, emphasis (authentic moments)
        if any(w in lowered.split() for w in ("i mean", "actually", "literally", "basically",
                                                "honestly", "truthfully", "frankly", "truly",
                                                "really", "seriously", "absolutely", "definitely")):
            score += 4

        scored.append((score, i, entry["start"], entry["end"], text, lowered))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    # Build result with context windows: when we pick a high-scoring segment,
    # also include its immediate neighbors for narrative continuity
    result_indices = set()
    selected_segments = []
    total_chars = 0
    context_window = 2  # number of adjacent segments to include on each side (increased from 1)

    for score, i, start, end, text, lowered in scored:
        if total_chars >= max_total_chars:
            break
        if i in result_indices:
            continue

        # Add the segment itself
        line = f"Seg {i} [{start:.1f}-{end:.1f}s]: {text}\n"
        if total_chars + len(line) <= max_total_chars:
            result_indices.add(i)
            selected_segments.append((i, line))
            total_chars += len(line)
        else:
            continue

        # Add neighbor segments for context (before and after)
        for offset in range(1, context_window + 1):
            # Try adding segment after
            after_idx = i + offset
            if after_idx < len(transcript) and after_idx not in result_indices and total_chars < max_total_chars:
                entry_after = transcript[after_idx]
                line_after = f"Seg {after_idx} [{entry_after['start']:.1f}-{entry_after['end']:.1f}s]: {entry_after['text']}\n"
                if total_chars + len(line_after) <= max_total_chars:
                    result_indices.add(after_idx)
                    selected_segments.append((after_idx, line_after))
                    total_chars += len(line_after)

            # Try adding segment before
            before_idx = i - offset
            if before_idx >= 0 and before_idx not in result_indices and total_chars < max_total_chars:
                entry_before = transcript[before_idx]
                line_before = f"Seg {before_idx} [{entry_before['start']:.1f}-{entry_before['end']:.1f}s]: {entry_before['text']}\n"
                if total_chars + len(line_before) <= max_total_chars:
                    result_indices.add(before_idx)
                    selected_segments.append((before_idx, line_before))
                    total_chars += len(line_before)

    # Re-sort by segment index to maintain chronological order
    selected_segments.sort(key=lambda x: x[0])
    compressed = "".join(line for _, line in selected_segments)

    print(f"[INFO] Transcript compressed from {len(full_text)} to {len(compressed)} chars "
          f"({len(selected_segments)} of {len(transcript)} segments kept)")
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
                    "max_tokens": 65536,
                    "timeout": 480.0,
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
    """Try to repair a truncated JSON by balancing braces and brackets,
    fixing trailing commas, and adding missing array/object separators."""
    if not text:
        return ""

    # 1. Fix trailing commas before closing brackets/braces first
    repaired = re.sub(r',\s*([}\]])', r'\1', text.strip())

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


def _build_reel_plan_prompt(video_title: str, video_description: str, transcript_text: str) -> str:
    """Build the full LLM prompt for reel_plan generation with enhanced persona and instructions."""
    return f"""You are a world-class short-form content strategist who creates viral YouTube Shorts and Instagram Reels. Your reels consistently achieve high retention because you understand narrative structure, emotional pacing, and scroll-stopping hooks.

TITLE: {video_title}
DESC: {video_description[:500]}
TRANSCRIPT:
{transcript_text}

CRITICAL INSTRUCTION:
Create 5 to 6 distinct, high-impact reel_groups from the video. Each group MUST use 5-8 source clips to tell a complete, self-contained story. Do NOT create fewer than 5 groups. Each reel MUST be 60-90 seconds (75+ seconds ideal).

CLIP SELECTION — CHOOSE FOR CURIOSITY GAP:
- Pick clips that create strong curiosity gaps — moments where the viewer NEEDS to know what happens next
- Examples: setup of a challenge before we see the result, a surprising statement before the explanation, a tense moment before the resolution
- Choose clips that naturally build on each other: setup clip → tension clip → payoff clip
- Avoid picking clips that are just talking heads or filler explanations without narrative tension

VIRAL HOOK PSYCHOLOGY & CURIOSITY GAP:

1. VIRAL HOOK (first 0-3 seconds of each reel):
   - Must create a strong INFORMATION GAP that makes viewers NEED to watch
   - Formula: [specific unexpected observation] + [what's at stake / why it matters]
   - Trigger: FOMO, surprise, emotional investment, or "how is this possible?"
   
   GOOD examples:
   ✓ "What happened when the world's strongest man faced a 50x stronger robot — the answer wasn't what engineers predicted."
   ✓ "A self-driving car hit 220kph on a turn the human driver was too scared to take. That's when the race got interesting."
   ✓ "The robot literally shut itself down in anger after losing — watch the exact moment it rage quit."

   BAD examples:
   ✗ "This is crazy!" (generic, no curiosity gap)
   ✗ "You won't believe this!" (tired clickbait, no specific hook)
   ✗ "Let's talk about robots." (no tension, no stakes)

2. NARRATIVE ARC - Each reel must have a clear story structure:
   - SETUP (context): Establishes who, what, where, why this matters (1-2 clips)
   - RISING TENSION: Builds stakes, reveals complications, creates doubt (2-3 clips)
   - SATISFYING PAYOFF: The resolution, reveal, or key insight. Let powerful visual/audio moments play on original sound WITHOUT narration overlay (1-2 clips)

3. COMMENTARY (narration_events):
   - Use SPARINGLY — only when it adds real insight or explains "why this matters"
   - Let powerful original audio moments speak for themselves
   - Minimum 2-3 narration events per group (hook + 1-2 commentaries)
   - Maximum 3-5 events for 60-90s reels
   - Each commentary should reveal something the viewer can't see or wouldn't notice
   
   GOOD examples:
   ✓ "Notice his hands are shaking — that's years of muscle memory fighting with a new technique he's never tried in competition."
   ✓ "Most people miss the critical detail here: the instrument cluster shows the car was still in second gear. He never shifted."

   BAD examples:
   ✗ "This is amazing!" (hype without insight — tell us WHY)
   ✗ "He's driving really fast." (describes what we can already see)
   ✗ "This shows teamwork." (generic label, not specific insight)

4. SOURCE CLIPS (source_clips array):
   - Use 5-8 source clips per group to build a complete story
   - Each clip should serve a specific narrative purpose: establish context, add tension, or deliver payoff
   - Choose substantial clips (6-15 seconds each) — enough to feel the moment but not drag
   - The total raw source duration should be ~35-55s per group (narration fills the rest)

5. OUTPUT RULES:
   - Each reel_group MUST have estimated_duration_seconds between 60 and 90 seconds
   - 5-6 groups total with 5-8 clips each
   - The complete JSON must parse correctly — do NOT truncate or omit any fields
   - Do NOT output incomplete JSON. Always produce the full reel_groups array.
   - Use precise source timestamps that align with actual transcript segments

PRIORITIZATION (in order):
   emotional impact > surprise > skill demonstration > stakes > spectacle > interesting fact

Exclude: filler content, rambling, off-topic tangents, repetitive statements.

OUTPUT ONLY VALID JSON — no explanations, no thinking, no text before or after:
{{
  "reel_groups": [
    {{
      "group_index": 0,
      "group_reasoning": "Why this is a distinct compelling story unit",
      "estimated_duration_seconds": 75.0,
      "reel_summary": {{
        "title": "Scroll-stopping title with curiosity gap",
        "short_description": "One sentence hook description for social media posting (max 120 chars)",
        "source_understanding": "What the video is about",
        "narrative_angle": "Unique framing for this reel",
        "key_moment": "The payoff moment that resolves the tension"
      }},
      "source_clips": [
        {{"source_start": 12.3, "source_end": 18.7, "reason": "Why this clip is essential — what it establishes"}},
        {{"source_start": 35.1, "source_end": 42.0, "reason": "What tension this adds"}},
        {{"source_start": 55.0, "source_end": 65.0, "reason": "How the story escalates"}},
        {{"source_start": 80.0, "source_end": 85.5, "reason": "The payoff moment"}}
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
    Fallback: generate 4-6 reel groups from transcript segments.
    Each group gets 4-7 clips forming a mini-story arc spread across the video.
    """
    if not transcript:
        return {"reel_groups": []}

    # Score segments by informativeness
    scored_segments = []
    for i, entry in enumerate(transcript):
        duration = entry["end"] - entry["start"]
        text = entry["text"].strip()
        words = text.split()
        lowered = text.lower()
        score = len(words)
        if duration >= 3.0:
            score += 2
        if "?" in text:
            score += 3
        if text.endswith("!"):
            score += 2
        # Strong engagement signals
        strong_terms = (
            "but", "because", "actually", "important", "problem", "why", "how",
            "win", "lose", "beat", "amazing", "incredible", "secret", "hidden",
            "truth", "reveal", "shock", "insane", "crazy", "best", "worst",
            "most", "ever", "first", "finally", "impossible", "possible",
            "change", "different", "wait", "what", "really", "never",
        )
        for term in strong_terms:
            if term in lowered:
                score += 2
        scored_segments.append((score, i, entry["start"], entry["end"], text, duration))

    scored_segments.sort(key=lambda x: -x[0])

    # Pick top segments, spread across the video timeline
    total_duration = transcript[-1]["end"] - transcript[0]["start"]
    groups = []
    clips_per_group = 5  # Increased from 4 to 5 for richer groups
    num_groups = min(6, max(4, len(transcript) // 12))  # Min 4, max 6 groups

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
            if group_duration + duration > 40.0:  # Increased from 30s to allow more clips per group
                continue
            group_clips.append({
                "source_start": start,
                "source_end": end,
                "reason": f"Key moment: {text[:100]}"
            })
            group_duration += duration

        # If zone had few good segments, supplement from adjacent zones
        if len(group_clips) < clips_per_group:
            nearby_range = total_duration / num_groups
            nearby_start = max(0, target_zone_start - nearby_range * 0.75)
            nearby_end = min(total_duration, target_zone_end + nearby_range * 0.75)
            nearby_segments = [
                s for s in scored_segments
                if nearby_start <= s[2] <= nearby_end and s[1] not in [c["source_start"] for c in group_clips]
            ]
            nearby_segments.sort(key=lambda x: -x[0])
            for score, i, start, end, text, duration in nearby_segments:
                if len(group_clips) >= clips_per_group:
                    break
                if group_duration + duration > 40.0:
                    continue
                if any(abs(c["source_start"] - start) < 1.0 for c in group_clips):
                    continue
                group_clips.append({
                    "source_start": start,
                    "source_end": end,
                    "reason": f"Key moment: {text[:100]}"
                })
                group_duration += duration

        # If still not enough clips, grab from anywhere
        if len(group_clips) < 3:
            for score, i, start, end, text, duration in scored_segments:
                if len(group_clips) >= clips_per_group:
                    break
                if group_duration + duration > 40.0:
                    continue
                group_clips.append({
                    "source_start": start,
                    "source_end": end,
                    "reason": f"Key moment: {text[:100]}"
                })
                group_duration += duration

        if not group_clips:
            continue

        hook_duration = 3.0
        commentary_count = min(len(group_clips), 4)
        commentary_duration = commentary_count * 3.0
        estimated = group_duration + hook_duration + commentary_duration

        group_title = f"{video_title[:50]} - Part {g+1}" if video_title else f"Part {g+1}"

        groups.append({
            "group_index": g,
            "group_reasoning": f"Fallback group {g+1}: {len(group_clips)} clips from video segment (approx {group_duration:.0f}s of source footage)",
            "estimated_duration_seconds": min(max(estimated, 60.0), 90.0),  # Floor at 60s
            "reel_summary": {
                "title": group_title,
                "short_description": f"Key moments from {video_title[:60] if video_title else 'the video'} - Part {g+1}",
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
                for j, c in enumerate(group_clips[:commentary_count])
            ]
        })

    if not groups:
        # Absolute fallback: one group with first few segments
        source_clips = []
        total_dur = 0.0
        for entry in transcript[:10]:
            dur = entry["end"] - entry["start"]
            if total_dur + dur > 40.0:
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
            "estimated_duration_seconds": min(max(total_dur + 10.0, 60.0), 90.0),
            "reel_summary": {
                "title": video_title[:80] if video_title else "Untitled",
                "short_description": f"Key moments from {video_title[:60] if video_title else 'the video'}",
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

    # Log stats
    total_clips = sum(len(g['source_clips']) for g in groups)
    avg_duration = sum(g.get('estimated_duration_seconds', 0) for g in groups) / max(len(groups), 1)
    print(f"[INFO] Fallback generated {len(groups)} group(s) with {total_clips} total clips, avg duration {avg_duration:.0f}s")
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

    # Smart transcript summarization targeting 13000-16000 chars
    transcript_text = _summarize_transcript_for_llm(transcript, max_total_chars=15000)

    description = (video_description or "")[:500]

    if progress_cb:
        progress_cb("Sending transcript to LLM for reel planning...", 30)

    prompt = _build_reel_plan_prompt(video_title, description, transcript_text)
    print(f"[DEBUG] Prompt length: {len(prompt)} chars, transcript chunk: {len(transcript_text)} chars")

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
        result = _fallback_reel_plan(transcript, video_title)
        groups = result.get("reel_groups", [])
        total_clips = sum(len(g.get("source_clips", [])) for g in groups)
        print(f"[INFO] FALLBACK RESULT: {len(groups)} groups, {total_clips} total clips")
        return ReelPlan(**result, is_fallback=True)

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
        )

        try:
            raw_json = _extract_json_object(raw_content_retry)
            try:
                reel_plan = json.loads(raw_json)
            except json.JSONDecodeError:
                repaired = _try_repair_truncated_json(raw_json)
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
            print(f"[WARN] LLM failed to produce JSON after retry: {e2}")
            print("[INFO] Using fallback reel plan")
            result = _fallback_reel_plan(transcript, video_title)
            groups = result.get("reel_groups", [])
            total_clips = sum(len(g.get("source_clips", [])) for g in groups)
            print(f"[INFO] FALLBACK RESULT: {len(groups)} groups, {total_clips} total clips")
            return ReelPlan(**result, is_fallback=True)

    if not isinstance(reel_plan, dict) or "reel_groups" not in reel_plan:
        print(f"[WARN] LLM returned object without 'reel_groups' key: {reel_plan}")
        print("[INFO] Using fallback reel plan")
        result = _fallback_reel_plan(transcript, video_title)
        groups = result.get("reel_groups", [])
        total_clips = sum(len(g.get("source_clips", [])) for g in groups)
        print(f"[INFO] FALLBACK RESULT: {len(groups)} groups, {total_clips} total clips")
        return ReelPlan(**result, is_fallback=True)

    groups = reel_plan["reel_groups"]
    if not isinstance(groups, list) or len(groups) == 0:
        print(f"[WARN] 'reel_groups' must be a non-empty array, got {type(groups)}. Using fallback.")
        result = _fallback_reel_plan(transcript, video_title)
        groups = result.get("reel_groups", [])
        total_clips = sum(len(g.get("source_clips", [])) for g in groups)
        print(f"[INFO] FALLBACK RESULT: {len(groups)} groups, {total_clips} total clips")
        return ReelPlan(**result, is_fallback=True)

    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            raise RuntimeError(f"Group {i} must be an object")

        if "source_clips" not in group or not isinstance(group["source_clips"], list) or len(group["source_clips"]) == 0:
            print(f"[WARN] Group {i} missing valid 'source_clips'. Using fallback.")
            result = _fallback_reel_plan(transcript, video_title)
            groups = result.get("reel_groups", [])
            total_clips = sum(len(g.get("source_clips", [])) for g in groups)
            print(f"[INFO] FALLBACK RESULT: {len(groups)} groups, {total_clips} total clips")
            return ReelPlan(**result, is_fallback=True)

        if "narration_events" not in group or not isinstance(group["narration_events"], list):
            print(f"[WARN] Group {i} missing 'narration_events'. Using fallback.")
            result = _fallback_reel_plan(transcript, video_title)
            groups = result.get("reel_groups", [])
            total_clips = sum(len(g.get("source_clips", [])) for g in groups)
            print(f"[INFO] FALLBACK RESULT: {len(groups)} groups, {total_clips} total clips")
            return ReelPlan(**result, is_fallback=True)

        if group.get("estimated_duration_seconds", 0) > 90:
            print(f"[WARN] Group {i} estimated duration {group['estimated_duration_seconds']}s exceeds 90s cap")

        for event in group["narration_events"]:
            if event.get("event_type") == "hook" and event.get("reel_start", 1) != 0.0:
                print(f"[WARN] Group {i} hook must start at reel_start=0.0, got {event.get('reel_start')}")

    if progress_cb:
        progress_cb(f"Built reel plan with {len(groups)} group(s)", 100)

    # Log key stats
    total_clips = sum(len(g.get("source_clips", [])) for g in groups)
    total_narrations = sum(len(g.get("narration_events", [])) for g in groups)
    avg_duration = sum(g.get("estimated_duration_seconds", 0) for g in groups) / max(len(groups), 1)
    print(f"[INFO] REEL PLAN STATS: {len(groups)} groups, {total_clips} total clips, "
          f"{total_narrations} total narrations, avg duration {avg_duration:.1f}s")

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

    transcript_text = _summarize_transcript_for_llm(transcript, max_total_chars=15000)

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