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
from backend.models import ReelPlan
from backend.providers.llm import call_llm_sync
from typing import Callable, Optional


def _summarize_transcript_for_llm(transcript: list, max_total_chars: int = 30000) -> str:
    """
    Summarize a long transcript for LLM consumption, targeting ~30000 chars.
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
                temperature=0.1,
                max_tokens=131072,
                timeout=480.0,
            )
            truncated = raw_content[:300] + "..." if len(raw_content) > 300 else raw_content
            print(f"[DEBUG] LLM response preview (model {model}): {truncated}")
            
            # Log full raw content to a debug file
            try:
                import time
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
    """Build the full LLM prompt for reel_plan generation.

    Ultra-detailed prompt engineered for:
    - Scroll-stopping psychological hooks in 0-3 seconds
    - Ruthless clip selection prioritizing action, emotion, and contrast
    - Structured storytelling: Setup → Rising Tension → Emotional Payoff
    - Very short, punchy narration (max 8-12 words per event)
    - Zero voice overlap: narration ONLY in verified silent gaps
    - Clean text output: no special characters that break TTS/subtitles
    """
    return f"""You are an elite short-form content strategist and behavioral psychologist who creates viral YouTube Shorts and TikToks. You have a 95% viral hit rate because you weaponize curiosity gaps, emotional escalation, and information asymmetry. Every frame you select has a purpose. Every word of narration is surgical. You never waste a single second.

===== SOURCE MATERIAL =====
TITLE: {video_title}
DESC: {video_description[:500]}

TRANSCRIPT (segment index [start-end timestamps]):
{transcript_text}

===== YOUR MISSION =====
Create distinct, high-impact reel_groups from this video. Each group is a self-contained viral short (90-180 seconds, ideal 120-150s) that tells a COMPLETE story with emotional payoff. Number of groups depends on content richness (typically 3-8).

===== SECTION 1: VIRAL HOOK (first 0-3 seconds) =====
The hook is EVERYTHING. 70% of viewers leave in the first 2 seconds. Your hook must:

REQUIRED FORMULA: [Specific unexpected claim or visual] + [Why the stakes are massive]

HOOK RULES:
- MAXIMUM 3 WORDS. Shorter = more powerful. Aim for 1-3 words.
- Must create an INFORMATION GAP that the viewer NEEDS closed.
- Must reference something SPECIFIC from the actual content — never generic.
- Must imply STAKES: what will be gained, lost, or revealed.
- The hook plays OVER the opening visual — pick the most visually intense opening frame.

BANNED HOOK PATTERNS (never use these):
- "Watch what happens..." / "You won't believe..." / "This is insane..."
- "Let me show you..." / "Check this out..." / "Wait for it..."
- Any hook that could apply to ANY video. It must be specific to THIS content.

GOOD HOOK EXAMPLES:
- "His technique broke every rule. Then this happened."
- "Three seconds separated genius from disaster."
- "The part nobody noticed changes everything."
- "Engineers said impossible. He proved them wrong."

PSYCHOLOGICAL TRIGGERS TO USE IN HOOKS:
- CURIOSITY GAP: Hint at a secret, reversal, or hidden truth
- STAKES: Imply danger, failure, or a massive reward
- CONTRAST: Set up a before/after or expectation/reality clash
- AUTHORITY CHALLENGE: Someone defies experts or conventional wisdom
- FOMO: "The part most people miss..." / "What nobody talks about..."

===== SECTION 2: RUTHLESS CLIP SELECTION =====
You are selecting 4-8 source clips per group (6-30 seconds each, ~70-120s total raw footage).

CLIP PRIORITY HIERARCHY (select in this order):
1. HIGH-ACTION MOMENTS: Physical movement, demonstrations, transformations, reveals
2. EMOTIONAL FACES: Shock, awe, concentration, frustration, triumph — close-ups preferred
3. BEFORE/AFTER CONTRAST: Clear visual difference showing change or result
4. HIGH-STAKES DIALOGUE: Moments where someone makes a bold claim, asks a pivotal question, or delivers a verdict
5. SKILL DEMONSTRATIONS: Someone performing, building, explaining with visible results
6. TURNING POINTS: The exact moment where the situation shifts

CLIPS TO REJECT (never select these):
- Static talking heads with no conflict, question, or revelation
- Repeated angles showing the same thing twice
- Filler transitions, establishing shots with no content
- Segments where the speaker is rambling, hesitating, or repeating themselves
- Any moment that does not advance the story or escalate tension

EACH CLIP MUST PASS THIS TEST:
"If I removed this clip, would the story lose tension, context, or payoff?" If NO → cut it.

===== SECTION 3: STORY STRUCTURE PER GROUP =====
Every group MUST follow this dramatic arc:

ACT 1 — SETUP (1-2 clips, ~15-25s):
- Establish WHO, WHAT, and WHY the viewer should care
- Introduce the central question, challenge, or conflict
- The hook narration plays over the first clip

ACT 2 — RISING TENSION (2-3 clips, ~30-50s):
- Complicate the situation. Introduce obstacles, doubts, or surprising information.
- Each clip must ESCALATE — never plateau. The tension curve goes UP.
- This is where you build emotional investment.

ACT 3 — PAYOFF (1-2 clips, ~20-30s):
- The emotional release: the answer, the result, the transformation, the surprise.
- This should feel EARNED — the viewer stayed for this moment.
- End on the strongest possible note. Do NOT let it fizzle.

===== SECTION 4: NARRATION / COMMENTARY — CRITICAL RULES =====
Your narration is the secret weapon. It adds context the viewer CANNOT get from the footage alone.

WORD COUNT RULE (NON-NEGOTIABLE):
- MAXIMUM 8-12 WORDS per narration event. Aim for 8.
- MAXIMUM 5 narration events per group (including the hook).
- Fewer narrations = more powerful. Quality over quantity.

CONTENT RULES:
- "SHOW, DON'T TELL": Never describe what the viewer can already see.
- Each narration must do ONE of these:
  (a) Reveal a HIDDEN DETAIL the viewer would miss
  (b) Add STAKES or CONSEQUENCES ("This is where everything changes")
  (c) Create a MICRO-CURIOSITY-GAP for the next clip ("But watch what happens next")
  (d) Provide EXPERT CONTEXT that elevates the moment ("Most people get this wrong")
- Use present tense, active voice, punchy rhythm.
- Sound like a confident insider, not a narrator reading a script.

BANNED NARRATION PATTERNS:
- "As you can see..." / "Here we can observe..." / "Notice how..."
- "This is really interesting because..." (too wordy, too weak)
- Any sentence over 12 words
- Any narration that just summarizes what the speaker already said

GOOD NARRATION EXAMPLES:
- "Right here. Watch his left hand."
- "This is the moment everything shifts."
- "Nobody expected what comes next."
- "That hesitation cost him everything."
- "The real technique is invisible."

===== SECTION 5: NARRATION TIMING — ZERO OVERLAP GUARANTEE =====
Voice overlap DESTROYS viewer experience. These rules are ABSOLUTE:

TIMING RULES (NON-NEGOTIABLE):
1. Narration ONLY during SILENT GAPS in the transcript — when nobody is speaking.
2. MINIMUM 0.4 SECONDS gap between any transcript speech ending and narration starting.
3. MINIMUM 0.4 SECONDS gap between narration ending and next transcript speech starting.
4. Check the transcript timestamps: if a segment has speech from time X to Y, your narration CANNOT start before Y + 0.4s.
5. Hook narration (event_type "hook") starts at reel_start=0.0 — this is the ONLY exception because it plays before clip dialogue begins.
6. If there is no clean silent gap available, DO NOT add narration for that moment. Silence is better than overlap.

TIMING PLACEMENT STRATEGY:
- Scan the transcript for gaps of 2+ seconds where no speech occurs.
- Place narration in the CENTER of these gaps, leaving padding on both sides.
- Between clips is often a natural gap — use these transitions.
- After a speaker makes a key point and pauses — that pause is your window.

===== SECTION 6: CLEAN TEXT RULES =====
All narration text must be clean for TTS synthesis and subtitle rendering.

BANNED CHARACTERS (never use in narration text):
- Forward slash: /
- Backslash: \\
- Pipe: |
- Asterisk: *
- Hash: #
- Underscore: _
- Angle brackets: < >
- Square brackets: [ ]
- Curly braces: {{ }}
- HTML tags or markdown formatting

ALLOWED: Letters, numbers, periods, commas, exclamation marks, question marks, apostrophes, hyphens, em-dashes, quotation marks, colons, semicolons.

Write narration as natural spoken English. Use contractions ("don't" not "do not"). Be conversational.

===== OUTPUT FORMAT (STRICT JSON) =====
Output ONLY valid JSON. No explanations, no thinking, no text before or after the JSON.
Do NOT truncate. Always produce the COMPLETE reel_groups array.

{{
  "reel_groups": [
    {{
      "group_index": 0,
      "group_reasoning": "One sentence explaining why these clips form a compelling story unit with clear arc",
      "estimated_duration_seconds": 125.0,
      "reel_summary": {{
        "title": "Scroll-stopping title with specific curiosity gap (max 60 chars)",
        "short_description": "One sentence social media hook (max 150 chars)",
        "source_understanding": "What this section of the video covers",
        "narrative_angle": "The unique emotional or intellectual framing for this reel",
        "key_moment": "The specific payoff moment that resolves the tension"
      }},
      "source_clips": [
        {{"source_start": 12.3, "source_end": 18.7, "reason": "SETUP: Establishes the central challenge and why it matters"}},
        {{"source_start": 35.1, "source_end": 42.0, "reason": "TENSION: Introduces the complication that raises stakes"}},
        {{"source_start": 55.0, "source_end": 65.0, "reason": "ESCALATION: The situation intensifies beyond expectations"}},
        {{"source_start": 80.0, "source_end": 88.5, "reason": "CLIMAX: The turning point where everything changes"}},
        {{"source_start": 95.0, "source_end": 102.0, "reason": "PAYOFF: The satisfying resolution or reveal"}}
      ],
      "narration_events": [
        {{"event_type": "hook", "reel_start": 0.0, "reel_end": 2.5, "text": "His technique broke every known rule.", "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 19.5, "reel_end": 22.0, "text": "Watch what happens to his left hand.", "voice_id": null}},
        {{"event_type": "commentary", "reel_start": 45.0, "reel_end": 47.5, "text": "This changes everything.", "voice_id": null}}
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
    num_groups = min(8, max(2, len(transcript) // 15))  # Min 2, max 8 groups

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
        commentary_count = min(len(group_clips), 8)
        commentary_duration = commentary_count * 3.0
        estimated = group_duration + hook_duration + commentary_duration

        group_title = f"{video_title[:50]} - Part {g+1}" if video_title else f"Part {g+1}"

        groups.append({
            "group_index": g,
            "group_reasoning": f"Fallback group {g+1}: {len(group_clips)} clips from video segment (approx {group_duration:.0f}s of source footage)",
            "estimated_duration_seconds": min(max(estimated, float(MIN_OUTPUT_DURATION)), float(MAX_OUTPUT_DURATION)),
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
            "estimated_duration_seconds": min(max(total_dur + 10.0, float(MIN_OUTPUT_DURATION)), float(MAX_OUTPUT_DURATION)),
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

    # Smart transcript summarization — pass full transcript up to 150k chars.
    # step-3.7-flash has a 256k token context window; 150k chars is well within that.
    # The summarization/scoring fallback only triggers for very long videos that exceed this.
    transcript_text = _summarize_transcript_for_llm(transcript, max_total_chars=150000)

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

        if group.get("estimated_duration_seconds", 0) > 130:
            print(f"[WARN] Group {i} estimated duration {group['estimated_duration_seconds']}s exceeds 130s cap")

        print(f"\n[INFO] Group {i} Narration Events:")
        for j, event in enumerate(group["narration_events"]):
            ev_type = event.get("event_type", "unknown")
            text = event.get("text", "")
            r_start = event.get("reel_start", 0.0)
            r_end = event.get("reel_end", 0.0)
            print(f"  {j+1}. [{ev_type.upper()}] {r_start:.1f}s - {r_end:.1f}s: \"{text[:60]}...\"")
            
            if ev_type == "hook" and r_start != 0.0:
                print(f"[WARN] Group {i} hook must start at reel_start=0.0, got {r_start}")
                event["reel_start"] = 0.0
                
            if r_end > group.get("estimated_duration_seconds", 130):
                print(f"[WARN] Group {i} event ends at {r_end}s, which exceeds estimated duration {group.get('estimated_duration_seconds')}s")

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

    transcript_text = _summarize_transcript_for_llm(transcript, max_total_chars=30000)

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