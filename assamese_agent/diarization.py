"""Match speaker-diarization turns to transcript segments. Pure logic, no models.

A diarizer (see providers.PyannoteDiarizer) returns "turns": time spans
labelled with an anonymous speaker such as SPEAKER_00. Turns may overlap each
other when people talk at once. This module decides which speaker each
transcript segment belongs to and, crucially, says how sure it is.

Nothing here hides uncertainty. Each segment gets:

  speaker_id          the dominant speaker by time overlap
  speaker_confidence  fraction of the segment covered by that speaker (0-1)
  speaker_flags       list of strings a reviewer should look at:
                        overlap              another speaker also covers a large share
                        no-speech-detected   no turn touched this segment (music, noise,
                                             or the diarizer missed it); speaker was
                                             inferred from the neighbouring segment
                        low-confidence       dominant speaker covers less than half

The CLI `speakers` command lets a human fix these before approval.
"""
from .project import SPEAKER_ID_PATTERN, _is_number

OVERLAP = 'overlap'
NO_SPEECH = 'no-speech-detected'
LOW_CONFIDENCE = 'low-confidence'
FLAGS = (OVERLAP, NO_SPEECH, LOW_CONFIDENCE)

# A second speaker covering at least this fraction of a segment is a real overlap.
DEFAULT_OVERLAP_THRESHOLD = 0.3
# The dominant speaker must cover at least this fraction to be trusted.
DEFAULT_MIN_COVERAGE = 0.5


def _overlap_seconds(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def validate_turns(turns):
    """Raise ValueError unless `turns` is a non-empty list of well-formed turns."""
    if not isinstance(turns, list) or not turns:
        raise ValueError('Diarization produced no speech turns')
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or any(k not in t for k in ('start', 'end', 'speaker')):
            raise ValueError(f'Turn {i} must be an object with start, end and speaker')
        start, end, speaker = t['start'], t['end'], t['speaker']
        if not _is_number(start) or not _is_number(end) or not 0 <= start < end:
            raise ValueError(f'Turn {i}: need numeric 0 <= start < end')
        if not isinstance(speaker, str) or not SPEAKER_ID_PATTERN.match(speaker):
            raise ValueError(f'Turn {i}: speaker label must be 1-64 chars of letters/digits/_/-')


def coverage_by_speaker(segment, turns):
    """Seconds of each speaker's turns that fall inside the segment window."""
    cov = {}
    for t in turns:
        o = _overlap_seconds(segment['start'], segment['end'], t['start'], t['end'])
        if o > 0:
            cov[t['speaker']] = cov.get(t['speaker'], 0.0) + o
    return cov


def assign_speakers(segments, turns, overlap_threshold=DEFAULT_OVERLAP_THRESHOLD,
                    min_coverage=DEFAULT_MIN_COVERAGE):
    """Return (new_segments, report). Inputs are not mutated.

    new_segments are copies with speaker_id / speaker_confidence / speaker_flags
    set on every segment. report = {'speakers': [...labels...],
    'flagged': [...segment indices...], 'counts': {label: n_segments}}.
    """
    validate_turns(turns)
    if not 0 < overlap_threshold <= 1 or not 0 < min_coverage <= 1:
        raise ValueError('Thresholds must be in (0, 1]')
    out = []
    for s in segments:
        seg = dict(s)
        duration = seg['end'] - seg['start']
        cov = coverage_by_speaker(seg, turns)
        flags = []
        if not cov:
            seg['speaker_id'] = None      # filled from neighbours below
            seg['speaker_confidence'] = 0.0
            flags.append(NO_SPEECH)
        else:
            ranked = sorted(cov.items(), key=lambda kv: (-kv[1], kv[0]))
            best, best_cov = ranked[0]
            share = min(1.0, best_cov / duration)
            seg['speaker_id'] = best
            seg['speaker_confidence'] = round(share, 3)
            if len(ranked) > 1 and ranked[1][1] / duration >= overlap_threshold:
                flags.append(OVERLAP)
            if share < min_coverage:
                flags.append(LOW_CONFIDENCE)
        seg['speaker_flags'] = flags
        out.append(seg)

    # Segments with no detected speech: the transcript says someone spoke, so
    # borrow the previous segment's speaker, or the next one's at the very start.
    last = None
    for seg in out:
        if seg['speaker_id'] is None:
            seg['speaker_id'] = last
        else:
            last = seg['speaker_id']
    nxt = None
    for seg in reversed(out):
        if seg['speaker_id'] is None:
            seg['speaker_id'] = nxt
        else:
            nxt = seg['speaker_id']
    if any(seg['speaker_id'] is None for seg in out):
        raise ValueError('No transcript segment overlaps any detected speech turn. '
                         'Check that diarization ran on the right audio')

    labels = sorted({t['speaker'] for t in turns})
    counts = {label: 0 for label in labels}
    for seg in out:
        counts[seg['speaker_id']] = counts.get(seg['speaker_id'], 0) + 1
    report = {'speakers': labels,
              'flagged': [i for i, seg in enumerate(out) if seg['speaker_flags']],
              'counts': counts}
    return out, report


def summary(segments, speakers):
    """Per-speaker stats for `speakers list`: segments, seconds, flagged count."""
    rows = {}
    for sid in (speakers or {}):
        rows[sid] = {'segments': 0, 'seconds': 0.0, 'flagged': 0}
    for seg in segments:
        sid = seg.get('speaker_id')
        if sid is None:
            continue
        row = rows.setdefault(sid, {'segments': 0, 'seconds': 0.0, 'flagged': 0})
        row['segments'] += 1
        row['seconds'] += seg['end'] - seg['start']
        if seg.get('speaker_flags'):
            row['flagged'] += 1
    for row in rows.values():
        row['seconds'] = round(row['seconds'], 1)
    return rows


def parse_segment_selection(text, count):
    """'0,3-5,9' -> sorted list of 0-based indices, validated against `count`."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Segment selection must be like "0,3-5,9" (0-based, see speakers list)')
    chosen = set()
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition('-')
        try:
            lo_i = int(lo)
            hi_i = int(hi) if sep else lo_i
        except ValueError:
            raise ValueError(f'Bad segment selection {part!r}') from None
        if lo_i < 0 or hi_i < lo_i or hi_i >= count:
            raise ValueError(f'Segment selection {part!r} is outside 0-{count - 1}')
        chosen.update(range(lo_i, hi_i + 1))
    if not chosen:
        raise ValueError('Segment selection is empty')
    return sorted(chosen)
