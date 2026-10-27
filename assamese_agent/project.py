import hashlib
import json
import math
import re
from pathlib import Path

# Documented Assamese stock voices from the TTS adapter. Kept in one place so
# validation, CLI parsing and synthesis cannot drift apart.
STOCK_VOICES = ('Sita', 'Amit')
SPEAKER_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(data, path):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


SEGMENT_FIELDS = ('start', 'end', 'text')
PROJECT_FIELDS = ('video', 'video_sha256', 'source_language', 'duration', 'segments')


def _is_number(value):
    # bool is a subclass of int; "start": true must not pass as a timestamp.
    return isinstance(value, (float, int)) and not isinstance(value, bool)


SPEAKER_NAME_MAX = 100
SPEAKER_ENTRY_FIELDS = ('voice', 'name')


def speaker_voice(speakers, sid):
    """Stock voice for a speaker_id, or None if unassigned/unknown.

    Accepts both map forms: {"N": "Sita"} and {"N": {"voice": "Sita", "name": "..."}}.
    """
    entry = (speakers or {}).get(sid)
    if entry is None:
        return None
    return entry if isinstance(entry, str) else entry.get('voice')


def speaker_name(speakers, sid):
    entry = (speakers or {}).get(sid)
    return entry.get('name') if isinstance(entry, dict) else None


def speaker_entry(speakers, sid):
    """Normalised {"voice": ..., "name": ...} copy of a speakers entry (or empty)."""
    entry = (speakers or {}).get(sid)
    if entry is None:
        return {'voice': None, 'name': None}
    if isinstance(entry, str):
        return {'voice': entry, 'name': None}
    return {'voice': entry.get('voice'), 'name': entry.get('name')}


def validate_speakers(speakers):
    """Validate a top-level speakers map.

    Values are either a stock voice string (legacy form) or an object
    {"voice": "Sita"|"Amit"|null, "name": str|null}. A null voice means
    "detected but not yet assigned by a human"; base validation allows it,
    validate(..., translated=True) (approve/render) does not.

    Raises ValueError on any problem. Returns None. A missing/None map is
    allowed (single-speaker back-compat).
    """
    if speakers is None:
        return
    if not isinstance(speakers, dict) or not speakers:
        raise ValueError('"speakers" must be a non-empty object mapping speaker_id to a voice entry')
    for sid, entry in speakers.items():
        if not isinstance(sid, str) or not SPEAKER_ID_PATTERN.match(sid):
            raise ValueError(f'Invalid speaker_id {sid!r}: use 1-64 chars of letters/digits/_/-')
        if isinstance(entry, str):
            voice, name = entry, None
        elif isinstance(entry, dict):
            unknown = [k for k in entry if k not in SPEAKER_ENTRY_FIELDS]
            if unknown:
                raise ValueError(f'speaker {sid!r}: unknown field(s) {", ".join(unknown)}')
            voice, name = entry.get('voice'), entry.get('name')
        else:
            raise ValueError(f'speaker {sid!r} must be a stock voice string or an object with voice/name')
        if voice is not None and voice not in STOCK_VOICES:
            raise ValueError(f'speaker {sid!r} must map to a documented stock voice '
                             f'{STOCK_VOICES} or null, got {voice!r}')
        if name is not None and (not isinstance(name, str) or not name.strip()
                                 or len(name) > SPEAKER_NAME_MAX):
            raise ValueError(f'speaker {sid!r}: name must be a non-empty string of '
                             f'<= {SPEAKER_NAME_MAX} characters')


def validate(segments, seconds, translated=False, speakers=None):
    """Raise ValueError (never KeyError/TypeError) for any malformed transcript.

    When `speakers` is provided, each segment's optional `speaker_id` must be a
    key in the map. When it's None, a segment's `speaker_id` (if present) still
    has to be a well-formed string, but routing is deferred to the caller.
    """
    validate_speakers(speakers)
    if not _is_number(seconds) or not math.isfinite(seconds):
        raise ValueError('Video duration must be a finite number')
    if not isinstance(segments, list) or not 1 <= len(segments) <= 2000:
        raise ValueError('Transcript must be a JSON list of 1-2000 segments')
    last_end = 0
    for i, s in enumerate(segments):
        if not isinstance(s, dict):
            raise ValueError(f'Segment {i} must be an object with start, end and text')
        missing = [k for k in SEGMENT_FIELDS if k not in s]
        if missing:
            raise ValueError(f'Segment {i} is missing required field(s): {", ".join(missing)}')
        start, end = s['start'], s['end']
        if not _is_number(start) or not _is_number(end):
            raise ValueError(f'Segment {i}: timestamps must be numbers')
        if not all(math.isfinite(x) for x in (start, end)):
            raise ValueError(f'Segment {i}: non-finite timestamp')
        if start < last_end or not start < end <= seconds + 0.02:
            raise ValueError(f'Segment {i}: segments must be ordered, non-overlapping, and inside video duration')
        if not isinstance(s['text'], str) or not s['text'].strip():
            raise ValueError(f'Segment {i}: empty source text')
        if translated:
            text = s.get('assamese')
            if not isinstance(text, str) or not text.strip() or len(text) > 500:
                raise ValueError(f'Segment {i}: Assamese text required, <=500 characters per segment')
        if 'speaker_id' in s:
            sid = s['speaker_id']
            if not isinstance(sid, str) or not SPEAKER_ID_PATTERN.match(sid):
                raise ValueError(f'Segment {i}: speaker_id must be 1-64 chars of letters/digits/_/-')
            if speakers is not None and sid not in speakers:
                raise ValueError(f'Segment {i}: speaker_id {sid!r} not declared in top-level "speakers" map')
            if translated and speakers is not None and speaker_voice(speakers, sid) is None:
                raise ValueError(f'Segment {i}: speaker {sid!r} has no stock voice yet. Run: '
                                 f'assamese-agent speakers set <project> {sid} --voice Sita|Amit')
        if 'speaker_confidence' in s and s['speaker_confidence'] is not None:
            conf = s['speaker_confidence']
            if not _is_number(conf) or not 0 <= conf <= 1:
                raise ValueError(f'Segment {i}: speaker_confidence must be a number in [0, 1] or null')
        if 'speaker_flags' in s:
            flags = s['speaker_flags']
            if not isinstance(flags, list) or not all(isinstance(f, str) and f for f in flags):
                raise ValueError(f'Segment {i}: speaker_flags must be a list of strings')
        last_end = end


def validate_project(data):
    """Check the top-level project.json shape before any field is indexed."""
    if not isinstance(data, dict):
        raise ValueError('Project file must be a JSON object')
    if data.get('schema_version') != 1 or data.get('rights_confirmed') is not True:
        raise ValueError('Invalid project or missing rights assertion')
    missing = [k for k in PROJECT_FIELDS if k not in data]
    if missing:
        raise ValueError(f'Project is missing required field(s): {", ".join(missing)}')
    speakers = data.get('speakers')
    validate(data['segments'], data['duration'], speakers=speakers)
    # If any segment carries a speaker_id, a speakers map is mandatory so render
    # can route it to a concrete stock voice deterministically.
    if speakers is None and any(isinstance(s, dict) and 'speaker_id' in s for s in data['segments']):
        raise ValueError('Segments use speaker_id but project is missing top-level "speakers" map')


def review_hash(project):
    # "speakers" is included so changing the speaker_id -> stock voice mapping
    # invalidates approval even if the segment list itself is unchanged.
    fields = {k: project[k] for k in ('video_sha256', 'source_language', 'segments')}
    fields['speakers'] = project.get('speakers')
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def reviewed(project):
    if project.get('review', {}).get('sha256') != review_hash(project):
        raise ValueError('Transcript/translation needs native review. Run approve after checking it')
