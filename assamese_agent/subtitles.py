"""ASS subtitle generation for burned-in Assamese captions. Pure Python.

Why ASS and not SRT for burning: libass (used by ffmpeg's `ass` filter) gives
us an explicit font, outline/box styling, per-word colour overrides and
pixel-accurate layout via PlayResX/PlayResY. With HarfBuzz it also shapes
Bengali-script conjuncts and reordered vowel signs correctly, which plain
drawtext does not.

Layout scales with the video: font size is a fraction of the shorter edge, so
a 1080x1920 reel, a 1080x1080 square and a 1920x1080 landscape video all get
the same 54 px captions, and a 4K video gets 108 px. Vertical videos get a
taller bottom margin to clear the UI that social apps overlay there.

Word timing is ESTIMATED: within each segment, words are spread across the
actual synthesized speech length in proportion to their character count.
This is not forced alignment; it is a readable approximation.
"""
import math

STYLE_NAME = 'Assamese'

# Colours are ASS &HAABBGGRR& (alpha, blue, green, red). 00 alpha = opaque.
WHITE = '&H00FFFFFF'
BLACK = '&H00000000'
BOX = '&H80000000'          # 50% black box
HIGHLIGHT = '&H0000FFFF'    # yellow (B=00 G=FF R=FF)

# Inline override colours omit alpha: &HBBGGRR&
INLINE_WHITE = '&HFFFFFF&'
INLINE_HIGHLIGHT = '&H00FFFF&'

FONT_FRACTION = 0.05        # of the shorter edge
GLYPH_WIDTH_EM = 0.55       # rough mean advance of Bengali-script glyphs
MAX_CHARS_PER_LINE = 42     # subtitle readability ceiling, regardless of width
MIN_CHARS_PER_LINE = 12
MAX_LINES = 2
MIN_WORD_SECONDS = 0.05     # below this, per-word highlighting is pointless


def layout(width, height, scale=1.0, style='outline'):
    """Pixel layout for a video size. Deterministic, no I/O."""
    if not (isinstance(width, int) and isinstance(height, int)) or width < 16 or height < 16:
        raise ValueError('Video dimensions must be integers >= 16')
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) \
            or not math.isfinite(scale) or not 0.25 <= scale <= 4:
        raise ValueError('Subtitle scale must be between 0.25 and 4')
    if style not in ('outline', 'box'):
        raise ValueError('Subtitle style must be outline or box')
    aspect = width / height
    if aspect < 0.9:
        kind, margin_v_frac = 'portrait', 0.16    # clear reels/shorts UI at the bottom
    elif aspect > 1.1:
        kind, margin_v_frac = 'landscape', 0.06
    else:
        kind, margin_v_frac = 'square', 0.08
    font_size = max(12, round(min(width, height) * FONT_FRACTION * scale))
    margin_lr = round(width * 0.06)
    margin_v = round(height * margin_v_frac)
    usable = width - 2 * margin_lr
    chars_per_line = int(usable / (font_size * GLYPH_WIDTH_EM))
    chars_per_line = max(MIN_CHARS_PER_LINE, min(MAX_CHARS_PER_LINE, chars_per_line))
    if style == 'box':
        border, outline, shadow = 3, max(2, round(font_size * 0.18)), 0
    else:
        border, outline, shadow = 1, max(1, round(font_size * 0.08)), max(0, round(font_size * 0.04))
    return {
        'kind': kind, 'width': width, 'height': height, 'font_size': font_size,
        'margin_lr': margin_lr, 'margin_v': margin_v,
        'chars_per_line': chars_per_line, 'max_chars': chars_per_line * MAX_LINES,
        'border_style': border, 'outline': outline, 'shadow': shadow, 'style': style,
    }


def escape_text(text):
    """Make arbitrary text safe inside an ASS Dialogue line."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\\', '/').replace('{', '(').replace('}', ')')
    return '\\N'.join(' '.join(line.split()) for line in text.split('\n') if line.strip())


def chunk_words(words, max_chars):
    """Greedy word-boundary split so no chunk exceeds max_chars (unless one word does)."""
    chunks, current, length = [], [], 0
    for w in words:
        add = len(w) + (1 if current else 0)
        if current and length + add > max_chars:
            chunks.append(current)
            current, length = [w], len(w)
        else:
            current.append(w)
            length += add
    if current:
        chunks.append(current)
    return chunks


def proportional_times(start, end, weights):
    """Split [start, end] into consecutive spans proportional to weights."""
    total = float(sum(weights))
    if total <= 0 or end <= start:
        return [(start, end)] * len(weights)
    spans, t = [], start
    for i, w in enumerate(weights):
        nxt = end if i == len(weights) - 1 else t + (end - start) * (w / total)
        spans.append((t, nxt))
        t = nxt
    return spans


def timestamp(seconds):
    """ASS time: H:MM:SS.cc (centiseconds)."""
    cs = max(0, round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f'{h}:{m:02}:{s:02}.{cs:02}'


def events(segments, speech_ends, lay, highlight='none'):
    """Build Dialogue rows: list of (start, end, text).

    segments: project segments with start/end/assamese.
    speech_ends: per segment, the time the synthesized speech actually stops
                 (<= segment end). Text stays visible until the segment end.
    highlight: 'none' -> one row per chunk; 'word' -> one row per word with
               the current word coloured.
    """
    if highlight not in ('none', 'word'):
        raise ValueError('highlight must be none or word')
    if len(speech_ends) != len(segments):
        raise ValueError('speech_ends must have one entry per segment')
    rows = []
    for seg, speech_end in zip(segments, speech_ends):
        start, end = seg['start'], seg['end']
        speech_end = min(max(speech_end, start), end)
        words = escape_text(seg['assamese']).replace('\\N', ' ').split()
        if not words:
            continue
        chunks = chunk_words(words, lay['max_chars'])
        chunk_spans = proportional_times(start, speech_end, [sum(len(w) for w in c) + len(c) - 1
                                                            for c in chunks])
        for ci, (chunk, (c_start, c_end)) in enumerate(zip(chunks, chunk_spans)):
            last = ci == len(chunks) - 1
            display_end = end if last else c_end
            plain = ' '.join(chunk)
            if highlight == 'none' or (c_end - c_start) < MIN_WORD_SECONDS * len(chunk):
                rows.append((c_start, display_end, plain))
                continue
            word_spans = proportional_times(c_start, c_end, [len(w) for w in chunk])
            for wi, (w_start, w_end) in enumerate(word_spans):
                text = ' '.join(f'{{\\1c{INLINE_HIGHLIGHT}}}{w}{{\\1c{INLINE_WHITE}}}' if j == wi else w
                                for j, w in enumerate(chunk))
                rows.append((w_start, w_end, text))
            if display_end > c_end:
                rows.append((c_end, display_end, plain))   # hold plain text through padding
    # Drop rows that round to zero length; keep order stable.
    return [(s, e, t) for s, e, t in rows if timestamp(e) > timestamp(s)]


def render_ass(segments, speech_ends, lay, font, highlight='none'):
    """Full ASS document as a string (UTF-8 when written)."""
    if not isinstance(font, str) or not font.strip() or ',' in font:
        raise ValueError('Font family name must be a non-empty string without commas')
    if lay['style'] == 'box':
        outline_colour, back_colour = BOX, BOX
    else:
        outline_colour, back_colour = BLACK, '&H80000000'
    head = [
        '[Script Info]',
        '; Generated by assamese-agent. Word timings are estimated, not aligned.',
        'ScriptType: v4.00+',
        f'PlayResX: {lay["width"]}',
        f'PlayResY: {lay["height"]}',
        'WrapStyle: 0',
        'ScaledBorderAndShadow: yes',
        '',
        '[V4+ Styles]',
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, '
        'BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, '
        'BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
        f'Style: {STYLE_NAME},{font},{lay["font_size"]},{WHITE},{HIGHLIGHT},{outline_colour},'
        f'{back_colour},0,0,0,0,100,100,0,0,{lay["border_style"]},{lay["outline"]},'
        f'{lay["shadow"]},2,{lay["margin_lr"]},{lay["margin_lr"]},{lay["margin_v"]},1',
        '',
        '[Events]',
        'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    ]
    body = [f'Dialogue: 0,{timestamp(s)},{timestamp(e)},{STYLE_NAME},,0,0,0,,{t}'
            for s, e, t in events(segments, speech_ends, lay, highlight)]
    return '\n'.join(head + body) + '\n'
