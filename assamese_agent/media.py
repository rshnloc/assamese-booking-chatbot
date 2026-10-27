"""Local media helpers. No shell evaluation or network media inputs."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave
from array import array


def run(args, cwd=None):
    subprocess.run([str(x) for x in args], check=True, capture_output=True, cwd=cwd)


def require_tools():
    missing = [x for x in ('ffmpeg', 'ffprobe') if not shutil.which(x)]
    if missing:
        raise ValueError('Missing system tools: ' + ', '.join(missing))


def local_video(value):
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError('Input must be an existing local video file, not a URL')
    return path


def duration(path):
    data = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                   'format=duration', '-of', 'json', str(path)])
    result = float(json.loads(data)['format']['duration'])
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Invalid media duration')
    return result


def video_dimensions(path):
    """(width, height) as the filter graph will see them, i.e. after auto-rotation.

    Phone videos are often stored landscape with a 90-degree display matrix;
    ffmpeg rotates them on decode, so subtitles must be laid out for the
    rotated size or a vertical reel gets landscape-sized captions.
    """
    data = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                    '-show_entries', 'stream=width,height:stream_side_data=rotation',
                                    '-of', 'json', str(path)])
    streams = json.loads(data).get('streams') or []
    if not streams:
        raise ValueError('No video stream found')
    info = streams[0]
    width, height = info.get('width'), info.get('height')
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError('Could not read video dimensions')
    rotation = 0
    for side in info.get('side_data_list') or []:
        if 'rotation' in side:
            try:
                rotation = int(round(float(side['rotation'])))
            except (TypeError, ValueError):
                rotation = 0
    if rotation % 180 != 0:
        width, height = height, width
    return width, height


# Fonts known to cover Assamese (Bengali script incl. U+09F0/U+09F1), by preference.
PREFERRED_ASSAMESE_FONTS = ('Noto Sans Bengali', 'Noto Serif Bengali', 'Lohit Assamese',
                            'Lohit Bengali', 'Mukti Narrow', 'Kalpurush', 'Siyam Rupali')


def _fc_list(pattern):
    """Families matching a fontconfig pattern, or None if fc-list is unavailable."""
    if not shutil.which('fc-list'):
        return None
    out = subprocess.run(['fc-list', pattern, 'family'], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    families = set()
    for line in out.stdout.splitlines():
        for fam in line.split(','):
            if fam.strip():
                families.add(fam.strip())
    return families


def font_coverage(family):
    """'as' if fontconfig says the family covers Assamese, 'bn' if only Bengali
    (likely missing the Assamese letters ra/wa), '' if installed without either,
    None if the family is not installed or fontconfig cannot be queried."""
    installed = _fc_list(f':family={family}')
    if installed is None:
        return None
    if family not in installed:
        return None
    if family in (_fc_list(f':family={family}:lang=as') or ()):
        return 'as'
    if family in (_fc_list(f':family={family}:lang=bn') or ()):
        return 'bn'
    return ''


def find_assamese_font():
    """Best installed font for Assamese, or raise with install guidance."""
    if not shutil.which('fc-list'):
        raise ValueError('fontconfig (fc-list) not found; pass --subtitle-font or --subtitle-font-file')
    for family in PREFERRED_ASSAMESE_FONTS:
        if font_coverage(family) == 'as':
            return family
    for pattern in (':lang=as', ':lang=bn'):
        families = sorted(_fc_list(pattern) or ())
        if families:
            return families[0]
    raise ValueError('No installed font covers Assamese (Bengali script). Install Noto Sans '
                     'Bengali (Debian/Ubuntu: apt install fonts-noto-core) or pass '
                     '--subtitle-font-file path/to/NotoSansBengali-Regular.ttf')


def font_file_info(path):
    """(family, coverage) for a font file via fc-scan. coverage as in font_coverage()."""
    path = Path(path)
    if not path.is_file():
        raise ValueError('Subtitle font file does not exist')
    if not shutil.which('fc-scan'):
        raise ValueError('fontconfig (fc-scan) not found; cannot read the font family name')
    out = subprocess.run(['fc-scan', '--format', '%{family}\n%{lang}', str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise ValueError('fc-scan could not read the font file')
    family_line, _, lang_line = out.stdout.partition('\n')
    family = family_line.split(',')[0].strip()
    if not family:
        raise ValueError('Font file has no family name')
    langs = set(lang_line.strip().split('|'))
    coverage = 'as' if 'as' in langs else 'bn' if 'bn' in langs else ''
    return family, coverage


def extract(video, output):
    run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', video, '-vn', '-ac', 1,
         '-ar', 16000, output])


def atempo_chain(rate):
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Invalid tempo')
    parts = []
    while rate > 2:
        parts.append('atempo=2')
        rate /= 2
    while rate < 0.5:
        parts.append('atempo=0.5')
        rate /= 0.5
    parts.append(f'atempo={rate:.8f}')
    return ','.join(parts)


def timestamp(seconds):
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02},{ms:03}'


def subtitles(segments, output):
    body = '\n\n'.join(f'{i}\n{timestamp(s["start"])} --> {timestamp(s["end"])}\n'
                        + s['assamese'].replace('\r', ' ').replace('\n', ' ')
                        for i, s in enumerate(segments, 1))
    Path(output).write_text(body + '\n', encoding='utf-8')


def tone(output, seconds, sr=24000):
    """Synthetic TEST tone, never represented as a voice."""
    values = array('h', (int(2500 * math.sin(2 * math.pi * 440 * i / sr))
                         for i in range(round(seconds * sr))))
    with wave.open(str(output), 'wb') as f:
        f.setparams((1, 2, sr, 0, 'NONE', 'not compressed'))
        f.writeframes(values.tobytes())


def assemble(chunks, video_seconds, output):
    """Place bounded mono PCM chunks on a silent timeline, without input speech."""
    sr = 24000
    timeline = array('h', [0]) * round(video_seconds * sr)
    for start, path in chunks:
        with wave.open(str(path), 'rb') as f:
            if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, sr):
                raise ValueError('Expected 24kHz mono PCM chunks')
            data = array('h')
            data.frombytes(f.readframes(f.getnframes()))
        offset = round(start * sr)
        size = min(len(data), max(0, len(timeline) - offset))
        timeline[offset:offset + size] = data[:size]
    with wave.open(str(output), 'wb') as f:
        f.setparams((1, 2, sr, 0, 'NONE', 'not compressed'))
        f.writeframes(timeline.tobytes())


def extract_stereo(source, output):
    """Full-quality 16-bit stereo PCM at the source sample rate, for background use.

    Unlike extract(), this is not downsampled for ASR. Mono sources are upmixed
    so the background path always deals with a single channel layout.
    """
    run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', source, '-vn', '-ac', 2,
         '-c:a', 'pcm_s16le', output])


def wav_info(path):
    with wave.open(str(path), 'rb') as f:
        if f.getsampwidth() != 2:
            raise ValueError('Expected 16-bit PCM WAV')
        return f.getnchannels(), f.getframerate(), f.getnframes()


# Ducking constants. Attack is short so the background is already down when the
# first syllable lands; release is longer so the music swells back gently.
DUCK_ATTACK_SECONDS = 0.05
DUCK_RELEASE_SECONDS = 0.30
DUCK_MAX_DB = 60.0


def duck_gain(duck_db):
    """Linear gain for a ducking depth in dB. 0 dB -> 1.0 (no ducking)."""
    if not isinstance(duck_db, (int, float)) or isinstance(duck_db, bool):
        raise ValueError('Ducking depth must be a number of dB')
    if not math.isfinite(duck_db) or not 0 <= duck_db <= DUCK_MAX_DB:
        raise ValueError(f'Ducking depth must be between 0 and {DUCK_MAX_DB:g} dB')
    return 10 ** (-duck_db / 20)


def duck_runs(windows, seconds, duck_db, sr, attack=DUCK_ATTACK_SECONDS, release=DUCK_RELEASE_SECONDS):
    """Piecewise-linear gain envelope as (n_samples, gain_start, gain_end) runs.

    `windows` are (start, end) speech windows in seconds, sorted and
    non-overlapping (project.validate enforces this). The envelope is 1.0
    outside speech, `duck_gain(duck_db)` inside, with a linear attack ramp
    ending at each window start and a linear release ramp beginning at each
    window end. Gaps shorter than attack+release stay ducked rather than
    pumping up and down between quick lines. The runs always sum to exactly
    round(seconds * sr) samples.
    """
    g = duck_gain(duck_db)
    total = round(seconds * sr)
    if total <= 0:
        raise ValueError('Envelope length must be positive')
    if attack < 0 or release < 0:
        raise ValueError('Attack/release must be non-negative')
    # Merge windows whose gap is too short for a full release + attack.
    merged = []
    for start, end in windows:
        if not (0 <= start < end):
            raise ValueError('Speech windows must satisfy 0 <= start < end')
        if merged and start - merged[-1][1] < attack + release:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    runs = []
    pos = 0

    def emit(until, g0, g1):
        nonlocal pos
        until = max(pos, min(total, until))
        if until > pos:
            runs.append((until - pos, g0, g1))
            pos = until

    for start, end in merged:
        s, e = round(start * sr), round(end * sr)
        a0, r1 = s - round(attack * sr), e + round(release * sr)
        emit(a0, 1.0, 1.0)           # flat at unity until attack begins
        emit(s, 1.0, g)              # attack ramp (possibly truncated at t=0)
        emit(e, g, g)                # hold ducked through the speech window
        emit(r1, g, 1.0)             # release ramp back to unity
    emit(total, 1.0, 1.0)
    if g == 1.0:
        # No ducking requested: collapse to a single unity run.
        return [(total, 1.0, 1.0)]
    return runs


def expand_runs(runs):
    """Expand runs into 16-bit PCM envelope samples (1.0 -> 32767)."""
    out = array('h')
    for n, g0, g1 in runs:
        if g0 == g1:
            out.extend(array('h', [round(g0 * 32767)]) * n)
        else:
            step = (g1 - g0) / n
            out.extend(array('h', (round((g0 + step * i) * 32767) for i in range(n))))
    return out


def write_envelope(runs, sr, output, block=1 << 20):
    """Stream the envelope to a mono 16-bit WAV without holding it all in memory.

    Flat runs are emitted with C-speed array repetition. Ramps are generated in
    Python once per distinct (length, g0, g1) and reused: every attack ramp in
    a render is identical, as is every release ramp, so 2,000 segments cost
    two ramp computations rather than 4,000.
    """
    ramp_cache = {}
    with wave.open(str(output), 'wb') as f:
        f.setparams((1, 2, sr, 0, 'NONE', 'not compressed'))
        for n, g0, g1 in runs:
            if g0 == g1:
                done = 0
                flat = array('h', [round(g0 * 32767)])
                while done < n:
                    chunk = min(block, n - done)
                    f.writeframes((flat * chunk).tobytes())
                    done += chunk
                continue
            key = (n, g0, g1)
            if key not in ramp_cache:
                step = (g1 - g0) / n
                ramp_cache[key] = array('h', (round((g0 + step * i) * 32767)
                                              for i in range(n))).tobytes()
            f.writeframes(ramp_cache[key])


def _layout(channels):
    if channels == 1:
        return 'mono'
    if channels == 2:
        return 'stereo'
    raise ValueError('Background track must be mono or stereo')


def _to_layout(layout):
    """Filter fragment that brings a MONO input to `layout` at unity gain.

    swresample's default mono->stereo rematrix applies -3 dB, which would
    silently shrink the gain envelope and the voice. `pan` copies the single
    channel into both outputs unchanged.
    """
    return 'pan=stereo|c0=c0|c1=c0,' if layout == 'stereo' else ''


def duck(background, windows, duck_db, output, envelope=None):
    """Attenuate `background` inside speech windows by `duck_db`, sample-accurately.

    The gain envelope is generated deterministically in Python and applied by
    ffmpeg's amultiply filter, so arbitrary sample rates and channel layouts are
    handled at C speed with no extra Python dependencies.
    """
    channels, sr, frames = wav_info(background)
    envelope = Path(envelope) if envelope else Path(output).with_suffix('.envelope.wav')
    write_envelope(duck_runs(windows, frames / sr, duck_db, sr), sr, envelope)
    layout = _layout(channels)
    run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', background, '-i', envelope,
         '-filter_complex',
         f'[0:a]aformat=sample_fmts=fltp:channel_layouts={layout}[b];'
         f'[1:a]{_to_layout(layout)}aformat=sample_fmts=fltp:channel_layouts={layout}[e];'
         '[b][e]amultiply[o]',
         '-map', '[o]', '-c:a', 'pcm_s16le', output])


def mix(background, voice, output):
    """Sum the voice track onto the background at the background's rate/layout.

    amix normalize=0 keeps absolute levels; neither track is attenuated.
    duration=first bounds the result to the background length.
    """
    channels, sr, _ = wav_info(background)
    layout = _layout(channels)
    voice_channels, _, _ = wav_info(voice)
    if voice_channels != 1:
        raise ValueError('Voice track must be mono')
    run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', background, '-i', voice,
         '-filter_complex',
         f'[1:a]aresample={sr},{_to_layout(layout)}'
         f'aformat=sample_fmts=fltp:sample_rates={sr}:channel_layouts={layout}[v];'
         f'[0:a]aformat=sample_fmts=fltp:channel_layouts={layout}[b];'
         '[b][v]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[o]',
         '-map', '[o]', '-c:a', 'pcm_s16le', output])


def mux(video, audio, srt, output, burn=None, fonts_dir=None):
    """Combine video + dubbed audio into an MP4.

    Default: soft mov_text subtitle track from `srt` (switchable in players).
    burn=<ass path>: render the ASS file into the picture with libass so the
    captions show everywhere, including players and apps that ignore subtitle
    tracks. No soft track is added then, to avoid doubled captions.
    fonts_dir: extra directory libass should search (for --subtitle-font-file).

    The ASS and fonts paths are passed relative to their directory (ffmpeg cwd)
    so no filtergraph escaping of arbitrary user paths is needed.
    """
    args = ['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', video, '-i', audio]
    cwd = None
    if burn:
        burn = Path(burn).resolve()
        cwd = burn.parent
        option = f'filename={burn.name}'
        if fonts_dir:
            fonts_dir = Path(fonts_dir).resolve()
            if fonts_dir.parent != cwd:
                raise ValueError('fonts_dir must sit next to the ASS file')
            option += f':fontsdir={fonts_dir.name}'
        args += ['-map', '0:v:0', '-map', '1:a:0', '-vf', f'ass={option}']
    else:
        args += ['-i', srt, '-map', '0:v:0', '-map', '1:a:0', '-map', '2:s:0',
                 '-c:s', 'mov_text', '-metadata:s:s:0', 'language=asm']
    args += ['-c:v', 'libx264', '-preset', 'fast', '-crf', 20, '-pix_fmt', 'yuv420p',
             '-c:a', 'aac', '-metadata:s:a:0', 'language=asm',
             '-t', video_seconds_string(video), '-movflags', '+faststart', output]
    run(args, cwd=cwd)


def video_seconds_string(video):
    return str(duration(video))
