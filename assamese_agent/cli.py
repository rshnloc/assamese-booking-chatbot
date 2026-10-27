"""Review-first CLI. No upload, publishing, voice cloning or paid defaults."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from . import diarization, media, project, subtitles


def _parse_speaker_map(entries):
    """Turn a list of "ID=Voice" strings into a {id: voice} dict with clear errors."""
    result = {}
    for entry in entries or ():
        if not isinstance(entry, str) or '=' not in entry:
            raise ValueError(f'--speaker-map expects ID=Voice, got {entry!r}')
        sid, _, voice = entry.partition('=')
        sid, voice = sid.strip(), voice.strip()
        if sid in result and result[sid] != voice:
            raise ValueError(f'--speaker-map has conflicting entries for {sid!r}')
        result[sid] = voice
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    prep = sub.add_parser('prepare')
    prep.add_argument('video')
    prep.add_argument('--out', required=True)
    prep.add_argument('--source-language', required=True, help='ASR language e.g. en, hi; transcript import supports any language')
    prep.add_argument('--transcript', help='JSON segment list; bypasses ASR')
    prep.add_argument('--model', default='small')
    prep.add_argument('--device', default='cpu')
    prep.add_argument('--rights-confirmed', action='store_true')
    prep.add_argument('--speaker-map', action='append', default=[], metavar='ID=Voice',
                      help='Map a segment speaker_id to a documented stock voice (Sita|Amit). '
                           'Repeatable. Merged with any speakers map in --transcript.')
    tr = sub.add_parser('translate')
    tr.add_argument('project')
    tr.add_argument('--source-code', required=True, help='Explicit Sarvam code e.g. en-IN; no auto detection')
    tr.add_argument('--allow-cloud', action='store_true')
    ap = sub.add_parser('approve')
    ap.add_argument('project')
    ap.add_argument('--reviewer', required=True)
    ap.add_argument('--native-reviewed', action='store_true')
    render = sub.add_parser('render')
    render.add_argument('project')
    render.add_argument('--device', default='cpu')
    render.add_argument('--speaker', choices=list(project.STOCK_VOICES), default='Sita',
                        help='Fallback stock voice for segments without a speaker_id')
    render.add_argument('--test-tone', action='store_true', help='Offline plumbing test, NOT Assamese speech')
    render.add_argument('--background', choices=['separate', 'original', 'none'], default=None,
                        help='separate: remove source speech with Demucs and keep music/effects/room tone '
                             '(default for real renders). original: keep the full source mix, ducked, '
                             'without separation (source speech stays audible; default for --test-tone). '
                             'none: replace the audio track entirely.')
    render.add_argument('--burn-subtitles', action='store_true',
                        help='Render Assamese captions into the picture (libass) so they show in every '
                             'player and social app. Replaces the switchable subtitle track.')
    render.add_argument('--highlight', choices=['none', 'word'], default='none',
                        help='With --burn-subtitles: colour the word being spoken. Word timing is '
                             'estimated from speech length, not aligned.')
    render.add_argument('--subtitle-font', metavar='FAMILY',
                        help='Installed font family for captions. Default: auto-detect a font '
                             'covering Assamese (e.g. Noto Sans Bengali).')
    render.add_argument('--subtitle-font-file', metavar='PATH',
                        help='Use this .ttf/.otf (copied into the project) instead of an installed font')
    render.add_argument('--subtitle-style', choices=['outline', 'box'], default='outline',
                        help='outline: white text, black outline and shadow. box: text on a translucent box.')
    render.add_argument('--subtitle-scale', type=float, default=1.0, metavar='X',
                        help='Multiply the automatic font size (0.25-4). Default 1.')
    render.add_argument('--duck-db', type=float, default=12.0, metavar='DB',
                        help='How far the background is lowered while dubbed speech plays, in dB '
                             '(0 = no ducking, max %g). Default 12.' % media.DUCK_MAX_DB)
    dz = sub.add_parser('diarize', help='Detect who speaks when and label each segment with a speaker_id')
    dz.add_argument('project')
    dz.add_argument('--device', default='cpu')
    dz.add_argument('--num-speakers', type=int, help='Exact speaker count, if known')
    dz.add_argument('--min-speakers', type=int)
    dz.add_argument('--max-speakers', type=int)
    dz.add_argument('--isolate-vocals', action='store_true',
                    help='Separate vocals from music first (Demucs) and diarize the vocal stem. '
                         'Helps when background music confuses detection.')
    dz.add_argument('--force', action='store_true',
                    help='Overwrite existing speaker assignments (manual corrections are lost)')
    sp = sub.add_parser('speakers', help='Review and correct speaker assignments before approval')
    sp_sub = sp.add_subparsers(dest='action', required=True)
    sp_list = sp_sub.add_parser('list', help='Show speakers, voices and segments needing review')
    sp_list.add_argument('project')
    sp_list.add_argument('--all', action='store_true', help='List every segment, not only flagged ones')
    sp_set = sp_sub.add_parser('set', help='Name a speaker and/or assign its stock voice')
    sp_set.add_argument('project')
    sp_set.add_argument('speaker_id')
    sp_set.add_argument('--name', help='Human-readable label, any language')
    sp_set.add_argument('--voice', choices=list(project.STOCK_VOICES))
    sp_merge = sp_sub.add_parser('merge', help='Fold one detected speaker into another (over-split fix)')
    sp_merge.add_argument('project')
    sp_merge.add_argument('speaker_id')
    sp_merge.add_argument('--into', required=True, metavar='SPEAKER_ID')
    sp_rename = sp_sub.add_parser('rename', help='Change a speaker_id everywhere it is used')
    sp_rename.add_argument('project')
    sp_rename.add_argument('speaker_id')
    sp_rename.add_argument('new_id')
    sp_assign = sp_sub.add_parser('assign', help='Put specific segments under a speaker (mis-detection fix)')
    sp_assign.add_argument('project')
    sp_assign.add_argument('--segments', required=True, help='0-based, e.g. "3,7-9" (see speakers list)')
    sp_assign.add_argument('--speaker', required=True, metavar='SPEAKER_ID')
    inspired = sub.add_parser('inspired-plan')
    inspired.add_argument('--brief', required=True, help='Your own idea/learning objective, not copied dialogue')
    inspired.add_argument('--out', required=True)
    return p


def _caption_plan(a, video, out):
    """Resolve layout + font for burned captions before any expensive work."""
    width, height = media.video_dimensions(video)
    lay = subtitles.layout(width, height, a.subtitle_scale, a.subtitle_style)
    warnings, fonts_dir = [], None
    if a.subtitle_font_file and a.subtitle_font:
        raise ValueError('Give either --subtitle-font or --subtitle-font-file, not both')
    if a.subtitle_font_file:
        src = Path(a.subtitle_font_file)
        family, coverage = media.font_file_info(src)
        fonts_dir = out / 'fonts'
        fonts_dir.mkdir(exist_ok=True)
        shutil.copyfile(src, fonts_dir / src.name)
    elif a.subtitle_font:
        family = a.subtitle_font
        coverage = media.font_coverage(family)
        if coverage is None and shutil.which('fc-list'):
            raise ValueError(f'Font {family!r} is not installed (fc-list does not know it)')
    else:
        family = media.find_assamese_font()
        coverage = media.font_coverage(family)
    if coverage is None:
        warnings.append(f'Could not verify that font {family!r} covers Assamese; check the output')
    elif coverage == 'bn':
        warnings.append(f'Font {family!r} declares Bengali but not Assamese coverage; the letters '
                        'ৰ and ৱ may render as boxes. Prefer Noto Sans Bengali')
    elif coverage == '':
        warnings.append(f'Font {family!r} does not declare Bengali-script coverage; Assamese text '
                        'will likely render as boxes')
    return {'layout': lay, 'font': family, 'coverage': coverage, 'fonts_dir': fonts_dir,
            'warnings': warnings}


def _diarize(a, data):
    """Run speaker detection and label every segment. Voices stay unassigned."""
    if not a.force and any('speaker_id' in s for s in data['segments']):
        raise ValueError('Segments already carry speaker assignments. Re-run with --force to '
                         'overwrite them (manual corrections will be lost)')
    for name, value in (('--num-speakers', a.num_speakers), ('--min-speakers', a.min_speakers),
                        ('--max-speakers', a.max_speakers)):
        if value is not None and value < 1:
            raise ValueError(f'{name} must be >= 1')
    if a.min_speakers and a.max_speakers and a.min_speakers > a.max_speakers:
        raise ValueError('--min-speakers cannot exceed --max-speakers')
    out = Path(a.project).resolve().parent
    audio = out / 'source.wav'
    if not audio.is_file():
        raise ValueError('source.wav is missing next to project.json; re-run prepare')
    isolated = False
    if a.isolate_vocals:
        video = media.local_video(data['video'])
        if project.file_hash(video) != data['video_sha256']:
            raise ValueError('Input video changed. Prepare a new project')
        full = out / 'source-full.wav'
        media.extract_stereo(video, full)
        from .providers import DemucsSeparator
        audio = DemucsSeparator(a.device).separate_stems(full, out / 'separation')['vocals']
        isolated = True
    from .providers import PyannoteDiarizer
    diarizer = PyannoteDiarizer(a.device)
    turns = diarizer.diarize(audio, a.num_speakers, a.min_speakers, a.max_speakers)
    diarization.validate_turns(turns)
    project.save(turns, out / 'diarization.json')
    segments, report = diarization.assign_speakers(data['segments'], turns)
    previous = data.get('speakers') or {}
    speakers = {}
    for label in report['speakers']:
        # Keep a name/voice the reviewer already gave this label on a previous run.
        speakers[label] = project.speaker_entry(previous, label) if label in previous \
            else {'voice': None, 'name': None}
    data['segments'] = segments
    data['speakers'] = speakers
    data['diarization'] = {'model': getattr(diarizer, 'MODEL', 'unknown'),
                           'turns': 'diarization.json',
                           'isolated_vocals': isolated,
                           'speakers_detected': len(report['speakers']),
                           'flagged_segments': report['flagged']}
    data.pop('review', None)
    project.validate_project(data)
    project.save(data, a.project)
    print(f"Detected {len(report['speakers'])} speaker(s): {', '.join(report['speakers'])}")
    print(f"{len(report['flagged'])} of {len(segments)} segment(s) need review "
          f"(overlap, no speech detected, or low confidence).")
    print(f'Next: assamese-agent speakers list {a.project}')
    print('Then: assamese-agent speakers set <project> <SPEAKER_ID> --voice Sita|Amit --name "..."')
    return 0


def _speakers(a):
    """Human correction of speaker labels. Every edit invalidates prior approval."""
    data = project.load(a.project)
    project.validate_project(data)
    segments = data['segments']
    speakers = data.get('speakers')

    if a.action == 'list':
        if not speakers and not any('speaker_id' in s for s in segments):
            print('No speaker assignments. Run diarize, or set speaker_id in the transcript.')
            return 0
        stats = diarization.summary(segments, speakers)
        print(f'Speakers ({len(stats)}):')
        for sid, row in stats.items():
            entry = project.speaker_entry(speakers, sid)
            print(f"  {sid:<14} name={entry['name'] or '-':<20} voice={entry['voice'] or 'UNASSIGNED':<10} "
                  f"segments={row['segments']:<5} seconds={row['seconds']:<8} flagged={row['flagged']}")
        rows = [(i, s) for i, s in enumerate(segments) if a.all or s.get('speaker_flags')]
        title = 'Segments' if a.all else 'Segments needing review'
        print(f'{title} ({len(rows)}):')
        for i, s in rows:
            conf = s.get('speaker_confidence')
            conf_text = '-' if conf is None else f'{conf:.2f}'
            flags = ','.join(s.get('speaker_flags') or []) or '-'
            text = s['text'].replace('\n', ' ')
            text = text if len(text) <= 50 else text[:47] + '...'
            print(f"  #{i:<4} [{s['start']:.1f}-{s['end']:.1f}] {s.get('speaker_id') or '(unassigned)':<14} "
                  f"conf={conf_text:<5} flags={flags:<35} {text}")
        unassigned = [sid for sid in stats if project.speaker_voice(speakers, sid) is None]
        if unassigned:
            print(f"Voices still needed for: {', '.join(unassigned)}")
        return 0

    if a.action == 'set':
        if a.name is None and a.voice is None:
            raise ValueError('Give --name and/or --voice')
        speakers = dict(speakers or {})
        entry = project.speaker_entry(speakers, a.speaker_id)
        created = a.speaker_id not in speakers
        if a.name is not None:
            entry['name'] = a.name
        if a.voice is not None:
            entry['voice'] = a.voice
        speakers[a.speaker_id] = entry
        data['speakers'] = speakers
        verb = 'Added' if created else 'Updated'
        print(f"{verb} {a.speaker_id}: name={entry['name'] or '-'} voice={entry['voice'] or 'UNASSIGNED'}")

    elif a.action == 'merge':
        if not speakers or a.speaker_id not in speakers or a.into not in speakers:
            raise ValueError('Both speakers must exist in the project "speakers" map')
        if a.speaker_id == a.into:
            raise ValueError('Cannot merge a speaker into itself')
        moved = 0
        for s in segments:
            if s.get('speaker_id') == a.speaker_id:
                s['speaker_id'] = a.into
                moved += 1
        speakers = dict(speakers)
        del speakers[a.speaker_id]
        data['speakers'] = speakers
        print(f'Merged {a.speaker_id} into {a.into}: {moved} segment(s) moved')

    elif a.action == 'rename':
        if not speakers or a.speaker_id not in speakers:
            raise ValueError(f'Unknown speaker {a.speaker_id!r}')
        if a.new_id in speakers:
            raise ValueError(f'{a.new_id!r} already exists; use merge instead')
        if not project.SPEAKER_ID_PATTERN.match(a.new_id):
            raise ValueError('New speaker_id must be 1-64 chars of letters/digits/_/-')
        speakers = {(a.new_id if sid == a.speaker_id else sid): entry for sid, entry in speakers.items()}
        for s in segments:
            if s.get('speaker_id') == a.speaker_id:
                s['speaker_id'] = a.new_id
        data['speakers'] = speakers
        print(f'Renamed {a.speaker_id} -> {a.new_id}')

    elif a.action == 'assign':
        if not speakers or a.speaker not in speakers:
            raise ValueError(f'Unknown speaker {a.speaker!r}; run speakers set first')
        chosen = diarization.parse_segment_selection(a.segments, len(segments))
        for i in chosen:
            segments[i]['speaker_id'] = a.speaker
            # A human decided: clear machine doubt markers.
            segments[i]['speaker_confidence'] = None
            segments[i]['speaker_flags'] = []
        print(f'Assigned {len(chosen)} segment(s) to {a.speaker}')

    data.pop('review', None)
    project.validate_project(data)
    project.save(data, a.project)
    return 0


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.command == 'doctor':
            print(json.dumps({'ffmpeg': shutil.which('ffmpeg'), 'ffprobe': shutil.which('ffprobe'),
                              'python': sys.version.split()[0],
                              'note': 'Only checks system tooling, not GPU/models/API credentials'}))
            media.require_tools()
            return 0
        if a.command == 'inspired-plan':
            project.save({'status': 'human-script-required', 'brief': a.brief,
                          'constraints': ['Write a new Assamese script', 'Do not copy source dialogue or shots',
                                          'Use only owned/licensed assets', 'No voice cloning'],
                          'scenes': [], 'next': 'Author scenes with Assamese narration and original visuals. '
                                               'Video generation adapter is not implemented in v0.1.'}, a.out)
            return 0
        if a.command == 'speakers':
            return _speakers(a)
        media.require_tools()
        if a.command == 'prepare':
            if not a.rights_confirmed:
                raise ValueError('Confirm you have dubbing rights and participant consent with --rights-confirmed')
            video = media.local_video(a.video)
            out = Path(a.out).resolve()
            out.mkdir(parents=True, exist_ok=True)
            dest = out / 'project.json'
            if dest.exists():
                raise ValueError('Project exists; choose a fresh output directory')
            seconds = media.duration(video)
            if seconds > 1800:
                raise ValueError('MVP supports videos up to 30 minutes. Split this input first')
            audio = out / 'source.wav'
            media.extract(video, audio)
            speakers_from_file = None
            if a.transcript:
                imported = project.load(a.transcript)
                # Accept either a bare segment list (original form) or
                # {"segments": [...], "speakers": {...}} for multi-speaker transcripts.
                if isinstance(imported, dict):
                    if 'segments' not in imported:
                        raise ValueError('Transcript object must contain a "segments" field')
                    segments = imported['segments']
                    speakers_from_file = imported.get('speakers')
                else:
                    segments = imported
            else:
                from .providers import WhisperASR
                # ASR does not do diarization here; segments will have no speaker_id
                # and render() will fall back to the --speaker default.
                segments = WhisperASR(a.model, a.device).transcribe(audio, a.source_language)
            speakers = dict(speakers_from_file or {})
            speakers.update(_parse_speaker_map(a.speaker_map))
            for sid, voice in list(speakers.items()):
                if speakers_from_file and sid in speakers_from_file and sid in _parse_speaker_map(a.speaker_map):
                    if speakers_from_file[sid] != voice:
                        raise ValueError(f'Transcript and --speaker-map disagree on speaker {sid!r}')
            speakers = speakers or None
            project.validate(segments, seconds, speakers=speakers)
            record = dict(schema_version=1, video=str(video), video_sha256=project.file_hash(video),
                          source_language=a.source_language, duration=seconds,
                          rights_confirmed=True, segments=segments)
            if speakers:
                record['speakers'] = speakers
            # Final belt-and-braces check: the saved record must pass the same
            # top-level validation the review/render commands will apply.
            project.validate_project(record)
            project.save(record, dest)
            print(dest)
            return 0
        data = project.load(a.project)
        project.validate_project(data)
        if a.command == 'translate':
            from .providers import sarvam_translate
            # Save incremental progress. Never overwrite an existing human translation.
            for segment in data['segments']:
                if not segment.get('assamese'):
                    segment['assamese'] = sarvam_translate(segment['text'], a.source_code, a.allow_cloud)
                    data.pop('review', None)
                    project.save(data, a.project)
            return 0
        speakers_map = data.get('speakers')
        if a.command == 'diarize':
            return _diarize(a, data)
        if a.command == 'approve':
            if not a.native_reviewed or not a.reviewer.strip():
                raise ValueError('Supply reviewer name and --native-reviewed only after checking language/timing')
            project.validate(data['segments'], data['duration'], translated=True, speakers=speakers_map)
            data['review'] = {'reviewer': a.reviewer, 'sha256': project.review_hash(data)}
            project.save(data, a.project)
            return 0
        if a.command == 'render':
            project.validate(data['segments'], data['duration'], translated=True, speakers=speakers_map)
            project.reviewed(data)
            video = media.local_video(data['video'])
            if project.file_hash(video) != data['video_sha256']:
                raise ValueError('Input video changed. Prepare and review a new project')
            out = Path(a.project).resolve().parent
            test_tone = a.test_tone
            # Fail fast on a bad ducking value before any model loads or audio is written.
            media.duck_gain(a.duck_db)
            caption = _caption_plan(a, video, out) if a.burn_subtitles else None
            # Test-tone is an offline plumbing check, so it must not require a
            # separation model unless the operator asks for one explicitly.
            background_mode = a.background or ('original' if test_tone else 'separate')
            stem = 'TEST-TONE-not-assamese' if test_tone else 'assamese-dub'
            output = out / (f'{stem}-subtitled.mp4' if caption else f'{stem}.mp4')
            # Lazily instantiate one IndicParlerVoice per distinct stock voice in use.
            # A two-speaker video loads each model exactly once, not per segment.
            voices = {}

            def _voice_for(segment):
                sid = segment.get('speaker_id')
                if sid is not None:
                    stock = project.speaker_voice(speakers_map, sid)
                    if stock is None:
                        raise ValueError(f'Speaker {sid!r} has no stock voice; run speakers set')
                else:
                    stock = a.speaker
                if stock not in voices:
                    from .providers import IndicParlerVoice
                    voices[stock] = IndicParlerVoice(a.device, stock)
                return voices[stock]

            chunks, warnings, voice_log, speech_ends = [], [], [], []
            for i, s in enumerate(data['segments']):
                raw, fitted = out / f'voice-{i:04}.wav', out / f'fitted-{i:04}.wav'
                slot = s['end'] - s['start']
                if test_tone:
                    media.tone(raw, slot)
                    voice_log.append({'segment': i, 'speaker_id': s.get('speaker_id'), 'voice': 'TEST-TONE'})
                else:
                    voice = _voice_for(s)
                    voice.synthesize(s['assamese'], raw)
                    voice_log.append({'segment': i, 'speaker_id': s.get('speaker_id'),
                                      'voice': next(k for k, v in voices.items() if v is voice)})
                spoken = media.duration(raw)
                ratio = max(1.0, spoken / slot)
                if ratio > 1.35:
                    raise ValueError(f'Segment {i} needs {ratio:.2f}x speed. Shorten translation or adjust timing and re-review')
                if ratio > 1.05:
                    warnings.append(f'Segment {i}: speed {ratio:.2f}x, review intelligibility')
                media.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', raw,
                           '-af', media.atempo_chain(ratio) + f',apad,atrim=duration={slot}',
                           '-ac', 1, '-ar', 24000, '-c:a', 'pcm_s16le', fitted])
                chunks.append((s['start'], fitted))
                # Where speech really stops inside the slot; the rest is padding.
                speech_ends.append(s['start'] + min(slot, spoken / ratio))
            track = out / ('TEST-TONE.wav' if test_tone else 'assamese.wav')
            srt = out / 'assamese.srt'
            media.assemble(chunks, data['duration'], track)
            media.subtitles(data['segments'], srt)
            final = track
            if background_mode != 'none':
                # Keep music, effects and room tone under the dub. Source speech
                # is removed by separation, or left in place with --background original.
                full = out / 'source-full.wav'
                media.extract_stereo(video, full)
                if background_mode == 'separate':
                    from .providers import DemucsSeparator
                    stem = DemucsSeparator(a.device).separate(full, out / 'separation')
                    background = out / 'background.wav'
                    media.extract_stereo(stem, background)
                else:
                    background = full
                ducked = out / 'background-ducked.wav'
                media.duck(background, [(s['start'], s['end']) for s in data['segments']],
                           a.duck_db, ducked)
                final = out / ('TEST-TONE-mix.wav' if test_tone else 'assamese-mix.wav')
                media.mix(ducked, track, final)
            if caption:
                ass = out / 'assamese.ass'
                ass.write_text(subtitles.render_ass(data['segments'], speech_ends, caption['layout'],
                                                    caption['font'], a.highlight), encoding='utf-8')
                warnings.extend(caption['warnings'])
                media.mux(video, final, srt, output, burn=ass, fonts_dir=caption['fonts_dir'])
            else:
                media.mux(video, final, srt, output)
            project.save({'output': str(output), 'test_tone': test_tone, 'warnings': warnings,
                          'subtitles': ({'mode': 'burned', 'font': caption['font'],
                                         'font_coverage': caption['coverage'],
                                         'highlight': a.highlight, 'style': a.subtitle_style,
                                         'video_size': [caption['layout']['width'], caption['layout']['height']],
                                         'layout': caption['layout']['kind'],
                                         'font_size_px': caption['layout']['font_size'],
                                         'word_timing': 'estimated' if a.highlight == 'word' else None}
                                        if caption else {'mode': 'soft', 'track': 'mov_text'}),
                          'voices_used': sorted(voices.keys()),
                          'segment_voices': voice_log,
                          'background': background_mode,
                          'duck_db': a.duck_db if background_mode != 'none' else None,
                          'source_speech_audible': background_mode == 'original',
                          'publication_status': 'NOT_PUBLISHED',
                          'requires_final_native_listening_review': True}, out / 'render-report.json')
            print(output)
            return 0
    except (ValueError, KeyError, ImportError, OSError, subprocess.CalledProcessError) as exc:
        # Do not log API keys, raw payloads or subprocess stderr containing private media paths.
        print(f'Pipeline stopped: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
