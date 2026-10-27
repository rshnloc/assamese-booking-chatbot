import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave
from array import array
from unittest.mock import patch
from assamese_agent import diarization, media, project, subtitles
from assamese_agent.cli import main
from assamese_agent.providers import sarvam_translate


class FakeVoice:
    """Drop-in for IndicParlerVoice used in multi-speaker integration tests.

    Records every synthesize() call and writes a tiny mono PCM WAV that
    media.assemble() will accept, so we exercise the real routing and mux
    plumbing without downloading a TTS model.
    """
    instances = []  # class-level log across test cases
    calls = []

    def __init__(self, device, speaker):
        assert speaker in project.STOCK_VOICES
        self.speaker = speaker
        FakeVoice.instances.append(speaker)

    def synthesize(self, text, output, sr=24000, seconds=0.4):
        FakeVoice.calls.append({'voice': self.speaker, 'text': text, 'output': str(output)})
        # 440Hz for Sita, 660Hz for Amit so a human reviewer could spot a swap.
        freq = 440 if self.speaker == 'Sita' else 660
        values = array('h', (int(2500 * math.sin(2 * math.pi * freq * i / sr))
                             for i in range(round(seconds * sr))))
        with wave.open(str(output), 'wb') as f:
            f.setparams((1, 2, sr, 0, 'NONE', 'not compressed'))
            f.writeframes(values.tobytes())

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.calls = []


class FakeSeparator:
    """Drop-in for DemucsSeparator: records the call and returns the input
    unchanged as the "no_vocals" stem, so the rest of the background pipeline
    (normalise, duck, mix, mux) runs for real on ffmpeg."""
    calls = []

    def __init__(self, device='cpu', model='htdemucs'):
        self.device, self.model = device, model

    def separate_stems(self, audio, workdir):
        FakeSeparator.calls.append({'audio': str(audio), 'workdir': str(workdir),
                                    'device': self.device})
        return {'vocals': Path(audio), 'no_vocals': Path(audio)}

    def separate(self, audio, workdir):
        return self.separate_stems(audio, workdir)['no_vocals']

    @classmethod
    def reset(cls):
        cls.calls = []


class FakeDiarizer:
    """Drop-in for PyannoteDiarizer returning canned turns."""
    MODEL = 'fake-diarizer'
    turns = []
    calls = []

    def __init__(self, device='cpu'):
        self.device = device

    def diarize(self, audio, num_speakers=None, min_speakers=None, max_speakers=None):
        FakeDiarizer.calls.append({'audio': str(audio), 'device': self.device,
                                   'num_speakers': num_speakers,
                                   'min_speakers': min_speakers, 'max_speakers': max_speakers})
        return [dict(t) for t in FakeDiarizer.turns]

    @classmethod
    def reset(cls, turns=None):
        cls.calls = []
        cls.turns = turns or []


def write_tone(path, seconds, freq=440, amp=8000, sr=24000, channels=1):
    frames = round(seconds * sr)
    mono = [int(amp * math.sin(2 * math.pi * freq * i / sr)) for i in range(frames)]
    data = array('h')
    for v in mono:
        data.extend([v] * channels)
    with wave.open(str(path), 'wb') as f:
        f.setparams((channels, 2, sr, 0, 'NONE', 'not compressed'))
        f.writeframes(data.tobytes())


def rms(path, t0, t1):
    """RMS over [t0, t1) seconds, averaged across channels."""
    with wave.open(str(path), 'rb') as f:
        ch, sr = f.getnchannels(), f.getframerate()
        f.setpos(round(t0 * sr))
        data = array('h')
        data.frombytes(f.readframes(round((t1 - t0) * sr)))
    if not data:
        return 0.0
    return math.sqrt(sum(v * v for v in data) / len(data))


class UnitTests(unittest.TestCase):
    def test_timestamps(self):
        self.assertEqual(media.timestamp(3661.007), '01:01:01,007')
        self.assertEqual(media.timestamp(59.9999), '00:01:00,000')

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            project.validate([dict(start=0, end=2, text='a'), dict(start=1, end=3, text='b')], 4)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            project.validate([dict(start=0, end=float('nan'), text='a')], 4)

    def test_malformed_segments_raise_value_error(self):
        # Missing keys or wrong segment types must be reported as ValueError,
        # never leak as KeyError / TypeError from indexing.
        bad = [
            [dict(end=1, text='a')],                      # missing start
            [dict(start=0, text='a')],                    # missing end
            [dict(start=0, end=1)],                       # missing text
            [dict(start=True, end=1, text='a')],          # bool is not a timestamp
            [dict(start='0', end=1, text='a')],           # string timestamp
            ['not an object'],
            [[0, 1, 'a']],
            [None],
            {'segments': [dict(start=0, end=1, text='a')]},  # dict instead of list
        ]
        for segments in bad:
            with self.subTest(segments=segments):
                with self.assertRaises(ValueError):
                    project.validate(segments, 4)
        with self.assertRaises(ValueError):
            project.validate([dict(start=0, end=1, text='a')], '4')

    def test_missing_field_message_names_segment_and_field(self):
        with self.assertRaisesRegex(ValueError, r'Segment 1 .*end'):
            project.validate([dict(start=0, end=1, text='a'), dict(start=1, text='b')], 4)

    def test_malformed_project_raises_value_error(self):
        bad = [
            [1, 2, 3],
            dict(schema_version=1, rights_confirmed=True),
            dict(schema_version=1, rights_confirmed=True, video='v', video_sha256='x',
                 source_language='en', duration='5', segments=[dict(start=0, end=1, text='a')]),
            dict(schema_version=1, rights_confirmed=True, video='v', video_sha256='x',
                 source_language='en', duration=5, segments=[dict(end=1, text='a')]),
        ]
        for data in bad:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    project.validate_project(data)

    def test_malformed_project_file_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'project.json'
            project.save(dict(schema_version=1, rights_confirmed=True, video='v', video_sha256='x',
                              source_language='en', duration=5, segments=[dict(end=1, text='a')]), p)
            self.assertEqual(main(['approve', str(p), '--reviewer', 'x', '--native-reviewed']), 1)
            project.save([1, 2, 3], p)
            self.assertEqual(main(['approve', str(p), '--reviewer', 'x', '--native-reviewed']), 1)

    def test_review_invalidated(self):
        p = dict(video_sha256='abc', source_language='en',
                 segments=[dict(start=0, end=1, text='Hello', assamese='নমস্কাৰ')])
        p['review'] = {'sha256': project.review_hash(p)}
        project.reviewed(p)
        p['segments'][0]['assamese'] = 'Changed'
        with self.assertRaises(ValueError):
            project.reviewed(p)

    def test_review_invalidated_when_speakers_map_changes(self):
        # Re-routing a speaker_id to a different stock voice must force a fresh
        # native review, even if the segment list is byte-identical.
        p = dict(video_sha256='abc', source_language='en',
                 speakers={'N': 'Sita'},
                 segments=[dict(start=0, end=1, text='Hi', assamese='নমস্কাৰ', speaker_id='N')])
        p['review'] = {'sha256': project.review_hash(p)}
        project.reviewed(p)
        p['speakers'] = {'N': 'Amit'}
        with self.assertRaises(ValueError):
            project.reviewed(p)

    def test_speakers_map_shape(self):
        # Empty map, wrong types, unknown stock voice, bad speaker_id charset.
        for bad in [{}, {'N': 'Priya'}, {'bad id!': 'Sita'}, {'N': 1}, 'Sita']:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    project.validate_speakers(bad)
        project.validate_speakers(None)
        project.validate_speakers({'N': 'Sita', 'GUEST-1': 'Amit'})

    def test_segment_speaker_id_requires_map_entry(self):
        segs = [dict(start=0, end=1, text='a', assamese='ক', speaker_id='N')]
        # Undeclared id
        with self.assertRaisesRegex(ValueError, 'not declared'):
            project.validate(segs, 4, translated=True, speakers={'OTHER': 'Sita'})
        # Declared id
        project.validate(segs, 4, translated=True, speakers={'N': 'Sita'})
        # Malformed id
        bad = [dict(start=0, end=1, text='a', assamese='ক', speaker_id='bad id!')]
        with self.assertRaisesRegex(ValueError, 'speaker_id'):
            project.validate(bad, 4, translated=True, speakers={'bad id!': 'Sita'})

    def test_segments_with_speaker_id_require_project_speakers_map(self):
        # validate_project must reject a project that uses speaker_id without
        # declaring the top-level routing map.
        data = dict(schema_version=1, rights_confirmed=True, video='v', video_sha256='x',
                    source_language='en', duration=4,
                    segments=[dict(start=0, end=1, text='a', speaker_id='N')])
        with self.assertRaisesRegex(ValueError, 'speakers'):
            project.validate_project(data)
        data['speakers'] = {'N': 'Sita'}
        project.validate_project(data)

    def test_legacy_single_speaker_project_still_valid(self):
        # Back-compat: projects with no speaker_id and no speakers map are fine.
        data = dict(schema_version=1, rights_confirmed=True, video='v', video_sha256='x',
                    source_language='en', duration=4,
                    segments=[dict(start=0, end=1, text='a')])
        project.validate_project(data)

    def test_cloud_opt_in(self):
        with patch('urllib.request.urlopen') as network:
            with self.assertRaises(ValueError):
                sarvam_translate('Hello', 'en-IN')
            network.assert_not_called()

    def test_translation_response_contract(self):
        from io import BytesIO
        with patch.dict('os.environ', {'SARVAM_API_KEY': 'test-key'}):
            with patch('urllib.request.urlopen', return_value=BytesIO(
                    json.dumps({'translated_text': 'নমস্কাৰ'}).encode())) as call:
                self.assertEqual(sarvam_translate('Hello', 'en-IN', True), 'নমস্কাৰ')
                sent = json.loads(call.call_args.args[0].data)
                self.assertEqual(sent['model'], 'sarvam-translate:v1')
                self.assertEqual(sent['target_language_code'], 'as-IN')

    def test_missing_rights_stops_prepare(self):
        self.assertEqual(main(['prepare', '/does-not-exist', '--out', '/tmp/no-write',
                               '--source-language', 'en']), 1)

    def test_planner_explicitly_unimplemented(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'plan.json'
            self.assertEqual(main(['inspired-plan', '--brief', 'Teach fractions', '--out', str(dest)]), 0)
            self.assertEqual(project.load(dest)['status'], 'human-script-required')

    def test_url_input_rejected(self):
        with self.assertRaises(ValueError):
            media.local_video('https://example.com/movie.mp4')


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class MediaIntegrationTests(unittest.TestCase):
    def test_offline_video_render(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source.mp4'
            media.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                       'color=c=navy:s=320x180:d=3', '-f', 'lavfi', '-i',
                       'sine=frequency=200:duration=3', '-c:v', 'libx264', '-pix_fmt',
                       'yuv420p', '-c:a', 'aac', '-shortest', source])
            transcript = root / 'segments.json'
            project.save([dict(start=0.2, end=1.2, text='Hello', assamese='নমস্কাৰ'),
                          dict(start=1.5, end=2.5, text='Welcome', assamese='স্বাগতম')], transcript)
            job = root / 'job'
            self.assertEqual(main(['prepare', str(source), '--out', str(job),
                                   '--source-language', 'en', '--transcript', str(transcript),
                                   '--rights-confirmed']), 0)
            p = job / 'project.json'
            self.assertEqual(main(['render', str(p), '--test-tone']), 1)
            self.assertEqual(main(['approve', str(p), '--reviewer', 'Test fixture only',
                                   '--native-reviewed']), 0)
            self.assertEqual(main(['render', str(p), '--test-tone']), 0)
            output = job / 'TEST-TONE-not-assamese.mp4'
            self.assertAlmostEqual(media.duration(output), 3, places=1)
            streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                                '-show_streams', '-of', 'json', str(output)]))['streams']
            self.assertEqual([s['codec_type'] for s in streams], ['video', 'audio', 'subtitle'])
            self.assertEqual(streams[1]['tags']['language'], 'asm')
            self.assertIn('নমস্কাৰ', (job / 'assamese.srt').read_text())


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class MultiSpeakerIntegrationTests(unittest.TestCase):
    def _build_source(self, root):
        source = root / 'source.mp4'
        media.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                   'color=c=navy:s=320x180:d=4', '-f', 'lavfi', '-i',
                   'sine=frequency=200:duration=4', '-c:v', 'libx264', '-pix_fmt',
                   'yuv420p', '-c:a', 'aac', '-shortest', source])
        return source

    def test_prepare_records_speaker_id_per_segment(self):
        # prepare with --speaker-map and an object-form transcript must persist
        # both the per-segment speaker_id and the top-level speakers map.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self._build_source(root)
            transcript = root / 'segments.json'
            project.save({
                'speakers': {'GUEST': 'Amit'},
                'segments': [
                    dict(start=0.2, end=1.2, text='Hi', assamese='নমস্কাৰ', speaker_id='NARRATOR'),
                    dict(start=1.5, end=2.5, text='Hello', assamese='স্বাগতম', speaker_id='GUEST'),
                ],
            }, transcript)
            job = root / 'job'
            self.assertEqual(main(['prepare', str(source), '--out', str(job),
                                   '--source-language', 'en', '--transcript', str(transcript),
                                   '--speaker-map', 'NARRATOR=Sita',
                                   '--rights-confirmed']), 0)
            data = project.load(job / 'project.json')
            self.assertEqual(data['speakers'], {'GUEST': 'Amit', 'NARRATOR': 'Sita'})
            self.assertEqual([s['speaker_id'] for s in data['segments']],
                             ['NARRATOR', 'GUEST'])
            # And the whole thing re-validates cleanly.
            project.validate_project(data)

    def test_prepare_rejects_undeclared_speaker_id(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self._build_source(root)
            transcript = root / 'segments.json'
            project.save([dict(start=0.2, end=1.2, text='Hi', assamese='নমস্কাৰ',
                               speaker_id='GHOST')], transcript)
            job = root / 'job'
            self.assertEqual(main(['prepare', str(source), '--out', str(job),
                                   '--source-language', 'en', '--transcript', str(transcript),
                                   '--rights-confirmed']), 1)

    def test_render_routes_each_segment_to_its_own_voice(self):
        FakeVoice.reset()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self._build_source(root)
            transcript = root / 'segments.json'
            project.save({
                'speakers': {'N': 'Sita', 'G': 'Amit'},
                'segments': [
                    dict(start=0.2, end=1.2, text='Hi', assamese='নমস্কাৰ', speaker_id='N'),
                    dict(start=1.5, end=2.5, text='Guest', assamese='স্বাগতম', speaker_id='G'),
                    dict(start=2.7, end=3.7, text='Back', assamese='ধন্যবাদ', speaker_id='N'),
                ],
            }, transcript)
            job = root / 'job'
            self.assertEqual(main(['prepare', str(source), '--out', str(job),
                                   '--source-language', 'en', '--transcript', str(transcript),
                                   '--rights-confirmed']), 0)
            p = job / 'project.json'
            # Cannot render before approval.
            with patch('assamese_agent.providers.IndicParlerVoice', FakeVoice), \
                    patch('assamese_agent.providers.DemucsSeparator', FakeSeparator):
                self.assertEqual(main(['render', str(p)]), 1)
            self.assertEqual(main(['approve', str(p), '--reviewer', 'Test fixture only',
                                   '--native-reviewed']), 0)
            with patch('assamese_agent.providers.IndicParlerVoice', FakeVoice), \
                    patch('assamese_agent.providers.DemucsSeparator', FakeSeparator):
                self.assertEqual(main(['render', str(p)]), 0)

            # One model per stock voice, no matter how many segments map to it.
            self.assertEqual(sorted(FakeVoice.instances), ['Amit', 'Sita'])
            # Each segment got synthesized by the right voice, in order.
            self.assertEqual([c['voice'] for c in FakeVoice.calls],
                             ['Sita', 'Amit', 'Sita'])
            self.assertEqual([c['text'] for c in FakeVoice.calls],
                             ['নমস্কাৰ', 'স্বাগতম', 'ধন্যবাদ'])

            report = project.load(job / 'render-report.json')
            self.assertEqual(report['voices_used'], ['Amit', 'Sita'])
            self.assertEqual([r['voice'] for r in report['segment_voices']],
                             ['Sita', 'Amit', 'Sita'])
            self.assertEqual([r['speaker_id'] for r in report['segment_voices']],
                             ['N', 'G', 'N'])
            self.assertTrue((job / 'assamese-dub.mp4').exists())

    def test_render_falls_back_to_cli_speaker_for_unlabelled_segments(self):
        # Legacy single-speaker flow: no speaker_id, no speakers map, --speaker Amit.
        FakeVoice.reset()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self._build_source(root)
            transcript = root / 'segments.json'
            project.save([dict(start=0.2, end=1.2, text='Hi', assamese='নমস্কাৰ'),
                          dict(start=1.5, end=2.5, text='Hello', assamese='স্বাগতম')], transcript)
            job = root / 'job'
            self.assertEqual(main(['prepare', str(source), '--out', str(job),
                                   '--source-language', 'en', '--transcript', str(transcript),
                                   '--rights-confirmed']), 0)
            p = job / 'project.json'
            self.assertEqual(main(['approve', str(p), '--reviewer', 'Test fixture only',
                                   '--native-reviewed']), 0)
            with patch('assamese_agent.providers.IndicParlerVoice', FakeVoice):
                self.assertEqual(main(['render', str(p), '--speaker', 'Amit',
                                       '--background', 'none']), 0)
            self.assertEqual(FakeVoice.instances, ['Amit'])
            self.assertEqual([c['voice'] for c in FakeVoice.calls], ['Amit', 'Amit'])

    def test_changing_speakers_map_forces_re_approval(self):
        # End-to-end: approve, then edit the speakers map; render must refuse.
        FakeVoice.reset()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self._build_source(root)
            transcript = root / 'segments.json'
            project.save({
                'speakers': {'N': 'Sita'},
                'segments': [dict(start=0.2, end=1.2, text='Hi',
                                   assamese='নমস্কাৰ', speaker_id='N')],
            }, transcript)
            job = root / 'job'
            self.assertEqual(main(['prepare', str(source), '--out', str(job),
                                   '--source-language', 'en', '--transcript', str(transcript),
                                   '--rights-confirmed']), 0)
            p = job / 'project.json'
            self.assertEqual(main(['approve', str(p), '--reviewer', 'Test fixture only',
                                   '--native-reviewed']), 0)
            data = project.load(p)
            data['speakers']['N'] = 'Amit'  # reroute without re-approval
            project.save(data, p)
            with patch('assamese_agent.providers.IndicParlerVoice', FakeVoice):
                self.assertEqual(main(['render', str(p), '--background', 'none']), 1)


class DuckingUnitTests(unittest.TestCase):
    """Pure-Python envelope maths. No ffmpeg needed."""

    def test_duck_gain(self):
        self.assertEqual(media.duck_gain(0), 1.0)
        self.assertAlmostEqual(media.duck_gain(6), 0.5012, places=3)
        self.assertAlmostEqual(media.duck_gain(12), 0.2512, places=3)
        self.assertAlmostEqual(media.duck_gain(20), 0.1, places=6)
        for bad in (-1, 61, float('nan'), float('inf'), True, '12', None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    media.duck_gain(bad)

    def test_runs_sum_to_exact_length(self):
        sr = 1000
        for windows, seconds in [([], 2.0), ([(0.5, 1.0)], 2.0), ([(0.0, 0.3)], 1.0),
                                 ([(0.1, 0.2), (0.25, 0.4), (1.9, 2.0)], 2.0),
                                 ([(1.5, 3.0)], 2.0)]:  # window runs past the end
            with self.subTest(windows=windows):
                runs = media.duck_runs(windows, seconds, 12, sr)
                self.assertEqual(sum(n for n, _, _ in runs), round(seconds * sr))
                self.assertEqual(len(media.expand_runs(runs)), round(seconds * sr))

    def test_envelope_shape(self):
        sr, g = 1000, media.duck_gain(12)
        env = media.expand_runs(media.duck_runs([(1.0, 2.0)], 3.0, 12, sr,
                                                attack=0.1, release=0.2))
        q = lambda t: env[round(t * sr)] / 32767
        self.assertAlmostEqual(q(0.5), 1.0, places=3)      # before attack: unity
        self.assertAlmostEqual(q(0.95), (1 + g) / 2, places=2)  # mid-attack
        self.assertAlmostEqual(q(1.0), g, places=3)        # fully ducked at onset
        self.assertAlmostEqual(q(1.5), g, places=3)        # held through speech
        self.assertAlmostEqual(q(2.0), g, places=3)        # still ducked at end
        self.assertAlmostEqual(q(2.1), (1 + g) / 2, places=2)  # mid-release
        self.assertAlmostEqual(q(2.5), 1.0, places=3)      # after release: unity
        # Monotonic within ramps; no overshoot anywhere.
        self.assertTrue(all(0 <= v <= 32767 for v in env))
        self.assertTrue(all(env[i] >= env[i + 1] for i in range(900, 1000)))
        self.assertTrue(all(env[i] <= env[i + 1] for i in range(2000, 2200)))

    def test_short_gap_stays_ducked(self):
        # Gap (0.1 s) shorter than attack+release (0.35 s) => no pumping between lines.
        sr, g = 1000, media.duck_gain(12)
        env = media.expand_runs(media.duck_runs([(1.0, 1.5), (1.6, 2.0)], 3.0, 12, sr))
        for t in (1.52, 1.55, 1.58):
            self.assertAlmostEqual(env[round(t * sr)] / 32767, g, places=3)
        # A long gap (1 s) does return to unity.
        env = media.expand_runs(media.duck_runs([(0.5, 1.0), (2.0, 2.5)], 3.0, 12, sr))
        self.assertAlmostEqual(env[round(1.5 * sr)] / 32767, 1.0, places=3)

    def test_attack_truncated_at_start(self):
        sr, g = 1000, media.duck_gain(12)
        env = media.expand_runs(media.duck_runs([(0.0, 0.5)], 1.0, 12, sr))
        self.assertAlmostEqual(env[0] / 32767, g, places=3)
        self.assertAlmostEqual(env[250] / 32767, g, places=3)

    def test_zero_db_is_single_unity_run(self):
        runs = media.duck_runs([(0.5, 1.0), (1.5, 2.0)], 3.0, 0, 1000)
        self.assertEqual(runs, [(3000, 1.0, 1.0)])

    def test_bad_windows_rejected(self):
        with self.assertRaises(ValueError):
            media.duck_runs([(1.0, 0.5)], 2.0, 12, 1000)
        with self.assertRaises(ValueError):
            media.duck_runs([(-0.1, 0.5)], 2.0, 12, 1000)
        with self.assertRaises(ValueError):
            media.duck_runs([], 0, 12, 1000)

    def test_write_envelope_matches_expand(self):
        runs = media.duck_runs([(0.2, 0.4), (0.45, 0.7)], 1.0, 9, 8000)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'env.wav'
            media.write_envelope(runs, 8000, path, block=1000)  # force multi-block ramps
            with wave.open(str(path), 'rb') as f:
                self.assertEqual((f.getnchannels(), f.getsampwidth(), f.getframerate()),
                                 (1, 2, 8000))
                data = array('h')
                data.frombytes(f.readframes(f.getnframes()))
        self.assertEqual(list(data), list(media.expand_runs(runs)))


class DiarizationUnitTests(unittest.TestCase):
    """Turn -> segment matching. Pure logic; no models or ffmpeg."""

    SEGS = [dict(start=0.0, end=2.0, text='a'), dict(start=2.0, end=4.0, text='b'),
            dict(start=4.0, end=6.0, text='c')]

    def test_clean_assignment(self):
        turns = [dict(start=0.0, end=2.0, speaker='SPEAKER_00'),
                 dict(start=2.0, end=4.0, speaker='SPEAKER_01'),
                 dict(start=4.0, end=6.0, speaker='SPEAKER_00')]
        out, report = diarization.assign_speakers(self.SEGS, turns)
        self.assertEqual([s['speaker_id'] for s in out], ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_00'])
        self.assertEqual([s['speaker_confidence'] for s in out], [1.0, 1.0, 1.0])
        self.assertEqual([s['speaker_flags'] for s in out], [[], [], []])
        self.assertEqual(report, {'speakers': ['SPEAKER_00', 'SPEAKER_01'], 'flagged': [],
                                  'counts': {'SPEAKER_00': 2, 'SPEAKER_01': 1}})
        # Inputs untouched.
        self.assertNotIn('speaker_id', self.SEGS[0])

    def test_dominant_speaker_wins_and_overlap_is_flagged(self):
        # Segment 1 (2-4 s): SPEAKER_00 covers 1.2 s, SPEAKER_01 covers 0.8 s (40%).
        turns = [dict(start=0.0, end=3.2, speaker='SPEAKER_00'),
                 dict(start=3.2, end=6.0, speaker='SPEAKER_01')]
        out, report = diarization.assign_speakers(self.SEGS, turns)
        self.assertEqual(out[1]['speaker_id'], 'SPEAKER_00')
        self.assertAlmostEqual(out[1]['speaker_confidence'], 0.6, places=3)
        self.assertEqual(out[1]['speaker_flags'], [diarization.OVERLAP])
        self.assertEqual(report['flagged'], [1])

    def test_true_overlap_two_people_talking_at_once(self):
        # Both turns span the whole segment (diarizer emitted overlapping turns).
        segs = [dict(start=0.0, end=2.0, text='a')]
        turns = [dict(start=0.0, end=2.0, speaker='SPEAKER_00'),
                 dict(start=0.5, end=2.0, speaker='SPEAKER_01')]
        out, _ = diarization.assign_speakers(segs, turns)
        self.assertEqual(out[0]['speaker_id'], 'SPEAKER_00')
        self.assertIn(diarization.OVERLAP, out[0]['speaker_flags'])
        self.assertNotIn(diarization.LOW_CONFIDENCE, out[0]['speaker_flags'])

    def test_tie_breaks_deterministically(self):
        segs = [dict(start=0.0, end=2.0, text='a')]
        turns = [dict(start=0.0, end=1.0, speaker='SPEAKER_01'),
                 dict(start=1.0, end=2.0, speaker='SPEAKER_00')]
        out, _ = diarization.assign_speakers(segs, turns)
        self.assertEqual(out[0]['speaker_id'], 'SPEAKER_00')  # alphabetical on tie
        self.assertEqual(out[0]['speaker_flags'], [diarization.OVERLAP])

    def test_low_confidence_when_mostly_uncovered(self):
        # Only 0.4 s of the 2 s segment has detected speech (music masked the rest).
        segs = [dict(start=0.0, end=2.0, text='a')]
        turns = [dict(start=1.6, end=2.0, speaker='SPEAKER_00')]
        out, _ = diarization.assign_speakers(segs, turns)
        self.assertEqual(out[0]['speaker_id'], 'SPEAKER_00')
        self.assertAlmostEqual(out[0]['speaker_confidence'], 0.2, places=3)
        self.assertEqual(out[0]['speaker_flags'], [diarization.LOW_CONFIDENCE])

    def test_no_speech_detected_borrows_previous_speaker(self):
        turns = [dict(start=0.0, end=2.0, speaker='SPEAKER_00'),
                 dict(start=4.0, end=6.0, speaker='SPEAKER_01')]
        out, report = diarization.assign_speakers(self.SEGS, turns)
        self.assertEqual(out[1]['speaker_id'], 'SPEAKER_00')   # from segment 0
        self.assertEqual(out[1]['speaker_confidence'], 0.0)
        self.assertEqual(out[1]['speaker_flags'], [diarization.NO_SPEECH])
        self.assertEqual(report['flagged'], [1])

    def test_no_speech_at_start_borrows_next_speaker(self):
        turns = [dict(start=2.0, end=6.0, speaker='SPEAKER_01')]
        out, _ = diarization.assign_speakers(self.SEGS, turns)
        self.assertEqual(out[0]['speaker_id'], 'SPEAKER_01')
        self.assertEqual(out[0]['speaker_flags'], [diarization.NO_SPEECH])

    def test_no_overlap_at_all_is_an_error(self):
        turns = [dict(start=10.0, end=12.0, speaker='SPEAKER_00')]
        with self.assertRaisesRegex(ValueError, 'No transcript segment overlaps'):
            diarization.assign_speakers(self.SEGS, turns)

    def test_bad_turns_rejected(self):
        for bad in ([], None, 'x', [dict(start=0, end=1)],
                    [dict(start=1, end=0, speaker='S')],
                    [dict(start=0, end=1, speaker='bad label!')],
                    [dict(start='0', end=1, speaker='S')]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    diarization.validate_turns(bad)

    def test_thresholds_are_tunable(self):
        turns = [dict(start=0.0, end=3.2, speaker='SPEAKER_00'),
                 dict(start=3.2, end=6.0, speaker='SPEAKER_01')]
        out, _ = diarization.assign_speakers(self.SEGS, turns, overlap_threshold=0.5)
        self.assertEqual(out[1]['speaker_flags'], [])   # 40% second speaker no longer flagged
        with self.assertRaises(ValueError):
            diarization.assign_speakers(self.SEGS, turns, overlap_threshold=0)

    def test_summary(self):
        segs = [dict(start=0, end=2, text='a', speaker_id='A', speaker_flags=[]),
                dict(start=2, end=5, text='b', speaker_id='A', speaker_flags=['overlap']),
                dict(start=5, end=6, text='c', speaker_id='B', speaker_flags=[]),
                dict(start=6, end=7, text='d')]  # unassigned, ignored
        rows = diarization.summary(segs, {'A': 'Sita', 'B': {'voice': None, 'name': None}, 'C': 'Amit'})
        self.assertEqual(rows['A'], {'segments': 2, 'seconds': 5.0, 'flagged': 1})
        self.assertEqual(rows['B'], {'segments': 1, 'seconds': 1.0, 'flagged': 0})
        self.assertEqual(rows['C'], {'segments': 0, 'seconds': 0.0, 'flagged': 0})

    def test_parse_segment_selection(self):
        self.assertEqual(diarization.parse_segment_selection('0,3-5,9', 10), [0, 3, 4, 5, 9])
        self.assertEqual(diarization.parse_segment_selection(' 2 ', 3), [2])
        for bad, n in (('', 3), ('a', 3), ('5', 3), ('3-1', 5), ('-1', 3), (None, 3)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    diarization.parse_segment_selection(bad, n)


class SpeakerSchemaUnitTests(unittest.TestCase):
    def test_object_form_entries(self):
        project.validate_speakers({'A': 'Sita', 'B': {'voice': 'Amit', 'name': 'শিক্ষক'},
                                   'C': {'voice': None, 'name': None}})
        for bad in ({'A': {'voice': 'Priya'}}, {'A': {'voice': 'Sita', 'extra': 1}},
                    {'A': {'name': ''}}, {'A': {'name': 'x' * 101}}, {'A': 7}, {'A': None}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    project.validate_speakers(bad)

    def test_helpers(self):
        sp = {'A': 'Sita', 'B': {'voice': 'Amit', 'name': 'Guest'}, 'C': {'voice': None, 'name': 'X'}}
        self.assertEqual(project.speaker_voice(sp, 'A'), 'Sita')
        self.assertEqual(project.speaker_voice(sp, 'B'), 'Amit')
        self.assertIsNone(project.speaker_voice(sp, 'C'))
        self.assertIsNone(project.speaker_voice(sp, 'Z'))
        self.assertIsNone(project.speaker_voice(None, 'A'))
        self.assertEqual(project.speaker_name(sp, 'B'), 'Guest')
        self.assertIsNone(project.speaker_name(sp, 'A'))
        self.assertEqual(project.speaker_entry(sp, 'A'), {'voice': 'Sita', 'name': None})
        self.assertEqual(project.speaker_entry(sp, 'Z'), {'voice': None, 'name': None})

    def test_unassigned_voice_blocks_only_approval_stage(self):
        segs = [dict(start=0, end=1, text='a', assamese='ক', speaker_id='A',
                     speaker_confidence=0.9, speaker_flags=['overlap'])]
        sp = {'A': {'voice': None, 'name': 'Teacher'}}
        project.validate(segs, 2, speakers=sp)                      # base: ok
        with self.assertRaisesRegex(ValueError, 'no stock voice'):
            project.validate(segs, 2, translated=True, speakers=sp)  # approve/render: blocked
        sp['A']['voice'] = 'Sita'
        project.validate(segs, 2, translated=True, speakers=sp)

    def test_confidence_and_flags_shape(self):
        base = dict(start=0, end=1, text='a', speaker_id='A')
        sp = {'A': 'Sita'}
        project.validate([dict(base, speaker_confidence=None, speaker_flags=[])], 2, speakers=sp)
        for bad in (dict(base, speaker_confidence=1.5), dict(base, speaker_confidence='1'),
                    dict(base, speaker_flags='overlap'), dict(base, speaker_flags=[1]),
                    dict(base, speaker_flags=[''])):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    project.validate([bad], 2, speakers=sp)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class SpeakerWorkflowIntegrationTests(unittest.TestCase):
    """diarize -> list -> correct -> approve -> render, with a fake diarizer."""

    SEGMENTS = [dict(start=0.2, end=1.0, text='Welcome', assamese='স্বাগতম'),
                dict(start=1.2, end=2.0, text='Thanks', assamese='ধন্যবাদ'),
                dict(start=2.2, end=3.0, text='Question', assamese='প্ৰশ্ন'),
                dict(start=3.2, end=3.9, text='Answer', assamese='উত্তৰ')]
    # SPEAKER_00 speaks seg 0 and 3; SPEAKER_01 speaks seg 1; seg 2 is mostly
    # SPEAKER_01 with SPEAKER_00 talking over; SPEAKER_02 is a spurious split
    # covering the tail of seg 3. Nothing covers 1.2-1.7 (music) -> seg 1 is
    # only 37.5% covered -> low-confidence.
    TURNS = [dict(start=0.2, end=1.0, speaker='SPEAKER_00'),
             dict(start=1.7, end=2.0, speaker='SPEAKER_01'),
             dict(start=2.2, end=3.0, speaker='SPEAKER_01'),
             dict(start=2.6, end=3.0, speaker='SPEAKER_00'),
             dict(start=3.2, end=3.6, speaker='SPEAKER_00'),
             dict(start=3.6, end=3.9, speaker='SPEAKER_02')]

    def _prepare(self, root):
        source = root / 'source.mp4'
        media.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                   'color=c=navy:s=320x180:d=4', '-f', 'lavfi', '-i',
                   'sine=frequency=200:duration=4', '-c:v', 'libx264', '-pix_fmt',
                   'yuv420p', '-c:a', 'aac', '-shortest', source])
        transcript = root / 'segments.json'
        project.save(self.SEGMENTS, transcript)
        job = root / 'job'
        self.assertEqual(main(['prepare', str(source), '--out', str(job),
                               '--source-language', 'en', '--transcript', str(transcript),
                               '--rights-confirmed']), 0)
        return job, job / 'project.json'

    def _run(self, argv):
        from io import StringIO
        buf = StringIO()
        with patch('sys.stdout', buf):
            rc = main(argv)
        return rc, buf.getvalue()

    def test_full_workflow(self):
        FakeDiarizer.reset(self.TURNS)
        FakeVoice.reset()
        FakeSeparator.reset()
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d))

            # 1. Detect speakers.
            with patch('assamese_agent.providers.PyannoteDiarizer', FakeDiarizer):
                rc, out = self._run(['diarize', str(p), '--num-speakers', '3'])
            self.assertEqual(rc, 0, out)
            self.assertIn('Detected 3 speaker(s)', out)
            self.assertEqual(FakeDiarizer.calls[0]['audio'], str(job / 'source.wav'))
            self.assertEqual(FakeDiarizer.calls[0]['num_speakers'], 3)
            self.assertTrue((job / 'diarization.json').exists())
            data = project.load(p)
            self.assertEqual([s['speaker_id'] for s in data['segments']],
                             ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_01', 'SPEAKER_00'])
            self.assertEqual(data['segments'][1]['speaker_flags'], [diarization.LOW_CONFIDENCE])
            self.assertEqual(data['segments'][2]['speaker_flags'], [diarization.OVERLAP])
            self.assertEqual(data['segments'][3]['speaker_flags'], [diarization.OVERLAP])
            self.assertEqual(data['speakers'], {k: {'voice': None, 'name': None}
                                                for k in ('SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02')})
            self.assertEqual(data['diarization']['flagged_segments'], [1, 2, 3])
            self.assertEqual(data['diarization']['model'], 'fake-diarizer')

            # 2. Approval is blocked until every speaker in use has a voice.
            self.assertEqual(main(['approve', str(p), '--reviewer', 'Test fixture only',
                                   '--native-reviewed']), 1)

            # 3. Review list shows flagged segments and missing voices.
            rc, out = self._run(['speakers', 'list', str(p)])
            self.assertEqual(rc, 0)
            self.assertIn('Segments needing review (3)', out)
            self.assertIn('#2', out)
            self.assertIn('overlap', out)
            self.assertIn('Voices still needed for: SPEAKER_00, SPEAKER_01, SPEAKER_02', out)

            # 4. Human corrections.
            self.assertEqual(main(['speakers', 'set', str(p), 'SPEAKER_00',
                                   '--name', 'শিক্ষক', '--voice', 'Amit']), 0)
            self.assertEqual(main(['speakers', 'set', str(p), 'SPEAKER_01', '--voice', 'Sita']), 0)
            # SPEAKER_02 was an over-split of SPEAKER_00.
            self.assertEqual(main(['speakers', 'merge', str(p), 'SPEAKER_02', '--into', 'SPEAKER_00']), 0)
            # Segment 2 was actually the student, not the teacher talking over.
            self.assertEqual(main(['speakers', 'assign', str(p), '--segments', '2',
                                   '--speaker', 'SPEAKER_01']), 0)
            self.assertEqual(main(['speakers', 'rename', str(p), 'SPEAKER_01', 'STUDENT']), 0)
            data = project.load(p)
            self.assertEqual(set(data['speakers']), {'SPEAKER_00', 'STUDENT'})
            self.assertEqual(data['speakers']['SPEAKER_00'], {'voice': 'Amit', 'name': 'শিক্ষক'})
            self.assertEqual(data['speakers']['STUDENT'], {'voice': 'Sita', 'name': None})
            self.assertEqual([s['speaker_id'] for s in data['segments']],
                             ['SPEAKER_00', 'STUDENT', 'STUDENT', 'SPEAKER_00'])
            self.assertEqual(data['segments'][2]['speaker_flags'], [])       # cleared by assign
            self.assertIsNone(data['segments'][2]['speaker_confidence'])
            self.assertEqual(data['segments'][3]['speaker_flags'], [diarization.OVERLAP])  # untouched
            self.assertNotIn('review', data)

            # 5. Approve and render: voices routed per corrected assignment.
            self.assertEqual(main(['approve', str(p), '--reviewer', 'Test fixture only',
                                   '--native-reviewed']), 0)
            with patch('assamese_agent.providers.IndicParlerVoice', FakeVoice), \
                    patch('assamese_agent.providers.DemucsSeparator', FakeSeparator):
                self.assertEqual(main(['render', str(p)]), 0)
            self.assertEqual([c['voice'] for c in FakeVoice.calls], ['Amit', 'Sita', 'Sita', 'Amit'])
            report = project.load(job / 'render-report.json')
            self.assertEqual([r['speaker_id'] for r in report['segment_voices']],
                             ['SPEAKER_00', 'STUDENT', 'STUDENT', 'SPEAKER_00'])

    def test_every_correction_invalidates_approval(self):
        FakeDiarizer.reset([dict(start=0.2, end=2.0, speaker='SPEAKER_00'),
                            dict(start=2.2, end=3.9, speaker='SPEAKER_01')])
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d))
            with patch('assamese_agent.providers.PyannoteDiarizer', FakeDiarizer):
                self.assertEqual(main(['diarize', str(p)]), 0)
            main(['speakers', 'set', str(p), 'SPEAKER_00', '--voice', 'Sita'])
            main(['speakers', 'set', str(p), 'SPEAKER_01', '--voice', 'Amit'])
            for edit in (['speakers', 'set', str(p), 'SPEAKER_00', '--name', 'T'],
                         ['speakers', 'assign', str(p), '--segments', '1', '--speaker', 'SPEAKER_01'],
                         ['speakers', 'rename', str(p), 'SPEAKER_01', 'S'],
                         ['speakers', 'merge', str(p), 'S', '--into', 'SPEAKER_00']):
                with self.subTest(edit=edit[1]):
                    self.assertEqual(main(['approve', str(p), '--reviewer', 'x',
                                           '--native-reviewed']), 0)
                    self.assertIn('review', project.load(p))
                    self.assertEqual(main(edit), 0)
                    self.assertNotIn('review', project.load(p))
                    self.assertEqual(main(['render', str(p), '--test-tone']), 1)

    def test_diarize_refuses_to_clobber_without_force(self):
        FakeDiarizer.reset([dict(start=0.2, end=3.9, speaker='SPEAKER_00')])
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d))
            with patch('assamese_agent.providers.PyannoteDiarizer', FakeDiarizer):
                self.assertEqual(main(['diarize', str(p)]), 0)
                main(['speakers', 'set', str(p), 'SPEAKER_00', '--name', 'Kept', '--voice', 'Sita'])
                self.assertEqual(main(['diarize', str(p)]), 1)           # refuses
                self.assertEqual(main(['diarize', str(p), '--force']), 0)
            # A label seen again keeps the name/voice the reviewer already gave it.
            self.assertEqual(project.load(p)['speakers']['SPEAKER_00'],
                             {'voice': 'Sita', 'name': 'Kept'})

    def test_isolate_vocals_diarizes_the_vocal_stem(self):
        FakeDiarizer.reset([dict(start=0.2, end=3.9, speaker='SPEAKER_00')])
        FakeSeparator.reset()
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d))
            with patch('assamese_agent.providers.PyannoteDiarizer', FakeDiarizer), \
                    patch('assamese_agent.providers.DemucsSeparator', FakeSeparator):
                self.assertEqual(main(['diarize', str(p), '--isolate-vocals', '--device', 'cuda']), 0)
            self.assertEqual(len(FakeSeparator.calls), 1)
            self.assertEqual(Path(FakeSeparator.calls[0]['audio']), job / 'source-full.wav')
            self.assertEqual(FakeSeparator.calls[0]['device'], 'cuda')
            # Fake separator returns its input as the vocal stem; the diarizer got that.
            self.assertEqual(FakeDiarizer.calls[0]['audio'], str(job / 'source-full.wav'))
            self.assertTrue(project.load(p)['diarization']['isolated_vocals'])

    def test_diarize_without_pyannote_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d))
            with patch.dict('sys.modules', {'pyannote': None, 'pyannote.audio': None}):
                self.assertEqual(main(['diarize', str(p)]), 1)
            self.assertNotIn('speakers', project.load(p))  # nothing half-written

    def test_diarize_with_no_detected_speech_stops_cleanly(self):
        FakeDiarizer.reset([])
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d))
            with patch('assamese_agent.providers.PyannoteDiarizer', FakeDiarizer):
                self.assertEqual(main(['diarize', str(p)]), 1)
            self.assertNotIn('speakers', project.load(p))

    def test_speakers_edit_errors(self):
        FakeDiarizer.reset([dict(start=0.2, end=2.0, speaker='SPEAKER_00'),
                            dict(start=2.2, end=3.9, speaker='SPEAKER_01')])
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d))
            with patch('assamese_agent.providers.PyannoteDiarizer', FakeDiarizer):
                main(['diarize', str(p)])
            before = project.load(p)
            for argv in (['speakers', 'set', str(p), 'SPEAKER_00'],                      # nothing to set
                         ['speakers', 'merge', str(p), 'SPEAKER_00', '--into', 'SPEAKER_00'],
                         ['speakers', 'merge', str(p), 'GHOST', '--into', 'SPEAKER_00'],
                         ['speakers', 'rename', str(p), 'SPEAKER_00', 'SPEAKER_01'],    # exists
                         ['speakers', 'rename', str(p), 'SPEAKER_00', 'bad id!'],
                         ['speakers', 'assign', str(p), '--segments', '9', '--speaker', 'SPEAKER_00'],
                         ['speakers', 'assign', str(p), '--segments', '0', '--speaker', 'GHOST']):
                with self.subTest(argv=argv[1:]):
                    self.assertEqual(main(argv), 1)
            self.assertEqual(project.load(p), before)  # failed edits never write

    def test_speakers_list_on_plain_project(self):
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d))
            rc, out = self._run(['speakers', 'list', str(p)])
            self.assertEqual(rc, 0)
            self.assertIn('No speaker assignments', out)


class SubtitleUnitTests(unittest.TestCase):
    """ASS generation and layout maths. No ffmpeg, no fonts."""

    def test_layout_scales_with_shorter_edge_across_formats(self):
        cases = {(1920, 1080): ('landscape', 54), (1080, 1920): ('portrait', 54),
                 (1080, 1080): ('square', 54), (1080, 1350): ('portrait', 54),   # 4:5
                 (3840, 2160): ('landscape', 108), (1280, 720): ('landscape', 36),
                 (720, 1280): ('portrait', 36)}
        for (w, h), (kind, size) in cases.items():
            with self.subTest(size=(w, h)):
                lay = subtitles.layout(w, h)
                self.assertEqual((lay['kind'], lay['font_size']), (kind, size))
                self.assertEqual((lay['width'], lay['height']), (w, h))

    def test_portrait_gets_taller_bottom_margin(self):
        land, port = subtitles.layout(1920, 1080), subtitles.layout(1080, 1920)
        self.assertAlmostEqual(land['margin_v'] / 1080, 0.06, places=2)
        self.assertAlmostEqual(port['margin_v'] / 1920, 0.16, places=2)
        self.assertGreater(port['margin_v'], land['margin_v'])

    def test_chars_per_line_bounded(self):
        self.assertEqual(subtitles.layout(3840, 2160)['chars_per_line'], subtitles.MAX_CHARS_PER_LINE)
        self.assertEqual(subtitles.layout(1920, 1080)['chars_per_line'], subtitles.MAX_CHARS_PER_LINE)
        port = subtitles.layout(1080, 1920)
        self.assertLess(port['chars_per_line'], subtitles.MAX_CHARS_PER_LINE)
        self.assertGreaterEqual(port['chars_per_line'], subtitles.MIN_CHARS_PER_LINE)
        self.assertEqual(port['max_chars'], port['chars_per_line'] * subtitles.MAX_LINES)

    def test_scale_and_style(self):
        self.assertEqual(subtitles.layout(1920, 1080, scale=1.5)['font_size'], 81)
        self.assertEqual(subtitles.layout(1920, 1080, scale=0.5)['font_size'], 27)
        box = subtitles.layout(1920, 1080, style='box')
        self.assertEqual((box['border_style'], box['shadow']), (3, 0))
        outline = subtitles.layout(1920, 1080)
        self.assertEqual(outline['border_style'], 1)
        self.assertGreater(outline['outline'], 0)
        for bad in [dict(width=8, height=8), dict(width=1920, height=1080, scale=0),
                    dict(width=1920, height=1080, scale=5), dict(width=1920, height=1080, scale='1'),
                    dict(width=1920, height=1080, style='neon'), dict(width=1920.0, height=1080)]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    subtitles.layout(**bad)

    def test_escape_text(self):
        self.assertEqual(subtitles.escape_text('a {b} c\\d'), 'a (b) c/d')
        self.assertEqual(subtitles.escape_text('one\r\ntwo\n\n  three  '), 'one\\Ntwo\\Nthree')
        self.assertEqual(subtitles.escape_text('  spaced   out  '), 'spaced out')

    def test_chunk_words(self):
        words = ['aa', 'bbb', 'cc', 'dddd', 'e']
        self.assertEqual(subtitles.chunk_words(words, 6), [['aa', 'bbb'], ['cc'], ['dddd', 'e']])
        self.assertEqual(subtitles.chunk_words(['averyverylongword'], 5), [['averyverylongword']])
        self.assertEqual(subtitles.chunk_words(words, 100), [words])
        self.assertEqual(subtitles.chunk_words([], 10), [])

    def test_proportional_times(self):
        spans = subtitles.proportional_times(1.0, 3.0, [1, 1, 2])
        self.assertAlmostEqual(spans[0][1], 1.5)
        self.assertAlmostEqual(spans[1][1], 2.0)
        self.assertEqual(spans[2], (2.0, 3.0))
        self.assertEqual(spans[0][0], 1.0)
        self.assertEqual(subtitles.proportional_times(1.0, 1.0, [1, 1]), [(1.0, 1.0), (1.0, 1.0)])

    def test_timestamp(self):
        self.assertEqual(subtitles.timestamp(0), '0:00:00.00')
        self.assertEqual(subtitles.timestamp(3661.257), '1:01:01.26')
        self.assertEqual(subtitles.timestamp(59.999), '0:01:00.00')

    LAY = subtitles.layout(1920, 1080)

    def test_events_plain_holds_last_chunk_to_segment_end(self):
        segs = [dict(start=1.0, end=4.0, assamese='নমস্কাৰ আপোনাৰ নাম কি')]
        rows = subtitles.events(segs, [3.0], self.LAY, 'none')
        self.assertEqual(rows, [(1.0, 4.0, 'নমস্কাৰ আপোনাৰ নাম কি')])

    def test_events_split_long_text_into_readable_chunks(self):
        text = ' '.join(['শব্দ'] * 60)   # 5 chars x 60 = far beyond 2 lines
        segs = [dict(start=0.0, end=10.0, assamese=text)]
        rows = subtitles.events(segs, [9.0], self.LAY, 'none')
        self.assertGreater(len(rows), 1)
        for s_, e_, t in rows:
            self.assertLessEqual(len(t), self.LAY['max_chars'])
        # Contiguous, in order, ending at segment end.
        for (s0, e0, _), (s1, e1, _) in zip(rows, rows[1:]):
            self.assertAlmostEqual(e0, s1)
        self.assertEqual(rows[0][0], 0.0)
        self.assertEqual(rows[-1][1], 10.0)            # last chunk holds through the padding
        self.assertLess(rows[-1][0], 9.0)              # ...but all chunk starts lie in the speech span
        self.assertAlmostEqual(rows[-2][1], rows[-1][0])

    def test_events_word_highlight(self):
        segs = [dict(start=1.0, end=3.0, assamese='এক দুই তিনি')]
        rows = subtitles.events(segs, [2.2], self.LAY, 'word')
        self.assertEqual(len(rows), 4)   # 3 words + plain hold through padding
        self.assertEqual(rows[0][0], 1.0)
        for (s0, e0, _), (s1, e1, _) in zip(rows, rows[1:]):
            self.assertAlmostEqual(e0, s1)
        self.assertAlmostEqual(rows[2][1], 2.2)
        self.assertEqual(rows[3], (2.2, 3.0, 'এক দুই তিনি'))
        self.assertEqual(rows[0][2], '{\\1c&H00FFFF&}এক{\\1c&HFFFFFF&} দুই তিনি')
        self.assertEqual(rows[1][2], 'এক {\\1c&H00FFFF&}দুই{\\1c&HFFFFFF&} তিনি')
        self.assertEqual(rows[2][2], 'এক দুই {\\1c&H00FFFF&}তিনি{\\1c&HFFFFFF&}')
        # Word durations are proportional to character count.
        self.assertAlmostEqual(rows[0][1] - rows[0][0], 1.2 * 2 / 9)
        self.assertAlmostEqual(rows[2][1] - rows[2][0], 1.2 * 4 / 9)

    def test_events_word_highlight_falls_back_when_too_fast(self):
        segs = [dict(start=1.0, end=1.1, assamese='এক দুই তিনি চাৰি')]
        rows = subtitles.events(segs, [1.1], self.LAY, 'word')
        self.assertEqual(rows, [(1.0, 1.1, 'এক দুই তিনি চাৰি')])

    def test_events_clamps_speech_end_and_skips_empty(self):
        segs = [dict(start=1.0, end=2.0, assamese='ক'), dict(start=3.0, end=4.0, assamese='  ')]
        rows = subtitles.events(segs, [9.0, 3.5], self.LAY, 'none')
        self.assertEqual(rows, [(1.0, 2.0, 'ক')])
        with self.assertRaises(ValueError):
            subtitles.events(segs, [1.5], self.LAY, 'none')
        with self.assertRaises(ValueError):
            subtitles.events(segs, [1.5, 3.5], self.LAY, 'rainbow')

    def test_render_ass_document(self):
        segs = [dict(start=0.5, end=2.0, assamese='নমস্কাৰ')]
        doc = subtitles.render_ass(segs, [1.5], subtitles.layout(1080, 1920), 'Noto Sans Bengali')
        self.assertIn('PlayResX: 1080\nPlayResY: 1920', doc)
        self.assertIn('ScaledBorderAndShadow: yes', doc)
        style = [l for l in doc.splitlines() if l.startswith('Style: ')][0]
        fields = style[len('Style: '):].split(',')
        self.assertEqual(fields[:3], ['Assamese', 'Noto Sans Bengali', '54'])
        self.assertEqual(fields[3], subtitles.WHITE)
        self.assertEqual(fields[15], '1')                    # BorderStyle outline
        self.assertEqual(fields[18], '2')                    # bottom-centre
        self.assertEqual(fields[19:22], ['65', '65', '307'])  # margins L,R,V for 1080x1920
        self.assertIn('Dialogue: 0,0:00:00.50,0:00:02.00,Assamese,,0,0,0,,নমস্কাৰ', doc)
        box = subtitles.render_ass(segs, [1.5], subtitles.layout(1080, 1920, style='box'), 'X')
        bfields = [l for l in box.splitlines() if l.startswith('Style: ')][0].split(',')
        self.assertEqual(bfields[15], '3')
        self.assertEqual(bfields[5], subtitles.BOX)
        self.assertEqual(bfields[6], subtitles.BOX)
        for bad in ('', ' ', 'A,B', None):
            with self.assertRaises(ValueError):
                subtitles.render_ass(segs, [1.5], self.LAY, bad)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class SubtitleBurnIntegrationTests(unittest.TestCase):
    """Burning with libass. Uses whatever font is installed to exercise the
    plumbing; Assamese glyph coverage is asserted via the report warning."""

    FONT = 'DejaVu Sans'
    FONT_FILE = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

    def setUp(self):
        if media.font_coverage(self.FONT) is None:
            self.skipTest(f'{self.FONT} not installed')

    def _source(self, root, size='1920x1080', rotation=None):
        source = root / 'source.mp4'
        media.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                   f'color=c=navy:s={size}:d=4', '-f', 'lavfi', '-i',
                   'sine=frequency=200:duration=4', '-c:v', 'libx264', '-pix_fmt',
                   'yuv420p', '-c:a', 'aac', '-shortest', source])
        if rotation is not None:
            rotated = root / 'rotated.mp4'
            media.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-display_rotation', rotation,
                       '-i', source, '-c', 'copy', rotated])
            return rotated
        return source

    def _project(self, root, source):
        transcript = root / 'segments.json'
        project.save([dict(start=0.5, end=2.0, text='Hello', assamese='নমস্কাৰ, আপোনাৰ নাম কি?'),
                      dict(start=2.5, end=3.8, text='Bye', assamese='ধন্যবাদ')], transcript)
        job = root / 'job'
        self.assertEqual(main(['prepare', str(source), '--out', str(job), '--source-language', 'en',
                               '--transcript', str(transcript), '--rights-confirmed']), 0)
        p = job / 'project.json'
        self.assertEqual(main(['approve', str(p), '--reviewer', 'x', '--native-reviewed']), 0)
        return job, p

    @staticmethod
    def _bottom_band_mean(video, at, band=0.25):
        """Mean grey level of the bottom `band` of the frame at time `at`."""
        w, h = media.video_dimensions(video)
        crop_h = int(h * band)
        raw = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(at), '-i', str(video),
                                       '-frames:v', '1', '-vf', f'crop={w}:{crop_h}:0:{h - crop_h}',
                                       '-f', 'rawvideo', '-pix_fmt', 'gray', '-'])
        return sum(raw) / len(raw)

    def test_video_dimensions_follow_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plain = self._source(root)
            self.assertEqual(media.video_dimensions(plain), (1920, 1080))
            self.assertEqual(media.video_dimensions(self._source(root, rotation='90')), (1080, 1920))
            self.assertEqual(media.video_dimensions(self._source(root, rotation='180')), (1920, 1080))
            self.assertEqual(media.video_dimensions(self._source(root, rotation='-90')), (1080, 1920))

    def test_burn_draws_captions_only_where_speech_is(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job, p = self._project(root, self._source(root))
            self.assertEqual(main(['render', str(p), '--test-tone', '--burn-subtitles',
                                   '--subtitle-font', self.FONT]), 0)
            out = job / 'TEST-TONE-not-assamese-subtitled.mp4'
            self.assertTrue(out.exists())
            self.assertFalse((job / 'TEST-TONE-not-assamese.mp4').exists())
            # No soft subtitle stream when burned (avoids doubled captions).
            streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                                          '-of', 'json', str(out)]))['streams']
            self.assertEqual([s_['codec_type'] for s_ in streams], ['video', 'audio'])
            # Pixels: caption band is brighter during speech than in the gap / before.
            background = self._bottom_band_mean(out, 0.1)
            self.assertGreater(self._bottom_band_mean(out, 1.0) - background, 1.0)
            self.assertGreater(self._bottom_band_mean(out, 3.0) - background, 0.5)
            self.assertAlmostEqual(self._bottom_band_mean(out, 2.2), background, delta=0.3)
            ass = (job / 'assamese.ass').read_text(encoding='utf-8')
            self.assertIn('PlayResX: 1920', ass)
            self.assertIn(f'Style: Assamese,{self.FONT},54,', ass)
            report = project.load(job / 'render-report.json')
            self.assertEqual(report['subtitles']['mode'], 'burned')
            self.assertEqual(report['subtitles']['layout'], 'landscape')
            self.assertEqual(report['subtitles']['font_size_px'], 54)
            self.assertEqual(report['subtitles']['highlight'], 'none')
            self.assertIsNone(report['subtitles']['word_timing'])
            # DejaVu has no Bengali glyphs: the report must say so.
            self.assertTrue(any('boxes' in w for w in report['warnings']))

    def test_word_highlight_events_in_burned_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job, p = self._project(root, self._source(root, size='1080x1920'))
            self.assertEqual(main(['render', str(p), '--test-tone', '--burn-subtitles',
                                   '--highlight', 'word', '--subtitle-style', 'box',
                                   '--subtitle-font', self.FONT]), 0)
            ass = (job / 'assamese.ass').read_text(encoding='utf-8')
            self.assertIn('PlayResX: 1080\nPlayResY: 1920', ass)
            dialogues = [l for l in ass.splitlines() if l.startswith('Dialogue:')]
            self.assertEqual(len(dialogues), 5)      # 4 words + 1 word
            self.assertEqual(sum('&H00FFFF&' in l for l in dialogues), 5)
            report = project.load(job / 'render-report.json')
            self.assertEqual(report['subtitles']['layout'], 'portrait')
            self.assertEqual(report['subtitles']['word_timing'], 'estimated')
            self.assertEqual(report['subtitles']['style'], 'box')
            out = job / 'TEST-TONE-not-assamese-subtitled.mp4'
            self.assertGreater(self._bottom_band_mean(out, 1.0) - self._bottom_band_mean(out, 0.1), 1.0)

    def test_rotated_phone_video_is_laid_out_as_portrait(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job, p = self._project(root, self._source(root, rotation='90'))
            self.assertEqual(main(['render', str(p), '--test-tone', '--burn-subtitles',
                                   '--subtitle-font', self.FONT]), 0)
            report = project.load(job / 'render-report.json')
            self.assertEqual(report['subtitles']['video_size'], [1080, 1920])
            self.assertEqual(report['subtitles']['layout'], 'portrait')

    def test_font_file_is_copied_into_project(self):
        if not Path(self.FONT_FILE).is_file() or not shutil.which('fc-scan'):
            self.skipTest('font file / fc-scan unavailable')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job, p = self._project(root, self._source(root))
            self.assertEqual(main(['render', str(p), '--test-tone', '--burn-subtitles',
                                   '--subtitle-font-file', self.FONT_FILE]), 0)
            self.assertTrue((job / 'fonts' / 'DejaVuSans.ttf').exists())
            report = project.load(job / 'render-report.json')
            self.assertEqual(report['subtitles']['font'], 'DejaVu Sans')
            ass = (job / 'assamese.ass').read_text(encoding='utf-8')
            self.assertIn('Style: Assamese,DejaVu Sans,', ass)

    def test_auto_font_detection_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job, p = self._project(root, self._source(root))
            # Auto-detect succeeds -> used.
            with patch('assamese_agent.media.find_assamese_font', return_value=self.FONT):
                self.assertEqual(main(['render', str(p), '--test-tone', '--burn-subtitles']), 0)
            self.assertEqual(project.load(job / 'render-report.json')['subtitles']['font'], self.FONT)
            # No suitable font -> stops before any synthesis work.
            job2 = root / 'job2'
            shutil.copytree(job, job2, ignore=shutil.ignore_patterns('voice-*', 'fitted-*', '*.mp4'))
            p2 = job2 / 'project.json'
            with patch('assamese_agent.media.find_assamese_font',
                       side_effect=ValueError('No installed font covers Assamese')):
                self.assertEqual(main(['render', str(p2), '--test-tone', '--burn-subtitles']), 1)
            self.assertFalse((job2 / 'voice-0000.wav').exists())

    def test_bad_subtitle_options_stop_early(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job, p = self._project(root, self._source(root))
            for extra in (['--subtitle-font', 'No Such Font Family 12345'],
                          ['--subtitle-font', self.FONT, '--subtitle-font-file', self.FONT_FILE],
                          ['--subtitle-font', self.FONT, '--subtitle-scale', '9'],
                          ['--subtitle-font-file', '/does/not/exist.ttf']):
                with self.subTest(extra=extra):
                    self.assertEqual(main(['render', str(p), '--test-tone', '--burn-subtitles'] + extra), 1)
                    self.assertFalse((job / 'voice-0000.wav').exists())

    def test_soft_subtitles_unchanged_without_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job, p = self._project(root, self._source(root))
            self.assertEqual(main(['render', str(p), '--test-tone']), 0)
            self.assertTrue((job / 'TEST-TONE-not-assamese.mp4').exists())
            self.assertFalse((job / 'assamese.ass').exists())
            self.assertEqual(project.load(job / 'render-report.json')['subtitles'],
                             {'mode': 'soft', 'track': 'mov_text'})


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class BackgroundIntegrationTests(unittest.TestCase):
    def test_duck_attenuates_only_inside_windows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bg, out = root / 'bg.wav', root / 'ducked.wav'
            write_tone(bg, 3.0, freq=200, amp=10000, sr=44100, channels=2)
            media.duck(bg, [(1.0, 2.0)], 20, out)
            self.assertEqual(media.wav_info(out), (2, 44100, round(3.0 * 44100)))
            before, inside, after = rms(out, 0.2, 0.8), rms(out, 1.2, 1.8), rms(out, 2.5, 2.9)
            ref = rms(bg, 0.2, 0.8)
            self.assertAlmostEqual(before / ref, 1.0, delta=0.02)   # untouched
            self.assertAlmostEqual(inside / ref, 0.1, delta=0.01)   # -20 dB
            self.assertAlmostEqual(after / ref, 1.0, delta=0.02)    # recovered

    def test_duck_strength_flag_scales_attenuation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bg = root / 'bg.wav'
            write_tone(bg, 2.0, freq=200, amp=10000, sr=24000, channels=1)
            ref = rms(bg, 0.6, 1.4)
            for db, expect in [(0, 1.0), (6, 0.501), (12, 0.251), (30, 0.0316)]:
                out = root / f'd{db}.wav'
                media.duck(bg, [(0.5, 1.5)], db, out)
                with self.subTest(db=db):
                    self.assertAlmostEqual(rms(out, 0.6, 1.4) / ref, expect, delta=0.01)

    def test_zero_db_duck_is_transparent_for_stereo(self):
        # Verifies the mono envelope -> stereo upmix has unity gain on both channels.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bg, out = root / 'bg.wav', root / 'out.wav'
            write_tone(bg, 1.0, freq=300, amp=12000, sr=48000, channels=2)
            media.duck(bg, [(0.2, 0.8)], 0, out)
            with wave.open(str(bg), 'rb') as a, wave.open(str(out), 'rb') as b:
                ia, ib = array('h'), array('h')
                ia.frombytes(a.readframes(a.getnframes()))
                ib.frombytes(b.readframes(b.getnframes()))
            self.assertEqual(len(ia), len(ib))
            self.assertLessEqual(max(abs(x - y) for x, y in zip(ia, ib)), 2)

    def test_mix_preserves_levels_and_background_format(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bg, voice, out = root / 'bg.wav', root / 'voice.wav', root / 'mix.wav'
            write_tone(bg, 2.0, freq=200, amp=6000, sr=44100, channels=2)
            # Voice is 24 kHz mono, present only in the second half.
            silence = [0] * 24000
            tone = [int(6000 * math.sin(2 * math.pi * 1000 * i / 24000)) for i in range(24000)]
            with wave.open(str(voice), 'wb') as f:
                f.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                f.writeframes(array('h', silence + tone).tobytes())
            media.mix(bg, voice, out)
            self.assertEqual(media.wav_info(out)[:2], (2, 44100))
            bg_ref = rms(bg, 0.1, 0.9)
            # First half: only background, at its original level (normalize=0).
            self.assertAlmostEqual(rms(out, 0.1, 0.9) / bg_ref, 1.0, delta=0.03)
            # Second half: power adds (uncorrelated tones) => RMS ~ sqrt(2) x.
            self.assertAlmostEqual(rms(out, 1.1, 1.9) / bg_ref, math.sqrt(2), delta=0.08)

    def _prepare(self, root, segments, speakers=None):
        source = root / 'source.mp4'
        media.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                   'color=c=navy:s=320x180:d=4', '-f', 'lavfi', '-i',
                   'sine=frequency=200:duration=4', '-c:v', 'libx264', '-pix_fmt',
                   'yuv420p', '-c:a', 'aac', '-shortest', source])
        transcript = root / 'segments.json'
        project.save({'segments': segments, **({'speakers': speakers} if speakers else {})},
                     transcript)
        job = root / 'job'
        self.assertEqual(main(['prepare', str(source), '--out', str(job),
                               '--source-language', 'en', '--transcript', str(transcript),
                               '--rights-confirmed']), 0)
        p = job / 'project.json'
        self.assertEqual(main(['approve', str(p), '--reviewer', 'Test fixture only',
                               '--native-reviewed']), 0)
        return job, p

    SEGMENTS = [dict(start=1.0, end=2.0, text='Hello', assamese='নমস্কাৰ')]

    def test_render_keeps_background_outside_speech(self):
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d), self.SEGMENTS)
            self.assertEqual(main(['render', str(p), '--test-tone', '--duck-db', '20']), 0)
            mixed = job / 'TEST-TONE-mix.wav'
            self.assertTrue(mixed.exists())
            self.assertTrue((job / 'background-ducked.wav').exists())
            report = project.load(job / 'render-report.json')
            self.assertEqual(report['background'], 'original')  # test-tone default
            self.assertEqual(report['duck_db'], 20.0)
            self.assertTrue(report['source_speech_audible'])
            # Background (200 Hz sine from the source) is present where nobody speaks...
            outside = rms(mixed, 0.2, 0.8)
            self.assertGreater(outside, 500)
            # ...and ducked under the speech window. Tone and background are
            # uncorrelated, so powers add: expected RMS = sqrt(tone^2 + (bg*g)^2).
            inside = rms(mixed, 1.2, 1.8)
            tone_only = rms(job / 'TEST-TONE.wav', 1.2, 1.8)
            expected = math.sqrt(tone_only ** 2 + (outside * media.duck_gain(20)) ** 2)
            self.assertAlmostEqual(inside / expected, 1.0, delta=0.03)
            # Sanity: without ducking the same window would be much louder.
            self.assertLess(inside, math.sqrt(tone_only ** 2 + outside ** 2) * 0.75)
            # The ducked stem itself is exactly -20 dB inside the window.
            ducked = job / 'background-ducked.wav'
            self.assertAlmostEqual(rms(ducked, 1.2, 1.8) / rms(ducked, 0.2, 0.8), 0.1, delta=0.01)
            # Final mp4 still has video/audio/subtitle streams.
            streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                                '-show_streams', '-of', 'json',
                                str(job / 'TEST-TONE-not-assamese.mp4')]))['streams']
            self.assertEqual([s['codec_type'] for s in streams], ['video', 'audio', 'subtitle'])

    def test_render_background_none_is_silent_outside_speech(self):
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d), self.SEGMENTS)
            self.assertEqual(main(['render', str(p), '--test-tone', '--background', 'none']), 0)
            self.assertFalse((job / 'TEST-TONE-mix.wav').exists())
            self.assertFalse((job / 'background-ducked.wav').exists())
            self.assertEqual(rms(job / 'TEST-TONE.wav', 0.2, 0.8), 0.0)
            report = project.load(job / 'render-report.json')
            self.assertEqual(report['background'], 'none')
            self.assertIsNone(report['duck_db'])

    def test_render_separate_routes_through_separator(self):
        FakeVoice.reset()
        FakeSeparator.reset()
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d), self.SEGMENTS)
            with patch('assamese_agent.providers.IndicParlerVoice', FakeVoice), \
                    patch('assamese_agent.providers.DemucsSeparator', FakeSeparator):
                self.assertEqual(main(['render', str(p), '--device', 'cuda', '--duck-db', '9']), 0)
            self.assertEqual(len(FakeSeparator.calls), 1)
            call = FakeSeparator.calls[0]
            self.assertEqual(Path(call['audio']), job / 'source-full.wav')
            self.assertEqual(Path(call['workdir']), job / 'separation')
            self.assertEqual(call['device'], 'cuda')
            for name in ('source-full.wav', 'background.wav', 'background-ducked.wav',
                         'assamese.wav', 'assamese-mix.wav', 'assamese-dub.mp4'):
                self.assertTrue((job / name).exists(), name)
            report = project.load(job / 'render-report.json')
            self.assertEqual(report['background'], 'separate')
            self.assertEqual(report['duck_db'], 9.0)
            self.assertFalse(report['source_speech_audible'])
            # Background stem preserved outside speech, ducked ~9 dB inside.
            self.assertGreater(rms(job / 'background-ducked.wav', 0.2, 0.8), 500)
            ratio = rms(job / 'background-ducked.wav', 1.2, 1.8) / \
                rms(job / 'background-ducked.wav', 0.2, 0.8)
            self.assertAlmostEqual(ratio, media.duck_gain(9), delta=0.02)

    def test_render_separate_without_demucs_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d), self.SEGMENTS)
            # Explicitly request separation in test-tone mode with no demucs installed.
            with patch.dict('sys.modules', {'demucs': None}):
                self.assertEqual(main(['render', str(p), '--test-tone',
                                       '--background', 'separate']), 1)

    def test_bad_duck_db_rejected_before_any_work(self):
        with tempfile.TemporaryDirectory() as d:
            job, p = self._prepare(Path(d), self.SEGMENTS)
            for bad in ('-3', '99', 'nan'):
                with self.subTest(bad=bad):
                    self.assertEqual(main(['render', str(p), '--test-tone',
                                           '--duck-db', bad]), 1)
                    self.assertFalse((job / 'voice-0000.wav').exists())


if __name__ == '__main__':
    unittest.main()
