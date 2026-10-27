"""Lazy-loaded model adapters. Paid/external translation is explicit opt-in."""
import json
import os
import urllib.request


class WhisperASR:
    def __init__(self, model='small', device='cpu'):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device,
                                  compute_type='int8' if device == 'cpu' else 'float16')

    def transcribe(self, audio, language):
        if language == 'as':
            raise ValueError('Assamese input ASR is not validated by this adapter. '
                             'Import a native-reviewed timestamped transcript instead.')
        segments, _ = self.model.transcribe(str(audio), language=language, vad_filter=True)
        return [dict(start=s.start, end=s.end, text=s.text.strip())
                for s in segments if s.text.strip()]


def sarvam_translate(text, source, allow_cloud=False):
    if not allow_cloud:
        raise ValueError('Cloud translation needs --allow-cloud: sends text to Sarvam and may cost money')
    if not 0 < len(text) <= 2000:
        raise ValueError('Sarvam segment must contain 1-2000 characters; split it first')
    key = os.environ.get('SARVAM_API_KEY')
    if not key:
        raise ValueError('Set SARVAM_API_KEY securely in the process environment')
    payload = dict(input=text, source_language_code=source,
                   target_language_code='as-IN', model='sarvam-translate:v1', mode='formal')
    request = urllib.request.Request('https://api.sarvam.ai/translate',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json',
                                              'api-subscription-key': key}, method='POST')
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    result = data.get('translated_text')
    if not isinstance(result, str) or not result.strip():
        raise ValueError('Translation response contained no translated_text')
    return result.strip()


class IndicParlerVoice:
    def __init__(self, device='cpu', speaker='Sita'):
        if speaker not in ('Sita', 'Amit'):
            raise ValueError('Use a documented Assamese stock voice: Sita or Amit')
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        self.device = device
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            'ai4bharat/indic-parler-tts').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-parler-tts')
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path)
        self.description = (f'{speaker} speaks Assamese with a clear, natural voice, '
                            'moderate speed and a very clean, close recording.')
        self.torch = torch

    def synthesize(self, text, output):
        import soundfile as sf
        if len(text) > 500:
            raise ValueError('Split long Assamese segments into <=500 characters before synthesis')
        desc = self.description_tokenizer(self.description, return_tensors='pt').to(self.device)
        prompt = self.tokenizer(text, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            result = self.model.generate(input_ids=desc.input_ids,
                                         attention_mask=desc.attention_mask,
                                         prompt_input_ids=prompt.input_ids,
                                         prompt_attention_mask=prompt.attention_mask)
        sf.write(str(output), result.cpu().numpy().squeeze(), self.model.config.sampling_rate)


class DemucsSeparator:
    """Vocal/background source separation via Demucs (htdemucs, two stems).

    Lazy: the import check only runs when a render with --background separate
    actually starts. Output is the "no_vocals" stem: music, effects and room
    tone with the original speech removed.
    """

    def __init__(self, device='cpu', model='htdemucs'):
        try:
            import demucs  # noqa: F401
        except ImportError as exc:
            raise ImportError("Background separation needs: pip install -e '.[separation]' "
                              "(or use --background original / none)") from exc
        self.device = device
        self.model = model

    def separate_stems(self, audio, workdir):
        """Run two-stem separation. Returns {'vocals': Path, 'no_vocals': Path}."""
        import subprocess
        import sys
        from pathlib import Path
        audio, workdir = Path(audio), Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'demucs.separate', '--two-stems', 'vocals',
                        '-n', self.model, '-d', self.device, '-o', str(workdir), str(audio)],
                       check=True, capture_output=True)
        stems = {name: workdir / self.model / audio.stem / f'{name}.wav'
                 for name in ('vocals', 'no_vocals')}
        missing = [name for name, path in stems.items() if not path.is_file()]
        if missing:
            raise ValueError(f'Separation finished but produced no {", ".join(missing)} stem')
        return stems

    def separate(self, audio, workdir):
        """Background only (music, effects, room tone with speech removed)."""
        return self.separate_stems(audio, workdir)['no_vocals']


class PyannoteDiarizer:
    """Who-spoke-when via pyannote.audio speaker-diarization-3.1.

    Lazy: nothing is imported or downloaded until `diarize` runs. The model is
    gated on Hugging Face: accept its terms at hf.co/pyannote/speaker-diarization-3.1
    and hf.co/pyannote/segmentation-3.0, then set HF_TOKEN in the environment.
    Audio stays local; only the model weights are fetched.

    The 3.1 pipeline is overlap-aware, but overlapped speech and loud music
    still degrade it. Downstream code flags segments rather than trusting the
    output blindly; see diarization.assign_speakers.
    """
    MODEL = 'pyannote/speaker-diarization-3.1'

    def __init__(self, device='cpu'):
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError("Speaker detection needs: pip install -e '.[diarization]'") from exc
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise ValueError('Set HF_TOKEN in the process environment after accepting the '
                             f'terms of {self.MODEL} on Hugging Face')
        import torch
        self.pipeline = Pipeline.from_pretrained(self.MODEL, use_auth_token=token)
        if self.pipeline is None:
            raise ValueError(f'Could not load {self.MODEL}. Check HF_TOKEN and model terms')
        self.pipeline.to(torch.device(device))

    def diarize(self, audio, num_speakers=None, min_speakers=None, max_speakers=None):
        """Return [{'start': s, 'end': e, 'speaker': 'SPEAKER_00'}, ...] sorted by start."""
        options = {k: v for k, v in (('num_speakers', num_speakers),
                                     ('min_speakers', min_speakers),
                                     ('max_speakers', max_speakers)) if v is not None}
        annotation = self.pipeline(str(audio), **options)
        turns = [dict(start=float(turn.start), end=float(turn.end), speaker=str(label))
                 for turn, _, label in annotation.itertracks(yield_label=True)]
        return sorted(turns, key=lambda t: (t['start'], t['end'], t['speaker']))
