# Assamese AI | eSkillVeda

A review-first local MVP for translating and dubbing an owned or licensed video into Assamese. This is a command-line pipeline, not a deployed web platform or a trained new language model.

**Status:** offline media plumbing tested. Production ASR, translation and Assamese TTS adapters are implemented but have not been run end-to-end with real models in the build environment. Native Assamese listening review is still required. Inspired-video generation is a planning interface, not a video generator.

## What works

- Local video -> mono audio -> timestamped transcript (import JSON or optional faster-whisper).
- Optional Sarvam translation to Assamese, with explicit cloud/cost opt-in. You can instead type a human translation in the project JSON.
- Native-review checkpoint, tied to a hash of the video, transcript and translation.
- Assamese stock voice adapter using AI4Bharat Indic Parler-TTS, Sita or Amit. No cloning.
- Per-segment timing fit, separate WAV/SRT, and dubbed MP4 with optional soft subtitles.
- Background preservation: vocals separated from music/effects/room tone, background dynamically ducked under the dub with a configurable depth (`--duck-db`).
- Burned-in Assamese captions (`--burn-subtitles`) rendered with libass/HarfBuzz for correct Bengali-script shaping, auto-sized for landscape, portrait and square video, with optional word-by-word highlighting.
- New-video brief/planning JSON. Original scene/script authoring and video-generation adapter remain future work.

No public uploads, automatic publishing, training, voice cloning, background paid jobs or storage service are included. By default the renderer separates the source speech from music/effects/room tone (Demucs, optional extra) and mixes the dub over the ducked background; `--background none` restores full audio replacement. This is voiceover dubbing, not lip sync or multi-speaker impersonation.

## Setup

Python 3.10+ and system `ffmpeg` + `ffprobe` are required.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
assamese-agent doctor
python3 -m unittest discover -s tests -v
# Optional transcription models, downloaded only when an ASR job starts:
pip install -e '.[asr]'
# Optional Assamese TTS. GPU recommended; CPU execution can be slow:
pip install -e '.[tts]'
# Optional vocal/background separation (Demucs). Needed for the default --background separate:
pip install -e '.[separation]'
# Optional speaker detection (pyannote.audio). Gated model: accept terms on Hugging Face, set HF_TOKEN:
pip install -e '.[diarization]'
```

The development TTS dependency follows the upstream Git repository. Before production, pin an audited commit and all package versions in a lockfile. Dependency installation and model inference have NOT been validated here; use a clean deployment environment and run the acceptance plan in `docs/EVALUATION.md`.

### 1. Prepare transcript

Use a short video you own or have permission to adapt. Confirm participant consent separately, especially when minors appear. The flag records your assertion; it is not legal verification.

```bash
assamese-agent prepare media/input.mp4 --out outputs/lesson-1 \
  --source-language en --rights-confirmed
```

Default faster-whisper model is `small` on CPU. Choose `--model large-v3 --device cuda` only after checking GPU capacity. The source can be English, Hindi or another language supported by the installed Whisper model. Assamese-source ASR is deliberately blocked by this adapter because its quality is not validated. Import a native-reviewed transcript in that case.

To bypass ASR/model download:

```bash
assamese-agent prepare media/input.mp4 --out outputs/lesson-1 \
  --source-language en --transcript examples/segments.json --rights-confirmed
```

Inputs must be local files. The MVP accepts up to 30-minute inputs and at most 2,000 ordered, non-overlapping segments. Keep segments short; Assamese text is limited to 500 characters per segment. Split long segments and review their times.

### 2. Translate

Either edit `outputs/lesson-1/project.json`, adding `assamese` to each segment, or explicitly opt into external translation. Never paste keys into chat or commit them. Set `SARVAM_API_KEY` using your normal secret manager; `.env.example` documents the field but the CLI does not automatically load it.

```bash
# This sends source text to Sarvam and may charge your account.
# Confirm provider terms, privacy policy and budget first.
assamese-agent translate outputs/lesson-1/project.json \
  --source-code en-IN --allow-cloud
```

The request explicitly selects `sarvam-translate:v1`, `formal` mode and `as-IN`, not the default model. Sarvam currently documents 22 scheduled Indian languages for this model. For other source languages, use human translation or add another verified adapter. Segments over the API limit (2,000 characters) stop instead of being truncated. Successfully translated segments are saved incrementally and existing human translations are not overwritten.

#### Multi-speaker dubbing

Each segment may carry an optional `speaker_id`, and the project declares a top-level `speakers` map from those ids to documented stock voices (`Sita` or `Amit`). See `examples/segments.json` for the shape. You can supply the map either inside an imported transcript (`{"speakers": {...}, "segments": [...]}`) or via repeated `--speaker-map ID=Voice` flags on `prepare`:

```bash
assamese-agent prepare media/input.mp4 --out outputs/lesson-1 \
  --source-language en --transcript examples/segments.json \
  --speaker-map NARRATOR=Sita --speaker-map GUEST=Amit \
  --rights-confirmed
```

Each `speakers` entry is either a bare stock voice (`"NARRATOR": "Sita"`) or an object with a voice and a human-readable name in any language (`"SPEAKER_00": {"voice": "Amit", "name": "শিক্ষক"}`). Validation enforces that every `speaker_id` used in a segment is declared in the project's `speakers` map, and that each mapped voice is one of the documented stock voices before approval.

#### Automatic speaker detection and correction

Instead of labelling segments by hand, detect who speaks when and then correct the result:

```bash
# Accept the model terms at hf.co/pyannote/speaker-diarization-3.1 and
# hf.co/pyannote/segmentation-3.0 first. Weights are downloaded; audio stays local.
HF_TOKEN=... assamese-agent diarize outputs/lesson-1/project.json --num-speakers 2
assamese-agent speakers list outputs/lesson-1/project.json
```

`diarize` writes `diarization.json` (raw turns) and gives every segment a `speaker_id`, a `speaker_confidence` (share of the segment covered by that speaker) and `speaker_flags`:

- `overlap` - another speaker also covers at least 30% of the segment (people talking at once).
- `no-speech-detected` - no turn touched the segment (music, noise, or a miss); the speaker was borrowed from the neighbouring segment.
- `low-confidence` - the chosen speaker covers less than half of the segment.

Detected speakers start with **no voice**. `approve` and `render` refuse until a human assigns one, so a wrong gender or a mis-split speaker cannot reach synthesis unreviewed. If background music confuses detection, run `diarize --isolate-vocals` to separate vocals first (needs the `[separation]` extra) and diarize the vocal stem. `--num-speakers`, `--min-speakers` and `--max-speakers` constrain the model when you know the cast.

Corrections, all of which invalidate any prior approval:

```bash
assamese-agent speakers set   P SPEAKER_00 --name "শিক্ষক" --voice Amit   # name and/or voice; creates if missing
assamese-agent speakers merge P SPEAKER_02 --into SPEAKER_00             # diarizer split one person in two
assamese-agent speakers assign P --segments 3,7-9 --speaker SPEAKER_01   # fix individual segments (0-based)
assamese-agent speakers rename P SPEAKER_01 STUDENT                      # readable id everywhere
assamese-agent speakers list  P --all                                    # every segment, not only flagged
```

`speakers assign` clears the confidence and flags on the segments it touches, because a person decided. Re-running `diarize` on a labelled project requires `--force`; names and voices already given to a label that is detected again are kept. Diarization is a starting point, not ground truth: always walk through the flagged segments before approving. Segments without a `speaker_id` fall back to `render --speaker` (default `Sita`), so existing single-speaker projects keep working unchanged. The renderer instantiates each stock voice at most once and records the per-segment routing in `render-report.json`. Changing the `speakers` map is treated as a content change and requires fresh native review before the next render.

### 3. Review and synthesize

A native Assamese reviewer must check meaning, technical terms, names, numbers and pacing before approval. Every change to transcript or translation requires fresh approval.

```bash
assamese-agent approve outputs/lesson-1/project.json \
  --reviewer 'Your native Assamese reviewer' --native-reviewed
assamese-agent render outputs/lesson-1/project.json --device cuda --speaker Sita
```

Outputs: `assamese.wav` (voice only), `assamese-mix.wav` (voice over ducked background), `assamese.srt`, `assamese-dub.mp4`, per-segment WAVs, `background.wav` / `background-ducked.wav`, and `render-report.json`. By default the subtitle track is switchable in the player, not burnt into the picture; see below to burn it in.

#### Burned-in subtitles for social media and every player

Switchable subtitle tracks are ignored by most social apps and some players. To draw the Assamese captions into the picture:

```bash
assamese-agent render outputs/lesson-1/project.json --burn-subtitles --highlight word
```

This writes `assamese.ass` and `assamese-dub-subtitled.mp4` (no separate soft track, to avoid doubled captions). Rendering uses libass with HarfBuzz, which shapes Bengali-script conjuncts and reordered vowel signs correctly; `drawtext` does not and is deliberately not used.

- **Font.** A font that covers Assamese is required; `render` auto-detects one via fontconfig (Noto Sans Bengali preferred) or you pass `--subtitle-font FAMILY` / `--subtitle-font-file path.ttf` (copied into the project so the job is reproducible). Debian/Ubuntu: `apt install fonts-noto-core`. Note that fontconfig's Bengali (`bn`) coverage does not include the Assamese letters ৰ and ৱ; the report warns when a font declares only `bn`, or nothing, and those letters would show as boxes.
- **Automatic sizing.** Font size is 5% of the shorter edge (54 px on any 1080-short-side video, 108 px on 4K), so landscape 16:9, vertical 9:16 reels, 4:5 and square posts all get the same readable size. Vertical video gets a taller bottom margin to clear app UI. Rotated phone footage is measured after rotation. `--subtitle-scale 1.3` scales the result. Long segments are split into chunks of at most two lines, timed proportionally.
- **Style.** `--subtitle-style outline` (default: white text, black outline and shadow) or `box` (translucent black box). Bottom-centred.
- **Word highlighting.** `--highlight word` colours the word currently being spoken. Timing is **estimated**: words are spread across the synthesized speech length in proportion to their character count. It reads naturally but is not forced alignment; `render-report.json` records `word_timing: estimated`.

`render-report.json` lists the font used, its coverage, the layout kind and the pixel font size.

#### Background audio and ducking

The renderer keeps the original music, effects and room tone under the dub:

```bash
assamese-agent render outputs/lesson-1/project.json --background separate --duck-db 12
```

- `--background separate` (default for real renders): runs Demucs (`htdemucs`, two stems) on the full-quality source audio and keeps the `no_vocals` stem. Requires `pip install -e '.[separation]'`. Separation quality is model-dependent; listen for vocal bleed.
- `--background original`: keeps the full source mix without separation (default for `--test-tone`). The source speech stays audible underneath — useful for plumbing checks and previews, not for publication. `render-report.json` records `source_speech_audible: true` in this mode.
- `--background none`: previous behaviour; the audio track is replaced entirely.
- `--duck-db N`: how far the background is lowered while dubbed speech plays (0-60 dB, default 12). Ducking is sample-accurate from the reviewed segment windows, with a 50 ms attack before each segment and a 300 ms release after it. Gaps shorter than 350 ms stay ducked so quick exchanges do not pump.

Levels are not normalised or limited; if the source is already hot, lower the dub or raise `--duck-db` and re-listen. The input video must remain in its original path and its SHA-256 must match. Listen to the whole result before publishing. A reviewed script is not a reviewed synthesized voice.

The renderer fits speech to the original segment window. If fitting would need more than 1.35x speed, it stops and asks for a shorter translation or new timestamps rather than silently cutting words. Slower speech can leave silence. It does not guarantee lip sync. Failed jobs leave intermediate files for inspection; use a fresh output folder for a separate project.

For an entirely offline smoke test, populate Assamese text manually, approve it, then run:

```bash
assamese-agent render outputs/lesson-1/project.json --test-tone
```

This creates `TEST-TONE-not-assamese.mp4`. Its sound is a synthetic tone, NOT speech, and must never be shown as a working Assamese voice demo.

### Original inspired-video roadmap

```bash
assamese-agent inspired-plan --brief 'Explain water conservation to Class 8 students using new examples from Assam' \
  --out outputs/original-plan.json
```

This creates a planning JSON only. Next: author a new Assamese script and storyboard, review them, select owned/licensed or newly generated visuals, then connect an approved rendering adapter. Do not copy protected dialogue, frames, characters or music from the reference. Inspiration does not supply adaptation rights. Any paid generation provider or external media upload needs a separate explicit decision.

## Architecture and deployment

See `docs/ARCHITECTURE.md` for components and roadmap, `docs/EVALUATION.md` for acceptance tests, and `docs/SOURCES.md` for model/API documentation and license notes.

No credentials or user/student media should enter this repository. Keep the repo private initially. A production platform additionally needs authentication, isolated workers, private object storage, quota/budget limits, retention/deletion controls, consent records, rate limits and an audited queue. Review deployment and distribution licenses first. No hosting, GPU spend or paid account is provisioned by this scaffold.
