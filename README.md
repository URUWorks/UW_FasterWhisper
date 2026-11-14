# UWFaster-Whisper

Standalone [faster-whisper](https://github.com/SYSTRAN/faster-whisper) command line
for [Tero Subtitler](https://github.com/URUWorks/TeroSubtitler), for Windows, macOS
(Apple Silicon and Intel) and Linux.

It's a drop-in replacement of the Purfview's `whisper-faster` that Tero Subtitler used: same
executable name and the same parameters, so Tero Subtitler works with it without changes.

## Usage

```
whisper-faster audio.wav --model medium --model_dir <models> --output_dir <out> --output_format srt
```

- `--model_dir`: folder with a `faster-whisper-<model>` folder for each model
  (`model.bin`, `config.json`, `tokenizer.json`...), as Tero Subtitler downloads them. If the
  model is not there, it's downloaded from Hugging Face.
- `--output_format`: `srt`, `vtt`, `txt`, `tsv`, `json` (openai-whisper format, with
  `--word_timestamps True` the times of each word) or `all`.
- `--sentence`, `--max_line_width N`, `--max_line_count N`, `--max_comma N`: subtitles
  by sentences and line limits (they use the word times).
- `--device auto|cpu|cuda`, `--compute_type auto|int8|float16...`: `auto` uses the
  NVIDIA GPU if the CUDA libraries are available, otherwise the CPU (`int8`). Before
  using the GPU it checks that cuBLAS/cuDNN are there and runs a short test on it; if
  something fails (missing DLLs, an old GPU, not enough memory) it continues on the
  CPU and says why in the log, also if the GPU fails before the first segment.
- `--vad_filter True` and the `--vad_*` options, `--language`, `--task translate`,
  `--beam_size`, `--condition_on_previous_text`, `--initial_prompt`, `--hotwords`,
  `--hallucination_silence_threshold`... `whisper-faster --help` lists all of them.
- `--multilingual True`: for audio that mixes languages. Without it the language is
  detected once, with the first 30 seconds, and everything is transcribed in that language
  (the lines in other languages are lost or come out wrong). With it, the language is
  detected again in each 30 s window. Use it without `--language`.
  `--language_detection_threshold` and `--language_detection_segments` tune the detection.
- Unknown parameters (of other versions) are ignored with a warning.
- `--loop_recovery True` (on by default): when Whisper gets stuck repeating a word or a
  phrase ("no, no, no, no...", the same "Thank you." again and again), the rest of that
  30 s window is usually lost. The looped segments are collapsed to one and that part of
  the audio, up to the next good segment, is decoded again starting a bit later (up to
  three starts are tried, and a result that loops again is not used). If the VAD says
  there's no voice there, it's only collapsed. The log says what was recovered.
- `--retime True` (on by default): a segment that starts long before its words are said
  (after a silence, a line can appear 10 or 20 s too early) is aligned again with its own
  text and the next segment's, only in that window (the rest of the transcription doesn't
  need word times), and moved to where the words are if the alignment is reliable
  (`--retime_threshold`, average word probability, 0.4).
- `--diarize True`: who speaks in each segment. A voice embedding model (3D-Speaker CAM++,
  Apache 2.0, about 28 MB, downloaded once from the sherpa-onnx releases on GitHub to
  `<model_dir>/diarization`) runs with onnxruntime, and the voices are grouped into
  speakers. No Hugging Face token and no new libraries. `--num_speakers N` if it's known,
  `--max_speakers N`, `--diarize_threshold` (0.5, higher = more speakers). The json has
  `"speaker": 1, 2...` in each segment, srt/txt/tsv a `[Speaker 1] ` label
  (`--speaker_format`, `%d` is the number, empty for none) and vtt the `<v Speaker 1>`
  tag. With `--sentence` a new speaker starts a new subtitle. Not in the macOS 10.11 build
  (no onnxruntime). The labels are in the files, not in the lines printed while
  transcribing.
- While transcribing, each segment is printed as `[mm:ss.zzz --> mm:ss.zzz] text`
  (`hh:mm:ss.zzz` after one hour), Tero Subtitler shows them as they come.

The builds read WAV files (any PCM wav, resampled to 16 kHz mono), which is what Tero
Subtitler gives to it. Without PyAV the build is about 130 MB smaller. Running from
the sources with PyAV installed, any audio or video format works.

## GPU (NVIDIA)

CTranslate2 4.x needs CUDA 12 and cuDNN 9. For Windows, `fasterwhisper_cuda_win64.zip`
has those DLLs (from the NVIDIA pip packages), extracted next to `whisper-faster.exe`
(Tero Subtitler asks before downloading it, it's about 1.3 GB). On Linux they are
used from the system when installed. macOS has no CUDA, it runs on the CPU.

## Intel GPUs (OpenVINO)

faster-whisper (CTranslate2) only uses NVIDIA GPUs. `--device openvino` uses
[OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) instead, for Intel GPUs
(Arc, Iris Xe and the integrated ones), NPUs and CPUs: the GPU if there is one, else the
CPU; `openvino:NPU`, `openvino:GPU.1`... to choose. If the GPU fails it continues on the CPU.

- It's the optional `fasterwhisper_openvino_win64.zip` / `fasterwhisper_openvino_linux.zip`
  (about 200 MB, `requirements-openvino.txt`), extracted next to `whisper-faster` (an
  `openvino` folder). Without it, `--device openvino` says so and uses faster-whisper.
- The models are OpenVINO ones, not the faster-whisper ones: `<model_dir>/whisper-<model>-int8-ov`,
  downloaded from `OpenVINO/whisper-<model>-int8-ov` on Hugging Face when they're not
  there (or a model folder with `openvino_encoder_model.xml` in `--model`).
- The audio is given by blocks of about 4 minutes (cut at a silence), so the segments are
  printed while transcribing. `--language`, `--task`, `--beam_size`, `--initial_prompt`,
  `--hotwords`, `--word_timestamps`, `--sentence`, `--loop_recovery`, `--diarize` and the
  output options work as with faster-whisper; `--retime` and the temperature/threshold
  options are faster-whisper only.

## macOS 10.11 to 10.15 (Intel)

`fasterwhisper_macos_legacy_x86_64.zip` is for the Intel Macs that can't run macOS 11
(Mac Pro 2009-2012 and others). It's built apart, with `.github/workflows/macos-legacy.yml`
(Actions > macOS legacy > Run workflow): Python 3.11 of python.org, CTranslate2 built for
10.11 with its own libc++ and Apple Accelerate (`scripts/legacy-macos`), and the last numpy
and tokenizers for those versions. `scripts/legacy-macos/check-macho.sh` stops the build if
something needs a newer macOS. Differences: no `--vad_filter` (onnxruntime has no build for
those versions, it transcribes without it), and on a CPU without AVX2 `auto` uses `float32`,
faster than `int8` there.

## Build

The workflow `.github/workflows/build.yml` builds everything:

- **Actions > Build > Run workflow**: the zips are left as artifacts of the run.
- **A tag `v*`** (for example `git tag v1.0.0 && git push --tags`): they're published
  in a release.

| File | System |
|---|---|
| `fasterwhisper_win64.zip` | Windows x64 |
| `fasterwhisper_cuda_win64.zip` | Windows, NVIDIA GPU libraries (optional) |
| `fasterwhisper_openvino_win64.zip` | Windows, OpenVINO for Intel GPUs (optional) |
| `fasterwhisper_openvino_linux.zip` | Linux, OpenVINO for Intel GPUs (optional) |
| `fasterwhisper_macos_arm64.zip` | macOS 11+, Apple Silicon |
| `fasterwhisper_macos_x86_64.zip` | macOS 11+, Intel |
| `fasterwhisper_macos_legacy_x86_64.zip` | macOS 10.11 to 10.15, Intel |
| `fasterwhisper_linux.zip` | Linux x64 (glibc 2.35+, Ubuntu 22.04 and later) |

Tero Subtitler downloads them from `releases/latest/download/<file>`, so a new release is used
without changing Tero Subtitler.

Locally: `pip install -r requirements.txt -r requirements-build.txt` and
`pyinstaller --noconfirm --clean whisper-faster.spec` (the result is in
`dist/whisper-faster`).

Updates are automatic: on day 1 of each month the workflow looks for new versions of
faster-whisper, CTranslate2 and the rest (`tools/update_pins.py`). If there are, it
builds all the systems and tests a real transcription on each one, and only then
commits the new pins, tags `vX.Y.0` and publishes the release. If nothing is new, or a
build or test fails, nothing changes (GitHub emails the failure). It can also be run by
hand: Actions > Build > Run workflow > "update".

To change the versions by hand: edit `requirements.txt` and `__version__` in
`whisper_faster.py`, and push a new tag.

## License

MPL 2.0, as Tero Subtitler. faster-whisper and CTranslate2 are MIT, the Whisper models
are MIT (OpenAI). OpenVINO and OpenVINO GenAI are Apache 2.0. The speaker model
(3D-Speaker CAM++, VoxCeleb, from the sherpa-onnx releases) is Apache 2.0.

Tests without a model: `python -m unittest discover -s tests`.
