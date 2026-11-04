# UW_FasterWhisper

Standalone [faster-whisper](https://github.com/SYSTRAN/faster-whisper) command line
for [Tero Subtitler](https://github.com/URUWorks/TeroSubtitler), for Windows, macOS
(Apple Silicon and Intel) and Linux.

It uses the `whisper-faster` executable name and command line parameters, so Tero Subtitler
works with it without changes.

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
  NVIDIA GPU if the CUDA libraries are available, otherwise the CPU (`int8`). If the GPU
  fails it continues on the CPU.
- `--vad_filter True` and the `--vad_*` options, `--language`, `--task translate`,
  `--beam_size`, `--condition_on_previous_text`, `--initial_prompt`, `--hotwords`,
  `--hallucination_silence_threshold`... `whisper-faster --help` lists all of them.
- Unknown parameters (of other versions) are ignored with a warning.
- While transcribing, each segment is printed as `[mm:ss.zzz --> mm:ss.zzz] text`
  (`hh:mm:ss.zzz` after one hour), Tero Subtitler shows them as they come.

The builds read WAV files (any PCM wav, resampled to 16 kHz mono), which is what Tero
Subtitler gives to it. Without PyAV the build is about 130 MB smaller. Running from the sources
with PyAV installed, any audio or video format works.

## GPU (NVIDIA)

CTranslate2 4.x needs CUDA 12 and cuDNN 9. For Windows, `fasterwhisper_cuda_win64.zip`
has those DLLs (from the NVIDIA pip packages), extracted next to `whisper-faster.exe`
(Tero Subtitler asks before downloading it, it's about 1.3 GB). On Linux they are used from the
system when installed. macOS has no CUDA, it runs on the CPU.

## Build

The workflow `.github/workflows/build.yml` builds everything:

- **Actions > Build > Run workflow**: the zips are left as artifacts of the run.
- **A tag `v*`** (for example `git tag v1.0.0 && git push --tags`): they're published
  in a release.

| File | System |
|---|---|
| `fasterwhisper_win64.zip` | Windows x64 |
| `fasterwhisper_cuda_win64.zip` | Windows, NVIDIA GPU libraries (optional) |
| `fasterwhisper_macos_arm64.zip` | macOS 11+, Apple Silicon |
| `fasterwhisper_macos_x86_64.zip` | macOS 11+, Intel |
| `fasterwhisper_linux.zip` | Linux x64 (glibc 2.35+, Ubuntu 22.04 and later) |

Tero Subtitler downloads them from `releases/latest/download/<file>`, so a new release
is used without changing Tero Subtitler.

Locally: `pip install -r requirements.txt -r requirements-build.txt` and
`pyinstaller --noconfirm --clean whisper-faster.spec` (the result is in
`dist/whisper-faster`).

Each build is tested with a real transcription (the tiny model, a wav, with and
without VAD), so a library that doesn't work stops the release.

## Updates

The workflow `.github/workflows/update.yml` runs on the 1st of each month: it looks
for new versions of the libraries of `requirements.txt` on PyPI and, if there are
any, builds and tests everything with them. Only if all the builds work, it publishes
a new release (`v1.0.1`, `v1.0.2`...) and updates `requirements.txt` and the version
in `main`. Without new versions it does nothing.

- The macOS lines stay as they are (the newer onnxruntime/numpy need a recent macOS).
- CTranslate2 stays in its major version (4.x, the CUDA 12 / cuDNN 9 of the GPU zip).
  A new major version has to be updated by hand, with the NVIDIA libraries of
  `build.yml`.
- **Actions > Update > Run workflow** runs it now. With *dry run* it builds the new
  versions without publishing anything.

To update by hand: change the versions in `requirements.txt` and `__version__` in
`whisper_faster.py`, and make a new tag.

## License

MPL 2.0, as Tero Subtitler. faster-whisper and CTranslate2 are MIT, the Whisper models
are MIT (OpenAI).
