# Codex Notes

This file is for future LLM/Codex sessions working on SoundLearner.

## Project Intent

SoundLearner is trying to learn a compact oscillator/synthesizer representation from audio. Generated oscillator audio provides labelled training data. Real, non-generated WAV files should be used as the real performance benchmark.

After training works, the first serious playback target is a Raspberry Pi app for digital piano/MIDI-controller input and realtime audio output. A modern VST-style DAW plugin is also planned, but treat that as long-term work because it will require plugin SDK integration, realtime-safe DSP rules, host automation, packaging, UI, and cross-platform build decisions.

The important architectural direction is:

```text
WAV
  -> fixed crop/pad window
  -> Python-generated .slft audio feature tensor
  -> modern audio encoder
  -> structured oscillator parameter heads
  -> render predicted instrument
  -> evaluate in audio space
```

Preview images are for human inspection. Do not build the ML pipeline around BMP or PPM files.

## Build Commands

Default GCC build:

```bash
meson setup build -Dcpp_std=c++23
meson compile -C build
```

Reconfigure an existing build:

```bash
meson setup build --reconfigure -Dcpp_std=c++23
meson compile -C build
```

Strict warning check with the available compiler:

```bash
meson setup build-warnings --wipe -Dcpp_std=c++23 -Dwerror=true
meson compile -C build-warnings
```

Clang build, once `clang++` is installed:

```bash
meson setup build-clang --native-file meson/clang.ini
meson compile -C build-clang
```

The current Clang native file uses `clang++-18`, matching the WSL toolchain available during this cleanup pass.

`clangd` reads `build/compile_commands.json` by default through `.clangd`.

## Main Executables

Dataset builder:

```bash
./build/dataset_builder/dataset_builder -n 10 -t 5
```

Variable oscillator counts per sample:

```bash
./build/dataset_builder/dataset_builder -n 1000 -t 5 --min-instrument-size 8 --max-instrument-size 64 --min-uncoupled-oscilators 0 --max-uncoupled-oscilators 12
```

F0-randomized generation:

```bash
./build/dataset_builder/dataset_builder -n 1000 -t 5 --min-instrument-size 1 --max-instrument-size 8 --min-note-frequency 55 --max-note-frequency 440
```

`-s/--instrument-size` and `-c/--uncoupled-oscilators` still work as fixed-count shortcuts. Prefer the min/max flags when you want the generator to cover a broader instrument space.

Dataset preparation / feature extraction:

```bash
python -m deep_trainer.prepare_dataset --dataset-root . --freq-bins 512 --time-frames 512 --crop-seconds 5
```

Python feature prep behavior:

- Long WAV files are cropped.
- Short WAV files are zero-padded.
- Default crop is 5 seconds from the start.
- Use `--crop-start-seconds` for later windows.
- `--resolution` is square shorthand; use `--freq-bins` and `--time-frames` for rectangular high-frequency tensors.

Python trainer:

```bash
python -m deep_trainer.train --dataset-root . --epochs 50 --batch-size 8 --resolution 512 --amp
```

Trainer config file:

```bash
python -m deep_trainer.train --config deep_trainer/configs/curriculum_v1_clean.toml
```

`train.py` now supports TOML or JSON configs. CLI flags override config-file values, which is handy for one-off width/output-dir changes.
It also supports:

```bash
python -m deep_trainer.train --resume runs/some_run/last.pt
python -m deep_trainer.train --init-checkpoint runs/previous_stage/best.pt
```

Use `--resume` to continue the same run. Use `--init-checkpoint` for curriculum fine-tuning into a new run directory.

Audio-aligned render loss:

```bash
python -m deep_trainer.train --dataset-root datasets/my_dataset --freq-bins 1024 --time-frames 512 --render-loss-weight 1.0 --render-loss-seconds 1.0 --render-loss-sample-rate 11025
```

This keeps the old activity/parameter supervision but also adds a differentiable surrogate render + feature reconstruction loss so training is no longer graded only on oscillator-table similarity.

Current `renderloss` implementation:

```text
predicted oscillator parameters
  -> predicted f0
  -> differentiable surrogate oscillator bank
  -> surrogate waveform
  -> multi-resolution differentiable 3-channel feature extraction
  -> weighted feature reconstruction loss
  -> RMS penalty against source WAV RMS
```

It is currently a first surrogate, not a faithful differentiable clone of the C++ player. If `val_render` barely moves, assume the signal may be too weak or too forgiving before assuming the code path is broken.

The trainer also has an f0 head and a frequency-crowding penalty. The f0 head predicts log-normalized Hz and renderloss uses the predicted f0 when synthesizing the surrogate waveform. Track `val_f0_cents` for readable pitch error. Activity BCE is dynamically class-balanced by default because active oscillator slots are sparse. The crowding penalty discourages active oscillator slots from collapsing to the same frequency factor, but it detaches activity so the penalty cannot reward silence. RMS render loss is log-scaled so zero-energy predictions are punished more strongly.

Dataset augmentor:

```bash
python -m deep_trainer.dataset_augmentor --input-root datasets/synth_1024x512_1k --output-root datasets/synth_1024x512_1k_realish --variants-per-input 2
```

This keeps oscillator labels untouched while augmenting the rendered audio before feature extraction. Use it to create a more recording-like sibling dataset rather than baking research-heavy augmentation logic into the C++ generator.
The augmentor now also writes mel spectrogram PNG previews under `mel_preview/`.

The current oscillator renderer uses a small octave frequency prior: `0.5, 1, 2, 4, 8, 16, 32` times f0, with a small detune window. New curriculum datasets randomize f0 over `55..440 Hz` by default.

Prediction:

```bash
python -m deep_trainer.predict --checkpoint runs/baseline/best.pt --feature features/data0.slft --output prediction.data
```

Evaluation harness:

```bash
python -m deep_trainer.evaluate --checkpoint runs/baseline_256_1k/best.pt --input sounds/o_a2_1.wav --output-dir sounds/eval/o_a2_1_eval --resolution 256 --device cpu
```

Use `--device cpu` if a long GPU training run is active. The harness now extracts features in Python and only calls the C++ player for rendering.
Evaluation runs now also write `ab_listen/` WAV pairs and mel spectrogram PNGs for original/predicted/A-B comparison.

Curriculum scripts:

```bat
scripts\build_curriculum_v1.bat
scripts\train_curriculum_v1.bat
scripts\build_complexity_curriculum_v1.bat
scripts\train_complexity_curriculum_v1.bat
```

Complexity curriculum:

```text
c1: coupled 1..3,  uncoupled 0
c2: coupled 1..5,  uncoupled 0
c3: coupled 1..8,  uncoupled 0
c4: coupled 1..12, uncoupled 0..2
c5: coupled 1..20, uncoupled 0..4
c6: coupled 1..32, uncoupled 0..8
c7: coupled 1..64, uncoupled 0..12
```

These stages are meant to be trained progressively, initializing each stage from the previous stage's `best.pt`.

Training wrappers support explicit objective modes:

```bat
scripts\train_complexity_curriculum_v1.bat c3 parameter
scripts\train_complexity_curriculum_v1.bat c3 renderloss
scripts\train_complexity_curriculum_v1.bat c3 renderloss_light
```

Parameter-space collapse analysis:

```bash
python -m deep_trainer.analyze_parameter_space --dataset-root datasets/synth_1024x512_1k --evaluation-root sounds/eval/baseline_1024x512_1k_b4_e40_10 --output-dir sounds/eval/baseline_1024x512_1k_b4_e40_10/analysis
```

This compares true synthetic instrument vectors against predicted `.data` files using PCA/SVD, pairwise distances, and per-parameter variance. Use it when predictions all sound suspiciously similar.

Local analysis frontend:

```powershell
.\.venv\Scripts\python.exe -m deep_trainer.analysis_frontend
```

Open `http://127.0.0.1:8765`. The frontend lets the user drop a WAV, choose a checkpoint and PCA reference dataset, hear source vs predicted render, and inspect mel previews, SLFT previews, metrics, and the prediction's 2D PCA placement. It can render with model f0, a classical audio pitch estimate, or a manual note to isolate pitch failure from timbre failure. Runs are saved under `analysis_frontend/runs/`.

## Current Data Formats

Canonical ML input:

```text
features/dataN.slft
```

Human preview outputs:

```text
preview/dataN_rgb.bmp
preview/dataN_logfreq_rgb.bmp
```

Legacy/generated support files:

```text
dataN.wav
dataN.data
dataN.meta
metadata/dataN.json
```

## Coding Guidelines

- Keep C++ at C++23.
- The ML implementation is PyTorch-based under `deep_trainer`.
- Prefer small `.cpp` implementations over large header-only code.
- Keep the `.slft` tensor as the model-facing artifact.
- Keep previews optional and human-facing.
- Avoid changing legacy Python trainer behavior unless the user asks for ML implementation work.
- Use Meson targets rather than ad hoc compile commands.
- Run `meson compile -C build` before finishing C++ changes.
- For warning work, run the strict warning build if it is available.

## ML Guidance

The README currently recommends a supervised encoder-regressor baseline before diffusion:

- The current implementation is a small ConvNeXt-style PyTorch encoder.
- Structured parameter heads instead of one unstructured flat output.
- Add uncertainty or mixture-density heads if the inverse mapping is ambiguous.
- Consider conditional diffusion only after the baseline shows where ambiguity hurts.

Real audio evaluation should compare resynthesized output against the source audio using spectral and perceptual metrics, because real piano/bird/etc. clips do not have ground-truth oscillator labels.

## Playback Roadmap

- First post-training target: Raspberry Pi app for model playback from digital piano MIDI input.
- The Pi player should prioritize low latency, stable realtime audio, model loading, MIDI handling, and practical deployment.
- Long-term target: DAW plugin as a modern VST-style shared library.
- The plugin work should be planned separately from the core training milestone because it is a large product/platform project rather than just another executable.
