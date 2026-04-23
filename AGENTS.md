# Codex Notes

This file is for future LLM/Codex sessions working on SoundLearner.

## Project Intent

SoundLearner is trying to learn a compact oscillator/synthesizer representation from audio. Generated oscillator audio provides labelled training data. Real, non-generated WAV files should be used as the real performance benchmark.

After training works, the first serious playback target is a Raspberry Pi app for digital piano/MIDI-controller input and realtime audio output. A modern VST-style DAW plugin is also planned, but treat that as long-term work because it will require plugin SDK integration, realtime-safe DSP rules, host automation, packaging, UI, and cross-platform build decisions.

The important architectural direction is:

```text
WAV
  -> fixed crop/pad window
  -> .slft audio feature tensor
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
./build/dataset_builder/dataset_builder -n 10 -t 5 -r 512
```

Variable oscillator counts per sample:

```bash
./build/dataset_builder/dataset_builder -n 1000 -t 5 --min-instrument-size 8 --max-instrument-size 64 --min-uncoupled-oscilators 0 --max-uncoupled-oscilators 12 --freq-bins 1024 --time-frames 512
```

`-s/--instrument-size` and `-c/--uncoupled-oscilators` still work as fixed-count shortcuts. Prefer the min/max flags when you want the generator to cover a broader instrument space.

Feature extractor:

```bash
./build/feature_extractor/feature_extractor -i input.wav -o output.slft -r 512 -t 5
```

Feature extractor behavior:

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

Dataset augmentor:

```bash
python -m deep_trainer.dataset_augmentor --input-root datasets/synth_1024x512_1k --output-root datasets/synth_1024x512_1k_realish --variants-per-input 2 --tool-mode wsl
```

This keeps oscillator labels untouched while augmenting the rendered audio before feature extraction. Use it to create a more recording-like sibling dataset rather than baking research-heavy augmentation logic into the C++ generator.
The augmentor now also writes mel spectrogram PNG previews under `mel_preview/`.

Prediction:

```bash
python -m deep_trainer.predict --checkpoint runs/baseline/best.pt --feature features/data0.slft --output prediction.data
```

Evaluation harness:

```bash
python -m deep_trainer.evaluate --checkpoint runs/baseline_256_1k/best.pt --input sounds/o_a2_1.wav --output-dir sounds/eval/o_a2_1_eval --resolution 256 --device cpu
```

Use `--device cpu` if a long GPU training run is active. The harness calls the WSL-built C++ feature extractor/player by default.
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

Parameter-space collapse analysis:

```bash
python -m deep_trainer.analyze_parameter_space --dataset-root datasets/synth_1024x512_1k --evaluation-root sounds/eval/baseline_1024x512_1k_b4_e40_10 --output-dir sounds/eval/baseline_1024x512_1k_b4_e40_10/analysis
```

This compares true synthetic instrument vectors against predicted `.data` files using PCA/SVD, pairwise distances, and per-parameter variance. Use it when predictions all sound suspiciously similar.

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
