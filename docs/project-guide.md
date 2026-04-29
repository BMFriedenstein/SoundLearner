# Project Guide

This page holds the more detailed project-operating notes that used to make the home README too dense.

## Dataset Builder

`dataset_builder` creates labelled synthetic examples from the oscillator instrument model.

```bash
./build/dataset_builder/dataset_builder
```

Useful options:

```text
-n --num-samples <count>       number of examples to generate
-t --sample-time <seconds>     generated clip length
--min-instrument-size <count>  minimum coupled oscillator count per sample
--max-instrument-size <count>  maximum coupled oscillator count per sample
--min-uncoupled-oscilators     minimum uncoupled oscillator count per sample
--max-uncoupled-oscilators     maximum uncoupled oscillator count per sample
--note-frequency <hz>          fixed generated base note frequency
--min-note-frequency <hz>      minimum generated base note frequency
--max-note-frequency <hz>      maximum generated base note frequency
--frequency-factor <0..1>      fixed normalized oscillator frequency factor
--min-frequency-factor <0..1>  minimum normalized oscillator frequency factor
--max-frequency-factor <0..1>  maximum normalized oscillator frequency factor
```

Old fixed-count flags still work:

```text
-s --instrument-size <count>
-c --uncoupled-oscilators <count>
```

but they now mean "use the same count for every generated sample". For more realistic datasets, prefer per-sample ranges:

```bash
./build/dataset_builder/dataset_builder -n 1000 -t 5 --min-instrument-size 8 --max-instrument-size 64 --min-uncoupled-oscilators 0 --max-uncoupled-oscilators 0 --min-note-frequency 55 --max-note-frequency 440
```

Outputs include:

```text
dataN.wav                      generated audio
dataN.data                     legacy oscillator data
dataN.meta                     legacy metadata
```

Feature tensors, metadata JSON, and previews are prepared afterward by the Python pipeline.

Prepared outputs include:

```text
features/dataN.slft            canonical feature tensor
metadata/dataN.json            structured metadata
preview/dataN_rgb.bmp          feature preview
preview/dataN_logfreq_rgb.bmp  feature preview alias
mel_preview/dataN_mel.png      mel spectrogram preview
```

The current oscillator generator no longer samples completely free frequency factors. It now renders oscillators against a deliberately small octave anchor ladder with a small detune window around each anchor:

```text
0.5*f0, 1*f0, 2*f0, 4*f0, 8*f0, 16*f0, 32*f0
```

This is intentionally simple while f0 prediction is being debugged. The curriculum build scripts also randomize the generated base note over `55..440 Hz`, so the model cannot learn a constant f0 shortcut.

The adaptive curriculum's first grade fixes the normalized frequency factor at `0.21428571428571427`, which is the center of the `1.0*f0` anchor. Without this, a one-oscillator sound is still pitch-ambiguous because the same tone can be represented by different base-note and octave-multiplier pairs.
Early adaptive grades can also pass an explicit coupled factor sequence, such as `0.21428571428571427,0.35714285714285715,0.5`, to build fixed harmonic ladders before reintroducing variable harmonic counts.

## Dataset Preparation

`deep_trainer.prepare_dataset` converts dataset WAV files into the `.slft` tensors and previews used by training and evaluation.

```bash
python -m deep_trainer.prepare_dataset --dataset-root . --freq-bins 1024 --time-frames 512 --crop-seconds 5
```

Useful options:

```text
--dataset-root <path>          dataset folder containing dataN.wav files
--resolution <pixels>          square output tensor resolution, default unset
--freq-bins <pixels>           frequency bins, overrides square resolution
--time-frames <pixels>         time frames, overrides square resolution
-t --crop-seconds <seconds>    fixed analysis window, default 5
--crop-start-seconds <seconds> crop offset, default 0
--skip-previews                skip BMP feature previews
--skip-mel-previews            skip mel PNG previews
```

The Python feature pipeline uses fixed-window cropping. Long WAV files are cropped. Short WAV files are zero-padded. This avoids stretching the time axis, which would hide the true temporal scale of the sound.

`--resolution` is square shorthand. For higher frequency detail without exploding memory, prefer rectangular tensors:

```bash
python -m deep_trainer.prepare_dataset --dataset-root . --freq-bins 2048 --time-frames 512 --crop-seconds 5
```

## Dataset Augmentation

The Python-side augmentor builds alternate curricula of synthetic datasets that keep the same oscillator labels but pass the audio through a more recording-like world before feature extraction.

```bash
python -m deep_trainer.dataset_augmentor --input-root datasets/synth_1024x512_1k --output-root datasets/synth_1024x512_1k_realish --variants-per-input 2
```

Use it to experiment with:

- gain variation
- noise floor
- tone tilt
- tiny room coloration
- mild saturation

The augmentor also writes mel spectrogram PNGs under `mel_preview/`.

## Complexity Curriculum

The progressive complexity curriculum is intended for stage-to-stage fine-tuning rather than independent scratch runs.

Stage ladder:

```text
c1: coupled 1..3,  uncoupled 0
c2: coupled 1..5,  uncoupled 0
c3: coupled 1..8,  uncoupled 0
c4: coupled 1..12, uncoupled 0
c5: coupled 1..20, uncoupled 0
c6: coupled 1..32, uncoupled 0
c7: coupled 1..64, uncoupled 0
```

Uncoupled oscillators are parked for now. Add them back later as a separate branch once coupled harmonic recovery is reliable.

Build it:

```bat
set DATASET_WORKERS=8
scripts\build_complexity_curriculum_v1.bat
```

Train progressively through the ladder:

```bat
scripts\train_complexity_curriculum_v1.bat
```

Or stop at an intermediate stage:

```bat
scripts\train_complexity_curriculum_v1.bat c4
```

The complexity curriculum configs default to `width = 128`, `batch_size = 8`, and `20` epochs per stage.

Dataset orchestration now lives in [../scripts/build_datasets.py](../scripts/build_datasets.py), with the batch files acting as thin wrappers.

The build path supports process-level parallel dataset generation through the `DATASET_WORKERS` environment variable. It now merges raw audio/labels first and then prepares Python-side feature tensors and previews afterward.

## Adaptive Curriculum

The adaptive curriculum is the preferred next experiment. It builds one generated grade at a time, trains in short chunks, reads validation metrics, and only promotes to the next grade when the configured thresholds pass.

```powershell
scripts\run_adaptive_curriculum_v1.bat --dry-run --skip-native-build --stop-grade g01_single_oscillator
scripts\run_adaptive_curriculum_v1.bat
```

The first grade is a deliberately simple one-oscillator dataset. Later grades increase coupled oscillator count and harmonic range while keeping uncoupled oscillators disabled. This is meant to avoid guessing how fast the model should progress.

Details live in [adaptive-curriculum.md](./adaptive-curriculum.md).

## Feature Tensor

The canonical model input is a compact binary `.slft` tensor:

```text
magic:        SLFT
version:      1
sample_rate:  44100
shape:        channels x frequency_bins x time_frames
data:         float32, channel-major
```

Current channels:

1. Log-frequency magnitude.
2. Temporal delta.
3. Onset emphasis.

Preview images are for humans. The ML model should train on `.slft` tensors, not on BMP or PPM files.

## ML Direction

The current direction is a modern supervised PyTorch baseline first, then richer probabilistic or generative approaches only after the failure modes are clear.

Recommended shape:

```text
audio feature tensor
  -> modern audio/image encoder
  -> latent embedding
  -> structured oscillator-parameter heads
  -> optional differentiable or offline resynthesis loss
```

Current priorities:

1. Train a strong encoder-regressor baseline on `.slft` tensors.
2. Evaluate by resynthesizing predicted parameters back to audio.
3. Use real non-generated WAVs as the real benchmark.
4. Add uncertainty or mixture-style outputs only if direct regression keeps collapsing to bland averages.

Practical target for a 12GB GPU:

```text
input:        3 x 512 x 512 or 3 x 768 x 768
precision:    mixed precision
batch size:   tune around memory, likely 4-32 depending on encoder
model:        small-to-medium ConvNet or small AST
optimizer:    AdamW
training:     supervised synthetic labels first
validation:   held-out synthetic plus curated real audio benchmark
```

For oscillator recovery, rectangular inputs are often more useful than very large square inputs:

```text
3 x 512 x 512
3 x 1024 x 512
3 x 2048 x 512
3 x 4096 x 512
```

## Proposed ML Milestones

1. Keep `.slft` as the stable model-facing artifact.
2. Build dataset manifests so training and evaluation are explicit rather than directory-driven.
3. Train with the adaptive curriculum from a one-oscillator inverse problem upward.
4. Measure progress on real non-generated holdout audio after promoted grades, not just synthetic validation loss.
5. Improve resynthesis quality with stronger audio-space losses and evaluation metrics.
6. Widen synthetic realism carefully so the generator teaches useful robustness without breaking label meaning.
7. Add uncertainty or mixture-style outputs if direct regression continues to average plausible solutions together.
8. Revisit diffusion only after the simpler probabilistic path has clearly hit a limit.
