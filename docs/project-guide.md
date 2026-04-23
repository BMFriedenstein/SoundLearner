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
-r --resolution <pixels>       feature and preview resolution
--min-instrument-size <count>  minimum coupled oscillator count per sample
--max-instrument-size <count>  maximum coupled oscillator count per sample
--min-uncoupled-oscilators     minimum uncoupled oscillator count per sample
--max-uncoupled-oscilators     maximum uncoupled oscillator count per sample
--write-ppm-preview            also write PPM preview images
```

Old fixed-count flags still work:

```text
-s --instrument-size <count>
-c --uncoupled-oscilators <count>
```

but they now mean "use the same count for every generated sample". For more realistic datasets, prefer per-sample ranges:

```bash
./build/dataset_builder/dataset_builder -n 1000 -t 5 --min-instrument-size 8 --max-instrument-size 64 --min-uncoupled-oscilators 0 --max-uncoupled-oscilators 12 --freq-bins 1024 --time-frames 512
```

Outputs include:

```text
dataN.wav                      generated audio
dataN.data                     legacy oscillator data
dataN.meta                     legacy metadata
features/dataN.slft            canonical feature tensor
metadata/dataN.json            structured metadata
preview/dataN_rgb.bmp          linear-frequency RGB preview
preview/dataN_logfreq_rgb.bmp  log-frequency RGB preview
```

By default, preview images are BMP RGB files. PPM files are only written when requested.

## Feature Extractor

`feature_extractor` converts a WAV file into the same feature tensor format used by the generated dataset.

```bash
./build/feature_extractor/feature_extractor -i input.wav -o output.slft
```

Useful options:

```text
-i --input <input.wav>         source WAV file
-o --output <output.slft>      feature tensor output
-r --resolution <pixels>       output tensor resolution, default 512
--freq-bins <pixels>           frequency bins, overrides square resolution
--time-frames <pixels>         time frames, overrides square resolution
-p --preview-prefix <path>     optional preview image prefix
-t --crop-seconds <seconds>    fixed analysis window, default 5
--crop-start-seconds <seconds> crop offset, default 0
--write-ppm-preview            also write a PPM preview
```

The extractor uses fixed-window cropping. Long WAV files are cropped. Short WAV files are zero-padded. This avoids stretching the time axis, which would hide the true temporal scale of the sound.

`--resolution` is square shorthand. For higher frequency detail without exploding memory, prefer rectangular tensors:

```bash
./build/feature_extractor/feature_extractor -i input.wav -o output.slft --freq-bins 2048 --time-frames 512 -p preview/input
```

## Dataset Augmentation

The Python-side augmentor builds alternate curricula of synthetic datasets that keep the same oscillator labels but pass the audio through a more recording-like world before feature extraction.

```bash
python -m deep_trainer.dataset_augmentor --input-root datasets/synth_1024x512_1k --output-root datasets/synth_1024x512_1k_realish --variants-per-input 2 --tool-mode wsl
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
c4: coupled 1..12, uncoupled 0..2
c5: coupled 1..20, uncoupled 0..4
c6: coupled 1..32, uncoupled 0..8
c7: coupled 1..64, uncoupled 0..12
```

Build it:

```bat
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
3. Train the complexity curriculum progressively from simple to dense oscillator mixtures.
4. Measure progress on real non-generated holdout audio, not just synthetic validation loss.
5. Improve resynthesis quality with stronger audio-space losses and evaluation metrics.
6. Widen synthetic realism carefully so the generator teaches useful robustness without breaking label meaning.
7. Add uncertainty or mixture-style outputs if direct regression continues to average plausible solutions together.
8. Revisit diffusion only after the simpler probabilistic path has clearly hit a limit.
