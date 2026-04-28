# SoundLearner Trainer

This directory contains the modern ML baseline for SoundLearner.

The legacy TensorFlow 1.x image trainer has been replaced with a PyTorch pipeline that trains directly on `.slft` feature tensors. The current goal is a supervised baseline: map audio features to oscillator-machine parameters, prove the end-to-end path, then improve the model once the failure modes are visible.

## Design Goal

SoundLearner is an inverse synthesis project:

```text
audio clip -> compact oscillator instrument -> rendered audio
```

The trainer does not try to generate waveform samples directly. It predicts the parameters of the existing oscillator model. That keeps the output small, inspectable, and usable by the C++ player.

## Architecture Overview

```mermaid
flowchart LR
    A["dataset_builder"] --> B["WAV audio"]
    A --> C["oscillator CSV labels"]
    B --> D["Python feature prep"]
    D --> E["SLFT feature tensor"]
    C --> F["fixed oscillator slots"]
    D --> G["metadata JSON"]
    E --> H["PyTorch Dataset"]
    F --> H
    G --> H
    H --> I["ConvNeXt-style encoder"]
    I --> J["activity head"]
    I --> K["parameter head"]
    J --> L["active oscillator probabilities"]
    K --> M["oscillator parameter values"]
    L --> N["CSV prediction"]
    M --> N
    N --> O["future C++ render/playback path"]
```

Generated audio gives us perfect labels. Real audio does not, so real clips should be treated as a performance benchmark after the synthetic baseline works.

## Files

```text
deep_trainer/
  slft.py                 binary .slft reader
  dataset.py              dataset discovery and label loading
  model.py                ConvNeXt-style baseline model
  losses.py               activity + parameter loss
  train.py                training CLI
  predict.py              prediction CLI
  evaluate.py             holdout evaluation harness
  audio_features.py       Python feature extraction and SLFT writing
  prepare_dataset.py      dataset WAV -> SLFT/metadata/preview preparation
  dataset_augmentor.py    audio-domain augmentation for synthetic datasets
  analyze_parameter_space.py
                          PCA/SVD collapse analysis for predicted .data files
  cnn_model_trainer.py    compatibility wrapper for old script name
  model_predictor.py      compatibility wrapper for old script name
  requirements.txt        Python dependencies
```

## Data Layout

The trainer expects data prepared from `dataset_builder` output:

```text
dataset-root/
  features/
    data0.slft
    data1.slft
  metadata/
    data0.json
    data1.json
  data0.data
  data1.data
```

`metadata/dataN.json` is preferred because it explicitly links the feature tensor to the oscillator CSV target. If metadata is missing, the dataset loader falls back to matching:

```text
features/dataN.slft -> dataN.data
```

The lower-level discovery rules live in [dataset-discovery.md](./dataset-discovery.md).

Use the Python preparation step to build those tensors:

```bash
python -m deep_trainer.prepare_dataset --dataset-root datasets/my_dataset --freq-bins 1024 --time-frames 512 --crop-seconds 5
```

## SLFT Tensor

The `.slft` format is the canonical ML input:

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

For a default 512-resolution dataset, one model input has shape:

```text
3 x 512 x 512
```

Rectangular tensors are supported:

```text
3 x 1024 x 512
3 x 2048 x 512
3 x 4096 x 512
```

Preview BMP/PPM images are only for humans. The trainer should not depend on image files.

## Label Format

Each instrument is encoded as a fixed number of oscillator slots. The default is `64` slots.

Each slot has 8 values:

```text
active
start_amplitude_factor
start_frequency_factor
phase_factor
amplitude_decay_factor
amplitude_attack_factor
frequency_decay_factor
base_frequency_coupled
```

The `active` flag lets the model represent variable oscillator counts while keeping a fixed output shape.

```mermaid
flowchart LR
    A["dataN.data CSV rows"] --> B["parse up to max_oscillators"]
    B --> C["active = 1 for present rows"]
    B --> D["active = 0 for padded rows"]
    C --> E["target tensor: max_oscillators x 8"]
    D --> E
    E --> F["mask tensor: max_oscillators"]
```

Rows beyond `--max-oscillators` are ignored.

## Model

The current model is a small ConvNeXt-style encoder, implemented in `model.py`.

```mermaid
flowchart TD
    A["SLFT tensor<br/>B x C x F x T"] --> B["stem<br/>4x stride convolution"]
    B --> C["ConvNeXt block"]
    C --> D["downsample stage<br/>width x 2"]
    D --> E["downsample stage<br/>width x 4"]
    E --> F["downsample stage<br/>width x 8"]
    F --> G["adaptive average pool"]
    G --> H["shared latent embedding"]
    H --> I["activity head<br/>B x max_oscillators"]
    H --> J["parameter head<br/>B x max_oscillators x 7"]
```

The two-head output matters:

1. The activity head predicts which oscillator slots are present.
2. The parameter head predicts values for active oscillator slots.

All parameter outputs are passed through `sigmoid`, because the current oscillator CSV values are normalized to `0..1`.

## Loss

```text
total_loss = activity_weight * BCE(activity_logits, active_flags)
           + parameter_weight * masked_smooth_l1(parameters, target_parameters)
```

Default weights:

```text
activity_weight = 1.0
parameter_weight = 10.0
```

The parameter loss is masked so padded inactive slots do not dominate training.

## Training

Install dependencies:

```bash
python -m pip install -r deep_trainer/requirements.txt
```

Smoke train:

```bash
python -m deep_trainer.train --dataset-root . --epochs 2 --batch-size 4 --resolution 256 --amp --output-dir runs/smoke
```

Baseline train:

```bash
python -m deep_trainer.train --dataset-root datasets/synth_256_1k --epochs 30 --batch-size 8 --resolution 256 --amp --output-dir runs/baseline_256_1k
```

Rectangular train:

```bash
python -m deep_trainer.train --dataset-root datasets/synth_2048x512_1k --epochs 30 --batch-size 2 --freq-bins 2048 --time-frames 512 --amp --output-dir runs/baseline_2048x512_1k
```

Useful options:

```text
--config configs/run.toml
--max-oscillators 64
--learning-rate 3e-4
--weight-decay 1e-4
--validation-split 0.15
--width 64
--dropout 0.1
--num-workers 0
--seed 1337
--amp
--tensorboard
--resume runs/previous/last.pt
--init-checkpoint runs/previous/best.pt
--render-loss-weight 1.0
--render-loss-sample-rate 11025
--render-loss-seconds 1.0
```

### Config Files

Training parameters can live in a TOML or JSON file instead of a long shell command.

Example:

```bash
python -m deep_trainer.train --config deep_trainer/configs/curriculum_v1_clean.toml
```

CLI flags still override config-file values.

### Checkpoint Continuation

`train.py` supports:

```text
--resume            continue the same run, including optimizer state and epoch count
--init-checkpoint   load model weights only, then start a fresh run
```

Use `--resume` when the dataset and run are the same. Use `--init-checkpoint` for curriculum fine-tuning.

### Audio-Aligned Render Loss

The trainer can now add a differentiable surrogate render loss on top of the existing activity and parameter losses.

This path:

```text
predicted oscillator parameters
  -> differentiable surrogate oscillator bank
  -> multi-resolution differentiable Python feature extraction
  -> weighted feature reconstruction loss against the input tensor
  -> rendered RMS penalty against source RMS
```

Enable it with:

```bash
python -m deep_trainer.train --dataset-root datasets/my_dataset --freq-bins 1024 --time-frames 512 --batch-size 8 --render-loss-weight 1.0 --render-loss-seconds 1.0 --render-loss-sample-rate 11025
```

This is intentionally a first surrogate, not a bit-perfect clone of the C++ renderer. The goal is to give training an audio-shaped objective instead of only a slot-table objective.

#### How `renderloss` Is Calculated

Current implementation:

```text
predicted activity logits + predicted oscillator parameters
  + predicted f0
  -> DifferentiableOscillatorBank
  -> surrogate mono waveform
  -> DifferentiableFeatureExtractor
  -> 3-channel feature tensor
  -> Smooth L1 loss against the input feature tensor
```

More explicitly:

```text
total_loss =
  activity_loss_weight  * balanced_BCEWithLogits(activity_logits, active_flags)
+ parameter_loss_weight * masked_smooth_l1(predicted_parameters, target_parameters)
+ f0_loss_weight        * smooth_l1(predicted_log_f0, target_log_f0)
+ crowding_loss_weight  * pairwise_frequency_crowding_penalty
+ render_loss_weight    * multi_resolution_weighted_feature_loss
+ render_rms_loss_weight * smooth_l1(log(rendered_rms), log(source_rms))
```

The current surrogate render path uses:

- predicted `sigmoid(activity_logits)` as a soft per-oscillator gain
- predicted oscillator parameters as normalized `0..1` controls
- predicted note frequency for rendering, supervised by dataset metadata
- velocity from dataset metadata
- a simplified oscillator bank with:
  - amplitude attack
  - amplitude decay
  - frequency decay
  - the same octave frequency-factor ladder as the C++ player: `0.5, 1, 2, 4, 8, 16, 32`
  - summed sine output

The f0 head predicts log-frequency normalized over the configured range, currently `40..2000 Hz`. The curriculum dataset generator now randomizes synthetic f0 over `55..440 Hz`, because training every example at one fixed base note makes f0 prediction meaningless.

Training logs include `val_f0_cents`, the mean absolute f0 error in cents. This is much easier to interpret than normalized f0 loss: `100` cents is one semitone, `1200` cents is one octave.

The f0 target is still partly ambiguous because the oscillator representation can trade base f0 against octave frequency factors. A prediction can be audio-plausible while reporting an octave-shifted base note. Use the analysis frontend's model f0 / audio pitch estimate / manual render-note switch to separate pitch estimation failure from timbre failure.

Activity loss is dynamically class-balanced by default because most oscillator slots are inactive. Without that balancing, the model can reduce loss by predicting silence.

The crowding penalty compares active oscillator frequency factors in log2 space and penalizes pairs that land too close together. It is activity-weighted, but activity is detached inside the crowding term so crowding cannot reduce its own loss by turning oscillators off. It is meant to push the model away from the current "one bland low-frequency cluster" failure mode without adding high-frequency weighting yet.

The current differentiable feature extractor uses:

- fixed render duration
- fixed sample rate
- Hann-window STFT
- linear-to-log-frequency projection
- per-example dB normalization
- the same 3 output channels as `.slft`:
  - log-frequency magnitude
  - temporal delta
  - onset emphasis

The current feature loss is strengthened by:

- multi-resolution comparison at full / half / quarter resolution
- per-channel weights
  - log-frequency magnitude weighted highest
  - temporal delta weighted lower
  - onset emphasis weighted above delta
- low-frequency emphasis through a descending frequency-band weight map
- an explicit log-RMS penalty using source WAV RMS from dataset metadata, so zero-energy renders are expensive

Relevant code:

- [losses.py](/C:/Users/Brandon/Documents/wip/SoundLearner/deep_trainer/losses.py)
- [differentiable_audio.py](/C:/Users/Brandon/Documents/wip/SoundLearner/deep_trainer/differentiable_audio.py)
- [train.py](/C:/Users/Brandon/Documents/wip/SoundLearner/deep_trainer/train.py)

#### Why It Can Look Almost Flat

If `val_render` barely moves, the most likely reasons are:

1. The rendered-feature loss is still numerically small relative to the parameter loss.
2. The surrogate renderer is too simple, so many different parameter sets produce very similar surrogate features.
3. Per-example dB normalization makes the render loss less sensitive to amplitude mistakes than your ears are.
4. Even with the current weighting, the feature loss may still not emphasize the perceptually important parts strongly enough.

So a nearly flat `val_render` does **not** mean the code path is broken by itself. It usually means the render objective is still too weak or too forgiving.

### Metrics Dashboard

Each run writes:

```text
runs/<run-name>/metrics.csv
```

Use `--tensorboard` to also write TensorBoard event logs:

```bash
python -m deep_trainer.train --dataset-root datasets/synth_256_1k --epochs 100 --batch-size 16 --resolution 256 --amp --tensorboard --output-dir runs/baseline_256_1k_long
```

Start TensorBoard:

```bash
tensorboard --logdir runs
```

## Prediction

```bash
python -m deep_trainer.predict --checkpoint runs/baseline/best.pt --feature features/data0.slft --output prediction.data
```

Useful flags:

```text
--write-all-slots
--activity-threshold 0.35
```

## Evaluation Harness

`evaluate.py` runs the full comparison loop for real holdout audio:

```text
real WAV
  -> source SLFT + Python preview
  -> checkpoint prediction
  -> predicted oscillator .data
  -> C++ player render
  -> rendered SLFT + Python preview
  -> summary metrics
  -> A/B listen folder + mel spectrogram PNGs
```

Single-file evaluation:

```bash
python -m deep_trainer.evaluate --checkpoint runs/baseline_256_1k/best.pt --input sounds/o_a2_1.wav --output-dir sounds/eval/o_a2_1_eval --resolution 256
```

Manifest evaluation:

```bash
python -m deep_trainer.evaluate --checkpoint runs/baseline_256_1k/best.pt --manifest sounds/manifest.csv --output-dir sounds/eval/baseline_256_1k --resolution 256 --limit 10
```

## Parameter-Space Analysis

When predictions start sounding suspiciously similar, analyze the oscillator vectors directly.

The analyzer writes:

- `pc_coordinates.csv`
- `parameter_variance.csv`
- `summary.json`
- `scatter.svg`
- `singular_values.svg`
- `pairwise_distance_histogram.svg`
- `report.md`

Example:

```bash
python -m deep_trainer.analyze_parameter_space --dataset-root datasets/synth_1024x512_1k --evaluation-root sounds/eval/baseline_1024x512_1k_b4_e40_10 --output-dir sounds/eval/baseline_1024x512_1k_b4_e40_10/analysis
```

## Curriculum Scripts

Windows batch helpers:

```text
scripts/build_curriculum_v1.bat
scripts/train_curriculum_v1.bat
scripts/build_complexity_curriculum_v1.bat
scripts/train_complexity_curriculum_v1.bat
```

Training wrappers now make the objective explicit:

```bat
scripts\train_complexity_curriculum_v1.bat c3 parameter
scripts\train_complexity_curriculum_v1.bat c3 renderloss
scripts\train_complexity_curriculum_v1.bat c3 renderloss_light
scripts\train_complexity_curriculum_v1.bat c6 renderloss c6
```

`parameter` is the old activity+parameter objective. `renderloss` adds the surrogate audio-feature reconstruction loss. `renderloss_light` is a cheaper first pass.
The optional third argument is the start stage. For example, `c6 renderloss c6` runs only `c6`, while `c6 renderloss c1` runs progressively from `c1` through `c6`.
The current wrapper writes rebalanced runs with `_balanced` / `_renderloss_balanced` suffixes so older collapsed runs remain available for comparison.

## Interpreting Loss

Rule of thumb:

```text
val_activity low, val_params flat -> model learned active slots but needs more data or better parameter supervision
train_loss down, val_loss up      -> overfitting
both losses flat                  -> pipeline, labels, learning rate, or architecture issue
```

## Current Limitations

1. It predicts fixed oscillator slots, so slot ordering matters.
2. The current audio reconstruction loss is only a first surrogate, not a faithful differentiable clone of the C++ renderer.
3. It does not rank multiple plausible oscillator solutions.
4. It does not train on real audio labels, because real audio has no oscillator ground truth.
5. It assumes parameter values are normalized to `0..1`.

## Next Improvements

1. Add a dataset manifest writer so training does not rely on directory scanning.
2. Add validation summaries as JSON/CSV.
3. Add per-parameter loss metrics.
4. Add parameter-specific loss weights.
5. Add predicted-instrument rendering and audio-space evaluation.
6. Add domain randomization to generated training data.
7. Add uncertainty or mixture-density heads for ambiguous inverse mappings.

## References

Selected references:

1. [ConvNeXt](https://arxiv.org/abs/2201.03545)
2. [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
3. [AdamW](https://arxiv.org/abs/1711.05101)
4. [PyTorch serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
5. [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)
6. [CLAP](https://arxiv.org/abs/2206.04769)
7. [BEATs](https://arxiv.org/abs/2212.09058)
8. [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)
9. [DDPM](https://arxiv.org/abs/2006.11239)
10. [DDIM](https://arxiv.org/abs/2010.02502)
