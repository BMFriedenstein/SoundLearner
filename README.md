# SoundLearner

SoundLearner is an experiment in learning a compact synthesizer model from audio.

The project generates labelled audio from an oscillator-based instrument model, extracts fixed-size audio feature tensors, and aims to train a model that maps an input sound back to synthesizer parameters. The long-term goal is not just to classify sounds, but to recover enough structure that the player can resynthesize similar sounds from MIDI.

![SoundLearner](SoundLearner.jpg)

## Current Shape

SoundLearner is being modernized from an older C++17 and TensorFlow 1.x project into a cleaner C++23 dataset and feature pipeline, with the ML architecture intentionally left open for redesign.

The repo currently contains:

1. `instrument` - oscillator-based sound generation.
2. `dataset_builder` - synthetic training data generation with known oscillator labels.
3. `feature_extractor` - WAV to fixed-size feature tensor conversion.
4. `deep_trainer` - legacy TensorFlow training scripts kept for reference.
5. `player` - instrument model loading and WAV rendering.
6. `Analyse` - older analysis scripts and experiments.

## Build

The C++ tools are built with Meson and C++23.

```bash
meson setup build -Dcpp_std=c++23
meson compile -C build
```

If the build directory already exists:

```bash
meson setup build --reconfigure -Dcpp_std=c++23
meson compile -C build
```

The native tools currently depend on FFTW3.

## Developer Tooling

The repo includes project-level Clang tooling:

```text
.clang-format       formatting rules for C++23-era code
.clangd             clangd editor/indexing configuration
meson/clang.ini     optional Meson native file for Clang builds
AGENTS.md           quick-start notes for future Codex/LLM sessions
```

`clangd` is configured to read `build/compile_commands.json`, so generate or reconfigure the default build directory before relying on editor diagnostics.

To build with Clang once `clang++` is installed:

```bash
meson setup build-clang --native-file meson/clang.ini
meson compile -C build-clang
```

The provided native file currently uses Ubuntu's versioned `clang++-18` binary. If your system provides an unversioned `clang++`, update `meson/clang.ini` accordingly.

To run a strict warning pass with the currently available compiler:

```bash
meson setup build-warnings --wipe -Dcpp_std=c++23 -Dwerror=true
meson compile -C build-warnings
```

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
--write-ppm-preview            also write PPM preview images
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
-p --preview-prefix <path>     optional preview image prefix
-t --crop-seconds <seconds>    fixed analysis window, default 5
--crop-start-seconds <seconds> crop offset, default 0
--write-ppm-preview            also write a PPM preview
```

The extractor uses fixed-window cropping. Long WAV files are cropped. Short WAV files are zero-padded. This avoids stretching the time axis, which would hide the true temporal scale of the sound.

## Feature Tensor

The current canonical model input is a compact binary `.slft` tensor:

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

## ML Architecture Direction

The old `deep_trainer` CNN was designed around 512 px images and TensorFlow 1.x. That was reasonable for a small GPU years ago, but it is no longer the best center of gravity for this project.

The strongest next architecture is likely not a plain image CNN and not diffusion as the first model. The recommended first serious model is:

```text
audio feature tensor
  -> modern audio/image encoder
  -> latent embedding
  -> structured oscillator-parameter heads
  -> optional differentiable or offline resynthesis loss
```

### Why Not Start With Diffusion?

Diffusion models are excellent generative models, especially when the output is high dimensional and there are many valid answers. This project has an inverse problem: map audio to a compact set of oscillator parameters.

Diffusion could help later because many oscillator configurations may sound similar. That makes the target distribution multi-modal. A diffusion head could sample multiple plausible parameter sets instead of predicting one average answer.

However, diffusion is heavier to train, slower to evaluate, and harder to debug. On a 12GB GPU it is possible, but it adds complexity before the project has a strong supervised baseline. The first milestone should be a model that predicts parameters directly and can be evaluated clearly.

Recommended path:

1. Build a strong supervised encoder-regressor baseline.
2. Add uncertainty or mixture-density outputs for ambiguous parameters.
3. Add audio reconstruction/resynthesis metrics.
4. Consider diffusion only if the direct model collapses ambiguous sounds into bad average parameters.

### Recommended Baseline

Use a modern ConvNet-style encoder over the `.slft` tensor.

Good first choices:

1. ConvNeXt-Tiny style encoder.
2. EfficientNetV2-small style encoder.
3. ResNet with modern training choices if simplicity matters most.

The model output should be structured, not a single flat vector if avoidable.

Suggested heads:

```text
global sound embedding
  -> oscillator count / activity head
  -> frequency trajectory head
  -> amplitude envelope head
  -> phase / waveform / modulation head, if retained
  -> confidence or uncertainty head
```

This better matches the synthesizer domain than treating the output as unrelated regression numbers.

### Transformer Option

An Audio Spectrogram Transformer or small hierarchical transformer is also reasonable, especially if the input resolution grows beyond 512 or if longer clips are used.

For a 12GB GPU, a full large transformer is probably not the best first move. A small AST-like model or hybrid ConvNet-plus-attention model is more practical.

Use a transformer when:

1. The model needs longer temporal context.
2. The feature tensor resolution becomes large.
3. Pretraining on real unlabeled audio becomes important.
4. The ConvNet baseline misses long-range time/frequency relationships.

### Probabilistic Output

A single set of oscillator parameters may not be the only valid answer for a real sound. This matters for piano, birds, voice, percussion, and noisy recordings.

Instead of forcing one answer, the model should eventually expose uncertainty:

1. Predict mean and variance for continuous parameters.
2. Use a mixture density head for parameters with multiple plausible values.
3. Predict top-k candidate instruments and resynthesize all candidates.
4. Rank candidates by audio reconstruction quality.

This gives the project a path toward "several good approximations" rather than pretending there is one perfect inverse.

### Real Audio Evaluation

Generated oscillator audio is the training dataset because it has perfect labels. Real audio should be used as the test and performance benchmark.

Real audio examples:

1. Piano notes.
2. Plucked strings.
3. Bowed strings.
4. Human whistle or voice vowels.
5. Bird chirps.
6. Percussion and transient sounds.

For real audio there may be no ground-truth oscillator labels. Evaluation should therefore include audio-domain metrics:

1. Resynthesize predicted parameters.
2. Compare input audio and rendered audio.
3. Measure spectral distance, multi-resolution STFT loss, onset timing, pitch contour, and loudness envelope.
4. Keep human preview spectrograms for qualitative debugging.
5. Build a small curated listening/evaluation set that never appears in training.

The most important test is not whether the model matches generated labels. It is whether predicted parameters resynthesize convincing audio for sounds outside the oscillator generator.

### Synthetic-To-Real Gap

The main ML risk is domain gap. A model trained only on clean generated oscillator sounds may fail on piano, birds, room noise, microphone coloration, compression, and performance variation.

The dataset builder should eventually add domain randomization:

1. Noise.
2. Room impulse responses.
3. EQ and filtering.
4. Compression and clipping.
5. Pitch drift.
6. Timing jitter.
7. Reverb.
8. Multiple simultaneous partials and transient layers.

The goal is not to make the synthetic data pretty. The goal is to make it broad enough that real audio does not look alien to the model.

### 12GB VRAM Target

A practical training target for a 12GB GPU:

```text
input:        3 x 512 x 512 or 3 x 768 x 768
precision:    mixed precision
batch size:   tune around memory, likely 4-32 depending on encoder
model:        small-to-medium ConvNet or small AST
optimizer:    AdamW
training:     supervised synthetic labels first
validation:   held-out synthetic plus curated real audio benchmark
```

If memory becomes tight, prefer gradient accumulation over shrinking the input too aggressively. The frequency and time resolution are part of the signal.

## Proposed ML Milestones

1. Freeze the `.slft` tensor format as the model input.
2. Build a dataset manifest that records audio path, feature path, labels, and generator settings.
3. Replace the TensorFlow 1.x trainer with a modern training stack.
4. Train a ConvNeXt-style supervised baseline on generated data.
5. Render predicted oscillator parameters back to WAV.
6. Score predictions with both parameter loss and audio reconstruction loss.
7. Create a real-audio evaluation set and never train on it.
8. Add domain randomization to the generator.
9. Add uncertainty or mixture-density heads.
10. Revisit diffusion after the baseline exposes where ambiguity is hurting.

## Notes On Diffusion

Diffusion is worth keeping in the design space, but as a second-stage model:

```text
audio encoder embedding
  -> conditional parameter diffusion
  -> sample several plausible oscillator parameter sets
  -> render candidates
  -> rank by audio reconstruction loss
```

This is a good architecture if direct regression produces smeared or average parameters. It is not the fastest route to a useful first model.

## Further Reading

Useful architecture references:

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
2. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
3. [A ConvNet for the 2020s / ConvNeXt](https://arxiv.org/abs/2201.03545)
4. [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)
5. [CLAP: Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769)
6. [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058)

## Milestones

### Sound Learner

- [x] Develop oscillator-based instrument model.
- [x] Add basic WAV read and write.
- [x] Add instrument model file write.
- [x] Build generated deep learning dataset.
- [x] Add standalone feature extractor.
- [x] Add flexible feature resolution.
- [x] Add fixed-window crop/pad feature extraction.
- [x] Write canonical `.slft` feature tensors.
- [ ] Add dataset manifest generation.
- [ ] Replace legacy TensorFlow trainer.
- [ ] Train modern supervised baseline.
- [ ] Add audio reconstruction evaluation.
- [ ] Test on real non-generated audio.
- [ ] Add domain randomization.
- [ ] Evaluate probabilistic or diffusion-based parameter heads.
- [ ] Add MIDI file read.

### Player

- [x] Open instrument model files.
- [x] Write to WAV file.
- [ ] Read MIDI input from IO.
- [ ] Playback to audio device.
- [ ] Add polyphonic playback.
- [ ] Add stereoization.
