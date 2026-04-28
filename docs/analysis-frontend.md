# Analysis Frontend

The analysis frontend is a local browser workbench for listening to and inspecting model predictions on real WAV files.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m deep_trainer.analysis_frontend
```

Then open:

```text
http://127.0.0.1:8765
```

## What It Does

For each dropped WAV file, the frontend runs this pipeline:

```text
uploaded WAV
  -> Python SLFT feature extraction
  -> checkpoint prediction
  -> oscillator .data file
  -> C++ player render
  -> A/B audio pair
  -> mel previews
  -> feature previews
  -> feature-space metrics
  -> 2D PCA placement against a selected reference dataset
```

It writes each analysis run under:

```text
analysis_frontend/runs/<timestamp>_<sample>_<id>/
```

Those folders contain the uploaded source WAV, predicted `.data`, rendered WAV, feature tensors, preview images, and `result.json`.

## Controls

- `Checkpoint`: the model checkpoint used for prediction. The newest `runs/**/best.pt` is selected by default.
- `PCA Reference Dataset`: the synthetic dataset used as the green reference cloud in the PCA chart.
- `Frequency Bins` / `Time Frames`: feature tensor shape for the analysis pass.
- `Crop Seconds`: fixed input window length.
- `Activity Threshold`: oscillator slots below this predicted activity are not written to the `.data` file.
- `Render Velocity`: velocity passed to the C++ player.
- `Render Note`: choose whether the C++ player uses the model f0, a classical audio pitch estimate, or a manual note.
- `Manual Note Hz`: note frequency used when `Render Note` is set to manual.
- `Device`: use `cpu` when a training run is occupying the GPU.

## Notes

The browser UI is intentionally thin. The Python server owns the analysis work so it can reuse the same feature extraction, model loading, PCA helpers, and WSL player invocation as the command-line tools.

The audio pitch estimate is a diagnostic, not ground truth. It can lock to a strong harmonic when the fundamental is weak or absent. That is useful in itself: if model f0, estimated pitch, and the note you hear disagree, the current oscillator representation is probably allowing octave-shifted explanations.

The PCA view is a quick visual diagnostic, not proof of audio quality. If the current prediction dot keeps landing in the same tiny area for many different WAVs, that is a strong signal that the model is still collapsing toward one generic solution.
