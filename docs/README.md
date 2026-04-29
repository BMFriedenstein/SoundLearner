# SoundLearner Docs

This folder holds detailed implementation notes that are too low-level for the top-level README.

Current documents:

- [Adaptive Curriculum](./adaptive-curriculum.md) - supervisor-driven dataset generation and training promotion loop
- [Dataset Discovery](./dataset-discovery.md) - how the trainer locates feature tensors and label files inside a dataset root
- [Analysis Frontend](./analysis-frontend.md) - local drag-and-drop A/B listening, spectrogram, metric, and PCA workbench
- [Project Guide](./project-guide.md) - dataset generation, feature extraction, curriculum notes, and ML direction
- [Trainer Guide](./trainer.md) - the PyTorch training, evaluation, and analysis stack

The home README should stay focused on project intent, roadmap, core commands, and where to go next. Use this folder for subsystem detail.
