# Legacy baseline snapshot

The pre-refactor repository is preserved by Git tag
`pre-yolopose-stgcn-refactor` (`04bdab6`). This document is evidence only; none
of the legacy checkpoints are used by the new runtime.

## Verified repository facts

- The legacy model consumed AlphaPose-style 17-joint arrays in 45-frame windows.
- All six `.pth` files had a two-class `(2, 256, 1)` classifier head, including
  the misleadingly named `NTU_87acc.pth`.
- Training applied `softmax` before `CrossEntropyLoss` and returned training
  accuracy under the `train_loss` key.
- Evaluation and notebooks contained author-local `D:\ASH` paths.

## Historical notebook outputs

These values were stored outputs, not a fresh reproducible run:

- one training notebook recorded validation accuracy about 0.947;
- a cross-scene unseen test recorded accuracy about 0.733;
- other fold/test outputs varied substantially.

They are not directly comparable to the new YOLO26n-pose pipeline. The new
baseline is the controlled scratch-versus-pretrained experiment produced by the
cloud notebook on identical pose caches and scene splits.
