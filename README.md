# AmsiNet: uncertainty-triggered multi-agent context fusion for script identification in Indian scene text

Code for the paper *AmsiNet: Uncertainty-Triggered Multi-Agent Context Fusion for Efficient Script Identification in Indian Scene Text* (Shinde, Malemath et al., submitted to Expert Systems with Applications).

A lightweight word agent classifies every word. Free leave-one-out context from the other words of the same line and block is fused log-linearly, and line (ViT-S) and block (ConvNeXt-T) agents are called only where the word scores stay uncertain (threshold chosen on validation data). The headline system, **AmsiNet-E**, uses an EfficientNet-B0 word agent; **AmsiNet-T** uses an EfficientViT-B0 word agent.

## Data (not included)
The datasets are not redistributed here. Download them from their owners and follow their terms.
* **BSTD** (Bharat Scene Text Dataset, De et al.): https://github.com/Bhashini-IITJ/BharatSceneTextDataset (arXiv:2511.23071). Check the licence stated by the authors.
* **MDIW-13** (Ferrer et al.): IEEE DataPort, doi 10.21227/656q-hc18.
* **CVSI-2015** (Sharma et al., ICDAR 2015): obtained from the competition organisers after registration, for research use.

## How to run
Experiments were run on Google Colab (NVIDIA A100 80 GB; the Step 25 agents on an NVIDIA T4; CPU for fusion and latency). Run the notebooks in `notebooks/` in numeric order. Each notebook mounts Google Drive and works under `MyDrive/amsinet/`; change `ROOT` at the top of a notebook to use another location. Finished steps are skipped on rerun.

| Notebook | Purpose |
|---|---|
| 01 | BSTD word crops, ten script classes, image-level splits |
| 02 | flat ViT-B/16 baseline (five seeds) |
| 03a | page, block, line and word agents (five seeds, linear and MoE heads) |
| 04 | MDIW-13 index and zero-shot transfer (AmsiNet-T) |
| 06-08 | context fusion, word-agent variants, negotiator / stacking / distillation negatives |
| 09 | five-seed fusion and cascade, mixed-script lines, Latin sink |
| 10 | disagreement gate and pull test |
| 11 | paired bootstrap against the flat ViT-B |
| 12 | region-agent gain and per-script F1 |
| 13 | unmerged 12-way comparison |
| 14, 15 | validation macro-F1 of the two word heads |
| 16 | cheaper flat baselines (ViT-S/16, EfficientNet-B0) |
| 17 | flat classifiers with free context, including the CNN-BiLSTM (Bhunia-style) baseline |
| 18 | example figures |
| 19 | paired tests for the EfficientNet-B0 word agent (AmsiNet-E) |
| 20 | numbers quoted in the paper |
| 21 | single-environment CPU latency of all agents |
| 22, 24 | confusion rates and confusion-matrix figure |
| 23 | CNN-BiLSTM (Bhunia-style) baseline, our reimplementation, not the authors' code |
| 25 | AmsiNet-E on MDIW-13: zero-shot, with line and document context |
| 26 | ablations with the EfficientNet-B0 word agent |
| 27 | AmsiNet-E word agent on CVSI-2015 (zero-shot) |

Protocol: ten script classes (Hindi+Marathi and Bengali+Assamese merged), splits by image, five seeds, mean ± 95% t-interval, paired bootstrap over test images, selection on validation, test set used once.

## Notes
* Model weights and large prediction files (`*.pt`, `*.npy`) are not in the repository; the notebooks regenerate them.
* The CNN-BiLSTM baseline in notebook 23 is our reimplementation of a published design.

## Citation
See `CITATION.cff`.

## Licence
Code: MIT. The datasets have their own licences.
