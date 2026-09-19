# Explainable Change-Aware Building Damage Assessment from Bi-Temporal Satellite Imagery

**MSc Data Science and Artificial Intelligence — Sheffield Hallam University**  
**Researcher:** Ranga Rama Raviteja Chakka  
**Project type:** Secondary-data, building-level deep-learning research

## Overview

This research investigates whether comparing satellite imagery captured **before and after a disaster** improves automated building-damage assessment over using the post-disaster image alone. It uses the **xBD/xView2** benchmark to classify individual building-centred image crops into four categories:

1. `no-damage`
2. `minor-damage`
3. `major-damage`
4. `destroyed`

The study compares three models from the **ConvNeXt-Tiny** backbone family and a validation-selected probability ensemble. It combines exploratory data analysis, paired-image preprocessing, scene-level leakage controls, class-sensitive evaluation and Grad-CAM diagnostics.

**Scope:** This repository documents a research pipeline, **not** a certified structural inspection or operational disaster-response system. The dissertation does not claim that its headline result generalises to completely unseen disasters. Flask, TorchScript and application deployment are not part of the final reported research scope.

## Research question

To what extent can a leakage-aware building-level computer-vision pipeline use pre- and post-disaster satellite imagery to classify xBD damage severity, and how do temporal representations affect class-wise performance and Grad-CAM error patterns?

## Dataset

- **Benchmark:** xBD/xView2, paired pre-/post-disaster optical satellite imagery and annotated building polygons.
- **Kaggle access:** [xView2 Challenge Dataset — train and test](https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test).
- **Foundational publication:** Gupta et al. (2019), [*xBD: A Dataset for Assessing Building Damage from Satellite Imagery*](https://arxiv.org/abs/1911.09296).

The executed study parsed **2,799 labelled scene pairs** and **159,794 classified building polygons**, then retained **60,000 paired building crops** at **256 × 256 pixels** for modelling. `un-classified` or unusable annotations were excluded from four-class training.

**Data access and licence:** Obtain the imagery independently from the dataset provider. Review the applicable access, attribution and non-commercial/reuse terms at the time of download. The raw xBD/xView2 imagery, extracted crop dataset and annotations are **not redistributed in this repository**.

## Research workflow

```text
Research problem and literature review
                |
        xBD/xView2 access
                |
       JSON/WKT parsing
                |
    EDA and quality assessment
                |
 Paired building crop extraction
                |
  Scene-grouped train/val/test
                |
  Three ConvNeXt-Tiny models
                |
 Validation checkpoint selection
                |
 Validation-selected ensemble
                |
 Held-out test-set evaluation
                |
 Confusion matrices and Grad-CAM
                |
 Results, figures and discussion
```

### Preprocessing and evaluation controls

- Match pre-disaster and post-disaster images using scene identifiers.
- Parse building polygons and corresponding four-class damage labels.
- Remove unclassified, invalid and unsuitable building examples.
- Extract the **same spatial region** from both images, retaining local building context.
- Resize paired crops to **256 × 256**; apply consistent geometric augmentation to each pair and normalise the model inputs.
- Separate **entire satellite scenes** across training, validation and test; verify disjoint `scene_id` and `crop_id` sets.
- Select checkpoints and ensemble weights on the **validation set only**. Reserve test data for final evaluation.

The primary experiment is a **scene-independent, in-domain** split: disaster types appear across partitions, but individual scenes do not. This is **not** leave-one-disaster-out evaluation.

| Partition | Building crops | Satellite scenes |
|---|---:|---:|
| Training | 39,892 | 712 |
| Validation | 9,477 | 174 |
| Test | 10,631 | 222 |
| **Total** | **60,000** | **1,108** |

## Models

| Approach | Input and method | Role in the comparison |
|---|---|---|
| **Post-only ConvNeXt-Tiny** | Post-disaster RGB crop | Strong baseline using only visible post-event cues |
| **9-channel Change-Stack ConvNeXt-Tiny** | Pre RGB + post RGB + absolute RGB difference | Tests explicit pixel-level temporal change |
| **Siamese Change-Aware ConvNeXt-Tiny** | Shared pre/post backbone with feature-change and interaction fusion | Tests learned feature-level temporal comparison |
| **Validation-optimised ensemble** | Weighted average of model class probabilities | Tests whether complementary errors improve classification |

The final validation-selected ensemble used **0.50 post-only + 0.10 change-stack + 0.40 Siamese** probability weights. The weights were not fitted to the test set.

## Reported results

The following results belong to the **executed in-domain experiment**, not an unseen-disaster benchmark.

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Majority-class baseline | 66.37% | 25.00% | — |
| Change-stack ConvNeXt-Tiny | 80.01% | 66.78% | 67.83% |
| Siamese Change-Aware ConvNeXt-Tiny | 83.54% | 71.72% | 72.79% |
| Post-only ConvNeXt-Tiny | 83.83% | 72.05% | 72.69% |
| **Validation-optimised ensemble** | **84.92%** | **73.31%** | **74.40%** |

The ensemble also achieved approximately **0.936 macro one-vs-rest ROC-AUC**. Final class-wise recall was **93.11%** (no damage), **50.21%** (minor damage), **76.73%** (major damage) and **73.21%** (destroyed).

**Interpretation:** The post-only model was the strongest individual model by accuracy. Explicit temporal modelling did **not** consistently outperform that baseline, but the temporal models supplied complementary information that improved the ensemble. Minor-damage recognition remains the clearest limitation. The majority-class baseline demonstrates why accuracy must be reported alongside balanced accuracy, macro-F1 and per-class recall.

## Repository contents

A concise submission structure is:

```text
.
├── README.md
├── notebook/
│   └── chekka.ipynb            # Executed FINAL ConvNeXt notebook
├── figures/                   # Selected figures actually included in the repo
├── results/                   # Selected small CSV / JSON summaries, if included
└── .gitignore
```

**Only include files that are actually present.** The `figures/` and `results/` folders are optional; if they are not uploaded, the executed notebook remains the primary source of the figures and metrics. Do not commit the raw dataset, generated crop images, large model checkpoints, temporary files or credentials.

> **Notebook identity is important:** the `chekka.ipynb` copy available alongside the project materials is an *older ResNet18/Flask version*, whereas the executed **ConvNeXt** experiment with the reported 84.92% result is in `chekka(1).ipynb`. When preparing this repository, use the **executed ConvNeXt notebook** and, if the dissertation cites `chekka.ipynb`, rename that correct file to `notebook/chekka.ipynb`. Do **not** accidentally upload the older ResNet notebook as the final experiment. An older export/deployment section may still exist in the executed source notebook, but it is **outside the scope of this final research report**.

## Reproducing the experiment on Kaggle

1. Create a Kaggle notebook and attach the [xBD/xView2 dataset](https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test) as input.
2. Import the **executed ConvNeXt** `notebook/chekka.ipynb` from this repository.
3. Enable a **GPU** in notebook settings. Internet may be necessary on the first run to download pretrained ConvNeXt weights if they are not cached.
4. Keep `EXPERIMENT_MODE = "in_domain"` to reproduce the reported evaluation protocol. Do not substitute `"cross_event"` and expect the same results.
5. Run the research sections in order: dependency checks, configuration, dataset discovery, EDA, crop extraction, scene-grouped split, paired preprocessing, model training, ensemble selection, evaluation and **Grad-CAM**.
6. The research results are written under the Kaggle working directory shown in the notebook. With the supplied configuration, the main location is:

   ```text
   /kaggle/working/xbd_80plus_accuracy_in_domain/
   ├── figures/
   ├── tables/
   ├── building_crops/
   └── models/
   ```

7. Important result files include `tables/individual_model_results.csv` and `tables/final_model_comparison_results.csv`; figures include the saved EDA, learning curves, comparison plots, confusion matrices and Grad-CAM diagnostics.

**Execution note:** The source notebook contains a legacy optional export section after Grad-CAM. That section is not needed to reproduce the dissertation's reported research findings and is not described here as a project deliverable. The original notebook's subsequent run-record cell relies on variables defined in that optional section; it is **not independently runnable if that section is skipped**. The core analysis and its metrics/figures are produced beforehand. For an exclusively research-only published notebook, remove or refactor the optional export and dependent run-record cells before pushing; keep a separate archived copy of the original executed notebook if auditability is required.

**Reproduction caveat:** Exact scores can vary with library versions, random seeds, GPU kernels, pretrained-weight availability and dataset layout. The reported values above are the saved results of the executed experiment, not a guaranteed result from every fresh run.

### Main software

Python, Jupyter/Kaggle, PyTorch, torchvision, NumPy, pandas, scikit-learn, Pillow, Matplotlib, Shapely and kagglehub. The notebook checks or installs some missing packages automatically. For reproducibility, record versions from the executed Kaggle environment rather than asserting unverified pinned versions.

## Explainability and research limitations

Grad-CAM heatmaps were generated for correct examples and high-confidence errors to inspect influential regions. Heatmap overlap with a building is **diagnostic evidence only**: it does not prove causal reasoning, predictive correctness, user trust or structural safety.

Additional limitations include scene-independent **in-domain** testing rather than external unseen-disaster validation, substantial variation between disaster events, difficulty identifying subtle minor damage, possible registration/illumination differences and the inability of optical images to establish internal structural condition.

Suggested follow-up research includes leave-one-disaster-out validation, improved spatial registration, probability calibration and uncertainty handling, minority-class analysis, and independently licensed multimodal datasets. Any future evaluation involving human participants would require a separate, approved ethics design.

## Research ethics and responsible use

This is a **secondary-data-only** study under the project's UREC1 scope. It did not recruit participants, conduct interviews or surveys, collect personal data, or perform human evaluation of predictions or Grad-CAM. Please respect xBD/xView2 licence terms, acknowledge its creators, and do not use experimental predictions as certified emergency-response or structural-safety decisions.

## Citation and acknowledgement

If you use the dataset or build on this research, cite the original xBD publication:

> Gupta, R., et al. (2019). *xBD: A dataset for assessing building damage from satellite imagery*. arXiv:1911.09296. https://arxiv.org/abs/1911.09296

ConvNeXt architecture: Liu, Z., et al. (2022), *A ConvNet for the 2020s*. Grad-CAM: Selvaraju, R. R., et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. Full academic references are provided in the dissertation.

**Academic project:** Ranga Rama Raviteja Chakka, MSc Data Science and Artificial Intelligence, Sheffield Hallam University.
