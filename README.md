# WineSensed one-shot wine-label benchmark

Fixed benchmark splits and evaluation evidence for **When Does Geometric View Synthesis Help Wine Label Retrieval? A Public One-Shot Benchmark Across Self-Supervised and Vision–Language Backbones**, by Yueh-Cheng Huang.

The benchmark uses WineSensed's `vintage_id` as its class label. It has 1,000 classes, one enrollment image per class, and 4,295 real query images. A nested 100-class subset has 351 queries; a separate 1,384-class development split has 5,442 queries.

## Download and scope

[Evidence release v1](https://github.com/EdenHuang/winesensed-one-shot-benchmark/releases/tag/evidence-v1) contains the unchanged **2026-09-24 evidence snapshot**: 96 files, including 73 per-query ranking CSVs. The downloadable archive is 11,741,894 bytes. Its SHA-256 is:

```text
36394cd3819efb389977aa5dc35845ae23235e734090903eb60a23eb63ec5f6c
```

The archive's filename ends in `v2` because the packaging revision added `.gitattributes` to preserve exact bytes across operating systems; its data and rankings remain evidence release v1. The same 96 files are available in [`evidence-v1/`](evidence-v1/). Repository-level README, citation metadata and checksum files sit outside that snapshot and are not part of the archive hash.

| Included | Scope |
|---|---|
| Image identifiers, split assignments and hashes | 12,121 distinct images across the benchmark and development splits |
| Source provenance | Pinned WineSensed image-archive revision and metadata hash |
| Synthesis records | Six slots per source for SAM-3D, Li 2022-style and the original 2D baseline, including fallback status |
| Per-query rankings | 73 frozen, fixed-budget SigLIP/DINO, validation-selected SigLIP and frozen multi-view enrollment runs |
| Evaluation utilities | Split reconstruction, integrity verification, embedding ranking and paired class-cluster bootstrap |

This snapshot does **not** contain the later flip-free 2D control, linear-head or LoRA rankings, the fourth synthesis bank, or cosine-weighted enrollment results. These remain available from the author. It also does not include original or synthetic images, backgrounds, source reviews, embeddings, weights, or complete training and synthesis implementations. Exact image generation needs the original implementations and background assets; the slot records alone cannot reproduce those pixels.

The [snapshot README](evidence-v1/README.md) contains detailed protocols and reconstruction instructions. Its publication-pending sentence describes the snapshot's preparation date; this repository is its public destination. The underlying development repository and its history are separate.

## Method names in the manuscript

The IEEE Access submission manuscript (v8, naming update dated 2026-09-27) uses the following display names. They are aliases for the existing benchmark pipelines, not names of new models or Meta's SAM 3D systems. Historical method IDs, filenames and labels in the evidence snapshot remain unchanged.

| Manuscript display name | Stable method ID | Localization and construction | Ranking CSVs in evidence-v1 |
|---|---|---|---:|
| Edge-based [4] | `icpr_hong` | Edge-based localization + rim-estimating construction | 15 |
| SAM-based [5] | `sam_3d` | SAM localization + rim-estimating construction | 18 |
| Frontal-assumption, adapted from Li et al. [7] | `li2022_style` | SAM localization + adapted frontal-assumption construction | 18 |
| 2D | `original_2d` | Original 2D augmentation baseline | 18 |
| Frozen | `frozen_raw` | Initial backbone without benchmark adaptation | 4 |

Thus the legacy labels `ICPR`, `SAM-3D` and `Li 2022-style` correspond to Edge-based, SAM-based and Frontal-assumption, respectively. Both `sam_3d` and `li2022_style` use SAM localization; their geometric constructions differ. The frontal-assumption pipeline adapts the geometry of [7] to this benchmark's SAM localization and one-image, six-view protocol. It is not a reproduction of [7]'s original FCN-based pipeline. Construction-only tests that use ground-truth masks refer to **rim-estimating construction** and **frontal-assumption construction**, rather than the full localization pipelines.

The reference numbers above follow the manuscript: [4] is [Single-Image Driven 3D Viewpoint Training Data Augmentation for Effective Label Recognition](https://doi.org/10.1007/978-3-031-78125-4_14); [5] is [HierarchicalWine](https://doi.org/10.1109/ICRCV67407.2025.11349206); [7] is [On Wine Label Image Data Augmentation Through Viewpoint Based Transformation](https://doi.org/10.16798/j.issn.1003-0530.2022.01.006).

These five stable IDs are the only method values in the 73-run `evidence-v1/results/run-index.json`. Later `icpr_li` (edge-based localization + frontal-assumption construction), `original_2d_noflip`, linear-head and LoRA results are **not included in evidence-v1**. This naming clarification adds no results and does not replace the fixed snapshot, release archive, tag or checksums.

## Check the package

Use Python 3.10 or later. NumPy is needed for ranking and paired comparisons; the independent checks used NumPy 2.4.4.

```text
python -m pip install numpy==2.4.4
python evidence-v1/eval/verify_hashes.py
```

The verifier checks 95 content files; the 96th file is the manifest itself. To check the downloaded archive, compare its SHA-256 with the value above or the release's `SHA256SUMS.txt`.

A saved-ranking comparison requires no images, model weights, or GPU:

```text
python evidence-v1/eval/evaluate.py paired --targets evidence-v1/results/benchmark1000-T3-sam_3d-s42.csv evidence-v1/results/benchmark1000-T3-sam_3d-s43.csv evidence-v1/results/benchmark1000-T3-sam_3d-s44.csv --reference evidence-v1/results/benchmark1000-frozen-frozen_raw-s42.csv --output sam-vs-frozen.json
```

Expected top-1 counts are **4,069 / 4,295** for frozen SigLIP and **3,607, 3,707, 3,704 / 4,295** for the three fixed-budget SAM-3D seeds. Their seed-averaged difference is **−9.228 percentage points**, with a class-cluster 95% interval of **[−12.339, −6.625]**. These intervals are conditional on the three training seeds.

For top-k results from any included ranking CSV, save the following as a local Python script and run it from the repository directory. It reads only the published results:

```python
import csv
import json
from pathlib import Path

root = Path("evidence-v1")
runs = json.loads((root / "results/run-index.json").read_text(encoding="utf-8"))
for run in runs:
    with (root / run["file"]).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {k: sum(int(row["true_rank"]) <= k for row in rows)
              for k in (1, 5, 10, 20)}
    assert len(rows) == run["queries"] and counts[1] == run["correct"]
    print(run["file"], len(rows), counts)
```

## Rebuild the image splits

Obtain the original data from [WineSensed at DTU](https://data.dtu.dk/articles/dataset/WineSensed_Learning_to_Taste_A_Multimodal_Wine_Dataset/23376560) or the [Hugging Face mirror](https://huggingface.co/datasets/Dakhoo/L2T-NeurIPS-2023), using the pinned archive specified in [`source-provenance.json`](evidence-v1/splits/source-provenance.json). Follow [`evidence-v1/README.md`](evidence-v1/README.md) to reconstruct the splits and verify image hashes. The source archive is approximately 35.4 GB; it is not distributed here.

## Citation and rights

Please cite the underlying WineSensed dataset and this versioned evidence package. Citation metadata for the package is in [`CITATION.cff`](CITATION.cff); the snapshot README provides the dataset citation. No accepted-paper citation or archival DOI is claimed.

The existing [snapshot license](evidence-v1/LICENSE) applies within its stated scope: MIT for the snapshot's evaluation utilities and CC BY 4.0 for its newly authored results and documentation. Dataset identifiers and source records retain applicable upstream terms. WineSensed's Hugging Face card states CC BY-NC-ND 4.0, while its DTU record states CC BY-NC 4.0; consult the provider for the specific files used. This repository does not relicense source images or third-party content, and makes no additional license grant for the repository-level metadata.
