# WineSensed one-shot wine-label benchmark: evidence release v1

This image-free release contains the fixed IDs, source hashes, augmentation-slot records, evaluation utilities, and per-query top-20 rankings for the 2026-09-24 evidence snapshot. The identity is **metadata `vintage_id`**, not a recovered wine hierarchy. No year-recognition task is claimed.

## Contents

| Split | Gallery | Real queries | Classes |
|---|---:|---:|---:|
| benchmark1000 | 1,000 | 4,295 | 1,000 |
| ablation100 (nested in benchmark1000) | 100 | 351 | 100 |
| val1384 | 1,384 | 5,442 | 1,384 |

- `splits/`: public image IDs, archive member names, byte SHA-256, vintage IDs and frozen roles. The nested subset preserves the parent IDs.
- `banks/`: 6,000 slots each for SAM-3D, Li 2022-style and 2D; fixed view recipes, actual generation parameters, source geometry where recorded, background hashes and fallback status. A failed geometric slot consumes the exact paired 2D bytes. Original source images are represented by the gallery split, not duplicated as synthetic slots.
- `results/`: 73 run CSVs: 26 SigLIP frozen/T3 runs, 26 recovered-OML DINO R runs, 12 validation-selected G runs, and all nine B combinations. Each row records query ID, class ID, true gallery ID, full true rank, and top-20 gallery IDs. T3 and R include both split sizes; G and B were evaluated on benchmark1000. `run-index.json` maps every file to its seed, step and source hashes.
- `eval/`: portable CPU NumPy ranking, paired rescued/harmed counts, class-cluster bootstrap, byte-preserving image selection and package-integrity checking.

No original images, generated images, source reviews/CSV, backgrounds, human notes, embeddings, model weights, or private project history are included. This is an evaluation/reconstruction release, not a standalone training distribution. Existing embeddings can be supplied to the ranking utility in the documented NPZ format; hashes identify the original experiment artifacts. Exact synthesis also requires the original background assets and segmentation/generation implementations, which are not redistributed here. Slot parameters alone are not a promise that arbitrary replacement backgrounds reproduce the same pixels.

## Obtain and reconstruct the released data

Dataset sources: [official WineSensed data](https://data.dtu.dk/articles/dataset/WineSensed_Learning_to_Taste_A_Multimodal_Wine_Dataset/23376560), [WineSensed project](https://thoranna.github.io/learning_to_taste/), and the [Hugging Face mirror](https://huggingface.co/datasets/Dakhoo/L2T-NeurIPS-2023).

This snapshot used the mirror image archive at revision `a198b934bb7267216a280295bb2273b359a80b98`, path `data/all/all.tar.gz`, SHA-256 `80be08379db6e76d804baede82b5aa5b7325661a028a3b0384bb7a72fb2f9dee` (35,409,352,118 bytes). Use the pinned URL in `splits/source-provenance.json`; do not silently substitute a changed `main` archive. Metadata came from the recorded DTU `metadata.zip`; extracted `images_reviews_attributes.csv` SHA-256 is `52fa0b65c6c0c9d2030f7e6d893ee3be30b865da31346cd9cdfcb406c0983b74`. The CSV actually used has no wine-name/wine-ID column, despite descriptions of other releases. The provenance file lists its actual columns and the exact metadata archive download/MD5.

After obtaining the archive under the source terms, from this directory:

```text
python eval/verify_hashes.py
python eval/rebuild_splits.py --archive /path/to/all.tar.gz --splits splits/benchmark1000.csv splits/ablation100.csv splits/val1384.csv --output /path/to/new-reconstruction
```

Alternatively, use `--images-root /path/to/extracted/images` instead of `--archive`. Every selected image is checked against its byte SHA-256. The script uses controlled hash filenames and never executes or blindly extracts paths from an archive. The output directory must be new. `--dry-run` with `--images-root` verifies bytes and writes an index referencing the external files without copying them. Reading the full compressed archive can be slow; it is not a model-compute step. Exact hashes take precedence over filename similarity. Dataset access is not guaranteed by this package.

## Evaluate stored embeddings

Python 3.10+ and NumPy are sufficient. No GPU, FAISS, project checkout, or trained-model code is required for these utilities.

An NPZ must contain `ids` (one-dimensional Unicode image IDs) and `features` (same-order finite nonzero image embeddings). IDs, not row position, join the split. SigLIP uses float64 L2-normalized cosine with deterministic gallery-order ties; recovered-OML DINO uses float64 squared Euclidean distance without normalization.

```text
python eval/evaluate.py rank --split splits/benchmark1000.csv --features /path/to/frozen-features.npz --metric cosine --compare-csv results/benchmark1000-frozen-frozen_raw-s42.csv --output frozen-recomputed.json
python eval/evaluate.py paired --targets results/benchmark1000-T3-sam_3d-s42.csv results/benchmark1000-T3-sam_3d-s43.csv results/benchmark1000-T3-sam_3d-s44.csv --reference results/benchmark1000-frozen-frozen_raw-s42.csv --output sam-vs-frozen.json
```

For R use `--metric sqeuclidean`. For B additionally supply `--bank /path/to/bank-features.npz --mode max` (or `mean`, `views-only`); the bank NPZ contains `source_ids`, `view_indices` (0–5), and `features`, six per source. `max` takes the maximum across original plus six views; `mean` averages seven unit embeddings and normalizes again; `views-only` takes maximum over six synthetic slots. The exact same original frozen feature file provides the real gallery and queries.

Paired differences average the specified training seeds before resampling **whole classes**, with 5,000 draws and seed 42. Query-weighted and class-macro differences are both returned; rescued/harmed counts are returned for every seed. Intervals are conditional on those seeds and exclude training-seed uncertainty; no multiple-comparison correction is applied. T3 mean±SD uses sample SD (`ddof=1`).

## Validation and interpretation

`validation/clean-room-check.json` records independent execution from a clean directory using only this release and externally supplied frozen/T3 NPZs. It verifies the frozen row and the three-seed T3 SAM-3D row against the historical formal result and checks all compared query ranks and top-20 lists. Per-query release CSVs are additionally checked against all 73 original case files. The release does not modify any historical formal output.

The frozen SigLIP row is **step 0 of the same pretrained initialization**, not a separate baseline competitor. T3 uses the final fixed 4,000 steps on benchmark1000 (1,000 on ablation100); G selects one step per method by mean three-seed val query-weighted Top-1, earliest checkpoint for exact ties. G choices are 2D/ICPR/Li 100 and SAM-3D 250. R is the **recovered OML regime on RTX 3080**, with different loss/miner/sampler/preprocessing/retrieval settings; it must not be presented as the newer Blackwell backbone-only protocol.

Metadata identity and independent capture sessions are not fully human verified. The benchmark and development images have prior inspection/use history; this is not an unseen blind test. A dense class can contribute many errors: fg0924 contributes 31 of 226 frozen errors. Diagnostic human/assistant judgments do not relabel these split files. Native-success-only subsets are secondary analyses. No claim that augmentation beats frozen, that SAM-3D significantly beats Li, or that geometry alone explains pipeline differences follows from this release.

## Citation, relationship and publication status

Please cite the underlying dataset:

```bibtex
@article{bender2023learning,
  title={Learning to Taste: A Multimodal Wine Dataset},
  author={Bender, Thoranna and S{\o}rensen, Simon M{\o}e and Kashani, Alireza and Hjorleifsson, K Eldjarn and Hyldig, Grethe and Hauberg, S{\o}ren and Belongie, Serge and Warburg, Frederik},
  journal={arXiv preprint arXiv:2308.16900},
  year={2023}
}
```

Dataset record: Warburg, Hauberg and Belongie, 2023, DOI `10.11583/DTU.23376560.v1`. For this evidence package cite “WineSensed one-shot wine-label benchmark, evidence release v1, 2026-09-24” plus its version/hash; the manuscript authors, final title and archival DOI will be supplied by the authors upon publication. No publication DOI or public release URL has been invented.

This work extends the earlier ICPR/ICRCV single-image geometric-augmentation line using a public metadata-defined benchmark and separate training, adaptation and frozen multiview-bank evaluations; it is not a direct reproduction of the earlier private-data scores.

The existing development repository remains private. This self-contained package is prepared for a separate public destination; publication is pending author destination/approval. See `LICENSE` for scope and upstream notices.
