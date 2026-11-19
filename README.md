# ldct-iqa-quality-map-audit

Code for *Do Quality Maps Point at the Artifact? Auditing Regional Low-Dose CT Quality Assessment* (submitted to IEEE ISBI 2027).

Regional quality maps of low-dose CT (LDCT) image quality assessment (IQA) models are usually judged only by their global agreement with radiologist scores. This code audits the maps directly. It injects noise, blur, streak or shading of known position and strength into one patch of a slice, at a matched patch PSNR, and scores each map with two AUCs: Localization (Loc) asks whether the injected patch changes more than the other patches, and Direction (Dir) asks whether its quality score drops (below 0.5: the artifact is rated as an improvement).

Repository maintainer: [Mervan Ocal](https://github.com/mrvnocl), [Biomedical Imaging and AI Laboratory](https://biailab.com), Istanbul Technical University ([lab GitHub](https://github.com/itu-biai)).

## Run the study code

Use Python 3.11 or newer and run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
make ci
python scripts/train.py --data-root data/ldct_iqa_2023
python scripts/evaluate_global.py --data-root data/ldct_iqa_2023
python scripts/run_audit.py --data-root data/ldct_iqa_2023
python scripts/analyze.py
python scripts/make_figures.py --heatmap-values outputs/heatmap_values.csv --example-maps --data-root data/ldct_iqa_2023
```

The scripts expect the public [LDCT-IQA 2023 dataset](https://doi.org/10.5281/zenodo.7833096) in `data/ldct_iqa_2023/`: `train.json` and `test.json` with the mean opinion scores (MOS) of the radiologists, and the slices in one sub-folder per split (for example `image/` and `test images/`; the folders are found automatically). The data have 1,000 training and 300 test slices with MOS from 0 to 4. The training slices are split 80/20 into training and validation sets with a fixed random seed. The split is at slice level because the release has no patient identifiers, so patient-level grouping could not be verified. CT images, trained weights and per-instance outputs are excluded from the repository.

`scripts/train.py` trains the two-stream EfficientNet-B0 model with and without the Smart ROI body mask, a patch ResNet18 and a global ResNet18 for the seeds 42, 123, 456, 789 and 1011 (25 epochs, Adam, learning rate 1e-4, Smooth L1 loss, batch size 32, best checkpoint on validation loss). `scripts/run_audit.py` runs every injection of `manifests/benchmark_manifest.csv` through the seven map sources of each seed. The manifest lists the 168 test slices with MOS of at least 2, three target patches per slice with all four artifacts at 35, 30 and 25 dB, and 30 dB noise in air for the 29 slices with a background patch; `scripts/make_manifest.py` shows how it was built. All settings are in the `Config` dataclass in `src/config.py`.

This code is a cleaned-up version of the Colab notebooks used for the paper; the computations are unchanged. The seed-42 models were trained in the original development notebooks, and the pipeline reproduced their test SROCC within 0.005 before the other seeds were trained with the same training loop. Results can differ slightly between GPU types (for example with TF32 enabled); cells where the target patch barely responds, such as the HF stream under shading, are the most sensitive to this.

`results/` contains Table 1 of the manuscript (`table_1.tex`) and the summary files of the five seeds in `results/paper_5seed/`: global PLCC/SROCC/KROCC, per-seed Loc and Dir AUC, the Wilcoxon tests with Holm correction, bootstrap intervals, the background analysis, the values of Fig. 2 and the training log of seeds 123 to 1011. Table 1 can be rebuilt from them without the data, and the unit tests check it against the manuscript:

```sh
python scripts/analyze.py --from-summary results/paper_5seed
```

In Table 1, a dagger marks a Dir AUC that differs from 0.5 (Holm p < 0.05) and lies on the same side of 0.5 in all seeds; a double dagger marks a Loc AUC that differs from the local noise reference map.

## Development

`make ci` runs the unit tests, the Ruff format check and the Ruff linter (`make test`, `make format_check`, `make lint`). `make format` applies import sorting and formatting. Ruff settings are in `pyproject.toml`; experiment settings are in `src/config.py`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ocal2027qualitymaps,
  author    = {Ocal, Mervan and {\"U}nal, Mehmet Ozan and Erta{\c{s}}, Metin and Y{\i}ld{\i}r{\i}m, {\.I}sa},
  title     = {Do Quality Maps Point at the Artifact? Auditing Regional Low-Dose {CT} Quality Assessment},
  booktitle = {Proc. IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027},
  note      = {Under review}
}
```

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial research and educational use is permitted with attribution; commercial use requires prior written permission from the authors. See [LICENSE](LICENSE).
