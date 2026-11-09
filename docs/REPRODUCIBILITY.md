# Reproduction notes

This document maps the research archive to its main entry points. The thesis PDF is the source for the reported results; the experiments have not been rerun as part of this documentation update.

## 1. Obtain the repository and large files

With Git and Git LFS installed:

```bash
git lfs install
git clone https://github.com/2Duffman/Diplomarbeit_public.git
cd Diplomarbeit_public
git lfs pull
```

The repository tracks CSV datasets and pickle artifacts through Git LFS. Downloading source files without their LFS content leaves pointer files in place of usable data or models. The full archive includes intermediate datasets and can be substantially larger than the thesis PDF.

## 2. Inspect the environment exports

The root `environment.yml` and `Case Study/environment.yml` describe the original `ml_env` environment. They contain pinned dependencies and platform-specific packages, including Windows-oriented TensorFlow packages. They are historical environment exports, not a verified cross-platform installation specification.

On a compatible system with Conda, the starting point is:

```bash
conda env create -f environment.yml
conda activate ml_env
```

Check dependencies against the script you intend to run; local explanation code also imports `shap`. A successful environment solve alone does not verify reproduction of the results.

## 3. Preserve the experimental split

Both `Machine Learning/` and `Case Study/` contain `train_ids.txt`, `test_ids.txt`, and a `data_utils.py` loader. The loaders check for overlap between training and test IDs. Their default paths currently point to the original author's machine.

For a direct data-loading check, run Python from `Case Study/` after obtaining the LFS data and use explicit relative paths:

```python
from data_utils import load_split

X_train, X_test, y_train, y_test = load_split(
    csv_path="data.csv",
    train_ids_path="train_ids.txt",
    test_ids_path="test_ids.txt",
)
print(X_train.shape, X_test.shape)
```

For scripts that rely on the loader defaults, adapt those paths before execution. Keep the supplied split for the corresponding analyses instead of regenerating it.

## 4. Choose an entry point

| Goal | Entry point | Prerequisites |
| :--- | :--- | :--- |
| Inspect model construction | `Case Study/model_training.py` | Review preprocessing and model parameters |
| Evaluate saved models | `Case Study/model_performance.py` | Dataset, corrected split paths, compatible saved artifacts |
| Study permutation importance | `Case Study/PFI.py`, `Case Study/calculate_PFI.py` | Dataset and fitted models; inspect script configuration |
| Study feature effects | `Case Study/ALE_clean.py`, `Case Study/calculate_ALE_clean.py` | Dataset and fitted models; inspect script configuration |
| Study interactions | `Case Study/calculate_H.py`, `Case Study/calculate_H_pairwise.py` | Dataset and fitted models; inspect selected features and computation settings |
| Generate local explanations | `Case Study/Generate_local_explainers.py` | Dataset, model artifacts, SHAP, and configured game IDs |
| Inspect tuning experiments | `Machine Learning/*_Optuna.py` | Dataset, script-specific paths and dependencies; potentially long runtimes |

Run scripts from the directory they expect: several input and output paths are relative to the current working directory. Inspect output paths first because experiments may replace previously generated outputs.

## Current limitations

- Some scripts use absolute paths from the original research environment.
- The directory contains historical experiments as well as case-study code. They are not a single orchestrated pipeline.
- `Machine Learning/Final_best_models.py` currently contains a stray `y` after a `print(...)` statement, so that historical script requires a syntax correction before use.
- Serialized model compatibility depends on the original libraries and versions. Keras models also require their corresponding preprocessing artifacts.
- Reproduction of the reported table requires the matching data, split, preprocessing, model artifacts, and environment. No claim of bit-for-bit reproduction is made here.

For the experimental interpretation and limitations, read Chapter 5 and Appendix A of the [thesis](diploma-thesis.pdf).
