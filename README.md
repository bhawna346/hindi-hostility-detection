# Hidden Shortcuts and Cheap Baselines in Hindi Hostility Detection

Code, predictions and result files for the paper by Bhawna Mittal and Mohit Tiwari
(Bharati Vidyapeeth's College of Engineering, New Delhi).

**Main finding.** In the CONSTRAINT-2021 Hindi hostility dataset, 84% of non-hostile posts contain a URL,
against at most 9% of hostile posts. Trained models (classical and fine-tuned transformers) learn this as a
shortcut. Removing URLs, mentions and hashtags lowers logistic regression from 0.970 to 0.893 coarse macro-F1.
Appending one random link flips 53–61% of correct hostile decisions of the best classical models, only in the
hostile → non-hostile direction, while a same-length random string flips at most 9%. On MACD Hindi, a dataset
without the cue, the same test has almost no effect. On cleaned text, MuRIL (0.922) leads the best classical
model (char SVM, 0.897) by a significant 2.5 points. Prompted Qwen2.5 LLMs reach at most 0.775.

**Status: all experiments are complete.** Every number in the paper is in `results/`.

## Data

- CONSTRAINT-2021 Hindi: https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021 (use its `Dataset/` folder; official splits 5,728 / 811 / 1,653)
- MACD (control dataset): https://github.com/ShareChatAI/MACD (use `dataset/hindi_{train,val,test}.csv`)

## Hardware used

| Part | Hardware |
|---|---|
| Classical models, ablations, URL tests, significance tests | Cloud CPU: Intel Xeon 2.30 GHz, 2 vCPU, ~7 GB RAM, no GPU |
| Transformers (mBERT, MuRIL, IndicBERT) | Free Google Colab, 1× NVIDIA T4 (16 GB), ~10 min per model per task |
| LLMs (Qwen2.5-1.5B / 7B Instruct, 4-bit) | Free Google Colab, 1× NVIDIA T4 |

Classical inference times are end to end (TF-IDF + classifier), median of five runs. Neural times are single runs.

## Setup

    pip install scikit-learn xgboost pandas scipy numpy
    # notebooks (Colab): transformers, accelerate, bitsandbytes, torch

Below, `D` is the path to the CONSTRAINT `Dataset/` folder.

## Paper item → script → result file

| Paper item | Command | Output in `results/` |
|---|---|---|
| Table 1, Section 3 counts | `python code/classical.py D results` | `data_stats.json` |
| Table 2 (15 classical models, raw text) | `python code/classical.py D results`, then `python code/classical_v2.py D results raw` (tuned fine-grained thresholds) | `classical_results.csv`, `classical_v2_raw_*.csv`, `classical_preds.json` |
| Table 2 inference column | `python code/timing.py D results` | `classical_timing_end_to_end.csv` |
| Figure 1, Figure 2, Table 3 (cue statistics and ablation) | `python code/ablation.py D results LR` and `python code/ablation.py D results SVM`; `python code/classical_v2.py D results "no URLs"` (also `"no URLs/mentions/hashtags"`, `"Devanagari only"`) | `surface_stats.csv`, `ablation_LR.csv`, `ablation_SVM.csv`, `classical_v2_no_URLs*.csv`, `classical_v2_Devanagari_only.csv` |
| Section 6.1 (word-level ablation) | `python code/wordclean.py D` | `word_ablation.txt` |
| Section 5.2 counterfactual link injection / removal + random-string control | `python code/url_injection.py D results` | `url_injection_classical.csv` |
| Section 5.2 / 6.2 directional test (sensitivity vs shortcut) | `python code/url_direction.py D results` | `url_direction_classical.csv` |
| Table 4 (transformers, seed 42) | Colab: `code/hindi_hostility_transformers_llms_v2.ipynb`, Part A | `transformer_coarse.csv`, `transformer_fine.csv` |
| Table 5 (LLMs) | Same notebook, Part B (if it runs out of GPU memory, use `code/partB_llm_cell_memory_safe.py` as the Part B cell) | `llm_results.csv` |
| Table 4 (seeds 13, 77) + transformer URL test | Colab: `code/hindi_hostility_v3_seeds_urltest.ipynb` | `v3_results.csv`, `v3_url_test.csv`, `v3_preds/*.npy` |
| Section 5.3 significance tests (McNemar + paired bootstrap, seeds 13 and 77) | `python code/sigtest.py D results/v3_preds results` | `significance_tests.csv` |
| Section 5.4, Table 7 (MACD control) | `python code/second_dataset.py MACD/dataset/hindi_train.csv MACD/dataset/hindi_test.csv text label 0 results/macd MACD/dataset/hindi_val.csv` | `macd/d2_results.csv`, `macd/d2_surface_stats.csv` |
| Section 6.5 error rate by post length | `python code/ablation.py D results LR` (and `SVM`) | `length_errors_LR.csv`, `length_errors_SVM.csv` |

Table 4 reports mean ± standard deviation over seeds 42, 13 and 77 for mBERT and MuRIL, combining
`transformer_*.csv` (seed 42) and `v3_results.csv` (seeds 13, 77). IndicBERT and the LLMs were run once.

## Notes on the Colab notebooks

- IndicBERT is a gated model: accept its terms on Hugging Face and log in with your own token
  (`from huggingface_hub import login; login()`). **Never commit a notebook that contains a token.**
- Results are saved to `MyDrive/hostility/` incrementally, so a disconnected run can be resumed.
- Settings: 4 epochs, AdamW, lr 2e-5, batch 16, max length 128; the fine-grained task uses BCE with
  pos_weight and per-label thresholds tuned on validation. LLMs: 4-bit, greedy decoding, 0-shot and 5-shot.

## Citation

Please cite the paper (citation to be added after publication) and the CONSTRAINT-2021 dataset
(Bhardwaj et al., 2020; Patwa et al., 2021).
