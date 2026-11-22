# Hindi hostility detection — code and results

Dataset: https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021 (clone it; use Dataset/ folder)

## Classical experiments (CPU, done)
    pip install scikit-learn xgboost pandas scipy
    python code/classical.py <path-to>/Dataset results      # Table 2 (~50 min on 2 CPUs, XGBoost is the slow part)
    python code/ablation.py  <path-to>/Dataset results SVM  # Figure 2, char SVM
    python code/ablation.py  <path-to>/Dataset results LR   # Figure 2 + Table 3, word+char LR
    python code/wordclean.py <path-to>/Dataset              # word-only ablation (Section 6.1)

## Transformer + LLM experiments (GPU, still to run)
Open code/hindi_hostility_transformers_llms_v2.ipynb in Google Colab, set runtime to T4 GPU, run all.
If Part B runs out of GPU memory, replace its cell with code/partB_llm_cell_memory_safe.py (smaller batches, resumes finished runs).
Outputs: transformer_coarse.csv, transformer_fine.csv (in results/) and llm_results.csv (copy it here from your Drive folder MyDrive/hostility).

results/ holds every number used in the paper.

## Fairness fixes (classical)
    python code/classical_v2.py <path-to>/Dataset results raw            # fine-grained with per-label thresholds tuned on validation
    python code/classical_v2.py <path-to>/Dataset results "no URLs/mentions/hashtags"   # same for cleaned-text variants
    python code/timing.py <path-to>/Dataset results                      # end-to-end inference time incl. TF-IDF, median of 5

## Counterfactual URL test and second dataset
    python code/url_injection.py <path-to>/Dataset results     # add/remove links in single test posts, classical models
    python code/second_dataset.py TRAIN.tsv TEST.tsv TEXT_COL LABEL_COL POS_LABEL results   # e.g. HASOC 2019 Hindi
    code/hindi_hostility_v3_seeds_urltest.ipynb                 # Colab: 2 extra seeds for mBERT/MuRIL + URL test for transformers
    python code/sigtest.py <path-to>/Dataset results/v3_preds results   # McNemar + paired bootstrap, transformer vs classical
    python code/second_dataset.py MACD/dataset/hindi_train.csv MACD/dataset/hindi_test.csv text label 0 results/macd MACD/dataset/hindi_val.csv   # control dataset (github.com/ShareChatAI/MACD)
