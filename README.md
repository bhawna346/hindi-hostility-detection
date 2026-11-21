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
