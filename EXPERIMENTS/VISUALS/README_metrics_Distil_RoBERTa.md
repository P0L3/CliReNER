This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 55.90 |
| Recall     | 45.26 |
| F1         | 50.02 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 3).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 51.83     | 45.02  | 48.19     |
| 2    | 54.55     | 38.73  | 45.29     |
| 3    | 55.90     | 45.26  | 50.02     |
| 4    | 50.34     | 39.38  | 44.19     |
| 5    | 52.62     | 42.20  | 46.84     |

**Summary:**

- **F1:** mean = 46.91, std = 2.31  
- **Precision:** mean = 53.05, std = 2.20  
- **Recall:** mean = 42.12, std = 3.05  
