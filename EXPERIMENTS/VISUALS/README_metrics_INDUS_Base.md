This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 53.65 |
| Recall     | 51.31 |
| F1         | 52.45 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 5).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 53.64     | 49.02  | 51.23     |
| 2    | 51.31     | 42.48  | 46.48     |
| 3    | 54.02     | 48.53  | 51.13     |
| 4    | 51.34     | 46.85  | 49.00     |
| 5    | 53.65     | 51.31  | 52.45     |

**Summary:**

- **F1:** mean = 50.06, std = 2.35  
- **Precision:** mean = 52.79, std = 1.35  
- **Recall:** mean = 47.64, std = 3.29  
