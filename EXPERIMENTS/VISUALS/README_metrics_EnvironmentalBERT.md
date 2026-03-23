This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 51.06 |
| Recall     | 43.10 |
| F1         | 46.74 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 3).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 51.74     | 42.48  | 46.66     |
| 2    | 49.62     | 42.28  | 45.66     |
| 3    | 51.06     | 43.10  | 46.74     |
| 4    | 50.68     | 42.44  | 46.20     |
| 5    | 48.61     | 40.60  | 44.25     |

**Summary:**

- **F1:** mean = 45.90, std = 1.02  
- **Precision:** mean = 50.34, std = 1.24  
- **Recall:** mean = 42.18, std = 0.93  
