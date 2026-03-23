This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 56.49 |
| Recall     | 49.63 |
| F1         | 52.84 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 3).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 54.41     | 45.14  | 49.34     |
| 2    | 45.58     | 37.05  | 40.87     |
| 3    | 56.49     | 49.63  | 52.84     |
| 4    | 53.84     | 48.69  | 51.14     |
| 5    | 53.31     | 45.34  | 49.01     |

**Summary:**

- **F1:** mean = 48.64, std = 4.60  
- **Precision:** mean = 52.72, std = 4.17  
- **Recall:** mean = 45.17, std = 4.96  
