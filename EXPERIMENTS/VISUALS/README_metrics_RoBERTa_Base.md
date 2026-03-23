This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 55.33 |
| Recall     | 49.18 |
| F1         | 52.08 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 4).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 55.39     | 48.69  | 51.83     |
| 2    | 58.32     | 44.12  | 50.23     |
| 3    | 54.80     | 45.92  | 49.97     |
| 4    | 55.33     | 49.18  | 52.08     |
| 5    | 51.19     | 43.95  | 47.30     |

**Summary:**

- **F1:** mean = 50.28, std = 1.91  
- **Precision:** mean = 55.01, std = 2.54  
- **Recall:** mean = 46.37, std = 2.47  
