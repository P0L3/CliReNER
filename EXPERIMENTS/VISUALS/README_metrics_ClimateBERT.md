This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 55.54 |
| Recall     | 45.63 |
| F1         | 50.10 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 3).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 50.38     | 42.97  | 46.38     |
| 2    | 46.34     | 36.44  | 40.80     |
| 3    | 55.54     | 45.63  | 50.10     |
| 4    | 51.94     | 47.47  | 49.61     |
| 5    | 50.80     | 44.36  | 47.36     |

**Summary:**

- **F1:** mean = 46.85, std = 3.72  
- **Precision:** mean = 51.00, std = 3.31  
- **Recall:** mean = 43.37, std = 4.22  
