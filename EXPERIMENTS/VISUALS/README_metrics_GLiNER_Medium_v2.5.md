This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 61.78 |
| Recall     | 62.54 |
| F1         | 62.16 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 3).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 61.06     | 61.68  | 61.37     |
| 2    | 61.63     | 62.21  | 61.92     |
| 3    | 61.78     | 62.54  | 62.16     |
| 4    | 61.27     | 62.05  | 61.66     |
| 5    | 60.90     | 62.09  | 61.49     |

**Summary:**

- **F1:** mean = 61.72, std = 0.32  
- **Precision:** mean = 61.33, std = 0.37  
- **Recall:** mean = 62.12, std = 0.31  
