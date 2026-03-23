This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 58.86 |
| Recall     | 59.44 |
| F1         | 59.15 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 4).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 58.61     | 58.54  | 58.57     |
| 2    | 58.62     | 59.44  | 59.03     |
| 3    | 57.99     | 58.25  | 58.12     |
| 4    | 58.86     | 59.44  | 59.15     |
| 5    | 57.17     | 58.62  | 57.89     |

**Summary:**

- **F1:** mean = 58.55, std = 0.55  
- **Precision:** mean = 58.25, std = 0.68  
- **Recall:** mean = 58.86, std = 0.55  
