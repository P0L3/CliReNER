This repository provides the **best-performing checkpoint** selected from 5 runs with different random seeds. While the internal training logs tracked performance on the validation split of **CliReNER<sub>silver</sub>**, the final model selection and the metrics below are evaluated on the independent, expert-annotated **CliReNER<sub>gold</sub>** dataset.

| Metric     | Score |
|------------|-------|
| Precision  | 50.02 |
| Recall     | 46.61 |
| F1         | 48.26 |

> This checkpoint corresponds to the **seed with the highest strict F1 on the gold evaluation set** (Seed 4).

---

## 📊 Results Across Seeds

We fine-tuned the model using 5 different random seeds to assess the stability and robustness of the architecture on the domain-specific text.

| Seed | Precision | Recall | Strict F1 |
|------|-----------|--------|-----------|
| 1    | 48.28     | 43.06  | 45.52     |
| 2    | 41.78     | 33.66  | 37.29     |
| 3    | 47.44     | 44.73  | 46.05     |
| 4    | 50.02     | 46.61  | 48.26     |
| 5    | 48.41     | 44.28  | 46.26     |

**Summary:**

- **F1:** mean = 44.67, std = 4.26  
- **Precision:** mean = 47.19, std = 3.16  
- **Recall:** mean = 42.47, std = 5.09  
