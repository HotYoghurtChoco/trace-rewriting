# Clean and MaxLoss AF on 100 held-out GSM8K questions

Each cell gives Clean / MaxLoss AF (%).

| Student | Seed 888 | Seed 889 | Seed 890 | Mean |
|---|---:|---:|---:|---:|
| Qwen3-1.7B-Base | 85 / 85 | 83 / 81 | 82 / 82 | 83.33 / 82.67 |
| Llama-3.2-3B | 23 / 20 | 19 / 17 | 25 / 18 | 22.33 / 18.33 |
| Qwen3-4B-Base | 40 / 39 | 42 / 34 | 46 / 55 | 42.67 / 42.67 |

The seven new runs used commit `795ae07`. The earlier seed-888 results come from [Step 2.3](../step2_seed888) and [Step 3.1](../step3_llama3b_seed888).
