# Assets

## Paper figures

- `remory.png`: the residual-compensation figure from the paper, rendered from
  `figures/iclr_residual_compensation.pdf` at 216 dpi. All four panels are retained.
- `benchmarks.png`: unchanged copy of `figures/headline_benchmarks.png` from the
  paper. GLM BrowseComp scores are 84.9 without and 89.0 with residual memory.

Source: [Remory preprint](https://huggingface.co/mocoV3/Qwen3.8-27B-REMORY-1.9B/tree/main/paper).

## ASI-Bench result

`asi-bench-result.pdf` is the ASI-Bench submission portal export captured on
October 31, 2026. It records a **29.3328** score for the Qwen3.8-27B + REMORY-1.9B
run, submitted on October 25, with **240 instances**, `sandbox=os`, and status
`scored`.

The October 31 news item compares this score with the published overall means
on the [official ASI-Bench leaderboard](https://asibench.apexin.ai/leaderboard/).
For each reference model, it uses the highest-scoring listed harness:
**Claude Code, Max effort**, averaged over three adopted rounds.

| Model | Published ASI-Bench score | Total parameters | Official model source |
| --- | ---: | ---: | --- |
| Kimi-K2.7 | 27.34 | 1T | [Moonshot model card](https://huggingface.co/moonshotai/Kimi-K2.7-Code#2-model-summary) |
| DeepSeek-V4-Pro | 24.98 | 1.6T | [DeepSeek release](https://api-docs.deepseek.com/news/news260424/) |
| MiMo-V2.5-Pro | 23.25 | 1T | [Xiaomi model release](https://mimo.mi.com/docs/en-US/updates/model) |

Scores are rounded to two decimal places from the
[official aggregate data](https://asibench.apexin.ai/data/leaderboard_preview.json).
Parameter counts refer to total model parameters.
