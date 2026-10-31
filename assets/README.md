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

## Demo

The 55.6-second `remory.mp4` opens with **REMORY**, the paper title, and a rotating
anatomical brain illustration. It then uses the paper's method diagram and
recorded Qwen and GLM task traces, moving
from a paired Core War replay to nine tasks in a 3×3 grid,
covering coding and web research. Each task shows both runs side by side, with
matching context-chart axes and separate request, repeat, error, and compaction
counts. Repeated outputs are marked yellow and tool errors red.

The nine examples were selected for a correct residual result, at least 20%
fewer requests, and compaction in both runs. Request reductions range from
21% to 83%; baseline success is allowed. Eight cases also have no increase in
repeats, errors, or compactions. The chess case has 7 repeated outputs versus
4 in the baseline, shown as recorded. These selected cases do not represent an
unbiased sample or isolate the causal effect of memory. Replay progress follows
request indices, not elapsed time.

The aggregate charts are unchanged and use the full benchmark results from
the paper. Accuracy lines use a 25–100% right axis; error, repeat, and cost bars
retain a zero baseline. Final outcomes and chart reveals are edited for
presentation.

The singer-search panel shows a recorded 30-frame contact sheet used to inspect
candidate footage. That footage does not depict the singer ultimately identified.

The 3×3 grid transitions directly to the Terminal-Bench and BrowseComp results,
followed by the paper, code, and model links.

Music: [Electric Dreams](https://www.scottbuckley.com.au/library/electric-dreams/)
by [Scott Buckley](https://www.scottbuckley.com.au), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The excerpt is trimmed to 55.60 seconds, faded in and out, and level-adjusted.

Brain illustration and animation: adapted from the
[Harvard–Oxford atlas distributed by FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html).
Credit: David Kennedy, Christian Haselgrove, Bruce Fischl, Janis Breeze,
Jean Frazier, Larry Seidman, and Jill Goldstein. The adapted brain mesh and
animation are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Changes include a left-hemisphere cortical mesh, transparent cortex, TokyoNight
colors, 3D camera rotation, and motion-graphics composition. This is an anatomical
illustration, not an activation map.
