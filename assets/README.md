# Assets

## Paper figures

- `remory.png`: the residual-compensation figure from the paper, rendered from
  `figures/iclr_residual_compensation.pdf` at 216 dpi. All four panels are retained.
- `benchmarks.png`: unchanged copy of `figures/headline_benchmarks.png` from the
  paper. GLM BrowseComp scores are 84.9 without and 89.0 with residual memory.

Source: [Remory preprint](https://huggingface.co/mocoV3/Qwen3.8-27B-REMORY-1.9B/tree/main/paper).

## Demo

The 93.5-second video opens with **REMORY — Reinventing agent memory** alongside
a rotating anatomical brain illustration. This combined opening goes directly
into the paper's method diagram and recorded Qwen and GLM task traces. It moves
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

The research segment, titled **Fully Automatic Research**, shows
**Qwen3.8-27B + REMORY-1.9B** exploring flow-matching drafters with DFlash as a
reference. It replays a 34-hour session through literature searches, code edits,
experiments, audits, and revisions to a 14-page working paper. A step rail follows
the recorded work, and branching panels replay eight actual `spawn_agent` calls:
seven returned reports and one child was stopped while running. Each branching
panel labels its own condensed event span.

The workflow includes a human-defined goal, assisted setup, and occasional
feedback. The snapshot is from October 13, 2026, at 15:38 UTC; research is ongoing,
the paper is a draft, and evaluation and correctness checks remain open. Session
time includes waiting and service restarts, rather than continuous inference.

Music: [Electric Dreams](https://www.scottbuckley.com.au/library/electric-dreams/)
by [Scott Buckley](https://www.scottbuckley.com.au), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The excerpt is trimmed to 93.50 seconds, faded in and out, and level-adjusted.

Brain illustration and animation: adapted from the
[Harvard–Oxford atlas distributed by FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html).
Credit: David Kennedy, Christian Haselgrove, Bruce Fischl, Janis Breeze,
Jean Frazier, Larry Seidman, and Jill Goldstein. The adapted brain mesh and
animation are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Changes include a left-hemisphere cortical mesh, transparent cortex, TokyoNight
colors, 3D camera rotation, and motion-graphics composition. This is an anatomical
illustration, not an activation map.
