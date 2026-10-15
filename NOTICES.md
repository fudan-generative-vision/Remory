# Third-party notices

`src/remory/models/layers.py`, `src/remory/models/compressor.py`, and
`src/remory/pyramid.py` are inference-focused adaptations of the Recursive Memory
implementation (MIT, copyright 2026 Recursive Memory contributors). The loader
supports the public Qwen and GLM memory releases. The runtime package excludes
training-loss code and training launchers. The separate `training/` directory
contains Qwen3.8-27B Stage I/II source snapshots and their shared dependencies;
provenance and checksums are recorded in `training/source-manifest.json`.

The compressor architecture incorporates work from
[SpecForge](https://github.com/sgl-project/SpecForge) (MIT; upstream provenance
commit `204299a02fa31e56b5d425661107ae6cf247e4c6`) and DFlash (MIT).
`src/remory/models/grouped_conv.py` derives from SpecForge's DFlash2 grouped
convolution at commit `ed64d275bac8a48e126adb2827368603566a3029`;
`src/remory/models/dflash2.py` contains the corresponding residual inference
adaptation. Their notices are retained in `LICENSES/SpecForge-MIT.txt` and
`LICENSES/DFlash-MIT.txt`. The GLM hook and frontier handling are inference-only
adaptations of the same local implementation. The package imports PyTorch and
Transformers. Three GLM processor components from Transformers 5.17.0 are retained
under `src/remory/backends/glm_processor_sources` for the shared serving runtime;
their Apache-2.0 headers are preserved and the license is in
`LICENSES/Transformers-Apache-2.0.txt`.
Deployment fetches a pinned SGLang checkout (Apache-2.0) and applies the
patch in `deploy/sglang.patch`. Its source revision and exact patch contents are
recorded in `src/remory/backends/sglang_recipe.json`; the Apache-2.0 license is
included in `LICENSES/SGLang-Apache-2.0.txt`. CUDA components are downloaded from
NVIDIA with checksum verification and retain their bundled license files. Model
weight licenses are separate from this code license.

Codex is an independent upstream project, licensed under Apache-2.0. The
launcher downloads its unmodified upstream binary and reads its bundled model
instructions at runtime. The Codex binary is not redistributed. The local compact
prompt from Codex CLI 0.149.0 is retained in
`training/context_compression/codex_compact_v0149.py` for training-summary
preparation, with its exact prompt checksum and upstream provenance. Its
Apache-2.0 license is included in `LICENSES/Codex-Apache-2.0.txt`. The
integration does not imply upstream endorsement. The provider and launcher
adapt the local Recursive Memory inference harness (MIT).

The demo soundtrack, [Electric Dreams](https://www.scottbuckley.com.au/library/electric-dreams/)
by Scott Buckley, is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The video uses a trimmed, faded, and level-adjusted excerpt. This music license
is separate from the repository's code license.
