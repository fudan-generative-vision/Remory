# Third-party notices

Parts of the compressor inference and training implementations were adapted and modified
from [SpecForge](https://github.com/sgl-project/SpecForge), revision
`204299a02fa31e56b5d425661107ae6cf247e4c6` (MIT). The compressor geometry is
based on the public [DFlash](https://github.com/z-lab/dflash) model contracts
(MIT). Full notices are included in `LICENSES/`.

Historical source notices are retained in `LICENSES/` where applicable to shared
helpers. No BrowseComp dataset or grading workflow is shipped. The optional
`training/` source snapshot includes modified SpecForge compressor trainers and
their dependency closure under MIT; its source provenance and SHA256 hashes are
in `training/source-manifest.json`, with the SpecForge license in `training/LICENSE`.
Training data, summary caches, optimizer states and pretrained weights are not
included. Dataset/model access and their respective licenses remain separate.

Terminal-Bench integration targets
[Harbor](https://github.com/harbor-framework/harbor) 0.21.0
(Apache-2.0). The Codex adapter subclasses Harbor interfaces and ships a
modified local PRoot environment provider. The Apache-2.0 text is included as
`LICENSES/Harbor-Apache-2.0.txt`. Harbor task images and datasets are not
redistributed here.

Qwen checkpoints and benchmark datasets are downloaded separately. They retain
their publishers' licenses and terms. The residual Compressor is distributed
separately through the public Hugging Face repository documented in the README.

The patched SGLang source is distributed as a checksummed overlay under
`src/recursive_memory/stack/payload/sglang_overlay` with its Apache-2.0 LICENSE.
The exact public upstream base commit and modified-file hashes are recorded in
`payload/manifest.json`; no private submodule is required. The installer combines
that upstream revision with the overlay. SpecForge's MIT notice is also retained
at `payload/workspace/SPECFORGE_LICENSE`.

AutomationBench, Codex, Docling and MinerU are installed from their respective
upstreams and retain their own licenses. Provider API/data-access terms apply
separately from this repository's license.

The installer downloads CUDA compiler components from NVIDIA's official
redistributions and a checksum-pinned Ubuntu bubblewrap package when needed.
These binaries are not included in this repository and retain their upstream
licenses. CUDA component versions and checksums are in `payload/cuda-toolchains.json`;
the bubblewrap package's upstream notices are retained in its downloaded archive.
