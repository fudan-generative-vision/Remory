# Third-party notices

This standalone r-codex branch is derived from the official
[fudan-generative-vision/Remory](https://github.com/fudan-generative-vision/Remory)
repository, commit `8ee4b0ecf1f3765c5d32d2bc2d97cbef87789915`.
The provider and launcher originated in the Recursive Memory inference harness.
Their MIT copyright notices are preserved in `LICENSE`.
The REMORY model, residual-memory runtime, training and deployment components
are not included in this branch.

Codex is an independent upstream project licensed under Apache-2.0. The launcher
can download its unmodified, checksum-verified 0.149.0 release binary and reads
its bundled model instructions at runtime. The binary is not redistributed.
The compact instruction retained in `src/r_codex/provider.py` derives from the
public local compact prompt in Codex CLI 0.149.0. This does not reproduce
OpenAI's private remote compaction implementation. The license is retained in
`LICENSES/Codex-Apache-2.0.txt`. This integration does not imply upstream endorsement.

Optional MCP tools are installed separately and retain their respective licenses.
