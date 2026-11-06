# Security

Do not report credentials in a public issue. Use GitHub's private security
advisory flow for this repository.

Load only trusted Compressor exports. The current residual export is a
checksum-pinned `compressor.pt`, not a safetensors file; it contains no target
model weights. HF tokens and search/document credentials must not be committed.

PRoot provides task compatibility, not a security boundary. Run benchmark
workers on dedicated machines/containers without unrelated sensitive files.
Interactive Codex uses an isolated account configuration and host workspace
approvals by default. `--unsafe` removes those protections. MinerU sends selected
documents to an external service; use it only for documents you may upload.
