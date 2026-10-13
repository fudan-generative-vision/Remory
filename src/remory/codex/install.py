"""Fetch the exact upstream Codex binary used by the Responses adapter."""
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = '0.149.0'
ARTIFACTS = {
    'x86_64': ('x86_64-unknown-linux-musl', '7368b2055ed02157fea2695bb9f5af3ee7b0e40c5a3bebc81dfc596704244cfd'),
    'aarch64': ('aarch64-unknown-linux-musl', '1cc3eb4c2fbab048c8afae0bebb1e54745f88d91e5249a448765d34a2a2ba9bb'),
}


def ensure_codex(cache, explicit=None):
    if explicit:
        import shutil
        found = shutil.which(str(explicit))
        binary = Path(found or explicit).expanduser().resolve()
    else:
        if platform.system() != 'Linux' or platform.machine() not in ARTIFACTS:
            raise RuntimeError('Use --codex with a Codex 0.149.0 binary for this platform')
        target, checksum = ARTIFACTS[platform.machine()]
        destination = Path(cache) / 'codex' / VERSION / target
        binary = destination / 'codex'
        if not binary.exists():
            destination.mkdir(parents=True, exist_ok=True)
            url = f'https://github.com/openai/codex/releases/download/rust-v{VERSION}/codex-{target}.tar.gz'
            print(f'[r-codex] Installing Codex {VERSION}...', flush=True)
            with tempfile.TemporaryDirectory(dir=destination) as td:
                archive = Path(td) / 'codex.tar.gz'
                urllib.request.urlretrieve(url, archive)
                with archive.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != checksum:
                        raise RuntimeError('Codex download checksum mismatch')
                with tarfile.open(archive) as bundle:
                    member = bundle.getmember('codex-' + target)
                    if not member.isfile():
                        raise RuntimeError('Codex archive does not contain the expected binary')
                    with bundle.extractfile(member) as source:
                        temporary = Path(td) / 'codex'
                        with temporary.open('wb') as output:
                            import shutil
                            shutil.copyfileobj(source, output)
                temporary.chmod(0o755)
                os.replace(temporary, binary)
    result = subprocess.run([str(binary), '--version'], capture_output=True, text=True, timeout=30)
    if result.returncode or result.stdout.strip() != 'codex-cli ' + VERSION:
        raise RuntimeError(f'r-codex requires Codex {VERSION}; got {result.stdout.strip()}')
    return binary
