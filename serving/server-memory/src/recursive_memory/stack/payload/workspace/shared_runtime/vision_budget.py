"""CPU-only image admission estimates for the attested Qwen3.8 processor.

SGLang's Qwen3.5 image path delegates resizing to Qwen2VLImageProcessor(Fast).
Its local ``qwen_vl.smart_resize`` (factor 28) is used for *video*, not these
still images. Image token count is ``grid_t * grid_h * grid_w / merge_size**2``;
a still image repeats its one frame to temporal_patch_size, so grid_t is 1.

Only the geometry of immutable image receipts is inspected; no pixels are
resized, dropped, sent to a server, or fetched from network URLs. Unknown
processors/images retain the legacy large reservation, with visible fallback
telemetry instead of pretending that a precise estimate was obtained.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import stat
import warnings

try:
    from PIL import Image
except ImportError:  # Text-only workers must not acquire a new hard dependency.
    Image = None


LEGACY_IMAGE_RESERVE = 16384
IMAGE_MARKER_TOKENS = 2
IMAGE_SAFETY_TOKENS = 64
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_CACHE_ENTRIES = 256
POLICY = "qwen_vl_processor_geometry_with_margin_v1"


def _positive_int(value):
    return type(value) is int and value > 0


def _read_json(path: Path):
    with path.open("rb") as stream:
        data = stream.read(_MAX_CONFIG_BYTES + 1)
    if len(data) > _MAX_CONFIG_BYTES:
        raise ValueError("oversized processor configuration")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("processor configuration is not an object")
    return value, hashlib.sha256(data).hexdigest()


def _smart_resize(height, width, *, factor, min_pixels, max_pixels):
    """Exact arithmetic of transformers Qwen2-VL smart_resize (CPU only)."""
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("unsupported image aspect ratio")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@dataclass(frozen=True)
class ImageBudgetEstimate:
    reserved_tokens: int
    estimated_images: int
    fallback_images: int
    image_estimates: tuple[dict, ...]
    processor_attestation: dict

    def to_dict(self):
        return {
            "policy": POLICY,
            "reserved_tokens": self.reserved_tokens,
            "estimated_images": self.estimated_images,
            "fallback_images": self.fallback_images,
            "image_estimates": [dict(item) for item in self.image_estimates],
            "processor_attestation": dict(self.processor_attestation),
        }


class ImageTokenBudget:
    """Bounded immutable-content cache; instantiate once per backend session."""

    def __init__(self, *, geometry=None, attestation=None):
        self.geometry = geometry
        self.attestation = attestation or {"attested": False, "reason": "unknown_processor"}
        self._cache = OrderedDict()

    @classmethod
    def from_server_info(cls, server_info):
        """Attest the actual server model's on-disk still-image processor.

        This is local configuration attestation, not remote cryptographic
        attestation. Never infer it from a model display name or the client's
        unrelated processor defaults. Runtime image overrides are unsupported.
        """
        try:
            if not isinstance(server_info, Mapping):
                raise ValueError("missing_server_info")
            if server_info.get("language_only") is True or server_info.get("enable_multimodal") is False:
                raise ValueError("vision_disabled")
            model_path = Path(server_info.get("model_path", ""))
            if not model_path.is_absolute():
                raise ValueError("missing_local_model_path")
            mm_config = server_info.get("mm_process_config")
            if mm_config is not None and (not isinstance(mm_config, Mapping) or mm_config.get("image")):
                raise ValueError("unsupported_image_runtime_override")
            if server_info.get("json_model_override_args") not in (None, "", "{}", {}):
                raise ValueError("unsupported_model_runtime_override")
            config, config_sha = _read_json(model_path / "config.json")
            processor, processor_sha = _read_json(model_path / "preprocessor_config.json")
            if config.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
                raise ValueError("unsupported_model_type")
            if processor.get("processor_class") != "Qwen3VLProcessor" or processor.get("image_processor_type") not in (
                "Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast",
            ):
                raise ValueError("unsupported_image_processor")
            if processor.get("do_resize", True) is not True:
                raise ValueError("image_resize_disabled")
            vision = config.get("vision_config", {})
            patch = processor.get("patch_size")
            merge = processor.get("merge_size")
            temporal = processor.get("temporal_patch_size")
            if not all(_positive_int(value) for value in (patch, merge, temporal)) or (
                patch != vision.get("patch_size") or merge != vision.get("spatial_merge_size")
                or temporal != vision.get("temporal_patch_size")
            ):
                raise ValueError("vision_geometry_mismatch")
            size = processor.get("size", {})
            # HF's constructor treats these legacy fields as size overrides.
            minimum = processor.get("min_pixels", size.get("shortest_edge"))
            maximum = processor.get("max_pixels", size.get("longest_edge"))
            if not _positive_int(minimum) or not _positive_int(maximum) or minimum > maximum:
                raise ValueError("invalid_resize_limits")
            geometry = dict(factor=patch * merge, min_pixels=minimum, max_pixels=maximum)
            return cls(geometry=geometry, attestation={
                "attested": True, "source": "server_model_path_processor_config",
                "model_config_sha256": config_sha, "preprocessor_config_sha256": processor_sha,
                "image_processor_type": processor["image_processor_type"],
                "patch_size": patch, "merge_size": merge, "temporal_patch_size": temporal,
                "min_pixels": minimum, "max_pixels": maximum,
                "marker_tokens": IMAGE_MARKER_TOKENS, "safety_tokens": IMAGE_SAFETY_TOKENS,
            })
        except (OSError, ValueError, TypeError, AttributeError, OverflowError) as exc:
            # Do not put file contents, private image paths or data URLs in receipts.
            reason = str(exc) if isinstance(exc, ValueError) and len(str(exc)) < 80 else type(exc).__name__
            return cls(attestation={"attested": False, "reason": reason})

    @staticmethod
    def _image_bytes(image):
        value = image.get("image_data") if isinstance(image, Mapping) else getattr(image, "image_data", image)
        if isinstance(value, bytes):
            data = value
        elif isinstance(value, str) and value.startswith("data:image/"):
            header, separator, body = value.partition(",")
            if not separator or not header.endswith(";base64"):
                raise ValueError("unsupported_data_url")
            if len(body) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
                raise ValueError("image_bytes_exceed_limit")
            data = base64.b64decode(body, validate=True)
        elif isinstance(value, str) and value.startswith("/"):
            # Only gateway-owned private immutable receipts should use paths.
            # Open nonblocking and require a regular file, never a pipe/device.
            fd = os.open(value, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_IMAGE_BYTES:
                    raise ValueError("unsupported_image_file")
                data = stream.read(MAX_IMAGE_BYTES + 1)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns,
                ):
                    raise ValueError("image_file_changed_during_read")
        else:
            raise ValueError("unsupported_image_receipt")
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image_bytes_exceed_limit")
        return data

    def _fallback(self, reason):
        # Respect a known larger image ceiling too. Unknown processors preserve
        # the former 16K assumption and explicitly report that it is unattested.
        tokens = LEGACY_IMAGE_RESERVE
        if self.geometry is not None:
            tokens = max(tokens, math.ceil(self.geometry["max_pixels"] / self.geometry["factor"] ** 2)
                + IMAGE_MARKER_TOKENS + IMAGE_SAFETY_TOKENS)
        return {"reserved_tokens": tokens, "estimated": False, "reason": reason}

    def _estimate_one(self, image):
        if self.geometry is None:
            return self._fallback("unattested_processor")
        if Image is None:
            return self._fallback("pillow_unavailable")
        try:
            data = self._image_bytes(image)
            digest = hashlib.sha256(data).hexdigest()
            cached = self._cache.get(digest)
            if cached is not None:
                self._cache.move_to_end(digest)
                return dict(cached)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as parsed:
                    if parsed.format not in ("PNG", "JPEG", "WEBP", "BMP") or getattr(parsed, "n_frames", 1) != 1:
                        raise ValueError("unsupported_or_animated_image")
                    width, height = parsed.size
                    if width * height > MAX_IMAGE_PIXELS:
                        raise ValueError("image_pixels_exceed_limit")
                    # Header-only inspection avoids allocating decoded rasters.
                    parsed.verify()
            resized_h, resized_w = _smart_resize(height, width, **self.geometry)
            visual_tokens = (resized_h // self.geometry["factor"]) * (resized_w // self.geometry["factor"])
            result = {
                "reserved_tokens": visual_tokens + IMAGE_MARKER_TOKENS + IMAGE_SAFETY_TOKENS,
                "visual_tokens": visual_tokens, "estimated": True, "sha256": digest,
                "width": width, "height": height, "resized_width": resized_w, "resized_height": resized_h,
            }
            self._cache[digest] = dict(result)
            if len(self._cache) > _MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)
            return result
        except (OSError, ValueError, TypeError, OverflowError, Warning, Image.DecompressionBombError) as exc:
            return self._fallback(type(exc).__name__)

    def estimate(self, images):
        rows = tuple(self._estimate_one(image) for image in images)
        estimated = sum(row["estimated"] for row in rows)
        return ImageBudgetEstimate(sum(row["reserved_tokens"] for row in rows), estimated,
            len(rows) - estimated, rows, dict(self.attestation))

    def reserve(self, images):
        return self.estimate(images).reserved_tokens
