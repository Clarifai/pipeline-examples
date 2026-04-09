"""
Image Preprocessor Module

Handles chart/graph image acquisition from multiple sources:
- URL download
- Local file path
- Base64-encoded string

Validates, decodes, and normalizes images for VLM consumption.
"""

import base64
import io
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger("chart-analysis-step.image_preprocessor")


class ImagePreprocessor:
    """Load and validate chart images from various input sources."""

    # Supported image MIME types and their magic bytes
    MAGIC_BYTES = {
        b"\xff\xd8\xff": "image/jpeg",
        b"\x89PNG": "image/png",
        b"GIF8": "image/gif",
        b"RIFF": "image/webp",  # WebP starts with RIFF
    }

    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB

    def load_image(
        self,
        image_url: str = "",
        image_base64: str = "",
        image_path: str = "",
    ) -> Tuple[bytes, str]:
        """
        Load a chart image from one of three sources.

        Args:
            image_url: URL to download the image from.
            image_base64: Base64-encoded image string.
            image_path: Local filesystem path to the image.

        Returns:
            Tuple of (image_bytes, mime_type).

        Raises:
            ValueError: If no valid source is provided or image is invalid.
        """
        if image_url:
            return self._from_url(image_url)
        elif image_base64:
            return self._from_base64(image_base64)
        elif image_path:
            return self._from_path(image_path)
        else:
            raise ValueError(
                "Provide at least one of: image_url, image_base64, or image_path"
            )

    def _from_url(self, url: str) -> Tuple[bytes, str]:
        """Download image from a URL."""
        import requests

        logger.info("Downloading image from: %s", url)
        headers = {
            "User-Agent": "ClarifaiChartAnalysis/1.0 (https://clarifai.com)"
        }
        try:
            resp = requests.get(url, timeout=60, stream=True, headers=headers)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to download image from {url}: {e}") from e

        image_bytes = resp.content
        if len(image_bytes) > self.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image too large: {len(image_bytes)} bytes "
                f"(max {self.MAX_IMAGE_SIZE})"
            )

        # Detect MIME from Content-Type header or magic bytes
        content_type = resp.headers.get("Content-Type", "")
        if "image/" in content_type:
            mime = content_type.split(";")[0].strip()
        else:
            mime = self._detect_mime(image_bytes)

        logger.info("Downloaded %d bytes (%s)", len(image_bytes), mime)
        return image_bytes, mime

    def _from_base64(self, b64_string: str) -> Tuple[bytes, str]:
        """Decode a base64-encoded image."""
        logger.info("Decoding base64 image (%d chars)", len(b64_string))

        # Strip optional data URI prefix: data:image/png;base64,...
        if b64_string.startswith("data:"):
            header, _, b64_string = b64_string.partition(",")
            # Extract MIME from header
            mime = header.split(":")[1].split(";")[0] if ":" in header else ""
        else:
            mime = ""

        try:
            image_bytes = base64.b64decode(b64_string)
        except Exception as e:
            raise ValueError(f"Invalid base64 string: {e}") from e

        if len(image_bytes) > self.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Decoded image too large: {len(image_bytes)} bytes"
            )

        if not mime:
            mime = self._detect_mime(image_bytes)

        logger.info("Decoded %d bytes (%s)", len(image_bytes), mime)
        return image_bytes, mime

    def _from_path(self, path: str) -> Tuple[bytes, str]:
        """Read an image from a local file path."""
        logger.info("Reading image from: %s", path)

        if not os.path.isfile(path):
            raise ValueError(f"Image file not found: {path}")

        file_size = os.path.getsize(path)
        if file_size > self.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image file too large: {file_size} bytes "
                f"(max {self.MAX_IMAGE_SIZE})"
            )

        with open(path, "rb") as f:
            image_bytes = f.read()

        mime = self._detect_mime(image_bytes)
        logger.info("Read %d bytes (%s)", len(image_bytes), mime)
        return image_bytes, mime

    def _detect_mime(self, data: bytes) -> str:
        """Detect MIME type from magic bytes."""
        for magic, mime in self.MAGIC_BYTES.items():
            if data[:len(magic)] == magic:
                return mime
        # Fallback — assume PNG
        logger.warning("Could not detect image type, assuming image/png")
        return "image/png"

    @staticmethod
    def get_image_dimensions(image_bytes: bytes) -> Optional[Tuple[int, int]]:
        """Return (width, height) of an image, or None on failure."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            return img.size
        except Exception:
            return None

    @staticmethod
    def resize_if_needed(
        image_bytes: bytes, max_dim: int = 2048
    ) -> bytes:
        """Resize image if any dimension exceeds max_dim, preserving aspect ratio."""
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size

            if w <= max_dim and h <= max_dim:
                return image_bytes

            ratio = min(max_dim / w, max_dim / h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)

            buf = io.BytesIO()
            fmt = img.format or "PNG"
            img.save(buf, format=fmt)
            logger.info("Resized image from %dx%d to %dx%d", w, h, *new_size)
            return buf.getvalue()
        except Exception as e:
            logger.warning("Could not resize image: %s", e)
            return image_bytes
