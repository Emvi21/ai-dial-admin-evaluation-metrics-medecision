"""VCR configuration for integration tests with cassette recording/replay."""
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse, urlunparse

import pytest
from vcr.persisters.filesystem import FilesystemPersister
from vcr.serialize import serialize

from tests.fixtures.dial import NORMALIZED_DIAL_HOST

ALLOWED_REQUEST_HEADERS: Set[str] = {
    "content-type",
    "api-key",
    "authorization",
    "host",
}
ALLOWED_RESPONSE_HEADERS: Set[str] = {"content-type"}


class LFFilesystemPersister(FilesystemPersister):
    """Custom persister that preserves LF line endings on all platforms.

    The default VCR YAML serializer already produces LF-only strings.
    However, Python's text mode on Windows converts \n to \r\n during write.
    This persister uses newline='' to prevent that conversion.
    """

    @staticmethod
    def save_cassette(cassette_path, cassette_dict, serializer):
        """Save cassette to disk preserving LF line endings."""
        data = serialize(cassette_dict, serializer)
        cassette_path = Path(cassette_path)

        cassette_folder = cassette_path.parent
        if not cassette_folder.exists():
            cassette_folder.mkdir(parents=True)

        # Use newline='' to prevent Python from converting \n to \r\n on Windows
        with cassette_path.open("w", newline="") as f:
            f.write(data)


def pytest_recording_configure(config, vcr):
    """Hook to register custom persister with VCR instance.

    Uses LFFilesystemPersister to ensure cassettes have LF line endings
    on all platforms (prevents CRLF on Windows).
    """
    vcr.register_persister(LFFilesystemPersister)


def _filter_headers(
    headers: Dict[str, str], allowed: Set[str]
) -> Dict[str, str]:
    """Return a new dict with only allowed (case-insensitive) headers."""
    allowed_lower = {h.lower() for h in allowed}
    return {k: v for k, v in headers.items() if k.lower() in allowed_lower}


def _is_dial_request(
    parsed_uri: Any, real_dial_host: Optional[str] = None
) -> bool:
    """Return True if the request is for the DIAL host (normalized or real)."""
    if parsed_uri.netloc == NORMALIZED_DIAL_HOST:
        return True
    if real_dial_host is not None and parsed_uri.netloc == real_dial_host:
        return True
    return False


def _before_record_request(
    request_obj: Any, real_dial_host: Optional[str]
) -> Any:
    parsed_uri = urlparse(request_obj.uri)
    if not _is_dial_request(parsed_uri, real_dial_host):
        return None
    if parsed_uri.netloc == real_dial_host:
        normalized = parsed_uri._replace(netloc=NORMALIZED_DIAL_HOST)
        request_obj.uri = urlunparse(normalized)
    request_obj.headers = _filter_headers(
        request_obj.headers,
        allowed=ALLOWED_REQUEST_HEADERS,
    )
    return request_obj


def _before_record_response(response_obj: Any) -> Any:
    if "headers" in response_obj:
        response_obj["headers"] = _filter_headers(
            response_obj["headers"], allowed=ALLOWED_RESPONSE_HEADERS
        )
    return response_obj


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR to filter sensitive data and normalize URLs in cassettes.

    This configuration applies to all tests using @pytest.mark.vcr.

    URL Normalization:
        - During recording: Uses real DIAL_URL from environment
        - In cassettes: Saves normalized URL (dial.test.example.com)
        - During replay: Matches against normalized URL

    This allows different developers to use different DIAL instances
    while keeping cassettes portable and consistent.

    Request Filtering:
        - Only DIAL requests are recorded
        - Other requests (NLTK, GitHub, etc.) pass through without recording
    """
    real_dial_url = os.getenv("DIAL_URL")
    real_dial_host = urlparse(real_dial_url).netloc if real_dial_url else None

    return {
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("api-key", "REDACTED"),
            ("host", NORMALIZED_DIAL_HOST),
        ],
        # Do not set record_mode here, because it will overwrite command-line
        # options like --record-mode=rewrite (see ../cassettes.md)
        # Default record_mode is 'none'.
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "body",
        ],
        "before_record_request": partial(
            _before_record_request, real_dial_host=real_dial_host
        ),
        "before_record_response": _before_record_response,
        "decode_compressed_response": True,  # Human-readable responses in cassettes
    }
