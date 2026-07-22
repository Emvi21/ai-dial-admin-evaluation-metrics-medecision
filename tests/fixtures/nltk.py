"""NLTK fixtures that keep the downloader off the network during tests.

``aidial_rag_eval`` calls ``nltk.download("punkt_tab", quiet=True)`` on every
text segmentation. This fixture replaces ``nltk.download`` with a no-op to
prevent network access during tests.
The actual download of the punkt_tab data is done in the noxfile.py test session.
"""

import nltk
import pytest


def _noop_download(*_args, **_kwargs) -> bool:
    return True


@pytest.fixture(scope="session", autouse=True)
def block_nltk_download():
    """Blocks nltk network downloads for the session."""

    with pytest.MonkeyPatch.context() as m:
        m.setattr(nltk, "download", _noop_download)
        yield
