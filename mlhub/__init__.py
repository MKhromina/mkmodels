# mypy: disable-error-code="attr-defined"
"""MLHubMK is a repository dedicated to machine learning models implemented from scratch."""

from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()
