import shutil
from pathlib import Path
from typing import List

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["test"]

SRC = "."


def format_with_args(session: nox.Session, *args):
    session.run("autoflake", *args)
    session.run("isort", *args)
    session.run("black", *args)


MIN_PYTHON_VERSION = "3.14"


@nox.session(python=[MIN_PYTHON_VERSION])
def lint(session: nox.Session):
    """Runs linters and fixers"""
    try:
        session.run("poetry", "sync", "--with", "lint", external=True)
        session.run("poetry", "check", "--lock", external=True)
        session.run("pyright", SRC)
        session.run("flake8", SRC)
        format_with_args(session, SRC, "--check")
    except Exception:
        session.error(
            "linting has failed. Run 'make format' to fix formatting and fix other errors manually"
        )


@nox.session(python=[MIN_PYTHON_VERSION])
def format(session: nox.Session):
    """Runs linters and fixers"""
    session.run("poetry", "sync", "--only", "lint", external=True)
    format_with_args(session, SRC)


@nox.session(python=[MIN_PYTHON_VERSION])
def test(session: nox.Session):
    """Runs unit tests and doc tests"""
    session.run("poetry", "sync", "--only", "main, test", external=True)
    session.run("python", "-m", "nltk.downloader", "punkt_tab")
    session.run("pytest", *session.posargs)


@nox.session(python=False)
def clean(_: nox.Session):
    """Cleans up the project by removing unnecessary files and directories"""

    def remove_dir(directory_path: Path):
        if directory_path.is_dir():
            shutil.rmtree(directory_path)
            print(f"Removed: {directory_path}")

    def remove_recursively(pattern: str, exclude_dirs: List[str] | None = None):
        if exclude_dirs is None:
            exclude_dirs = []

        def is_excluded(file: Path) -> bool:
            return any(dir in str(file) for dir in exclude_dirs)

        for file in Path(".").rglob(pattern):
            if not is_excluded(file):
                remove_dir(file)

    remove_dir(Path(".nox"))
    remove_dir(Path("dist"))
    remove_recursively("__pycache__", exclude_dirs=[".venv"])
    remove_recursively(".pytest_cache", exclude_dirs=[".venv"])
