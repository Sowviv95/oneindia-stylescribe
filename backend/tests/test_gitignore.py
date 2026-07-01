from pathlib import Path


def test_review_outputs_is_gitignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "review_outputs/" in gitignore
