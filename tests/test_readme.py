from pathlib import Path

README = (Path(__file__).parents[1] / "README.md").read_text()


def test_readme_uses_a_generic_badge_for_the_zenodo_concept_doi():
    assert "https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21765976-blue" in README
