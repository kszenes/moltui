from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_release_please_publishes_created_release_to_pypi():
    workflow = (REPO_ROOT / ".github/workflows/release-please.yml").read_text()

    assert "id: release" in workflow
    assert "release_created: ${{ steps.release.outputs.release_created }}" in workflow
    assert "publish-pypi:" in workflow
    assert "needs.release-please.outputs.release_created == 'true'" in workflow
    assert "ref: ${{ needs.release-please.outputs.tag_name }}" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow


def test_release_workflow_can_publish_a_named_existing_tag():
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()

    assert "workflow_dispatch:" in workflow
    assert "tag:" in workflow
    assert "ref: ${{ inputs.tag || github.ref }}" in workflow
    assert "TAG_NAME: ${{ inputs.tag || github.ref_name }}" in workflow


def test_release_workflow_skips_pypi_for_prereleases():
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()

    assert "is_prerelease=" in workflow
    assert "if: needs.build.outputs.is_prerelease != 'true'" in workflow
