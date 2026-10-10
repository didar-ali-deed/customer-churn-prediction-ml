"""Check that the learning artifacts are complete and executed without errors."""

import nbformat

from src.config import ROOT


def test_nine_executed_teaching_notebooks() -> None:
    """Require saved outputs and the requested lesson structure in all notebooks."""
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 9
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        for heading in ("Goal of this step", "Why this matters", "Common mistakes", "Key takeaways", "Next step"):
            assert heading in markdown, (path.name, heading)
        assert "from src." in "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                assert index > 0 and notebook.cells[index - 1].cell_type == "markdown"
                assert cell.execution_count is not None, path.name
                assert not any(output.output_type == "error" for output in cell.outputs), path.name
        if path.name[:2] in {"01", "03", "04", "06", "07", "08"}:
            assert any("image/png" in output.get("data", {})
                       for cell in notebook.cells if cell.cell_type == "code"
                       for output in cell.outputs), f"Missing inline plots in {path.name}"
