"""Build teaching notebooks; execution is left to nbconvert."""

from pathlib import Path

import nbformat

BOOTSTRAP = '''from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display
plt.rcParams["figure.figsize"] = (8, 4)
'''


def write_notebook(
    path: Path, title: str, goal: str, why: str,
    chunks: list[tuple[str, str, str]], mistakes: list[str],
    takeaways: list[str], next_step: str, final_markdown: str = "",
) -> None:
    """Write ordered explanation/code/interpretation cells with kernel metadata."""
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [md(f"# {title}\n\n## Goal of this step\n\n{goal}"),
             md(f"## Why this matters\n\n{why}"),
             md("Find the project root and import display tools. Reusable logic lives in src."),
             code(BOOTSTRAP)]
    for explanation, source, interpretation in chunks:
        cells.extend([md(explanation), code(source), md(f"**Interpretation:** {interpretation}")])
    cells.append(md("## Common mistakes\n\n" + "\n".join(f"- {x}" for x in mistakes)))
    cells.append(md("## Key takeaways\n\n" + "\n".join(f"- {x}" for x in takeaways)
                    + f"\n\n## Next step\n\n{next_step}"))
    if final_markdown:
        cells.append(md(final_markdown))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3 (project venv)",
                              "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.11"}
    nbformat.write(nb, path)
