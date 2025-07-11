# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The Jupyter notebook reproducibility crisis

    Jupyter enabled _literate_ scientific programming. Despite its popularity, some common headaches still exist.

    ## Has this ever happened to you?

    /// details | 👻 Hidden State 

    Jupyter notebooks can develop hidden state in many ways.

    <img src="public/hidden_state.png" width="80%" />
    **Figure from **J. F. Pimentel, L. Murta, V. Braganholo and J. Freire, "A Large-Scale Study About Quality and Reproducibility of Jupyter Notebooks," 2019 IEEE/ACM 16th International Conference on Mining Software Repositories (MSR), Montreal, QC, Canada, 2019, pp. 507-517, doi: 10.1109/MSR.2019.00077.


    ///


    /// details | 📜 Very Large Git Diffs 

    `.ipynb` files are stored as JSON. Images are `base64` encoded into the file by default. Git diffs are often incomprehensible.  
    <img src="public/large_diff.png" width="80%" />

    ///

    /// details | 🩹 Restart & Run All 

    Personally, I "Restart & Run All" before sharing/committing my Jupyter notebooks. Difficult to automate reproducible workflows. 
    <img src="public/restart_run_all.png" width="80%" />

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# marimo - A reactive Python notebook""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    From the marimo README: 

     **marimo** is a reactive Python notebook: run a cell or interact with a UI
    element, and marimo automatically runs dependent cells (or <a href="#expensive-notebooks">marks them as stale</a>), keeping code and outputs
    consistent. marimo notebooks are stored as pure Python (with first-class SQL support), executable as scripts,
    and deployable as apps.

     **Highlights**.

    - 🚀 **batteries-included:** replaces `jupyter`, `streamlit`, `jupytext`, `ipywidgets`, `papermill`, and more
    - ⚡️ **reactive**: run a cell, and marimo reactively [runs all dependent cells](https://docs.marimo.io/guides/reactivity.html) or <a href="#expensive-notebooks">marks them as stale</a>
    - 🖐️ **interactive:** [bind sliders, tables, plots, and more](https://docs.marimo.io/guides/interactivity.html) to Python — no callbacks required
    - 🐍 **git-friendly:** stored as `.py` files
    - 🛢️ **designed for data**: query dataframes, databases, warehouses, or lakehouses [with SQL](https://docs.marimo.io/guides/working_with_data/sql.html), filter and search [dataframes](https://docs.marimo.io/guides/working_with_data/dataframes.html)
    - 🤖 **AI-native**: [generate cells with AI](https://docs.marimo.io/guides/generate_with_ai/) tailored for data work
    - 🔬 **reproducible:** [no hidden state](https://docs.marimo.io/guides/reactivity.html#no-hidden-state), deterministic execution, [built-in package management](https://docs.marimo.io/guides/package_management/)
    - 🏃 **executable:** [execute as a Python script](https://docs.marimo.io/guides/scripts.html), parameterized by CLI args
    - 🛜 **shareable**: [deploy as an interactive web app](https://docs.marimo.io/guides/apps.html) or [slides](https://docs.marimo.io/guides/apps.html#slides-layout), [run in the browser via WASM](https://docs.marimo.io/guides/wasm.html)
    - 🧩 **reusable:** [import functions and classes](https://docs.marimo.io/guides/reusing_functions/) from one notebook to another
    - 🧪 **testable:** [run pytest](https://docs.marimo.io/guides/testing/) on notebooks
    - ⌨️ **a modern editor**: [GitHub Copilot](https://docs.marimo.io/guides/editor_features/ai_completion.html#github-copilot), [AI assistants](https://docs.marimo.io/guides/editor_features/ai_completion.html), vim keybindings, variable explorer, and [more](https://docs.marimo.io/guides/editor_features/index.html)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Some nice features of the editor: 

    marimo has many IDE features:

    - Variable explorer
    - Live docs
    - Basic profiling
    - Scratchpad
    - Terminal
    - vim keybindings
    - [and many more...](https://docs.marimo.io/guides/editor_features/)
    """
    )
    return


@app.cell
def _():
    apple = "🍍"  # Hmm, This doesn't seem right. Maybe it should be 🍎.
    return (apple,)


@app.cell(hide_code=True)
def _(apple, mo):
    mo.md(
        rf"""
    # marimo for reproducible science

    Just like Jupyter notebooks, marimo enables literate programming. Better yet, interpolating Python variables into Markdown is very easy. For example, here is my apple: {apple} 

    Reactivity in marimo notebooks ensures that there is **no hidden state**. Your notebook is **always** up-to-date.
    """
    )
    return


if __name__ == "__main__":
    app.run()
