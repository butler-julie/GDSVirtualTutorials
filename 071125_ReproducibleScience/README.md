# Modern Python Tools for Reproducible Science

Reproducibility is a cornerstone of robust scientific research, yet achieving it in computational workflows can often be challenging. Modern Python tools can enable reproducible workflows by default. In this tutorial, we'll explore marimo notebooks, a next-generation reactive notebook environment that enables, by default, inherently reproducible code execution compared to traditional Jupyter notebooks. We will also discuss "uv", a pip replacement that's ~10-100x faster.


## Installation and Usage

You will need the "uv" package manager. Follow installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Next, install marimo as a tool using the following command:

```bash
uv tool install marimo
```

This makes the `marimo` command available in your terminal everywhere. Then you can run the tutorial using the following command:

```bash
uvx marimo edit --sandbox <filename-or-link-to-notebook>
```

This will open a new browser window with the marimo notebook server running. You can then edit the notebook and run the cells.