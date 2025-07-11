# Modern Python Tools for Reproducible Science

Reproducibility is a cornerstone of robust scientific research, yet achieving it in computational workflows can often be challenging. Modern Python tools can enable reproducible workflows by default. In this tutorial, we'll explore marimo notebooks, a next-generation reactive notebook environment that enables, by default, inherently reproducible code execution compared to traditional Jupyter notebooks. We will also discuss "uv", a pip replacement that's ~10-100x faster.

We will cover three notebooks in this tutorial:

- [Introduction to marimo](01_intro_to_marimo.py): Introduction to marimo notebooks. (Note that the images are not displayed when using a link. Run locally to see the images.)
- [A simple pendulum example](https://marimo.app/github.com/butler-julie/GDSVirtualTutorials/blob/main/071125_ReproducibleScience/02_simple_pendulum.py): A simple pendulum example. 
- [Embedding Visualizer](https://marimo.app/l/xalyyf): Visualize the embedding of the MNIST dataset (from [marimo examples](https://github.com/marimo-team/examples)).


You do not need to install anything to run the notebooks online. However, in the spirit of reproducibility, we recommend running the notebooks locally.

## Installation and Usage

You will need the "uv" package manager to run the tutorial locally. Follow installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Next, install marimo as a tool using the following command:

```bash
uv tool install marimo
```

This makes the `marimo` command available in your terminal everywhere. Then you can run the tutorial using the following command:

```bash
uvx marimo edit --sandbox <filename-or-link-to-notebook>
```

This will open a new browser window with the marimo notebook server running. You can then edit the notebook and run the cells.



