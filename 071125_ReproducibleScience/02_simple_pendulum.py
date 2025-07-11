# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
#     "scipy==1.16.0",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.integrate import solve_ivp
    return mo, np, plt, solve_ivp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # A simple pendulum example

    marimo provides [several](https://docs.marimo.io/api/inputs/) interactive UI elements for exploratory simulations/analysis. There's also Vincent Warmerdam's [wigglystuff](https://github.com/koaning/wigglystuff) library with other fun widgets (these work in Jupyter too!)

    Here, we explore the dynamics of a simple pendulum on various planets in our solar system.
    """
    )
    return


@app.cell
def _():
    solar_system_g_values = {
        "Mercury": 3.7,
        "Venus": 8.87,
        "Earth": 9.81,
        "Mars": 3.71,
        "Jupiter": 24.79,
        "Saturn": 10.44,
        "Uranus": 8.87,
        "Neptune": 11.15,
    }
    return (solar_system_g_values,)


@app.cell(hide_code=True)
def _(mo, solar_system_g_values):
    # Create UI element for pendulum length
    pendulum_length = mo.ui.slider(
        start=0.1,
        stop=10.0,
        step=0.1,
        value=0.5,
        label="Pendulum Length (m): ",
        show_value=True,
        # Hmmm, maybe adding a debounce is a good idea.
    )

    simulation_time = mo.ui.number(
        start=1,
        stop=50.0,
        step=1,
        value=10,
        label="Simulation time (m): ",
    )
    simulation_planet = mo.ui.radio(
        options=solar_system_g_values, value="Earth", inline=True, label="Planet: "
    )
    mo.vstack([pendulum_length, simulation_time, simulation_planet])
    return pendulum_length, simulation_planet, simulation_time


@app.cell
def _(np, simulation_planet):
    # Define the equations of motion for the pendulum
    g = simulation_planet.value  # acceleration due to gravity (m/s^2)


    def pendulum_ode(t, y, length):
        theta, omega = y
        dydt = [omega, -g / length * np.sin(theta)]
        return dydt
    return (pendulum_ode,)


@app.cell
def _(np, pendulum_length, pendulum_ode, simulation_time, solve_ivp):
    # Initial conditions
    theta0 = np.pi / 4  # initial angle (radians)
    omega0 = 0.0  # initial angular velocity (rad/s)

    # Time span for the simulation
    t_span = (0, simulation_time.value)

    # Solve the ODEs using solve_ivp
    solution = solve_ivp(
        pendulum_ode,
        t_span,
        [theta0, omega0],
        args=(pendulum_length.value,),
        t_eval=np.linspace(t_span[0], t_span[1], 1000),
    )

    # solution.t, solution.y
    return (solution,)


@app.cell
def _(plt, simulation_time, solution):
    # Plot the results
    # Hmmm, maybe we can style this a bit better.
    fig, (top_ax, bottom_ax) = plt.subplots(
        2,
        1,
        layout="constrained",
        sharex=True,
    )

    # Plot angle over time
    top_ax.plot(solution.t, solution.y[0])
    top_ax.set(xlim=(0, simulation_time.value), title="Angle (rad)")
    # Plot angular velocity over time
    bottom_ax.plot(solution.t, solution.y[1], color="tab:orange")
    bottom_ax.set(xlabel="Time (s)", title="Angular Velocity (rad/s)")

    fig
    return


if __name__ == "__main__":
    app.run()
