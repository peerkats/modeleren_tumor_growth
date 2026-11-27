"""
Zhang et al. (2022): https://doi.org/10.7554/eLife.76284

The growth rates below are their UNSCALED estimates. Their fitted model
multiplies both rates by 8, uses beta = 6, and models treatment by changing
carrying capacity. This script instead implements the assignment model.
It is an exploratory simulation, not a reproduction of their patient fits.

The initial populations come from Appendix 2, Table 4, patient P1003 at
initial treatment, divided by their K = 10000. They are model estimates.
alpha = beta = 1 follows the assignment. The treatment loss rate is an
illustrative assumption. The stopping rule is inspired by the trial's
half-baseline / baseline PSA rule, applied here directly to S + R.
"""

import argparse
from dataclasses import dataclass, replace

import numpy as np

#verandering
# 1. Model parameters: edit these to change the starting experiment.

POLICIES = ("Untreated", "Continuous", "Adaptive")
SIMULATION_DAYS = 1500
TIME_STEP = 0.02


@dataclass
class Parameters:
    rS: float = 0.0156       # Sensitive growth rate, per day
    rR: float = 0.0091       # Resistant growth rate, per day
    alpha: float = 1.0       # Effect of R on S's crowding
    beta: float = 1.0        # Effect of S on R's crowding
    treatment_rate: float = 0.03  # lambda in the equations, per day
    S0: float = 0.578298     # Initial sensitive population / K
    R0: float = 0.010532     # Initial resistant population / K
    stop_fraction: float = 0.5  # Stop at this fraction of the initial total


@dataclass
class Simulation:
    time: np.ndarray
    sensitive: np.ndarray
    resistant: np.ndarray
    treatment_on: np.ndarray
    days_on_treatment: np.ndarray

    @property
    def total(self):
        return self.sensitive + self.resistant


# 2. The differential equations and treatment rule.


def growth_rates(population, treatment_on, parameters):
    """Calculate dS/dt and dR/dt at the current population."""
    S, R = population
    p = parameters

    sensitive_growth = p.rS * S * (1 - S - p.alpha * R)
    resistant_growth = p.rR * R * (1 - p.beta * S - R)

    treatment_loss = p.treatment_rate * S if treatment_on else 0.0
    dS_dt = sensitive_growth - treatment_loss
    dR_dt = resistant_growth

    return np.array([dS_dt, dR_dt])


def choose_treatment(policy, treatment_on, total, initial_total, stop_fraction):
    """Keep the previous on/off state between the adaptive thresholds."""
    if policy == "Untreated":
        return False
    if policy == "Continuous":
        return True

    stop_threshold = initial_total * stop_fraction
    restart_threshold = initial_total

    if treatment_on and total <= stop_threshold:
        return False
    if not treatment_on and total >= restart_threshold:
        return True

    return treatment_on



def rk4_step(population, step_size, treatment_on, parameters):
    """Advance one step using fourth-order Runge-Kutta"""
    slope_start = growth_rates(population, treatment_on, parameters)

    midpoint = population + step_size * slope_start / 2
    slope_middle_1 = growth_rates(midpoint, treatment_on, parameters)

    midpoint = population + step_size * slope_middle_1 / 2
    slope_middle_2 = growth_rates(midpoint, treatment_on, parameters)

    endpoint = population + step_size * slope_middle_2
    slope_end = growth_rates(endpoint, treatment_on, parameters)

    average_slope = (
        slope_start + 2 * slope_middle_1 + 2 * slope_middle_2 + slope_end
    ) / 6
    return population + step_size * average_slope


def simulate(parameters, policy="Adaptive", days=SIMULATION_DAYS, dt=TIME_STEP):
    """Solve the model; switching times are resolved to approximately dt."""
    p = parameters
    if policy not in POLICIES:
        raise ValueError(f"Unknown treatment policy: {policy}")

    values = list(vars(p).values()) + [days, dt]
    if not np.isfinite(values).all():
        raise ValueError("All parameters and times must be finite.")
    if min(p.rS, p.rR, p.alpha, p.beta, p.treatment_rate, p.S0, p.R0) < 0:
        raise ValueError("Rates, competition coefficients and populations must be nonnegative.")
    if days <= 0 or dt <= 0 or not 0 < p.stop_fraction < 1:
        raise ValueError("Use positive days and dt, and a stop_fraction between 0 and 1.")
    if p.S0 + p.R0 == 0:
        raise ValueError("Start with a nonzero tumour population.")

    number_of_steps = int(np.ceil(days / dt))
    time = np.linspace(0, days, number_of_steps + 1)
    step_size = days / number_of_steps

    population = np.zeros((len(time), 2))
    population[0] = [p.S0, p.R0]
    treatment_history = np.zeros(len(time), dtype=bool)
    treatment_days = np.zeros(len(time))

    initial_total = p.S0 + p.R0
    treatment_on = policy != "Untreated"

    for step in range(len(time)):
        total = population[step].sum()
        treatment_on = choose_treatment(
            policy, treatment_on, total, initial_total, p.stop_fraction
        )
        treatment_history[step] = treatment_on

        if step == number_of_steps:
            break  

        population[step + 1] = rk4_step(
            population[step], step_size, treatment_on, p
        )
        treatment_days[step + 1] = treatment_days[step]
        if treatment_on:
            treatment_days[step + 1] += step_size

    # Do not hide numerical errors by forcing negative populations to zero.
    if not np.isfinite(population).all() or population.min() < -1e-10:
        raise ArithmeticError("Invalid numerical solution. Try a smaller time step.")

    return Simulation(
        time=time,
        sensitive=population[:, 0],
        resistant=population[:, 1],
        treatment_on=treatment_history,
        days_on_treatment=treatment_days,
    )


# 4. Visualisation. The numerical model above works without this class.


class SimulationViewer:
    """Keep the plots, controls and their callbacks together."""

    def __init__(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider

        self.parameters = Parameters()
        self.policy = "Adaptive"
        self.playing = False
        self.results = {}

        self.figure = plt.figure(figsize=(13, 9))
        grid = self.figure.add_gridspec(
            2, 2, left=0.075, right=0.97, bottom=0.35, top=0.9,
            hspace=0.48, wspace=0.28,
        )
        self.population_axes = self.figure.add_subplot(grid[0, 0])
        self.comparison_axes = self.figure.add_subplot(grid[0, 1])
        self.cell_axes = self.figure.add_subplot(grid[1, 0])
        self.treatment_axes = self.figure.add_subplot(grid[1, 1])
        self.figure.suptitle("Tumour competition: sensitive and resistant cells", fontsize=16)

        self.status = self.figure.text(0.075, 0.305, "", fontsize=10)

        self.day_slider = Slider(
            self.figure.add_axes([0.16, 0.255, 0.70, 0.025]),
            "Day", 0, SIMULATION_DAYS, valinit=0, valstep=1,
        )
        self.treatment_slider = Slider(
            self.figure.add_axes([0.16, 0.205, 0.32, 0.025]),
            "lambda / day", 0, 0.1, valinit=self.parameters.treatment_rate,
        )
        self.competition_slider = Slider(
            self.figure.add_axes([0.64, 0.205, 0.25, 0.025]),
            "beta", 0, 6, valinit=self.parameters.beta,
        )
        initial_total = self.parameters.S0 + self.parameters.R0
        self.resistance_slider = Slider(
            self.figure.add_axes([0.16, 0.155, 0.32, 0.025]),
            "Initial R (%)", 0, 30,
            valinit=100 * self.parameters.R0 / initial_total,
        )
        self.policy_buttons = RadioButtons(
            self.figure.add_axes([0.64, 0.065, 0.17, 0.105]), POLICIES, active=2,
        )
        self.play_button = Button(
            self.figure.add_axes([0.84, 0.09, 0.09, 0.045]), "Play",
        )
        self.apply_button = Button(
            self.figure.add_axes([0.49, 0.045, 0.095, 0.045]), "Apply",
        )



        self.cell_positions = np.array([
            (column + 0.5, row + 0.5)
            for row in range(10) for column in range(50)
        ])
        np.random.default_rng(42).shuffle(self.cell_positions)

        self.day_slider.on_changed(self.show_day)
        self.policy_buttons.on_clicked(self.select_policy)
        self.apply_button.on_clicked(self.run_experiments)
        self.play_button.on_clicked(self.toggle_play)
        self.run_experiments()

    def run_experiments(self, event=None):
        initial_total = self.parameters.S0 + self.parameters.R0
        resistant_fraction = self.resistance_slider.val / 100
        parameters = replace(
            self.parameters,
            treatment_rate=self.treatment_slider.val,
            beta=self.competition_slider.val,
            S0=initial_total * (1 - resistant_fraction),
            R0=initial_total * resistant_fraction,
        )
        for policy in POLICIES:
            self.results[policy] = simulate(parameters, policy)


        largest_total = max(result.total.max() for result in self.results.values())
        self.display_capacity = max(1.0, largest_total)
        self.population_per_dot = self.display_capacity / len(self.cell_positions)
        self.draw_plots()

    def draw_plots(self):
        result = self.results[self.policy]
        axes = self.population_axes
        axes.clear()
        axes.plot(result.time, result.sensitive, color="tab:blue", label="Sensitive S")
        axes.plot(result.time, result.resistant, "--", color="tab:red", label="Resistant R")
        axes.plot(result.time, result.total, ":", color="black", label="Total S + R")
        axes.set_title(self.policy)
        axes.legend(fontsize=9)

        axes = self.comparison_axes
        axes.clear()
        styles = {
            "Untreated": ("grey", ":"),
            "Continuous": ("tab:red", "--"),
            "Adaptive": ("tab:blue", "-"),
        }
        for policy, run in self.results.items():
            color, style = styles[policy]
            axes.plot(run.time, run.total, color=color, linestyle=style, label=policy)
        axes.set_title("Total population under each treatment policy")
        axes.legend(fontsize=9)

        for axes in (self.population_axes, self.comparison_axes):
            axes.set(
                xlabel="Time (days)", ylabel="Population / K",
                xlim=(0, SIMULATION_DAYS), ylim=(0, 1.05 * self.display_capacity),
            )
            axes.grid(alpha=0.2)

        axes = self.treatment_axes
        axes.clear()
        axes.step(result.time, result.treatment_on.astype(int), where="post", color="grey")
        axes.set(
            xlabel="Time (days)", ylabel="Treatment",
            xlim=(0, SIMULATION_DAYS), ylim=(-0.1, 1.1),
            yticks=[0, 1], yticklabels=["OFF", "ON"],
        )
        self.population_cursor = self.population_axes.axvline(0, color="grey", linewidth=1)
        self.treatment_cursor = self.treatment_axes.axvline(0, color="grey", linewidth=1)

        axes = self.cell_axes
        axes.clear()
        axes.set(xlim=(0, 50), ylim=(0, 10), aspect="equal")
        axes.axis("off")
        positions = self.cell_positions
        axes.scatter(positions[:, 0], positions[:, 1], s=12, color="0.9")
        self.sensitive_dots = axes.scatter([], [], s=14, color="tab:blue", marker="o")
        self.resistant_dots = axes.scatter([], [], s=14, color="tab:red", marker="s")
        axes.text(
            0.5, -0.2,
            f"One mark = {100 * self.population_per_dot:.3f}% of K; no spatial dynamics",
            transform=axes.transAxes, ha="center", fontsize=9,
        )
        self.show_day(self.day_slider.val)

    def show_day(self, day):
        result = self.results[self.policy]
        index = min(np.searchsorted(result.time, day), len(result.time) - 1)
        S = result.sensitive[index]
        R = result.resistant[index]

        sensitive_count = int(round(S / self.population_per_dot))
        resistant_count = int(round(R / self.population_per_dot))
        end = sensitive_count + resistant_count
        self.sensitive_dots.set_offsets(self.cell_positions[:sensitive_count])
        self.resistant_dots.set_offsets(self.cell_positions[sensitive_count:end])
        self.cell_axes.set_title(
            f"Day {day:.0f} | circles: sensitive | squares: resistant", fontsize=11,
        )
        self.population_cursor.set_xdata([day, day])
        self.treatment_cursor.set_xdata([day, day])

        treatment_label = "ON" if result.treatment_on[index] else "OFF"
        treatment_days = result.days_on_treatment[index]
        self.status.set_text(
            f"S = {S:.4f} K    R = {R:.4f} K    Total = {S + R:.4f} K    "
            f"Treatment: {treatment_label}    Days on treatment: {treatment_days:.1f}"
        )
        self.figure.canvas.draw_idle()

    def select_policy(self, policy):
        self.policy = policy
        self.draw_plots()

    def toggle_play(self, event):
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")
        if self.playing and self.day_slider.val >= SIMULATION_DAYS:
            self.day_slider.set_val(0)

    def advance_animation(self, frame):
        if not self.playing:
            return
        next_day = min(SIMULATION_DAYS, self.day_slider.val + 5)
        self.day_slider.set_val(next_day)
        if next_day == SIMULATION_DAYS:
            self.playing = False
            self.play_button.label.set_text("Play")


# 5. Checks against mathematical properties of the model.


def check_numerics():
    parameters = Parameters()
    for policy in POLICIES:
        coarse = simulate(parameters, policy, dt=TIME_STEP)
        fine = simulate(parameters, policy, dt=TIME_STEP / 2)
        # Every second fine time point matches a coarse time point.
        sensitive_error = np.max(np.abs(coarse.sensitive - fine.sensitive[::2]))
        resistant_error = np.max(np.abs(coarse.resistant - fine.resistant[::2]))
        error = max(sensitive_error, resistant_error)
        assert error < 0.002, f"Step-size check failed for {policy}: {error}"

        # These properties apply to the default alpha = beta = 1 case.
        assert coarse.total.max() <= 1 + 1e-9
        assert np.diff(coarse.resistant).min() >= -1e-12
        print(f"{policy}: maximum change after halving dt = {error:.3g}")

    # Without resistant cells or treatment, the solution is logistic.
    parameters = replace(parameters, R0=0, treatment_rate=0)
    result = simulate(parameters, "Untreated", days=200)
    exponential = np.exp(parameters.rS * result.time)
    exact = parameters.S0 * exponential / (1 + parameters.S0 * (exponential - 1))
    assert np.max(np.abs(result.sensitive - exact)) < 1e-8
    assert np.all(result.resistant == 0)

    # Check both adaptive thresholds and the memory between them.
    assert not choose_treatment("Adaptive", True, 0.3, 0.6, 0.5)
    assert choose_treatment("Adaptive", False, 0.6, 0.6, 0.5)
    assert choose_treatment("Adaptive", True, 0.4, 0.6, 0.5)
    assert not choose_treatment("Adaptive", False, 0.4, 0.6, 0.5)
    print("All numerical and treatment-rule checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Simulate competing tumour populations.")
    parser.add_argument("--save", metavar="PNG", help="Save a static figure")
    parser.add_argument("--check", action="store_true", help="Run numerical checks")
    arguments = parser.parse_args()

    if arguments.check:
        check_numerics()
        return

    import matplotlib
    if arguments.save:
        matplotlib.use("Agg")  # Saving does not require a graphical window.

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    viewer = SimulationViewer()
    if arguments.save:
        viewer.day_slider.set_val(500)
        viewer.figure.savefig(arguments.save, dpi=150)
        plt.close(viewer.figure)
    else:
        # Keep a reference so Matplotlib does not discard the animation.
        viewer.animation = FuncAnimation(
            viewer.figure, viewer.advance_animation, interval=50,
            cache_frame_data=False,
        )
        plt.show()


if __name__ == "__main__":
    main()
