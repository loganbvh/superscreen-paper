import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import superscreen as sc

from squid import make_squid

plt.rcParams["font.size"] = 14
plt.rcParams["savefig.dpi"] = 150


if __name__ == "__main__":

    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_triangles",
        type=int,
        default=8000,
        help="Minimum number of trianges in the mesh.",
    )
    parser.add_argument(
        "--optimesh_steps",
        type=int,
        default=400,
        help="Number of optimesh steps to perform when making the mesh."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=9,
        help="Number of solver iterations to perform."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(os.pardir, "images", "small_squid")
    os.makedirs(imagedir, exist_ok=True)

    device = make_squid()
    device.layers["BE"].london_lambda = 1
    device.make_mesh(min_triangles=args.min_triangles, optimesh_steps=args.optimesh_steps)
    print(device)

    # Plot the device's mesh and polygons
    ax = device.plot_mesh(figsize=(6, 6), color="k", alpha=0.4)
    ax = device.plot_polygons(ax=ax, linewidth=4)
    npoints = device.points.shape[0]
    ntriangles = device.triangles.shape[0]
    ax.set_title(f"{device.name} mesh:\n{npoints} points, {ntriangles} triangles")
    ax.figure.savefig(
        os.path.join(imagedir, f"{device.name}_mutual_mesh.png"),
        bbox_inches="tight",
    )

    applied_field = sc.sources.ConstantField(0)
    circulating_currents = {"fc_center": "1 mA"}
    I_fc = device.ureg(circulating_currents["fc_center"]).to("A")

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=circulating_currents,
        field_units="mT",
        current_units="uA",
        iterations=args.iterations,
    )

    # Calculate and plot polygon flux vs. solve iteration
    records = [s.polygon_flux(units="Phi_0", with_units=False) for s in solutions]
    df = pd.DataFrame.from_records(records)
    df.index.name = "Iteration"
    df.columns.name = "Polygon flux [Phi_0]"
    print(df)

    pl_flux = df["pl_hull"].values[-1] * device.ureg("Phi_0")
    mutual0 = (pl_flux / I_fc).to("Phi_0 / A")
    print(f"Pickup loop flux = {pl_flux:.3e~P}")
    print(
        f"Mutual inductance = {mutual0:.3f~P} = "
        f"{mutual0.to('pH'):.3f~P}"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True)
    for col in df.columns:
        ys = df[col].values
        with np.errstate(divide="ignore"):
            ys = np.abs(np.diff(ys)[1:] / ys[1:-1])
        ax.plot(np.arange(len(ys)) + 1, ys, 'o--', lw=3, label=col)
    title = (
        "Fractional change in polygon flux vs. iteration"
        + f"\n$\\Phi$ = {pl_flux.m:.4f} $\\Phi_0$, $M_0$ = {mutual0.m:.2f} $\\Phi_0$ / A"
    )
    ax.set_title(title)
    ax.set_ylabel("$\\left|(\\Phi_{(i+1)} -\\Phi_{(i)}) / \\Phi_{(i)}\\right|$")
    ax.set_xlabel("Iteration, $i$")
    ax.set_xticks(np.arange(len(ys)) + 1)
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(
        os.path.join(imagedir, f"{device.name}_mutual_convergence.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(fig)

    # Plot the fields.
    fig, axes = solutions[-1].plot_fields(
        figsize=(9, 5),
        layers=["W2", "W1"],
        auto_range_cutoff=0.05,
    )
    for ax in axes:
        device.plot_polygons(ax=ax, color="w", linewidth=1, legend=False)
    fig.savefig(
        os.path.join(imagedir, f"{device.name}_fields.png"),
        bbox_inches="tight",
    )

    # Plot the currents.
    fig, axes = solutions[-1].plot_currents(
        figsize=(14, 5),
        units="mA/um",
        streamplot=True,
    )
    for ax in axes:
        device.plot_polygons(ax=ax, color="w", linewidth=1, legend=False)
    fig.savefig(
        os.path.join(imagedir, f"{device.name}_currents.png"),
        bbox_inches="tight",
    )