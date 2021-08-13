import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import logging
import tempfile

import ray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import superscreen as sc

from .huber_squid import squid_with_sample

plt.rcParams["font.size"] = 14
plt.rcParams["savefig.dpi"] = 150


def layer_updater(layer, Lambda=None, z0=None):
    if layer.name == "sample_layer":
        if Lambda is not None:
            layer.Lambda = Lambda
        if z0 is not None:
            layer.z0 = z0
    return layer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_type",
        choices=["Lambda", "z0"],
        help="Parameter to sweep, either Lambda or z0."
    )
    parser.add_argument(
        "-n",
        "--num_points",
        type=int,
        default=5,
        help="Number of points in the sweep."
    )
    parser.add_argument(
        "-c",
        "--num_cpus",
        type=int,
        default=0,
        help="Number of CPUs to use. Set to 0 for serial execution."
    )
    parser.add_argument(
        "-f",
        "--figures",
        type=bool,
        default=True,
        help="Whether to save all figures."
    )
    parser.add_argument(
        "-s",
        "--sweep_num_cpus",
        type=bool,
        default=False,
        help="If True, then sweep the number of cpus from 0 to num_cpus.",
    )
    args = parser.parse_args()

    if args.sweep_num_cpus:
        num_cpus_list = list(range(args.num_cpus))
    else:
        num_cpus_list = [args.num_cpus]

    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(os.pardir, "images", "squid_susceptometry")
    os.makedirs(imagedir, exist_ok=True)
    
    squid = squid_with_sample()
    print(squid)
    squid.make_mesh(min_triangles=8500, optimesh_steps=400)

    if args.figures:
        ax = squid.plot_mesh(figsize=(8, 6), color="k", alpha=0.5)
        ax = squid.plot_polygons(ax=ax, linewidth=4)
        npoints = squid.points.shape[0]
        ntriangles = squid.triangles.shape[0]
        ax.set_title(f"{squid.name} mesh:\n{npoints} points, {ntriangles} triangles")
        ax.figure.savefig(
            os.path.join(imagedir, f"{squid.name}_mesh.png"),
            bbox_inches="tight",
            dpi=150,
        )

    # Current flowing in the field coil
    circulating_currents = {"fc_center": "1 mA"}
    I_fc = squid.ureg(circulating_currents["fc_center"]).to("A")

    # Calculcate M_0, the mutual inductance between field coil and pickup loop
    # in the absence of a superconducting sample.
    # Here we turn off the sample's diamagnetic response by setting Lambda
    # to a very large value.
    squid.layers["sample_layer"].Lambda = 1e5

    # Solve the model with many iterations so that we can check convergence.
    solutions = sc.solve(
        device=squid,
        circulating_currents=circulating_currents,
        iterations=11,
    )
    # Calculate and plot polygon flux vs. solve iteration
    records = []
    for s in solutions:
        records.append(s.polygon_flux(units="Phi_0", with_units=False))
    df = pd.DataFrame.from_records(records)
    df.index.name = "Iteration"
    print("Polygon flux (Phi_0):")
    print(df)

    pl_flux = df["pl_hull"].values[4] * squid.ureg("Phi_0")
    mutual0 = (pl_flux / I_fc).to("Phi_0 / A")
    print("Without sample:")
    print(f"\tPickup loop flux = {pl_flux:.3e~P}")
    print(
        f"\tMutual inductance = {mutual0:.3f~P} = "
        f"{mutual0.to('pH'):.3f~P}"
    )

    if args.figures:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True)
        for col in df.columns:
            ys = df[col].values
            ys = np.abs(np.diff(ys)[1:] / ys[1:-1])
            ax.plot(np.arange(len(ys)) + 1, ys, 'o--', lw=3, label=col)
        title = (
            "Fractional change in polygon flux vs. iteration"
            + f"\n$\\Phi =$ {pl_flux.m:.3f} $\\Phi_0$, $M_0$ = {mutual0.m:.0f} $\\Phi_0$ / A"
        )
        ax.set_title(title)
        ax.set_ylabel("$\\left|(\\Phi_{(i+1)} -\\Phi_{(i)}) / \\Phi_{(i)}\\right|$")
        ax.set_xlabel("Iteration, $i$")
        ax.set_xticks(np.arange(len(ys)) + 1)
        ax.set_yscale("log")
        ax.legend(loc=0)
        fig.tight_layout()
        fig.savefig(
            os.path.join(imagedir, f"{squid.name}_M0_convergence.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

    # All polygon flux values converge to around 1 percent by 3 iterations
    solve_iterations = 3

    squid.layers["sample_layer"].Lambda = 1  # micron

    if args.sweep_type == "Lambda":
        # Sample effective penetration depth in microns
        sweep_vals = Lambdas = np.linspace(1, 10, args.num_points)
        layer_update_kwargs = [dict(Lambda=Lambda, z0=None) for Lambda in Lambdas]
        plot_xlabel = "Sample $\\Lambda$ [$\\mu$m]"
        plot_title = "Susceptibility vs. sample $\\Lambda$"
        plot_fname = "susc_vs_Lambda"
    else:
        # Distance between sample and SQUID in microns
        sweep_vals = zs = np.linspace(1, 10, args.num_points)
        layer_update_kwargs = [dict(Lambda=None, z0=-z) for z in zs]
        plot_xlabel = "SQUID - sample distance, $z$ [$\\mu$m]"
        plot_title = "Susceptibility vs. SQUID - sample distance"
        plot_fname = "susc_vs_z0"

    # Save solutions to a temporary directory so as not to fill up
    # memory with Soltution objects.
    squid_arrays = squid.get_arrays(copy_arrays=False, dense=False)

    elapsed_times = []

    for num_cpus in num_cpus_list:

        parallel_method = "ray" if (num_cpus > 0) else None

        t0 = time.time()

        if parallel_method == "ray":
            ray.init(num_cpus=args.num_cpus)

        with tempfile.TemporaryDirectory() as directory:
            _, paths = sc.solve_many(
                device=squid,
                circulating_currents=circulating_currents,
                layer_updater=layer_updater,
                layer_update_kwargs=layer_update_kwargs,
                iterations=solve_iterations,
                directory=directory,
                return_solutions=False,
                keep_only_final_solution=True,
                parallel_method=parallel_method,
            )
            pl_flux = []
            for path, kwargs in zip(paths, layer_update_kwargs):
                solution = sc.Solution.from_file(path)
                solution.device.set_arrays(squid_arrays)
                flux_dict = solution.polygon_flux(units="Phi_0", with_units=False)
                pl_flux.append(flux_dict["pl_hull"])
                if args.figures:
                    if kwargs["Lambda"] is None:
                        Lambda = f"{squid.layers['sample_layer'].Lambda:.2f} $\\mu$m"
                        z0 = f"{kwargs['z0']:.2f} $\\mu$m"
                    else:
                        Lambda = f"{kwargs['Lambda']:.2f} $\\mu$m"
                        z0 = f"{squid.layers['sample_layer'].z0:.2f} $\\mu$m"
                    title = f"$\\Lambda$={Lambda}, $z$={z0}"
                    label = title.replace(
                            ".", "_"
                        ).replace(
                            ",", ""
                        ).replace(
                            " ", "__"
                        ).replace(
                            "$\\mu$", "u"
                        ).replace(
                            "$", ""
                        ).replace(
                            "\\", ""
                        )
                    fig, axes = solution.plot_currents(
                        figsize=(16, 4.5),
                        units="uA / um",
                        max_cols=4,
                        # cross_section_ys=[0],
                        streamplot=False,
                    )
                    for ax in axes.flat:
                        _ = squid.plot_polygons(ax=ax, color="w", linewidth=0.75, legend=False)
                    fig.suptitle(title)
                    fig.subplots_adjust(top=1.2)
                    fig.savefig(
                        os.path.join(imagedir, f"{squid.name}_currents_{label}.png"),
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close(fig)

                    fig, axes = solution.plot_fields(
                        figsize=(16, 4.5),
                        units="mT",
                        max_cols=4,
                        share_color_scale=True,
                        symmetric_color_scale=True,
                        # cross_section_ys=[0],
                    )
                    for ax in axes.flat:
                        _ = squid.plot_polygons(ax=ax, color="w", linewidth=0.75, legend=False)
                    fig.suptitle(title)
                    fig.subplots_adjust(top=1.2)
                    fig.savefig(
                        os.path.join(imagedir, f"{squid.name}_fields_{label}.png"),
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close(fig)

        if parallel_method == "ray":
            ray.shutdown()

        t1 = time.time()
        n = len(layer_update_kwargs)
        dt = t1 - t0
        elapsed_times.append(dt)

        if parallel_method == "ray":
            method = f"ray with num_cpus = {args.num_cpus}"
        else:
            method = "serial execution"
        print(
            f"Elapsed time for {n} models ({method}): {dt:.3f} seconds"
            f" ({dt / n:.3f} seconds per model)."
        )

        pl_flux = np.array(pl_flux) * squid.ureg("Phi_0")
        mutual = (pl_flux / I_fc - mutual0).to("Phi_0 / A")

        if args.figures:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(sweep_vals, mutual.magnitude, "o-", linewidth=4)
            ax.grid(True)
            ax.set_xlabel(plot_xlabel)
            ax.set_ylabel("Susceptibility [$\\Phi_0$ / A]")
            ax.set_title(plot_title)
            fig.savefig(
                os.path.join(imagedir, f"{squid.name}_{plot_fname}.png"),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig)

    if args.sweep_num_cpus:
        print(list(zip(num_cpus_list, elapsed_times)))
        import csv
        csv_file = os.path.join(os.path.dirname(__file__), "results.csv")
        with open(csv_file, "w") as f:
            writer = csv.writer(f)
            writer.writerows(zip(num_cpus_list, elapsed_times))
