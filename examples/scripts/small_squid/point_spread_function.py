import os
import itertools
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import superscreen as sc

from squid import make_squid, split_layer

plt.rcParams["font.size"] = 14
plt.rcParams["savefig.dpi"] = 150


def vortex_field(x, y, z, x0=0, y0=0, z0=0, nPhi0=1):
    """Field from an isolated vortex.

    Args:
        x, y, z: Position coordinates.
        x0, y0, z0: Vortex position
        nPhi0: Number of flux quanta contained in the vortex.
    """
    xp = x - x0
    yp = y - y0
    zp = z - z0
    Hz0 = zp / (xp ** 2 + yp ** 2 + zp ** 2) ** (3 / 2) / (2 * np.pi)
    return nPhi0 * Hz0


def VortexField(x0=0, y0=0, z0=0, nPhi0=1):
    return sc.Parameter(vortex_field, x0=x0, y0=y0, z0=z0, nPhi0=nPhi0)


if __name__ == "__main__":

    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vortex_height",
        type=float,
        default=-2.0,
        help="Height of the vortex relative to the SQUID."
    )
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
        default=5,
        help="Number of solver iterations to perform."
    )
    parser.add_argument(
        "--max_thickness",
        type=float,
        default=None,
        help="Maximum thickness of Layers making up the pickup loop shield."
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=0.25,
        help="Linear pixel size in microns."
    )
    parser.add_argument(
        "--use_ray",
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(os.pardir, "images", "small_squid")
    os.makedirs(imagedir, exist_ok=True)

    device = make_squid()
    if args.max_thickness is not None:
        device = split_layer(device, "W2", max_thickness=args.max_thickness)
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
        os.path.join(imagedir, f"{device.name}_psf_mesh.png"),
        bbox_inches="tight",
    )

    # Vortex positions
    xs = np.arange(-1, 1 + args.pixel_size, args.pixel_size)
    ys = np.arange(-2.5, 1 + args.pixel_size, args.pixel_size)

    # Vertical location of the vortex relative to the SQUID
    z0 = args.vortex_height

    applied_fields = [
        VortexField(x0=x0, y0=y0, z0=z0) for x0, y0 in itertools.product(xs, ys)
    ]

    parallel_method = "ray" if args.use_ray else None

    arrays = device.get_arrays(dense=False)

    with tempfile.TemporaryDirectory() as directory:
        _, paths = sc.solve_many(
            device=device,
            applied_fields=applied_fields,
            field_units="Phi_0 / um**2",
            current_units="uA",
            iterations=args.iterations,
            parallel_method=parallel_method,
            return_solutions=False,
            keep_only_final_solution=True,
            directory=directory,
        )
        # Calculate and plot polygon flux vs. solve iteration
        records = []
        for path, vortex in zip(paths, applied_fields):
            solution = sc.Solution.from_file(os.path.join(directory, path))
            solution.device.set_arrays(arrays)
            flux_dict = solution.polygon_flux(units="Phi_0", with_units=False)
            flux_dict["x0"] = vortex.kwargs["x0"]
            flux_dict["y0"] = vortex.kwargs["y0"]
            flux_dict["z0"] = vortex.kwargs["z0"]
            records.append(flux_dict)

    df = pd.DataFrame.from_records(records).set_index(["x0", "y0", "z0"])
    df.index.name = "Vortex position"
    df.columns.name = "Polygon flux [Phi_0]"
    print(df)

    df.to_csv(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "vortex_image.csv",
        )
    )
