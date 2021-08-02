import os
import logging

import numpy as np

import superscreen as sc

from plot_solutions import plot_solutions


def make_disk(Lambda=1, radius=5, length_units="um"):
    layers = [
        sc.Layer("base", Lambda=Lambda, z0=0),
    ]
    films = [
        sc.Polygon(
            "disk",
            layer="base",
            points=sc.geometry.circle(radius, points=401),
        ),
    ]
    abstract_regions = [
        sc.Polygon(
            "bounding_box",
            layer="base",
            points=sc.geometry.square(2 * (radius + 1), points_per_side=11),
        ),
    ]
    disk = sc.Device(
        "disk",
        layers=layers,
        films=films,
        abstract_regions=abstract_regions,
        length_units=length_units,
    )
    return disk


def Lambda_linear_ramp(x, y, xmin=0, xmax=1, Lambda0=0, Lambda1=0):
    Lambda = Lambda0 + (Lambda1 - Lambda0) * ((x - xmin) / (xmax - xmin))
    return Lambda


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(os.pardir, "images", "disk")
    os.makedirs(imagedir, exist_ok=True)

    disk_radius = 5
    length_units = "um"
    # Plot the fields and currents for this value of Lambda
    show_Lambda = 1.0

    # Make the disk Device
    disk = make_disk(radius=disk_radius, length_units=length_units)
    disk.make_mesh(min_triangles=8000, optimesh_steps=100)

    # Plot the disk's mesh and polygons
    ax = disk.plot_mesh(figsize=(6, 6))
    ax = disk.plot_polygons(ax=ax, color="k")
    npoints = disk.points.shape[0]
    ntriangles = disk.triangles.shape[0]
    ax.set_title(f"Mesh: {npoints} points, {ntriangles}")
    ax.figure.savefig(
        os.path.join(imagedir, f"{disk.name}_mesh.png"),
        bbox_inches="tight",
    )

    # Define a uniform applied field.
    applied_field = sc.sources.ConstantField(1)
    field_units = "mT"

    # Simulate screening of a uniform applied magnetic field
    # over a wide range of spatially-uniform effective
    # penetration depths (Lambdas).
    Lambdas = np.logspace(-1, 1, 5)
    solutions = []
    for Lambda in Lambdas:
        # Update the film's effective penetration depth, Lambda
        disk.layers["base"].Lambda = Lambda
        # Solve for the disk's magnetic response
        solutions.append(
            sc.solve(
                device=disk,
                applied_field=applied_field,
                field_units=field_units,
            )[-1]
        )

    fig, axes = plot_solutions(solutions, Lambdas, show_Lambda=show_Lambda)
    fig.savefig(
        os.path.join(imagedir, f"{disk.name}_uniform_field.png"),
        bbox_inches="tight",
    )

    # Simulate screening of a uniform applied magnetic field
    # by a disk with a spatially-nonuniform Lambda.

    # Lambda will vary linearly between Lambda0 and Lambda1 as you
    # go from xmin to xmax.
    Lambda = sc.Parameter(
        Lambda_linear_ramp,
        xmin=-disk_radius,
        xmax=disk_radius,
        Lambda0=1,
        Lambda1=10,
    )
    disk.layers["base"].Lambda = Lambda
    solution = sc.solve(
        device=disk,
        applied_field=applied_field,
        field_units=field_units,
    )[-1]

    # Plot the fields.
    fig, axes = solution.plot_fields(
        figsize=(5, 6),
        cross_section_ys=[-2, 0, 2],
        auto_range_cutoff=0.1,
    )
    fig.savefig(
        os.path.join(imagedir, f"{disk.name}_fields_nonuniform_lambda.png"),
        bbox_inches="tight",
    )

    # Plot the currents.
    fig, axes = solution.plot_currents(
        figsize=(5, 6),
        cross_section_ys=[-2, 0, 2],
        units="mA/um",
        auto_range_cutoff=0.1,
    )
    fig.savefig(
        os.path.join(imagedir, f"{disk.name}_currents_nonuniform_lambda.png"),
        bbox_inches="tight",
    )
