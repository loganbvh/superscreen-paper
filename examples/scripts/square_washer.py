import os
import logging

import numpy as np

import superscreen as sc

from plot_solutions import plot_solutions


def make_washer(inner_radius=2.5, outer_radius=5, Lambda=1, length_units="um"):
    layers = [
        sc.Layer("base", Lambda=Lambda, z0=0),
    ]
    films = [
        sc.Polygon(
            "washer",
            layer="base",
            points=sc.geometry.circle(outer_radius),
        ),
    ]
    holes = [
        sc.Polygon(
            "hole",
            layer="base",
            points=sc.geometry.circle(inner_radius),
        ),
    ]
    abstract_regions = [
        sc.Polygon(
            "bounding_box",
            layer="base",
            points=sc.geometry.circle(outer_radius * 1.4, points=21),
        )
    ]
    washer = sc.Device(
        "washer",
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
        length_units=length_units,
    )
    return washer


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(os.pardir, "images")

    inner_radius = 2.5
    outer_radius = 5.0
    Lambda = 1
    length_units = "um"
    field_units = "mT"

    # Make the disk Device
    washer = make_washer(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        Lambda=Lambda,
        length_units=length_units,
    )
    washer.make_mesh(min_triangles=8000, optimesh_steps=400)

    # Simulate a current circulating around the hole in the washer.
    circulating_currents = {"hole": "1 mA"}

    solution = sc.solve(device=washer, circulating_currents=circulating_currents)[-1]
    fig, axes = solution.plot_fields(figsize=(6, 7), cross_section_ys=[0])
    fig.savefig(os.path.join(imagedir, f"{washer.name}_plot_fields.png"))

    fig, axes = solution.plot_currents(
        figsize=(6, 7), cross_section_ys=[0], units="mA/um"
    )
    fig.savefig(os.path.join(imagedir, f"{washer.name}_plot_currents.png"))

    # Perform a surface integral of the resulting fields to calculcate the
    # flux through the washer and the washer's self-inductance.
    flux_dict = solution.polygon_flux()
    # Self-inductance is the total flux through the washer
    # divided by the circulating current.
    I_circ = washer.ureg(circulating_currents["hole"])
    inductance = (flux_dict["washer"] / I_circ).to("pH")
    print(
        f"{washer.name} self-inductance: "
        f"{inductance:.3f~P} = {inductance.to('Phi_0/A'):.1f~P}"
    )

    # Simulate the field and current distributions for a given total
    # circulating current across a wide range of Lambdas.
    Lambdas = np.logspace(-2, 2, 11)
    solutions = []
    for Lambda in Lambdas:
        washer.layers["base"].Lambda = Lambda
        solutions.append(
            sc.solve(
                device=washer,
                circulating_currents=circulating_currents,
                field_units=field_units,
            )[-1]
        )

    fig, axes = plot_solutions(solutions, Lambdas)
    fig.savefig(os.path.join(imagedir, f"{washer.name}_circulating_current.png"))

    # Simulate the field and current distributions for a given
    # uniform applied field across a wide range of Lambdas,
    # assuming no trapped flux.
    applied_field = sc.sources.ConstantField(0.2)
    Lambdas = np.logspace(-2, 2, 11)
    solutions = []
    for Lambda in Lambdas:
        washer.layers["base"].Lambda = Lambda
        solutions.append(
            sc.solve(
                device=washer,
                applied_field=applied_field,
                circulating_currents=None,
                field_units=field_units,
            )[-1]
        )

    fig, axes = plot_solutions(solutions, Lambdas)
    fig.savefig(os.path.join(imagedir, f"{washer.name}_uniform_field.png"))
