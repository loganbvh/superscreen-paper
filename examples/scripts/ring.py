import os
import logging

import numpy as np

import superscreen as sc

from plot_solutions import plot_solutions


def make_ring(
    inner_radius=2.5,
    outer_radius=5,
    Lambda=1,
    length_units="um",
    square=False,
):
    if square:
        outer_points = sc.geometry.square(outer_radius * 2, points_per_side=60)
        inner_points = sc.geometry.square(inner_radius * 2, points_per_side=20)
        bbox_points = sc.geometry.square(outer_radius * 2 * 1.4, points_per_side=5)
    else:
        outer_points = sc.geometry.circle(outer_radius)
        inner_points = sc.geometry.circle(inner_radius)
        bbox_points = sc.geometry.circle(outer_radius * 1.4, points=21)

    layers = [
        sc.Layer("base", Lambda=Lambda, z0=0),
    ]
    films = [
        sc.Polygon(
            "ring",
            layer="base",
            points=outer_points,
        ),
    ]
    holes = [
        sc.Polygon(
            "hole",
            layer="base",
            points=inner_points,
        ),
    ]
    abstract_regions = [
        sc.Polygon(
            "bounding_box",
            layer="base",
            points=bbox_points,
        )
    ]
    name = "square_ring" if square else "ring"
    ring = sc.Device(
        name,
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
        length_units=length_units,
    )
    return ring


def layer_updater(layer, Lambda=None):
    if layer.name == "base":
        layer.Lambda = Lambda
    return layer


def vortex_field(x, y, z, vortex_position=(0, 0, 0), nPhi0=1):
    """The field from an isolated vortex.

    Args:
        x, y, z: Position coordinates.
        vortex_position: (x, y, z) position of the vortex.
        nPhi0: Number of flux quanta contained in the vortex.

    Returns:
        Magnetic field in units of Phi_0 / (length_units)^2,
        where length_units are the units of x, y, z, etc.
    """
    xv, yv, zv = vortex_position
    xp = x - xv
    yp = y - yv
    zp = z - zv
    Hz0 = zp / (xp ** 2 + yp ** 2 + zp ** 2) ** (3 / 2) / (2 * np.pi)
    return nPhi0 * Hz0


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(os.pardir, "images", "ring")
    os.makedirs(imagedir, exist_ok=True)

    inner_radius = 2.5
    outer_radius = 5.0
    Lambda = 1
    length_units = "um"
    field_units = "mT"
    # Plot the fields and currents for this value of Lambda
    show_Lambda = 0.1

    # The method used for parallel execution.
    # Valid options are None (i.e. serial execution), "ray", and "multiprocessing".
    # Note that for models this quick to solve, you should not expect any significatnt
    # speedup from parallel processing (especially using "multiprocessing").
    parallel_method = None

    for square in (False, True):

        # Make the disk Device
        ring = make_ring(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            Lambda=Lambda,
            length_units=length_units,
            square=square,
        )
        ring.make_mesh(min_triangles=8000, optimesh_steps=400)

        ax = ring.plot_mesh(figsize=(6, 6))
        ax = ring.plot_polygons(ax=ax, color="k")
        npoints = ring.points.shape[0]
        ntriangles = ring.triangles.shape[0]
        ax.set_title(f"Mesh: {npoints} points, {ntriangles}")
        ax.figure.savefig(
            os.path.join(imagedir, f"{ring.name}_mesh.png"),
            bbox_inches="tight",
        )

        # Simulate a current circulating around the hole in the ring.
        circulating_currents = {"hole": "1 mA"}

        solution = sc.solve(device=ring, circulating_currents=circulating_currents)[-1]
        fig, axes = solution.plot_fields(
            figsize=(6, 7),
            cross_section_ys=[0],
            auto_range_cutoff=0.1,
        )
        fig.savefig(
            os.path.join(imagedir, f"{ring.name}_plot_fields.png"),
            bbox_inches="tight",
        )

        fig, axes = solution.plot_currents(
            figsize=(6, 7),
            cross_section_ys=[0],
            units="mA/um",
            auto_range_cutoff=0.1,
        )
        fig.savefig(
            os.path.join(imagedir, f"{ring.name}_plot_currents.png"),
            bbox_inches="tight",
        )

        # Perform a surface integral of the resulting fields to calculcate the
        # flux through the ring and the ring's self-inductance.
        flux_dict = solution.polygon_flux()
        # Self-inductance is the total flux through the ring
        # divided by the circulating current.
        I_circ = ring.ureg(circulating_currents["hole"])
        inductance = (flux_dict["ring"] / I_circ).to("pH")
        print(
            f"{ring.name} self-inductance: "
            f"{inductance:.3f~P} = {inductance.to('Phi_0/A'):.1f~P}"
        )

        # Simulate the field and current distributions for a given total
        # circulating current across a wide range of Lambdas.
        Lambdas = np.logspace(-2, 2, 13)
        assert np.any(Lambdas == show_Lambda), "show_Lambda not in Lambdas"

        layer_update_kwargs = [dict(Lambda=Lambda) for Lambda in Lambdas]

        solutions, _ = sc.solve_many(
            device=ring,
            parallel_method=parallel_method,
            circulating_currents=circulating_currents,
            layer_updater=layer_updater,
            layer_update_kwargs=layer_update_kwargs,
            return_solutions=True,
            keep_only_final_solution=True,
        )

        # # Using solve_many is equivalent to the loop below, except that
        # # with solve_many we have the option of solving models in parallel.
        # solutions = []
        # for Lambda in Lambdas:
        #     ring.layers["base"].Lambda = Lambda
        #     solutions.append(
        #         sc.solve(
        #             device=ring,
        #             circulating_currents=circulating_currents,
        #             field_units=field_units,
        #         )[-1]
        #     )

        fig, axes = plot_solutions(solutions, Lambdas, show_Lambda=show_Lambda)
        fig.savefig(
            os.path.join(imagedir, f"{ring.name}_circulating_current.png"),
            bbox_inches="tight",
        )

        # Simulate the field and current distributions for a given
        # uniform applied field across a wide range of Lambdas,
        # assuming no trapped flux.
        applied_field = sc.sources.ConstantField(0.2)

        solutions, _ = sc.solve_many(
            device=ring,
            parallel_method=parallel_method,
            applied_fields=applied_field,
            circulating_currents=None,
            layer_updater=layer_updater,
            layer_update_kwargs=layer_update_kwargs,
            return_solutions=True,
            keep_only_final_solution=True,
        )

        # # Using solve_many is equivalent to the loop below, except that
        # # with solve_many we have the option of solving models in parallel.
        # solutions = []
        # for Lambda in Lambdas:
        #     ring.layers["base"].Lambda = Lambda
        #     solutions.append(
        #         sc.solve(
        #             device=ring,
        #             applied_field=applied_field,
        #             circulating_currents=None,
        #             field_units=field_units,
        #         )[-1]
        #     )

        fig, axes = plot_solutions(solutions, Lambdas, show_Lambda=show_Lambda)
        fig.savefig(
            os.path.join(imagedir, f"{ring.name}_uniform_field.png"),
            bbox_inches="tight",
        )

        # Simulate the field and current distribution for a nonuniform applied field,
        # namely the field from several isolated vortcies pinned below
        # the plane of the ring.
        vortex_positions = [
            # (x, y, z),
            (-3.75, 0, -0.5),
            (2.5, -2.5, -0.5),
            (0, 5, -0.5),
        ]
        # Define the applied field as a sum of Parameters
        applied_field = sum(
            sc.Parameter(vortex_field, vortex_position=position)
            for position in vortex_positions
        )

        solutions, _ = sc.solve_many(
            device=ring,
            parallel_method=parallel_method,
            applied_fields=applied_field,
            circulating_currents=None,
            layer_updater=layer_updater,
            layer_update_kwargs=layer_update_kwargs,
            return_solutions=True,
            keep_only_final_solution=True,
        )

        # # Using solve_many is equivalent to the loop below, except that
        # # with solve_many we have the option of solving models in parallel.
        # solutions = []
        # for Lambda in Lambdas:
        #     ring.layers["base"].Lambda = Lambda
        #     solutions.append(
        #         sc.solve(
        #             device=ring,
        #             applied_field=applied_field,
        #             field_units="Phi_0 / um**2",
        #         )[-1]
        #     )

        fig, axes = plot_solutions(solutions, Lambdas, show_Lambda=show_Lambda)
        fig.savefig(
            os.path.join(imagedir, f"{ring.name}_nonuniform_field.png"),
            bbox_inches="tight",
        )
