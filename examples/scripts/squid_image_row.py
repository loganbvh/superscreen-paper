"""
This script uses SuperScreen to simulate a single row of a SQUID susceptometry
image. The script can be called from squid_image.sbatch in order to simulate
many rows in parallel using a slurm job array.
"""

import os
import sys
import time
import logging

import numpy as np

import superscreen as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import squids


def mirror_layers(device, about=0, in_place=False):
    new_layers = []
    for layer in device.layers_list:
        new_layer = layer.copy()
        new_layer.z0 = about - layer.z0
        new_layers.append(new_layer)
    if not in_place:
        device = device.copy()
    device.layers_list = new_layers
    return device


def flip_device(device, about_axis="y"):
    device = device.copy(with_arrays=False)
    assert about_axis in "xy"
    index = 0 if about_axis == "y" else 1
    polygons = list(device.polygons.values())
    for polygon in polygons:
        polygon.points[:, index] *= -1
    return device


def update_origin(device, x0=0, y0=0):
    device = device.copy(with_arrays=True, copy_arrays=True)
    polygons = list(device.polygons.values())
    p0 = np.array([[x0, y0]])
    for polygon in polygons:
        polygon.points += p0
    if getattr(device, "points", None) is not None:
        device.points += p0
    return device


def sample_applied_field(x, y, z, fc_solution=None, field_units="mT"):
    x, y = np.atleast_1d(x, y)
    f = fc_solution.field_at_position(
        np.stack([x, y], axis=1),
        zs=z,
        units=field_units,
        with_units=False,
    )
    return f


def squid_applied_field(x, y, z, sample_solution=None, field_units="mT"):
    x, y = np.atleast_1d(x, y)
    f = sample_solution.field_at_position(
        np.stack([x, y], axis=1),
        zs=z,
        units=field_units,
        with_units=False,
        return_sum=True,
    )
    return f


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5000,
        help="Minimum number of points to use in the two SQUID meshes.",
    )
    parser.add_argument(
        "--optimesh-steps",
        type=int,
        default=None,
        help="Number of optimesh steps to perform.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of solver iterations to perform.",
    )
    parser.add_argument(
        "--squid-height",
        type=float,
        default=0,
        help="Relative distance between the two SQUIDs",
    )
    parser.add_argument(
        "--x-range",
        type=str,
        help="start, stop for x axis in microns",
    )
    parser.add_argument(
        "--y-range",
        type=str,
        help="start, stop for y axis in microns",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=1,
        help="Number of rows in the image (ignored if run via a slurm job array).",
    )
    args = parser.parse_args()

    field_units = "mT"
    outdir = args.outdir
    squid_height = args.squid_height
    min_points = args.min_points
    x_range = args.x_range
    y_range = args.y_range
    iterations = args.iterations
    optimesh_steps = args.optimesh_steps

    if "SLURM_ARRAY_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_ARRAY_JOB_ID"]
        array_id = os.environ["SLURM_ARRAY_TASK_ID"]
        num_rows = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    else:
        job_id = time.strfime("%y%m%d_%H%M%S")
        array_id = 0
        num_rows = args.num_rows

    outfile = os.path.join(outdir, f"{job_id}_{array_id}_image_squid.npz")

    logging.basicConfig(level=logging.INFO)

    x_range = [
        float(x.strip()) for x in x_range.replace("(", "").replace(")", "").split(",")
    ]
    y_range = [
        float(y.strip()) for y in y_range.replace("(", "").replace(")", "").split(",")
    ]

    xstart, xstop = x_range
    ystart, ystop = y_range
    pixel_size = (ystop - ystart) / num_rows

    xs = np.linspace(xstart, xstop, int(np.ceil((xstop - xstart) / pixel_size)))
    ys = np.linspace(ystart, ystop, int(np.ceil((ystop - ystart) / pixel_size)))

    squid = squids.medium.make_squid(align_layers="bottom")
    sample = squids.large.make_squid(align_layers="bottom")

    squid = flip_device(squid, about_axis="x")
    sample = flip_device(sample, about_axis="x")

    logging.info(squid)
    logging.info(sample)

    squid.make_mesh(min_points=min_points, optimesh_steps=optimesh_steps)
    sample.make_mesh(min_points=min_points, optimesh_steps=optimesh_steps)

    logging.info("Computing bare mutual inductance...")
    circulating_currents = {"fc_center": "1 mA"}
    I_fc = squid.ureg(circulating_currents["fc_center"])
    fc_solution = sc.solve(
        device=squid,
        circulating_currents=circulating_currents,
        iterations=iterations,
    )[-1]

    pl_fluxoid = sum(fc_solution.hole_fluxoid("pl_center", units="Phi_0"))
    m_no_sample = (pl_fluxoid / I_fc).to("Phi_0/A")
    logging.info(f"\tPhi = {pl_fluxoid:~.3fP}")
    logging.info(f"\tM = {m_no_sample:~.3fP}")

    sample_x0s = xs
    sample_y0 = ys[int(array_id)]

    pl_total_flux = []
    flux_part = []
    supercurrent_part = []
    for i, x0 in enumerate(sample_x0s):

        logging.info(
            f"({i + 1} / {len(xs)}) Solving for sample response to field coil..."
        )
        _sample = mirror_layers(sample, about=squid_height)
        _sample = update_origin(_sample, x0=x0, y0=sample_y0)

        applied_field = sc.Parameter(
            sample_applied_field,
            fc_solution=fc_solution,
            field_units=field_units,
        )

        sample_solution, _ = sc.find_fluxoid_solution(
            _sample,
            fluxoids={"pl_center": 0},
            applied_field=applied_field,
            field_units=field_units,
            iterations=iterations,
        )

        logging.info("\tSolving for squid response to sample...")
        applied_field = sc.Parameter(
            squid_applied_field,
            sample_solution=sample_solution,
            field_units=field_units,
        )

        solution = sc.solve(
            device=squid,
            applied_field=applied_field,
            circulating_currents=None,
            field_units=field_units,
            iterations=iterations,
        )[-1]
        logging.info("\tComputing pickup loop flux...")
        pl_flux = solution.polygon_flux(polygons="pl", units="Phi_0", with_units=False)[
            "pl"
        ]
        pl_total_flux.append(pl_flux)
        fluxoid = solution.hole_fluxoid("pl_center", units="Phi_0")
        flux_part.append(fluxoid.flux_part.magnitude)
        supercurrent_part.append(fluxoid.supercurrent_part.magnitude)
        logging.info(
            f"({i + 1} / {len(xs)}) mutual: "
            f"{(sum(fluxoid) / I_fc - m_no_sample).to('Phi_0 / A')}"
        )
        logging.info(f"({i + 1} / {len(xs)}) flux: {flux_part}")
        logging.info(f"({i + 1} / {len(xs)}) pl_total_flux: {pl_total_flux}")

    # Units: Phi_0
    pl_total_flux = np.array(pl_total_flux)
    flux = np.array(flux_part)
    supercurrent = np.array(supercurrent_part)
    # Units: Phi_0 / A
    mutual = (flux + supercurrent) / I_fc.to("A").magnitude
    data = dict(
        row=int(array_id),
        I_fc=I_fc.to("A").magnitude,
        current_units="A",
        pl_total_flux=pl_total_flux,
        flux=flux,
        supercurrent=supercurrent,
        flux_units="Phi_0",
        mutual=mutual,
        mutual_no_sample=m_no_sample.to("Phi_0 / A").m,
        mutual_units="Phi_0/A",
        xs=xs,
        ys=ys,
        y=sample_y0,
        length_units="um",
    )
    np.savez(outfile, **data)
    logging.info(f"Data saved to {outfile}.")
    logging.info("Done.")


if __name__ == "__main__":
    main()
