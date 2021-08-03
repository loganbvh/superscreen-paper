import os
import time
import logging

import ray
import numpy as np
import matplotlib.pyplot as plt

import superscreen as sc
import superscreen.geometry as geo


def susceptibility_analytical(z, Lambda, a=8.4):
    """See Table 1 of 10.1103/PhysRevB.85.224518 (arXiv:1204.3355).
    
    Args:
        z: Vertical distance between the sample and sensor.
        Lambda: Effective penetration depth of the sample,
            Lambda = london_lambda**2/thickness.
        a: Effective field coil radius = 8.4 microns from the paper.
    
    Returns:
        Normalized susceptibility signal (M - M_no_sample) / M_no_sample,
        where M is the field coil - pickup loop mutual inductance.
    """
    # In the above paper, Lambda = 2 * lambda_london**2 / thickness, i.e. Pearl length.
    Lambda_Pearl = 2 * Lambda
    zbar = z / a
    return -a / Lambda_Pearl * (1 - 2 * zbar / np.sqrt(1 + 4 * zbar ** 2))


def sensor_and_sample(
    *,
    field_coil_inner_radius,
    field_coil_outer_radius,
    field_coil_Lambda,
    pickup_loop_inner_radius,
    pickup_loop_outer_radius,
    pickup_loop_Lambda,
    sample_size,
    sample_Lambda,
    sample_z0=-1,
    length_units="um",
):
    layers = [
        sc.Layer("fc_layer", Lambda=field_coil_Lambda, z0=0.57),
        sc.Layer("pl_layer", Lambda=pickup_loop_Lambda, z0=0),
        sc.Layer("sample_layer", Lambda=sample_Lambda, z0=sample_z0)
    ]
    films = [
        sc.Polygon(
            "field_coil", layer="fc_layer", points=geo.circle(field_coil_outer_radius)
        ),
        sc.Polygon(
            "pickup_loop", layer="pl_layer", points=geo.circle(pickup_loop_outer_radius)
        ),
        sc.Polygon(
            "sample", layer="sample_layer", points=geo.square(sample_size)
        ),
    ]
    holes = [
        sc.Polygon(
            "fc_hole", layer="fc_layer", points=geo.circle(field_coil_inner_radius)
        ),
        sc.Polygon(
            "pl_hole", layer="pl_layer", points=geo.circle(pickup_loop_inner_radius)
        ),
    ]
    
    return sc.Device(
        "sensor_and_sample",
        layers=layers,
        films=films,
        holes=holes,
        length_units=length_units,
    )


def layer_updater(layer, z0=0):
    if layer.name == "sample_layer":
        layer.z0 = z0
    return layer


def sweep_sample_height(device, sample_heights, ncpus, solve_iterations):
    parallel_method = "ray" if ncpus else None
    # Sweep the sample height to calculate susceptibility vs.
    # sensor - sample distance.
    start_time = time.time()
    layer_update_kwargs = [dict(z0=z0) for z0 in sample_heights]

    if parallel_method == "ray":
        ray.init(num_cpus=ncpus)

    solutions, _ = sc.solve_many(
        device=device,
        parallel_method=parallel_method,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        circulating_currents=circulating_currents,
        iterations=solve_iterations,
        return_solutions=True,
        keep_only_final_solution=True,
    )

    stop_time = time.time()
    elapsed_seconds = stop_time - start_time

    if parallel_method == "ray":
        ray.shutdown()
        parallel_method_str = f"ray:{ncpus}"
    else:
        parallel_method_str = "None"

    return solutions, elapsed_seconds, parallel_method_str
        

if __name__ == "__main__":

    import argparse

    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(os.pardir, "images", "susceptometry")
    os.makedirs(imagedir, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "max_ncpus",
        type=int,
        help="Maximum number of CPUs to use. 0 means no parallelism.",
    )
    parser.add_argument(
        "sample_Lambda",
        type=float,
        help="Sample layer effective penetration depth in microns."
    )
    args = parser.parse_args()

    ncpus = range(args.max_ncpus + 1)

    sample_Lambda = args.sample_Lambda

    sample_heights = -np.linspace(1, 20, 20)

    device = sensor_and_sample(
        field_coil_inner_radius=5.5,
        field_coil_outer_radius=8.0,
        field_coil_Lambda=0.08**2 / 0.20,
        pickup_loop_inner_radius=1.7,
        pickup_loop_outer_radius=2.7,
        pickup_loop_Lambda=0.08**2 / 0.23,
        sample_size=30,
        sample_Lambda=sample_Lambda,
        length_units="um",
    )

    device.make_mesh(min_triangles=8000, optimesh_steps=400)

    ax = device.plot_mesh(figsize=(6, 6))
    ax = device.plot_polygons(ax=ax, color="k", legend=False)
    npoints = device.points.shape[0]
    ntriangles = device.triangles.shape[0]
    ax.set_title(f"{device.name} mesh:\n{npoints} points, {ntriangles} triangles")
    ax.figure.savefig(
        os.path.join(imagedir, f"{device.name}_mesh.png"),
        bbox_inches="tight",
    )

    # Define the field coil current
    circulating_currents = {"fc_hole": "1 mA"}
    I_fc = device.ureg(circulating_currents["fc_hole"])

    solve_iterations = 4

    # Calculcate field coil - pickup loop mutual inductance
    # when the sample is very far from the sensor
    device.layers["sample_layer"].z0 = -50
    solutions = sc.solve(
        device=device,
        circulating_currents=circulating_currents,
        iterations=solve_iterations,
    )

    flux_dict = solutions[-1].polygon_flux(with_units=True)

    mutual_no_sample = (flux_dict["pickup_loop"] / I_fc).to("Phi_0 / A")
    print("Field coil - pickup loop mutual inductance with no sample:")
    print(f"\t{mutual_no_sample:.3f~P}")
    mutual_no_sample = mutual_no_sample.magnitude

    susc_analytical = susceptibility_analytical(
        -sample_heights,
        sample_Lambda,
        a=(5.5 + 8.0) / 2,
    )

    report = []

    lines = [f"num_cpus, elapsed_seconds"]

    for ncpu in ncpus:

        solutions, elapsed_seconds, parallel_method = sweep_sample_height(
            device, sample_heights, ncpu, solve_iterations,
        )
        mutual = []
        for solution in solutions:
            flux_dict = solution.polygon_flux(with_units=True)
            mutual.append((flux_dict["pickup_loop"] / I_fc).to("Phi_0 / A").magnitude)
        mutual = np.array(mutual)

        susc = (mutual - mutual_no_sample) / mutual_no_sample

        title = [
            "Normalized susceptibility vs. analytical expression",
            f" ($\\Lambda$ = {sample_Lambda:.1f} $\\mu$m)"
            f"\nparallel_method = {parallel_method}",
        ]
        title.append(f"\nElapsed time: {elapsed_seconds:.3f} seconds")
        title = "".join(title)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True)
        ax.plot(-sample_heights, susc_analytical, "-", label="Analytical")
        ax.plot(-sample_heights, susc, "s", label="Simulated")
        ax.set_xlabel("Sensor-sample distance [$\\mu$m]")
        ax.set_ylabel("Normalized susceptibility")
        ax.set_title(title)
        ax.legend(loc="best")

        ax.figure.savefig(
            os.path.join(imagedir, f"{device.name}_{parallel_method}_sweep_height.png"),
            bbox_inches="tight",
        )
        report.append(title)

        lines.append(f"{ncpu}, {elapsed_seconds}")

    print("\n\n".join(report))

    with open("result.csv", "w") as f:
        f.writelines(lines)
