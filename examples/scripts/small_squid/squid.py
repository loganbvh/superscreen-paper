import os

import numpy as np
from scipy.io import loadmat

from superscreen import Device, Layer, Polygon
from superscreen.geometry import rotate


def split_layer(device, layer_name, max_thickness=0.05):
    """Splits a given layer into multiple thinner layers."""
    layers = device.layers
    films = device.films
    holes = device.holes
    abstract_regions = device.abstract_regions
    
    layer_to_split = layers.pop(layer_name)
    london_lambda = layer_to_split.london_lambda
    z0 = layer_to_split.z0
    d = layer_to_split.thickness
    
    num_layers, mod = divmod(d, max_thickness)
    num_layers = int(num_layers)
    new_ds = [max_thickness for _ in range(num_layers)]
    if mod:
        num_layers += 1
        new_ds.append(mod)
    new_layers = {}
    for i, new_d in enumerate(new_ds):
        name = f"{layer_name}_{i}"
        z = i * max_thickness + new_d / 2
        new_layers[name] = Layer(
            name, london_lambda=london_lambda, thickness=new_d, z0=z
        )

    new_films = {}
    for name, film in films.items():
        if film.layer == layer_name:
            for i, new_layer_name in enumerate(new_layers):
                film_name = f"{name}_{i}"
                new_film = film.copy()
                new_film.name = film_name
                new_film.layer = new_layer_name
                new_films[film_name] = new_film
        else:
            new_films[name] = film
            
    new_holes = {}
    for name, hole in holes.items():
        if hole.layer == layer_name:
            for i, new_layer_name in enumerate(new_layers):
                hole_name = f"{name}_{i}"
                new_hole = hole.copy()
                new_hole.name = hole_name
                new_hole.layer = new_layer_name
                new_holes[film_name] = new_hole
        else:
            new_holes[name] = hole
            
    new_abstract_regions = {}
    for name, region in abstract_regions.items():
        if region.layer == layer_name:
            for i, new_layer_name in enumerate(new_layers):
                region_name = f"{name}_{i}"
                new_region = region.copy()
                new_region.name = region_name
                new_region.layer = new_layer_name
                new_abstract_regions[region_name] = new_region
        else:
            new_abstract_regions[name] = region
            
    new_layers.update(layers)
            
    return Device(
        device.name,
        layers=new_layers,
        films=new_films,
        holes=new_holes,
        abstract_regions=new_abstract_regions,
        length_units=device.length_units
    )


def make_layout_small_susc_jkr():
    mat_path = os.path.join(os.path.dirname(__file__), "small_susc.mat")
    layout = loadmat(mat_path)
    origin = layout["origin"]
    pl = layout["pl"]
    pl_centers = layout["pl_centers"]
    pl_shield = layout["pl_shield"]
    pl_shield_2 = layout["pl_shield_2"]
    # A = layout['A']
    fc_in = layout["fc_in"]
    fc_out = layout["fc_out"]
    fc_shield = layout["fc_shield"]
    two_micron_scale = layout["two_micron_scale"]

    z0 = 0.0  # microns
    london_lambda = 0.080  # 80 nm London penetration depth for Nb films

    scale_factor = 2 / (two_micron_scale[1, 0] - two_micron_scale[1, 1])

    components = {}

    fc_in[:, 0] = (fc_in[:, 0] - origin[0, 0]) * scale_factor
    fc_in[:, 1] = -(fc_in[:, 1] - origin[0, 1]) * scale_factor
    components["fc_in"] = fc_in
    fc_out[:, 0] = (fc_out[:, 0] - origin[0, 0]) * scale_factor
    fc_out[:, 1] = -(fc_out[:, 1] - origin[0, 1]) * scale_factor
    components["fc_out"] = fc_out
    fc_shield[:, 0] = (fc_shield[:, 0] - origin[0, 0]) * scale_factor
    fc_shield[:, 1] = -(fc_shield[:, 1] - origin[0, 1]) * scale_factor
    components["fc_shield"] = fc_shield
    pl[:, 0] = (pl[:, 0] - origin[0, 0]) * scale_factor
    pl[:, 1] = -(pl[:, 1] - origin[0, 1]) * scale_factor
    components["pl"] = pl
    pl_shield[:, 0] = (pl_shield[:, 0] - origin[0, 0]) * scale_factor
    pl_shield[:, 1] = -(pl_shield[:, 1] - origin[0, 1]) * scale_factor
    components["pl_shield"] = pl_shield
    pl_shield_2[:, 0] = (pl_shield_2[:, 0] - origin[0, 0]) * scale_factor
    pl_shield_2[:, 1] = -(pl_shield_2[:, 1] - origin[0, 1]) * scale_factor
    components["pl_shield2"] = pl_shield_2
    pl_centers[:, 0] = (pl_centers[:, 0] - origin[0, 0]) * scale_factor
    pl_centers[:, 1] = -(pl_centers[:, 1] - origin[0, 1]) * scale_factor

    thicknesses = {"W2": 0.2, "I2": 0.13, "W1": 0.1, "I1": 0.15, "BE": 0.16}
    heights = {
        "W2": z0 + thicknesses["W2"] / 2,
        "W1": (
            z0
            + sum([thicknesses[k] for k in ["W2", "I2"]])
            + thicknesses["W1"] / 2
        ),
        "BE": (
            z0
            + sum([thicknesses[k] for k in ["W2", "I2", "W1", "I1"]])
            + thicknesses["BE"] / 2
        ),
    }
    polygons = {
        "fc": np.concatenate([fc_in[:-2, :], np.flipud(fc_out)]),
        "fc_shield": np.append(
            fc_shield, fc_shield[0].reshape([-1, 2]), axis=0
        ),
        "pl_shield": np.append(
            pl_shield, pl_shield[0].reshape([-1, 2]), axis=0
        ),
        "pl_shield2": np.append(
            pl_shield_2, pl_shield_2[0].reshape([-1, 2]), axis=0
        ),
        "pl": np.append(pl, pl[0].reshape([-1, 2]), axis=0),
    }

    layers = {
        "W2": Layer(
            "W2",
            thickness=thicknesses["W2"],
            london_lambda=london_lambda,
            z0=heights["W2"],
        ),
        "W1": Layer(
            "W1",
            thickness=thicknesses["W1"],
            london_lambda=london_lambda,
            z0=heights["W1"],
        ),
        "BE": Layer(
            "BE",
            thickness=thicknesses["BE"],
            london_lambda=london_lambda,
            z0=heights["BE"],
        ),
    }
    films = {
        "fc": Polygon("fc", layer="BE", points=polygons["fc"]),
        "pl_shield2": Polygon(
            "pl_shield2", layer="BE", points=polygons["pl_shield2"]
        ),
        "fc_shield": Polygon(
            "fc_shield", layer="W1", points=polygons["fc_shield"]
        ),
        "pl": Polygon("pl", layer="W1", points=polygons["pl"]),
        "pl_shield": Polygon(
            "pl_shield", layer="W2", points=polygons["pl_shield"]
        ),
    }
    holes = {}
    flux_regions = {
        "pl_center": Polygon(
            "pl_center",
            layer="W1",
            points=np.append(
                pl_centers, pl_centers[0].reshape([-1, 2]), axis=0
            )
        ),
    }
    return Device(
        name="small_squid",
        layers=layers.values(),
        films=films.values(),
        holes=holes.values(),
        abstract_regions=flux_regions.values(),
    )


def make_squid():

    d = make_layout_small_susc_jkr()

    fc0 = rotate(d.films["fc"].points, -45)

    fc = np.concatenate(
        [
            fc0[29:-2],
            np.array([[1.80, -0.19]]),
            np.array([[1.95, -0.50]]),
            np.array([[1.75, -0.90]]),
            np.array([[1.50, -1.20]]),
            np.array([[1.30, -1.30]]),
        ]
    )

    fc = np.concatenate([
        fc[1:-4],
        [[1.95, -0.5]],
        [[1.97, -0.8]],
        [[1.97, -1.2]],
        [[1.97, -1.5]],
        [[1.97, -2.0]],
        [[1.90, -2.1]],
        [[1.80, -2.10]],
        [[1.60, -2.10]],
        [[1.40, -2.00]],
        [[1.30, -1.95]],
        [[1.20, -1.80]],
        [[1.15, -1.50]],
        [[1.10, -1.40]],
    ])

    fc_center = np.concatenate(
        [
            np.array([[1.45, -0.65]]),
            fc0[1:28][::-1],
        ]
    )

    fc_center = np.concatenate([
        fc_center[4:-1],
        [[1.20, -0.20]],
        [[1.40, -0.45]],
        [[1.50, -0.65]],
        [[1.55, -0.75]],
        [[1.60, -0.85]],
        [[1.62, -0.95]],
        [[1.64, -1.00]],
        [[1.66, -1.2]],
        [[1.63, -1.30]],
        [[1.50, -1.30]],
        [[1.40, -1.15]],
        [[1.30, -1.00]],
        [[1.20, -0.85]],
        [[1.10, -0.76]],
    ])

    fc_shield = np.array(
        [
            [0.81450159, -1.62204163],
            [0.65, -1.1],
            [0.75880918, -0.04873086],
            [1.25, 0.45],
            [1.55, 0.25],
            [1.8, 0.05],
            [2.0, -0.13],
            [2.15, -0.31326984],
            [2.15, -3.0],
            [1.3, -3.0],
        ]
    )

    pl_shield = np.array(
        [
            [-0.31326984, -0.10442328],
            [0.30630829, -0.11138483],
            [0.71007831, -1.74038802],
            [1.1, -3],
            [-1.1, -3],
            [-0.70311676, -1.55242611],
        ]
    )

    pl_shield2 = np.array(
        [
            [-0.45250089, -1.44104128],
            [0.42465468, -1.44104128],
            [0.57084727, -1.87961906],
            [0.92, -2.9],
            [-0.92, -2.9],
            [-0.54996261, -1.71950336],
        ]
    )

    pl0 = rotate(d.films["pl"].points, -45)
    pl = np.concatenate(
        [
            pl0[:8],
            np.array([[+0.75, -2.8]]),
            np.array([[0.15, -2.8]]),
            pl0[10:17],
            np.array([[-0.15, -2.8]]),
            np.array([[-0.75, -2.8]]),
        ]
    )
    pl_hull = np.concatenate(
        [
            pl0[:8],
            np.array([[+0.70, -2.7]]),
            np.array([[+0.75, -2.8]]),
            np.array([[-0.75, -2.8]]),
            np.array([[-0.70, -2.7]]),
        ]
    )

    bbox = np.array(
        [
            [-1.25, -3.15],
            [-1.25, 1.05],
            [2.20, 1.05],
            [2.20, -3.15],
        ]
    )

    polygons = {
        "fc": fc,
        "fc_center": fc_center,
        "pl_shield2": pl_shield2,
        "fc_shield": fc_shield,
        "pl": pl,
        "pl_shield": pl_shield,
        "pl_hull": pl_hull,
        "bounding_box": bbox,
    }

    films = [
        Polygon("fc", layer="BE", points=polygons["fc"]),
        Polygon("pl_shield2", layer="BE", points=polygons["pl_shield2"]),
        Polygon("fc_shield", layer="W1", points=polygons["fc_shield"]),
        Polygon("pl", layer="W1", points=polygons["pl"]),
        Polygon("pl_shield", layer="W2", points=polygons["pl_shield"]),
    ]

    holes = [
        Polygon("fc_center", layer="BE", points=polygons["fc_center"]),
    ]

    abstract_regions = [
        Polygon("bounding_box", layer="W1", points=polygons["bounding_box"]),
        Polygon("pl_hull", layer="W1", points=polygons["pl_hull"]),
    ]

    return Device(
        d.name,
        layers=d.layers_list,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
    )
