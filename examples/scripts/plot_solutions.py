import numpy as np
import matplotlib.colors as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import superscreen as sc

plt.rcParams["font.size"] = 18
plt.rcParams["savefig.dpi"] = 150


def plot_solutions(
    solutions,
    Lambdas,
    figsize=(14, 10),
    grid_shape=200,
    line_cmap="viridis",
    lw=2.5,
    show_Lambda=1,
    plot_polygons=True,
    auto_range_cutoff=0.1,
):
    N = grid_shape // 2

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax, bx), (cx, dx) = axes
    ax.set_aspect("equal")
    bx.set_aspect("equal")
    cx.grid(True)
    dx.grid(True)

    colors = plt.get_cmap(line_cmap)(np.linspace(0, 1, len(Lambdas)))

    for i, (solution, Lambda, color) in enumerate(zip(solutions, Lambdas, colors)):

        xgrid, ygrid, fields = solution.grid_data("fields", grid_shape=grid_shape)
        field = fields["base"]

        if Lambda == show_Lambda:
            vmin, vmax = sc.visualization.auto_range_iqr(field, auto_range_cutoff)
            im = ax.pcolormesh(
                xgrid,
                ygrid,
                field,
                shading="auto",
                cmap="cividis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("$x$ [$\\mu$m]")
            ax.set_ylabel("$y$ [$\\mu$m]")
            ax.set_title("Magnetic field, $\\mu_0H_z$\n$\\Lambda$ = 1 $\\mu$m")
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="8%", pad="4%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(f"$\\mu_0H_z$ [{solution.field_units}]")
            ax.axhline(xvec[N], ls="--", color="w", lw=lw)
            if plot_polygons:
                solution.device.plot_polygons(ax=ax, color="w", lw=1, legend=False)

        xvec, yvec = sc.grids_to_vecs(xgrid, ygrid)
        cx.plot(xvec, field[N, :], color=color, lw=lw)

        xgrid, ygrid, jcs = solution.current_density(
            grid_shape=grid_shape, units="mA / um", with_units=True
        )
        jx, jy = jcs["base"].magnitude
        units = jcs["base"].units
        J = np.sqrt(jx ** 2 + jy ** 2)
        xvec, yvec = sc.grids_to_vecs(xgrid.magnitude, ygrid.magnitude)

        if i == 0:
            jxlabel = "$J_x$"
            jylabel = "$J_y$"
        else:
            jxlabel = jylabel = None

        dx.plot(xvec, jx[N, :], color=color, ls="--", lw=lw, label=jxlabel)
        dx.plot(xvec, jy[N, :], color=color, ls="-", lw=lw, label=jylabel)

        if Lambda == show_Lambda:
            im = bx.pcolormesh(xvec, yvec, J, shading="auto", cmap="inferno")
            bx.set_xlabel("$x$ [$\\mu$m]")
            bx.set_ylabel("$y$ [$\\mu$m]")
            bx.set_title("Sheet current, $|\\vec{J}|$\n$\\Lambda$ = 1 $\\mu$m")
            ax_divider = make_axes_locatable(bx)
            cax = ax_divider.append_axes("right", size="6%", pad="4%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("$|\\vec{J}|$" + f" [{units:~P}]")
            bx.axhline(xvec[N], ls="--", color="w", lw=lw)
            jx[J < 0.075 * np.nanmax(J)] = np.nan
            jy[J < 0.075 * np.nanmax(J)] = np.nan
            bx.streamplot(
                xgrid.m,
                ygrid.m,
                jx,
                jy,
                color="w",
                density=1.25,
                linewidth=1,
            )
            if plot_polygons:
                solution.device.plot_polygons(ax=bx, color="w", lw=1, legend=False)

    cx.set_xlabel("$x$ [$\\mu$m]")
    cx.set_ylabel(f"$\\mu_0H_z$ [{solution.field_units}]")

    dx.set_xlabel("$x$ [$\\mu$m]")
    dx.set_ylabel(f"$J_x$, $J_y$ [{units:~P}]")
    dx.legend(loc=1)

    sm = plt.cm.ScalarMappable(
        cmap=line_cmap, norm=cm.LogNorm(Lambdas.min(), Lambdas.max())
    )
    cbar = fig.colorbar(sm, ax=(cx, dx), orientation="horizontal", aspect=50, pad=0.25)
    cbar.set_label(f"Effective penetration depth, $\\Lambda$ [{xgrid.units:~P}]")

    return fig, axes
