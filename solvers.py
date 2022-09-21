import jax.numpy as np

from functools import partial
from jax import jit
from routines import Grid, add_source, diffuse, advect, project


@partial(jit, static_argnums=0)
def dens_step(N: int, grid_points: np.ndarray, x: Grid, x0: Grid, u: Grid, v: Grid, diff: float, dt: float) -> (Grid, Grid):
    x = add_source(N, x, x0, dt)

    x, x0 = x0, x
    x = diffuse(N, 0, x, x0, diff, dt)

    x, x0 = x0, x
    x = advect(N, 0, grid_points, x0, u, v, dt)

    return x, x0


@partial(jit, static_argnums=0)
def vel_step(N: int, u: Grid, v: Grid, u0: Grid, v0: Grid, visc: float, dt: float):
    u = add_source(N, u, u0, dt)
    v = add_source(N, v, v0, dt)

    u, u0 = u0, u
    u = diffuse(N, 1, u, u0, visc, dt)

    v, v0 = v0, v
    v = diffuse(N, 2, v, v0, visc, dt)

    u, v = project(N, u, v, u0, v0)
    u, u0 = u0, u
    v, v0 = v0, v

    u = advect(N, 1, u0, u0, v0, dt)
    v = advect(N, 2, v0, u0, v0, dt)

    u, v = project(N, u, v, u0, v0)

    return u, v, u0, v0
