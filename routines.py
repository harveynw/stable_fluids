import jax
import jax.numpy as np

from jax import jit
from functools import partial
from jax._src.lax.control_flow import fori_loop


Grid = np.ndarray


@partial(jit, static_argnums=(0, 1))
def set_bnd(N: int, b: int, x: Grid) -> Grid:
    y_bounds = (x[1, 1:(N+1)], x[N, 1:(N+1)])
    x_bounds = (x[1:(N+1), 1], x[1:(N+1), N])

    x = x.at[0, 1:(N+1)].set(-y_bounds[0] if b == 1 else y_bounds[0])
    x = x.at[N+1, 1:(N+1)].set(-y_bounds[1] if b == 1 else y_bounds[1])
    x = x.at[1:(N+1), 0].set(-x_bounds[0] if b == 2 else x_bounds[0])
    x = x.at[1:(N+1), N+1].set(-x_bounds[1] if b == 2 else x_bounds[1])

    x = x.at[0, 0].set(0.5*(x[1, 0] + x[0, 1]))
    x = x.at[0, N+1].set(0.5*(x[1, N+1] + x[0, N]))
    x = x.at[N+1, 0].set(0.5*(x[N, 0] + x[N+1, 1]))
    x = x.at[N+1, N+1].set(0.5*(x[N, N+1] + x[N+1, N]))

    return x


@partial(jit, static_argnums=0)
def add_source(N: int, x: Grid, s: Grid, dt: float) -> Grid:
    return x + dt * s


@partial(jit, static_argnums=(0, 1))
def diffuse(N: int, b: int, x: Grid, x0: Grid, diff: float, dt: float) -> Grid:
    a = dt * diff * N * N

    def compute_shift_sum(x_dash):
        # x[i-1,j] + x[i+1,j] + x[i,j-1] + x[i,j+1]
        return np.roll(a=x_dash, shift=1, axis=0)\
               + np.roll(a=x_dash, shift=-1, axis=0)\
               + np.roll(a=x_dash, shift=1, axis=1)\
               + np.roll(a=x_dash, shift=-1, axis=1)

    def g_s_iteration(_, x_dash):
        # x[IX(i,j)] = (x0[IX(i,j)] + a*compute_shift_sum(x))/(1+4*a)
        x_dash = (x0 + a * compute_shift_sum(x_dash)) / (1+4*a)
        return set_bnd(N, b, x_dash)

    return fori_loop(0, 20, g_s_iteration, x)


@partial(jit, static_argnums=(0, 1))
def advect(N: int, b: int, grid_points: np.ndarray, d0: Grid, u: Grid, v: Grid, dt: float) -> Grid:
    dt0 = dt*N  # ???

    n_grid_cells = (N+2) * (N+2)
    vel_u, vel_v = u.reshape((n_grid_cells,)), v.reshape((n_grid_cells,))

    backwards_i = np.clip(grid_points[:, 0] - dt0*vel_u, 0.5, N+0.5)
    backwards_j = np.clip(grid_points[:, 1] - dt0*vel_v, 0.5, N+0.5)

    # This is just lerping
    d = jax.scipy.ndimage.map_coordinates(d0, [backwards_i, backwards_j], order=1).reshape((N+2, N+2))
    d = set_bnd(N, b, d)
    return d


@partial(jit, static_argnums=0)
def project(N: int, u: Grid, v: Grid) -> (Grid, Grid):
    h = 1/N

    # STEP 1
    #     for i in range(1, N+1):
    #         for j in range(1, N+1):
    #             div[i,j] = -0.5*h*(u[i+1,j]-u[i-1,j]+v[i,j+1]-v[i,j-1])
    div = -0.5*h*(np.roll(a=u, shift=-1, axis=0) - np.roll(a=u, shift=1, axis=0)
                  + np.roll(a=v, shift=-1, axis=1) - np.roll(a=v, shift=1, axis=1))
    p = np.zeros((N+2, N+2))

    # STEP 2
    div = set_bnd(N, 0, div)
    p = set_bnd(N, 0, p)

    # STEP 3
    #     for _ in range(20):  # G-S
    #         for i in range(1, N+1):
    #             for j in range(1, N+1):
    #                 p[i,j] = (div[i,j]+p[i-1,j]+p[i+1,j]+p[i,j-1]+p[i,j+1])/4
    #         set_bnd (N, 0, p)
    def update_pressure(_, p_dash):
        p_dash = (div + np.roll(a=p_dash, shift=1, axis=0) + np.roll(a=p_dash, shift=-1, axis=0)
                  + np.roll(a=p_dash, shift=1, axis=1) + np.roll(a=p_dash, shift=-1, axis=1))/4.0
        return set_bnd(N, 0, p_dash)
    p = fori_loop(0, 20, update_pressure, p)  # G-S

    # STEP 4
    #     for i in range(1, N+1):
    #         for j in range(1, N+1):
    #             u[i,j] -= 0.5*(p[i+1,j]-p[i-1,j])/h
    #             v[i,j] -= 0.5*(p[i,j+1]-p[i,j-1])/h
    u = u - 0.5 * (np.roll(a=p, shift=-1, axis=0) - np.roll(a=p, shift=1, axis=0)) / h
    v = v - 0.5 * (np.roll(a=p, shift=-1, axis=1) - np.roll(a=p, shift=1, axis=1)) / h

    # STEP 5
    u = set_bnd(N, 1, u)
    v = set_bnd(N, 2, v)

    return u, v

