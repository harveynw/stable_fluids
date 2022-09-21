import jax
import jax.numpy as np

from jax import jit, vmap
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


# @partial(jit, static_argnums=(0, 1))
# def advect(N: int, b: int, d0: Grid, u: Grid, v: Grid, dt: float) -> Grid:
#     dt0 = dt*N  # ???
#
#     indices = np.array([
#         (i, j)
#         for i in range(1, N+1)
#         for j in range(1, N+1)
#     ])
#     indices_float = np.array([
#         np.array([i, j])
#         for i in range(1, N+1)
#         for j in range(1, N+1)
#     ])
#
#     lerp_grid_1d = np.arange(N+2)
#
#     def update(d_dash, pair):
#         (i, j), (ii, jj) = pair
#
#         k = np.clip(i - dt0*u[ii, jj], 0.5, N+0.5)
#         l = np.clip(j - dt0*v[ii, jj], 0.5, N+0.5)
#
#         # Lerp density method 1
#         interp_vmap = vmap(np.interp, (None, None, 0), 0)
#         lerp_y = interp_vmap(l, lerp_grid_1d, d_dash)
#         lerp_x_y = np.interp(k, lerp_grid_1d, lerp_y)
#
#         return d_dash.at[ii, jj].set(lerp_x_y), 0.0
#
#     d, _ = jax.lax.scan(update, np.array(d0), (indices_float, indices))
#     d = set_bnd(N, b, d)
#     return d


@partial(jit, static_argnums=(0, 1))
def advect(N: int, b: int, grid_points: np.ndarray, d0: Grid, u: Grid, v: Grid, dt: float) -> Grid:
    dt0 = dt*N  # ???

    n_grid_cells = (N+2) * (N+2)
    vel_u, vel_v = u.reshape((n_grid_cells,)), v.reshape((n_grid_cells,))

    backwards_i = np.clip(grid_points[:, 0] - dt0*vel_u, 0.5, N+0.5)
    backwards_j = np.clip(grid_points[:, 1] - dt0*vel_v, 0.5, N+0.5)

    d = jax.scipy.ndimage.map_coordinates(d0, [backwards_i, backwards_j], order=1).reshape((N+2, N+2))
    d = set_bnd(N, b, d)
    return d


@partial(jit, static_argnums=0)
def project(N: int, u: Grid, v: Grid, p: Grid, div: Grid) -> (Grid, Grid):
    h = 1/N
    indices = np.array([
        (i, j)
        for i in range(1, N+1)
        for j in range(1, N+1)
    ])

    # STEP 1
    #     for i in range(1, N+1):
    #         for j in range(1, N+1):
    #             div[i,j] = -0.5*h*(u[i+1,j]-u[i-1,j]+v[i,j+1]-v[i,j-1])
    def init_div(div, idx):
        ii, jj = idx
        div = div.at[ii, jj].set(-0.5*h*(u[ii+1,jj]-u[ii-1,jj]+v[ii,jj+1]-v[ii,jj-1]))
        return div, 0.0
    div, _ = jax.lax.scan(init_div, div, indices)

    # STEP 2
    div = set_bnd(N, 0, div)
    p = set_bnd(N, 0, p)

    # STEP 3
    #     for _ in range(20):  # G-S
    #         for i in range(1, N+1):
    #             for j in range(1, N+1):
    #                 p[i,j] = (div[i,j]+p[i-1,j]+p[i+1,j]+p[i,j-1]+p[i,j+1])/4
    #         set_bnd (N, 0, p)
    def update_pressure(p, idx):
        ii, jj = idx
        p = p.at[ii, jj].set((div[ii,jj]+p[ii-1,jj]+p[ii+1,jj]+p[ii,jj-1]+p[ii,jj+1])/4.0)
        return p, 0.0
    def g_s_iteration(_, p):
        p, _ = jax.lax.scan(update_pressure, p, indices)
        return set_bnd(N, 0, p)
    p = fori_loop(0, 20, g_s_iteration, p)

    # STEP 4
    #     for i in range(1, N+1):
    #         for j in range(1, N+1):
    #             u[i,j] -= 0.5*(p[i+1,j]-p[i-1,j])/h
    #             v[i,j] -= 0.5*(p[i,j+1]-p[i,j-1])/h
    def update_velocity(vel, idx):
        ii, jj = idx
        u, v = vel
        u = u.at[ii, jj].set(u[ii,jj] - 0.5*(p[ii+1,jj]-p[ii-1,jj])/h)
        v = v.at[ii, jj].set(u[ii,jj] - 0.5*(p[ii,jj+1]-p[ii,jj-1])/h)
        return np.array([u, v]), 0.0
    (u, v), _ = jax.lax.scan(update_velocity, np.array([u, v]), indices)

    # STEP 5
    u = set_bnd (N, 1, u)
    v = set_bnd (N, 2, v)

    return u, v


