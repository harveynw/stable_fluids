import numpy as onp
import jax.numpy as np


def setup_1(grid):
    # Linear movement
    u = 0.1*onp.ones(grid)
    v = 0.3*onp.ones(grid)

    # Density initial conditions
    dens, dens_prev = onp.zeros(grid), onp.zeros(grid)

    # Diffusion, viscosity
    diff = 0.000005
    visc = 0.0001

    return np.array(u), np.array(v), np.array(dens), np.array(dens_prev), diff, visc


def setup_2(grid):
    # Swirl
    u = np.zeros(grid)
    v = np.zeros(grid)

    for i in range(1, grid[0]-1):
        for j in range(1, grid[1]-1):
            if i < 50:
                if j < 50:
                    u[i, j], v[i, j] = 0.0, 1.0
                else:
                    u[i, j], v[i, j] = 1.0, 0.0
            else:
                if j < 50:
                    u[i, j], v[i, j] = -1.0, 0.0
                else:
                    u[i, j], v[i, j] = 0.0, -1.0

    u, v = 0.3 * u, 0.3 * v

    # Density initial conditions
    dens, dens_prev = np.zeros(grid), np.zeros(grid)

    dens[40:60, 40:60] = 1
    dens_prev[40:60, 40:60] = 1

    diff, dt = 0.002, 0.01

    return u, v, dens, dens_prev, diff, dt
