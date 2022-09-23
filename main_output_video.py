# For debugging only, use the interactive solver

import os
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation

from typing import List
from tqdm import tqdm
from routines import Grid
from solvers import dens_step, vel_step
from initial_conditions import setup_1

matplotlib.use('TkAgg')


def simulate(frames=10) -> (List[Grid], List[Grid], List[Grid]):
    dens_history = []
    u_vel_history, v_vel_history = [], []

    N = 100

    grid = (N+2, N+2)
    grid_points = np.column_stack([np.repeat(np.arange(N+2), N+2), np.tile(np.arange(N+2), N+2)])

    u, v, dens, dens_prev, diff, dt = setup_1(grid)
    u_prev, v_prev = np.array(u), np.array(v)
    visc = 10.0

    print('Beginning full solver...')
    for _ in tqdm(range(frames)):
        u, v, u_prev, v_prev = vel_step(N, grid_points, u, v, u_prev, v_prev, visc, dt)
        dens, dens_prev = dens_step(N, grid_points, dens, dens_prev, u, v, diff, dt)

        dens_history.append(dens)
        u_vel_history.append(u)
        v_vel_history.append(v)

    return dens_history, u_vel_history, v_vel_history


def start_animation(dens_history: List[Grid], u_vel_history: List[Grid], v_vel_history: List[Grid]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    im_1 = dens_history[0].transpose()
    image_1 = ax1.imshow(im_1, interpolation='None', origin='lower', animated=True)

    im_2 = u_vel_history[0].transpose()
    image_2 = ax2.imshow(im_2, interpolation='None', origin='lower', animated=True)

    im_3 = v_vel_history[0].transpose()
    image_3 = ax3.imshow(im_3, interpolation='None', origin='lower', animated=True)

    def function_for_animation(frame_index):
        image_1.set_data(dens_history[frame_index].transpose())
        image_2.set_data(u_vel_history[frame_index].transpose())
        image_3.set_data(v_vel_history[frame_index].transpose())

        fig.suptitle(str(frame_index))
        return ax1, ax2, ax3

    ani = matplotlib.animation.FuncAnimation(fig, function_for_animation, interval=200, frames=len(dens_history))
    return ani


# Sim
dens_history, u_vel_history, v_vel_history = simulate(1000)

# Animate
anim = start_animation(dens_history, u_vel_history, v_vel_history)

# Save
writer = matplotlib.animation.FFMpegWriter(fps=30)
if not os.path.isdir('outputs'):
    os.mkdir('outputs')
anim.save('outputs/full_solver.mp4', writer=writer)
