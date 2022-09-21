import numpy as onp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import jax.numpy as np

from typing import List
from tqdm import tqdm
from routines import Grid
from solvers import dens_step
from initial_conditions import setup_1

matplotlib.use('TkAgg')


def simulate(frames=10) -> List[Grid]:
    history = []

    N = 100
    grid = (N+2, N+2)
    u, v, dens, dens_prev, diff, dt = setup_1(grid)

    grid_points = np.array([
        np.array([i, j])
        for i in range(N+2)
        for j in range(N+2)
    ])

    print('Beginning density ONLY solver...')
    for _ in tqdm(range(frames)):
        dens, dens_prev = dens_step(N, grid_points, dens, dens_prev, u, v, diff, dt)
        history.append(onp.array(dens))

    return history


def start_animation_1(history: List[Grid]):
    f = plt.figure()
    ax = f.gca()

    im = history[0].transpose()
    image = plt.imshow(im, interpolation='None', origin='lower', animated=True)

    def function_for_animation(frame_index):
        image.set_data(history[frame_index].transpose())
        ax.set_title(str(frame_index))
        return image,

    ani = matplotlib.animation.FuncAnimation(f, function_for_animation, interval=200, frames=len(dens_history),
                                             blit=True)
    return ani


dens_history = simulate(1000)
anim = start_animation_1(dens_history)
writervideo = matplotlib.animation.FFMpegWriter(fps=30)
anim.save('outputs/density_only_new_advect.mp4', writer=writervideo)
