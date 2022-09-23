import time
import cv2
import jax.numpy as np
import numpy as onp

from routines import add_source
from solvers import dens_step, vel_step
from initial_conditions import setup_1


class MouseTracker:
    def __init__(self, N):
        self.N = N
        self.mouse_x, self.mouse_y = 0.0, 0.0
        self.mouse_dx, self.mouse_dy = 0.0, 0.0
        self.mouse_down = False

    def retrieve_update_matrices(self) -> (np.ndarray, np.ndarray, np.ndarray):
        add_source = np.zeros((self.N+2, self.N+2))
        add_u, add_v = np.zeros((self.N+2, self.N+2)), np.zeros((self.N+2, self.N+2))

        if self.mouse_down:
            i, j = self.N - int(self.mouse_y), int(self.mouse_x)
            add_source = add_source.at[i, j].set(1.0)

            add_u = add_u.at[i, j].set(self.mouse_dx)
            add_v = add_v.at[i, j].set(self.mouse_dy)

            self.mouse_dx, self.mouse_dy = 0.0, 0.0

        return add_source, add_u, add_v

    def register(self, window_name):
        cv2.setMouseCallback(window_name, self.mouse_update)

    def mouse_update(self, event, x, y, flags, param):
        # This is the OpenCV hook

        if event == cv2.EVENT_LBUTTONDOWN:
            # print('Mouse down')
            self.mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            # print('Mouse up')
            self.mouse_down = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_down:
            self.mouse_dx += x - self.mouse_x
            self.mouse_dy += y - self.mouse_y
            # print('Mouse move', self.mouse_dx, self.mouse_dy)
            self.mouse_x, self.mouse_y = x, y


def run_simulation():
    N = 100
    grid = (N+2, N+2)
    grid_points = np.array([
        np.array([i, j])
        for i in range(N+2)
        for j in range(N+2)
    ])
    u, v, dens, dens_prev, diff, dt = setup_1(grid)
    u_prev, v_prev = np.array(u), np.array(v)
    visc = 0.5

    print('Beginning full solver...')
    window_name = 'Stable Fluids'
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    mouse = MouseTracker(N=N)
    mouse.register(window_name)

    prev_frame_time, new_frame_time = 0, 0
    while True:
        u, v, u_prev, v_prev = vel_step(N, grid_points, u, v, u_prev, v_prev, visc, dt)
        dens, dens_prev = dens_step(N, grid_points, dens, dens_prev, u, v, diff, dt)

        frame = onp.flip(onp.array(dens), axis=0)
        frame = np.repeat(frame[:, :, onp.newaxis], 3, axis=2)
        frame = onp.ascontiguousarray(frame)

        # User Input
        del_dens, del_u, del_v = mouse.retrieve_update_matrices()
        dens = add_source(N, dens, del_dens, dt)
        u = add_source(N, u, del_u, dt)
        v = add_source(N, v, del_v, dt)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = 'FPS: ' + str(int(fps))

        fontScale, thickness = 0.2, 1
        cv2.putText(frame, fps, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, f'Total Density: {np.sum(dens)}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow(window_name, frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Closes all the frames
    cv2.destroyAllWindows()


run_simulation()
