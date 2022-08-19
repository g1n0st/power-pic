import taichi as ti
import utils
from fluid_simulator import *
import numpy as np

@ti.data_oriented
class Visualizer2D:
    def __init__(self, grid_res, res):
        self.grid_res = grid_res
        self.res = res
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))

    @ti.func
    def ij_to_xy(self, i, j):
        return int((i + 0.5) / self.res * self.grid_res), \
               int((j + 0.5) / self.res * self.grid_res)

    @ti.kernel
    def fill_pressure(self, p : ti.template()):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            m = (ti.log(min(p[x, y], 1e6) + 1) / ti.log(10)) / 6
            self.color_buffer[i, j] = ti.Vector([m, m, m])

    @ti.kernel
    def fill_levelset(self, phi : ti.template(), dx : ti.template()):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            p = min(phi[x, y] / (dx * self.grid_res) * 1e2, 1)
            if p > 0: self.color_buffer[i, j] = ti.Vector([p, 0, 0])
            else: self.color_buffer[i, j] = ti.Vector([0, 0, -p])

    @ti.kernel
    def visualize_kernel(self, phi : ti.template(), cell_type : ti.template()):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            if cell_type[x, y] == utils.SOLID: 
                self.color_buffer[i, j] = ti.Vector([0, 0, 0])
            elif phi[x, y] <= 0: 
                self.color_buffer[i, j] = ti.Vector([113 / 255, 131 / 255, 247 / 255]) # fluid
            else: 
                self.color_buffer[i, j] = ti.Vector([1, 1, 1])

    def visualize_factory(self, simulator):
        if self.mode == 'pressure':
            self.fill_pressure(simulator.pressure)
        elif self.mode == 'levelset':
            self.fill_levelset(simulator.level_set.phi, simulator.dx)
        elif self.mode == 'visual':
            self.visualize_kernel(simulator.level_set.phi, simulator.cell_type)

    def visualize(self, simulator):
        assert 0, 'Please use GUIVisualizer2D'

@ti.data_oriented
class GUIVisualizer2D(Visualizer2D):
    def __init__(self, grid_res, res, mode, title = 'demo'):
        super().__init__(grid_res, res)
        self.mode = mode
        self.window = ti.ui.Window(title, (res, res))
        self.canvas = self.window.get_canvas()
        self.frame = 0

    def visualize(self, simulator):
        self.canvas.set_background_color(color=(0.0, 0.0, 0.0))
        if self.mode == 'p':
            self.canvas.circles(simulator.p_x, 0.001, (1.0, 1.0, 1.0))
        else:
            self.visualize_factory(simulator)
            self.canvas.set_image(self.color_buffer)
        self.window.show()
        self.frame += 1