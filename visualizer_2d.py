import taichi as ti
import utils
from fluid_simulator import *
import numpy as np

@ti.data_oriented
class Visualizer2D:
    def __init__(self, grid_res, res):
        self.grid_res = grid_res
        self.res = res
        self.tmp = ti.Vector.field(3, dtype=ti.f32, shape=(self.grid_res, self.grid_res))
        self.tmp_w = ti.field(dtype=ti.f32, shape=(self.grid_res, self.grid_res))
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))

    @ti.func
    def ij_to_xy(self, i, j):
        return int((i + 0.5) / self.res * self.grid_res), \
               int((j + 0.5) / self.res * self.grid_res)

    @ti.kernel
    def fill_power(self, sim : ti.template()):
        for i, j in self.tmp:
            self.tmp[i, j].fill(0.0)
            self.tmp_w[i, j] = 0.0
        for I in ti.grouped(ti.ndrange(sim.total_mk[None], *(sim.R, ) * sim.dim)):
            p, I_x, pos = self.get_base(I)
            if self.check_Tg(I_x):
                T = sim.T(p, I_x)
                self.tmp[I_x] += T * sim.color_p[p]
                self.tmp_w[I_x] += T
        
        V_p = (1.0 / sim.total_mk[None])
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)
            self.color_buffer[i, j] = self.tmp[x, y] / self.tmp_w[x, y]

    def visualize_factory(self, simulator):
        if self.mode == 'power':
            self.fill_power(simulator)

    def visualize(self, simulator):
        assert 0, 'Please use GUIVisualizer2D'
    
    def end(self):
        pass

@ti.data_oriented
class GUIVisualizer2D(Visualizer2D):
    def __init__(self, grid_res, res, mode, title = 'demo', export=""):
        super().__init__(grid_res, res)
        self.mode = mode
        self.window = ti.ui.Window(title, (res, res), vsync=True)
        self.canvas = self.window.get_canvas()
        self.frame = 0
        self.export = False
        if export != "":
            self.export = True
            self.video_manager = ti.tools.VideoManager(export)

    def visualize(self, simulator):
        self.canvas.set_background_color(color=(0.0, 0.0, 0.0))
        if self.mode == 'p':
            self.canvas.circles(simulator.p_x, 0.002, per_vertex_color=simulator.color_p)
        else:
            self.visualize_factory(simulator)
            self.canvas.set_image(self.color_buffer)
        
        if self.export:
            img = self.window.get_image_buffer_as_numpy()
            self.video_manager.write_frame(img)

        self.window.show()
        self.frame += 1
    
    def end(self):
        if self.export:
            self.video_manager.make_video(gif=True, mp4=True)