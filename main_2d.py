import taichi as ti
from fluid_simulator import *
from flip_extension import *
from initializer_2d import *
from visualizer_2d import *

if __name__ == '__main__':
    res = 128
    fs = False
    initializer = SphereInitializer2D(res, 0.5, 0.5, 0.2, free_surface=fs)

    solver = FLIPSimulator(dim=2, res=(res, res), dt=1e-2, substeps=1, dx=1 / res, p0=0, 
                           gravity = [0.0, -9.8] if fs else [0.0, 0.0], 
                           free_surface=fs)

    visualizer = GUIVisualizer2D(res, 1024, 'p')
    solver.initialize(initializer)
    solver.run(-1, visualizer)
    # ti.kernel_profiler_print()
