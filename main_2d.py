import argparse
import taichi as ti
from fluid_simulator import *
from flip_extension import *
from power_pic import *
from initializer_2d import *
from visualizer_2d import *

parser = argparse.ArgumentParser()
parser.add_argument('--flip', action="store_true")
parser.add_argument('--res', type=int, default=64)
parser.add_argument('--show', type=str, default="p")
args = parser.parse_args()

if __name__ == '__main__':
    fs = False
    initializer = SphereInitializer2D(args.res, 0.5, 0.5, 0.2, free_surface=fs)
    if args.flip:
        solver = FLIPSimulator(dim=2, res=(args.res, args.res), dt=1e-2, substeps=1, dx=1 / args.res, p0=0, 
                            gravity = [0.0, -9.8] if fs else [0.0, 0.0], 
                            free_surface=fs)
    else:
        solver = PowerPICSimulator(dim=2, res=(args.res, args.res), dt=1e-2, substeps=1, dx=1 / args.res, p0=0, 
                            gravity = [0.0, -9.8] if fs else [0.0, 0.0], 
                            free_surface=fs)

    visualizer = GUIVisualizer2D(args.res, 1024, args.show)
    solver.initialize(initializer)
    solver.run(-1, visualizer)
    # ti.kernel_profiler_print()
