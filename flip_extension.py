from re import L
import taichi as ti

from fluid_simulator import FluidSimulator
from utils import *
import utils

from functools import reduce
import time
import numpy as np

@ti.data_oriented
class FLIPSimulator(FluidSimulator):
    def __init__(self,
        dim = 2,
        res = (128, 128),
        dt = 1.25e-2,
        substeps = 1,
        dx = 1.0,
        rho = 1000.0,
        gravity = [0, -9.8],
        p0 = 1e-3,
        real = float,
        free_surface = True):
            super().__init__(dim, res, dt, substeps, dx, rho, gravity, p0, real)
            self.free_surface=free_surface
            self.p_v = ti.Vector.field(dim, dtype=real) # velocities

            max_particles = reduce(lambda x, y : x * y, res) * (4 ** dim)
            ti.root.dense(ti.i, max_particles).place(self.p_v)

    @ti.kernel
    def p2g(self):
        for p in range(self.total_mk[None]):
            for k in ti.static(range(self.dim)):
                utils.splat(self.velocity[k], self.velocity_backup[k], self.p_v[p][k], self.p_x[p] / self.dx - 0.5 * (1 - ti.Vector.unit(self.dim, k)))

        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.velocity_backup[k]): # reuse velocity_backup as weight
                if self.velocity_backup[k][I] > 0:
                    self.velocity[k][I] /= self.velocity_backup[k][I]

    @ti.func
    def vel_old_interp(self, pos):
        v = ti.Vector.zero(self.real, self.dim)
        for k in ti.static(range(self.dim)):
            v[k] = utils.sample(self.velocity_backup[k], pos / self.dx - 0.5 * (1 - ti.Vector.unit(self.dim, k)))
        return v

    @ti.kernel
    def g2p(self, dt : ti.f32):
        for p in range(self.total_mk[None]):
            old_v = self.p_v[p]
            self.p_v[p] = self.vel_interp(self.p_x[p]) + 0.99 * (old_v - self.vel_old_interp(self.p_x[p]))
            mispos = self.p_x[p] + self.vel_interp(self.p_x[p]) * (0.5 * dt)
            self.p_x[p] += self.vel_interp(mispos) * dt
    
    @ti.kernel
    def apply_markers_p(self):
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] != utils.SOLID:
                self.cell_type[I] = utils.AIR

        for p in range(self.total_mk[None]):
            I = int(self.p_x[p] / self.dx - 0.5)
            for offset in ti.grouped(ti.ndrange(*((-1, 2), ) * self.dim)):
                if self.cell_type[I+offset] != utils.SOLID and self.is_valid(I+offset):
                    self.cell_type[I+offset] = utils.FLUID

    def substep(self, dt):
        if self.free_surface:
            # self.level_set.build_from_markers(self.p_x, self.total_mk[None])
            self.apply_markers_p()

        for k in range(self.dim):
            self.velocity[k].fill(0)
            self.velocity_backup[k].fill(0)
        self.p2g()
        for k in range(self.dim):
            self.velocity_backup[k].copy_from(self.velocity[k])

        self.extrap_velocity()
        self.enforce_boundary()

        self.add_gravity(dt)
        self.enforce_boundary()
        self.solve_pressure(dt, self.strategy)

        if self.verbose:
            prs = np.max(self.pressure.to_numpy())
            print(f'\033[36mMax pressure: {prs}\033[0m')

        self.apply_pressure(dt)
        self.g2p(dt)
        # self.advect_markers(dt)
        self.total_t += self.dt

    @ti.kernel
    def init_markers(self):
        self.total_mk[None] = 0
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] == utils.FLUID:
                for offset in ti.grouped(ti.ndrange(*((0, 2), ) * self.dim)):
                    num = ti.atomic_add(self.total_mk[None], 1)
                    self.p_x[num] = (I + (offset + [ti.random() for _ in ti.static(range(self.dim))]) / 2) * self.dx
                    self.p_v[num] = self.vel_interp(self.p_x[num])

