import taichi as ti

from flip_extension import FLIPSimulator
from utils import *
import utils
import numpy as np

@ti.data_oriented
class PowerPICSimulator(FLIPSimulator):
    def __init__(self,
        dim = 2,
        res = (128, 128),
        t_res = (256, 256),
        dt = 1.25e-2,
        substeps = 1,
        dx = 1.0,
        rho = 1000.0,
        gravity = [0, -9.8],
        p0 = 1e-3,
        real = float,
        free_surface = True):
            super().__init__(dim, res, dt, substeps, dx, rho, gravity, p0, real, free_surface)
            self.s_p = ti.field(real, shape=self.max_particles)
            self.sum_p = ti.field(real, shape=self.max_particles)
            self.Z_p = ti.field(real, shape=self.max_particles) # log-domain Z=max{a[i]+b[i]}
            self.c_p = ti.Vector.field(dim, real, shape=self.max_particles)
            self.u_p = ti.Vector.field(dim, real, shape=self.max_particles)
            
            self.t_res = t_res
            self.t_dx = 1 / self.t_res[0]
            self.s_j = ti.field(real)
            self.sum_j = ti.field(real)
            self.Z_j = ti.field(real) # log-domain Z=max{a[i]+b[i]}
            ti.root.dense(self.indices, self.t_res).place(self.s_j, self.sum_j, self.Z_j)

            self.entropy_eps = 2.0 * (self.t_dx ** 2)
            self.sinkhorn_delta = 0.1
            self.V_j = self.t_dx ** self.dim

            self.R_2 = 6 # 3 * sqrt(2) + EPS
            self.R = self.R_2 * 2
    
    @ti.func
    def K_pj(self, x_p, x_j):
        return -(x_p - x_j).norm_sqr() / self.entropy_eps
    
    @ti.func
    def check_Tg(self, I):
        return all(I >= 0) and all(I < self.t_res)
    
    @ti.func
    def T(self, p, I):
        pos = (I + 0.5) * self.t_dx
        return ti.math.exp(self.K_pj(self.p_x[p], pos) + self.s_p[p] + self.s_j[I])
    
    @ti.func
    def get_base(self, I):
        p, I_i = I[0], I[1:]
        base = (self.p_x[p] / self.t_dx).cast(ti.i32)
        I_x = base + I_i - self.R_2
        pos = (I_x + 0.5) * self.t_dx
        return p, I_x, pos

    def init_sinkhorn_algo(self):
        self.s_p.fill(0.0)
        self.s_j.fill(0.0)

    @ti.kernel
    def calc_max_sum_j(self) -> ti.f32:
        for I in ti.grouped(self.sum_j): self.sum_j[I] = 0.0
        for I in ti.grouped(ti.ndrange(self.total_mk[None], *(self.R, ) * self.dim)):
            p, I_x, pos = self.get_base(I)
            if self.check_Tg(I_x):
                self.sum_j[I_x] += self.T(p, I_x)
        
        ret = 0.0
        for I in ti.grouped(self.sum_j):
            if self.check_Tg(I):
                val = ti.abs(self.sum_j[I] / self.V_j - 1)
                ti.atomic_max(ret, val)
        return ret

    @ti.kernel
    def sinkhorn_algo(self):
        V_p = 1.0 / self.total_mk[None]
        log_Vp = ti.math.log(V_p)
        log_Vj = ti.math.log(self.V_j)
        stencil = ti.static((self.total_mk[None], ) + (self.R, ) * self.dim)

        for I in ti.grouped(self.sum_j): 
                self.sum_j[I] = 0.0
                self.Z_j[I] = -1e10
        
        # solve in log-domain
        for I in ti.grouped(ti.ndrange(*stencil)):
            p, I_x, pos = self.get_base(I)
            if self.check_Tg(I_x):
                ti.atomic_max(self.Z_j[I_x], self.K_pj(self.p_x[p], pos) + self.s_p[p])
        for I in ti.grouped(ti.ndrange(*stencil)):
            p, I_x, pos = self.get_base(I)
            if self.check_Tg(I_x):
                self.sum_j[I_x] += ti.math.exp(self.K_pj(self.p_x[p], pos) + self.s_p[p] - self.Z_j[I_x])

        for I in ti.grouped(self.s_j):
            self.s_j[I] = log_Vj - (self.Z_j[I] + ti.math.log(self.sum_j[I]))


        for p in self.sum_p: 
            self.sum_p[p] = 0.0
            self.Z_p[p] = -1e10
        for I in ti.grouped(ti.ndrange(*stencil)):
            p, I_x, pos = self.get_base(I)
            if self.check_Tg(I_x):
                ti.atomic_max(self.Z_p[p], self.K_pj(self.p_x[p], pos) + self.s_j[I_x])
        for I in ti.grouped(ti.ndrange(*stencil)):
            p, I_x, pos = self.get_base(I)
            if self.check_Tg(I_x):
                self.sum_p[p] += ti.math.exp(self.K_pj(self.p_x[p], pos) + self.s_j[I_x] - self.Z_p[p])

        for p in self.s_p:
            self.s_p[p] = log_Vp - (self.Z_p[p] + ti.math.log(self.sum_p[p]))

    @ti.kernel
    def p2g(self):
        V_p = 1.0 / self.total_mk[None]
        for I in ti.grouped(ti.ndrange(self.total_mk[None], *(self.R, ) * self.dim)):
            p, I_x, pos = self.get_base(I)
            if self.check_Tg(I_x):
                for k in ti.static(range(self.dim)):
                    utils.splat_w(self.velocity[k], self.velocity_backup[k], self.p_v[p][k], pos / self.dx - 0.5 * (1 - ti.Vector.unit(self.dim, k)), self.T(p, I_x) / V_p)

        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.velocity_backup[k]): # reuse velocity_backup as weight
                if self.velocity_backup[k][I] > 0:
                    self.velocity[k][I] /= self.velocity_backup[k][I]

    @ti.kernel
    def g2p(self, dt : ti.f32):
        V_p = 1.0 / self.total_mk[None]
        for p in range(self.total_mk[None]): 
            self.c_p[p].fill(0.0)
            self.u_p[p].fill(0.0)
        for I in ti.grouped(ti.ndrange(self.total_mk[None], *(self.R, ) * self.dim)):
            p, I_x, pos = self.get_base(I)
            if self.check_Tg(I_x):
                T = self.T(p, I_x)
                self.p_v[p] += T * (self.vel_interp(pos) - self.vel_old_interp(pos)) / V_p
                self.c_p[p] += pos * T / V_p
                self.u_p[p] += T * self.vel_interp(pos) / V_p

        for p in range(self.total_mk[None]):
            self.p_x[p] = self.c_p[p] + dt * self.u_p[p]

    def substep(self, dt):
        self.init_sinkhorn_algo()
        while self.calc_max_sum_j() > self.sinkhorn_delta:
            self.sinkhorn_algo()

        for k in range(self.dim):
            self.velocity[k].fill(0)
            self.velocity_backup[k].fill(0)
        self.p2g()
        for k in range(self.dim):
            self.velocity_backup[k].copy_from(self.velocity[k])

        # self.extrap_velocity()
        # self.enforce_boundary()

        self.add_gravity(dt)
        self.enforce_boundary()
        self.solve_pressure(dt, self.strategy)

        if self.verbose:
            prs = np.max(self.pressure.to_numpy())
            print(f'\033[36mMax pressure: {prs}\033[0m')

        self.apply_pressure(dt)
        self.g2p(dt)
        self.total_t += self.dt