"""Robotics domain: tune an operational-space controller for a simulated 2-link planar arm.

The 'reality gap' is modelled explicitly: the controller believes in nominal link masses,
while the plant it drives has perturbed masses. Validation uses mild perturbations; the
hidden TEST split uses larger perturbations, sensor noise and a 1-step actuation delay.
A controller that overfits the validation sim will do worse on test.
"""
from __future__ import annotations

import numpy as np

from research_loop import Domain

L1, L2 = 0.30, 0.25          # link lengths (m)
M1, M2 = 1.2, 0.8            # nominal link masses (kg)
DT, T = 0.005, 1.5           # step, episode length (s)
TAU_MAX = 25.0               # hardware torque limit (N m)
STEPS = int(T / DT)


def _inertia(q2, m1, m2):
    lc1, lc2 = L1 / 2, L2 / 2
    i1, i2 = m1 * L1**2 / 12, m2 * L2**2 / 12
    c2 = np.cos(q2)
    m11 = m1 * lc1**2 + m2 * (L1**2 + lc2**2 + 2 * L1 * lc2 * c2) + i1 + i2
    m12 = m2 * (lc2**2 + L1 * lc2 * c2) + i2
    m22 = m2 * lc2**2 + i2
    return np.array([[m11, m12], [m12, m22]])


def _coriolis(q2, qd, m2):
    h = m2 * L1 * (L2 / 2) * np.sin(q2)
    return np.array([-h * (2 * qd[0] * qd[1] + qd[1] ** 2), h * qd[0] ** 2])


def _fk(q):
    return np.array([L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]),
                     L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])])


def _jac(q):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def _ik(target):
    """Analytic elbow-down IK, used only to pick a sane initial pose reference."""
    x, y = target
    d = (x * x + y * y - L1**2 - L2**2) / (2 * L1 * L2)
    q2 = np.arccos(np.clip(d, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


def episode(cfg, target, q0, mass_scale, noise, delay, rng):
    """Simulate one reach. Returns (cost, final_error, settle_time)."""
    m1, m2 = M1 * mass_scale[0], M2 * mass_scale[1]
    q, qd = q0.copy(), np.zeros(2)
    prev_tau = np.zeros(2)
    err_int = 0.0
    effort = 0.0
    settle = 0.0
    for k in range(STEPS):
        qm = q + rng.normal(0, noise, 2) if noise else q
        qdm = qd + rng.normal(0, noise * 10, 2) if noise else qd
        x = _fk(qm)
        J = _jac(qm)
        err = target - x
        # Cartesian PD -> joint torques, with joint damping and optional model-based feedforward
        F = cfg["kp"] * err - cfg["kd"] * (J @ qdm)
        tau = J.T @ F - cfg["kj"] * qdm
        if cfg["ff"] > 0.5:
            tau = tau + _coriolis(qm[1], qdm, M2)        # NOMINAL model, not the true plant
        tau = np.clip(tau, -cfg["tau_lim"], cfg["tau_lim"])
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        applied = prev_tau if delay else tau
        prev_tau = tau
        # True plant (perturbed masses) + viscous joint friction
        M = _inertia(q[1], m1, m2)
        acc = np.linalg.solve(M, applied - _coriolis(q[1], qd, m2) - 0.05 * qd)
        qd = qd + acc * DT                                # semi-implicit Euler
        q = q + qd * DT
        e = float(np.linalg.norm(target - _fk(q)))
        if not np.isfinite(e) or e > 5.0:
            return 50.0, 5.0, T
        err_int += e * DT
        effort += float(applied @ applied) * DT
        if e > 0.01:
            settle = (k + 1) * DT          # time of the most recent step outside 1 cm
    final = float(np.linalg.norm(target - _fk(q)))
    # cost: tracking error integral + final error + settle time + small effort term
    cost = 4.0 * err_int + 20.0 * final + 0.3 * settle + 0.0005 * effort
    return cost, final, settle


class RoboticsDomain(Domain):
    name = "robotics_arm_reach"
    space = {
        "kp": ("log", 5.0, 600.0),         # Cartesian stiffness
        "kd": ("log", 0.2, 60.0),          # Cartesian damping
        "kj": ("log", 0.005, 3.0),         # joint damping
        "tau_lim": ("log", 2.0, 25.0),     # software torque limit
        "ff": ("cat", [0, 1]),             # nominal Coriolis feedforward on/off
    }

    def __init__(self, n_targets: int = 10, seed: int = 123, robust_val: bool = False, test_seed: int | None = None):
        self.robust_val = robust_val
        rng = np.random.default_rng(seed)
        self.n = n_targets
        # Targets are reachable points; q0 is the shared home pose.
        def targets():
            out = []
            for _ in range(n_targets):
                r = rng.uniform(0.15, 0.50)
                th = rng.uniform(0.2, 1.6)
                out.append(np.array([r * np.cos(th), r * np.sin(th)]))
            return out
        self.val_targets, self.test_targets = targets(), targets()
        if test_seed is not None:   # fresh, never-before-seen test set
            rng = np.random.default_rng(test_seed)
            self.test_targets = targets()
            self.test_noise_seed = test_seed
        else:
            self.test_noise_seed = 99
        self.q0 = np.array([0.6, 1.2])

    def baseline(self):
        return {"kp": 60.0, "kd": 6.0, "kj": 0.1, "tau_lim": 10.0, "ff": 0}

    def evaluate(self, config, split):
        test = split == "test"
        rng = np.random.default_rng(7 if not test else self.test_noise_seed)
        targets = self.test_targets if test else self.val_targets
        costs, finals = [], []
        for t in targets:
            # reachable check
            if np.linalg.norm(t) > L1 + L2 - 0.02 or np.linalg.norm(t) < abs(L1 - L2) + 0.02:
                continue
            harsh = test or self.robust_val          # robust_val = domain randomisation during search
            spread = 0.40 if test else (0.35 if self.robust_val else 0.15)
            for _ in range(2 if (self.robust_val and not test) else 1):
                ms = rng.uniform(1 - spread, 1 + spread, 2)
                c, f, _ = episode(config, t, self.q0, ms,
                                  noise=0.002 if harsh else 0.0, delay=harsh, rng=rng)
                costs.append(c)
                finals.append(f)
        mean_cost = float(np.mean(costs))
        return {"score": -mean_cost, "cost": mean_cost,
                "final_err_mm": 1000 * float(np.mean(finals)),
                "worst_err_mm": 1000 * float(np.max(finals))}
