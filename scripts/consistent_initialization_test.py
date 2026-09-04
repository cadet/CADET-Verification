#!/usr/bin/env python3
"""
This script implements an artificial chemical equilibrium benchmark.

"""

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import root


# ==========================================================
# Utilities
# ==========================================================

def initial_value_generator(x, distance, seed=None):
    """
    Generate a random positive N-dimensional point exactly
    'distance' away from x.
    """

    rng = np.random.default_rng(seed)

    while True:

        # create normalized random point (1.0 distance from 0.0)
        d = rng.normal(size=len(x))
        d /= np.linalg.norm(d)

        y = x + distance * d

        # ensure positive mass
        if np.all(y > 0):
            return y


# ==========================================================
# Reaction system
# ==========================================================

class ChemicalSystem:

    def __init__(self,
                 n_species,
                 n_reactions,
                 seed=None):

        rng = np.random.default_rng(seed)

        self.n_species = n_species

        if n_reactions >= n_species:
            raise ValueError(
                "Number of reactions must be less than number of species, need at least one conservation equation"
            )

        if n_reactions < 1 or n_species < 3:
            raise ValueError(
                "Invalid number of species or reactions"
            )

        self.n_reactions = n_reactions

        # --------------------------------------------------
        # Random (sparse) stoichiometric matrix
        # --------------------------------------------------

        N = np.zeros((n_species, n_reactions))

        for j in range(n_reactions):

            # choose 2-4 participating species -> sparse system

            k = rng.integers(2, min(5, n_species))

            idx = rng.choice(n_species,
                             size=k,
                             replace=False)

            coeff = rng.integers(-2, 3, size=k)

            coeff[coeff == 0] = 1

            N[idx, j] = coeff

        # remove zero columns

        for j in range(n_reactions):
            if np.all(N[:, j] == 0):
                N[rng.integers(n_species), j] = 1

        self.N = N

        # --------------------------------------------------
        # Left nullspace = conservation matrix
        # --------------------------------------------------

        L = null_space(N.T).T

        self.L = L

        self.n_conservation = L.shape[0]

        # --------------------------------------------------
        # Exact equilibrium
        # --------------------------------------------------

        self.x_eq = np.exp(rng.normal(size=n_species))

        # equilibrium constants

        self.logK = N.T @ np.log(self.x_eq)

        # conserved quantities

        self.b = L @ self.x_eq

    # ------------------------------------------------------

    def residual(self, x):

        # Concentrations must be strictly positive because the
        # mass-action equations use logarithms of the species concentrations.
        # Invalid values are penalized with a large residual so the solver
        # moves away from this region.
        if np.any(x <= 0):
            return np.ones(self.n_species) * 1e20

        # Reaction equilibrium residual:
        #
        # For a reaction such as A + 2B ⇌ C:
        #   K = C / (A * B^2)
        #
        # Taking the logarithm gives:
        #   log(K) = log(C) - log(A) - 2*log(B)
        #
        # The stoichiometric matrix contains these coefficients, so:
        #   N.T @ log(x) = log(K)
        #
        # At equilibrium this residual must be zero.
        r1 = self.N.T @ np.log(x) - self.logK

        # Conservation residual:
        #
        # Chemical reactions only redistribute species; conserved quantities
        # (e.g. total atoms/mass) must remain constant:
        #
        #   L @ x = b
        #
        # where L contains the conservation laws and b the conserved totals.
        # At a valid solution this residual must also be zero.
        r2 = self.L @ x - self.b

        # Combine all equations into one residual vector for the nonlinear solver:
        # first the reaction equilibrium equations, then the conservation equations.
        return np.concatenate((r1, r2))

    # ------------------------------------------------------

    def jacobian(self, x):

        # The Jacobian contains the derivatives of all residual equations
        # with respect to all species concentrations.
        #
        # The first part of the residual is:
        #
        #   r1 = N.T @ log(x) - logK
        #
        # The derivative of log(x) is 1/x, therefore:
        #
        #   dr1/dx = N.T / x
        #
        # This gives the sensitivity of each reaction equilibrium equation
        # to changes in the species concentrations.
        J1 = self.N.T / x

        # The conservation residual is:
        #
        #   r2 = L @ x - b
        #
        # Since this is linear in x, its derivative is simply:
        #
        #   dr2/dx = L
        #
        # The full Jacobian is obtained by stacking the two parts:
        #
        #   J = [ reaction derivatives     ]
        #       [ conservation derivatives ]
        #
        # This matrix is used by Newton-type solvers to find better updates.
        J = np.vstack((J1, self.L))

        return J

# ==========================================================
# SMA binding system
# ==========================================================

class ClassSMA:

    def __init__(self,
                 n_bindings,
                 seed=None):

        rng = np.random.default_rng(seed)

        self.n_bindings = n_bindings

        # ionic capacities
        self.Lambda = 100.0
        self.q_ref = 1.0
        self.c_ref = 1.0

        # SMA parameters
        self.nu = rng.integers(1, 5, size=n_bindings)
        self.sigma = rng.integers(0, 3, size=n_bindings)

        self.Keq = np.exp(rng.normal(size=n_bindings))

        # liquid concentrations (treated as fixed)
        self.c = np.exp(rng.normal(size=n_bindings))
        self.c_salt = 1.0

        # generate an exact equilibrium
        self.q = np.exp(rng.normal(size=n_bindings))

        qbar = self.Lambda - np.sum((self.nu + self.sigma) * self.q)

        # choose q so qbar stays positive
        while qbar <= 0:
            self.q *= 0.5
            qbar = self.Lambda - np.sum((self.nu + self.sigma) * self.q)

        self.q_eq = self.q.copy()

        self.logK = (
            np.log(self.c)
            - self.nu * np.log(self.c_salt)
            - np.log(self.q_eq)
            + self.nu * np.log(qbar)
        )

    def residual(self, q):

        if np.any(q <= 0):
            return np.ones(self.n_bindings) * 1e20

        qbar = self.Lambda - np.sum((self.nu + self.sigma) * q)

        if qbar <= 0:
            return np.ones(self.n_bindings) * 1e20

        r = (
            np.log(q)
            - self.nu * np.log(qbar)
            + self.logK
            - np.log(self.c)
            + self.nu * np.log(self.c_salt)
        )

        return r

    def jacobian(self, q):

        qbar = self.Lambda - np.sum((self.nu + self.sigma) * q)

        J = np.diag(1.0 / q)

        J += np.outer(
            self.nu,
            self.nu + self.sigma
        ) / qbar

        return J


# ==========================================================
# Demo with reactions example
# ==========================================================

def main():

    np.set_printoptions(precision=4,
                        suppress=True)

    system = ChemicalSystem(
        n_species=15,
        n_reactions=14,
        seed=123
    )

    print("=" * 60)
    print("Reaction system")
    print("=" * 60)

    print("Species            :", system.n_species)
    print("Reactions          :", system.n_reactions)
    print("Conservation laws  :", system.n_conservation)
    print("Stoichiometric matrix:\n", system.N)

    print()

    print("Stoichiometric matrix shape:",
          system.N.shape)

    print("Conservation matrix shape:",
          system.L.shape)

    print()

    # --------------------------------------------------

    distance = 2.0

    x0 = initial_value_generator(
        system.x_eq,
        distance,
        seed=123
    )

    print("Initial guess:", x0)

    print("Distance of initial guess:",
          np.linalg.norm(x0 - system.x_eq))

    print()

    # solve the nonlinear system using scipy.optimize.root
    # @Matthias here you'll substitute your own solver
    sol = root(
            system.residual,
            x0,
            jac=system.jacobian,
            method="hybr",
        )

    print("=" * 60)
    print("Solver")
    print("=" * 60)

    print("success :", sol.success)
    print("message :", sol.message)

    print()

    print("Residual norm")
    print(np.linalg.norm(system.residual(sol.x)))

    print()

    print("Distance to exact equilibrium")

    print(np.linalg.norm(sol.x - system.x_eq))

    print()

    print("Relative error")

    print(
        np.linalg.norm(sol.x - system.x_eq)
        /
        np.linalg.norm(system.x_eq)
    )

    print()

    print("Conservation error")

    print(
        np.linalg.norm(
            system.L @ sol.x - system.b
        )
    )

    print()

    print("Mass-action error")

    print(
        np.linalg.norm(
            system.N.T @ np.log(sol.x)
            - system.logK
        )
    )


if __name__ == "__main__":
    main()

# ==========================================================
# Demo with SMA equilibrium example
# ==========================================================

def main():

    np.set_printoptions(
        precision=4,
        suppress=True
    )

    system = ClassSMA(
        n_bindings=10,
        seed=123
    )

    print("=" * 60)
    print("Steric Mass Action (SMA) system")
    print("=" * 60)

    print("Binding components :", system.n_bindings)
    print("Ionic capacity     :", system.Lambda)

    print()

    print("Charges (nu)       :", system.nu)
    print("Steric factors     :", system.sigma)

    print()

    print("Liquid concentrations")
    print(system.c)

    print()

    print("Equilibrium constants")
    print(np.exp(system.logK))

    print()

    # --------------------------------------------------

    distance = 2.0

    q0 = initial_value_generator(
        system.q_eq,
        distance,
        seed=123
    )

    print("Initial guess:")
    print(q0)

    print()

    print("Distance of initial guess:",
          np.linalg.norm(q0 - system.q_eq))

    print()

    # --------------------------------------------------

    sol = root(
        system.residual,
        q0,
        jac=system.jacobian,
        method="hybr",
    )

    print("=" * 60)
    print("Solver")
    print("=" * 60)

    print("success :", sol.success)
    print("message :", sol.message)

    print()

    print("Residual norm")

    print(np.linalg.norm(system.residual(sol.x)))

    print()

    print("Distance to exact equilibrium")

    print(np.linalg.norm(sol.x - system.q_eq))

    print()

    print("Relative error")

    print(
        np.linalg.norm(sol.x - system.q_eq)
        /
        np.linalg.norm(system.q_eq)
    )

    print()

    qbar = (
        system.Lambda
        - np.sum((system.nu + system.sigma) * sol.x)
    )

    print("Free binding sites")

    print(qbar)

    print()

    print("Minimum bound concentration")

    print(np.min(sol.x))


if __name__ == "__main__":
    main()