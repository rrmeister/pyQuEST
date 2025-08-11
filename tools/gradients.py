import numpy as np
from pyquest import Register
from pyquest import Circuit
from pyquest.unitaries import Ry, Rx, Rz, X, Y, Z, H

from typing import Callable

Observable = Callable[[Register], Register]


class Gradients:
    r"""Functionality to compute gradients."""

    def gradient(
        self, circ: Circuit, observable: Callable[[Register], Register]
    ) -> Register:
        r"""
        Compute the unnormalized reverse-mode gradient.

        This method implements Algorithm 1 from:
        "Efficient Classical Calculation of the Quantum Natural Gradient" (arXiv:2011.02991),
        specifically the calculation of the standard gradient ∇E(θ), using a backpropagation-like
        traversal of the circuit in reverse order.

        Args:
            circ (Circuit): A pyQuEST Circuit object containing differentiable gates (e.g., Ry, Rx, Rz).
            observable (Callable[[Register], Register]): A function that applies the observable
                (e.g., Z(0)) to a given Register and returns a new Register representing O|ψ⟩.

        Returns:
            Register: A quantum Register object representing the unnormalized state-vector
            direction of the gradient ∇E(θ) in Hilbert space.

        Side Effects:
            - Sets `self.grad_vec` to a NumPy array of scalar gradient components ∂E/∂θ_i.
            - Sets `self.gradient_register` to the full unnormalized gradient Register.

        """
        n_qbits = self.infer_num_qubits(circ)
        lambda_state = Register(n_qbits)
        lambda_state.apply_circuit(circ)
        phi_state = Register(copy_reg=lambda_state)
        lambda_state = observable(lambda_state)
        gradient_register = Register(n_qbits)
        gradient_register.init_blank_state()

        deriv_states = []
        grad_vec = []
        for n, adj_gate in enumerate(circ.inverse):
            orig_gate = adj_gate.inverse
            phi_state.apply_operator(adj_gate)
            mu_state = Register(copy_reg=phi_state)
            mu_state = self.unitary_differentiation(mu_state, orig_gate)
            overlap = 2 * (lambda_state * mu_state).real
            if n < len(circ) - 1:
                lambda_state.apply_operator(adj_gate)
            gradient_register += overlap * mu_state
            deriv_states.append(mu_state)
            grad_vec.append(overlap)

        self.deriv_states = deriv_states[::-1]
        self.grad_vec = np.array(grad_vec[::-1])
        self.gradient_register = gradient_register

        return gradient_register

    def nat_gradient(
        self, circ: Circuit, observable: Callable[[Register], Register]
    ) -> Register:
        r"""
        Compute the unnormalized quantum natural gradient direction in state-vector form.

        This method implements the full Algorithm 1 from the paper:
        "Efficient Classical Calculation of the Quantum Natural Gradient" (arXiv:2011.02991),
        by computing both the gradient ∇E(θ) and the quantum Fisher information matrix G(θ),
        and returning the preconditioned gradient vector G⁻¹ ∇E(θ) as a quantum Register.

        Args:
            circ (Circuit): A pyQuEST Circuit object consisting of differentiable gates.
            observable (Callable[[Register], Register]): A function that applies a Hermitian
                observable (e.g., Z(0)) to a given Register and returns the result of O|ψ⟩.

        Returns:
            Register: A Register representing the unnormalized direction of the natural gradient
            in Hilbert space, corresponding to G⁻¹ ∇E(θ).

        Side Effects:
            - Calls and caches `self.gradient_register` and `self.grad_vec` from `gradient(...)`.
            - Computes and caches `self.G` using `fisher_information_matrix(...)`.
            - Caches `self.natural_gradient_register`.
        """
        # compute gradients.
        self.gradient(circ, observable)
        self.fisher_information_matrix(circ)

        # compute coefficients.
        G_inv = np.linalg.pinv(self.G)
        coeffs = G_inv @ self.grad_vec

        # find natural gradient register.
        n_qbits = self.infer_num_qubits(circ)
        natural_gradient_register = Register(n_qbits)
        natural_gradient_register.init_blank_state()
        for coeff, mu in zip(coeffs, self.deriv_states):
            natural_gradient_register += coeff * mu

        self.natural_gradient_register = natural_gradient_register
        return natural_gradient_register

    def infer_num_qubits(self, circ: Circuit) -> int:
        r"""
        Infer the number of qubits required to run a given circuit.

        This utility function inspects all gates in the circuit and determines the
        highest qubit index used (including targets and controls). It assumes qubit
        indices are zero-based and returns one more than the highest index found.

        Args:
            circ (Circuit): A pyQuEST Circuit object containing gates applied to qubits.

        Returns:
            int: The inferred total number of qubits used in the circuit.
        """
        max_index = 0
        for gate in circ:
            indices = []
            if hasattr(gate, "target"):
                indices.append(gate.target)
            if hasattr(gate, "controls"):
                indices.extend(gate.controls or [])
            if hasattr(gate, "targets"):
                indices.extend(gate.targets or [])
            if indices:
                max_index = max(max_index, max(indices))
        return max_index + 1

    def unitary_differentiation(
        self, mu_state: Register, orig_gate: object
    ) -> Register:
        r"""
        Apply the derivative of a parameterized gate to a quantum state.

        This function implements analytic differentiation rules for supported
        single-qubit rotation gates (Ry, Rx, Rz) based on their generator
        (Y, X, Z respectively). The result is a new quantum state representing
        the action of dU/dθ on the input state.

        Args:
            mu_state (Register): A quantum register to be modified in-place.
                Should be a copy of the current state before the target gate is applied.
            orig_gate (Unitary): The original parameterized gate to differentiate.
                Supported gates: Ry, Rx, Rz.

        Raises:
            NotImplementedError: Unsupported gates will raise.

        Returns:
            Register: The modified register representing (dU/dθ)·|ψ⟩ (unnormalized).
        """
        q = getattr(orig_gate, "target", None)

        # Parameterized single-qubit rotations.
        if isinstance(orig_gate, Ry):
            mu_state.apply_operator(orig_gate)
            mu_state.apply_operator(Y(q))
            mu_state = (-1j / 2) * mu_state
        elif isinstance(orig_gate, Rx):
            mu_state.apply_operator(orig_gate)
            mu_state.apply_operator(X(q))
            mu_state = (-1j / 2) * mu_state
        elif isinstance(orig_gate, Rz):
            mu_state.apply_operator(orig_gate)
            mu_state.apply_operator(Z(q))
            mu_state = (-1j / 2) * mu_state
        # Non-parameterized gates — gradient is zero direction.
        elif isinstance(orig_gate, (X, H)):
            mu_state.init_blank_state()
        else:
            raise NotImplementedError(
                f"Gradient not implemented for gate: {type(orig_gate)}"
            )
        return mu_state

    def fisher_information_matrix(self, circ: Circuit) -> np.ndarray:
        r"""
        Compute the Fisher Information Matrix, for a given parameterized circuit.

        This method implements Algorithm 1 from:
        "Efficient Classical Calculation of the Quantum Natural Gradient" (arXiv:2011.02991),
        which computes the full QGT using reverse-mode simulation with O(P²) complexity.

        Args:
            circ (Circuit): A pyQuEST circuit consisting of P parameterized gates.

        Returns:
            np.ndarray: A real-valued (P x P) Fisher Information Matrix G,
                        where G[i,j] encodes the inner product between partial
                        derivatives of the circuit with respect to θᵢ and θⱼ.
        """
        n_qbits = self.infer_num_qubits(circ)

        chi = Register(n_qbits)  # Automatically initialized to |0...0>
        chi.apply_operator(circ[0])
        psi = Register(copy_reg=chi)
        phi = Register(n_qbits)
        first_gate = circ[0]
        phi = self.unitary_differentiation(phi, first_gate)

        P = len(circ)
        T = np.zeros(P, dtype=complex)
        L = np.zeros((P, P), dtype=complex)
        T[0] = chi * phi
        L[0, 0] = phi * phi

        for j, orig_gate in enumerate(circ):
            if j == 0:
                continue  # Already handled j=0 outside the loop

            lam = Register(copy_reg=psi)
            phi = Register(copy_reg=psi)
            phi = self.unitary_differentiation(phi, orig_gate)
            L[j, j] = (phi * phi).real

            for i in reversed(range(j)):
                phi.apply_operator(circ[i + 1].inverse)
                lam.apply_operator(circ[i].inverse)
                mu = Register(copy_reg=lam)
                mu = self.unitary_differentiation(mu, circ[i])
                L[i, j] = (mu * phi).real

            T[j] = chi * phi

        G = np.zeros((P, P), dtype=complex)
        for i in range(P):
            for j in range(P):
                if i <= j:
                    G[i, j] = L[i, j] - np.conj(T[i]) * T[j]
                else:
                    G[i, j] = L[j, i] - np.conj(T[i]) * T[j]
        G = G.real

        self.G = G
        return G


# Use Case:
g = Gradients()
circ = Circuit([Ry(0, 0.5), Rx(1, 0.3)])


def observable(register):
    reg = Register(copy_reg=register)
    reg.apply_operator(Z(0))
    return reg


g.nat_gradient(circ, observable)
print(f"gradient_register:")
print(g.gradient_register[:])
print()

print(f"natural_gradient_register:")
print(g.natural_gradient_register[:])
