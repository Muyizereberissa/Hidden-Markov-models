"""Gaussian Hidden Markov Model (diagonal covariance) with Viterbi and Baum-Welch."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

LOG_2PI = np.log(2.0 * np.pi)


def logsumexp(a: NDArray[np.floating], axis: int | None = None) -> NDArray[np.floating]:
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    if axis is not None:
        return np.squeeze(out, axis=axis)
    return np.squeeze(out)


def normalize_rows(x: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    s = x.sum(axis=1, keepdims=True)
    s[s < eps] = 1.0
    return x / s


def normalize_vec(x: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    s = x.sum()
    if s < eps:
        return np.full_like(x, 1.0 / len(x))
    return x / s


class GaussianHMM:
    def __init__(self, n_states: int, n_features: int, random_state: int = 42):
        self.n_states = n_states
        self.n_features = n_features
        self.rng = np.random.default_rng(random_state)
        self.pi = normalize_vec(np.ones(n_states))
        self.A = normalize_rows(np.full((n_states, n_states), 1.0 / n_states))
        self.means = self.rng.normal(size=(n_states, n_features))
        self.vars = np.ones((n_states, n_features))

    def _log_emission(self, obs: NDArray[np.floating]) -> NDArray[np.floating]:
        # obs: (T, D) -> (T, N)
        diff = obs[:, None, :] - self.means[None, :, :]
        inv_var = 1.0 / np.maximum(self.vars[None, :, :], 1e-6)
        log_det = np.sum(np.log(np.maximum(self.vars, 1e-6)), axis=1)
        quad = np.sum(diff * diff * inv_var, axis=2)
        return -0.5 * (self.n_features * LOG_2PI + log_det[None, :] + quad)

    def _forward_backward(self, log_b: NDArray[np.floating]) -> tuple[NDArray[np.floating], float, NDArray[np.floating], NDArray[np.floating]]:
        t_len, n = log_b.shape
        log_pi = np.log(np.maximum(self.pi, 1e-12))
        log_a = np.log(np.maximum(self.A, 1e-12))

        alpha = np.zeros((t_len, n))
        alpha[0] = log_pi + log_b[0]
        for t in range(1, t_len):
            alpha[t] = log_b[t] + logsumexp(alpha[t - 1][:, None] + log_a, axis=0)

        beta = np.zeros((t_len, n))
        for t in range(t_len - 2, -1, -1):
            beta[t] = logsumexp(log_a + log_b[t + 1] + beta[t + 1], axis=1)

        log_likelihood = float(logsumexp(alpha[-1]))
        gamma = np.exp(alpha + beta - log_likelihood)

        xi = np.zeros((t_len - 1, n, n))
        for t in range(t_len - 1):
            log_xi = (
                alpha[t][:, None]
                + log_a
                + log_b[t + 1][None, :]
                + beta[t + 1][None, :]
                - log_likelihood
            )
            xi[t] = np.exp(log_xi)
        return gamma, log_likelihood, xi, alpha

    def fit(
        self,
        sequences: list[NDArray[np.floating]],
        max_iter: int = 80,
        tol: float = 1e-4,
    ) -> list[float]:
        log_liks: list[float] = []
        prev = -np.inf
        for _ in range(max_iter):
            exp_counts = np.zeros(self.n_states)
            exp_start = np.zeros(self.n_states)
            exp_trans = np.zeros((self.n_states, self.n_states))
            exp_mean = np.zeros((self.n_states, self.n_features))
            exp_var = np.zeros((self.n_states, self.n_features))
            total_ll = 0.0

            for obs in sequences:
                log_b = self._log_emission(obs)
                gamma, ll, xi, _ = self._forward_backward(log_b)
                total_ll += ll
                exp_counts += gamma.sum(axis=0)
                exp_start += gamma[0]
                exp_trans += xi.sum(axis=0)
                exp_mean += gamma.T @ obs
                exp_var += gamma.T @ (obs ** 2)

            exp_counts = np.maximum(exp_counts, 1e-6)
            self.pi = normalize_vec(exp_start)
            self.A = normalize_rows(exp_trans + 1e-6)
            self.means = exp_mean / exp_counts[:, None]
            second_moment = exp_var / exp_counts[:, None]
            self.vars = np.maximum(second_moment - self.means ** 2, 1e-3)

            log_liks.append(total_ll)
            if len(log_liks) > 1 and abs(log_liks[-1] - prev) < tol:
                break
            prev = log_liks[-1]
        return log_liks

    def predict(self, obs: NDArray[np.floating]) -> NDArray[np.int64]:
        log_b = self._log_emission(obs)
        t_len, n = log_b.shape
        log_pi = np.log(np.maximum(self.pi, 1e-12))
        log_a = np.log(np.maximum(self.A, 1e-12))

        delta = np.zeros((t_len, n))
        psi = np.zeros((t_len, n), dtype=int)
        delta[0] = log_pi + log_b[0]
        for t in range(1, t_len):
            scores = delta[t - 1][:, None] + log_a
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = log_b[t] + np.max(scores, axis=0)

        path = np.zeros(t_len, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(t_len - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def score(self, obs: NDArray[np.floating]) -> float:
        log_b = self._log_emission(obs)
        _, ll, _, _ = self._forward_backward(log_b)
        return ll
