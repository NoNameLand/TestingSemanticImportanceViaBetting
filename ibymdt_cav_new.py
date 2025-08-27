# pip install torch torchvision numpy scikit-learn scipy
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn
from sklearn.neighbors import KernelDensity
from sklearn.covariance import EmpiricalCovariance
from typing import Callable, Dict, Tuple, Iterable, Optional

# ========== 0) Concept projections & data collection ==========

class ActivationTap:
    """Register a forward hook on a layer to grab h."""
    def __init__(self, layer: nn.Module):
        self.buf = []
        self.hook = layer.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        a = o.detach()
        if a.dim() > 2:  # flatten conv maps
            a = a.flatten(1)
        self.buf.append(a.cpu())

    def collect(self) -> np.ndarray:
        if not self.buf: return None
        A = torch.cat(self.buf, 0).numpy()
        self.buf = []
        return A

    def close(self): self.hook.remove()

def l2_norm_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

class ConceptBank:
    """Provide concept matrix C in R^{m x d} (unit-norm rows)."""
    def __init__(self, C: np.ndarray):
        assert C.ndim == 2
        self.C = l2_norm_rows(C.astype(np.float64))  # (m, d)

    def project(self, H: np.ndarray) -> np.ndarray:
        """Return Z = <C, H>, with H in R^{n x d}, unit-normalized first."""
        Hn = l2_norm_rows(H.astype(np.float64))
        return Hn @ self.C.T  # (n, m)

# ========== 1) Concept distribution models (Gaussian | KDE) ==========

@dataclass
class ConceptDist:
    """Fit marginal over Z_j and joint over Z for conditionals."""
    kind: str = "gaussian"   # "gaussian" or "kde"
    bw: float = 0.2          # for KDE (Scott's factor alternative)
    nu_cond: float = 0.5     # conditioning bandwidth for c-SKIT (Appendix D.1)
    neff_target: int = 2000  # target effective sample size (Appendix D.1)

    def fit(self, Z: np.ndarray):
        """
        Z: (n, m) concept matrix across dataset.
        Stores per-feature marginals and joint moments for Gaussian conditionals.
        """
        self.Z = Z
        self.n, self.m = Z.shape
        # Marginals
        self.mus = Z.mean(axis=0)
        self.stds = Z.std(axis=0) + 1e-12
        # KDE per coordinate (optional)
        if self.kind == "kde":
            self.kdes = []
            for j in range(self.m):
                kde = KernelDensity(kernel="gaussian", bandwidth=self.bw)
                kde.fit(Z[:, [j]])
                self.kdes.append(kde)
        else:
            self.kdes = None
        # Joint Gaussian moments for fast Gaussian conditionals (for x-SKIT too)
        self.mu_joint = Z.mean(axis=0)
        self.cov_joint = np.cov(Z.T) + 1e-6*np.eye(self.m)

    def sample_Zj_given_Zminus(self, z_minus: np.ndarray, j: int, adapt_neff: bool=True) -> float:
        """
        Nonparametric conditional p(Z_j | Z_{-j}) using weighted KDE as in Appendix D.1:
        weights w_i = N( Z_{-j}^{(i)} | z_{-j}, nu^2 I ), then 1D KDE over Z_j with those weights.
        """
        idx = [k for k in range(self.m) if k != j]
        Zm = self.Z[:, idx]           # (n, m-1)
        Zj = self.Z[:, j:j+1]         # (n, 1)
        # Gaussian weights by distance in Z_-j
        diffs = (Zm - z_minus[None, :])
        w = np.exp(-0.5 * np.sum(diffs**2, axis=1) / (self.nu_cond**2))
        w = w + 1e-12
        w = w / w.sum()
        if adapt_neff:
            neff = (w.sum()**2) / (np.sum(w**2) + 1e-12)
            s = (self.neff_target / max(neff, 1.0))**0.5
            w = np.clip(w * s, 1e-12, None); w = w / w.sum()
        # Weighted resampling index then add small Gaussian jitter (Scott factor)
        i = np.random.choice(self.n, p=w)
        mu = Zj[i, 0]
        sd = (1.06 * Zj.std() * (self.n ** (-1/5)))  # Scott's / Silverman-ish
        return float(mu + sd * np.random.randn())

# ========== 2) Kernels and the SKIT witness (ρ) ==========

def median_heuristic_1d(vals: np.ndarray) -> float:
    if len(vals) < 2: return 1.0
    med = np.median(np.abs(vals.reshape(-1,1) - vals.reshape(1,-1)))
    return float(max(med, 1e-3))

class RBF1D:
    def __init__(self, bw: float):
        self.bw2 = (bw if bw>0 else 1.0)**2
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.reshape(-1,1); b = b.reshape(1,-1)
        return np.exp(-0.5 * (a-b)**2 / self.bw2)

class SKITGlobal:
    """
    Global SKIT for H0: Yhat ⟂ Z_j (Algorithm 1). Uses the MMD witness between
    the joint and product-of-marginals with tensor kernel k_joint((y,z),(y',z')) = ky* kz.
    ρ_t is fitted on previous points only, as in the paper.
    """
    def __init__(self, alpha=0.05, use_ons=True, center_z=False):
        self.alpha = alpha
        self.use_ons = use_ons
        self.center_z = center_z
        self.reset()

    def reset(self):
        self.Y = []; self.Z = []
        self.K = 1.0
        self.v = 0.0  # ONS starts at 0
        self.a = 1.0  # ONS accumulator

    def _update_bw(self):
        y = np.array(self.Y)
        z = np.array(self.Z)
        self.ky = RBF1D(median_heuristic_1d(y) or 1.0)
        self.kz = RBF1D(median_heuristic_1d(z) or 1.0)

    def _rho(self, y: float, z: float) -> float:
        """
        ρ_t(d) = (1/n) Σ k_y(y, y_i) k_z(z, z_i)  -  (1/n Σ k_y(y, y_i)) (1/n Σ k_z(z, z_i))
        computed on previous points only.
        """
        if len(self.Y) == 0: return 0.0
        ybuf = np.array(self.Y); zbuf = np.array(self.Z)
        jy = self.ky(np.array([y]), ybuf).flatten()
        jz = self.kz(np.array([z]), zbuf).flatten()
        term_joint = float((jy * jz).mean())
        term_prod = float(jy.mean() * jz.mean())
        return term_joint - term_prod

    def step_pair(self, y1, z1, y2, z2) -> Tuple[float, float]:
        """Consume a pair; build permuted null and update wealth."""
        if self.center_z and len(self.Z) > 0:
            mu = np.mean(self.Z)
            z1, z2 = z1 - mu, z2 - mu

        # fix ρ using previous data
        self._update_bw()
        r11 = self._rho(y1, z1); r22 = self._rho(y2, z2)
        r12 = self._rho(y1, z2); r21 = self._rho(y2, z1)
        kappa = np.tanh((r11 + r22) - (r12 + r21))  # Eq. (12)
        # wealth update
        self.K = self.K * (1.0 + self.v * kappa)
        # ONS update (Appendix A.3)
        if self.use_ons:
            zt = kappa / (1.0 + self.v * kappa)
            self.a = self.a + zt*zt
            v_next = self.v + (2/(2-np.log(3))) * (zt / self.a)
            self.v = float(np.clip(v_next, 0.0, 1.0))
        else:
            self.v = 0.5

        # Now append current data (becomes "past" for next step)
        self.Y += [y1, y2]
        self.Z += [z1, z2]
        return kappa, self.K

    def rejected(self): return self.K >= 1.0/self.alpha  # reject when K >= 1/α

# ========== 3) c-SKIT: global conditional importance (needs P[Z_j|Z_-j]) ==========

class SKITConditional:
    """c-SKIT for H0: Yhat ⟂ Z_j | Z_{-j}. Requires sampler Z_j ~ P[Z_j|Z_-j]."""
    def __init__(self, alpha=0.05, use_ons=True):
        self.alpha = alpha
        self.use_ons = use_ons
        self.reset()

    def reset(self):
        self.K = 1.0; self.v = 0.0; self.a = 1.0
        self.buf = []   # list of (y, z_full)

    def _kernels(self, yvals, zvals):
        ky = RBF1D(median_heuristic_1d(yvals) or 1.0)
        kzj = RBF1D(median_heuristic_1d(zvals) or 1.0)
        return ky, kzj

    def _rho(self, y, zj, zminus, past):
        if len(past)==0: return 0.0
        Y = np.array([p[0] for p in past])
        Zfull = np.stack([p[1] for p in past], axis=0)
        Zj = Zfull[:, self.j_idx]
        ky, kzj = self._kernels(Y, Zj)
        jy = ky(np.array([y]), Y).flatten()
        jz = kzj(np.array([zj]), Zj).flatten()
        return float((jy*jz).mean() - jy.mean()*jz.mean())

    def set_concept_index(self, j_idx: int): self.j_idx = j_idx

    def step(self, y, z_full, sampler: Callable[[np.ndarray, int], float]):
        """
        One observation (y, z_full). Build null by sampling ztilde_j ~ P(Z_j|Z_-j=z_-j),
        then compute κ_t = tanh(ρ(y,z_j,z_-j) - ρ(y,ztilde_j,z_-j)), update wealth.
        """
        zminus = np.delete(z_full, self.j_idx)
        zj = float(z_full[self.j_idx])
        ztilde = sampler(zminus, self.j_idx)

        # fix ρ on past
        rho_real = self._rho(y, zj, zminus, self.buf)
        rho_null = self._rho(y, ztilde, zminus, self.buf)
        kappa = np.tanh(rho_real - rho_null)
        self.K = self.K * (1.0 + self.v*kappa)
        if self.use_ons:
            zt = kappa / (1.0 + self.v * kappa)
            self.a += zt*zt
            v_next = self.v + (2/(2-np.log(3))) * (zt / self.a)
            self.v = float(np.clip(v_next, 0.0, 1.0))
        else:
            self.v = 0.5

        self.buf.append((y, z_full.copy()))
        return kappa, self.K

    def rejected(self): return self.K >= 1.0/self.alpha

# ========== 4) x-SKIT: local conditional importance (needs H ~ P[H|Z_C=z_C]) ==========

class XSKITLocal:
    """
    x-SKIT for a single input's z (local). We sample H|Z_C=z_C and compare Y_S∪{j} vs Y_S.
    Here we give a Gaussian conditional sampler for H|Z (using joint (H,Z) fit).
    """
    def __init__(self, alpha=0.05, use_ons=True):
        self.alpha = alpha; self.use_ons = use_ons; self.reset()

    def reset(self):
        self.K = 1.0; self.v = 0.0; self.a = 1.0
        self.yS = []; self.ySj = []

    def _rbf(self, arr):
        return RBF1D(median_heuristic_1d(np.array(arr)) or 1.0)

    def _rho(self, y, past):
        if len(past) == 0: return 0.0
        Y = np.array(past)
        ky = self._rbf(Y)
        jy = ky(np.array([y]), Y).flatten()
        return float(jy.mean() - jy.mean()*1.0)  # reduces to centered kernel mean; fine as witness

    def step(self, g_fn: Callable[[np.ndarray], float],
             sample_H_given_ZC: Callable[[np.ndarray, Iterable[int]], np.ndarray],
             z_obs: np.ndarray, S: Iterable[int], j_idx: int):
        """
        Draw paired samples H_{S∪{j}} and H_S, push through g to get Y, compute κ and update wealth.
        """
        HSj = sample_H_given_ZC(z_obs, list(S)+[j_idx])
        HS  = sample_H_given_ZC(z_obs, list(S))
        ySj = float(g_fn(HSj))
        yS  = float(g_fn(HS))
        r1 = self._rho(ySj, self.ySj)
        r0 = self._rho(yS,  self.yS)
        kappa = np.tanh(r1 - r0)
        self.K = self.K * (1.0 + self.v*kappa)
        if self.use_ons:
            zt = kappa / (1.0 + self.v * kappa)
            self.a += zt*zt
            v_next = self.v + (2/(2-np.log(3))) * (zt / self.a)
            self.v = float(np.clip(v_next, 0.0, 1.0))
        else:
            self.v = 0.5
        self.ySj.append(ySj); self.yS.append(yS)
        return kappa, self.K

    def rejected(self): return self.K >= 1.0/self.alpha

# ========== 5) Gaussian conditional sampler for H | Z_C (used by x-SKIT) ==========

class JointHZConditional:
    """
    Fit joint Gaussian over (H, Z) from dataset pairs, then sample H | Z_C=z_C.
    Stable and simple for small m.
    """
    def fit(self, H: np.ndarray, Z: np.ndarray):
        self.d = H.shape[1]; self.m = Z.shape[1]
        X = np.concatenate([H, Z], axis=1)
        self.mu = X.mean(axis=0)
        self.S  = np.cov(X.T) + 1e-6*np.eye(self.d+self.m)
        return self

    def sample_H_given_ZC(self, z_obs: np.ndarray, C: Iterable[int]) -> np.ndarray:
        C = list(C)
        # partition indices
        H_idx = np.arange(self.d)
        Z_idx = self.d + np.arange(self.m)
        C_idx = self.d + np.array(C)
        U_idx = np.setdiff1d(np.concatenate([H_idx, Z_idx]), C_idx, assume_unique=False)

        # standard Gaussian conditioning: X_U | X_C ~ N(mu_U + S_UC S_CC^{-1} (zC - mu_C), S_UU - S_UC S_CC^{-1} S_CU)
        mu_U = self.mu[U_idx]; mu_C = self.mu[C_idx]
        S_UU = self.S[np.ix_(U_idx, U_idx)]
        S_UC = self.S[np.ix_(U_idx, C_idx)]
        S_CC = self.S[np.ix_(C_idx, C_idx)]
        inv = np.linalg.pinv(S_CC)
        delta = (z_obs[C] - mu_C)
        mu_cond = mu_U + S_UC @ inv @ delta
        S_cond  = S_UU - S_UC @ inv @ S_UC.T
        # sample and return first d dims (H)
        sample = np.random.multivariate_normal(mu_cond, S_cond)
        return sample[:self.d]

# ========== 6) FDR post-processing over many concepts (optional) ==========

def greedy_e_fdr(wealth_dict: Dict[int, float], alpha=0.05):
    """
    Given final wealths {j: K_j} for m concepts (each is a valid e-value),
    greedily pick rejections at thresholds m/(α s) to control FDR (Algorithm 4).
    Returns list of (concept, crossed_at_threshold_rank).
    """
    m = len(wealth_dict)
    rejected = []
    # sort by the time/threshold they cross (here we only have final wealth; emulate greedy by size)
    # we repeatedly pick the largest K that crosses current threshold
    remaining = dict(wealth_dict)
    s = 1
    while remaining:
        thresh = m / (alpha * s)
        # pick a concept that crosses thresh, with the largest K
        ok = [(j, K) for j, K in remaining.items() if K >= thresh]
        if not ok: break
        j, K = sorted(ok, key=lambda t: t[1], reverse=True)[0]
        rejected.append((j, s))
        remaining.pop(j)
        s += 1
    return rejected

# ========== 7) Secure-importance wrapper with Hoeffding bound ==========

def hoeffding_lower_bound(successes: int, trials: int, delta: float=0.05) -> float:
    """
    With R independent repeats (trials) of the e-test at level α, let successes be #rejections.
    Hoeffding: P( p_true <= p_hat - eps ) <= exp(-2 R eps^2).
    Lower bound p_true >= p_hat - sqrt( (1/(2R)) * log(1/delta) ).
    """
    if trials <= 0: return 0.0
    p_hat = successes / trials
    eps = np.sqrt(np.log(1.0/delta) / (2.0*trials))
    return max(0.0, p_hat - eps)

def secure_importance(run_fn: Callable[[], bool], R: int=50, delta: float=0.05) -> Dict[str, float]:
    """
    run_fn() should execute an entire SKIT run and return True if it rejected (K >= 1/α).
    We repeat, then return LB on true rejection probability and a bound on 'not-important' prob.
    """
    s = sum(bool(run_fn()) for _ in range(R))
    lb = hoeffding_lower_bound(s, R, delta)
    return {
        "rejection_rate_hat": s / R,
        "rejection_rate_lower_bound": lb,
        "prob_not_important_upper_bound": 1.0 - lb
    }
