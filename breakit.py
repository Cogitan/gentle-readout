"""Break-it test for the readability bound (v1.4; Discussion section of the paper).

    python -u breakit.py            # harness check, then Arm A, then Arm B
    -> breakit_results.json next to this script, printed log

Asks whether Gamma_self >= gamma * max(2 kappa^2, eps_KL^2 / 2) survives outside
the hypotheses it is proven under: Arm A uses non-Hermitian meters (a, a^2, the
exponential phase operator E and E^2) at gamma = 0.05; Arm B uses the twelve
Table III Hermitian meters at 4x, 16x and 64x that strength. The rules (harness
tolerance, the 0.8 violation margin, the 1e-6 judging floor) were fixed in
data/breakit/PREREG_2026-10-24.md before the script was written.

Reuses this repository unchanged: the paper's cat (alpha = sqrt(1.9), d = 2) and
GKP (d = 2, N = 80) codes with their own stabilizers, and the Table III
self-rate estimator (autocorr_rates_fixed + stable). Shipped outputs:
data/breakit/. A full run is dense: 56 cells, of which the twelve harness cells
alone take about 20 minutes. Results are cached per cell in breakit_results.json,
so an interrupted run resumes where it stopped.
"""
import json
import os
import sys

import numpy as np
import qutip as qt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from autocorr_extract_d3fix import TABLE2_PRINTED, autocorr_rates_fixed, stable  # noqa: E402
from perception_frontier import cat_code, gkp_code, observables  # noqa: E402

GAMMA0 = 0.05
STRONG = (0.2, 0.8, 3.2)
HERMITIAN = ["parity", "mod-q", "mod-p", "q", "p", "n"]
NONHERM = ["a", "a2", "E", "E2"]
HARNESS_TOL = 0.15
VIOLATION = 0.8          # Gamma_self < 0.8 x bound
BOUND_FLOOR = 1e-6       # below this the bound is not judged


def nonherm_ops(N):
    a = qt.destroy(N)
    E = qt.Qobj(np.diag(np.ones(N - 1), 1).astype(complex))   # sum_n |n><n+1|
    return {"a": a, "a2": a * a, "E": E, "E2": E * E}


def quantities(z_states, L):
    """kappa_g^2 = (|L01|^2 + |L10|^2)/2 and eps_KL^2 = |Lambda_01|^2 / Tr Lambda,
    with Lambda = (P_perp L P)^dag (P_perp L P) on orthonormal read-basis codewords.
    Both reduce to the paper's kappa^2 and eps_KL^2 for Hermitian L."""
    V = np.column_stack([s.full().ravel() for s in z_states])
    V, _ = np.linalg.qr(V)
    Lm = L.full()
    LV = Lm @ V
    PLP = V.conj().T @ LV
    E = LV - V @ PLP
    Lam = E.conj().T @ E
    tr = float(np.real(np.trace(Lam)))
    kg2 = (abs(PLP[0, 1]) ** 2 + abs(PLP[1, 0]) ** 2) / 2
    ekl2 = abs(Lam[0, 1]) ** 2 / tr if tr > 0 else 0.0
    return float(kg2), float(ekl2)


def cell(code, zl, L, gamma):
    c_ops = code["stab"] + [np.sqrt(gamma) * L]
    g, _ = autocorr_rates_fixed(c_ops, zl, code["Pcode"], 2)
    s, flag = stable(g)
    kg2, ekl2 = quantities(code["z_states"], L)
    bound = gamma * max(2 * kg2, ekl2 / 2)
    unstable = bool(flag.strip())
    judged = (not unstable) and bound > BOUND_FLOOR
    return {"gamma": gamma, "gamma_self": s, "windows": g, "unstable": unstable,
            "kappa_g2": kg2, "eps_kl2": ekl2, "bound": bound,
            "kappa_term": gamma * 2 * kg2, "kl_term": gamma * ekl2 / 2,
            "ratio": s / bound if bound > 0 else None, "judged": judged,
            "violates": judged and s < VIOLATION * bound}


def line(tag, r):
    v = "VIOLATES" if r["violates"] else ("ok" if r["judged"] else "not judged")
    print(f"{tag:<26s} g={r['gamma']:<5g} Gamma_self={r['gamma_self']:.3e} bound={r['bound']:.3e} "
          f"(kappa {r['kappa_term']:.2e}, KL {r['kl_term']:.2e}) "
          f"ratio={r['ratio'] if r['ratio'] is None else round(r['ratio'], 3)} "
          f"{'UNSTABLE ' if r['unstable'] else ''}{v}", flush=True)


def main():
    out_path = os.path.join(HERE, "breakit_results.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}

    def save():
        json.dump(res, open(out_path, "w"), indent=1)

    codes = [("cat", cat_code(np.sqrt(1.9), 2), 2), ("GKP", gkp_code(2, 80), 1)]

    # harness: the twelve Table III cells at gamma = 0.05
    print("== HARNESS: Table III (tab:bound) at gamma = 0.05", flush=True)
    ok = True
    for short, code, zli in codes:
        obs, _ = observables(code["N"], 2)
        for m in HERMITIAN:
            key = f"harness|{short}|{m}"
            if key not in res:
                res[key] = cell(code, code["logops"][zli], obs[m], GAMMA0)
                save()
            r = res[key]
            pr = TABLE2_PRINTED[(short, m)]
            dev = r["gamma_self"] / pr - 1
            r["printed"], r["dev_vs_printed"] = pr, dev
            line(f"{short}/{m}", r)
            if abs(dev) > HARNESS_TOL:
                ok = False
                print(f"   HARNESS MISS: {dev:+.1%} vs printed {pr:.1e}", flush=True)
    res["harness_pass"] = ok
    save()
    if not ok:
        print("HARNESS FAILED: stopping (PREREG).", flush=True)
        return

    print("== ARM A: non-Hermitian meters at gamma = 0.05", flush=True)
    for short, code, zli in codes:
        ops = nonherm_ops(code["N"])
        for m in NONHERM:
            key = f"A|{short}|{m}"
            if key not in res:
                res[key] = cell(code, code["logops"][zli], ops[m], GAMMA0)
                save()
            line(f"A {short}/{m}", res[key])

    print("== ARM B: Hermitian Table III meters at strong gamma", flush=True)
    for short, code, zli in codes:
        obs, _ = observables(code["N"], 2)
        for m in HERMITIAN:
            for gm in STRONG:
                key = f"B|{short}|{m}|{gm}"
                if key not in res:
                    res[key] = cell(code, code["logops"][zli], obs[m], gm)
                    save()
                line(f"B {short}/{m}", res[key])

    a_cells = [v for k, v in res.items() if k.startswith("A|")]
    b_cells = [v for k, v in res.items() if k.startswith("B|")]
    res["summary"] = {
        "arm_A_judged": sum(c["judged"] for c in a_cells),
        "arm_A_violations": [k for k, v in res.items() if k.startswith("A|") and v["violates"]],
        "arm_B_judged": sum(c["judged"] for c in b_cells),
        "arm_B_violations": [k for k, v in res.items() if k.startswith("B|") and v["violates"]],
    }
    save()
    print("== SUMMARY", json.dumps(res["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
