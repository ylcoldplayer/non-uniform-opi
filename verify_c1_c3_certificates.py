#!/usr/bin/env python3
"""
Finite certificates C1--C3 for the nonuniform OPI counterexample.

This verifier is intended to be paired with verify_c4plus_section_interval.py.
It proves the finite certificates that do not require the positive-radius
Poincare-section contraction box:

  C1. Krawczyk certificate for the algebraic orbit equations G(y)=0 on
      B = ybar + [-1e-30,1e-30]^9.
      It computes G(ybar), DG(ybar) exactly over QQ, constructs a fixed
      80-digit rational-decimal preconditioner A approximating
      DG(ybar)^{-1}, interval-encloses DG(B), and prints the Krawczyk
      inclusion bounds. Krawczyk's theorem does not require A to be the
      exact inverse.

  C2. Guard-sign certificate along all six orbit segments.  It evaluates the
      degree-40 Bernstein coefficients of the signed guard polynomials over
      the orbit box B.  Endpoint coefficients corresponding to the intended
      entry/exit roots are allowed to contain zero; all other Bernstein
      coefficients are certified positive.

  C3. One-way transversality / no-sliding certificate.  It interval-encloses
      n_k^T f_{mu_k}(x_{k+1}) and n_k^T f_{mu_{k+1}}(x_{k+1}) over B and
      checks that incoming and outgoing normal velocities have the same
      nonzero sign.

The code parses all MDP data as exact rational decimals.  Exact algebra is used for C1 center quantities and for all MDP input
parsing; outward-rounded Decimal interval arithmetic is used for B-dependent
enclosures.
"""
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import json
import math
import importlib.util
import mpmath as mp

BASE_PATH = Path(__file__).with_name("verify_c4plus_decimal_interval.py")
spec = importlib.util.spec_from_file_location("base_c4", str(BASE_PATH))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

Interval = base.Interval
F = base.F

# Basic data from the shared base verifier.
alpha_q = base.alpha_q
p_q = base.p_q
g_q = base.g_q
P_q = base.P_q
policies = base.policies
guard_indices = base.guard_indices
x0_str = base.x0_str
z_str = base.z_str
Jpol_q = base.Jpol_q
Jpol = base.Jpol
alpha = base.alpha
P = base.P
g = base.g
p = base.p

exponents = [1, 9, 40]
rho_str = "1e-30"
rho_dec = base.decimal_exact(rho_str)

# ---------------------------------------------------------------------------
# Exact rational functions for C1 center and derivative.
# ---------------------------------------------------------------------------

def qdiff_fraction(J):
    out = []
    for s in range(3):
        v = g_q[s][1] - g_q[s][0]
        for j in range(3):
            v += alpha_q * (P_q[s][1][j] - P_q[s][0][j]) * J[j]
        out.append(v)
    return out


def grad_guard_fraction(s):
    return [alpha_q * (P_q[s][1][j] - P_q[s][0][j]) for j in range(3)]


def flow_fraction(mu, X, z):
    J = Jpol_q[mu]
    return [J[i] + (X[i] - J[i]) * (z ** exponents[i]) for i in range(3)]


def compute_G_and_DG_fraction():
    # Variables: y = (x0_1,x0_2,x0_3,z0,...,z5)
    X = [F(s) for s in x0_str]
    z = [F(s) for s in z_str]
    dX = [[Fraction(1 if i == j else 0) for j in range(9)] for i in range(3)]
    G = []
    DG_rows = []
    for k, mu in enumerate(policies):
        Z = z[k]
        J = Jpol_q[mu]
        Xnew = []
        dXnew = [[Fraction(0) for _ in range(9)] for __ in range(3)]
        for i, m in enumerate(exponents):
            Zm = Z ** m
            Xnew_i = J[i] + (X[i] - J[i]) * Zm
            Xnew.append(Xnew_i)
            for col in range(9):
                dXnew[i][col] = Zm * dX[i][col]
            # direct derivative wrt z_k
            dXnew[i][3+k] += (X[i] - J[i]) * Fraction(m) * (Z ** (m-1))
        X = Xnew
        dX = dXnew
        # guard equation at the exit point
        s = guard_indices[k]
        gd = qdiff_fraction(X)[s]
        grad = grad_guard_fraction(s)
        grow = []
        for col in range(9):
            val = sum(grad[i] * dX[i][col] for i in range(3))
            grow.append(val)
        G.append(gd)
        DG_rows.append(grow)
    # return equations x6-x0
    x0_frac = [F(s) for s in x0_str]
    for i in range(3):
        G.append(X[i] - x0_frac[i])
        row = [dX[i][col] for col in range(9)]
        row[i] -= Fraction(1)
        DG_rows.append(row)
    return G, DG_rows


def frac_abs(fr):
    return abs(fr)


def frac_to_sci(fr, digits=18):
    # Reliable short decimal representation for logs.
    dec = base.dec_from_frac(fr, base.ROUND_CEILING if fr >= 0 else base.ROUND_FLOOR)
    return f"{dec:.{digits}E}"


def invert_matrix_fraction(M):
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] + [Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if A[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            raise RuntimeError("singular matrix")
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
        piv = A[col][col]
        for j in range(2*n):
            A[col][j] /= piv
        for r in range(n):
            if r == col:
                continue
            fac = A[r][col]
            if fac:
                for j in range(2*n):
                    A[r][j] -= fac * A[col][j]
    return [[A[i][n+j] for j in range(n)] for i in range(n)]

# ---------------------------------------------------------------------------
# Interval derivative propagation for DG(B).
# ---------------------------------------------------------------------------

def interval_box_center_values():
    x0 = [base.interval_center_radius(s, rho_dec) for s in x0_str]
    z = [base.interval_center_radius(s, rho_dec) for s in z_str]
    return x0, z


def compute_DG_interval_on_B():
    X, zints = interval_box_center_values()
    dX = [[Interval(1 if i == j else 0) for j in range(9)] for i in range(3)]
    rows = []
    Xs = [X]
    for k, mu in enumerate(policies):
        Z = zints[k]
        J = Jpol[mu]
        Xnew = []
        dXnew = [[Interval(0) for _ in range(9)] for __ in range(3)]
        for i, m in enumerate(exponents):
            Zm = Z ** m
            Xnew_i = J[i] + (X[i] - J[i]) * Zm
            Xnew.append(Xnew_i)
            for col in range(9):
                dXnew[i][col] = Zm * dX[i][col]
            dXnew[i][3+k] = dXnew[i][3+k] + (X[i] - J[i]) * Interval(m) * (Z ** (m-1))
        X = Xnew
        dX = dXnew
        Xs.append(X)
        s = guard_indices[k]
        grad = base.grad_guard(s)
        row = []
        for col in range(9):
            val = Interval(0)
            for i in range(3):
                val += grad[i] * dX[i][col]
            row.append(val)
        rows.append(row)
    for i in range(3):
        row = [dX[i][col] for col in range(9)]
        row[i] = row[i] - Interval(1)
        rows.append(row)
    return rows, Xs


def interval_matmul(A, B):
    n = len(A)
    m = len(B[0])
    pnum = len(B)
    return [[sum((A[i][k] * B[k][j] for k in range(pnum)), Interval(0)) for j in range(m)] for i in range(n)]


def interval_inf_norm(M):
    rows = [base.directed_sum((M[i][j].abs_upper() for j in range(len(M[i]))), base.ROUND_CEILING) for i in range(len(M))]
    return max(rows), rows


def decimal_preconditioner_from_DG(DG_frac, digits=90):
    """Return a fixed rational-decimal interval matrix A approximating DG^{-1}.

    Krawczyk's theorem allows any fixed matrix A; it need not be the exact
    inverse.  We compute A numerically at high precision, then freeze the
    printed decimal entries as exact decimal intervals.
    """
    mp.mp.dps = 110
    M = mp.matrix([[mp.mpf(fr.numerator) / mp.mpf(fr.denominator) for fr in row] for row in DG_frac])
    Minv = mp.inverse(M)
    A = []
    for i in range(9):
        row = []
        for j in range(9):
            row.append(Interval(mp.nstr(Minv[i, j], digits)))
        A.append(row)
    return A


def interval_matrix_point_inf_norm(A):
    rows = [base.directed_sum((cell.abs_upper() for cell in row), base.ROUND_CEILING) for row in A]
    return max(rows), rows

# ---------------------------------------------------------------------------
# C2 Bernstein guard sign verification.
# ---------------------------------------------------------------------------

def poly_add(a, b):
    n = max(len(a), len(b))
    out = [Interval(0) for _ in range(n)]
    for i in range(n):
        if i < len(a):
            out[i] = out[i] + a[i]
        if i < len(b):
            out[i] = out[i] + b[i]
    return out


def poly_mul(a, b, maxdeg=40):
    out = [Interval(0) for _ in range(min(maxdeg, len(a)+len(b)-2)+1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            if i + j <= maxdeg:
                out[i+j] = out[i+j] + ai * bj
    return out


def poly_pow_linear(a0, a1, n):
    # (a0 + a1 u)^n as interval power-basis coefficients up to degree 40.
    poly = [Interval(1)]
    basep = [a0, a1]
    k = n
    while k:
        if k & 1:
            poly = poly_mul(poly, basep)
        k >>= 1
        if k:
            basep = poly_mul(basep, basep)
    # Pad to degree 40.
    while len(poly) < 41:
        poly.append(Interval(0))
    return poly[:41]


def power_to_bernstein(power_coeffs, degree=40):
    # p(u)=sum_j c_j u^j, b_i = sum_{j<=i} c_j binom(i,j)/binom(n,j).
    bs = []
    for i in range(degree + 1):
        val = Interval(0)
        for j in range(i + 1):
            if j < len(power_coeffs):
                factor = Fraction(math.comb(i, j), math.comb(degree, j))
                val += power_coeffs[j] * Interval.from_fraction(factor)
        bs.append(val)
    return bs


def signed_guard_bernstein_intervals(Xk, Zk, mu, guard_i):
    sigma = Interval(1 if mu[guard_i] == 0 else -1)
    # d_guard_i(J(r)) = gdiff + alpha DeltaP dot Jmu + sum_j coeff_j (X_j-Jmu_j) r^m_j
    poly = [Interval(0) for _ in range(41)]
    const = g[guard_i][1] - g[guard_i][0]
    for j in range(3):
        const += alpha * (P[guard_i][1][j] - P[guard_i][0][j]) * Jpol[mu][j]
    poly[0] = const
    # r = Z + (1-Z)u
    r0 = Zk
    r1 = Interval(1) - Zk
    for j, m in enumerate(exponents):
        coeff = alpha * (P[guard_i][1][j] - P[guard_i][0][j]) * (Xk[j] - Jpol[mu][j])
        powpoly = poly_pow_linear(r0, r1, m)
        for deg in range(41):
            poly[deg] = poly[deg] + coeff * powpoly[deg]
    poly = [sigma * c for c in poly]
    return power_to_bernstein(poly, 40)


def propagate_orbit_intervals_on_B():
    X, zints = interval_box_center_values()
    Xs = [X]
    for k, mu in enumerate(policies):
        X = base.flow(mu, X, zints[k])
        Xs.append(X)
    return Xs, zints

# Map zero endpoints: None, 'start' (u=1, b_40 may be zero), or 'exit' (u=0, b_0 may be zero)
zero_endpoint = {
    (0,0): 'start', (0,1): 'exit', (0,2): None,
    (1,0): 'exit',  (1,1): 'start', (1,2): None,
    (2,0): 'start', (2,1): None,    (2,2): 'exit',
    (3,0): None,    (3,1): 'exit',  (3,2): 'start',
    (4,0): None,    (4,1): 'start', (4,2): 'exit',
    (5,0): 'exit',  (5,1): None,    (5,2): 'start',
}


def verify_C2():
    Xs, zints = propagate_orbit_intervals_on_B()
    rows = []
    global_min = None
    for k, mu in enumerate(policies):
        Xk = Xs[k]
        Zk = zints[k]
        for gi in range(3):
            bs = signed_guard_bernstein_intervals(Xk, Zk, mu, gi)
            excluded = []
            ze = zero_endpoint[(k, gi)]
            if ze == 'exit':
                excluded = [0]
            elif ze == 'start':
                excluded = [40]
            lower = Decimal('Infinity')
            bad = []
            for idx, b in enumerate(bs):
                if idx in excluded:
                    continue
                if b.lo < lower:
                    lower = b.lo
                if b.lo <= 0:
                    bad.append((idx, b.to_pair()))
            endpoint_interval = None
            if excluded:
                endpoint_interval = bs[excluded[0]].to_pair()
            if global_min is None or lower < global_min:
                global_min = lower
            rows.append({
                "segment": k,
                "policy": str(mu),
                "guard": gi + 1,
                "zero_endpoint": ze or "none",
                "excluded_bernstein_indices": excluded,
                "min_nonendpoint_lower_bound": str(lower),
                "endpoint_interval": endpoint_interval,
                "bad_nonendpoint_coefficients": bad,
            })
    ok = all(len(r["bad_nonendpoint_coefficients"]) == 0 for r in rows)
    return {"ok": ok, "global_min_nonendpoint_lower_bound": str(global_min), "rows": rows}

# ---------------------------------------------------------------------------
# C3 transversality intervals over B.
# ---------------------------------------------------------------------------

def verify_C3():
    Xs, _ = propagate_orbit_intervals_on_B()
    rows = []
    ok = True
    for k, mu in enumerate(policies):
        s = guard_indices[k]
        x_cross = Xs[k+1]
        n = base.grad_guard(s)
        incoming = base.dot(n, base.f_field(mu, x_cross))
        mu_out = policies[(k+1) % 6]
        outgoing = base.dot(n, base.f_field(mu_out, x_cross))
        sign_in = incoming.sign()
        sign_out = outgoing.sign()
        same_nonzero = (sign_in == sign_out and sign_in in ("positive", "negative"))
        if not same_nonzero:
            ok = False
        rows.append({
            "segment": k,
            "guard": s+1,
            "incoming_policy": str(mu),
            "outgoing_policy": str(mu_out),
            "incoming": incoming.to_pair(),
            "outgoing": outgoing.to_pair(),
            "incoming_sign": sign_in,
            "outgoing_sign": sign_out,
            "same_nonzero_sign": same_nonzero,
        })
    return {"ok": ok, "rows": rows}

# ---------------------------------------------------------------------------
# Main C1 function.
# ---------------------------------------------------------------------------

def mpmath_inverse_as_decimal_rationals(DG_frac, digits=80):
    mp.mp.dps = digits + 20
    M = mp.matrix([[mp.mpf(x.numerator) / mp.mpf(x.denominator) for x in row] for row in DG_frac])
    A_mp = mp.inverse(M)
    # Round to fixed decimal strings.  These rounded decimals are then treated
    # as exact rational numbers by Interval(...).  Krawczyk only requires a
    # nonsingular preconditioner A, not the exact inverse.
    A_dec = []
    for i in range(9):
        row = []
        for j in range(9):
            row.append(mp.nstr(A_mp[i,j], n=digits, strip_zeros=False))
        A_dec.append(row)
    return A_dec


def verify_C1():
    G_frac, DG_frac = compute_G_and_DG_fraction()
    max_G = max(abs(x) for x in G_frac)
    # Use an exact decimal rational preconditioner obtained from a high-precision
    # inverse.  The subsequent Krawczyk inclusion is checked by interval
    # arithmetic, so exact inversion is not needed.
    A_dec = mpmath_inverse_as_decimal_rationals(DG_frac, digits=80)
    A_int = [[Interval(A_dec[i][j]) for j in range(9)] for i in range(9)]
    A_norm_rows = [base.directed_sum((A_int[i][j].abs_upper() for j in range(9)), base.ROUND_CEILING) for i in range(9)]
    A_norm_dec = max(A_norm_rows)
    # Interval DG(B)
    DGB, Xs = compute_DG_interval_on_B()
    ADGB = interval_matmul(A_int, DGB)
    I_minus = []
    for i in range(9):
        row = []
        for j in range(9):
            val = (Interval(1) if i == j else Interval(0)) - ADGB[i][j]
            row.append(val)
        I_minus.append(row)
    C_norm, C_rows = interval_inf_norm(I_minus)
    G_dec = base.dec_from_frac(max_G, base.ROUND_CEILING)
    k_radius = base.directed_sum_product(C_norm, rho_dec, base.directed_mul(A_norm_dec, G_dec, base.ROUND_CEILING), base.ROUND_CEILING)
    ok_G = G_dec < Decimal("8e-43")
    ok_A = A_norm_dec < Decimal("13")
    ok_C = C_norm < Decimal("1.2e-20")
    ok_K = k_radius < rho_dec
    return {
        "ok": bool(ok_G and ok_A and ok_C and ok_K),
        "box_radius": rho_str,
        "preconditioner": "80-digit rounded decimal rational approximate inverse of DG(ybar)",
        "max_G_decimal_upper": str(G_dec),
        "A_infinity_norm_upper": str(A_norm_dec),
        "I_minus_A_DG_B_row_sums": [str(x) for x in C_rows],
        "I_minus_A_DG_B_infinity_norm_upper": str(C_norm),
        "krawczyk_radius_upper": str(k_radius),
        "krawczyk_radius_lt_box_radius": bool(ok_K),
        "bounds_pass": {
            "max_G_lt_8e-43": bool(ok_G),
            "A_norm_lt_13": bool(ok_A),
            "I_minus_norm_lt_1p2e-20": bool(ok_C),
            "Krawczyk_radius_lt_1e-30": bool(ok_K),
        }
    }


def main():
    C1 = verify_C1()
    C2 = verify_C2()
    C3 = verify_C3()
    all_ok = C1["ok"] and C2["ok"] and C3["ok"]
    cert = {"all_C1_C3_certificates_pass": all_ok, "C1": C1, "C2": C2, "C3": C3}
    lines = []
    lines.append("C1--C3 FINITE CERTIFICATE VERIFIER")
    lines.append("====================================")
    lines.append(f"All C1--C3 certificates pass: {all_ok}")
    lines.append("")
    lines.append("C1: Krawczyk orbit certificate")
    lines.append(f"  Box radius: {C1['box_radius']}")
    lines.append(f"  ||G(ybar)||_inf upper: {Decimal(C1['max_G_decimal_upper']):.18E}")
    lines.append(f"  ||A||_inf upper: {Decimal(C1['A_infinity_norm_upper']):.18E}")
    lines.append(f"  ||I - A DG(B)||_inf upper: {Decimal(C1['I_minus_A_DG_B_infinity_norm_upper']):.18E}")
    lines.append(f"  Krawczyk radius upper: {Decimal(C1['krawczyk_radius_upper']):.18E}")
    lines.append(f"  K(B) subset int(B): {C1['krawczyk_radius_lt_box_radius']}")
    lines.append("")
    lines.append("C2: Bernstein guard-sign certificate")
    lines.append(f"  Pass: {C2['ok']}")
    lines.append(f"  Global minimum non-endpoint Bernstein lower bound: {Decimal(C2['global_min_nonendpoint_lower_bound']):.18E}")
    for r in C2["rows"]:
        lines.append(
            f"  k={r['segment']} policy={r['policy']} guard=d_{r['guard']} "
            f"zero={r['zero_endpoint']} min_nonendpoint={Decimal(r['min_nonendpoint_lower_bound']):.12E}"
        )
        if r["endpoint_interval"] is not None:
            lines.append(f"      endpoint coefficient interval: {r['endpoint_interval']}")
    lines.append("")
    lines.append("C3: one-way transversality / no-sliding certificate")
    lines.append(f"  Pass: {C3['ok']}")
    for r in C3["rows"]:
        lines.append(
            f"  k={r['segment']} guard=d_{r['guard']} incoming={r['incoming']} "
            f"outgoing={r['outgoing']} same_nonzero_sign={r['same_nonzero_sign']}"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    out_dir = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
    (out_dir / "c1_c3_certificate.log").write_text(text)
    (out_dir / "c1_c3_certificate.json").write_text(json.dumps(cert, indent=2))

if __name__ == "__main__":
    main()
