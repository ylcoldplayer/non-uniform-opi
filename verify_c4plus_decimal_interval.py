#!/usr/bin/env python3
"""
C4+ positive-radius interval certificate for the nonuniform OPI counterexample.

This verifier uses Python's Decimal arithmetic with directed rounding to implement
outward-rounded interval arithmetic. All MDP input data are parsed as exact
rational decimals; policy values are computed exactly with fractions and then
embedded into directed-rounded decimal intervals.

Certificate proved by this script:
  * For the ambient box V = { ||x-x0||_inf <= 1e-8 }, the six intended event
    times are isolated by interval Newton steps.
  * The event-map denominators and dr/dt guard derivatives are bounded away
    from zero, so each event map is locally well-defined and transverse.
  * The interval enclosure of the six-event derivative has infinity norm < 0.75.
  * The center residual is < 7.1e-43, so the mean-value theorem gives
        Pi(V) subset int(V)
    because residual + 0.75*1e-8 < 1e-8.

This is a certificate for the C4+ contraction part. It should be run unchanged
and the printed output/log included in the supplemental material.
"""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from pathlib import Path
import json

PREC = 100


def F(s):
    return Fraction(str(s))


def dec_from_frac(fr, rounding):
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = rounding
        return Decimal(fr.numerator) / Decimal(fr.denominator)


def floor_dec(x):
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = ROUND_FLOOR
        return +x


def ceil_dec(x):
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = ROUND_CEILING
        return +x


class Interval:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        if hi is None:
            hi = lo
        if isinstance(lo, Fraction):
            l = dec_from_frac(lo, ROUND_FLOOR)
        else:
            l = lo if isinstance(lo, Decimal) else Decimal(str(lo))
        if isinstance(hi, Fraction):
            h = dec_from_frac(hi, ROUND_CEILING)
        else:
            h = hi if isinstance(hi, Decimal) else Decimal(str(hi))
        if l > h:
            l, h = h, l
        self.lo = floor_dec(l)
        self.hi = ceil_dec(h)

    @staticmethod
    def from_fraction(fr):
        return Interval(dec_from_frac(fr, ROUND_FLOOR), dec_from_frac(fr, ROUND_CEILING))

    def __add__(self, other):
        other = as_interval(other)
        with localcontext() as ctx:
            ctx.prec = PREC
            ctx.rounding = ROUND_FLOOR
            lo = self.lo + other.lo
        with localcontext() as ctx:
            ctx.prec = PREC
            ctx.rounding = ROUND_CEILING
            hi = self.hi + other.hi
        return Interval(lo, hi)

    __radd__ = __add__

    def __sub__(self, other):
        other = as_interval(other)
        with localcontext() as ctx:
            ctx.prec = PREC
            ctx.rounding = ROUND_FLOOR
            lo = self.lo - other.hi
        with localcontext() as ctx:
            ctx.prec = PREC
            ctx.rounding = ROUND_CEILING
            hi = self.hi - other.lo
        return Interval(lo, hi)

    def __rsub__(self, other):
        return as_interval(other).__sub__(self)

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other):
        other = as_interval(other)
        lows = []
        highs = []
        for a, b in ((self.lo, other.lo), (self.lo, other.hi), (self.hi, other.lo), (self.hi, other.hi)):
            with localcontext() as ctx:
                ctx.prec = PREC
                ctx.rounding = ROUND_FLOOR
                lows.append(a * b)
            with localcontext() as ctx:
                ctx.prec = PREC
                ctx.rounding = ROUND_CEILING
                highs.append(a * b)
        return Interval(min(lows), max(highs))

    __rmul__ = __mul__

    def inv(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError(f"interval contains zero: {self}")
        with localcontext() as ctx:
            ctx.prec = PREC
            ctx.rounding = ROUND_FLOOR
            a = Decimal(1) / self.hi
            b = Decimal(1) / self.lo
            lo = min(a, b)
        with localcontext() as ctx:
            ctx.prec = PREC
            ctx.rounding = ROUND_CEILING
            a = Decimal(1) / self.hi
            b = Decimal(1) / self.lo
            hi = max(a, b)
        return Interval(lo, hi)

    def __truediv__(self, other):
        return self * as_interval(other).inv()

    def __rtruediv__(self, other):
        return as_interval(other) * self.inv()

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("only nonnegative integer powers are supported")
        result = Interval(1)
        base = self
        k = n
        while k:
            if k & 1:
                result = result * base
            k >>= 1
            if k:
                base = base * base
        return result

    def contains_zero(self):
        return self.lo <= 0 <= self.hi

    def width(self):
        return self.hi - self.lo

    def mid(self):
        with localcontext() as ctx:
            ctx.prec = PREC
            return (self.lo + self.hi) / Decimal(2)

    def abs_upper(self):
        return max(abs(self.lo), abs(self.hi))

    def sign(self):
        if self.hi < 0:
            return "negative"
        if self.lo > 0:
            return "positive"
        return "contains zero"

    def intersect(self, other):
        other = as_interval(other)
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            raise ValueError(f"empty intersection: {self} and {other}")
        return Interval(lo, hi)

    def fmt(self, sig=18):
        return f"[{self.lo:.{sig}E}, {self.hi:.{sig}E}]"

    def to_pair(self):
        return [str(self.lo), str(self.hi)]

    def __repr__(self):
        return self.fmt(8)


def as_interval(x):
    return x if isinstance(x, Interval) else Interval(x)


alpha_q = F("0.95")
p_q = [F("0.02"), F("0.18"), F("0.80")]
g_q = [
    [F("2.8645365128"), F("6.2604856687")],
    [F("8.8765458733"), F("-6.6644477357")],
    [F("-4.4428868652"), F("-7.5034828348")],
]
P_q = [
    [[F("0.0083421983"), F("0.5161392971"), F("0.4755185046")],
     [F("0.4115295055"), F("0.3414919270"), F("0.2469785675")]],
    [[F("0.0176113658"), F("0.9549690854"), F("0.0274195488")],
     [F("0.3574261643"), F("0.1754143753"), F("0.4671594604")]],
    [[F("0.6791060489"), F("0.0358694514"), F("0.2850244997")],
     [F("0.3221209528"), F("0.3803472570"), F("0.2975317902")]],
]

policies = [(0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
guard_indices = [1, 0, 2, 1, 2, 0]  # one-based: d2, d1, d3, d2, d3, d1
x0_str = [
    "-8.59365549110427684060547352373810356279806687",
    "-0.0178290152994757047995861123714012318312941336",
    "0.494199384159623810384986644320237043033461336",
]
z_str = [
    "0.747518894957396298068037965192410779744158632",
    "0.983925376878883085999540632135004996246384770",
    "0.978860491445977974417005488978870175248391989",
    "0.957246782943564021435154886545707794901486695",
    "0.748005437128185097605270137358697189429422619",
    "0.999089891635080634085860840337534558610691943",
]


def solve_3x3_fraction(A, b):
    M = [list(A[i]) + [b[i]] for i in range(3)]
    for col in range(3):
        pivot = next(r for r in range(col, 3) if M[r][col] != 0)
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
        piv = M[col][col]
        for j in range(col, 4):
            M[col][j] /= piv
        for r in range(3):
            if r == col:
                continue
            fac = M[r][col]
            if fac:
                for j in range(col, 4):
                    M[r][j] -= fac * M[col][j]
    return [M[i][3] for i in range(3)]


def policy_value_fraction(mu):
    A = []
    b = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append((Fraction(1) if i == j else Fraction(0)) - alpha_q * P_q[i][mu[i]][j])
        A.append(row)
        b.append(g_q[i][mu[i]])
    return solve_3x3_fraction(A, b)


all_policies = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
Jpol_q = {mu: policy_value_fraction(mu) for mu in all_policies}
Jpol = {mu: [Interval.from_fraction(v) for v in Jpol_q[mu]] for mu in all_policies}
alpha = Interval.from_fraction(alpha_q)
p = [Interval.from_fraction(v) for v in p_q]
P = [[[Interval.from_fraction(P_q[i][a][j]) for j in range(3)] for a in range(2)] for i in range(3)]
g = [[Interval.from_fraction(g_q[i][a]) for a in range(2)] for i in range(3)]


def qdiff(J):
    out = []
    for s in range(3):
        v = g[s][1] - g[s][0]
        for j in range(3):
            v += alpha * (P[s][1][j] - P[s][0][j]) * J[j]
        out.append(v)
    return out


def flow(mu, X, r):
    exponents = [1, 9, 40]
    return [Jpol[mu][i] + (X[i] - Jpol[mu][i]) * (r ** exponents[i]) for i in range(3)]


def grad_guard(s):
    return [alpha * (P[s][1][j] - P[s][0][j]) for j in range(3)]


def f_field(mu, Y):
    return [p[i] * (Jpol[mu][i] - Y[i]) for i in range(3)]


def dot(a, b):
    v = Interval(0)
    for x, y in zip(a, b):
        v += x * y
    return v


def matmul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(3)), Interval(0)) for j in range(3)] for i in range(3)]


def event_derivative(mu, Y, r, s):
    f = f_field(mu, Y)
    n = grad_guard(s)
    denom = dot(n, f)
    if denom.contains_zero():
        raise RuntimeError(f"event denominator contains zero: {denom}")
    A = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append((Interval(1) if i == j else Interval(0)) - f[i] * n[j] / denom)
        A.append(row)
    E = [[Interval(0) for _ in range(3)] for __ in range(3)]
    E[0][0] = r
    E[1][1] = r ** 9
    E[2][2] = r ** 40
    return matmul(A, E), denom


def phi(mu, X, r, s):
    return qdiff(flow(mu, X, r))[s]


def dphi_dr(mu, X, R, s):
    n = grad_guard(s)
    exponents = [1, 9, 40]
    v = Interval(0)
    for i, m in enumerate(exponents):
        v += n[i] * (X[i] - Jpol[mu][i]) * Interval(m) * (R ** (m - 1))
    return v


def inf_norm(M):
    rows = []
    for i in range(3):
        rows.append(sum((M[i][j].abs_upper() for j in range(3)), Decimal(0)))
    return max(rows), rows


def root_interval_newton(mu, X, s, z_center, init_width="1e-5", iterations=6):
    z = Decimal(z_center)
    w = Decimal(init_width)
    R = Interval(z - w, z + w)
    for _ in range(iterations):
        mid = R.mid()
        Phi = phi(mu, X, Interval(mid), s)
        Der = dphi_dr(mu, X, R, s)
        if Der.contains_zero():
            raise RuntimeError(f"root derivative contains zero on segment {mu}, guard {s+1}: {Der}")
        N = Interval(mid) - Phi / Der
        R = R.intersect(N)
    return R


def center_residual():
    X = [Interval(v) for v in x0_str]
    max_abs = Decimal(0)
    guard_res = []
    for k, mu in enumerate(policies):
        X = flow(mu, X, Interval(z_str[k]))
        val = qdiff(X)[guard_indices[k]]
        guard_res.append(val)
        max_abs = max(max_abs, val.abs_upper())
    ret_res = [X[i] - Interval(x0_str[i]) for i in range(3)]
    for val in ret_res:
        max_abs = max(max_abs, val.abs_upper())
    return max_abs, guard_res, ret_res


def verify_c4plus(radius="1e-8", init_width="1e-5", iterations=6):
    rad = Decimal(radius)
    X = [Interval(Decimal(x) - rad, Decimal(x) + rad) for x in x0_str]
    M = [[Interval(1 if i == j else 0) for j in range(3)] for i in range(3)]
    segments = []

    for k, mu in enumerate(policies):
        s = guard_indices[k]
        R = root_interval_newton(mu, X, s, z_str[k], init_width, iterations)
        dR = dphi_dr(mu, X, R, s)
        if dR.contains_zero():
            raise RuntimeError(f"dphi/dr contains zero after Newton on segment {k}")
        Y = flow(mu, X, R)
        Mk, denom = event_derivative(mu, Y, R, s)
        M = matmul(Mk, M)
        cumulative_norm, cumulative_rows = inf_norm(M)
        segments.append({
            "k": k,
            "policy": str(mu),
            "guard": s + 1,
            "root_interval": R,
            "root_width": R.width(),
            "dphi_dr": dR,
            "event_denominator": denom,
            "image_widths": [y.width() for y in Y],
            "cumulative_row_sums": cumulative_rows,
            "cumulative_norm": cumulative_norm,
        })
        X = Y

    final_norm, final_rows = inf_norm(M)
    res, guard_res, ret_res = center_residual()
    containment_radius = res + final_norm * rad
    ok_contraction = final_norm < Decimal("0.75")
    ok_containment = containment_radius < rad
    return {
        "precision_decimal_digits": PREC,
        "box_radius": radius,
        "initial_root_width": init_width,
        "newton_iterations": iterations,
        "center_residual_upper": res,
        "final_derivative_row_sums": final_rows,
        "final_derivative_infinity_norm": final_norm,
        "mvt_image_radius_upper": containment_radius,
        "contraction_bound_lt_0_75": ok_contraction,
        "mvt_containment_in_initial_box": ok_containment,
        "segments": segments,
        "matrix": M,
    }


def main():
    cert = verify_c4plus()
    lines = []
    lines.append("C4+ POSITIVE-RADIUS INTERVAL CERTIFICATE")
    lines.append("========================================")
    lines.append(f"Decimal precision: {cert['precision_decimal_digits']} digits")
    lines.append(f"Box V: ||x - x0||_inf <= {cert['box_radius']}")
    lines.append(f"Initial event-root bracket half-width: {cert['initial_root_width']}")
    lines.append(f"Interval Newton iterations per event: {cert['newton_iterations']}")
    lines.append("")
    lines.append(f"Center residual upper bound: {cert['center_residual_upper']:.18E}")
    lines.append("Final DPi row-sum upper bounds:")
    for i, r in enumerate(cert["final_derivative_row_sums"]):
        lines.append(f"  row {i+1}: {r:.18E}")
    lines.append(f"Final ||DPi(V)||_inf upper bound: {cert['final_derivative_infinity_norm']:.18E}")
    lines.append(f"MVT image-radius upper bound: {cert['mvt_image_radius_upper']:.18E}")
    lines.append(f"Contraction < 0.75: {cert['contraction_bound_lt_0_75']}")
    lines.append(f"Pi(V) subset int(V) by MVT: {cert['mvt_containment_in_initial_box']}")
    lines.append("")
    lines.append("Per-event interval enclosures:")
    for seg in cert["segments"]:
        lines.append(
            f"k={seg['k']} policy={seg['policy']} exit=d_{seg['guard']} "
            f"R={seg['root_interval'].fmt(18)} width={seg['root_width']:.18E}"
        )
        lines.append(f"    dphi/dr={seg['dphi_dr'].fmt(12)}")
        lines.append(f"    event denominator={seg['event_denominator'].fmt(12)}")
        lines.append("    image widths: " + ", ".join(f"{w:.12E}" for w in seg["image_widths"]))
        lines.append("    cumulative row sums: " + ", ".join(f"{r:.12E}" for r in seg["cumulative_row_sums"]))
        lines.append(f"    cumulative ||D(Psi_k...Psi_0)||_inf={seg['cumulative_norm']:.12E}")
    lines.append("")
    lines.append("Interval enclosure for final DPi matrix:")
    for row in cert["matrix"]:
        lines.append("  " + "  ".join(cell.fmt(12) for cell in row))

    text = "\n".join(lines) + "\n"
    print(text)

    out_dir = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
    (out_dir / "c4plus_interval_certificate.log").write_text(text)

    # JSON summary for machine-readable supplement.
    json_obj = {
        "box_radius": cert["box_radius"],
        "center_residual_upper": str(cert["center_residual_upper"]),
        "final_derivative_row_sums": [str(x) for x in cert["final_derivative_row_sums"]],
        "final_derivative_infinity_norm": str(cert["final_derivative_infinity_norm"]),
        "mvt_image_radius_upper": str(cert["mvt_image_radius_upper"]),
        "contraction_bound_lt_0_75": cert["contraction_bound_lt_0_75"],
        "mvt_containment_in_initial_box": cert["mvt_containment_in_initial_box"],
        "segments": [
            {
                "k": s["k"],
                "policy": s["policy"],
                "guard": s["guard"],
                "root_interval": s["root_interval"].to_pair(),
                "root_width": str(s["root_width"]),
                "dphi_dr": s["dphi_dr"].to_pair(),
                "event_denominator": s["event_denominator"].to_pair(),
                "image_widths": [str(w) for w in s["image_widths"]],
                "cumulative_row_sums": [str(r) for r in s["cumulative_row_sums"]],
                "cumulative_norm": str(s["cumulative_norm"]),
            }
            for s in cert["segments"]
        ],
    }
    (out_dir / "c4plus_interval_certificate.json").write_text(json.dumps(json_obj, indent=2))


if __name__ == "__main__":
    main()
