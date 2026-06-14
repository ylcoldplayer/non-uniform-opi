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
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING, getcontext
from fractions import Fraction
from pathlib import Path
import json
import math

PREC = 100
getcontext().prec = PREC


def decimal_exact(x):
    """Parse a number-like object as a Decimal without binary-float conversion."""
    return x if isinstance(x, Decimal) else Decimal(str(x))


def directed_add(a, b, rounding):
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = rounding
        return decimal_exact(a) + decimal_exact(b)


def directed_sub(a, b, rounding):
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = rounding
        return decimal_exact(a) - decimal_exact(b)


def interval_center_radius(center, radius):
    """Outward-rounded interval [center-radius, center+radius]."""
    lo = directed_sub(center, radius, ROUND_FLOOR)
    hi = directed_add(center, radius, ROUND_CEILING)
    return Interval(lo, hi)


def interval_grid_cell(radius, index, splits):
    """Outward-rounded subinterval of [-radius,radius] in an equal grid."""
    r = decimal_exact(radius)
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = ROUND_FLOOR
        lo = -r + Decimal(2) * r * Decimal(index) / Decimal(splits)
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = ROUND_CEILING
        hi = -r + Decimal(2) * r * Decimal(index + 1) / Decimal(splits)
    return Interval(lo, hi)


def directed_mul(a, b, rounding):
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = rounding
        return decimal_exact(a) * decimal_exact(b)


def directed_sum_product(a, b, c=None, rounding=ROUND_CEILING):
    """Compute a*b+c with directed rounding; c defaults to 0."""
    if c is None:
        c = Decimal(0)
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = rounding
        return decimal_exact(a) * decimal_exact(b) + decimal_exact(c)


def directed_sum(values, rounding=ROUND_CEILING):
    """Directed-rounded sum of Decimal-compatible values."""
    with localcontext() as ctx:
        ctx.prec = PREC
        ctx.rounding = rounding
        total = Decimal(0)
        for value in values:
            total += decimal_exact(value)
        return total


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
            l = decimal_exact(lo)
        if isinstance(hi, Fraction):
            h = dec_from_frac(hi, ROUND_CEILING)
        else:
            h = decimal_exact(hi)
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
exponents = [1, 9, 40]
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


def min_abs_interval(I):
    """Certified lower bound for |I| when I does not contain zero."""
    if I.contains_zero():
        return Decimal(0)
    return min(abs(I.lo), abs(I.hi))



def outgoing_transversality_certificate(k, Y, incoming_denom):
    """Check outgoing normal velocity at the same crossing point.

    The incoming denominator n_k^T f_{mu_k}(Y) is the event-map denominator.
    For no Filippov sliding on the positive-radius tube, the outgoing field
    f_{mu_{k+1}} must cross the same guard in the same nonzero direction.
    """
    s = guard_indices[k]
    mu_out = policies[(k + 1) % len(policies)]
    n = grad_guard(s)
    outgoing = dot(n, f_field(mu_out, Y))
    sign_in = incoming_denom.sign()
    sign_out = outgoing.sign()
    same_nonzero = sign_in == sign_out and sign_in in ("positive", "negative")
    if not same_nonzero:
        raise RuntimeError(
            f"outgoing transversality failed on segment {k}: "
            f"incoming={incoming_denom}, outgoing={outgoing}"
        )
    return {
        "outgoing_policy": str(mu_out),
        "outgoing_denominator": outgoing,
        "incoming_sign": sign_in,
        "outgoing_sign": sign_out,
        "same_nonzero_sign": same_nonzero,
        "outgoing_abs_lower": min_abs_interval(outgoing),
    }


def crossing_nonactive_guard_certificate(k, mu, Y):
    """Check non-active guards at the certified crossing point.

    At segment k only guard s_k is allowed to vanish.  Every other signed
    guard must be strictly positive at Y, ruling out simultaneous unintended
    guard hits throughout the enclosed crossing box.
    """
    s = guard_indices[k]
    rows = []
    min_lower = Decimal("Infinity")
    for gi in range(3):
        if gi == s:
            continue
        val = signed_guard_value(mu, Y, gi)
        if val.lo <= 0:
            raise RuntimeError(
                f"non-active guard sign failed at crossing segment {k}, guard d_{gi+1}: {val}"
            )
        min_lower = min(min_lower, val.lo)
        rows.append({"guard": gi + 1, "signed_guard": val})
    return {"rows": rows, "min_lower": min_lower}


def matmul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(3)), Interval(0)) for j in range(3)] for i in range(3)]


def normal_velocity(mu, Y, s):
    """Interval enclosure of grad(d_s)^T f_mu(Y)."""
    return dot(grad_guard(s), f_field(mu, Y))


def guard_sigma(mu, guard_i):
    """Sign that makes sigma*d_i nonnegative inside policy mu's cell."""
    return Interval(1 if mu[guard_i] == 0 else -1)


def signed_guard_value(mu, Y, guard_i):
    """Interval enclosure of the signed guard sigma*d_i(Y)."""
    return guard_sigma(mu, guard_i) * qdiff(Y)[guard_i]


def _poly_mul(a, b, maxdeg=40):
    out = [Interval(0) for _ in range(min(maxdeg, len(a) + len(b) - 2) + 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            if i + j <= maxdeg:
                out[i + j] = out[i + j] + ai * bj
    return out


def _poly_pow_linear(a0, a1, n, degree=40):
    """Power-basis coefficients of (a0+a1*u)^n, padded to degree.

    Computed directly by the binomial formula to avoid intermediate
    polynomial growth in the certificate loops.
    """
    a0 = as_interval(a0)
    a1 = as_interval(a1)
    pow0 = [Interval(1)]
    pow1 = [Interval(1)]
    for _ in range(n):
        pow0.append(pow0[-1] * a0)
        pow1.append(pow1[-1] * a1)
    coeffs = []
    for j in range(n + 1):
        coeffs.append(Interval(math.comb(n, j)) * pow0[n - j] * pow1[j])
    while len(coeffs) < degree + 1:
        coeffs.append(Interval(0))
    return coeffs[:degree + 1]

def _power_to_bernstein(power_coeffs, degree=40):
    """Convert p(u)=sum_j c_j u^j to degree-'degree' Bernstein coefficients."""
    bs = []
    for i in range(degree + 1):
        val = Interval(0)
        for j in range(i + 1):
            if j < len(power_coeffs):
                factor = Fraction(math.comb(i, j), math.comb(degree, j))
                val += power_coeffs[j] * Interval.from_fraction(factor)
        bs.append(val)
    return bs


def signed_guard_bernstein_on_r_interval(mu, X, guard_i, rlo, rhi, degree=40):
    """Bernstein coefficients for sigma*d_i(Phi_mu(X,r)), r in [rlo,rhi].

    The affine substitution r = rlo + (rhi-rlo) u maps u in [0,1] to the
    full candidate segment.  Positivity of all non-excluded coefficients
    certifies that the trajectory remains in the intended policy cell over
    that whole r-interval.
    """
    r0 = as_interval(rlo)
    r1 = as_interval(rhi) - r0
    poly = [Interval(0) for _ in range(degree + 1)]
    const = g[guard_i][1] - g[guard_i][0]
    for j in range(3):
        const += alpha * (P[guard_i][1][j] - P[guard_i][0][j]) * Jpol[mu][j]
    poly[0] = const
    for j, m in enumerate(exponents):
        coeff = alpha * (P[guard_i][1][j] - P[guard_i][0][j]) * (X[j] - Jpol[mu][j])
        powpoly = _poly_pow_linear(r0, r1, m, degree)
        for deg in range(degree + 1):
            poly[deg] = poly[deg] + coeff * powpoly[deg]
    sigma = guard_sigma(mu, guard_i)
    return _power_to_bernstein([sigma * c for c in poly], degree)


def bernstein_positive_certificate(coeffs, excluded=()):
    """Return positivity data for Bernstein coefficients outside exclusions."""
    excluded = set(excluded)
    lower = Decimal('Infinity')
    bad = []
    for idx, coeff in enumerate(coeffs):
        if idx in excluded:
            continue
        if coeff.lo < lower:
            lower = coeff.lo
        if coeff.lo <= 0:
            bad.append((idx, coeff.to_pair()))
    return {"ok": len(bad) == 0, "min_lower": lower, "bad": bad}


def event_derivative(mu, Y, r, s):
    f = f_field(mu, Y)
    n = grad_guard(s)
    denom = normal_velocity(mu, Y, s)
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
        rows.append(directed_sum((M[i][j].abs_upper() for j in range(3)), ROUND_CEILING))
    return max(rows), rows


def interval_abs_lower(I):
    """Lower bound for |I| when I does not contain zero; otherwise 0."""
    I = as_interval(I)
    if I.contains_zero():
        return Decimal(0)
    return min(abs(I.lo), abs(I.hi))


def poly_mul_truncated(a, b, maxdeg=40):
    """Interval polynomial product truncated to maxdeg."""
    out = [Interval(0) for _ in range(min(maxdeg, len(a) + len(b) - 2) + 1)]
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            if i + j <= maxdeg:
                out[i + j] = out[i + j] + ai * bj
    return out


def poly_pow_linear(a0, a1, n, degree=40):
    """Power-basis coefficients of (a0+a1*u)^n, padded to degree.

    Computed directly by the binomial formula to keep the section certificate
    fast and memory-stable.
    """
    a0 = as_interval(a0)
    a1 = as_interval(a1)
    pow0 = [Interval(1)]
    pow1 = [Interval(1)]
    for _ in range(n):
        pow0.append(pow0[-1] * a0)
        pow1.append(pow1[-1] * a1)
    coeffs = []
    for j in range(n + 1):
        coeffs.append(Interval(math.comb(n, j)) * pow0[n - j] * pow1[j])
    while len(coeffs) < degree + 1:
        coeffs.append(Interval(0))
    return coeffs[:degree + 1]

_BERNSTEIN_FACTOR_CACHE = {}


def _bernstein_factors(degree=40):
    factors = _BERNSTEIN_FACTOR_CACHE.get(degree)
    if factors is None:
        factors = []
        for i in range(degree + 1):
            row = []
            for j in range(i + 1):
                row.append(Interval.from_fraction(Fraction(math.comb(i, j), math.comb(degree, j))))
            factors.append(row)
        _BERNSTEIN_FACTOR_CACHE[degree] = factors
    return factors


def power_to_bernstein(power_coeffs, degree=40):
    """Convert power-basis interval coefficients to Bernstein coefficients."""
    factors = _bernstein_factors(degree)
    bs = []
    for i in range(degree + 1):
        val = Interval(0)
        for j in range(i + 1):
            if j < len(power_coeffs):
                val += power_coeffs[j] * factors[i][j]
        bs.append(val)
    return bs


def power_to_bernstein_min_lower(power_coeffs, excluded_indices=(), degree=40):
    """Direct lower bound on non-excluded Bernstein coefficients.

    This avoids constructing full interval Bernstein coefficients when only a
    positive lower bound is needed.  Bernstein conversion coefficients are
    nonnegative, so each product lower bound is computed directly with
    downward rounding.
    """
    excluded = set(excluded_indices)
    factors = _bernstein_factors(degree)
    best = Decimal("Infinity")
    seen = False
    for i in range(degree + 1):
        if i in excluded:
            continue
        seen = True
        with localcontext() as ctx:
            ctx.prec = PREC
            ctx.rounding = ROUND_FLOOR
            val = Decimal(0)
            for j in range(i + 1):
                if j < len(power_coeffs):
                    coeff = power_coeffs[j]
                    factor = factors[i][j]
                    # factor is nonnegative.  If coeff.lo is negative, the
                    # smallest product uses the upper endpoint of the factor;
                    # otherwise it uses the lower endpoint.
                    scale = factor.hi if coeff.lo < 0 else factor.lo
                    val += coeff.lo * scale
        if val < best:
            best = val
    if not seen:
        raise ValueError("all Bernstein coefficients were excluded")
    return best


def guard_segment_power_polynomials(Rk, degree=40):
    """Precompute powers of r=Rk+(1-Rk)u used by all three guards."""
    r0 = Rk
    r1 = Interval(1) - Rk
    return {m: poly_pow_linear(r0, r1, m, degree) for m in (1, 9, 40)}


def signed_guard_power_coeffs_from_powers(Xk, mu, guard_i, power_polys, degree=40):
    """Power-basis interval coefficients for the signed guard."""
    sigma = Interval(1 if mu[guard_i] == 0 else -1)
    poly = [Interval(0) for _ in range(degree + 1)]
    const = g[guard_i][1] - g[guard_i][0]
    for j in range(3):
        const += alpha * (P[guard_i][1][j] - P[guard_i][0][j]) * Jpol[mu][j]
    poly[0] = const
    exponents = [1, 9, 40]
    for j, m in enumerate(exponents):
        coeff = alpha * (P[guard_i][1][j] - P[guard_i][0][j]) * (Xk[j] - Jpol[mu][j])
        powpoly = power_polys[m]
        for deg in range(degree + 1):
            poly[deg] = poly[deg] + coeff * powpoly[deg]
    return [sigma * c for c in poly]


def signed_guard_bernstein_intervals_from_powers(Xk, mu, guard_i, power_polys, degree=40):
    """Bernstein coefficients for the signed guard using precomputed powers."""
    signed_power = signed_guard_power_coeffs_from_powers(Xk, mu, guard_i, power_polys, degree)
    return power_to_bernstein(signed_power, degree)


def signed_guard_bernstein_min_lower_from_powers(Xk, mu, guard_i, power_polys, excluded_indices=(), degree=40):
    """Fast lower bound for signed guard Bernstein coefficients."""
    signed_power = signed_guard_power_coeffs_from_powers(Xk, mu, guard_i, power_polys, degree)
    return power_to_bernstein_min_lower(signed_power, excluded_indices, degree)


def signed_guard_bernstein_intervals(Xk, Rk, mu, guard_i, degree=40):
    """Bernstein coefficients for the signed guard on r in [Rk,1].

    The parameterization is r = Rk + (1-Rk) u, u in [0,1].
    The sign is chosen so that the intended greedy action under mu has
    nonnegative signed guard: +d_i for action 0 and -d_i for action 1.
    """
    powers = guard_segment_power_polynomials(Rk, degree)
    return signed_guard_bernstein_intervals_from_powers(Xk, mu, guard_i, powers, degree)


def bernstein_min_lower(coeffs, excluded_indices=()):
    """Minimum lower endpoint among non-excluded Bernstein coefficients."""
    excluded = set(excluded_indices)
    lows = [coeff.lo for idx, coeff in enumerate(coeffs) if idx not in excluded]
    if not lows:
        raise ValueError("all Bernstein coefficients were excluded")
    return min(lows)



def root_interval_newton(mu, X, s, z_center, init_width="1e-5", iterations=6):
    R = interval_center_radius(z_center, init_width)
    for _ in range(iterations):
        mid = R.mid()
        Phi = phi(mu, X, Interval(mid), s)
        Der = dphi_dr(mu, X, R, s)
        if Der.contains_zero():
            raise RuntimeError(f"root derivative contains zero on segment {mu}, guard {s+1}: {Der}")
        N = Interval(mid) - Phi / Der
        R = R.intersect(N)
    return R


def root_bracket_certificate(mu, X, s, R):
    """Certify existence/uniqueness of the event root on R for all X.

    If dphi/dr is bounded away from zero and phi has opposite strict signs at
    the two endpoints of R, every point in the parameter box has exactly one
    event root in R.
    """
    left = phi(mu, X, Interval(R.lo), s)
    right = phi(mu, X, Interval(R.hi), s)
    der = dphi_dr(mu, X, R, s)
    if der.contains_zero():
        raise RuntimeError(f"root derivative contains zero in bracket check on segment {mu}, guard {s+1}: {der}")
    if left.hi < 0 and right.lo > 0:
        sign_change = "negative_to_positive"
    elif left.lo > 0 and right.hi < 0:
        sign_change = "positive_to_negative"
    else:
        raise RuntimeError(
            f"event root not strictly bracketed on segment {mu}, guard {s+1}: "
            f"phi(lo)={left}, phi(hi)={right}"
        )
    return {
        "phi_left": left,
        "phi_right": right,
        "dphi_dr": der,
        "sign_change": sign_change,
    }


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
    rad = decimal_exact(radius)
    X = [interval_center_radius(x, rad) for x in x0_str]
    M = [[Interval(1 if i == j else 0) for j in range(3)] for i in range(3)]
    segments = []
    min_abs_outgoing = Decimal("Infinity")
    min_crossing_nonactive_guard = Decimal("Infinity")

    for k, mu in enumerate(policies):
        s = guard_indices[k]
        R = root_interval_newton(mu, X, s, z_str[k], init_width, iterations)
        bracket = root_bracket_certificate(mu, X, s, R)
        dR = dphi_dr(mu, X, R, s)
        if dR.contains_zero():
            raise RuntimeError(f"dphi/dr contains zero after Newton on segment {k}")
        Y = flow(mu, X, R)
        Mk, denom = event_derivative(mu, Y, R, s)
        outgoing = outgoing_transversality_certificate(k, Y, denom)
        nonactive_crossing = crossing_nonactive_guard_certificate(k, mu, Y)
        min_abs_outgoing = min(min_abs_outgoing, outgoing["outgoing_abs_lower"])
        min_crossing_nonactive_guard = min(min_crossing_nonactive_guard, nonactive_crossing["min_lower"])
        M = matmul(Mk, M)
        cumulative_norm, cumulative_rows = inf_norm(M)
        segments.append({
            "k": k,
            "policy": str(mu),
            "guard": s + 1,
            "root_interval": R,
            "root_width": R.width(),
            "root_bracket": bracket,
            "dphi_dr": dR,
            "event_denominator": denom,
            "outgoing_transversality": outgoing,
            "crossing_nonactive_guard_signs": nonactive_crossing,
            "image_widths": [y.width() for y in Y],
            "cumulative_row_sums": cumulative_rows,
            "cumulative_norm": cumulative_norm,
        })
        X = Y

    final_norm, final_rows = inf_norm(M)
    res, guard_res, ret_res = center_residual()
    containment_radius = directed_sum_product(final_norm, rad, res, ROUND_CEILING)
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
        "strict_event_root_bracketing": True,
        "positive_radius_outgoing_transversality": True,
        "min_abs_outgoing_event_denominator": min_abs_outgoing,
        "crossing_nonactive_guard_signs_certified": True,
        "min_crossing_nonactive_signed_guard_lower_bound": min_crossing_nonactive_guard,
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
    lines.append(f"Strict event-root bracketing: {cert['strict_event_root_bracketing']}")
    lines.append(f"Outgoing same-sign transversality: {cert['positive_radius_outgoing_transversality']}")
    lines.append(f"Minimum |outgoing event denominator| lower bound: {cert['min_abs_outgoing_event_denominator']:.18E}")
    lines.append(f"Non-active guard signs at crossings: {cert['crossing_nonactive_guard_signs_certified']}")
    lines.append(f"Minimum crossing non-active signed-guard lower bound: {cert['min_crossing_nonactive_signed_guard_lower_bound']:.18E}")
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
        lines.append(f"    root bracket: {seg['root_bracket']['sign_change']}, "
                     f"phi(left)={seg['root_bracket']['phi_left'].fmt(12)}, "
                     f"phi(right)={seg['root_bracket']['phi_right'].fmt(12)}")
        lines.append(f"    event denominator={seg['event_denominator'].fmt(12)}")
        lines.append(f"    outgoing denominator={seg['outgoing_transversality']['outgoing_denominator'].fmt(12)} "
                     f"same_sign={seg['outgoing_transversality']['same_nonzero_sign']}")
        lines.append(f"    crossing non-active guard min lower={seg['crossing_nonactive_guard_signs']['min_lower']:.12E}")
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
        "strict_event_root_bracketing": cert["strict_event_root_bracketing"],
        "positive_radius_outgoing_transversality": cert["positive_radius_outgoing_transversality"],
        "min_abs_outgoing_event_denominator": str(cert["min_abs_outgoing_event_denominator"]),
        "crossing_nonactive_guard_signs_certified": cert["crossing_nonactive_guard_signs_certified"],
        "min_crossing_nonactive_signed_guard_lower_bound": str(cert["min_crossing_nonactive_signed_guard_lower_bound"]),
        "contraction_bound_lt_0_75": cert["contraction_bound_lt_0_75"],
        "mvt_containment_in_initial_box": cert["mvt_containment_in_initial_box"],
        "segments": [
            {
                "k": s["k"],
                "policy": s["policy"],
                "guard": s["guard"],
                "root_interval": s["root_interval"].to_pair(),
                "root_width": str(s["root_width"]),
                "root_bracket": {
                    "sign_change": s["root_bracket"]["sign_change"],
                    "phi_left": s["root_bracket"]["phi_left"].to_pair(),
                    "phi_right": s["root_bracket"]["phi_right"].to_pair(),
                },
                "dphi_dr": s["dphi_dr"].to_pair(),
                "event_denominator": s["event_denominator"].to_pair(),
                "outgoing_transversality": {
                    "outgoing_policy": s["outgoing_transversality"]["outgoing_policy"],
                    "outgoing_denominator": s["outgoing_transversality"]["outgoing_denominator"].to_pair(),
                    "incoming_sign": s["outgoing_transversality"]["incoming_sign"],
                    "outgoing_sign": s["outgoing_transversality"]["outgoing_sign"],
                    "same_nonzero_sign": s["outgoing_transversality"]["same_nonzero_sign"],
                    "outgoing_abs_lower": str(s["outgoing_transversality"]["outgoing_abs_lower"]),
                },
                "crossing_nonactive_guard_signs": {
                    "min_lower": str(s["crossing_nonactive_guard_signs"]["min_lower"]),
                    "rows": [
                        {"guard": row["guard"], "signed_guard": row["signed_guard"].to_pair()}
                        for row in s["crossing_nonactive_guard_signs"]["rows"]
                    ],
                },
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
