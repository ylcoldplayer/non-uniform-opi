#!/usr/bin/env python3
"""
C4+ positive-radius Poincare-section interval certificate.

This verifier upgrades the ambient 1e-8 diagnostic check. It verifies the
Poincare return map on the actual local section d_1(x)=0 through x0, using a
section-parameter box of radius 1e-3. The section is parametrized by
    x_2 = x0_2 + u,  x_3 = x0_3 + v,
    x_1 = x0_1 - (n_2 u + n_3 v)/n_1,
where n=grad d_1 and |u|, |v| <= r.

For each subbox of the (u,v)-box, the script:
  * interval-isolates the six event roots and certifies strict endpoint bracketing;
  * checks nonzero dphi/dr and incoming event-map denominators;
  * checks incoming and outgoing event normal velocities have the same nonzero sign;
  * certifies the first-hit itinerary by Bernstein guard-sign checks: the
    active exit guard is positive on the pre-bracket interval [R.hi,1], and
    the non-exit guards are positive on [R.lo,1] except for the intended
    entry endpoint zero at r=1;
  * checks non-active guard signs at each crossing box;
  * encloses the derivative of the six-event return map in section coordinates;
  * aggregates a global row-sum bound for D Pi on the section box.

The output proves, by the mean-value theorem in section coordinates,
    Pi(V_r) subset int(V_r)
provided residual + ||D Pi||_inf r < r, and certifies that the intended
six-guard itinerary is the true first-hit itinerary on the full section box.
"""
from decimal import Decimal
from pathlib import Path
import json
import itertools
import importlib.util

BASE_PATH = Path(__file__).with_name("verify_c4plus_decimal_interval.py")
spec = importlib.util.spec_from_file_location("base_c4", str(BASE_PATH))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

Interval = base.Interval
F = base.F

# Exact normal to the initial/return section d_1=0.
n0_q = [base.alpha_q * (base.P_q[0][1][j] - base.P_q[0][0][j]) for j in range(3)]
# Parameterization: dx = B [u,v]^T with dx_2=u, dx_3=v and n0^T dx = 0.
a_q = -n0_q[1] / n0_q[0]
b_q = -n0_q[2] / n0_q[0]
B = [
    [Interval.from_fraction(a_q), Interval.from_fraction(b_q)],
    [Interval(1), Interval(0)],
    [Interval(0), Interval(1)],
]

entry_guard_indices = [base.guard_indices[(k - 1) % len(base.guard_indices)] for k in range(len(base.guard_indices))]


def x_from_uv(U, V):
    x0 = [Interval(s) for s in base.x0_str]
    return [x0[0] + B[0][0] * U + B[0][1] * V,
            x0[1] + U,
            x0[2] + V]


def mat3x2(A, Bmat):
    return [[sum((A[i][k] * Bmat[k][j] for k in range(3)), Interval(0)) for j in range(2)] for i in range(3)]


def section_derivative_from_ambient(M):
    # Input perturbations are section coordinates (u,v), so ambient perturbation is B [du,dv].
    MB = mat3x2(M, B)
    # Output section coordinates are (x_2-x0_2, x_3-x0_3), i.e. rows 2 and 3 in one-based indexing.
    return [MB[1], MB[2]]


def inf_norm2(M2):
    rows = [base.directed_sum((M2[i][j].abs_upper() for j in range(2)), base.ROUND_CEILING) for i in range(2)]
    return max(rows), rows


def first_hit_guard_certificate(X, R, mu, k):
    """Certify the intended first-hit guard order on one segment.

    The true exit root is known, by the bracketing certificate, to lie in
    R=[R.lo,R.hi].  To prove first-hit validity before the root, the active
    exit guard is checked on the pre-bracket interval [R.hi,1] with no
    excluded Bernstein coefficient.  The non-exit guards are checked on the
    larger interval [R.lo,1]; only the intended entry guard is allowed its
    already-crossed endpoint zero at r=1.
    """
    exit_guard = base.guard_indices[k]
    entry_guard = entry_guard_indices[k]
    powers_full = base.guard_segment_power_polynomials(Interval(R.lo))
    powers_pre_exit = base.guard_segment_power_polynomials(Interval(R.hi))
    rows = []
    segment_min = Decimal("Infinity")
    for gi in range(3):
        if gi == exit_guard:
            powers = powers_pre_exit
            excluded = []
            interval_checked = "[R.hi,1]"
            allowed_endpoint_zeros = ["none"]
        else:
            powers = powers_full
            excluded = [40] if gi == entry_guard else []
            interval_checked = "[R.lo,1]"
            allowed_endpoint_zeros = ["entry"] if gi == entry_guard else ["none"]
        lower = base.signed_guard_bernstein_min_lower_from_powers(X, mu, gi, powers, excluded)
        if lower <= 0:
            raise RuntimeError(
                f"first-hit Bernstein guard-sign check failed on segment {k}, guard d_{gi+1}: "
                f"lower={lower}, excluded={excluded}, interval={interval_checked}"
            )
        segment_min = min(segment_min, lower)
        rows.append({
            "guard": gi + 1,
            "interval_checked": interval_checked,
            "excluded_bernstein_indices": excluded,
            "allowed_endpoint_zeros": allowed_endpoint_zeros,
            "min_nonexcluded_lower_bound": str(lower),
        })
    return {"min_nonexcluded_lower_bound": str(segment_min), "rows": rows}


def crossing_nonactive_guard_certificate(k, mu, Y):
    """Check that no non-active guard vanishes at the crossing box."""
    active = base.guard_indices[k]
    rows = []
    min_lower = Decimal("Infinity")
    for gi in range(3):
        if gi == active:
            continue
        val = base.signed_guard_value(mu, Y, gi)
        if val.lo <= 0:
            raise RuntimeError(f"crossing non-active guard failed on segment {k}, guard d_{gi+1}: {val}")
        min_lower = min(min_lower, val.lo)
        rows.append({"guard": gi + 1, "signed_guard": val.to_pair(), "min_lower_bound": str(val.lo)})
    return {"min_lower_bound": str(min_lower), "rows": rows}


def verify_uv_subbox(U, V, init_width="1e-4", iterations=8):
    X = x_from_uv(U, V)
    M = [[Interval(1 if i == j else 0) for j in range(3)] for i in range(3)]
    segs = []
    for k, mu in enumerate(base.policies):
        s = base.guard_indices[k]
        R = base.root_interval_newton(mu, X, s, base.z_str[k], init_width, iterations)
        bracket = base.root_bracket_certificate(mu, X, s, R)
        dR = base.dphi_dr(mu, X, R, s)
        if dR.contains_zero():
            raise RuntimeError(f"dphi/dr contains zero on segment {k}: {dR}")

        first_hit = first_hit_guard_certificate(X, R, mu, k)

        Y = base.flow(mu, X, R)
        Mk, denom = base.event_derivative(mu, Y, R, s)
        if denom.contains_zero():
            raise RuntimeError(f"event denominator contains zero on segment {k}: {denom}")
        mu_out = base.policies[(k + 1) % len(base.policies)]
        outgoing = base.normal_velocity(mu_out, Y, s)
        incoming_sign = denom.sign()
        outgoing_sign = outgoing.sign()
        same_nonzero = incoming_sign == outgoing_sign and incoming_sign in ("positive", "negative")
        if not same_nonzero:
            raise RuntimeError(
                f"positive-radius no-sliding check failed on segment {k}: incoming={denom}, outgoing={outgoing}"
            )
        crossing_nonactive = crossing_nonactive_guard_certificate(k, mu, Y)

        M = base.matmul(Mk, M)
        X = Y
        segs.append({
            "k": k,
            "root_interval": R,
            "root_width": R.width(),
            "root_bracket": bracket,
            "dphi_dr": dR,
            "denom": denom,
            "outgoing_denom": outgoing,
            "incoming_sign": incoming_sign,
            "outgoing_sign": outgoing_sign,
            "same_nonzero_sign": same_nonzero,
            "first_hit_guard_signs": first_hit,
            "crossing_nonactive_guard_signs": crossing_nonactive,
        })
    M2 = section_derivative_from_ambient(M)
    norm, rows = inf_norm2(M2)
    return X, M2, norm, rows, segs


def center_section_residual():
    X = [Interval(v) for v in base.x0_str]
    for k, mu in enumerate(base.policies):
        X = base.flow(mu, X, Interval(base.z_str[k]))
    # Section coordinates are second and third state-coordinate differences.
    r2 = X[1] - Interval(base.x0_str[1])
    r3 = X[2] - Interval(base.x0_str[2])
    return max(r2.abs_upper(), r3.abs_upper()), [r2, r3]


def verify_section(radius="1e-3", splits=4, init_width="1e-4", iterations=8):
    r = base.decimal_exact(radius)
    max_norm = Decimal(0)
    max_rows = [Decimal(0), Decimal(0)]
    max_root_width = Decimal(0)
    min_abs_dphi = Decimal("Infinity")
    min_abs_denom = Decimal("Infinity")
    min_abs_outgoing_denom = Decimal("Infinity")
    min_first_hit_guard = Decimal("Infinity")
    min_crossing_nonactive_guard = Decimal("Infinity")
    cells = []
    for iu, iv in itertools.product(range(splits), repeat=2):
        U = base.interval_grid_cell(r, iu, splits)
        V = base.interval_grid_cell(r, iv, splits)
        _Xf, _M2, norm, rows, segs = verify_uv_subbox(U, V, init_width, iterations)
        if norm > max_norm:
            max_norm = norm
        for j in range(2):
            if rows[j] > max_rows[j]:
                max_rows[j] = rows[j]
        for seg in segs:
            if seg["root_width"] > max_root_width:
                max_root_width = seg["root_width"]
            min_abs_dphi = min(min_abs_dphi, base.interval_abs_lower(seg["dphi_dr"]))
            min_abs_denom = min(min_abs_denom, base.interval_abs_lower(seg["denom"]))
            min_abs_outgoing_denom = min(min_abs_outgoing_denom, base.interval_abs_lower(seg["outgoing_denom"]))
            min_first_hit_guard = min(
                min_first_hit_guard,
                Decimal(seg["first_hit_guard_signs"]["min_nonexcluded_lower_bound"]),
            )
            min_crossing_nonactive_guard = min(
                min_crossing_nonactive_guard,
                Decimal(seg["crossing_nonactive_guard_signs"]["min_lower_bound"]),
            )
        cell_min_first_hit = min(Decimal(seg["first_hit_guard_signs"]["min_nonexcluded_lower_bound"]) for seg in segs)
        cell_min_crossing_nonactive = min(
            Decimal(seg["crossing_nonactive_guard_signs"]["min_lower_bound"]) for seg in segs
        )
        cell_min_outgoing = min(base.interval_abs_lower(seg["outgoing_denom"]) for seg in segs)
        cells.append({
            "iu": iu,
            "iv": iv,
            "u_interval": U.to_pair(),
            "v_interval": V.to_pair(),
            "row_sums": [str(x) for x in rows],
            "norm": str(norm),
            "max_root_width": str(max(seg["root_width"] for seg in segs)),
            "min_first_hit_guard_lower_bound": str(cell_min_first_hit),
            "min_crossing_nonactive_guard_lower_bound": str(cell_min_crossing_nonactive),
            "min_outgoing_event_denominator_abs_lower": str(cell_min_outgoing),
            "all_event_roots_strictly_bracketed": True,
            "all_incoming_outgoing_same_sign": True,
        })
    res, res_vec = center_section_residual()
    image_radius_upper = base.directed_sum_product(max_norm, r, res, base.ROUND_CEILING)
    ok_contraction = max_norm < Decimal(1)
    ok_containment = image_radius_upper < r
    return {
        "section": "d_1(x)=0, parameters u=x_2-x0_2, v=x_3-x0_3",
        "section_normal": [str(x) for x in n0_q],
        "section_basis_B": [[str(cell.lo) + "," + str(cell.hi) for cell in row] for row in B],
        "radius": radius,
        "splits_per_axis": splits,
        "subboxes": splits * splits,
        "init_width": init_width,
        "iterations": iterations,
        "center_section_residual_upper": str(res),
        "center_section_residual_vector": [x.to_pair() for x in res_vec],
        "max_section_derivative_row_sums": [str(x) for x in max_rows],
        "max_section_derivative_infinity_norm": str(max_norm),
        "max_event_root_width": str(max_root_width),
        "min_abs_dphi_dr": str(min_abs_dphi),
        "min_abs_event_denominator": str(min_abs_denom),
        "min_abs_outgoing_event_denominator": str(min_abs_outgoing_denom),
        "incoming_outgoing_same_sign_on_section_box": True,
        "min_first_hit_guard_bernstein_lower_bound": str(min_first_hit_guard),
        "first_hit_guard_signs_certified": True,
        "min_crossing_nonactive_guard_lower_bound": str(min_crossing_nonactive_guard),
        "crossing_nonactive_guard_signs_certified": True,
        "mvt_image_radius_upper": str(image_radius_upper),
        "strict_event_root_bracketing": True,
        "contraction_lt_1": ok_contraction,
        "mvt_containment_in_section_box": ok_containment,
        "cells": cells,
    }


def main():
    cert = verify_section()
    lines = []
    lines.append("C4+ POINCARE-SECTION INTERVAL CERTIFICATE")
    lines.append("==========================================")
    lines.append(f"Section: {cert['section']}")
    lines.append(f"Section normal grad d_1: {cert['section_normal']}")
    lines.append(f"Section box radius in (u,v): {cert['radius']}")
    lines.append(f"Subdivision grid: {cert['splits_per_axis']} x {cert['splits_per_axis']} = {cert['subboxes']} boxes")
    lines.append(f"Initial root bracket half-width: {cert['init_width']}")
    lines.append(f"Interval Newton iterations per event: {cert['iterations']}")
    lines.append("")
    lines.append(f"Center section residual upper: {Decimal(cert['center_section_residual_upper']):.18E}")
    lines.append("Max section-derivative row-sum upper bounds:")
    for i, row_sum in enumerate(cert["max_section_derivative_row_sums"]):
        lines.append(f"  row {i+1}: {Decimal(row_sum):.18E}")
    lines.append(f"Max ||D Pi(V)||_inf in section coordinates: {Decimal(cert['max_section_derivative_infinity_norm']):.18E}")
    lines.append(f"Max event-root interval width: {Decimal(cert['max_event_root_width']):.18E}")
    lines.append(f"Minimum |dphi/dr| lower bound: {Decimal(cert['min_abs_dphi_dr']):.18E}")
    lines.append(f"Minimum |incoming event denominator| lower bound: {Decimal(cert['min_abs_event_denominator']):.18E}")
    lines.append(f"Minimum |outgoing event denominator| lower bound: {Decimal(cert['min_abs_outgoing_event_denominator']):.18E}")
    lines.append(f"Incoming/outgoing same nonzero sign: {cert['incoming_outgoing_same_sign_on_section_box']} for every event on every subbox")
    lines.append(f"Minimum first-hit Bernstein guard-sign lower bound: {Decimal(cert['min_first_hit_guard_bernstein_lower_bound']):.18E}")
    lines.append(f"First-hit guard-sign Bernstein check: {cert['first_hit_guard_signs_certified']}")
    lines.append(f"Minimum crossing non-active guard lower bound: {Decimal(cert['min_crossing_nonactive_guard_lower_bound']):.18E}")
    lines.append(f"Crossing non-active guard signs certified: {cert['crossing_nonactive_guard_signs_certified']}")
    lines.append(f"Strict event-root bracketing: {cert['strict_event_root_bracketing']} for every event on every subbox")
    lines.append(f"MVT image-radius upper: {Decimal(cert['mvt_image_radius_upper']):.18E}")
    lines.append(f"Contraction < 1: {cert['contraction_lt_1']}")
    lines.append(f"Pi(V) subset int(V) by MVT: {cert['mvt_containment_in_section_box']}")
    lines.append("")
    lines.append("Per-subbox derivative norms and guard minima:")
    for c in cert["cells"]:
        lines.append(
            f"  cell ({c['iu']},{c['iv']}): row sums {c['row_sums']}, norm {c['norm']}, "
            f"min_first_hit_guard {Decimal(c['min_first_hit_guard_lower_bound']):.12E}"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    out_dir = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
    (out_dir / "c4plus_section_interval_certificate.log").write_text(text)
    (out_dir / "c4plus_section_interval_certificate.json").write_text(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
