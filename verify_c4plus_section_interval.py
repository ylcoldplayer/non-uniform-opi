#!/usr/bin/env python3
"""
C4+ positive-radius Poincare-section interval certificate.

This verifier upgrades the earlier ambient 1e-8 C4+ check.  It verifies the
Poincare return map on the actual local section d_1(x)=0 through x0, using a
section-parameter box of radius 1e-3.  The section is parametrized by
    x_2 = x0_2 + u,  x_3 = x0_3 + v,
    x_1 = x0_1 - (n_2 u + n_3 v)/n_1,
where n=grad d_1 and |u|,|v| <= r.

For each subbox of the (u,v)-box, the script:
  * interval-isolates the six event roots;
  * checks event denominators stay away from zero;
  * encloses the derivative of the six-event return map in section coordinates;
  * aggregates a global row-sum bound for D Pi on the section box.

The output proves, by the mean-value theorem in section coordinates,
    Pi(V_r) subset int(V_r)
provided residual + ||D Pi||_inf r < r.
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
    rows = [sum((M2[i][j].abs_upper() for j in range(2)), Decimal(0)) for i in range(2)]
    return max(rows), rows


def verify_uv_subbox(U, V, init_width="1e-4", iterations=8):
    X = x_from_uv(U, V)
    M = [[Interval(1 if i == j else 0) for j in range(3)] for i in range(3)]
    segs = []
    for k, mu in enumerate(base.policies):
        s = base.guard_indices[k]
        R = base.root_interval_newton(mu, X, s, base.z_str[k], init_width, iterations)
        dR = base.dphi_dr(mu, X, R, s)
        if dR.contains_zero():
            raise RuntimeError(f"dphi/dr contains zero on segment {k}: {dR}")
        Y = base.flow(mu, X, R)
        Mk, denom = base.event_derivative(mu, Y, R, s)
        if denom.contains_zero():
            raise RuntimeError(f"event denominator contains zero on segment {k}: {denom}")
        M = base.matmul(Mk, M)
        X = Y
        segs.append({
            "k": k,
            "root_interval": R,
            "root_width": R.width(),
            "dphi_dr": dR,
            "denom": denom,
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
    r = Decimal(radius)
    max_norm = Decimal(0)
    max_rows = [Decimal(0), Decimal(0)]
    max_root_width = Decimal(0)
    min_abs_dphi = Decimal("Infinity")
    min_abs_denom = Decimal("Infinity")
    cells = []
    for iu, iv in itertools.product(range(splits), repeat=2):
        U = Interval(-r + Decimal(2) * r * Decimal(iu) / Decimal(splits),
                     -r + Decimal(2) * r * Decimal(iu + 1) / Decimal(splits))
        V = Interval(-r + Decimal(2) * r * Decimal(iv) / Decimal(splits),
                     -r + Decimal(2) * r * Decimal(iv + 1) / Decimal(splits))
        Xf, M2, norm, rows, segs = verify_uv_subbox(U, V, init_width, iterations)
        if norm > max_norm:
            max_norm = norm
        for j in range(2):
            if rows[j] > max_rows[j]:
                max_rows[j] = rows[j]
        for s in segs:
            if s["root_width"] > max_root_width:
                max_root_width = s["root_width"]
            min_abs_dphi = min(min_abs_dphi, min(abs(s["dphi_dr"].lo), abs(s["dphi_dr"].hi)))
            min_abs_denom = min(min_abs_denom, min(abs(s["denom"].lo), abs(s["denom"].hi)))
        cells.append({
            "iu": iu,
            "iv": iv,
            "u_interval": U.to_pair(),
            "v_interval": V.to_pair(),
            "row_sums": [str(x) for x in rows],
            "norm": str(norm),
            "root_intervals": [s["root_interval"].to_pair() for s in segs],
            "root_widths": [str(s["root_width"]) for s in segs],
            "dphi_dr": [s["dphi_dr"].to_pair() for s in segs],
            "event_denominators": [s["denom"].to_pair() for s in segs],
        })
    res, res_vec = center_section_residual()
    image_radius_upper = res + max_norm * r
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
        "mvt_image_radius_upper": str(image_radius_upper),
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
    for i, r in enumerate(cert["max_section_derivative_row_sums"]):
        lines.append(f"  row {i+1}: {Decimal(r):.18E}")
    lines.append(f"Max ||D Pi(V)||_inf in section coordinates: {Decimal(cert['max_section_derivative_infinity_norm']):.18E}")
    lines.append(f"Max event-root interval width: {Decimal(cert['max_event_root_width']):.18E}")
    lines.append(f"Minimum |dphi/dr| lower bound: {Decimal(cert['min_abs_dphi_dr']):.18E}")
    lines.append(f"Minimum |event denominator| lower bound: {Decimal(cert['min_abs_event_denominator']):.18E}")
    lines.append(f"MVT image-radius upper: {Decimal(cert['mvt_image_radius_upper']):.18E}")
    lines.append(f"Contraction < 1: {cert['contraction_lt_1']}")
    lines.append(f"Pi(V) subset int(V) by MVT: {cert['mvt_containment_in_section_box']}")
    lines.append("")
    lines.append("Per-subbox derivative norms:")
    for c in cert["cells"]:
        lines.append(f"  cell ({c['iu']},{c['iv']}): row sums {c['row_sums']}, norm {c['norm']}")
    text = "\n".join(lines) + "\n"
    print(text)
    out_dir = Path("/mnt/data") if Path("/mnt/data").exists() else Path(".")
    (out_dir / "c4plus_section_interval_certificate.log").write_text(text)
    (out_dir / "c4plus_section_interval_certificate.json").write_text(json.dumps(cert, indent=2))


if __name__ == "__main__":
    main()
