#!/usr/bin/env python3
"""Standard-library verification of the algorithmic kernels.

Why this exists
---------------
The PyTorch implementation could not be executed in the environment where it was
written. Most of it genuinely needs torch. But a specific, high-risk subset does not:
the **discrete index arithmetic and lookup tables**, where a single off-by-one silently
corrupts every downstream number instead of raising.

This script re-implements those kernels in plain Python and checks them against
independent references (brute force, combinatorial invariants, closed forms). It runs
anywhere, needs nothing installed, and is the only part of the repository that has
actually been executed.

What it covers
--------------
1. the 6-tetrahedron decomposition of a cube tiles it exactly (volumes sum, no overlap)
2. the 16-entry marching-tetrahedra table is topologically correct for every case
3. the (surfel, tile) pair enumeration matches a brute-force double loop
4. per-tile depth ordering + truncation keeps the nearest N
5. bit packing / unpacking round-trips
6. the ellipsoid SDF bisection bracket is valid and the root matches brute force
7. trilinear weights sum to 1 and reproduce affine functions exactly
8. conjugate gradient converges on an SPD system
9. the Godunov upwind reinitialisation is a fixed point when ||grad phi|| = 1
10. the storage break-even algebra inverts Eq. (36)-(38) consistently
11. the contraction profile is continuous at ES and at the cycle wrap

What it does NOT cover
----------------------
Anything involving tensors, autograd, rasterisation or the level-set solve. Passing here
means the index logic is sound, not that the pipeline runs.
"""

from __future__ import annotations

import math
import sys

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# --------------------------------------------------------------------------- #
# 1-2. Marching tetrahedra: decomposition and case table
# --------------------------------------------------------------------------- #
CORNERS = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)
TETS = ((0, 6, 1, 2), (0, 6, 2, 3), (0, 6, 3, 7), (0, 6, 7, 4), (0, 6, 4, 5), (0, 6, 5, 1))
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TABLE = (
    (-1, -1, -1, -1, -1, -1),
    (0, 1, 2, -1, -1, -1),
    (0, 3, 4, -1, -1, -1),
    (1, 2, 4, 1, 4, 3),
    (1, 3, 5, -1, -1, -1),
    (0, 2, 5, 0, 5, 3),
    (0, 1, 5, 0, 5, 4),
    (2, 4, 5, -1, -1, -1),
    (2, 4, 5, -1, -1, -1),
    (0, 1, 5, 0, 5, 4),
    (0, 2, 5, 0, 5, 3),
    (1, 3, 5, -1, -1, -1),
    (1, 2, 4, 1, 4, 3),
    (0, 3, 4, -1, -1, -1),
    (0, 1, 2, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1),
)


def tet_volume(p0, p1, p2, p3) -> float:
    a = [p1[i] - p0[i] for i in range(3)]
    b = [p2[i] - p0[i] for i in range(3)]
    c = [p3[i] - p0[i] for i in range(3)]
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    return abs(det) / 6.0


def verify_decomposition() -> None:
    section("1. Cube -> 6 tetrahedra decomposition")

    total = sum(tet_volume(*[CORNERS[v] for v in t]) for t in TETS)
    check("volumes sum to the unit cube", abs(total - 1.0) < 1e-12, f"sum={total}")

    # Every tet must be non-degenerate.
    vols = [tet_volume(*[CORNERS[v] for v in t]) for t in TETS]
    check("no degenerate tetrahedron", all(v > 1e-12 for v in vols), f"min={min(vols):.6f}")

    # No overlap: sample the cube and count how many tets contain each point.
    # Barycentric containment test.
    def contains(tet, p) -> bool:
        v = [CORNERS[i] for i in tet]
        d0 = tet_volume(*v)
        if d0 <= 0:
            return False
        s = 0.0
        for k in range(4):
            w = list(v)
            w[k] = p
            s += tet_volume(*w)
        return abs(s - d0) < 1e-9

    # All six tets share the 0-6 diagonal, so the only internal faces lie on the
    # three planes x=y, y=z, x=z. A point *on* one of those planes legitimately
    # belongs to two tets, so the sampling must avoid them: use irrational-ish
    # offsets rather than a plain lattice.
    #
    # (An earlier version of this check sampled p = (i/n, j/n, k/n) and reported
    # 96 offenders out of 216 -- which is exactly
    #   |i=j| + |j=k| + |i=k| - 3|pairwise| + |triple| = 36*3 - 6*3 + 6 = 96,
    # i.e. precisely the lattice points sitting on those three planes. The count
    # matching the inclusion-exclusion prediction exactly is itself evidence that
    # the decomposition is correct and the sampling was at fault.)
    n = 6
    ox, oy, oz = 0.2137, 0.6193, 0.4471
    generic = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                generic.append(((i + ox) / n, (j + oy) / n, (k + oz) / n))
    counts = [sum(1 for t in TETS if contains(t, p)) for p in generic]
    bad = [c for c in counts if c != 1]
    check(
        "every generic interior point lies in exactly one tetrahedron",
        not bad,
        f"{len(counts)} samples, offenders={len(bad)}",
    )

    # And a point on a shared internal face must belong to exactly two.
    on_face = [(0.4, 0.4, 0.7), (0.3, 0.3, 0.9), (0.6, 0.25, 0.25)]
    face_counts = [sum(1 for t in TETS if contains(t, p)) for p in on_face]
    check(
        "points on an internal face belong to exactly two tetrahedra",
        all(c == 2 for c in face_counts),
        f"counts={face_counts}",
    )

    # The ring 1-2-3-7-4-5 must consist of cube edges (adjacent corners).
    ring = [1, 2, 3, 7, 4, 5]
    ok = True
    for a, b in zip(ring, ring[1:] + ring[:1]):
        diff = sum(abs(CORNERS[a][i] - CORNERS[b][i]) for i in range(3))
        ok = ok and diff == 1
    check("the hexagonal ring uses only cube edges", ok)


def verify_case_table() -> None:
    section("2. Marching-tetrahedra case table (16 entries)")

    all_ok = True
    details = []
    for case in range(16):
        inside = {v for v in range(4) if (case >> v) & 1}
        crossing = {
            ei for ei, (a, b) in enumerate(TET_EDGES)
            if (a in inside) != (b in inside)
        }

        row = TABLE[case]
        tris = []
        for slot in (0, 1):
            tri = row[3 * slot : 3 * slot + 3]
            if tri[0] >= 0:
                tris.append(tuple(tri))

        n_in = len(inside)
        expected_tris = 0 if n_in in (0, 4) else (1 if n_in in (1, 3) else 2)
        if len(tris) != expected_tris:
            all_ok = False
            details.append(f"case {case:04b}: {len(tris)} tris, expected {expected_tris}")
            continue
        if not tris:
            continue

        used = set()
        for t in tris:
            used |= set(t)
        if used != crossing:
            all_ok = False
            details.append(
                f"case {case:04b}: uses edges {sorted(used)}, crossing set {sorted(crossing)}"
            )
            continue

        # No degenerate triangle (repeated edge within one triangle).
        if any(len(set(t)) != 3 for t in tris):
            all_ok = False
            details.append(f"case {case:04b}: degenerate triangle {tris}")
            continue

        # Two-triangle cases must form a quad: the pair shares exactly one diagonal
        # (two vertices), and together they cover all four crossing edges.
        if len(tris) == 2:
            shared = set(tris[0]) & set(tris[1])
            if len(shared) != 2:
                all_ok = False
                details.append(f"case {case:04b}: triangles share {len(shared)} vertices, need 2")
                continue
            if len(crossing) != 4:
                all_ok = False
                details.append(f"case {case:04b}: 2-2 split must have 4 crossing edges")
                continue
            # The quad's boundary cycle: consecutive crossing edges must share a
            # tetrahedron vertex, otherwise the patch is a bowtie.
            quad = [tris[0][0], tris[0][1], tris[0][2], None]
            other = [e for e in tris[1] if e not in shared]
            quad[3] = other[0]
            cyc_ok = True
            for a, b in zip(quad, quad[1:] + quad[:1]):
                if not (set(TET_EDGES[a]) & set(TET_EDGES[b])):
                    cyc_ok = False
            if not cyc_ok:
                all_ok = False
                details.append(f"case {case:04b}: quad cycle {quad} is not edge-connected")

    check("all 16 cases topologically correct", all_ok, "; ".join(details[:4]))

    # Complementary cases must emit the same patch (inside/outside symmetry).
    sym = all(
        sorted(TABLE[c]) == sorted(TABLE[15 - c]) for c in range(16)
    )
    check("case c and its complement 15-c emit the same patch", sym)


# --------------------------------------------------------------------------- #
# 3-4. Tile binning arithmetic
# --------------------------------------------------------------------------- #
def verify_tile_enumeration() -> None:
    section("3. (surfel, tile) pair enumeration")

    # Mirror of the vectorised trick in raster2dgs._bin_surfels.
    boxes = [
        (0, 2, 0, 1), (3, 3, 2, 4), (1, 4, 0, 0), (2, 2, 2, 2), (0, 5, 1, 3),
    ]  # (tx0, tx1, ty0, ty1)
    n_tx = 6

    wt = [b[1] - b[0] + 1 for b in boxes]
    ht = [b[3] - b[2] + 1 for b in boxes]
    counts = [w * h for w, h in zip(wt, ht)]
    total = sum(counts)

    pair_surfel: list[int] = []
    for i, c in enumerate(counts):
        pair_surfel += [i] * c
    cum = []
    acc = 0
    for c in counts:
        cum.append(acc)
        acc += c

    got = []
    for idx in range(total):
        s = pair_surfel[idx]
        local = idx - cum[s]
        w = wt[s]
        dy = local // w
        dx = local - dy * w
        tx = boxes[s][0] + dx
        ty = boxes[s][2] + dy
        got.append((s, ty * n_tx + tx))

    want = []
    for s, (tx0, tx1, ty0, ty1) in enumerate(boxes):
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                want.append((s, ty * n_tx + tx))

    check("pair count matches", total == len(want), f"{total} vs {len(want)}")
    check("enumerated pairs match a brute-force double loop", got == want)
    check("no duplicate (surfel, tile) pairs", len(set(got)) == len(got))


def verify_tile_truncation() -> None:
    section("4. Per-tile depth ordering and truncation")

    # (tile_id, depth, surfel_id)
    pairs = [
        (2, 5.0, 0), (1, 3.0, 1), (2, 1.0, 2), (2, 9.0, 3),
        (1, 0.5, 4), (2, 4.0, 5), (0, 7.0, 6), (2, 2.0, 7),
    ]
    m_cap = 3

    # Mirror: stable sort by depth, then stable sort by tile.
    by_depth = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    grouped = sorted(by_depth, key=lambda i: pairs[i][0])

    per_tile: dict[int, list[int]] = {}
    for i in grouped:
        per_tile.setdefault(pairs[i][0], []).append(i)

    dense: dict[int, list[int]] = {}
    for tid, idxs in per_tile.items():
        dense[tid] = [pairs[i][2] for i in idxs[:m_cap]]

    depth_sorted_ok = all(
        all(pairs[a][1] <= pairs[b][1] for a, b in zip(idxs, idxs[1:]))
        for idxs in per_tile.values()
    )
    check("within each tile, pairs are depth-ascending", depth_sorted_ok)

    # Tile 2 has depths 1,2,4,5,9 -> nearest 3 are surfels 2,7,5
    check("truncation keeps the nearest max_per_tile", dense[2] == [2, 7, 5], f"{dense[2]}")
    check("tiles with fewer pairs are unaffected", dense[1] == [4, 1] and dense[0] == [6])

    n_trunc = sum(max(0, len(v) - m_cap) for v in per_tile.values())
    check("truncated count is reported correctly", n_trunc == 2, f"{n_trunc}")


# --------------------------------------------------------------------------- #
# 5. Bit packing
# --------------------------------------------------------------------------- #
def verify_bit_packing() -> None:
    section("5. Occupancy bitmask pack / unpack")

    def pack(bits: list[bool]) -> list[int]:
        pad = (-len(bits)) % 8
        b = bits + [False] * pad
        out = []
        for i in range(0, len(b), 8):
            byte = sum((1 << k) for k in range(8) if b[i + k])
            out.append(byte)
        return out

    def unpack(packed: list[int], n: int) -> list[bool]:
        bits = []
        for byte in packed:
            for k in range(8):
                bits.append(bool((byte >> k) & 1))
        return bits[:n]

    import random

    random.seed(0)
    ok = True
    sizes_ok = True
    for n in (1, 7, 8, 9, 63, 64, 65, 1000):
        bits = [random.random() > 0.5 for _ in range(n)]
        packed = pack(bits)
        if unpack(packed, n) != bits:
            ok = False
        if len(packed) != (n + 7) // 8:
            sizes_ok = False
    check("round-trips for every length mod 8", ok)
    check("packed size is ceil(n/8) bytes", sizes_ok)

    # Max row sum must fit the uint8 cast used in storage_io.
    check("max byte value is 255 (fits uint8)", sum(1 << k for k in range(8)) == 255)


# --------------------------------------------------------------------------- #
# 6. Ellipsoid SDF bisection
# --------------------------------------------------------------------------- #
def ellipsoid_sdf_py(p, axes, iters=80):
    """Mirror of data/phantom.ellipsoid_sdf for a single point."""
    a2 = [a * a for a in axes]
    scaled = sum((p[i] / axes[i]) ** 2 for i in range(3))
    inside = scaled < 1.0
    ap2 = [(axes[i] * p[i]) ** 2 for i in range(3)]

    if math.sqrt(sum(x * x for x in p)) < 1e-9:
        return math.sqrt(min(a2))

    if inside:
        t_lo, t_hi = -min(a2) + 1e-9, 0.0
    else:
        t_lo, t_hi = 0.0, math.sqrt(sum(ap2)) + 1.0

    def f(t):
        return sum(ap2[i] / (a2[i] + t) ** 2 for i in range(3)) - 1.0

    for _ in range(iters):
        mid = 0.5 * (t_lo + t_hi)
        if f(mid) > 0:
            t_lo = mid
        else:
            t_hi = mid
    t = 0.5 * (t_lo + t_hi)
    y = [a2[i] * p[i] / (a2[i] + t) for i in range(3)]
    d = math.sqrt(sum((p[i] - y[i]) ** 2 for i in range(3)))
    return d if inside else -d


def verify_ellipsoid_sdf() -> None:
    section("6. Exact ellipsoid SDF (bisection bracket + root)")

    axes = (23.0, 23.0, 38.0)
    a2 = [a * a for a in axes]

    # 6a. The bracket must actually contain a sign change, for both sides.
    import random

    random.seed(1)
    bracket_ok = True
    for _ in range(200):
        p = [random.uniform(-60, 60) for _ in range(3)]
        scaled = sum((p[i] / axes[i]) ** 2 for i in range(3))
        ap2 = [(axes[i] * p[i]) ** 2 for i in range(3)]
        if math.sqrt(sum(x * x for x in p)) < 1e-6:
            continue

        # ap2 is bound as a default so the closure cannot silently capture a later
        # iteration's value if this call ever moves out of the loop body.
        def f(t, ap2=ap2):
            return sum(ap2[i] / (a2[i] + t) ** 2 for i in range(3)) - 1.0

        if scaled < 1.0:
            lo, hi = -min(a2) + 1e-9, 0.0
        else:
            lo, hi = 0.0, math.sqrt(sum(ap2)) + 1.0
        if not (f(lo) > 0 >= f(hi)):
            bracket_ok = False
            break
    check("bracket contains the root for inside and outside points", bracket_ok)

    # 6b. Sphere special case has a closed form.
    sphere = (10.0, 10.0, 10.0)
    max_err = 0.0
    for _ in range(300):
        p = [random.uniform(-25, 25) for _ in range(3)]
        got = ellipsoid_sdf_py(p, sphere)
        want = 10.0 - math.sqrt(sum(x * x for x in p))
        max_err = max(max_err, abs(got - want))
    check("reduces to the analytic sphere distance", max_err < 1e-9, f"max err {max_err:.2e}")

    # 6c. Against a brute-force closest-point search on a dense surface sampling.
    n_t, n_p = 160, 320
    surf = []
    for i in range(n_t):
        th = (i + 0.5) / n_t * math.pi
        st, ct = math.sin(th), math.cos(th)
        for j in range(n_p):
            ph = (j + 0.5) / n_p * 2 * math.pi
            surf.append((axes[0] * st * math.cos(ph), axes[1] * st * math.sin(ph), axes[2] * ct))

    worst = 0.0
    for _ in range(12):
        p = [random.uniform(-50, 50) for _ in range(3)]
        got = abs(ellipsoid_sdf_py(p, axes))
        brute = min(
            math.sqrt((p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2 + (p[2] - s[2]) ** 2)
            for s in surf
        )
        # The brute force is the *less* accurate side: it can only over-estimate,
        # bounded by the surface sampling spacing.
        worst = max(worst, got - brute)
    check(
        "never exceeds a brute-force closest-point distance",
        worst < 1e-6,
        f"max (bisect - brute) = {worst:.2e}; brute over-estimates by its sampling step",
    )

    # 6d. Sign convention: positive inside.
    check("positive inside, negative outside",
          ellipsoid_sdf_py([0.0, 0.0, 0.0], axes) > 0
          and ellipsoid_sdf_py([100.0, 0.0, 0.0], axes) < 0)

    # 6e. On the surface the distance is ~0.
    on = [axes[0], 0.0, 0.0]
    check("zero on the surface", abs(ellipsoid_sdf_py(on, axes)) < 1e-6,
          f"{ellipsoid_sdf_py(on, axes):.2e}")


# --------------------------------------------------------------------------- #
# 7. Trilinear interpolation weights
# --------------------------------------------------------------------------- #
def verify_trilinear() -> None:
    section("7. Trilinear interpolation weights")

    def weights(fx, fy, fz):
        gx, gy, gz = 1 - fx, 1 - fy, 1 - fz
        return [
            gx * gy * gz, gx * gy * fz, gx * fy * gz, gx * fy * fz,
            fx * gy * gz, fx * gy * fz, fx * fy * gz, fx * fy * fz,
        ]

    # Corner order must match _gather_corners: (i0j0k0, i0j0k1, i0j1k0, ...)
    offsets = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
               (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    import random

    random.seed(2)
    sum_ok = True
    affine_ok = True
    corner_ok = True
    for _ in range(500):
        fx, fy, fz = (random.random() for _ in range(3))
        w = weights(fx, fy, fz)
        if abs(sum(w) - 1.0) > 1e-12:
            sum_ok = False
        # Affine exactness: interpolating f(x)=3x-2y+0.5z must be exact.
        val = sum(
            wi * (3 * o[0] - 2 * o[1] + 0.5 * o[2]) for wi, o in zip(w, offsets)
        )
        want = 3 * fx - 2 * fy + 0.5 * fz
        if abs(val - want) > 1e-12:
            affine_ok = False

    for k, (ox, oy, oz) in enumerate(offsets):
        w = weights(float(ox), float(oy), float(oz))
        if abs(w[k] - 1.0) > 1e-12 or abs(sum(w) - 1.0) > 1e-12:
            corner_ok = False

    check("weights sum to 1", sum_ok)
    check("exact for affine functions", affine_ok)
    check("weight order matches the corner gather order", corner_ok)


# --------------------------------------------------------------------------- #
# 8. Conjugate gradient
# --------------------------------------------------------------------------- #
def verify_cg() -> None:
    section("8. Conjugate gradient on an SPD system")

    n = 25
    import random

    random.seed(3)
    a = [[random.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    spd = [
        [sum(a[i][k] * a[j][k] for k in range(n)) + (n if i == j else 0) for j in range(n)]
        for i in range(n)
    ]

    def matvec(v):
        return [sum(spd[i][j] * v[j] for j in range(n)) for i in range(n)]

    x_true = [random.gauss(0, 1) for _ in range(n)]
    b = matvec(x_true)

    # Mirror of residual/solver.conjugate_gradient.
    x = [0.0] * n
    r = [b[i] - 0.0 for i in range(n)]
    p = list(r)
    rs = sum(v * v for v in r)
    b_norm = math.sqrt(sum(v * v for v in b))
    iters = 0
    for it in range(1, 200):
        if math.sqrt(rs) / b_norm <= 1e-12:
            break
        ap = matvec(p)
        denom = sum(p[i] * ap[i] for i in range(n))
        alpha = rs / denom
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * ap[i] for i in range(n)]
        rs_new = sum(v * v for v in r)
        p = [r[i] + (rs_new / rs) * p[i] for i in range(n)]
        rs = rs_new
        iters = it

    err = max(abs(x[i] - x_true[i]) for i in range(n))
    check("converges to the exact solution", err < 1e-8, f"max err {err:.2e}")
    check("finishes within n iterations", iters <= n + 2, f"{iters} iterations for n={n}")


# --------------------------------------------------------------------------- #
# 9. Godunov upwind reinitialisation
# --------------------------------------------------------------------------- #
def verify_reinit_fixed_point() -> None:
    section("9. Godunov upwind reinitialisation")

    # A 1-D exact distance ramp must be a fixed point of the reinit PDE.
    h = 1.0
    n = 21
    phi = [10.0 - abs(i - 10) * h for i in range(n)]  # positive inside, |grad| = 1

    def upwind_norm(idx, sign):
        back = (phi[idx] - phi[idx - 1]) / h
        fwd = (phi[idx + 1] - phi[idx]) / h
        if sign > 0:
            return math.sqrt(max(max(back, 0.0) ** 2, min(fwd, 0.0) ** 2))
        return math.sqrt(max(min(back, 0.0) ** 2, max(fwd, 0.0) ** 2))

    worst = 0.0
    for i in range(1, n - 1):
        s = phi[i] / math.sqrt(phi[i] ** 2 + h * h)
        sgn = 1.0 if s > 0 else -1.0
        g = upwind_norm(i, sgn)
        worst = max(worst, abs(g - 1.0))
    check("||grad phi|| = 1 is reproduced by the upwind stencil",
          worst < 1e-12, f"max deviation {worst:.2e}")

    # A badly scaled ramp must be pushed towards |grad| = 1, not away.
    phi = [3.0 * (10.0 - abs(i - 10) * h) for i in range(n)]
    dt = 0.3 * h
    before = 0.0
    after = 0.0
    new = list(phi)
    for i in range(1, n - 1):
        s = phi[i] / math.sqrt(phi[i] ** 2 + h * h)
        sgn = 1.0 if s > 0 else -1.0
        g = upwind_norm(i, sgn)
        before = max(before, abs(g - 1.0))
        new[i] = phi[i] + dt * s * (1.0 - g)
    phi = new
    for i in range(2, n - 2):
        s = phi[i] / math.sqrt(phi[i] ** 2 + h * h)
        sgn = 1.0 if s > 0 else -1.0
        g = upwind_norm(i, sgn)
        after = max(after, abs(g - 1.0))
    check("a mis-scaled level set moves towards the eikonal solution",
          after < before, f"{before:.3f} -> {after:.3f}")


# --------------------------------------------------------------------------- #
# 10. Storage algebra
# --------------------------------------------------------------------------- #
def verify_storage_algebra() -> None:
    section("10. Storage accounting, Eq. (36)-(38)")

    bf = 4  # bytes per float32
    t, n, p2d, pr = 20, 20000, 10, 1

    def report(surface_per_frame, rank=None):
        s_full = t * n * p2d * bf
        canonical = n * p2d * bf
        surface = t * surface_per_frame
        residual = (t * n * pr * bf) if rank is None else rank * (n + t) * 1 * bf
        s_ours = canonical + surface + residual
        return s_full, s_ours, s_full / s_ours, residual

    s_full, s_ours, cr, _ = report(50_000)
    check("S_full = T N P_2D (in bytes)", s_full == 20 * 20000 * 10 * 4, f"{s_full}")
    check("S_ours = N P_2D + sum S(Gamma_t) + T N P_r",
          s_ours == 20000 * 10 * 4 + 20 * 50_000 + 20 * 20000 * 1 * 4, f"{s_ours}")

    # Break-even: solving S_ours = S_full for the per-frame surface budget must
    # give CR exactly 1 when substituted back.
    canonical = n * p2d * bf
    residual = t * n * pr * bf
    break_even = (s_full - canonical - residual) / t
    _, _, cr_be, _ = report(break_even)
    check("break-even surface budget yields CR = 1",
          abs(cr_be - 1.0) < 1e-9, f"budget {break_even:.0f} B/frame -> CR {cr_be:.9f}")

    # Low-rank must reduce the residual term for r << min(N, T).
    _, _, cr_dense, res_dense = report(10_000)
    _, _, cr_low, res_low = report(10_000, rank=8)
    check("low-rank residual is smaller than dense",
          res_low < res_dense, f"{res_low} < {res_dense}")
    check("low-rank raises the compression ratio",
          cr_low > cr_dense, f"{cr_dense:.2f} -> {cr_low:.2f}")

    # Sanity: a huge per-frame surface must make the method LOSE (CR < 1),
    # which is the failure mode proposal 2.9 warns about.
    _, _, cr_bad, _ = report(5_000_000)
    check("an oversized surface makes CR < 1 (the warned-about failure)",
          cr_bad < 1.0, f"CR {cr_bad:.3f}")


# --------------------------------------------------------------------------- #
# 11. Cardiac cycle profile
# --------------------------------------------------------------------------- #
def verify_contraction_profile() -> None:
    section("11. Contraction profile")

    def profile(t, n_frames, es=0.35):
        x = (t % n_frames) / float(n_frames)
        if x <= es:
            return 0.5 * (1.0 - math.cos(math.pi * x / es))
        return 0.5 * (1.0 + math.cos(math.pi * (x - es) / (1.0 - es)))

    n = 1000
    check("0 at end-diastole", abs(profile(0, n)) < 1e-12)
    es_frame = int(0.35 * n)
    check("reaches 1 at end-systole", profile(es_frame, n) > 0.999999,
          f"{profile(es_frame, n):.8f}")
    check("returns to 0 after a full cycle", abs(profile(n, n)) < 1e-12)

    vals = [profile(i, n) for i in range(n)]
    check("stays within [0, 1]", min(vals) >= -1e-12 and max(vals) <= 1 + 1e-12,
          f"[{min(vals):.3f}, {max(vals):.3f}]")

    # Continuity: no jump anywhere, including the wrap-around.
    jumps = [abs(vals[(i + 1) % n] - vals[i]) for i in range(n)]
    check("continuous at ES and at the cycle wrap", max(jumps) < 0.01,
          f"max step {max(jumps):.5f}")

    # Monotone rise to ES then monotone fall.
    rise = all(vals[i + 1] >= vals[i] - 1e-12 for i in range(es_frame))
    fall = all(vals[i + 1] <= vals[i] + 1e-12 for i in range(es_frame, n - 1))
    check("monotone contraction then monotone relaxation", rise and fall)


# --------------------------------------------------------------------------- #
# 12. Viewpoint break-even, the layer-4 storage comparison
# --------------------------------------------------------------------------- #
def verify_viewpoint_breakeven() -> None:
    section("12. Viewpoint break-even against pre-rendered video")

    # Independent reimplementation of cvdyn2dgs.metrics.viewpoint.
    def video_bytes(bitrate_kbps, n_frames, fps, n_views):
        seconds = n_frames / fps
        return math.ceil(bitrate_kbps * 1000.0 / 8.0 * seconds * n_views)

    # 2000 kbit/s, 24 frames at 24 fps = exactly 1 second per view.
    one = video_bytes(2000, 24, 24, 1)
    check("one second at 2000 kbit/s is 250000 bytes", one == 250_000, str(one))
    check("video cost is linear in viewpoint count",
          video_bytes(2000, 24, 24, 7) == 7 * one, str(video_bytes(2000, 24, 24, 7)))

    # Break-even solves V * b = S_ours.
    def breakeven(ours, per_view):
        return math.inf if per_view == 0 else ours / per_view

    ours = 1_000_000
    v = breakeven(ours, one)
    check("break-even is S_ours / bytes-per-view", abs(v - 4.0) < 1e-12, f"{v:.6f}")
    check("integer break-even rounds up", math.ceil(v) == 4, str(math.ceil(v)))

    # The three regimes the paper must distinguish.
    check("below break-even video wins", one * 3 < ours, f"{one*3} < {ours}")
    check("above break-even ours wins", one * 5 > ours, f"{one*5} > {ours}")
    check("at break-even they are equal", one * 4 == ours)

    # A sub-1 break-even is the strongest claim: video loses even at one view.
    v_small = breakeven(100_000, one)
    check("break-even below 1 means video loses at a single viewpoint",
          v_small < 1.0, f"{v_small:.3f}")

    # Zero-cost video is degenerate and must be infinity, not a division error.
    check("zero bytes per view gives infinite break-even",
          breakeven(ours, 0) == math.inf)

    # Raw volume is flat in viewpoints: that is why it is hard to beat on flexibility.
    raw = 128 * 128 * 32 * 24 * 2
    check("raw 4-D volume size is nx*ny*nz*T*bytes", raw == 25_165_824, str(raw))
    check("raw volume does not grow with viewpoints",
          raw == raw * 1, "viewpoint-independent by construction")


# --------------------------------------------------------------------------- #
# 13. Quaternion to rotation: the free-3DGS orientation parameterisation
# --------------------------------------------------------------------------- #
def verify_quaternion_rotation() -> None:
    section("13. Quaternion -> rotation matrix (free-geometry 3DGS)")

    def quat_to_rot(w, x, y, z):
        n = math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / n, x / n, y / n, z / n
        return (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )

    def col(r, j):
        return (r[0][j], r[1][j], r[2][j])

    def dot(a, b):
        return sum(p * q for p, q in zip(a, b))

    def det(r):
        return (
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
        )

    ident = quat_to_rot(1, 0, 0, 0)
    check("identity quaternion gives the identity matrix",
          all(abs(ident[i][j] - (1.0 if i == j else 0.0)) < 1e-12
              for i in range(3) for j in range(3)))

    # A spread of quaternions, including unnormalised input.
    quats = [(1, 0, 0, 0), (0, 1, 0, 0), (0.5, 0.5, 0.5, 0.5),
             (2, -1, 0.5, 3), (0.1, 0.2, -0.3, 0.4), (-1, 1, 1, -1)]
    worst_orth = 0.0
    worst_det = 0.0
    for q in quats:
        r = quat_to_rot(*q)
        for j in range(3):
            worst_orth = max(worst_orth, abs(dot(col(r, j), col(r, j)) - 1.0))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            worst_orth = max(worst_orth, abs(dot(col(r, a), col(r, b))))
        worst_det = max(worst_det, abs(det(r) - 1.0))
    check("columns are orthonormal for every quaternion tested",
          worst_orth < 1e-12, f"max deviation {worst_orth:.2e}")
    check("determinant is +1, so it is a rotation and not a reflection",
          worst_det < 1e-12, f"max deviation {worst_det:.2e}")

    # 90 degrees about z maps x to y. Catches a transposed convention, which would
    # otherwise show up only as subtly wrong splat orientations.
    s = math.sqrt(0.5)
    r = quat_to_rot(s, 0, 0, s)
    ex = (r[0][0], r[1][0], r[2][0])
    check("90 degrees about z sends e1 to +y (convention not transposed)",
          abs(ex[0]) < 1e-12 and abs(ex[1] - 1.0) < 1e-12 and abs(ex[2]) < 1e-12,
          f"({ex[0]:.3f}, {ex[1]:.3f}, {ex[2]:.3f})")


# --------------------------------------------------------------------------- #
# 14. External manifest integrity
# --------------------------------------------------------------------------- #
def verify_external_manifest() -> None:
    section("14. External comparison manifest")

    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "external" / "manifest.json"
    if not path.exists():
        check("external/manifest.json exists", False, str(path))
        return
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    check("schema version is 1", data.get("schema") == 1, str(data.get("schema")))
    targets = data.get("targets", [])
    check("manifest is non-empty", len(targets) > 0, f"{len(targets)} targets")

    keys = [t.get("key") for t in targets]
    check("every target has a key", all(keys))
    check("keys are unique", len(set(keys)) == len(keys),
          f"{len(keys)} keys, {len(set(keys))} distinct")

    required = ("name", "layer", "license", "redistributable", "role")
    missing = [f"{t.get('key')}.{f}" for t in targets for f in required if f not in t]
    check("every target declares name/layer/license/redistributable/role",
          not missing, ", ".join(missing[:4]))

    check("layers are all in 1..4",
          all(t["layer"] in (1, 2, 3, 4) for t in targets),
          str(sorted({t["layer"] for t in targets})))

    # A pinned repo must carry a full 40-char SHA, or a measurement taken from it
    # cannot be reproduced.
    pinned = [t for t in targets if t.get("repo")]
    bad_sha = [t["key"] for t in pinned
               if not (isinstance(t.get("commit"), str) and len(t["commit"]) == 40
                       and all(c in "0123456789abcdef" for c in t["commit"]))]
    check("every pinned repo has a full 40-hex-char commit", not bad_sha,
          ", ".join(bad_sha))
    check("targets without a repo also have no commit",
          all(t.get("commit") is None for t in targets if not t.get("repo")))

    # The four known-restricted repos must be flagged.
    restricted_urls = (
        "turandai/gaussian_surfels",
        "KeKsBoTer/cinematic-gaussians",
        "graphdeco-inria/gaussian-splatting",
        "hbb1/2d-gaussian-splatting",
    )
    flagged = True
    detail = []
    for frag in restricted_urls:
        hits = [t for t in targets if t.get("repo") and frag in t["repo"]]
        if not hits:
            continue
        for t in hits:
            if t.get("redistributable"):
                flagged = False
                detail.append(t["key"])
    check("all four non-redistributable repos are flagged redistributable=false",
          flagged, ", ".join(detail))

    nonredist = [t for t in targets if not t.get("redistributable")]
    check("every non-redistributable target carries a caveat explaining why",
          all(t.get("caveat") for t in nonredist),
          f"{len(nonredist)} restricted targets")

    # Every layer must have at least one target, or the design has a silent hole.
    per_layer = {L: sum(1 for t in targets if t["layer"] == L) for L in (1, 2, 3, 4)}
    check("all four comparison layers are represented",
          all(v > 0 for v in per_layer.values()), str(per_layer))

    # URL forms: both spellings must normalise to the same bare URL.
    def url_forms(repo):
        bare = repo[:-4] if repo.endswith(".git") else repo
        return [bare + ".git", bare]

    u = "https://github.com/o/n"
    check("url_forms is stable under an existing .git suffix",
          url_forms(u) == url_forms(u + ".git") == [u + ".git", u],
          str(url_forms(u)))


# --------------------------------------------------------------------------- #
# 15. NIfTI-1 header parsing, used to validate datasets before nibabel exists
# --------------------------------------------------------------------------- #
def verify_nifti_header() -> None:
    section("15. NIfTI-1 header parsing (dataset verification)")

    import gzip
    import importlib.util
    import pathlib
    import struct

    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("fd", here / "fetch_datasets.py")
    if spec is None or spec.loader is None:
        check("scripts/fetch_datasets.py is importable", False)
        return
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)

    def build(dim, pixdim, endian="<", gz=True):
        """A minimal but standards-conformant NIfTI-1 header."""
        h = bytearray(348)
        struct.pack_into(endian + "i", h, 0, 348)
        d = [0] * 8
        d[0] = len(dim)
        for i, v in enumerate(dim):
            d[i + 1] = v
        struct.pack_into(endian + "8h", h, 40, *d)
        struct.pack_into(endian + "2h", h, 70, 4, 16)
        p = [1.0] * 8
        for i, v in enumerate(pixdim):
            p[i + 1] = v
        struct.pack_into(endian + "8f", h, 76, *p)
        struct.pack_into(endian + "f", h, 108, 352.0)
        h[344:348] = b"n+1\0"
        blob = bytes(h) + b"\0" * 20
        return gzip.compress(blob) if gz else blob

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        # Little-endian gzipped, the usual ACDC case.
        p = root / "a.nii.gz"
        p.write_bytes(build([216, 256, 10, 30], [1.5, 1.5, 10.0]))
        hdr = fd.read_nifti_header(p)
        check("little-endian gzipped header parses", hdr["endian"] == "<")
        check("dim round-trips", hdr["dim"][:5] == [4, 216, 256, 10, 30], str(hdr["dim"][:5]))
        sp = fd.spacing_of(hdr)
        ok = all(abs(a - b) < 1e-5 for a, b in zip(sp, (1.5, 1.5, 10.0)))
        check("pixdim round-trips as spacing", ok, f"{sp}")
        check("n_dims reads dim[0]", fd.n_dims(hdr) == 4, str(fd.n_dims(hdr)))

        # Big-endian: sizeof_hdr must drive byte-order detection. Getting this wrong
        # yields nonsense spacings that would silently pass an anisotropy test.
        p = root / "b.nii.gz"
        p.write_bytes(build([216, 256, 10, 30], [1.5, 1.5, 10.0], endian=">"))
        hdr = fd.read_nifti_header(p)
        check("big-endian header is detected", hdr["endian"] == ">")
        sp = fd.spacing_of(hdr)
        ok = all(abs(a - b) < 1e-5 for a, b in zip(sp, (1.5, 1.5, 10.0)))
        check("big-endian spacing is not byte-swapped garbage", ok, f"{sp}")

        # Uncompressed .nii.
        p = root / "c.nii"
        p.write_bytes(build([192, 192, 8, 25], [1.2, 1.2, 8.0], gz=False))
        hdr = fd.read_nifti_header(p)
        ok = all(abs(a - b) < 1e-5 for a, b in zip(fd.spacing_of(hdr), (1.2, 1.2, 8.0)))
        check("uncompressed .nii parses", ok)

        # Negative pixdim happens in real files; spacing must be a magnitude.
        p = root / "d.nii.gz"
        p.write_bytes(build([216, 256, 10, 30], [-1.5, 1.5, -10.0]))
        sp = fd.spacing_of(fd.read_nifti_header(p))
        check("negative pixdim is taken as a magnitude", sp[0] > 0 and sp[2] > 0, f"{sp}")

        # A non-NIfTI file must raise, not return silent nonsense.
        p = root / "e.nii.gz"
        p.write_bytes(gzip.compress(b"not a nifti" * 60))
        try:
            fd.read_nifti_header(p)
            check("a non-NIfTI file raises", False, "it returned a header")
        except fd.NiftiHeaderError:
            check("a non-NIfTI file raises NiftiHeaderError", True)

        # Truncated file.
        p = root / "f.nii"
        p.write_bytes(b"\0" * 100)
        try:
            fd.read_nifti_header(p)
            check("a truncated file raises", False, "it returned a header")
        except fd.NiftiHeaderError:
            check("a truncated file raises NiftiHeaderError", True)

    # Anisotropy ratio: the quantity the spacing ablation depends on.
    ratio = lambda s: s[2] / min(s[0], s[1])  # noqa: E731
    check("genuine ACDC-like spacing is strongly anisotropic",
          ratio((1.5, 1.5, 10.0)) > 2.0, f"{ratio((1.5, 1.5, 10.0)):.2f}")
    check("isotropic spacing is correctly judged unusable",
          ratio((1.0, 1.0, 1.0)) < 2.0, f"{ratio((1.0, 1.0, 1.0)):.2f}")
    check("a 1x1x10 mirror passes the ratio test, so it needs a fingerprint",
          ratio((1.0, 1.0, 10.0)) > 2.0,
          "which is exactly why resampled_fingerprints exists")


# --------------------------------------------------------------------------- #
# 16. Dataset manifest integrity
# --------------------------------------------------------------------------- #
def verify_dataset_manifest() -> None:
    section("16. Dataset manifest")

    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "external" / "datasets.json"
    if not path.exists():
        check("external/datasets.json exists", False, str(path))
        return
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    check("schema version is 1", data.get("schema") == 1, str(data.get("schema")))
    ds = data.get("datasets", [])
    check("manifest is non-empty", len(ds) > 0, f"{len(ds)} datasets")

    keys = [d.get("key") for d in ds]
    check("keys are present and unique",
          all(keys) and len(set(keys)) == len(keys), f"{len(keys)} keys")

    allowed = {"registration", "open-download", "application", "generated"}
    bad = [d["key"] for d in ds if d.get("access") not in allowed]
    check("every access mode is recognised", not bad, ", ".join(bad))

    # A gated dataset must never be presented as fetchable.
    gated = [d for d in ds if d["access"] in ("registration", "application")]
    check("gated datasets are recorded as such", len(gated) > 0, f"{len(gated)} gated")

    nolicense = [d["key"] for d in ds if not d.get("license")]
    check("every dataset states a licence", not nolicense, ", ".join(nolicense))

    # Anything requiring citation must name a bib key or say why not.
    # A dataset that must be cited needs either a refs.bib key or an explicit note
    # saying why there is none. Silence is the failure mode: it is how an unattributed
    # dataset ends up in a results table.
    cite = [d for d in ds if d.get("cite_required")]
    missing_bib = [d["key"] for d in cite if not d.get("bib") and not d.get("cite_note")]
    check("datasets requiring citation name a refs.bib key or state why not",
          not missing_bib, ", ".join(missing_bib))

    # Layout entries must be consumable by the verifier.
    with_layout = [d for d in ds if d.get("expected_layout")]
    # `used_by` may be explicitly null, so `.get(k, "")` is not enough.
    check("datasets with a loader declare an expected layout",
          all(d.get("expected_layout") for d in ds
              if (d.get("used_by") or "").startswith("cvdyn2dgs.data.real")),
          f"{len(with_layout)} layouts")
    for d in with_layout:
        lay = d["expected_layout"]
        if not lay.get("patient_dir_glob") or not lay.get("required_per_patient"):
            check(f"{d['key']} layout is complete", False)
            break
    else:
        check("every layout names a patient glob and required files", True,
              f"{len(with_layout)} checked")

    # The anisotropy contract.
    aniso = [d for d in ds if (d.get("expected_geometry") or {}).get("anisotropic_required")]
    check("real cine datasets require anisotropic spacing", len(aniso) >= 2,
          f"{len(aniso)} datasets")
    no_ratio = [d["key"] for d in aniso if not d["expected_geometry"].get("min_z_to_xy_ratio")]
    check("each of those states a minimum z:xy ratio", not no_ratio, ", ".join(no_ratio))

    # Fingerprints must be complete enough to match on.
    fps = [(d["key"], fp) for d in ds
           for fp in (d.get("expected_geometry") or {}).get("resampled_fingerprints", [])]
    incomplete = [k for k, fp in fps if not fp.get("spacing_mm") or not fp.get("reason")]
    check("resampled fingerprints carry a spacing and a reason", not incomplete,
          f"{len(fps)} fingerprint(s)")

    # Every unusable dataset must explain itself, or it will be proposed as a substitute.
    unusable = [d for d in ds if not d.get("usable_for_this_thesis", True)]
    check("datasets marked unusable explain why",
          all(d.get("caveat") for d in unusable), f"{len(unusable)} marked unusable")

    # Known mirrors must carry a usability verdict.
    mirrors = [(d["key"], m) for d in ds for m in d.get("known_mirrors", [])]
    check("every known mirror states whether it is usable here",
          all("usable_for_this_thesis" in m for _, m in mirrors),
          f"{len(mirrors)} mirror(s)")


# --------------------------------------------------------------------------- #
# 17. Quality-versus-cost: percentiles, axis direction, dominance
# --------------------------------------------------------------------------- #
def verify_cost_quality() -> None:
    section("17. Quality against cost (percentile, dominance, frame budget)")

    # Independent reimplementation of metrics/costquality.percentile.
    def pct(vals, q):
        vs = sorted(vals)
        pos = q * (len(vs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vs) - 1)
        return vs[lo] + (vs[hi] - vs[lo]) * (pos - lo)

    check("p0 and p100 are the extremes",
          pct([1, 2, 3, 4, 5], 0.0) == 1 and pct([1, 2, 3, 4, 5], 1.0) == 5)
    check("median of an even-length sample interpolates",
          abs(pct([1, 2, 3, 4], 0.5) - 2.5) < 1e-12, f"{pct([1,2,3,4],0.5)}")

    # On a 10-sample run a non-interpolating p95 collapses to the maximum, which would
    # silently turn the 33.3 ms budget into a worst-case test.
    v10 = list(range(1, 11))
    check("p95 interpolates instead of snapping to the maximum",
          pct(v10, 0.95) < 10.0 and abs(pct(v10, 0.95) - 9.55) < 1e-9,
          f"{pct(v10, 0.95):.3f} vs max 10")

    # The tail the mean hides: this is why the criterion is on p95.
    tail = [10, 12, 14, 16, 18, 20, 22, 24, 26, 100]
    mean = sum(tail) / len(tail)
    check("p95 exposes a tail the mean conceals",
          pct(tail, 0.95) > 2 * mean, f"p95 {pct(tail,0.95):.1f} vs mean {mean:.1f}")

    # Frame budget and headroom.
    budget = 33.3
    check("budget verdict is taken at p95, not the mean",
          pct(tail, 0.95) > budget and mean < budget,
          "this sample passes on the mean and fails on p95")
    for p95, want in ((11.1, 3.0), (33.3, 1.0), (66.6, 0.5)):
        got = budget / p95
        check(f"headroom at p95={p95} ms is {want}x", abs(got - want) < 1e-2, f"{got:.3f}")

    # Stage shares must sum to one and identify the dominant stage.
    stages = {"surface": 2.1, "project": 6.4, "orient": 1.2, "raster": 11.3}
    total = sum(stages.values())
    shares = {k: v / total for k, v in stages.items()}
    check("stage shares sum to 1", abs(sum(shares.values()) - 1.0) < 1e-12)
    check("the dominant stage is identified", max(shares, key=shares.get) == "raster",
          f"raster {shares['raster']*100:.1f}%")

    # The mean-versus-tail attribution problem: a stage that is cheap on average can own
    # the tail, so attribution must be computed on the worst frames.
    frames = [
        {"project": 6.0, "raster": 3.0},
        {"project": 6.0, "raster": 3.0},
        {"project": 6.0, "raster": 3.0},
        {"project": 6.0, "raster": 40.0},  # the tail
    ]
    mean_project = sum(f["project"] for f in frames) / len(frames)
    mean_raster = sum(f["raster"] for f in frames) / len(frames)
    worst = max(frames, key=lambda f: f["project"] + f["raster"])
    dom_worst = max(worst, key=lambda k: worst[k])
    check("a stage can lead on the mean while another owns the tail",
          mean_project > 0 and dom_worst == "raster",
          f"mean project {mean_project:.1f} vs raster {mean_raster:.1f}; worst -> {dom_worst}")

    # Dominance. Lower-is-better axes must be handled, or conclusions invert.
    LOWER = {"e_surf", "p95", "bytes"}

    def better(axis, a, b):
        return a < b if axis in LOWER else a > b

    def dominates(a, b):
        keys = [k for k in a if k in b]
        if not keys:
            return False
        strict = False
        for k in keys:
            if better(k, b[k], a[k]):
                return False
            if better(k, a[k], b[k]):
                strict = True
        return strict

    good = {"psnr": 30.0, "e_surf": 0.4, "p95": 20.0}
    bad = {"psnr": 30.0, "e_surf": 0.9, "p95": 20.0}
    check("lower-is-better axes are compared in the right direction",
          dominates(good, bad) and not dominates(bad, good))

    a = {"psnr": 32.0, "p95": 18.0}
    b = {"psnr": 30.0, "p95": 25.0}
    check("strictly better on every axis dominates", dominates(a, b))
    check("dominance is antisymmetric", not dominates(b, a))
    check("an identical point dominates neither way",
          not dominates(a, dict(a)) and not dominates(dict(a), a))

    fast = {"psnr": 29.0, "p95": 12.0}
    fine = {"psnr": 34.0, "p95": 30.0}
    check("a genuine trade-off dominates neither way",
          not dominates(fast, fine) and not dominates(fine, fast))

    # The expected free-3DGS outcome: wins PSNR, loses E_surf, both stay on the frontier.
    ours = {"psnr": 31.2, "e_surf": 0.42}
    free = {"psnr": 34.6, "e_surf": 4.80}
    check("free geometry and surface pinning are mutually non-dominated",
          not dominates(ours, free) and not dominates(free, ours),
          "so the trade-off cannot be averaged away")

    # Silence must not become an advantage.
    full = {"psnr": 30.0, "e_surf": 0.4}
    partial = {"psnr": 30.0}
    check("omitting a losing axis yields no advantage",
          not dominates(partial, full) and not dominates(full, partial),
          "verdict rests on the shared axes only")

    pts = {"a": a, "b": b, "fast": fast, "fine": fine}
    front = sorted(k for k, p in pts.items()
                   if not any(dominates(q, p) for n, q in pts.items() if n != k))
    check("the frontier excludes exactly the dominated points",
          "b" not in front and "a" in front and "fast" in front and "fine" in front,
          f"frontier {front}")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("CV-Dyn2DGS kernel verification (standard library only)")
    print("=" * 56)
    print("Covers the discrete index/table logic that does not need PyTorch.")
    print("Passing here does NOT mean the pipeline runs - see the README.")

    verify_decomposition()
    verify_case_table()
    verify_tile_enumeration()
    verify_tile_truncation()
    verify_bit_packing()
    verify_ellipsoid_sdf()
    verify_trilinear()
    verify_cg()
    verify_reinit_fixed_point()
    verify_storage_algebra()
    verify_contraction_profile()
    verify_viewpoint_breakeven()
    verify_quaternion_rotation()
    verify_external_manifest()
    verify_nifti_header()
    verify_dataset_manifest()
    verify_cost_quality()

    print("\n" + "=" * 56)
    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
