"""Extract a triangle mesh from the Chan-Vese level set.

The mesh-only viewer is a **mandatory strong baseline**, not a straw man
(proposal §4.3, §5.8): if surfels do not beat a dense, smoothly shaded mesh on
silhouette / normal / depth / appearance at a comparable primitive budget, then
"a mesh is the better tool for this job" is the correct conclusion.  To keep the
comparison fair this extractor

* uses the *same* :math:`\\Gamma_t = \\{\\phi_t = 0\\}` as the surfel pipeline,
* interpolates crossings linearly along edges (sub-voxel accurate), and
* assigns per-vertex normals from the analytic level-set gradient
  :math:`\\nabla_h\\phi / \\|\\nabla_h\\phi\\|` rather than from face averaging,
  which gives the mesh the best shading quality available from this surface.

Marching **tetrahedra** is used instead of marching cubes: each cube is split
into 6 tetrahedra around the ``0-6`` diagonal, and a tetrahedron has only three
topological cases (0, 1 or 2 vertices inside).  That replaces the 256-entry
marching-cubes table - and its ambiguous-face cases - with a 16-entry table that
is easy to audit, at the cost of more triangles.

Triangle winding is fixed *geometrically*: every face normal is compared against
the level-set gradient at the face centroid and flipped if they disagree.  This
removes any dependence on getting the table's orientation right.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.grid import Grid, trilinear_sample_vector
from .operators import gradient_central

__all__ = ["TriangleMesh", "marching_tetrahedra"]


# Cube corner offsets (see module docstring for the ordering).
_CORNERS = (
    (0, 0, 0),  # 0
    (1, 0, 0),  # 1
    (1, 1, 0),  # 2
    (0, 1, 0),  # 3
    (0, 0, 1),  # 4
    (1, 0, 1),  # 5
    (1, 1, 1),  # 6
    (0, 1, 1),  # 7
)

# Six tetrahedra sharing the 0-6 diagonal; the other two vertices walk the
# hexagonal ring 1-2-3-7-4-5.
_TETS = (
    (0, 6, 1, 2),
    (0, 6, 2, 3),
    (0, 6, 3, 7),
    (0, 6, 7, 4),
    (0, 6, 4, 5),
    (0, 6, 5, 1),
)

# Local tetrahedron edges, indexed 0..5.
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# case -> up to two triangles, each a triple of edge indices; -1 pads.
# Bit i of `case` is set when local vertex i is INSIDE (phi > 0).
_TABLE = (
    (-1, -1, -1, -1, -1, -1),  # 0000
    (0, 1, 2, -1, -1, -1),  # 0001  v0
    (0, 3, 4, -1, -1, -1),  # 0010  v1
    (1, 2, 4, 1, 4, 3),  # 0011  v0,v1
    (1, 3, 5, -1, -1, -1),  # 0100  v2
    (0, 2, 5, 0, 5, 3),  # 0101  v0,v2
    (0, 1, 5, 0, 5, 4),  # 0110  v1,v2
    (2, 4, 5, -1, -1, -1),  # 0111  v3 outside
    (2, 4, 5, -1, -1, -1),  # 1000  v3
    (0, 1, 5, 0, 5, 4),  # 1001  v0,v3
    (0, 2, 5, 0, 5, 3),  # 1010  v1,v3
    (1, 3, 5, -1, -1, -1),  # 1011  v2 outside
    (1, 2, 4, 1, 4, 3),  # 1100  v2,v3
    (0, 3, 4, -1, -1, -1),  # 1101  v1 outside
    (0, 1, 2, -1, -1, -1),  # 1110  v0 outside
    (-1, -1, -1, -1, -1, -1),  # 1111
)


@dataclass
class TriangleMesh:
    """A triangle mesh in world (mm) coordinates."""

    vertices: Tensor  # (V, 3) float
    faces: Tensor  # (F, 3) long
    normals: Tensor  # (V, 3) float, unit

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def to(self, *args, **kwargs) -> "TriangleMesh":
        return TriangleMesh(
            self.vertices.to(*args, **kwargs),
            self.faces.to(self.faces.device if "device" not in kwargs else kwargs["device"]),
            self.normals.to(*args, **kwargs),
        )

    def storage_floats(self) -> int:
        """Numbers needed to store this mesh: 3 per vertex + 3 indices per face.

        Feeds the mesh row of the storage comparison, Eq. (37).
        """
        return self.n_vertices * 3 + self.n_faces * 3

    def face_areas(self) -> Tensor:
        v = self.vertices[self.faces]  # (F, 3, 3)
        cross = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1)
        return 0.5 * cross.norm(dim=-1)

    def surface_area(self) -> Tensor:
        return self.face_areas().sum()


@torch.no_grad()
def marching_tetrahedra(
    phi: Tensor,
    grid: Grid,
    *,
    dedup: bool = True,
    dedup_decimals: int = 4,
    spacing_aware: bool = True,
) -> TriangleMesh:
    """Extract :math:`\\{\\phi = 0\\}` as a triangle mesh.

    Parameters
    ----------
    phi:
        ``(nx, ny, nz)`` level set, ``phi > 0`` inside.
    dedup:
        Merge coincident vertices.  Needed for an honest vertex count in the
        storage comparison; harmless for rendering.

    Returns
    -------
    :class:`TriangleMesh` with per-vertex normals taken from
    :math:`\\nabla_h\\phi` (Eq. 7.3 / theory Eq. 3.2).
    """
    if phi.dim() != 3:
        raise ValueError(f"expected (nx,ny,nz), got {tuple(phi.shape)}")
    device, dtype = phi.device, phi.dtype
    nx, ny, nz = phi.shape
    if min(nx, ny, nz) < 2:
        raise ValueError("need at least 2 voxels per axis to march")

    # ---- cube corner values and integer coordinates ----------------------
    # phi_c[c] is the value at corner c of every cube -> (nc, 8)
    corner_vals = []
    corner_idx = []
    for ox, oy, oz in _CORNERS:
        corner_vals.append(phi[ox : ox + nx - 1, oy : oy + ny - 1, oz : oz + nz - 1].reshape(-1))
    corner_vals_t = torch.stack(corner_vals, dim=1)  # (nc, 8)

    ci = torch.arange(nx - 1, device=device)
    cj = torch.arange(ny - 1, device=device)
    ck = torch.arange(nz - 1, device=device)
    gi, gj, gk = torch.meshgrid(ci, cj, ck, indexing="ij")
    base_idx = torch.stack((gi.reshape(-1), gj.reshape(-1), gk.reshape(-1)), dim=1)  # (nc,3)
    for ox, oy, oz in _CORNERS:
        off = torch.tensor((ox, oy, oz), device=device, dtype=base_idx.dtype)
        corner_idx.append(base_idx + off)
    corner_idx_t = torch.stack(corner_idx, dim=1)  # (nc, 8, 3) integer voxel indices

    table = torch.tensor(_TABLE, device=device, dtype=torch.long)  # (16, 6)
    edge_lo = torch.tensor([e[0] for e in _TET_EDGES], device=device, dtype=torch.long)
    edge_hi = torch.tensor([e[1] for e in _TET_EDGES], device=device, dtype=torch.long)

    tri_chunks: list[Tensor] = []

    for tet in _TETS:
        tet_t = torch.tensor(tet, device=device, dtype=torch.long)
        vals = corner_vals_t[:, tet_t]  # (nc, 4)
        inside = vals > 0
        case = (
            inside[:, 0].long()
            + 2 * inside[:, 1].long()
            + 4 * inside[:, 2].long()
            + 8 * inside[:, 3].long()
        )
        active = (case > 0) & (case < 15)
        if not bool(active.any()):
            continue

        sel = torch.nonzero(active, as_tuple=False).squeeze(1)
        vals_a = vals[sel]  # (m, 4)
        pos_a = corner_idx_t[sel][:, tet_t, :].to(dtype)  # (m, 4, 3) voxel coords
        case_a = case[sel]  # (m,)

        # Zero crossings on all six local edges (only the crossing ones get used).
        p0 = vals_a[:, edge_lo]  # (m, 6)
        p1 = vals_a[:, edge_hi]
        denom = p0 - p1
        # Guard exact ties; the resulting point is then the edge midpoint.
        t = torch.where(denom.abs() > 1e-12, p0 / denom, torch.full_like(denom, 0.5))
        t = t.clamp(0.0, 1.0)
        a = pos_a[:, edge_lo, :]  # (m, 6, 3)
        b = pos_a[:, edge_hi, :]
        edge_pts = a + t.unsqueeze(-1) * (b - a)  # (m, 6, 3) voxel coords

        tri_edges = table[case_a]  # (m, 6)
        for tri_slot in (0, 1):
            e = tri_edges[:, 3 * tri_slot : 3 * tri_slot + 3]  # (m, 3)
            valid = e[:, 0] >= 0
            if not bool(valid.any()):
                continue
            vsel = torch.nonzero(valid, as_tuple=False).squeeze(1)
            ee = e[vsel]  # (k, 3)
            pts = torch.gather(
                edge_pts[vsel], 1, ee.unsqueeze(-1).expand(-1, -1, 3)
            )  # (k, 3, 3)
            tri_chunks.append(pts)

    if not tri_chunks:
        empty_v = torch.zeros((0, 3), device=device, dtype=dtype)
        return TriangleMesh(empty_v, torch.zeros((0, 3), device=device, dtype=torch.long), empty_v)

    tris_vox = torch.cat(tri_chunks, dim=0)  # (F, 3, 3) in voxel coords
    tris_world = grid.voxel_to_world(tris_vox)

    # ---- per-vertex normals from the level-set gradient ------------------
    grad = gradient_central(phi, grid.spacing, spacing_aware=spacing_aware)  # (3,nx,ny,nz)
    flat_vox = tris_vox.reshape(-1, 3)
    g = trilinear_sample_vector(grad, flat_vox)  # (F*3, 3)
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ---- fix winding against the gradient at the centroid ----------------
    fn = torch.cross(
        tris_world[:, 1] - tris_world[:, 0], tris_world[:, 2] - tris_world[:, 0], dim=-1
    )
    centroid_vox = tris_vox.mean(dim=1)
    g_center = trilinear_sample_vector(grad, centroid_vox)
    flip = (fn * g_center).sum(dim=-1) < 0
    if bool(flip.any()):
        idx = torch.nonzero(flip, as_tuple=False).squeeze(1)
        tris_world[idx] = tris_world[idx][:, [0, 2, 1], :]
        g = g.reshape(-1, 3, 3)
        g[idx] = g[idx][:, [0, 2, 1], :]
        g = g.reshape(-1, 3)

    verts = tris_world.reshape(-1, 3)
    norms = g

    if dedup:
        key = torch.round(verts * (10.0**dedup_decimals)).to(torch.int64)
        uniq, inverse = torch.unique(key, dim=0, return_inverse=True)
        n_uniq = int(uniq.shape[0])
        out_v = torch.zeros((n_uniq, 3), device=device, dtype=dtype)
        out_v.index_copy_(0, inverse, verts)
        out_n = torch.zeros((n_uniq, 3), device=device, dtype=dtype)
        out_n.index_add_(0, inverse, norms)
        out_n = out_n / out_n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        faces = inverse.reshape(-1, 3)
        return TriangleMesh(out_v, faces, out_n)

    faces = torch.arange(verts.shape[0], device=device, dtype=torch.long).reshape(-1, 3)
    return TriangleMesh(verts, faces, norms)
