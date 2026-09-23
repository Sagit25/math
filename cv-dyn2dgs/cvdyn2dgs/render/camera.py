"""Camera models: free perspective viewer and DICOM-style slice-plane cameras.

Two camera families are needed, for two different jobs.

**Perspective** (:meth:`Camera.look_at`, :meth:`Camera.orbit`) is the interactive
viewer: free rotation and zoom, the setting in which the 33.3 ms frame budget of
Eq. (34) has to be met.  Eq. (8.1)-(8.3) are written for exactly this model.

**Orthographic slice planes** (:meth:`Camera.from_slice_plane`) are the
supervision signal.  Theory §11.3 / proposal §10.3 are explicit that cine CMR is
*not* calibrated multi-view RGB: what is actually known is a set of MRI slice
planes in physical coordinates.  Supervision is therefore restricted to masks,
depth, normals and intensity observed on those known planes, and this camera
reproduces them.

Conventions
-----------
* ``x_cam = R @ x_world + t``; camera space is right-handed with ``+x`` right,
  ``+y`` down, ``+z`` forward.  Pixel ``(row, col)`` maps to ``(y, x)``.
* Ray directions are **unit** vectors, so the :math:`\\tau` of Eq. (8.2) is a
  distance in mm, directly comparable with voxel spacing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.grid import Grid

__all__ = ["Camera"]


@dataclass
class Camera:
    """A pinhole or orthographic camera."""

    R: Tensor
    """``(3, 3)`` world-to-camera rotation."""

    t: Tensor
    """``(3,)`` world-to-camera translation."""

    fx: float
    fy: float
    cx: float
    cy: float
    height: int
    width: int
    orthographic: bool = False
    pixel_size_mm: float = 1.0
    """mm per pixel; only used when ``orthographic`` is ``True``."""

    # ------------------------------------------------------------------ basics
    @property
    def device(self) -> torch.device:
        return self.R.device

    @property
    def dtype(self) -> torch.dtype:
        return self.R.dtype

    def to(self, device=None, dtype=None) -> "Camera":
        return Camera(
            R=self.R.to(device=device, dtype=dtype),
            t=self.t.to(device=device, dtype=dtype),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            height=self.height,
            width=self.width,
            orthographic=self.orthographic,
            pixel_size_mm=self.pixel_size_mm,
        )

    def center(self) -> Tensor:
        """Camera centre :math:`o_c` in world mm (Eq. 8.1)."""
        return -(self.R.transpose(0, 1) @ self.t)

    def forward_axis(self) -> Tensor:
        """World-space viewing direction (camera ``+z``)."""
        return self.R[2]

    # ------------------------------------------------------------------- rays
    def rays(self, *, height: int | None = None, width: int | None = None) -> tuple[Tensor, Tensor]:
        """Per-pixel ray origins and unit directions in world coordinates.

        ``height``/``width`` may exceed the nominal sensor size; the tiled
        rasteriser uses that to pad the image up to a whole number of tiles and
        crop afterwards.

        Returns
        -------
        ``(origins, directions)``.  For a perspective camera ``origins`` is
        ``(1, 1, 3)`` (broadcastable, since all rays share :math:`o_c`); for an
        orthographic camera it is ``(H, W, 3)``.  ``directions`` is ``(H, W, 3)``
        and unit-norm.
        """
        h = int(height if height is not None else self.height)
        w = int(width if width is not None else self.width)
        dev, dt = self.device, self.dtype

        ys = torch.arange(h, device=dev, dtype=dt) + 0.5
        xs = torch.arange(w, device=dev, dtype=dt) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        r_t = self.R.transpose(0, 1)  # camera -> world

        if self.orthographic:
            # Parallel rays; the origin slides across the image plane.
            d_cam = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt)
            d_world = (r_t @ d_cam).view(1, 1, 3).expand(h, w, 3).contiguous()
            ox = (gx - self.cx) * self.pixel_size_mm
            oy = (gy - self.cy) * self.pixel_size_mm
            o_cam = torch.stack((ox, oy, torch.zeros_like(ox)), dim=-1)  # (h,w,3)
            o_world = o_cam @ r_t.transpose(0, 1) + self.center().view(1, 1, 3)
            return o_world, d_world

        d_cam = torch.stack(
            ((gx - self.cx) / self.fx, (gy - self.cy) / self.fy, torch.ones_like(gx)), dim=-1
        )  # (h,w,3)
        d_world = d_cam @ r_t.transpose(0, 1)
        d_world = d_world / d_world.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return self.center().view(1, 1, 3), d_world

    # -------------------------------------------------------------- projection
    def project(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Project world points to pixel coordinates.

        Returns
        -------
        ``(uv, depth)`` with ``uv`` of shape ``(..., 2)`` ordered ``(x, y)`` and
        ``depth`` the camera-space ``z`` in mm.  Points behind the camera get a
        non-positive depth and must be culled by the caller (Prop. 8.1).
        """
        cam = points @ self.R.transpose(0, 1) + self.t
        z = cam[..., 2]
        if self.orthographic:
            u = cam[..., 0] / self.pixel_size_mm + self.cx
            v = cam[..., 1] / self.pixel_size_mm + self.cy
        else:
            zz = z.clamp_min(1e-9)
            u = cam[..., 0] / zz * self.fx + self.cx
            v = cam[..., 1] / zz * self.fy + self.cy
        return torch.stack((u, v), dim=-1), z

    def pixel_scale_at(self, depth: Tensor) -> Tensor:
        """Pixels per mm at a given camera-space depth.

        Used to convert surfel radii in mm into conservative screen-space bounds.
        """
        if self.orthographic:
            return torch.full_like(depth, 1.0 / self.pixel_size_mm)
        return max(self.fx, self.fy) / depth.clamp_min(1e-9)

    # ----------------------------------------------------------- constructors
    @classmethod
    def look_at(
        cls,
        eye: Tensor,
        target: Tensor,
        *,
        up: Tensor | None = None,
        fov_deg: float = 40.0,
        height: int = 512,
        width: int = 512,
    ) -> "Camera":
        """Perspective camera at ``eye`` looking at ``target``."""
        dev, dt = eye.device, eye.dtype
        if up is None:
            up = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt)

        z_c = target - eye
        z_c = z_c / z_c.norm().clamp_min(1e-12)
        # Guard the degenerate case where the view direction is parallel to `up`.
        if float(torch.abs((z_c * up / up.norm().clamp_min(1e-12)).sum()).item()) > 0.999:
            up = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dt)
        x_c = torch.cross(z_c, up, dim=0)
        x_c = x_c / x_c.norm().clamp_min(1e-12)
        y_c = torch.cross(z_c, x_c, dim=0)

        rot = torch.stack((x_c, y_c, z_c), dim=0)  # rows
        tr = -(rot @ eye)

        f = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)
        return cls(
            R=rot,
            t=tr,
            fx=f,
            fy=f,
            cx=width * 0.5,
            cy=height * 0.5,
            height=height,
            width=width,
        )

    @classmethod
    def orbit(
        cls,
        center: Tensor,
        radius_mm: float,
        azimuth_deg: float,
        elevation_deg: float,
        *,
        up: Tensor | None = None,
        fov_deg: float = 40.0,
        height: int = 512,
        width: int = 512,
    ) -> "Camera":
        """Perspective camera on a sphere around ``center``.

        This is the viewer interaction of proposal §3.3 ("free camera rotation and
        zoom") and is what the FPS measurement of RQ4 sweeps over.
        """
        dev, dt = center.device, center.dtype
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        offset = torch.tensor(
            [
                radius_mm * math.cos(el) * math.cos(az),
                radius_mm * math.cos(el) * math.sin(az),
                radius_mm * math.sin(el),
            ],
            device=dev,
            dtype=dt,
        )
        return cls.look_at(
            center + offset, center, up=up, fov_deg=fov_deg, height=height, width=width
        )

    @classmethod
    def from_slice_plane(
        cls,
        grid: Grid,
        axis: int = 2,
        slice_index: int | None = None,
        *,
        device=None,
        dtype=torch.float32,
        standoff_mm: float | None = None,
    ) -> "Camera":
        """Orthographic camera looking straight down one grid axis.

        This reproduces the geometry of an MRI slice: the image plane is a grid
        plane, the pixel size equals the in-plane voxel spacing, and the viewing
        direction is the slice normal.  Rendering through it yields exactly the
        projected silhouette / depth / normal / intensity that proposal §7.1
        allows as supervision.

        Parameters
        ----------
        axis:
            ``0``, ``1`` or ``2`` - the axis to look along (``2`` = short-axis
            stack direction for a standard SAX volume).
        slice_index:
            Which slice to centre on; defaults to the middle of the volume.
        """
        dev = device
        dims = [0, 1, 2]
        dims.remove(int(axis))
        a0, a1 = dims  # in-plane axes

        n_u = grid.shape[a0]
        n_v = grid.shape[a1]
        pix = min(grid.spacing[a0], grid.spacing[a1])

        # World-space basis
        e = torch.eye(3, device=dev, dtype=dtype)
        z_c = e[int(axis)]
        x_c = e[a0]
        y_c = torch.cross(z_c, x_c, dim=0)
        rot = torch.stack((x_c, y_c, z_c), dim=0)

        if slice_index is None:
            slice_index = grid.shape[int(axis)] // 2
        centre = grid.center_world(device=dev, dtype=dtype).clone()
        centre[int(axis)] = grid.origin[int(axis)] + slice_index * grid.spacing[int(axis)]

        if standoff_mm is None:
            standoff_mm = 4.0 * max(grid.extent_mm)
        eye = centre - z_c * float(standoff_mm)
        tr = -(rot @ eye)

        # Cover the in-plane extent, with the image centred on `centre`.
        width = int(math.ceil(n_u * grid.spacing[a0] / pix))
        height = int(math.ceil(n_v * grid.spacing[a1] / pix))
        cam_centre = rot @ centre + tr
        return cls(
            R=rot,
            t=tr,
            fx=1.0,
            fy=1.0,
            cx=width * 0.5 - float(cam_centre[0].item()) / pix,
            cy=height * 0.5 - float(cam_centre[1].item()) / pix,
            height=height,
            width=width,
            orthographic=True,
            pixel_size_mm=pix,
        )
