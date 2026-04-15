from __future__ import annotations

import math

import numpy as np

from .elements import Molecule
from .isosurface import IsosurfaceMesh


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class ImageRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        bg_color: tuple[int, int, int] = (0, 0, 0),
    ):
        self.width = width
        self.height = height
        self.fov = 1.5
        self.light_dir = np.array([0.4, 0.7, -0.6])
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.ambient = 0.2
        self.diffuse_strength = 0.6
        self.specular_strength = 0.4
        self.shininess = 32.0
        # View direction (camera looks along +Z, so view dir is -Z)
        self.view_dir = np.array([0.0, 0.0, -1.0])
        # Half vector for Blinn-Phong
        self.half_vec = self.light_dir + self.view_dir
        self.half_vec /= np.linalg.norm(self.half_vec)
        self.atom_scale = 0.35
        self.bond_radius = 0.08
        self.bg_color = np.array(bg_color, dtype=np.uint8)
        self.clear()

    def clear(self):
        self.pixels = np.tile(self.bg_color, (self.height, self.width, 1))
        self.z_buf = np.full((self.height, self.width), float("inf"))

    def _project(self, point: np.ndarray) -> tuple[float, float, float]:
        x, y, z = point
        if z <= 0.1:
            return (float("nan"), float("nan"), z)
        scale = min(self.width, self.height) / 2
        px = x * self.fov / z
        py = y * self.fov / z
        sx = self.width / 2 + px * scale
        sy = self.height / 2 - py * scale
        return (sx, sy, z)

    def render_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: tuple[int, int, int],
    ):
        sx, sy, sz = self._project(center)
        if math.isnan(sx):
            return

        scale = min(self.width, self.height) / 2
        pr = radius * self.fov / sz * scale
        if pr < 0.5:
            return

        x_min = max(0, int(sx - pr - 1))
        x_max = min(self.width - 1, int(sx + pr + 1))
        y_min = max(0, int(sy - pr - 1))
        y_max = min(self.height - 1, int(sy + pr + 1))

        if x_min > x_max or y_min > y_max:
            return

        ys = np.arange(y_min, y_max + 1)
        xs = np.arange(x_min, x_max + 1)
        px_grid, py_grid = np.meshgrid(xs, ys)

        dx = (px_grid - sx) / pr
        dy = (py_grid - sy) / pr
        dist_sq = dx * dx + dy * dy

        mask = dist_sq <= 1.0

        dz = np.sqrt(np.maximum(0.0, 1.0 - dist_sq))
        norm_len = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-10
        nx = dx / norm_len
        ny = -dy / norm_len
        nz = dz / norm_len

        n_dot_l = np.maximum(
            0.0,
            nx * self.light_dir[0] + ny * self.light_dir[1] + nz * self.light_dir[2],
        )
        n_dot_h = np.maximum(
            0.0,
            nx * self.half_vec[0] + ny * self.half_vec[1] + nz * self.half_vec[2],
        )
        specular = np.power(n_dot_h, self.shininess) * self.specular_strength
        diffuse = n_dot_l * self.diffuse_strength
        intensity = np.minimum(1.0, self.ambient + diffuse)

        point_z = sz - radius * dz

        z_slice = self.z_buf[y_min : y_max + 1, x_min : x_max + 1]
        valid = mask & (point_z < z_slice)

        z_slice[valid] = point_z[valid]

        color_arr = np.array(color, dtype=np.float64)
        shaded = np.minimum(
            255, color_arr[None, None, :] * intensity[:, :, None] + 255 * specular[:, :, None]
        ).astype(np.uint8)
        for c in range(3):
            channel = self.pixels[y_min : y_max + 1, x_min : x_max + 1, c]
            channel[valid] = shaded[:, :, c][valid]

    def render_bond(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        color1: tuple[int, int, int],
        color2: tuple[int, int, int],
    ):
        sx1, sy1, sz1 = self._project(p1)
        sx2, sy2, sz2 = self._project(p2)
        if math.isnan(sx1) or math.isnan(sx2):
            return

        scale = min(self.width, self.height) / 2
        mid_z = (sz1 + sz2) / 2
        pr = self.bond_radius * self.fov / mid_z * scale

        dx = sx2 - sx1
        dy = sy2 - sy1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1.0:
            return

        nx, ny = -dy / length, dx / length  # perpendicular
        half_w = max(1.0, pr)
        steps = int(length) + 1

        # Vectorize: all (step, offset) combinations at once
        hw = int(half_w + 1)
        ts = np.linspace(0, 1, steps + 1)
        offsets = np.arange(-hw, hw + 1, dtype=np.float64)
        d_norm = offsets / half_w
        off_mask = np.abs(d_norm) <= 1.0
        offsets = offsets[off_mask]
        d_norm = d_norm[off_mask]

        # Outer product: (n_steps, n_offsets)
        cxs = sx1 + dx * ts
        cys = sy1 + dy * ts
        czs = sz1 + (sz2 - sz1) * ts

        # All pixel positions: (n_steps, n_offsets)
        all_px = np.round(cxs[:, None] + nx * offsets[None, :]).astype(int)
        all_py = np.round(cys[:, None] + ny * offsets[None, :]).astype(int)

        # Cylinder shading (only varies with offset, not step)
        cyl_nz = np.sqrt(1.0 - d_norm * d_norm)
        cyl_nx_v = nx * d_norm
        cyl_ny_v = -ny * d_norm
        norm_len = np.sqrt(cyl_nx_v**2 + cyl_ny_v**2 + cyl_nz**2) + 1e-10
        cyl_nx_v /= norm_len
        cyl_ny_v /= norm_len
        cyl_nz /= norm_len
        diffuse = np.maximum(
            0.0,
            cyl_nx_v * self.light_dir[0]
            + cyl_ny_v * self.light_dir[1]
            + cyl_nz * self.light_dir[2],
        )
        intensity = np.minimum(1.0, self.ambient + (1.0 - self.ambient) * diffuse)
        # Broadcast to (n_steps, n_offsets)
        pz = czs[:, None] - self.bond_radius * cyl_nz[None, :]
        intensity_2d = np.broadcast_to(intensity[None, :], pz.shape)

        # Flatten everything
        flat_px = all_px.ravel()
        flat_py = all_py.ravel()
        flat_pz = pz.ravel()
        flat_int = intensity_2d.ravel()

        # Color: first half = color1, second half = color2
        n_steps = len(ts)
        n_off = len(offsets)
        step_idx = np.repeat(np.arange(n_steps), n_off)
        half = n_steps // 2
        is_first_half = step_idx <= half

        # Bounds check
        valid = (flat_px >= 0) & (flat_px < self.width) & (flat_py >= 0) & (flat_py < self.height)
        flat_px, flat_py, flat_pz, flat_int, is_first_half = (
            flat_px[valid],
            flat_py[valid],
            flat_pz[valid],
            flat_int[valid],
            is_first_half[valid],
        )

        if len(flat_px) == 0:
            return

        # Z-buffer test
        z_current = self.z_buf[flat_py, flat_px]
        z_pass = flat_pz < z_current
        flat_px = flat_px[z_pass]
        flat_py = flat_py[z_pass]
        flat_pz = flat_pz[z_pass]
        flat_int = flat_int[z_pass]
        is_first_half = is_first_half[z_pass]

        if len(flat_px) == 0:
            return

        self.z_buf[flat_py, flat_px] = flat_pz

        c1 = np.array(color1, dtype=np.float64)
        c2 = np.array(color2, dtype=np.float64)
        colors = np.where(is_first_half[:, None], c1[None, :], c2[None, :])
        shaded = np.minimum(255, colors * flat_int[:, None]).astype(np.uint8)
        self.pixels[flat_py, flat_px] = shaded

    def render_isosurface(
        self,
        mesh: IsosurfaceMesh,
        rot: np.ndarray,
        camera_distance: float,
        centroid: np.ndarray,
        pan: tuple[float, float] = (0.0, 0.0),
    ):
        if len(mesh.faces) == 0:
            return

        # Transform all vertices and normals at once
        transformed = (rot @ (mesh.vertices - centroid).T).T
        transformed[:, 0] += pan[0]
        transformed[:, 1] += pan[1]
        transformed[:, 2] += camera_distance
        rot_normals = (rot @ mesh.normals.T).T

        # Project all vertices (vectorized)
        scale = min(self.width, self.height) / 2
        valid_z = transformed[:, 2] > 0.1
        z = transformed[:, 2]
        safe_z = np.where(valid_z, z, 1.0)
        projected = np.column_stack(
            [
                self.width / 2 + transformed[:, 0] * self.fov / safe_z * scale,
                self.height / 2 - transformed[:, 1] * self.fov / safe_z * scale,
                z,
            ]
        )

        faces = mesh.faces
        # Vectorized: check all vertices visible
        face_valid = valid_z[faces[:, 0]] & valid_z[faces[:, 1]] & valid_z[faces[:, 2]]
        faces = faces[face_valid]
        if len(faces) == 0:
            return

        # Gather projected vertices per face
        s0 = projected[faces[:, 0]]  # (M, 3)
        s1 = projected[faces[:, 1]]
        s2 = projected[faces[:, 2]]

        # Vectorized backface cull
        cross = (s1[:, 0] - s0[:, 0]) * (s2[:, 1] - s0[:, 1]) - (s1[:, 1] - s0[:, 1]) * (
            s2[:, 0] - s0[:, 0]
        )
        front = cross > 0
        faces = faces[front]
        s0, s1, s2 = s0[front], s1[front], s2[front]
        if len(faces) == 0:
            return

        # Gather normals per face vertex
        n0 = rot_normals[faces[:, 0]]
        n1 = rot_normals[faces[:, 1]]
        n2 = rot_normals[faces[:, 2]]

        # Vectorized bounding boxes
        x_mins = np.maximum(
            0, np.floor(np.minimum(s0[:, 0], np.minimum(s1[:, 0], s2[:, 0]))).astype(int)
        )
        x_maxs = np.minimum(
            self.width - 1,
            np.ceil(np.maximum(s0[:, 0], np.maximum(s1[:, 0], s2[:, 0]))).astype(int),
        )
        y_mins = np.maximum(
            0, np.floor(np.minimum(s0[:, 1], np.minimum(s1[:, 1], s2[:, 1]))).astype(int)
        )
        y_maxs = np.minimum(
            self.height - 1,
            np.ceil(np.maximum(s0[:, 1], np.maximum(s1[:, 1], s2[:, 1]))).astype(int),
        )

        # Filter degenerate bboxes
        bbox_valid = (x_mins <= x_maxs) & (y_mins <= y_maxs)
        if not bbox_valid.any():
            return
        idx = np.where(bbox_valid)[0]
        s0, s1, s2 = s0[idx], s1[idx], s2[idx]
        n0, n1, n2 = n0[idx], n1[idx], n2[idx]
        x_mins, x_maxs = x_mins[idx], x_maxs[idx]
        y_mins, y_maxs = y_mins[idx], y_maxs[idx]
        n_faces = len(idx)

        # Barycentric denominators (per face)
        denom = (s1[:, 1] - s2[:, 1]) * (s0[:, 0] - s2[:, 0]) + (s2[:, 0] - s1[:, 0]) * (
            s0[:, 1] - s2[:, 1]
        )
        denom_valid = np.abs(denom) > 1e-10
        if not denom_valid.any():
            return
        filt = np.where(denom_valid)[0]
        s0, s1, s2 = s0[filt], s1[filt], s2[filt]
        n0, n1, n2 = n0[filt], n1[filt], n2[filt]
        x_mins, x_maxs = x_mins[filt], x_maxs[filt]
        y_mins, y_maxs = y_mins[filt], y_maxs[filt]
        denom = denom[filt]
        n_faces = len(filt)

        # Generate all candidate pixels (light loop, no per-pixel math)
        widths = x_maxs - x_mins + 1
        heights = y_maxs - y_mins + 1
        n_pixels = widths * heights
        total_pixels = int(n_pixels.sum())
        if total_pixels == 0:
            return

        # Generate all candidate pixel coords (fully vectorized, no Python loop)
        face_idx = np.repeat(np.arange(n_faces, dtype=np.int32), n_pixels)
        cumsum = np.empty(n_faces + 1, dtype=np.int64)
        cumsum[0] = 0
        np.cumsum(n_pixels, out=cumsum[1:])
        local_idx = np.arange(total_pixels, dtype=np.int32) - np.repeat(
            cumsum[:-1].astype(np.int32), n_pixels
        )
        w_rep = np.repeat(widths.astype(np.int32), n_pixels)
        all_x = np.repeat(x_mins.astype(np.int32), n_pixels) + (local_idx % w_rep)
        all_y = np.repeat(y_mins.astype(np.int32), n_pixels) + (local_idx // w_rep)

        # === Everything below is one big vectorized numpy operation ===
        # Pre-compute per-face edge coefficients (avoids repeated indexing)
        fi = face_idx
        inv_d = (1.0 / denom).astype(np.float32)
        e0a = ((s1[:, 1] - s2[:, 1]) * inv_d).astype(np.float32)
        e0b = ((s2[:, 0] - s1[:, 0]) * inv_d).astype(np.float32)
        e1a = ((s2[:, 1] - s0[:, 1]) * inv_d).astype(np.float32)
        e1b = ((s0[:, 0] - s2[:, 0]) * inv_d).astype(np.float32)
        s2x = s2[:, 0].astype(np.float32)
        s2y = s2[:, 1].astype(np.float32)

        px = all_x.astype(np.float32)
        py = all_y.astype(np.float32)
        dpx = px - s2x[fi]
        dpy = py - s2y[fi]

        w0 = e0a[fi] * dpx + e0b[fi] * dpy
        w1 = e1a[fi] * dpx + e1b[fi] * dpy
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            return

        # Filter to inside pixels and do z-test immediately
        px_i = all_x[inside]
        py_i = all_y[inside]
        fi_i = fi[inside]
        w0_i = w0[inside]
        w1_i = w1[inside]
        w2_i = w2[inside]

        tri_z = (
            w0_i * s0[fi_i, 2].astype(np.float32)
            + w1_i * s1[fi_i, 2].astype(np.float32)
            + w2_i * s2[fi_i, 2].astype(np.float32)
        )

        z_pass = tri_z < self.z_buf[py_i, px_i]
        if not z_pass.any():
            return

        px_f = px_i[z_pass]
        py_f = py_i[z_pass]
        fi_f = fi_i[z_pass]
        w0_f = w0_i[z_pass]
        w1_f = w1_i[z_pass]
        w2_f = w2_i[z_pass]
        tri_z_f = tri_z[z_pass]

        # Interpolate normals and shade (only for z-passing pixels)
        inx = w0_f * n0[fi_f, 0] + w1_f * n1[fi_f, 0] + w2_f * n2[fi_f, 0]
        iny = w0_f * n0[fi_f, 1] + w1_f * n1[fi_f, 1] + w2_f * n2[fi_f, 1]
        inz = w0_f * n0[fi_f, 2] + w1_f * n1[fi_f, 2] + w2_f * n2[fi_f, 2]
        in_len = np.sqrt(inx * inx + iny * iny + inz * inz) + 1e-10
        inx /= in_len
        iny /= in_len
        inz /= in_len

        # Flip normals facing away from camera
        flip = inz > 0
        inx = np.where(flip, -inx, inx)
        iny = np.where(flip, -iny, iny)
        inz = np.where(flip, -inz, inz)

        # Blinn-Phong lighting
        n_dot_l = np.maximum(
            0.0, inx * self.light_dir[0] + iny * self.light_dir[1] + inz * self.light_dir[2]
        )
        n_dot_h = np.maximum(
            0.0, inx * self.half_vec[0] + iny * self.half_vec[1] + inz * self.half_vec[2]
        )
        specular = np.power(n_dot_h, self.shininess) * self.specular_strength
        intensity = np.minimum(1.0, self.ambient + n_dot_l * self.diffuse_strength)

        # Sort back-to-front so closest fragments are written last and win
        order = np.argsort(-tri_z_f)
        px_f = px_f[order]
        py_f = py_f[order]
        tri_z_f = tri_z_f[order]

        # Write z-buffer and pixels
        self.z_buf[py_f, px_f] = tri_z_f
        color = np.array(mesh.color, dtype=np.float32)
        shaded = np.minimum(
            255, color[None, :] * intensity[:, None] + 255 * specular[:, None]
        ).astype(np.uint8)
        self.pixels[py_f, px_f] = shaded[order]

    @staticmethod
    def _highlight_color() -> tuple[int, int, int]:
        return (255, 255, 50)

    def render_molecule(
        self,
        molecule: Molecule,
        rot: np.ndarray,
        camera_distance: float,
        isosurfaces: list[IsosurfaceMesh] | None = None,
        pan: tuple[float, float] = (0.0, 0.0),
        highlighted_atoms: set[int] | None = None,
    ):
        self.clear()
        if not molecule.atoms:
            return

        centroid = molecule.center()
        hl = highlighted_atoms or set()
        has_hl = len(hl) > 0

        # Render isosurfaces first (they go behind atoms/bonds via z-buffer)
        if isosurfaces:
            for mesh in isosurfaces:
                self.render_isosurface(mesh, rot, camera_distance, centroid, pan)

        transformed = []
        for atom in molecule.atoms:
            pos = rot @ (atom.position - centroid)
            pos[0] += pan[0]
            pos[1] += pan[1]
            pos[2] += camera_distance
            transformed.append(pos)

        for i, j in molecule.bonds:
            c1 = molecule.atoms[i].element.cpk_color
            c2 = molecule.atoms[j].element.cpk_color
            if has_hl and i in hl and j in hl:
                c1 = self._highlight_color()
                c2 = self._highlight_color()
            self.render_bond(transformed[i], transformed[j], c1, c2)

        atom_order = sorted(
            range(len(molecule.atoms)),
            key=lambda idx: -transformed[idx][2],
        )
        for i in atom_order:
            atom = molecule.atoms[i]
            radius = atom.element.covalent_radius * self.atom_scale
            color = self._highlight_color() if has_hl and i in hl else atom.element.cpk_color
            self.render_sphere(transformed[i], radius, color)


def render_scene(
    width: int,
    height: int,
    molecule: Molecule,
    rot: np.ndarray,
    camera_distance: float,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    isosurfaces: list[IsosurfaceMesh] | None = None,
    ssaa: int = 2,
    pan: tuple[float, float] = (0.0, 0.0),
    highlighted_atoms: set[int] | None = None,
) -> np.ndarray:
    """Render with supersampling anti-aliasing. Returns (height, width, 3) uint8."""
    r = ImageRenderer(width * ssaa, height * ssaa, bg_color=bg_color)
    r.render_molecule(
        molecule,
        rot,
        camera_distance,
        isosurfaces=isosurfaces,
        pan=pan,
        highlighted_atoms=highlighted_atoms,
    )
    if ssaa == 1:
        return r.pixels
    # Box-filter downsample
    downsampled = r.pixels.reshape(height, ssaa, width, ssaa, 3).mean(axis=(1, 3)).astype(np.uint8)
    return downsampled
