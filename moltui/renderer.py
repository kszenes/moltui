import math

import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

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


class Renderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.char_aspect = 2.0  # terminal chars are ~2x tall as wide
        self.fov = 1.5
        self.light_dir = np.array([0.4, 0.7, -0.6])
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.ambient = 0.4
        self.atom_scale = 0.35  # scale factor for atom display radius
        self.bond_radius = 0.05  # 3D radius of bond cylinders in Angstrom
        self.clear()

    def clear(self):
        self.fg_buf = [[(0, 0, 0)] * self.width for _ in range(self.height)]
        self.bg_buf = [[None] * self.width for _ in range(self.height)]
        self.char_buf = [[" "] * self.width for _ in range(self.height)]
        self.z_buf = [[float("inf")] * self.width for _ in range(self.height)]

    def _project(self, point: np.ndarray) -> tuple[float, float, float]:
        x, y, z = point
        if z <= 0.1:
            return (float("nan"), float("nan"), z)
        px = x * self.fov / z * self.char_aspect
        py = y * self.fov / z
        sx = self.width / 2 + px * self.width / 2
        sy = self.height / 2 - py * self.height / 2
        return (sx, sy, z)

    def _set_pixel(
        self,
        x: int,
        y: int,
        z: float,
        char: str,
        fg: tuple[int, int, int],
        bg: tuple[int, int, int] | None = None,
    ):
        if 0 <= x < self.width and 0 <= y < self.height:
            if z < self.z_buf[y][x]:
                self.z_buf[y][x] = z
                self.char_buf[y][x] = char
                self.fg_buf[y][x] = fg
                self.bg_buf[y][x] = bg

    def _shade_color(
        self, base_color: tuple[int, int, int], intensity: float
    ) -> tuple[int, int, int]:
        r = int(base_color[0] * intensity)
        g = int(base_color[1] * intensity)
        b = int(base_color[2] * intensity)
        return (min(255, r), min(255, g), min(255, b))

    def render_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: tuple[int, int, int],
    ):
        sx, sy, sz = self._project(center)
        if math.isnan(sx):
            return

        projected_radius = radius * self.fov / sz * self.width / 2
        # Account for char aspect in x direction
        pr_x = projected_radius * self.char_aspect
        pr_y = projected_radius

        x_min = max(0, int(sx - pr_x - 1))
        x_max = min(self.width - 1, int(sx + pr_x + 1))
        y_min = max(0, int(sy - pr_y - 1))
        y_max = min(self.height - 1, int(sy + pr_y + 1))

        for py in range(y_min, y_max + 1):
            for px in range(x_min, x_max + 1):
                # Normalized position on sphere
                dx = (px - sx) / pr_x if pr_x > 0 else 0
                dy = (py - sy) / pr_y if pr_y > 0 else 0
                dist_sq = dx * dx + dy * dy
                if dist_sq > 1.0:
                    continue

                dz = math.sqrt(1.0 - dist_sq)
                normal = np.array([dx, -dy, dz])
                normal /= np.linalg.norm(normal) + 1e-10

                point_z = sz - radius * dz

                diffuse = max(0.0, float(np.dot(normal, self.light_dir)))
                intensity = min(1.0, self.ambient + (1.0 - self.ambient) * diffuse)

                shaded = self._shade_color(color, intensity)
                self._set_pixel(px, py, point_z, " ", (0, 0, 0), bg=shaded)

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

        # Projected bond radius at midpoint depth
        mid_z = (sz1 + sz2) / 2
        pr = self.bond_radius * self.fov / mid_z * self.width / 2

        dx = sx2 - sx1
        dy = sy2 - sy1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.5:
            return

        # Unit direction along bond and perpendicular
        ux, uy = dx / length, dy / length
        # Perpendicular (screen-space), adjusted for char aspect
        nx, ny = -uy, ux

        pr_x = pr * self.char_aspect
        pr_y = pr
        half_w = max(1.0, math.sqrt((nx * pr_x) ** 2 + (ny * pr_y) ** 2))

        steps = int(length) + 1
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            cx = sx1 + dx * t
            cy = sy1 + dy * t
            cz = sz1 + (sz2 - sz1) * t
            color = color1 if t < 0.5 else color2

            hw = int(half_w + 1)
            for offset in range(-hw, hw + 1):
                px = int(cx + nx * offset)
                py = int(cy + ny * offset)
                # Normalized perpendicular distance for cylinder shading
                d = offset / half_w if half_w > 0 else 0
                if abs(d) > 1.0:
                    continue
                # Cylinder normal: perpendicular component + outward z
                cyl_nz = math.sqrt(1.0 - d * d)
                cyl_normal = np.array([nx * d, -ny * d, cyl_nz])
                cyl_normal /= np.linalg.norm(cyl_normal) + 1e-10

                diffuse = max(0.0, float(np.dot(cyl_normal, self.light_dir)))
                intensity = min(
                    1.0, self.ambient + (1.0 - self.ambient) * diffuse
                )
                # Push cylinder surface slightly behind atom surface
                pz = cz - self.bond_radius * cyl_nz
                shaded = self._shade_color(color, intensity)
                self._set_pixel(px, py, pz, " ", (0, 0, 0), bg=shaded)

    def render_isosurface(
        self,
        mesh: IsosurfaceMesh,
        rot: np.ndarray,
        camera_distance: float,
        centroid: np.ndarray,
    ):
        if len(mesh.faces) == 0:
            return

        transformed = (rot @ (mesh.vertices - centroid).T).T
        transformed[:, 2] += camera_distance
        rot_normals = (rot @ mesh.normals.T).T

        for face in mesh.faces:
            v0, v1, v2 = transformed[face]
            n0, n1, n2 = rot_normals[face]

            s0x, s0y, s0z = self._project(v0)
            s1x, s1y, s1z = self._project(v1)
            s2x, s2y, s2z = self._project(v2)
            if math.isnan(s0x) or math.isnan(s1x) or math.isnan(s2x):
                continue

            # Backface cull
            cross = (s1x - s0x) * (s2y - s0y) - (s1y - s0y) * (s2x - s0x)
            if cross <= 0:
                continue

            x_min = max(0, int(min(s0x, s1x, s2x)))
            x_max = min(self.width - 1, int(max(s0x, s1x, s2x)) + 1)
            y_min = max(0, int(min(s0y, s1y, s2y)))
            y_max = min(self.height - 1, int(max(s0y, s1y, s2y)) + 1)

            denom = (s1y - s2y) * (s0x - s2x) + (s2x - s1x) * (s0y - s2y)
            if abs(denom) < 1e-10:
                continue
            inv_d = 1.0 / denom

            for py in range(y_min, y_max + 1):
                for px in range(x_min, x_max + 1):
                    dpx = px - s2x
                    dpy = py - s2y
                    w0 = ((s1y - s2y) * dpx + (s2x - s1x) * dpy) * inv_d
                    w1 = ((s2y - s0y) * dpx + (s0x - s2x) * dpy) * inv_d
                    w2 = 1.0 - w0 - w1
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue

                    pz = w0 * s0z + w1 * s1z + w2 * s2z
                    if pz >= self.z_buf[py][px]:
                        continue

                    nx = w0 * n0[0] + w1 * n1[0] + w2 * n2[0]
                    ny = w0 * n0[1] + w1 * n1[1] + w2 * n2[1]
                    nz = w0 * n0[2] + w1 * n1[2] + w2 * n2[2]
                    nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-10
                    nx /= nl; ny /= nl; nz /= nl

                    # Flip normals facing away from camera
                    if nz > 0:
                        nx, ny, nz = -nx, -ny, -nz

                    diffuse = max(0.0, nx * self.light_dir[0] + ny * self.light_dir[1] + nz * self.light_dir[2])
                    intensity = min(1.0, self.ambient + (1.0 - self.ambient) * diffuse)
                    shaded = self._shade_color(mesh.color, intensity)
                    self._set_pixel(px, py, pz, " ", (0, 0, 0), bg=shaded)

    def render_molecule(
        self,
        molecule: Molecule,
        rot: np.ndarray,
        camera_distance: float,
        isosurfaces: list[IsosurfaceMesh] | None = None,
    ):
        self.clear()
        if not molecule.atoms:
            return

        centroid = molecule.center()

        if isosurfaces:
            for mesh in isosurfaces:
                self.render_isosurface(mesh, rot, camera_distance, centroid)

        transformed = []
        for atom in molecule.atoms:
            pos = rot @ (atom.position - centroid)
            pos[2] += camera_distance
            transformed.append(pos)

        # Render bonds first (behind atoms)
        for i, j in molecule.bonds:
            self.render_bond(
                transformed[i],
                transformed[j],
                molecule.atoms[i].element.cpk_color,
                molecule.atoms[j].element.cpk_color,
            )

        # Sort atoms back-to-front for rendering order (painter's assist with z-buffer)
        atom_order = sorted(
            range(len(molecule.atoms)),
            key=lambda i: -transformed[i][2],
        )
        for i in atom_order:
            atom = molecule.atoms[i]
            radius = atom.element.covalent_radius * self.atom_scale
            self.render_sphere(transformed[i], radius, atom.element.cpk_color)

    def get_strip(self, y: int) -> Strip:
        if y < 0 or y >= self.height:
            return Strip.blank(self.width)
        segments = []
        x = 0
        while x < self.width:
            char = self.char_buf[y][x]
            fg = self.fg_buf[y][x]
            bg = self.bg_buf[y][x]
            # Batch consecutive cells with same style
            end = x + 1
            while (
                end < self.width
                and self.char_buf[y][end] == char
                and self.fg_buf[y][end] == fg
                and self.bg_buf[y][end] == bg
            ):
                end += 1
            text = char * (end - x)
            if bg is not None:
                style = Style(
                    color=f"rgb({fg[0]},{fg[1]},{fg[2]})",
                    bgcolor=f"rgb({bg[0]},{bg[1]},{bg[2]})",
                )
            elif char == " ":
                style = Style()
            else:
                style = Style(color=f"rgb({fg[0]},{fg[1]},{fg[2]})")
            segments.append(Segment(text, style))
            x = end
        return Strip(segments, self.width)
