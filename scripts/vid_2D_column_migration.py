
import matplotlib.pyplot as plt

from cadet import Cadet

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.benchmark_models import setting_Col2D_SMA_4comp_LWE_benchmark1


model = Cadet()
polyDeg = 3
axNElem = 3
radNElem = 3
parNElem = 3

eps_wall=0.5

model.root = setting_Col2D_SMA_4comp_LWE_benchmark1.get_model(
    polyDeg=polyDeg, axNElem=axNElem, radNElem=radNElem, parNElem=parNElem,
    write_solution_bulk=True, write_solution_particle=True, write_solution_solid=True,
    eps_wall=eps_wall, eps_inner=0.35, idas_reftol=1e-5
    )
modelName = f"2DLWE_radInlet_epsRc{eps_wall}_DG_P{polyDeg}Z{axNElem}radZ{radNElem}parZ{parNElem}"
model.filename = r"C:\Users\jmbr\software/" + modelName + ".h5"

model.save()
return_data = model.run_simulation()
print(return_data.return_code)
print(return_data.error_message)
model.load()
model.save()

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm, animation, ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse, Rectangle, Circle, Wedge
from src.utility import convergence

# ============================================================
# SETTINGS
# ============================================================

file = model.filename # r"C:\Users\jmbr\software\LWE_FV_Z250parZ100.h5"
vid_name = "LWE_FV_Z250parZ100" + ".mp4"

draw_outlet = True
draw_particles = True

section_names = ["LOAD", "WASH", "ELUTE"]
# section_names = ["Injection", "Wash"]

t_end = 750  # 10 # 750
component_idx = 3

n_particles = 5
min_particle_width_fig = 0.075
particle_img_res = 100  # 10 # 100 # root of pixels per particle, ie 100 -> 10000 pixels
bitrate = 100  # 100 # 800 # mp4 bitrate: kb per seco
frame_step = 1
fps = 10
dpi = 100  # spatial resolution

# Independent scale choices:
# - bulk + particles share one normalization
# - outlet uses its own normalization
use_log_norm_bulk_particles = True
use_log_norm_outlet = False

log_floor_bulk_particles = 1e-4   # only used when use_log_norm_bulk_particles=True
log_floor_outlet = 1e-4           # only used when use_log_norm_outlet=True

# time step assuming equidistant solution spacing in t
time_step = convergence.get_solution_times(file)[1] - convergence.get_solution_times(file)[0]

# For 1D bulk data only: homogeneous artificial radial resolution
n_radial_fake = 80

# For the concentration label in the outlet plot. Left out since repetitive with bulk label
outlet_zlabel = None  # "concentration (mol/L)"

# ============================================================
# LOAD COORDINATES
# ============================================================
ax_coords = np.asarray(convergence.get_axial_coordinates(file, unit="000"), dtype=float)

try:
    rad_coords = convergence.get_radial_coordinates(file, unit="000")
    rad_coords = np.asarray(rad_coords, dtype=float)
except Exception:
    rad_coords = None

par_coords = None
if draw_particles:
    try:
        par_coords = np.asarray(
            np.squeeze(
                convergence.sim_go_to(
                    convergence.get_simulation(file).root,
                    ['output', 'coordinates', 'unit_000', 'particle_coordinates_000']
                )
            ),
            dtype=float
        )
    except Exception:
        raise ValueError("draw_particles=True, but particle coordinates could not be loaded.")

# ============================================================
# LOAD DATA
# ============================================================
section_times = np.squeeze(
    convergence.sim_go_to(
        convergence.get_simulation(file).root,
        ['input', 'solver', 'sections', 'section_times']
    )
)

if not len(section_names) + 1 == len(section_times):
    raise Exception("section_names does not match the length of section_times (-1)")

column = np.squeeze(
    convergence.sim_go_to(
        convergence.get_simulation(file).root,
        ['output', 'solution', 'unit_000', 'solution_bulk']
    )
)
column = column[:t_end]

particle_pore = None
particle_solid = None

if draw_particles:
    particle_pore = np.squeeze(
        convergence.sim_go_to(
            convergence.get_simulation(file).root,
            ['output', 'solution', 'unit_000', 'solution_particle']
        )
    )
    particle_pore = particle_pore[:t_end]

    particle_solid = np.squeeze(
        convergence.sim_go_to(
            convergence.get_simulation(file).root,
            ['output', 'solution', 'unit_000', 'solution_solid']
        )
    )
    particle_solid = particle_solid[:t_end]

if draw_outlet:
    outlet = np.asarray(
        np.squeeze(
            convergence.sim_go_to(
                convergence.get_simulation(file).root,
                ['output', 'solution', 'unit_000', f'solution_outlet_comp_{component_idx:03}']
            )
        ),
        dtype=float
    )
    outlet = outlet[:t_end]

# ============================================================
# BULK: HANDLE COMPONENT AXIS
# ============================================================
if column.ndim == 4:
    # (nt, nx, nr, nc)
    column = column[:, :, :, component_idx]

elif column.ndim == 3:
    # Either (nt, nx, nr) or (nt, nx, nc)
    if rad_coords is None:
        column = column[:, :, component_idx]
    else:
        if column.shape[2] != len(rad_coords):
            column = column[:, :, component_idx]

column = column[::frame_step]
print("Max c^b: ", np.max(column))
print("Min c^b: ", np.min(column))

if column.shape[0] == 0:
    raise ValueError("Selected time range contains no frames.")

# ============================================================
# BULK: NORMALIZE DIMENSIONS
# ============================================================
if column.ndim == 2:
    # 1D bulk data: (nt, nx)
    nt, nx = column.shape

    if rad_coords is None:
        rad_coords = np.linspace(0.0, 1.0, n_radial_fake)
    else:
        r_max = float(np.max(rad_coords)) if len(rad_coords) > 0 else 1.0
        rad_coords = np.linspace(0.0, r_max, n_radial_fake)

    nr = len(rad_coords)
    column = np.repeat(column[:, :, np.newaxis], nr, axis=2)

elif column.ndim == 3:
    # 2D bulk data: (nt, nx, nr)
    nt, nx, nr = column.shape

    if rad_coords is None:
        rad_coords = np.linspace(0.0, 1.0, nr)
    elif len(rad_coords) != nr:
        raise ValueError(
            f"Mismatch between radial coordinate length ({len(rad_coords)}) "
            f"and bulk radial dimension ({nr})."
        )
else:
    raise ValueError(f"Unsupported bulk data shape after preprocessing: {column.shape}")

# ============================================================
# OUTLET: NORMALIZE DIMENSIONS
# ============================================================
if draw_outlet:
    outlet = outlet[::frame_step]
    print("Max outlet c^b: ", np.max(outlet))
    print("Min outlet c^b: ", np.min(outlet))

    if outlet.ndim == 1:
        if outlet.shape[0] != nt:
            raise ValueError(
                f"Outlet time dimension {outlet.shape[0]} does not match bulk time dimension {nt}."
            )

        # Create a thin pseudo-radial surface so the 3D history plot also works for 1D outlet data
        col_radius = np.sqrt(convergence.sim_go_to(
            convergence.get_simulation(file).root,
            ['input', 'model', 'unit_000', 'cross_section_area']
        ) / np.pi)
        outlet_rad_coords = np.asarray([0.0, col_radius], dtype=float)
        outlet_plot = np.repeat(outlet[:, np.newaxis], len(outlet_rad_coords), axis=1)

    elif outlet.ndim == 2:
        if outlet.shape[0] != nt:
            raise ValueError(
                f"Outlet time dimension {outlet.shape[0]} does not match bulk time dimension {nt}."
            )

        outlet_plot = outlet

        if rad_coords is not None and len(rad_coords) == outlet.shape[1]:
            outlet_rad_coords = np.asarray(rad_coords, dtype=float)
        else:
            outlet_rad_coords = np.linspace(0.0, 1.0, outlet.shape[1])
    else:
        raise ValueError(f"Unsupported outlet data shape after preprocessing: {outlet.shape}")

    outlet_plot = np.asarray(outlet_plot, dtype=float)
    outlet_time = np.arange(nt, dtype=float) * frame_step * time_step

# ============================================================
# PARTICLES: PREPARE HELPERS
# ============================================================
def prepare_particle_field(arr, par_coords, frame_step, field_name, atol=1e-14):
    """
    Normalize particle arrays to shape (nt, nx, npart)
    Supported after component selection:
      - 1D particles: (nt, nx, npart)
      - 2D particles: (nt, nx, ncolrad, npart) -> takes first column-radial position
    """
    arr = np.asarray(arr)
    npar = len(par_coords)

    if arr.ndim == 4:
        # assume (nt, nx, ncolrad, npart)
        arr = arr[:, :, 0, :]
    elif arr.ndim == 3:
        # assume (nt, nx, npart)
        pass
    else:
        raise ValueError(
            f"Unsupported shape for {field_name}: {arr.shape}. "
            f"Expected 3D or 4D after component handling."
        )

    arr = arr[::frame_step]

    if arr.ndim != 3:
        raise ValueError(f"{field_name} could not be normalized to (nt, nx, npart).")

    if arr.shape[2] != npar:
        raise ValueError(
            f"{field_name}: particle dimension {arr.shape[2]} does not match "
            f"particle coordinate length {npar}."
        )

    if arr.shape[0] != nt:
        raise ValueError(
            f"{field_name}: time dimension {arr.shape[0]} does not match bulk time dimension {nt}."
        )

    if arr.shape[1] != nx:
        raise ValueError(
            f"{field_name}: axial dimension {arr.shape[1]} does not match bulk axial dimension {nx}."
        )

    # Merge double interface DG coordinates
    groups = []
    start = 0
    order = np.argsort(par_coords)
    par_coords = par_coords[order]
    arr = arr[:, :, order]
    for i in range(1, npar + 1):
        if i == npar:
            groups.append(order[start:i])
        else:
            same = abs(par_coords[i] - par_coords[start]) <= atol
            if not same:
                groups.append(order[start:i])
                start = i

    # Nothing to merge
    if len(groups) == npar and all(len(g) == 1 for g in groups):
        return arr, par_coords

    par_coords_new = np.array([np.mean(par_coords[g]) for g in groups])

    arr_new = np.empty((arr.shape[0], arr.shape[1], len(groups)), dtype=arr.dtype)
    for k, g in enumerate(groups):
        arr_new[:, :, k] = np.mean(arr[:, :, g], axis=2)

    return arr_new, par_coords_new

# ============================================================
# PARTICLES: SELECT COMPONENT EXPLICITLY FIRST
# ============================================================
if draw_particles:
    # Typical raw shapes:
    # 1D pore/solid:   (nt, nx, npart, ncomp)
    # 2D pore/solid:   (nt, nx, ncolrad, npart, ncomp)
    if particle_pore.ndim >= 4:
        particle_pore = particle_pore[..., component_idx]
    if particle_solid.ndim >= 4:
        particle_solid = particle_solid[..., component_idx]

    particle_pore, _ = prepare_particle_field(
        particle_pore, par_coords, frame_step, "solution_particle"
    )

    print("Max c^p: ", np.max(particle_pore))
    print("Min c^p: ", np.min(particle_pore))

    particle_solid, par_coords = prepare_particle_field(
        particle_solid, par_coords, frame_step, "solution_solid"
    )

    print("Max c^s: ", np.max(particle_solid))
    print("Min c^s: ", np.min(particle_solid))

# ============================================================
# BULK: PREPARE DATA
# ============================================================
r_signed = np.concatenate((-rad_coords[:0:-1], rad_coords))


def make_signed_slice(frame_2d):
    return np.concatenate((frame_2d[:, :0:-1], frame_2d), axis=1)


frames_signed = np.empty((nt, nx, 2 * nr - 1), dtype=column.dtype)
for t in range(nt):
    frames_signed[t] = make_signed_slice(column[t])

frames_plot = np.transpose(frames_signed, (0, 2, 1))

# ============================================================
# NORMALIZATION HELPERS
# ============================================================
def build_norm(data_min, data_max, use_log_norm, log_floor):
    data_min = float(data_min)
    data_max = float(data_max)

    if use_log_norm:
        data_min = max(data_min, log_floor)
        data_max = max(data_max, data_min * 10.0)
        norm = colors.LogNorm(vmin=data_min, vmax=data_max, clip=True)
        norm_name = "LogNorm"
    else:
        if np.isclose(data_min, data_max):
            data_max = data_min + 1e-12
        norm = colors.Normalize(vmin=data_min, vmax=data_max, clip=True)
        norm_name = "Linear"

    return norm, norm_name


# ============================================================
# SHARED NORMALIZATION ACROSS BULK + PARTICLES
# ============================================================
bulk_particles_min = float(np.nanmin(column))
bulk_particles_max = float(np.nanmax(column))

if draw_particles:
    bulk_particles_min = min(
        bulk_particles_min,
        float(np.nanmin(particle_pore)),
        float(np.nanmin(particle_solid))
    )
    bulk_particles_max = max(
        bulk_particles_max,
        float(np.nanmax(particle_pore)),
        float(np.nanmax(particle_solid))
    )

bulk_particles_norm, bulk_particles_norm_name = build_norm(
    bulk_particles_min,
    bulk_particles_max,
    use_log_norm_bulk_particles,
    log_floor_bulk_particles,
)

# ============================================================
# OUTLET NORMALIZATION (INDEPENDENT)
# ============================================================
outlet_norm = None
outlet_norm_name = None
if draw_outlet:
    outlet_min = float(np.nanmin(outlet_plot))
    outlet_max = float(np.nanmax(outlet_plot))
    outlet_norm, outlet_norm_name = build_norm(
        outlet_min,
        outlet_max,
        use_log_norm_outlet,
        log_floor_outlet,
    )

# Use the same colormap everywhere
shared_cmap = cm.viridis

print(f"bulk/particles plotted color range min = {bulk_particles_norm.vmin:.3e}")
print(f"bulk/particles plotted color range max = {bulk_particles_norm.vmax:.3e}")
print(
    f"bulk/particles using {bulk_particles_norm_name} with "
    f"vmin = {bulk_particles_norm.vmin:.3e}, vmax = {bulk_particles_norm.vmax:.3e}"
)

if draw_outlet:
    print(f"outlet plotted color range min = {outlet_norm.vmin:.3e}")
    print(f"outlet plotted color range max = {outlet_norm.vmax:.3e}")
    print(
        f"outlet using {outlet_norm_name} with "
        f"vmin = {outlet_norm.vmin:.3e}, vmax = {outlet_norm.vmax:.3e}"
    )

# Only clip where LogNorm is used
if use_log_norm_bulk_particles:
    frames_plot = np.maximum(frames_plot, log_floor_bulk_particles)

if draw_outlet and use_log_norm_outlet:
    outlet_plot = np.maximum(outlet_plot, log_floor_outlet)

x0 = float(np.min(ax_coords))
x1 = float(np.max(ax_coords))
R = float(np.max(rad_coords))
L = x1 - x0

if R <= 0:
    R = 1.0

# ============================================================
# PARTICLE: PRECOMPUTE DISK GEOMETRY
# ============================================================
if draw_particles:
    par_max = float(np.max(par_coords))
    if par_max <= 0:
        par_r_norm = np.linspace(0.0, 1.0, len(par_coords))
    else:
        par_r_norm = par_coords / par_max

    grid = np.linspace(-1.0, 1.0, particle_img_res)
    XX, YY = np.meshgrid(grid, grid)
    RR = np.sqrt(XX**2 + YY**2)
    disk_mask = RR <= 1.0
    left_mask = (XX < 0.0) & disk_mask
    right_mask = (XX >= 0.0) & disk_mask

    def make_split_particle_rgba(pore_profile, solid_profile):
        pore_profile = np.asarray(pore_profile, dtype=float)
        solid_profile = np.asarray(solid_profile, dtype=float)

        pore_field = np.interp(
            np.clip(RR, 0.0, 1.0),
            par_r_norm, pore_profile,
            left=pore_profile[0], right=pore_profile[-1]
        )
        solid_field = np.interp(
            np.clip(RR, 0.0, 1.0),
            par_r_norm, solid_profile,
            left=solid_profile[0], right=solid_profile[-1]
        )

        if use_log_norm_bulk_particles:
            pore_field = np.maximum(pore_field, log_floor_bulk_particles)
            solid_field = np.maximum(solid_field, log_floor_bulk_particles)

        rgba = np.zeros((particle_img_res, particle_img_res, 4), dtype=float)
        rgba[left_mask] = shared_cmap(bulk_particles_norm(pore_field[left_mask]))
        rgba[right_mask] = shared_cmap(bulk_particles_norm(solid_field[right_mask]))
        rgba[~disk_mask, 3] = 0.0
        return rgba

# ============================================================
# FIGURE SETUP
# ============================================================
fig_h = 6.2 if draw_particles else 4.2
fig_w = 13.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

bg = "#0f1117"
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

if draw_particles:
    fig.subplots_adjust(left=0.05, right=0.72, top=0.88, bottom=0.30)
else:
    fig.subplots_adjust(left=0.05, right=0.72, top=0.90, bottom=0.10)

im = ax.imshow(
    frames_plot[0],
    extent=[x0, x1, -R, R],
    origin="lower",
    cmap=shared_cmap,
    norm=bulk_particles_norm,
    interpolation="bicubic",
    aspect="auto",
    zorder=2,
    animated=True
)

# ============================================================
# CYLINDER OVERLAY
# ============================================================
body = Rectangle(
    (x0, -R), L, 2 * R,
    linewidth=1.8,
    edgecolor=(1, 1, 1, 0.55),
    facecolor="none",
    zorder=5
)
ax.add_patch(body)

cap_width = 0.06 * L
left_cap = Ellipse(
    (x0, 0),
    width=cap_width,
    height=2 * R,
    linewidth=1.8,
    edgecolor=(1, 1, 1, 0.60),
    facecolor=(1, 1, 1, 0.05),
    zorder=6
)
right_cap = Ellipse(
    (x1, 0),
    width=cap_width,
    height=2 * R,
    linewidth=1.8,
    edgecolor=(1, 1, 1, 0.60),
    facecolor=(1, 1, 1, 0.05),
    zorder=6
)
ax.add_patch(left_cap)
ax.add_patch(right_cap)

ax.plot([x0, x1], [R, R], color=(1, 1, 1, 0.18), lw=2.0, zorder=6)
ax.plot([x0, x1], [-R, -R], color=(0, 0, 0, 0.22), lw=2.0, zorder=6)
ax.plot([x0, x1], [0, 0], color=(1, 1, 1, 0.12), lw=1.0, ls="--", zorder=6)

im.set_clip_path(body)

# Shared colorbar for bulk + particles only
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
cbar.set_label("Concentration (mol / L)", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.get_yticklabels(), color="white")
cbar.outline.set_edgecolor((1, 1, 1, 0.4))

if use_log_norm_bulk_particles:
    cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
else:
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

# ============================================================
# OUTLET HISTORY PANEL (3D)
# ============================================================
if draw_outlet:
    out_ax = fig.add_axes([0.79, 0.20 if draw_particles else 0.16, 0.19, 0.68], projection="3d")
    out_ax.set_facecolor(bg)
    out_ax.xaxis.pane.set_facecolor((0.10, 0.11, 0.15, 1.0))
    out_ax.yaxis.pane.set_facecolor((0.10, 0.11, 0.15, 1.0))
    out_ax.zaxis.pane.set_facecolor((0.10, 0.11, 0.15, 1.0))
    out_ax.xaxis.pane.set_edgecolor((1, 1, 1, 0.08))
    out_ax.yaxis.pane.set_edgecolor((1, 1, 1, 0.08))
    out_ax.zaxis.pane.set_edgecolor((1, 1, 1, 0.08))
    out_ax.tick_params(colors="white", labelsize=8, pad=1)
    out_ax.xaxis.label.set_color("white")
    out_ax.yaxis.label.set_color("white")
    out_ax.zaxis.label.set_color("white")
    out_ax.set_title("Outlet concentration", color="white", fontsize=11, pad=8)
    out_ax.set_xlabel("Time (s)", labelpad=4)
    out_ax.set_ylabel("Radial coordinate", labelpad=4)
    if outlet_zlabel is not None:
        out_ax.set_zlabel(f"{outlet_zlabel}", labelpad=4)
    out_ax.view_init(elev=30, azim=-125)
    out_ax.set_xlim(float(outlet_time[0]), float(outlet_time[-1]) if nt > 1 else float(outlet_time[0]) + 1.0)
    out_ax.set_ylim(float(outlet_rad_coords[0]), float(outlet_rad_coords[-1]))
    out_ax.set_zlim(0.0, 1.0)

    if use_log_norm_outlet:
        try:
            ztick_values = np.geomspace(float(outlet_norm.vmin), float(outlet_norm.vmax), 4)
        except Exception:
            ztick_values = np.linspace(float(outlet_norm.vmin), float(outlet_norm.vmax), 4)

        def format_tick(v):
            exp = int(np.floor(np.log10(v)))
            return rf"$10^{{{exp}}}$"
    else:
        ztick_values = np.linspace(float(outlet_norm.vmin), float(outlet_norm.vmax), 4)

        def format_tick(v):
            return f"{v:.2g}"

    ztick_positions = np.clip(outlet_norm(ztick_values), 0.0, 1.0)
    out_ax.set_zticks(ztick_positions)
    out_ax.set_zticklabels([format_tick(v) for v in ztick_values], color="white", fontsize=8)

    outlet_artists = {
        "surface": None,
        "profile_line": None,
    }

    def draw_outlet_surface(frame_idx):
        if frame_idx < 0:
            return []

        time_slice = outlet_time[:frame_idx + 1]
        data_raw = outlet_plot[:frame_idx + 1]
        data_slice = np.clip(outlet_norm(data_raw), 0.0, 1.0)

        T, RR = np.meshgrid(time_slice, outlet_rad_coords, indexing="ij")

        if outlet_artists["surface"] is not None:
            outlet_artists["surface"].remove()
        if outlet_artists["profile_line"] is not None:
            outlet_artists["profile_line"].remove()

        facecolors = shared_cmap(outlet_norm(data_raw))
        outlet_artists["surface"] = out_ax.plot_surface(
            T, RR, data_slice,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
            zorder=2
        )

        profile = data_slice[-1]
        outlet_artists["profile_line"], = out_ax.plot(
            np.full_like(outlet_rad_coords, time_slice[-1], dtype=float),
            outlet_rad_coords,
            profile,
            color=(1, 1, 1, 0.95),
            lw=1.4,
            zorder=3
        )

        return [outlet_artists["surface"], outlet_artists["profile_line"]]

    draw_outlet_surface(0)

title = ax.set_title(
    f"Concentration distribution in {section_names[0]} Step   |   Time (s) 0",
    color="white",
    fontsize=14,
    pad=12
)

ax.set_xlim(x0 - 0.04 * L, x1 + 0.04 * L)
ax.set_ylim(-1.18 * R, 1.18 * R)
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

# ============================================================
# PARTICLE INSETS BELOW COLUMN
# ============================================================
particle_ims = []
particle_marker_lines = []
particle_axes = []

if draw_particles:
    if n_particles < 1:
        raise ValueError("n_particles must be at least 1.")
    if n_particles > nx:
        raise ValueError(
            f"Requested n_particles={n_particles}, but only nx={nx} axial grid points exist."
        )

    particle_idx = np.linspace(0, nx - 1, n_particles)
    particle_idx = np.round(particle_idx).astype(int)
    particle_idx = np.unique(particle_idx)

    if len(particle_idx) != n_particles:
        raise ValueError(
            "n_particles is too large for the available axial grid; "
            "equidistant selection produced duplicate indices."
        )

    particle_x = np.asarray(ax_coords)[particle_idx]

    fig.canvas.draw()
    ax_bbox = ax.get_position()

    if np.isclose(x1, x0) or np.any(np.diff(np.asarray(ax_coords)) < 0):
        u = (particle_idx.astype(float) + 0.5) / nx
    else:
        denom = (x1 - x0)
        u = (particle_x - x0) / denom if denom > 0 else (particle_idx.astype(float) + 0.5) / nx

    u = np.clip(u, 0.02, 0.98)

    if len(u) == 1:
        min_center_spacing = 1.0
    else:
        min_center_spacing = float(np.min(np.diff(np.sort(u))))

    particle_width_fig = 0.75 * min_center_spacing * ax_bbox.width

    if particle_width_fig <= 1e-6:
        particle_width_fig = min(0.10, 0.8 * ax_bbox.width / n_particles)

    if particle_width_fig < min_particle_width_fig:
        raise ValueError(
            f"Too many particles for the figure width. "
            f"Maximum allowed inset width is {particle_width_fig:.3f}, "
            f"but minimum required is {min_particle_width_fig:.3f}. "
            f"Reduce n_particles or increase figure width."
        )

    particle_width_fig = min(particle_width_fig, 0.12)

    fig_w, fig_h_in = fig.get_size_inches()
    particle_height_fig = particle_width_fig * (fig_w / fig_h_in)

    particle_bottom = 0.025
    particle_top_margin = 0.02
    title_gap = 0.03
    available_height = ax_bbox.y0 - particle_top_margin - particle_bottom

    if available_height <= 0:
        raise ValueError(
            "Not enough vertical space for particle insets. "
            "Increase figure height or increase the subplot bottom margin."
        )

    if particle_height_fig > available_height:
        particle_height_fig = available_height
        particle_width_fig = particle_height_fig * (fig_h_in / fig_w)

    title_y = min(
        particle_bottom + particle_height_fig + title_gap,
        ax_bbox.y0 - 0.005
    )

    fig.text(
        0.35, title_y,
        "Particle zoom-ins at selected axial positions   |   left half: pore/liquid   |   right half: solid/adsorbed",
        color=(1, 1, 1, 0.78),
        ha="center", va="bottom", fontsize=10
    )

    for x_sel, idx_sel, u_sel in zip(particle_x, particle_idx, u):
        x_center_fig = ax_bbox.x0 + u_sel * ax_bbox.width
        left_fig = x_center_fig - 0.5 * particle_width_fig

        pax = fig.add_axes([left_fig, particle_bottom, particle_width_fig, particle_height_fig])
        pax.set_facecolor(bg)
        pax.set_xticks([])
        pax.set_yticks([])
        for s in pax.spines.values():
            s.set_visible(False)

        rgba0 = make_split_particle_rgba(
            particle_pore[0, idx_sel, :],
            particle_solid[0, idx_sel, :]
        )

        pim = pax.imshow(
            rgba0,
            extent=[-1, 1, -1, 1],
            origin="lower",
            interpolation="bicubic",
            animated=True
        )

        solid_half_hatch = Wedge(
            center=(0, 0),
            r=1.0,
            theta1=-90,
            theta2=90,
            facecolor=(1, 1, 1, 0.03),
            edgecolor=(1, 1, 1, 0.22),
            hatch='////',
            linewidth=0.0,
            zorder=3
        )
        pax.add_patch(solid_half_hatch)

        circle = Circle((0, 0), 1.0, facecolor="none", edgecolor=(1, 1, 1, 0.68), lw=1.4, zorder=4)
        pax.add_patch(circle)
        pax.plot([0, 0], [-1, 1], color=(1, 1, 1, 0.68), lw=1.0, zorder=4)

        pax.set_xlim(-1.05, 1.05)
        pax.set_ylim(-1.05, 1.05)
        pax.set_aspect("equal")
        pax.set_title(f"z = {float(x_sel):.3g}", color="white", fontsize=8, pad=2)

        line = ax.plot(
            [x_sel, x_sel], [-R, R],
            color=(1, 1, 1, 0.22), lw=1.0, ls=":",
            zorder=7
        )[0]

        particle_axes.append(pax)
        particle_ims.append((pim, idx_sel))
        particle_marker_lines.append(line)

# ============================================================
# ANIMATION
# ============================================================
def update(frame_idx):
    artists = []

    time = frame_idx * frame_step * time_step
    section_name = section_names[0]
    for i in range(len(section_times) - 1):
        if section_times[i] < time < section_times[i + 1]:
            section_name = section_names[i]
            break

    im.set_array(frames_plot[frame_idx])
    title.set_text(
        f"Concentration distribution in {section_name} Step  |   Time (s) {time}"
    )

    artists.extend([im, title])
    if draw_outlet:
        artists.extend(draw_outlet_surface(frame_idx))

    if draw_particles:
        for pim, idx_sel in particle_ims:
            rgba = make_split_particle_rgba(
                particle_pore[frame_idx, idx_sel, :],
                particle_solid[frame_idx, idx_sel, :]
            )
            pim.set_data(rgba)
            artists.append(pim)

    return artists


ani = animation.FuncAnimation(
    fig,
    update,
    frames=nt,
    interval=1000 // fps,
    blit=False
)

# ============================================================
# SAVE AS MP4
# ============================================================
writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
ani.save(vid_name, writer=writer, dpi=dpi)

plt.show()







