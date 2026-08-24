import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.colors as mcolors
import cmocean
import cartopy.crs as ccrs
import os

# PARAMETERS
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
simu = '/lus/store/CT1/c1601279/aperez/resultats/diff'
file = 'diff_1PSOURCE_15PSU_25deg.nc'
figures = '/lus/store/CT1/c1601279/aperez/figures'

SWIO = (25, 69, -36, 7)
annee = '2017'   # annee sur laquelle la moyenne (baseline) est calculee

gridline_style = {'draw_labels': True, 'linestyle': '--', 'linewidth': 0.5}
figsize = (20, 6)
velo_cmap = cmocean.cm.balance
vort_cmap = cmocean.cm.curl
ener_cmap = cmocean.cm.speed

pas_animation = 5   # 1 tous les 5 jours

os.makedirs(figures, exist_ok=True)

# ---------------------------------------------------------------
# GRILLE
# ---------------------------------------------------------------
g = xr.open_dataset(grid)
lon = g['lon_rho'].values
lat = g['lat_rho'].values
mask_rho = g['mask_rho'].values
pm = g['pm'].values
pn = g['pn'].values
angle = g['angle'].values
g.close()
print("Grid loaded.")

lon_c = lon[:-1, :-1]
lat_c = lat[:-1, :-1]
mask_c = mask_rho[:-1, :-1]
pm_c = pm[:-1, :-1]
pn_c = pn[:-1, :-1]
angle_c = angle[:-1, :-1]

# ---------------------------------------------------------------
# DONNEES
# ---------------------------------------------------------------
d = xr.open_dataset(os.path.join(simu, file))
u_full = d.u[:, -1, :, :]
v_full = d.v[:, -1, :, :]
w_full = d.w[:, -1, :, :]
time = d.time
d.close()
print('Data loaded.')

fill_value = 9.96921e+36
u_full = u_full.where((u_full != fill_value), np.nan).sel(time=annee)
v_full = v_full.where((v_full != fill_value), np.nan).sel(time=annee)
w_full = w_full.where((w_full != fill_value), np.nan).sel(time=annee)
time = u_full.time
print('NaN values added')

n_days = len(time)
indices_a_animer = list(range(0, n_days, pas_animation))
print(f"{len(indices_a_animer)} images dans l'animation (1 tous les {pas_animation} jours).")

# ---------------------------------------------------------------
# MOYENNE ANNUELLE (baseline), calculee UNE SEULE FOIS, hors boucle
# ---------------------------------------------------------------
u_yr = u_full.mean(dim='time').values
v_yr = v_full.mean(dim='time').values
w_yr = w_full.mean(dim='time').values
print("Time mean calculated (once)")

u_yr_geo = u_yr[:-1, :] * np.cos(angle_c) - v_yr[:, :-1] * np.sin(angle_c)
v_yr_geo = u_yr[:-1, :] * np.sin(angle_c) + v_yr[:, :-1] * np.cos(angle_c)


def calculer_champs_turbulents(t):
    """Calcule vitesse turbulente, vorticite turbulente et EKE pour un jour donne."""
    u_jour = u_full.isel(time=t).values
    v_jour = v_full.isel(time=t).values
    w_jour = w_full.isel(time=t).values

    ut = u_jour - u_yr
    vt = v_jour - v_yr
    wt = w_jour - w_yr

    ut_geo = ut[:-1, :] * np.cos(angle_c) - vt[:, :-1] * np.sin(angle_c)
    vt_geo = ut[:-1, :] * np.sin(angle_c) + vt[:, :-1] * np.cos(angle_c)
    wt_geo = wt[:-1, :-1]

    velocity_t = np.sqrt(ut_geo**2 + vt_geo**2)

    dvt_dlon = np.gradient(vt_geo, axis=1) * pm_c
    dut_dlat = np.gradient(ut_geo, axis=0) * pn_c
    vorticity_t = dvt_dlon - dut_dlat

    EKE = 0.5 * (ut_geo**2 + vt_geo**2 + wt_geo**2)

    return velocity_t, vorticity_t, EKE


# ---------------------------------------------------------------
# FIGURE DE BASE
# ---------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
titre = fig.suptitle("")

for ax in axs:
    ax.set_extent(SWIO)
    ax.coastlines(resolution='50m')
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, zorder=3)

    # Terre dessinee depuis mask_rho (robuste, pas de dependance reseau)
    terre_affichee = np.ma.masked_where(mask_c == 1, mask_c)
    ax.pcolormesh(lon_c, lat_c, terre_affichee, cmap=mcolors.ListedColormap(['#DEB887']),
                  transform=ccrs.PlateCarree(), zorder=4)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), **gridline_style)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'color': 'k'}

# Premiere frame, pour initialiser les 3 panneaux + colorbars (creees UNE FOIS)
velocity0, vorticity0, EKE0 = calculer_champs_turbulents(indices_a_animer[0])

# --- Vitesse ---
norm_v = mpl.colors.LogNorm(10**-1.5, 10**0.2)
pcm_v = axs[0].pcolormesh(lon_c, lat_c, velocity0, cmap=velo_cmap, norm=norm_v, transform=ccrs.PlateCarree())
cb_v = plt.colorbar(pcm_v, ax=axs[0], label=r'$v$ [m/s]', orientation='vertical')
axs[0].set_title("Velocity")

# --- Vorticite ---
norm_vort = mpl.colors.Normalize(vmin=-5e-5, vmax=5e-5)
pcm_vort = axs[1].pcolormesh(lon_c, lat_c, vorticity0, cmap=vort_cmap, norm=norm_vort, transform=ccrs.PlateCarree())
cb_vort = plt.colorbar(pcm_vort, ax=axs[1], label=r'$\omega$ [s⁻¹]', orientation='vertical')
axs[1].set_title("Vorticity")

# --- Energie (bornes fixees a partir de la 1ere frame) ---
a_exp = int(np.log10(np.nanmax(EKE0)))
b_exp = a_exp - 2.5
norm_eke = mpl.colors.LogNorm(vmin=10**b_exp, vmax=10**a_exp)
pcm_eke = axs[2].pcolormesh(lon_c, lat_c, EKE0, cmap=ener_cmap, norm=norm_eke, transform=ccrs.PlateCarree())
cb_eke = plt.colorbar(pcm_eke, ax=axs[2], label=r'$EKE$ [m².s⁻²]', orientation='vertical')
axs[2].set_title("Energy")

fig.tight_layout()

artistes_a_effacer = [pcm_v, pcm_vort, pcm_eke]


# ---------------------------------------------------------------
# FONCTION APPELEE POUR CHAQUE FRAME
# ---------------------------------------------------------------
def mettre_a_jour(t):
    global pcm_v, pcm_vort, pcm_eke, artistes_a_effacer

    for artiste in artistes_a_effacer:
        artiste.remove()
    artistes_a_effacer = []

    velocity_t, vorticity_t, EKE = calculer_champs_turbulents(t)
    date_str = str(time.values[t])[:10]

    pcm_v = axs[0].pcolormesh(lon_c, lat_c, velocity_t, cmap=velo_cmap, norm=norm_v, transform=ccrs.PlateCarree())
    pcm_vort = axs[1].pcolormesh(lon_c, lat_c, vorticity_t, cmap=vort_cmap, norm=norm_vort, transform=ccrs.PlateCarree())
    pcm_eke = axs[2].pcolormesh(lon_c, lat_c, EKE, cmap=ener_cmap, norm=norm_eke, transform=ccrs.PlateCarree())

    artistes_a_effacer = [pcm_v, pcm_vort, pcm_eke]

    titre.set_text(f"CROCO Turbulent {date_str}")
    print(f'Frame : {date_str}')
    return pcm_v, pcm_vort, pcm_eke


# ---------------------------------------------------------------
# ANIMATION
# ---------------------------------------------------------------
anim = animation.FuncAnimation(
    fig, mettre_a_jour, frames=indices_a_animer,
    interval=200, blit=False
)

sortie_gif = os.path.join(figures, 'animation_turbulent_diff_15PSU_croco.gif')
anim.save(sortie_gif, writer='pillow', fps=5)
print(f"\nAnimation sauvegardee : {sortie_gif}")
