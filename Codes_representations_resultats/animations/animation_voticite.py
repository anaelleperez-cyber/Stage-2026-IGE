import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cmocean
import os

# Chemins
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
diff_file = '/lus/store/CT1/c1601279/aperez/resultats/exp_1PSOURCE_1PSU_25deg/001swiose_avg.nc'
figures = '/lus/store/CT1/c1601279/aperez/figures'

SWIO = (25, 69, -36,-7)
seuil_aberrant = 1e5

figsize = (8, 7)

pas_animation = 5

# Grilles et donnees
g = xr.open_dataset(grid)
lon = g['lon_rho'].values
lat = g['lat_rho'].values
angle = g['angle'].values
pm = g['pm'].values
pn = g['pn'].values
mask_rho = g['mask_rho'].values
g.close()

d = xr.open_dataset(diff_file)
du_full = d['u']   # u et v sont deja des diff de vitesses
dv_full = d['v']
if 's_rho' in du_full.dims:
    du_full = du_full.isel(s_rho=-1)
    dv_full = dv_full.isel(s_rho=-1)

du_full = du_full.where(np.abs(du_full) < seuil_aberrant, np.nan)
dv_full = dv_full.where(np.abs(dv_full) < seuil_aberrant, np.nan)
time = du_full.time
d.close()

n_days = len(time)
indices_a_animer = list(range(0, n_days, pas_animation))
print(f"{len(indices_a_animer)} images dans l'animation (1 tous les {pas_animation} jours).")

lon_c = lon[:-1, :-1]
lat_c = lat[:-1, :-1]
mask_c = mask_rho[:-1, :-1]
pm_c = pm[:-1, :-1]
pn_c = pn[:-1, :-1]
angle_c = angle[:-1, :-1]
masque_zone = (lon_c >= SWIO[0]) & (lon_c <= SWIO[1]) & (lat_c >= SWIO[2]) & (lat_c <= SWIO[3])

def calculer_vorticite_jour(t):
    du_jour = du_full.isel(time=t).values
    dv_jour = dv_full.isel(time=t).values

    du_geo = du_jour[:-1, :] * np.cos(angle_c) - dv_jour[:, :-1] * np.sin(angle_c)
    dv_geo = du_jour[:-1, :] * np.sin(angle_c) + dv_jour[:, :-1] * np.cos(angle_c)

    dv_dlon = np.gradient(dv_geo, axis=1) * pm_c
    du_dlat = np.gradient(du_geo, axis=0) * pn_c
    vort = dv_dlon - du_dlat

    vort = np.where((mask_c == 1) & masque_zone, vort, np.nan)
    return vort


# FIGURE DE BASE
fig, ax = plt.subplots(figsize=figsize, dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent(SWIO)
ax.coastlines(resolution='50m')
ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, zorder=3)

# Terre dessinee directement depuis mask_rho
terre_affichee = np.ma.masked_where(mask_c == 1, mask_c)
ax.pcolormesh(lon_c, lat_c, terre_affichee, cmap=mcolors.ListedColormap(['#DEB887']),
              transform=ccrs.PlateCarree(), zorder=4)

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.4)
gl.top_labels = False
gl.right_labels = False

# Echelle de couleur FIXE, basee sur l'ordre de grandeur deja observe
max_abs = 2e-5
print(f"max_abs utilise pour la normalisation (fixe) : {max_abs:.3e}")

norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
vort0 = calculer_vorticite_jour(indices_a_animer[0])

pcm = ax.pcolormesh(lon_c, lat_c, vort0, cmap=cmocean.cm.curl, norm=norm,
                      transform=ccrs.PlateCarree(), zorder=1)
cb = plt.colorbar(pcm, ax=ax, label='Δζ [s⁻¹]', orientation='vertical')

titre = ax.set_title("")

artistes_a_effacer = [pcm]


def mettre_a_jour(t):
    global pcm, artistes_a_effacer

    for artiste in artistes_a_effacer:
        artiste.remove()
    artistes_a_effacer = []

    vort = calculer_vorticite_jour(t)
    date_str = str(time.values[t])[:10]

    pcm = ax.pcolormesh(lon_c, lat_c, vort, cmap=cmocean.cm.curl, norm=norm,
                      transform=ccrs.PlateCarree(), zorder=1)
    artistes_a_effacer = [pcm]

    titre.set_text(f"{date_str}")
    print(f'Frame : {date_str}')
    return pcm,

#creation animation
anim = animation.FuncAnimation(
    fig, mettre_a_jour, frames=indices_a_animer,
    interval=200, blit=False
)

# sortie en gif
sortie_gif = os.path.join(figures, 'vorticite_animation.gif')
anim.save(sortie_gif, writer='pillow', fps=5)
print(f"\nAnimation sauvegardee : {sortie_gif}")

