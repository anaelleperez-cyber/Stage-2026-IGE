import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import os

# Chemins
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
figures = '/lus/store/CT1/c1601279/aperez/figures'
simu = '/lus/store/CT1/c1601279/aperez/resultats/exp_12PSOURCE_1PSU_25deg/'
file = '001swiose_avg.nc'

SWIO = (32, 50, -28, -12)

gridline_style = {'draw_labels': True, 'linestyle': '--', 'linewidth': 0.5}
figsize = (8, 7)

os.makedirs(figures, exist_ok=True)

# Pas entre chaque image de l'animation (en jours)
pas_animation = 5

# Les 12 PSOURCE (Isrc, Jsrc)
psource_points = [
    (96, 200),   # Zambeze
    (188, 192),  # Tsiribihina
    (208, 240),  # Betsiboka
    (178, 171),  # Mangoky
    (187, 197),  # Manambolo
    (238, 272),  # Mahavavy
    (233, 265),  # Sambirano
    (70, 119),   # Limpopo
    (83, 173),   # Save
    (79, 185),   # Buzi et Pungoe
    (186, 184),  # Morondava
    (185, 128),  # Linta
]

#Grille et données
g = xr.open_dataset(grid)
lon = g['lon_rho'].values
lat = g['lat_rho'].values
mask_rho = g['mask_rho'].values
g.close()

d = xr.open_dataset(os.path.join(simu, file))
salt_full = d.salt
if 's_rho' in salt_full.dims:
    salt_full = salt_full.isel(s_rho=-1)

fill_value = 9.96921e+36
salt_full = salt_full.where((salt_full != fill_value), np.nan)
time = salt_full.time
d.close()

n_days = len(time)
indices_a_animer = range(0, n_days, pas_animation)
print(f"{len(indices_a_animer)} images dans l'animation (1 tous les {pas_animation} jours).")

# palette
fond_cmap = mcolors.LinearSegmentedColormap.from_list(
    'rainbow_20_35', ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
)
fond_cmap.set_over('firebrick')
niveaux = np.arange(20, 36, 1)

# Figure de base
fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent(SWIO)
ax.coastlines(resolution='50m')
ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, zorder=3)

# Terre dessinee DIRECTEMENT depuis mask_rho (0=terre), en beige uni
terre_affichee = np.ma.masked_where(mask_rho == 1, mask_rho)
ax.pcolormesh(lon[:, :], lat[:, :], terre_affichee, cmap=mcolors.ListedColormap(['#DEB887']),
              transform=ccrs.PlateCarree(), zorder=4)

gl = ax.gridlines(crs=ccrs.PlateCarree(), **gridline_style)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {'color': 'k'}

# Premiere image (jour 0), pour initialiser le fond colore
salt0 = np.where(mask_rho == 1, salt_full.isel(time=0).values, np.nan)
pcm = ax.contourf(lon[:, :], lat[:, :], salt0, levels=niveaux, cmap=fond_cmap,
                    extend='both', transform=ccrs.PlateCarree(), zorder=1)
cb = plt.colorbar(pcm, ax=ax, label='Salinité [PSU]', orientation='vertical')
cb.set_ticks(niveaux)


# Les 12 PSOURCE identifies
for Isrc, Jsrc in psource_points:
    ax.plot(lon[Jsrc, Isrc], lat[Jsrc, Isrc], marker='o', color='cyan',
            markeredgecolor='black', markeredgewidth=1, markersize=8,
            transform=ccrs.PlateCarree(), zorder=6)

titre = ax.set_title("")

# Liste des artistes à supprimer à chaque frame
artistes_a_effacer = [pcm]

# Fonction a appeler pour chaque frame
def mettre_a_jour(t):
    global pcm, artistes_a_effacer

    for artiste in artistes_a_effacer:
        artiste.remove()
    artistes_a_effacer = []

    salt = np.where(mask_rho == 1, salt_full.isel(time=t).values, np.nan)  # <-- terre masquee explicitement
    date_str = str(time.values[t])[:10]

    pcm = ax.contourf(lon[:, :], lat[:, :], salt, levels=niveaux, cmap=fond_cmap,
                        extend='both', transform=ccrs.PlateCarree(), zorder=1)
    c1 = ax.contour(lon[:, :], lat[:, :], salt, levels=niveaux[niveaux != 35],
               colors='black', linewidths=0.2, transform=ccrs.PlateCarree(), zorder=2)
    c2 = ax.contour(lon[:, :], lat[:, :], salt, levels=[35], colors='blue',
               linewidths=1.5, transform=ccrs.PlateCarree(), zorder=5)

    artistes_a_effacer = [pcm, c1, c2]

    titre.set_text(f"{date_str}")
    print(f'Frame : {date_str}')
    return pcm,

# Creation de l'animation
anim = animation.FuncAnimation(
    fig, mettre_a_jour, frames=list(indices_a_animer),
    interval=200, blit=False
)

# Sauvegarde en gif
sortie_gif = os.path.join(figures, 'isohaline_abs_animation_12_1PSU.gif')
anim.save(sortie_gif, writer='pillow', fps=5)
print(f"\nAnimation sauvegardee : {sortie_gif}")
