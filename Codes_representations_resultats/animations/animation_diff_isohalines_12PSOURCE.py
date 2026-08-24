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
simu = '/lus/store/CT1/c1601279/aperez/resultats/diff'
file = 'diff_12PSOURCE_1PSU_25deg.nc'

SWIO = (32, 50, -28, -12)

gridline_style = {'draw_labels': True, 'linestyle': '--', 'linewidth': 0.5}
figsize = (8, 7)

os.makedirs(figures, exist_ok=True)

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

# Pas entre chaque image de l'animation (1 = tous les jours, 5 = un jour sur 5...)
pas_animation = 5

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

seuil_aberrant = 1e5
salt_full = salt_full.where(np.abs(salt_full) < seuil_aberrant, np.nan)   # <-- CORRECTION : salt_full, pas salt
print('NaN values added')

time = salt_full.time
d.close()

n_days = len(time)
indices_a_animer = list(range(0, n_days, pas_animation))   # <-- list() ajoute directement, coherence avec le reste
print(f"{len(indices_a_animer)} images dans l'animation (1 tous les {pas_animation} jours).")

#Palette
diff_cmap = mcolors.LinearSegmentedColormap.from_list(
    'diff_salinite',
    ['darkviolet', 'darkblue', 'blue', 'cyan', 'white']
)
diff_cmap.set_under('#4C0099')   # tout ce qui est <= -10
diff_cmap.set_over('white')     # tout ce qui est >= 0

norm_diff = mcolors.Normalize(vmin=-10, vmax=0)

# Niveau des isohalines
niveaux = np.arange(-10, -0.2, 0.1)

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
pcm = ax.contourf(lon[:, :], lat[:, :], salt0, levels=niveaux, cmap=diff_cmap, norm=norm_diff,
                    extend='both', transform=ccrs.PlateCarree(), zorder=1)   # <-- CORRECTION : diff_cmap, pas fond_cmap ; norm ajoute pour coherence avec la boucle

# Colorbar CREEE UNE SEULE FOIS ICI, jamais recreee dans mettre_a_jour()
cb = plt.colorbar(pcm, ax=ax, label='ΔSSS [psu]', orientation='vertical')
cb.set_ticks(np.arange(-10, 1, 2))


titre = ax.set_title("")

# Liste des artistes à supprimer à chaque frame (colorbar RETIREE de cette liste)
artistes_a_effacer = [pcm]

#  Les 12 PSOURCES
for Isrc, Jsrc in psource_points:
    ax.plot(lon[Jsrc, Isrc], lat[Jsrc, Isrc], marker='o', color='red',
            markeredgecolor='red', markeredgewidth=1, markersize=5,
            transform=ccrs.PlateCarree(), zorder=6)

def mettre_a_jour(t):
    global pcm, artistes_a_effacer

    for artiste in artistes_a_effacer:
        artiste.remove()
    artistes_a_effacer = []

    salt = np.where(mask_rho == 1, salt_full.isel(time=t).values, np.nan)
    date_str = str(time.values[t])[:10]

    #isohalines
    pcm = ax.contourf(
        lon[:, :], lat[:, :], salt,
        levels=niveaux, cmap=diff_cmap, norm=norm_diff, extend='both',
        transform=ccrs.PlateCarree(), zorder=1
    )

    # Lignes de niveaux
    c1 = ax.contour(
        lon[:, :], lat[:, :], salt,
        levels=niveaux, colors='black',
        linewidths=0.2, transform=ccrs.PlateCarree(), zorder=4
    )

    # Ligne de niveau -1 PSU : plus epaisse et rouge, par-dessus le reste
    c2 = ax.contour(
        lon[:, :], lat[:, :], salt,
        levels=[-0.2], colors='red',
        linewidths=1.5, transform=ccrs.PlateCarree(), zorder=5
    )

    artistes_a_effacer = [pcm, c1, c2]

    titre.set_text(f" {date_str} ")
    print(f'Frame : {date_str}')
    return pcm,

# Creation de l'animation
anim = animation.FuncAnimation(
    fig, mettre_a_jour, frames=indices_a_animer,
    interval=200, blit=False
)

# Sauvegarde en gif
sortie_gif = os.path.join(figures, 'isohaline_animation_diff_12_1PSU.gif')
anim.save(sortie_gif, writer='pillow', fps=5)
print(f"\nAnimation sauvegardee : {sortie_gif}")
