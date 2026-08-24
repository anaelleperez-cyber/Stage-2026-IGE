import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import os

# Chemins
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
simu = '/lus/store/CT1/c1601279/aperez/resultats/exp_12PSOURCE_1PSU_25deg/'
file = '001swiose_avg.nc'

SWIO = (32, 50, -28, -12)

seuil_panache = 34.0
salinite_ambiante = 35.0

# Pas entre chaque image de l'animation (en nombre de jours)
pas_animation = 5

# Grille et données
g = xr.open_dataset(grid)
lon = g['lon_rho'].values
lat = g['lat_rho'].values
pm = g['pm'].values
pn = g['pn'].values
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
indices_a_animer = list(range(0, n_days, pas_animation))
print(f"{len(indices_a_animer)} images dans l'animation (1 tous les {pas_animation} jours).")

masque_zone = (lon >= SWIO[0]) & (lon <= SWIO[1]) & (lat >= SWIO[2]) & (lat <= SWIO[3])


def calculer_gradient_jour(t):
    champ = salt_full.isel(time=t).values
    champ = np.where(mask_rho == 1, champ, np.nan)
    champ = np.where(masque_zone, champ, np.nan)

    dS_dx = np.gradient(champ, axis=1) * pm
    dS_dy = np.gradient(champ, axis=0) * pn
    gradient_magnitude_km = np.sqrt(dS_dx**2 + dS_dy**2) * 1000
    gradient_magnitude_km = np.where(mask_rho == 1, gradient_magnitude_km, np.nan)

    return gradient_magnitude_km

# FIGURE DE BASE
fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent(SWIO)
ax.coastlines(resolution='50m')
ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, zorder=3)

# Terre dessinee directement depuis mask_rho 
terre_affichee = np.ma.masked_where(mask_rho == 1, mask_rho)
ax.pcolormesh(lon, lat, terre_affichee, cmap=mcolors.ListedColormap(['#DEB887']),
              transform=ccrs.PlateCarree(), zorder=4)

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.4)
gl.top_labels = False
gl.right_labels = False

cmap = plt.colormaps['inferno']
norm = mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=0.1)

# Premiere image (jour 0), pour initialiser la colorbar
gradient0 = calculer_gradient_jour(indices_a_animer[0])
pcm = ax.pcolormesh(lon, lat, gradient0, cmap=cmap, norm=norm,
                      transform=ccrs.PlateCarree(), zorder=1)
cb = plt.colorbar(pcm, ax=ax, label='Gradient salinité de surface [psu/km]', orientation='vertical')

titre = ax.set_title("")

artistes_a_effacer = [pcm] # ce qui est supprimé à chaque frame


# FONCTION APPELEE POUR CHAQUE FRAME DE L'ANIMATION
def mettre_a_jour(t):
    global pcm, artistes_a_effacer # pcm et artistes_a_effacer sont definis en dehors de la fonction

    for artiste in artistes_a_effacer:
        artiste.remove()
    artistes_a_effacer = []

    gradient_magnitude_km = calculer_gradient_jour(t)
    date_str = str(time.values[t])[:10]

    pcm = ax.pcolormesh(lon, lat, gradient_magnitude_km, cmap=cmap, norm=norm,
                          transform=ccrs.PlateCarree(), zorder=1)
    artistes_a_effacer = [pcm]

    titre.set_text(f"{date_str}")
    print(f'Frame : {date_str}')
    return pcm,


# Creation de l'animation
anim = animation.FuncAnimation(
    fig, mettre_a_jour, frames=indices_a_animer,
    interval=200, blit=False
)

# Sauvegarde en gif
sortie_gif = 'gradient_salinite_animation_12_1PSU.gif'
anim.save(sortie_gif, writer='pillow', fps=5)
print(f"\nAnimation sauvegardee : {sortie_gif}")
