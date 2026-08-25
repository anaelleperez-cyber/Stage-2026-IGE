# Code pour créer une carte représentant la zone et sa bathymetrie

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os


grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'

SWIO = (25, 69, -36, 7)
pas_grille = 10           # 1 ligne de grille affichee tous les N points


lon = g['lon_rho'].values
lat = g['lat_rho'].values
h = g['h'].values           # bathymetrie [m], toujours positive (profondeur)
mask_rho = g['mask_rho'].values
g.close()

# Bathymetrie affichee en negatif (convention -5000 m = fond), terre masquee
bathy = np.where(mask_rho == 1, -h, np.nan)

# Creation de la figure 

fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent(SWIO)
ax.coastlines(resolution='50m', linewidth=0.5)
ax.add_feature(ccrs.cartopy.feature.LAND, facecolor='#EAEAEA', edgecolor='black', zorder=3)

# Fond colore de bathymetrie
pcm = ax.pcolormesh(lon, lat, bathy, cmap='YlGnBu_r', vmin=-5500, vmax=0,
                      transform=ccrs.PlateCarree(), zorder=1)
cb = plt.colorbar(pcm, ax=ax, label='Bathymetry (m)', orientation='vertical',
                    fraction=0.04, pad=0.03)

# Isobaths, avec etiquettes
niveaux_isobaths = [-4000, -100]
couleurs_isobaths = ['white', 'red']
cs = ax.contour(lon, lat, bathy, levels=niveaux_isobaths, colors=couleurs_isobaths,
                  linewidths=0.6, transform=ccrs.PlateCarree(), zorder=2)
ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

# Lignes de la grille numerique affichees tous les N points
lon_grille = lon[::pas_grille, ::pas_grille]
lat_grille = lat[::pas_grille, ::pas_grille]
for i in range(lon_grille.shape[0]):
    ax.plot(lon_grille[i, :], lat_grille[i, :], color='gray', linewidth=0.2,
            transform=ccrs.PlateCarree(), zorder=1.5)
for j in range(lon_grille.shape[1]):
    ax.plot(lon_grille[:, j], lat_grille[:, j], color='gray', linewidth=0.2,
            transform=ccrs.PlateCarree(), zorder=1.5)

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.3, color='gray')
gl.top_labels = False
gl.right_labels = False

fig.tight_layout()
fig.savefig('bathymetrie_grille.png', dpi=300)
print("Figure sauvegardee : bathymetrie_grille.png")
plt.show()
