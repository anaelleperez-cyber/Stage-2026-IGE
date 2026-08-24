import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import os

# chemins
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
figures = '/lus/home/CT1/c1601279/aperez/figures'
simu = '/lus/store/CT1/c1601279/aperez/resultats/exp_1PSOURCE_1PSU_25deg/'
file = '001swiose_avg.nc'

date_demandee = '2017-02-04'

SWIO = (33, 43, -26.5, -16)

gridline_style = {'draw_labels': True, 'linestyle': '--', 'linewidth': 0.5}
figsize = (8, 7)

os.makedirs(figures, exist_ok=True)

#grille et donnees

g = xr.open_dataset(grid)
lon = g['lon_rho'][:, :]
lat = g['lat_rho'][:, :]
g.close()
print("Grid loaded.")

d = xr.open_dataset(os.path.join(simu, file))
salt_full = d.salt
time = d.time
d.close()
print('Data loaded.')

fill_value = 9.96921e+36
salt_full = salt_full[:, -1, :, :]      # niveau de surface (dernier niveau sigma)
salt_full = salt_full.where((salt_full != fill_value), np.nan)
print('NaN values added')

# palette
fond_cmap = mcolors.LinearSegmentedColormap.from_list(
    'rainbow_20_35', ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
)
fond_cmap.set_over('firebrick')

niveaux = np.arange(20, 36, 1)

if np.datetime64(date_demandee) not in time.values.astype('datetime64[D]'):
    print(f'{date_demandee} not found in simulation.')
    date_demandee = np.datetime_as_string(time.values.astype('datetime64[D]')[0])
    print(f'Setting {date_demandee} as the new date.')

salt = salt_full.sel(time=date_demandee).mean(dim='time')
print(f'--- {date_demandee} : donnees selectionnees ---')

fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"{date_demandee}")

ax.set_extent(SWIO)
ax.coastlines(resolution='50m')
ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='black', zorder=3)
ax.add_feature(ccrs.cartopy.feature.COASTLINE, linewidth=0.5, zorder=3)
ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, zorder=3)

land_color = ccrs.cartopy.feature.COLORS['land']
minor_islands = ccrs.cartopy.feature.NaturalEarthFeature(
    category='physical', name='minor_islands', scale='10m',
    facecolor=land_color, edgecolor='black'
)
ax.add_feature(minor_islands, zorder=3)

gl = ax.gridlines(crs=ccrs.PlateCarree(), **gridline_style)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {'color': 'k'}

# Fond de carte colore
pcm = ax.contourf(
    lon[:, :], lat[:, :], salt,
    levels=niveaux, cmap=fond_cmap, extend='both',
    transform=ccrs.PlateCarree(), zorder=1
)
cb = plt.colorbar(pcm, ax=ax, label='Salinité [PSU]', orientation='vertical')
cb.set_ticks(niveaux)

# Isohalines (sauf le niveau 35, traite ensuite)
niveaux_lignes = niveaux[niveaux != 35]
lignes_iso = ax.contour(
    lon[:, :], lat[:, :], salt,
    levels=niveaux_lignes, colors='black',
    linewidths=0.2, transform=ccrs.PlateCarree(), zorder=2
)

# Ligne de niveau 35 PSU
lignes_35 = ax.contour(
    lon[:, :], lat[:, :], salt,
    levels=[35], colors='blue',
    linewidths=1.5, transform=ccrs.PlateCarree(), zorder=5
)
segments_35 = lignes_35.allsegs[0]
if len(segments_35) > 0:
    tous_points_35 = np.concatenate(segments_35, axis=0)
    dans_la_zone_35 = (tous_points_35[:, 0] >= 36.0) & (tous_points_35[:, 0] <= 37.0)
    points_zone_35 = tous_points_35[dans_la_zone_35]
    if len(points_zone_35) > 0:
        point_35 = tuple(points_zone_35[len(points_zone_35) // 2])
        ax.clabel(lignes_35, inline=True, fontsize=6, fmt='%d', manual=[point_35])

# Point source du Zambeze (Isrc=96, Jsrc=200) : cyan, entoure de noir
ax.plot(lon[200, 96], lat[200, 96], marker='o', color='cyan',
        markeredgecolor='black', markeredgewidth=1, markersize=8,
        transform=ccrs.PlateCarree(), zorder=6)

fig.tight_layout()
filename = os.path.join(figures, f"isohaline_{simu.split('/')[-2]}_{date_demandee}.png")
fig.savefig(filename, dpi=300)
print(f'{filename} saved.')
plt.show()
