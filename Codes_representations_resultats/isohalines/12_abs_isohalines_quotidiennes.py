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
simu = '/lus/store/CT1/c1601279/aperez/resultats/exp_12PSOURCE_1PSU_25deg/'
file = '001swiose_avg.nc'

SWIO = (32, 50, -28, -12)

date = '2017-02-01'

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

#grille et donnees

g = xr.open_dataset(grid)
lon = g['lon_rho'][:, :]
lat = g['lat_rho'][:, :]
g.close()
print("Grid loaded.")

d = xr.open_dataset(os.path.join(simu, file))
salt = d.salt
time = d.time
d.close()
print('Data loaded.')

if np.datetime64(date) not in time.values.astype('datetime64[D]'):
    print(f'{date} not found in simulation.')
    date = np.datetime_as_string(time.values.astype('datetime64[D]')[0])
    print(f'Setting {date} as the new date.')

fill_value = 9.96921e+36
salt = salt[:, -1, :, :]      # niveau de surface (dernier niveau sigma)
salt = salt.where((salt != fill_value), np.nan)
print('NaN values added')

# palette
fond_cmap = mcolors.LinearSegmentedColormap.from_list(
    'rainbow_20_35', ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
)
fond_cmap.set_over('firebrick')

niveaux = np.arange(20, 36, 1)

salt = salt.sel(time=date).mean(dim='time')
print('Date selected.')

fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"{date} - Isohalines de surface")

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

# Les 12 PSOURCE : cyan, entoures de noir
for Isrc, Jsrc in psource_points:
    ax.plot(lon[Jsrc, Isrc], lat[Jsrc, Isrc], marker='o', color='cyan',
            markeredgecolor='black', markeredgewidth=1, markersize=8,
            transform=ccrs.PlateCarree(), zorder=6)

fig.tight_layout()
filename = os.path.join(figures, f"isohaline_{simu.split('/')[-2]}_{date}.png")
fig.savefig(filename, dpi=300)
print(f'{filename} saved.')
plt.show()
