import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import os

grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
figures = '/lus/home/CT1/c1601279/aperez/figures'
simu = '/lus/store/CT1/c1601279/aperez/resultats/diff'
file = 'diff_12PSOURCE_1PSU_25deg.nc'

date = '2017-12-26'

# Plot
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

#Grille
g = xr.open_dataset(grid)
lon = g['lon_rho'][:, :]
lat = g['lat_rho'][:, :]
g.close()
print("Grid loaded.")

SWIO = (32, 50, -28, -12)

#Donnees
d = xr.open_dataset(os.path.join(simu, file))
salt = d.salt
time = d.time
d.close()
print('Data loaded.')

if np.datetime64(date) not in time.values.astype('datetime64[D]'):
    print(f'{date} not found in simulation.')
    date = np.datetime_as_string(time.values.astype('datetime64[D]')[0])
    print(f'Setting {date} as the new date.')

seuil_aberrant = 1e5
salt = salt[:, -1, :, :]
salt = salt.where(np.abs(salt) < seuil_aberrant, np.nan)
print('NaN values added')

#Palette
diff_cmap = mcolors.LinearSegmentedColormap.from_list(
    'diff_salinite',
    ['darkviolet', 'darkblue', 'blue', 'cyan', 'white']
)
diff_cmap.set_under('#4C0099')   # tout ce qui est <= -10
diff_cmap.set_over('white')     # tout ce qui est >= 0

norm_diff = mcolors.Normalize(vmin=-10, vmax=0)

# Niveau des isohalines
niveaux = np.arange(-10, 1, 1)

salt = salt.sel(time=date).mean(dim='time')
print('Date selected.')

print(f"Min/Max de la difference de salinite ce jour : "
      f"{float(np.nanmin(salt.values)):.2f} / {float(np.nanmax(salt.values)):.2f} psu")

#Figures
fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"{date} - Isohalines de surface (écart à la référence)")

ax.set_extent(SWIO)
ax.coastlines(resolution='50m')
ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='black', zorder=3)
ax.add_feature(ccrs.cartopy.feature.COASTLINE, linewidth=0.5, zorder=3)
ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, zorder=3)

land_color = ccrs.cartopy.feature.COLORS['land']

minor_islands = ccrs.cartopy.feature.NaturalEarthFeature(
    category='physical',
    name='minor_islands',
    scale='10m',
    facecolor=land_color,
    edgecolor='black'
)
ax.add_feature(minor_islands, zorder=3)

gl = ax.gridlines(crs=ccrs.PlateCarree(), **gridline_style)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {'color': 'k'}

#isohalines
pcm = ax.contourf(
    lon[:, :], lat[:, :], salt,
    levels=niveaux, cmap=diff_cmap, norm=norm_diff, extend='both',
    transform=ccrs.PlateCarree(), zorder=1
)
cb = plt.colorbar(pcm, ax=ax, label='ΔSSS [psu]', orientation='vertical')
cb.set_ticks(np.arange(-10, -0.2, 0.1)

# Lignes de niveaux
lignes_iso = ax.contour(
    lon[:, :], lat[:, :], salt,
    levels=niveaux, colors='black',
    linewidths=0.2, transform=ccrs.PlateCarree(), zorder=4
)

# Les 12 PSOURCES
for Isrc, Jsrc in psource_points:
    ax.plot(lon[Jsrc, Isrc], lat[Jsrc, Isrc], marker='o', color='red',
            markeredgecolor='black', markeredgewidth=1, markersize=3,
            transform=ccrs.PlateCarree(), zorder=6)

# Ligne de niveau -0.2
ax.contour(
    lon[:, :], lat[:, :], salt,
    levels=[-0.2], colors='red',
    linewidths=1.5, transform=ccrs.PlateCarree(), zorder=5
)

fig.tight_layout()
filename = os.path.join(figures, f"diff_isohaline_{file.replace('.nc','')}_{date}.png")
fig.savefig(filename, dpi=300)
print(f'{filename} saved.')
plt.show()
