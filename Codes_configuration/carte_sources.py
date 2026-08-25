# Ce code permet de faire une carte représentant la zone SWIO, sa bathymetrie et les PSOURCES. Ces derniers sont entourés de cercles rouges, proportionnels à leurs débits.

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature


grid_file = '/lus/scratch/CT1/c1601279/aperez/SWIO/resultats_juillet/test2/swiose_grid.nc'

SWIO = (25, 69, -36, 7)  

# Les 12 PSOURCES (nom, Isrc, Jsrc, Dsrc, Qbar)

rivers = [
    ("Zambeze",         96, 200, 0,  3719.),
    ("Tsiribihina",    188, 192, 0,  -998.),
    ("Betsiboka",      208, 240, 1,   271.),
    ("Mangoky",        178, 171, 0,  -530.),
    ("Manambolo",      187, 197, 0,  -116.),
    ("Mahavavy",       238, 272, 1,   156.),
    ("Sambirano",      233, 265, 0,  -136.),
    ("Limpopo",         70, 119, 1,  -490.),
    ("Save",            83, 173, 0,   434.),
    ("Buzi et Pungoe",  79, 185, 0,   553.),
    ("Morondava",      186, 184, 0,   -51.),
    ("Linta",          185, 128, 0,   -11.),
]

# Chargement de la grille
g = xr.open_dataset(grid_file)
lon = g['lon_rho'].values
lat = g['lat_rho'].values
h = g['h'].values if 'h' in g else None   # bathymetrie, si disponible
mask = g['mask_rho'].values
g.close()

# Conversion des indices (Isrc, Jsrc) en coordonnees géographiques

rivers_geo = []
for name, Isrc, Jsrc, Dsrc, Qbar in rivers:
    lon_pt = lon[Jsrc, Isrc]
    lat_pt = lat[Jsrc, Isrc]
    rivers_geo.append((name, lon_pt, lat_pt, abs(Qbar)))

# Création de la carte

fig = plt.figure(figsize=(11, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(SWIO, crs=ccrs.PlateCarree())

# Representation de la bathymétrie si disponible  

if h is not None:
    h_masked = np.where(mask == 1, h, np.nan)
    bathy_cmap = plt.colormaps['GnBu']
    norm_bathy = mpl.colors.Normalize(vmin=25, vmax=np.nanpercentile(h_masked, 98))
    pcm = ax.pcolormesh(lon, lat, h_masked, cmap=bathy_cmap, norm=norm_bathy,
                          transform=ccrs.PlateCarree(), zorder=0)
    cb = plt.colorbar(pcm, ax=ax, label='Profondeur (m)', orientation='vertical',
                        fraction=0.04, pad=0.02)
    cb.ax.invert_yaxis()
else:
    ax.add_feature(cfeature.OCEAN, facecolor='#cfe8f3', zorder=0)

ax.add_feature(cfeature.LAND, facecolor='#808080', edgecolor='black', zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)

minor_islands = cfeature.NaturalEarthFeature(
    category='physical', name='minor_islands', scale='10m',
    facecolor='#808080', edgecolor='black'
)
ax.add_feature(minor_islands, zorder=2)

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.4, color='gray')
gl.top_labels = False
gl.right_labels = False

# Cercles proportionnels au débit : 
# La SURFACE du marqueur est directement proportionnelle au debit -> le rayon visuel est donc proportionnel a sqrt(debit)

qbar_values = [r[3] for r in rivers_geo]
max_qbar = max(qbar_values)

max_marker_area = 3000   # taille plafond
min_marker_area = 15     # taille plancher 

for name, lon_pt, lat_pt, qbar_abs in rivers_geo:
    area = max(min_marker_area, (qbar_abs / max_qbar) * max_marker_area)

    ax.scatter(lon_pt, lat_pt, s=area, facecolor='none',
               edgecolor='red', linewidth=1.8, transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(lon_pt, lat_pt, s=8, facecolor='white', edgecolor='black',
               linewidth=0.5, transform=ccrs.PlateCarree(), zorder=6)

ax.set_title("Points sources PSOURCE - SWIO\nCercles proportionnels au débit annuel (m³/s)",
              fontsize=13)

fig.tight_layout()
fig.savefig('psource_rivers_map.png', dpi=300, bbox_inches='tight')
print("Carte sauvegardee : psource_rivers_map.png")
plt.show()
