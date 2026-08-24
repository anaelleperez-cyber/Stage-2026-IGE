import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cmocean
import os

grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
diff_file = '/lus/store/CT1/c1601279/aperez/resultats/diff/diff_1PSOURCE_1PSU_25deg.nc'
date = '2017-02-01'

SWIO = (33, 43, -26.5, -16)
seuil_aberrant = 1e5

# Seuil de vorticite anomale definissant "une perturbation significative"
seuil_vorticite = 1e-2   # s^-1

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
du = d['u'] #u et v sont deja des diff de vitesses
dv = d['v']
if 's_rho' in du.dims:
    du = du.isel(s_rho=-1)
    dv = dv.isel(s_rho=-1)

du = du.where(np.abs(du) < seuil_aberrant, np.nan)
dv = dv.where(np.abs(dv) < seuil_aberrant, np.nan)
time = du.time
d.close()

if np.datetime64(date) not in time.values.astype('datetime64[D]'):
    date = np.datetime_as_string(time.values.astype('datetime64[D]')[0])
    print(f'Date non trouvee, utilisation de {date} a la place.')

du_jour = du.sel(time=date).mean(dim='time').values
dv_jour = dv.sel(time=date).mean(dim='time').values

# Rotation vers les coordonnes geographiques (grille tordue dans CROCO)
du_geo = du_jour[:-1, :] * np.cos(angle[:-1, :-1]) - dv_jour[:, :-1] * np.sin(angle[:-1, :-1])
dv_geo = du_jour[:-1, :] * np.sin(angle[:-1, :-1]) + dv_jour[:, :-1] * np.cos(angle[:-1, :-1])

lon_c = lon[:-1, :-1]
lat_c = lat[:-1, :-1]
mask_c = mask_rho[:-1, :-1]
pm_c = pm[:-1, :-1]
pn_c = pn[:-1, :-1]

# Calcul de l'anomalie de vorticite dv/dx - du/dy
dv_dlon = np.gradient(dv_geo, axis=1) * pm_c
du_dlat = np.gradient(du_geo, axis=0) * pn_c
vorticite_anomalie = dv_dlon - du_dlat   # s^-1

# Exclusion de la terre + restriction a la zone SWIO
masque_zone = (lon_c >= SWIO[0]) & (lon_c <= SWIO[1]) & (lat_c >= SWIO[2]) & (lat_c <= SWIO[3])
vorticite_anomalie = np.where((mask_c == 1) & masque_zone, vorticite_anomalie, np.nan)

print(f"Vorticite anomale - min: {np.nanmin(vorticite_anomalie):.3e} s^-1, "
      f"max: {np.nanmax(vorticite_anomalie):.3e} s^-1")
print(f"Vorticite anomale - moyenne |.|: {np.nanmean(np.abs(vorticite_anomalie)):.3e} s^-1")

# ---------------------------------------------------------------
# EMPREINTE SPATIALE : surface ou |vorticite| depasse le seuil
# ---------------------------------------------------------------
cell_area_km2 = 1.0 / (pm_c * pn_c) / 1e6
masque_perturbation = np.abs(vorticite_anomalie) >= seuil_vorticite
empreinte_km2 = np.nansum(cell_area_km2[masque_perturbation])

print(f"\nEmpreinte de la perturbation de vorticite (|Δζ| >= {seuil_vorticite:.1e} s^-1) : "
      f"{empreinte_km2:,.1f} km2")

# ---------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"CROCO {date} - Anomalie de vorticité de surface")

ax.set_extent(SWIO)
ax.coastlines(resolution='50m')
ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='black', zorder=3)
ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, zorder=3)

max_abs = np.nanpercentile(np.abs(vorticite_anomalie), 98)
norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

pcm = ax.pcolormesh(lon_c, lat_c, vorticite_anomalie, cmap=cmocean.cm.curl, norm=norm,
                      transform=ccrs.PlateCarree(), zorder=1)
cb = plt.colorbar(pcm, ax=ax, label='Δζ [s⁻¹]', orientation='vertical')

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.4)
gl.top_labels = False
gl.right_labels = False

fig.tight_layout()
fig.savefig('vorticite_anomalie.png', dpi=300)
print("\nFigure sauvegardee : vorticite_anomalie.png")
plt.show()
