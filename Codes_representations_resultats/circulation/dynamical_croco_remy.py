# LIBRAIRIES
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import metpy.calc as mpcalc
import cmocean
import cmcrameri
import cartopy.crs as ccrs
import os



# PARAMETERS
# Path
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
simu = '/lus/store/CT1/c1601279/aperez/resultats/diff'
file = 'diff_1PSOURCE_1PSU_25deg.nc'
figures = '/lus/store/CT1/c1601279/aperez/figures'

# Selection
date = '2017-02-11'
SWIO = (25, 69, -36, 7)

# Plot
gridline_style = {'draw_labels': True, 'linestyle': '--', 'linewidth': 0.5}
figsize = (20, 6)
velo_cmap = cmocean.cm.balance
vort_cmap = cmocean.cm.curl
ener_cmap = cmocean.cm.speed


g = xr.open_dataset(grid)
lon = g['lon_rho'][:, :]
lat = g['lat_rho'][:, :]
msk = g['mask_rho'][:, :]
pm = g['pm'][:,:]
pn = g['pn'][:,:]
msk_inv = np.where(msk == 0, msk, np.nan)
h = g['h'][:, :]
angle = g['angle'][:, :]
g.close()
print("Grid loaded.")

d = xr.open_dataset(os.path.join(simu, file))
u = d.u[:,-1,:,:]
v = d.v[:,-1,:,:]
w = d.w[:,-1,:,:]
time = d.time
d.close()
print('Data loaded.')



if np.datetime64(date) not in time.values.astype('datetime64[D]'):
    print(f'{date} not found in simulation.')
    date = np.datetime_as_string(time.values.astype('datetime64[D]')[0])
    print(f'Setting {date} as the new date.')

years = np.unique(time.astype('datetime64[Y]')).astype('datetime64[Y]')
year = np.datetime64(date).astype('datetime64[Y]')

u = u.sel(time=str(year))
v = v.sel(time=str(year))
w = w.sel(time=str(year))

fill_value = 9.96921e+36
u = u.where((u != fill_value), np.nan)
v = v.where((v != fill_value), np.nan)
w = w.where((w != fill_value), np.nan)
print('NaN values added')



# Moyenne annuelle
u_yr = u.mean(dim='time')
v_yr = v.mean(dim='time')
w_yr = w.mean(dim='time')
print("Time mean calculated")

u = u.sel(time=date).mean(dim='time')
v = v.sel(time=date).mean(dim='time')
w = w.sel(time=date).mean(dim='time')
print("Data sliced")

# Vitesse turbulente
ut = (u - u_yr).data[:,:]
vt = (v - v_yr).data[:,:]
wt = (w - w_yr).data[:,:]
print("Turbulent velocity calculated")

u = u.data
v = v.data
w = w.data

u_yr = u_yr.data
v_yr = v_yr.data
w_yr = w_yr.data

angle = angle.data

u_geo = u[:-1,:] * np.cos(angle[:-1,:-1]) - v[:,:-1] * np.sin(angle[:-1,:-1])
v_geo = u[:-1,:] * np.sin(angle[:-1,:-1]) + v[:,:-1] * np.cos(angle[:-1,:-1])
w_geo = w[:-1,:-1]

ut_geo = ut[:-1,:] * np.cos(angle[:-1,:-1]) - vt[:,:-1] * np.sin(angle[:-1,:-1])
vt_geo = ut[:-1,:] * np.sin(angle[:-1,:-1]) + vt[:,:-1] * np.cos(angle[:-1,:-1])
wt_geo = wt[:-1,:-1]

u_yr_geo = u_yr[:-1,:] * np.cos(angle[:-1,:-1]) - v_yr[:,:-1] * np.sin(angle[:-1,:-1])
v_yr_geo = u_yr[:-1,:] * np.sin(angle[:-1,:-1]) + v_yr[:,:-1] * np.cos(angle[:-1,:-1])
w_yr_geo = w_yr[:-1,:-1]
print("Velocity transformed")

velocity = np.sqrt(u_geo ** 2 + v_geo ** 2)
velocity_t = np.sqrt(ut_geo ** 2 + vt_geo ** 2)
velocity_yr = np.sqrt(u_yr_geo ** 2 + v_yr_geo ** 2)
print('Velocity calculated')

# Calculate derivatives
dv_dlon = np.gradient(v_geo, axis=1) * pm[:-1,:-1]
du_dlat = np.gradient(u_geo, axis=0) * pn[:-1,:-1]

# Calculate vorticity
vorticity = dv_dlon - du_dlat

dvt_dlon = np.gradient(vt_geo, axis=1) * pm[:-1,:-1]
dut_dlat = np.gradient(ut_geo, axis=0) * pn[:-1,:-1]

vorticity_t = dvt_dlon - dut_dlat

dv_yr_dlon = np.gradient(v_yr_geo, axis=1) * pm[:-1,:-1]
du_yr_dlat = np.gradient(u_yr_geo, axis=0) * pn[:-1,:-1]

vorticity_yr = dv_yr_dlon - du_yr_dlat
print('Vorticity calculated')

KE = 1 / 2 * ( u_geo ** 2 + v_geo ** 2 + w_geo ** 2)
MKE = 1 / 2 * ( u_yr_geo ** 2 + v_yr_geo ** 2 + w_yr_geo ** 2)
EKE = 1 / 2 * ( ut_geo ** 2 + vt_geo ** 2 + wt_geo ** 2)
print('Energy calculated')



fig, axs = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"CROCO {date}")

for ax in axs:
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

# Velocity
ax = axs[0]
ax.set_title("Velocity")

a = -1.5
b = 0.2
norm = mpl.colors.LogNorm(10**a, 10**b)
data = velocity
cmap = velo_cmap
label = r'$v$ [m/s]'

pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')

# Vorticity
ax = axs[1]
ax.set_title("Vorticity")

norm = mpl.colors.Normalize(vmin=-5e-5, vmax=5e-5)
data = vorticity
cmap = vort_cmap
label = r'$\omega$ [h⁻¹]'
pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')
ticks = cb.get_ticks()[1:-1]
cb.set_ticks(ticks)
cb.set_ticklabels([f'{tick:.0e}' for tick in ticks])

# Energy
ax = axs[2]
ax.set_title("Energy")

a = int(np.log10(np.nanmax(KE)))
b = a-2.5
norm = mpl.colors.LogNorm(vmin=10**b, vmax=10**a)
data = KE
cmap = ener_cmap
label = r'$KE$ [m².s⁻²]'

pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')

fig.tight_layout()
filename = os.path.join(figures, f"dynamical_croco_{simu.split('/')[-2]}_{date}.png")
fig.savefig(filename, dpi=300)
print(f'{filename} saved.')
plt.show()



fig, axs = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"CROCO")

for ax in axs:
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


# Velocity
ax = axs[0]
ax.set_title("Velocity")

a = -1.5
b = 0.2
norm = mpl.colors.LogNorm(10**a, 10**b)
data = velocity_yr
cmap = velo_cmap
label = r'$v$ [m/s]'

pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')

# Vorticity
ax = axs[1]
ax.set_title("Vorticity")

norm = mpl.colors.Normalize(vmin=-5e-5, vmax=5e-5)
data = vorticity_yr
cmap = vort_cmap
label = r'$\omega$ [h⁻¹]'
pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')
ticks = cb.get_ticks()[1:-1]
cb.set_ticks(ticks)
cb.set_ticklabels([f'{tick:.0e}' for tick in ticks])

# Energy
ax = axs[2]
ax.set_title("Energy")

a = int(np.log10(np.nanmax(MKE)))
b = a-2.5
norm = mpl.colors.LogNorm(vmin=10**b, vmax=10**a)
data = MKE
cmap = ener_cmap
label = r'$KE$ [m².s⁻²]'

pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')

fig.tight_layout()
filename = os.path.join(figures, f"dynamical_croco_{simu.split('/')[-2]}_{year}_mean.png")
fig.savefig(filename, dpi=300)
print(f'{filename} saved.')
plt.show()



fig, axs = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"CROCO Turbulent {date}")

for ax in axs:
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


# Velocity
ax = axs[0]
ax.set_title("Velocity")

a = -1.5
b = 0.2
norm = mpl.colors.LogNorm(10**a, 10**b)
data = velocity_t
cmap = velo_cmap
label = r'$v$ [m/s]'

pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')

# Vorticity
ax = axs[1]
ax.set_title("Vorticity")

norm = mpl.colors.Normalize(vmin=-5e-5, vmax=5e-5)
data = vorticity_t
cmap = vort_cmap
label = r'$\omega$ [h⁻¹]'
pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')
ticks = cb.get_ticks()[1:-1]
cb.set_ticks(ticks)
cb.set_ticklabels([f'{tick:.0e}' for tick in ticks])

# Energy
ax = axs[2]
ax.set_title("Energy")

a = int(np.log10(np.nanmax(EKE)))
b = a-2.5
norm = mpl.colors.LogNorm(vmin=10**b, vmax=10**a)
data = EKE
cmap = ener_cmap
label = r'$KE$ [m².s⁻²]'

pcm = ax.pcolormesh(lon[:, :], lat[:, :], data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cb = plt.colorbar(pcm, ax=ax, label=label, orientation='vertical')

fig.tight_layout()
filename = os.path.join(figures, f"dynamical_croco_{simu.split('/')[-2]}_{date}_turbulent.png")
fig.savefig(filename, dpi=300)
print(f'{filename} saved.')
plt.show()
