# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python [conda env:.conda-moje] *
#     language: python
#     name: conda-env-.conda-moje-py
# ---

# # Load libraries

import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path
import numpy as np


# # Functions

# +
def get_mean_std(da):
    return da.mean(['time','lon'], keep_attrs = True), da.std(['time','lon','year_ens'], keep_attrs = True)


def preprocess(da):
    da = da.set_coords(['lev','lon','lat'])
    da = da.rename({'levs': 'lev', 'lats': 'lat', 'lons': 'lon'})
    da['lev'].attrs['long_name'] = 'altitude'
    da['lat'].attrs['long_name'] = 'latitude'
    da['lon'].attrs['long_name'] = 'longitude'
    
    return da

def ttest_ind_wrap(a,b):
    _, pv = ttest_ind(a,b, axis = 0, equal_var = False)
    return pv


# -

# # Get list of input files

root_path = Path('/projekt5/hochatm/muam_mstober/')
in_folders = [f for f in root_path.iterdir() if f.is_dir()] # get all folder in root_path

# +
# define ENSO years
el_year_ls = [1983,1992,1998,2003,2010]
la_year_ls = [1989,1999,2000,2008,2013]

sel_month = 'Jan' 
# get El Nino and La Nina directories
el_in_folders = list(filter(lambda x: str(x)[-8:] in [f'{year}_{sel_month}' for year in el_year_ls], in_folders)) 
la_in_folders = list(filter(lambda x: str(x)[-8:] in [f'{year}_{sel_month}' for year in la_year_ls], in_folders))
# and particular input files
add_nc_suffix = lambda x: x / 'nc' / 'muam_Jan330.nc'
el_in_files = list(map(add_nc_suffix, el_in_folders))
la_in_files = list(map(add_nc_suffix, la_in_folders))
# -

# # Load data

sel_var = 'tem'
# concatenate files along 'year_ens' dimension
ds_el = xr.open_mfdataset(el_in_files, concat_dim = 'year_ens', parallel = True, combine='nested', preprocess=preprocess)
ds_la = xr.open_mfdataset(la_in_files, concat_dim = 'year_ens', parallel = True, combine='nested', preprocess=preprocess)
# mean and std calculation
mean_el, std_el = get_mean_std(ds_el[sel_var])
mean_la, std_la = get_mean_std(ds_la[sel_var])

# # P-values calculation

# vectorized ttest_ind calculation
da_pv = xr.apply_ufunc(ttest_ind_wrap, mean_el, mean_la, \
               vectorize = True, dask = 'allowed', \
               input_core_dims=[['year_ens'],['year_ens']], \
               output_core_dims=[[]])

# # Plotting

mean_el.std('year_ens').plot.contourf(levels = 11)

# +
mean_el_plot = mean_el.mean('year_ens', keep_attrs = True)
mean_la_plot = mean_la.mean('year_ens', keep_attrs = True)
diff = mean_el_plot-mean_la_plot
diff.attrs['units'] = 'K'
diff.attrs['long_name'] = 'difference'


plt.rcParams.update({'font.size': 20})
c_pad = 0.2
c_shrink = 0.99
c_aspect = 20
cf_levels = [100.,150.,200.,240.,260.,300.,400.,500.,600.,700.]
c_levels = np.arange(0,20,1)
fs = 16

fig, axes = plt.subplots(ncols = 3, figsize = (16,6), sharey = True)

ax = axes[0]
p = mean_el_plot.plot.contourf(ax = ax, levels = cf_levels, add_colorbar = False)
mean_el.std('year_ens').plot.contour(levels = c_levels, colors='k', ax = ax)
ax.set_title(f'average El Nino in {sel_month}', fontsize = fs)

ax = axes[1]
cbar = fig.colorbar(p, ax=axes[:2],  location='bottom', \
                    ticks=cf_levels, extend = 'both', spacing='proportional', \
                    pad = c_pad, shrink=c_shrink, aspect = c_aspect)
cbar.set_label('temperature [K]')
cbar.ax.tick_params(labelsize=12) 

mean_la_plot.plot.contourf(ax = ax, add_colorbar = False, levels = cf_levels, extend = 'both')
mean_la.std('year_ens').plot.contour(levels = c_levels, colors='k', ax = ax)
ax.set_ylabel('')
ax.set_title(f'average La Nina in {sel_month}', fontsize = fs)

ax = axes[2]
diff.plot(ax = ax, robust = True, cbar_kwargs={'orientation': 'horizontal', 'pad': c_pad, \
                                               'aspect': c_aspect/2., "shrink": c_shrink})
# hatching to display pvalues smaller than 0.05 and 0.01, respectively
plot_kwargs2 = dict(levels = [0,0.05], hatches = ['\\\\',None], colors='none', add_colorbar=False)
da_pv.plot.contourf(ax = ax, **plot_kwargs2)
plot_kwargs2['levels'] = [0,0.01]
plot_kwargs2['hatches'] = ['////',None]
da_pv.plot.contourf(ax = ax, **plot_kwargs2)
ax.set_title(f'Diff between El-Nino and La-Nina in {sel_month}', fontsize = fs)
ax.set_ylabel('')
