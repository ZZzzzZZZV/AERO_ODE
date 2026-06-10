"""
    RMSE computation and plotting utilities
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import xarray as xr

# ============================================
# Constants
# ============================================
# Multi-level variables (match model output channel order)
MULTI_LEVEL_VARS = [
    'geopotential', 'temperature', 'specific_humidity',
    'u_component_of_wind', 'v_component_of_wind'
]

# Display names for plotting
VAR_DISPLAY_NAMES = [
    'Geopotential Height', 'Temperature', 'Specific Humidity',
    'U Component', 'V Component'
]


# ============================================
# RMSE computation
# ============================================
def calculate_batch_rmse(pred, target, phy=None):
    """
    Compute RMSE for one batch
    
    Args:
        pred: Prediction tensor, shape (Batch, Channels, Lat, Lon, Time)
        target: Ground truth tensor, same shape
        phy: Physics tensor (optional), same shape
    
    Returns:
        rmse_pred: shape (Channels, Time)
        rmse_phy: shape (Channels, Time), None if phy is None
    """
    # Convert to tensor
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)
        target = torch.as_tensor(target)
        if phy is not None:
            phy = torch.as_tensor(phy)
    
    # RMSE: sqrt(mean((pred - target)^2)) over batch, lat, lon
    mse_pred = torch.mean((pred - target) ** 2, dim=(0, 2, 3))
    rmse_pred = torch.sqrt(mse_pred).cpu().detach().numpy()
    
    rmse_phy = None
    if phy is not None:
        mse_phy = torch.mean((phy - target) ** 2, dim=(0, 2, 3))
        rmse_phy = torch.sqrt(mse_phy).cpu().detach().numpy()
    
    return rmse_pred, rmse_phy


def calculate_rmse(pred_array, gt_array):
    """
    Compute RMSE for NumPy arrays
    
    Args:
        pred_array: shape (Time, Channels, Lat, Lon)
        gt_array: shape (Time, Channels, Lat, Lon)
    
    Returns:
        rmse: shape (Channels, Time)
    """
    num_channels = pred_array.shape[1]
    gt_upper = gt_array[:, :num_channels, :, :]
    
    # MSE averaged over spatial dims
    mse = np.mean((pred_array - gt_upper) ** 2, axis=(2, 3))
    rmse = np.sqrt(mse)  # (Time, Channels)
    
    return rmse.T  # (Channels, Time)


# ============================================
# Plotting
# ============================================
def plot_rmse_grid(rmse_pred, rmse_phy, target_levels, 
                   var_names=None, output_path="plots/rmse.png"):
    """
    Plot RMSE grid (rows=variables, cols=levels)
    
    Args:
        rmse_pred: shape (Channels, Time)
        rmse_phy: shape (Channels, Time), may be None
        target_levels: Level list, e.g. [50, 500, 850, 1000]
        var_names: Variable display names
        output_path: Output path
    """
    if var_names is None:
        var_names = VAR_DISPLAY_NAMES
    
    num_vars = len(var_names)
    num_levels = len(target_levels)
    time_steps = rmse_pred.shape[1]
    times = np.arange(time_steps)
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=num_vars, ncols=num_levels,
        figsize=(4 * num_levels, 3 * num_vars),
        squeeze=False
    )
    
    for v_idx, var_name in enumerate(var_names):
        for l_idx, level in enumerate(target_levels):
            ax = axes[v_idx, l_idx]
            ch_idx = v_idx * num_levels + l_idx
            
            if ch_idx >= rmse_pred.shape[0]:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                continue
            
            # Plot curves
            ax.plot(times, rmse_pred[ch_idx], 
                   label='Prediction', color='blue', marker='o', markersize=2)
            
            if rmse_phy is not None:
                ax.plot(times, rmse_phy[ch_idx],
                       label='Physics', color='red', linestyle='--', marker='x', markersize=2)
            
            # Style
            ax.set_title(f'{var_name}\n@{level} hPa', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            if l_idx == 0:
                ax.set_ylabel('RMSE')
            if v_idx == num_vars - 1:
                ax.set_xlabel('Time (h)')
            if v_idx == 0 and l_idx == num_levels - 1:
                ax.legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved: {output_path}")
    plt.close()


def calculate_rmse_and_plot(t_pred, t_phy, t_target, target_levels, output_dir="plots"):
    """
    Compute RMSE and plot in one call
    
    Args:
        t_pred, t_phy, t_target: shape (Batch, Channels, Lat, Lon, Time)
        target_levels: Level list
        output_dir: Output directory
    """
    rmse_pred, rmse_phy = calculate_batch_rmse(t_pred, t_target, t_phy)
    
    output_path = os.path.join(output_dir, "rmse_analysis.png")
    plot_rmse_grid(rmse_pred, rmse_phy, target_levels, output_path=output_path)
    
    return rmse_pred, rmse_phy


# Legacy interface
def plot_rmse_metrics(rmse_pred, rmse_phy, target_levels, output_dir="plots"):
    """Legacy interface"""
    output_path = os.path.join(output_dir, "rmse_analysis.png")
    plot_rmse_grid(rmse_pred, rmse_phy, target_levels, output_path=output_path)


def plot_rmse_results(rmse_pred, rmse_phy, target_levels, name="rmse.png"):
    """Legacy interface"""
    plot_rmse_grid(rmse_pred, rmse_phy, target_levels, output_path=f"./plot/{name}")


# ============================================
# ERA5 interpolation
# ============================================
def interpolate_prediction(ds, lat_grid, lon_grid, target_levels):
    """
    Interpolate xarray Dataset to target grid
    
    Args:
        ds: xarray Dataset (global ERA5 field)
        lat_grid: Target latitude grid, shape (H, W)
        lon_grid: Target longitude grid, shape (H, W)  
        target_levels: Target pressure levels
    
    Returns:
        pred_array: shape (Time, Channels, H, W)
    """
    # Convert coordinates to 0-360 range
    lon_360 = (lon_grid + 360) % 360
    lat_da = xr.DataArray(lat_grid, dims=("y", "x"))
    lon_da = xr.DataArray(lon_360, dims=("y", "x"))
    
    # Select levels
    ds_sel = ds.sel(level=target_levels, method='nearest')
    
    # Unit conversion: geopotential -> geopotential height
    if 'geopotential' in ds_sel:
        ds_sel['geopotential'] = ds_sel['geopotential'] / 9.80665
    
    # Interpolate
    print("Interpolating to regional grid...")
    ds_interp = ds_sel.interp(longitude=lon_da, latitude=lat_da)
    
    # Extract data
    data_list = []
    for var in MULTI_LEVEL_VARS:
        if var not in ds_interp:
            print(f"Warning: {var} not found")
            continue
        
        da = ds_interp[var]
        # Transpose to (time, level, y, x)
        dims = list(da.dims)
        t_dim = [d for d in dims if 'time' in d.lower()][0]
        l_dim = [d for d in dims if 'level' in d.lower()][0]
        da = da.transpose(t_dim, l_dim, 'y', 'x')
        data_list.append(da.values)
    
    # Concatenate to (Time, num_vars * num_levels, H, W)
    pred_array = np.concatenate(data_list, axis=1)
    print(f"Interpolated shape: {pred_array.shape}")
    
    return pred_array
