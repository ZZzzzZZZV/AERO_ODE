import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp

_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
if _cache:
    try:
        jax.config.update("jax_compilation_cache_dir", _cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    except Exception:
        pass
from jax.scipy.ndimage import map_coordinates
import neuralgcm
import torch
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

MULTI_LEVEL_VARS = [
    'geopotential', 'temperature', 'specific_humidity',
    'u_component_of_wind', 'v_component_of_wind',
    'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content'
]

SINGLE_LEVEL_VARS = [
    '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
    'mean_sea_level_pressure', 'sea_surface_temperature', 'sea_ice_cover'
]

def stack_dicts(dicts_list):
    return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *dicts_list)

def process_predictions(data_dict, level_indices, interp_indices=None):

    def interpolate_spatial(tensor, indices):
        if indices is None:
            return tensor

        B, T, H, W = tensor.shape
        reshaped = tensor.reshape(B * T, H, W)

        lon_idx, lat_idx = indices[0], indices[1]
        lon_idx = lon_idx % H
        lat_idx = jnp.clip(lat_idx, 0.0, W - 1.0)
        fixed_idx = jnp.stack([lon_idx, lat_idx], axis=0)

        interp_fn = lambda img: map_coordinates(img, fixed_idx, order=1, mode='wrap')
        out = jax.vmap(interp_fn)(reshaped)

        return out.reshape(B, T, indices.shape[1], indices.shape[2])

    def process_multi_level(arrays, level_idx, interp_idx):
        if not arrays:
            return None

        stacked = jnp.stack(arrays, axis=0)
        selected = stacked[:, :, :, level_idx, :, :]

        V, B, T, L, H, W = selected.shape
        permuted = jnp.transpose(selected, (0, 3, 1, 2, 4, 5))
        flat = permuted.reshape(-1, B, T, H, W)

        channels = [interpolate_spatial(flat[i], interp_idx) for i in range(flat.shape[0])]
        return jnp.stack(channels, axis=0)

    def process_single_level(arrays, interp_idx):
        if not arrays:
            return None

        cleaned = []
        for arr in arrays:
            if arr.ndim == 5 and arr.shape[2] == 1:
                arr = arr.squeeze(axis=2)
            if arr.ndim == 4:
                cleaned.append(arr)

        if not cleaned:
            return None

        stacked = jnp.stack(cleaned, axis=0)
        channels = [interpolate_spatial(stacked[i], interp_idx) for i in range(stacked.shape[0])]
        return jnp.stack(channels, axis=0)

    def process_dataset(prefix=''):

        multi_arrays = [data_dict[prefix + v] for v in MULTI_LEVEL_VARS if prefix + v in data_dict]
        multi = process_multi_level(multi_arrays, level_indices, interp_indices)

        single_arrays = [data_dict[prefix + v] for v in SINGLE_LEVEL_VARS if prefix + v in data_dict]
        single = process_single_level(single_arrays, interp_indices)

        parts = [x for x in [multi, single] if x is not None]
        if not parts:
            return None

        combined = jnp.concatenate(parts, axis=0)
        return jnp.transpose(combined, (1, 0, 3, 4, 2))

    return process_dataset(''), process_dataset('_physics_')

class NeuralGCMInference:

    def __init__(self, checkpoint_path: str, inner_steps: int = 1, outer_steps: int = 24):
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.timedelta = np.timedelta64(1, 'h') * inner_steps

        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)

        self.model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
        self.model_with_physics = self.model.with_physics_core_output(enable=True)

        self.all_levels = self.model.data_coords.vertical.centers

        self.model_lat = self.model.data_coords.horizontal.latitudes * 180 / np.pi
        self.model_lon = self.model.data_coords.horizontal.longitudes * 180 / np.pi

        self._compiled_fn = None
        self._compile_functions()

    def _compile_functions(self):

        def forward_and_process(inputs, input_forcings, rng_keys, all_forcings, level_idx, interp_idx):

            initial_state = jax.vmap(self.model.encode)(inputs, input_forcings, rng_keys)

            def unroll_op(state, forcings):
                return self.model_with_physics.unroll(
                    state, forcings,
                    steps=self.outer_steps,
                    timedelta=self.timedelta,
                    start_with_input=True
                )
            _, predictions = jax.vmap(unroll_op)(initial_state, all_forcings)

            return process_predictions(predictions, level_idx, interp_idx)

        n_devices = jax.local_device_count()
        if n_devices > 1:
            self._forward_fn = jax.pmap(
                forward_and_process,
                in_axes=(0, 0, 0, 0, None, None)
            )
            self._multi_gpu = True
        else:
            self._forward_fn = jax.jit(forward_and_process)
            self._multi_gpu = False

    def _compute_interp_indices(self, target_lon, target_lat):
        src_lat = np.array(self.model_lat, dtype=np.float64)
        src_lon = np.array(self.model_lon, dtype=np.float64)

        lat_idx_range = np.arange(len(src_lat), dtype=np.float64)
        if src_lat[0] > src_lat[-1]:
            src_lat, lat_idx_range = src_lat[::-1], lat_idx_range[::-1]

        flat_lat = np.array(target_lat, dtype=np.float64).ravel()
        lat_indices = np.interp(flat_lat, src_lat, lat_idx_range).reshape(target_lat.shape)

        flat_lon = np.array(target_lon, dtype=np.float64).ravel()
        if src_lon.min() >= 0:
            flat_lon = flat_lon % 360.0

        lon_step = src_lon[1] - src_lon[0]
        src_lon_ext = np.append(src_lon, src_lon[-1] + lon_step)
        lon_idx_ext = np.arange(len(src_lon) + 1, dtype=np.float64)
        lon_indices = np.interp(flat_lon, src_lon_ext, lon_idx_ext).reshape(target_lon.shape)

        for idx in [lat_indices, lon_indices]:
            rounded = np.round(idx)
            mask = np.abs(idx - rounded) < 1e-6
            idx[mask] = rounded[mask]

        indices = np.stack([lon_indices, lat_indices], axis=0)
        return jax.device_put(jnp.array(indices, dtype=jnp.float32))

    def _prepare_inputs(self, datasets, include_targets=True):
        inputs_list, forcings_list, all_forcings_list, targets_list = [], [], [], []

        for ds in datasets:
            inputs_list.append(self.model.inputs_from_xarray(ds.isel(time=0)))
            forcings_list.append(self.model.forcings_from_xarray(ds.isel(time=0)))
            all_forcings_list.append(self.model.forcings_from_xarray(ds.head(time=1)))
            if include_targets:
                targets_list.append(self.model.inputs_from_xarray(ds))

        batched = {
            'inputs': jax.device_put(stack_dicts(inputs_list)),
            'input_forcings': jax.device_put(stack_dicts(forcings_list)),
            'all_forcings': jax.device_put(stack_dicts(all_forcings_list)),
            'rng_keys': jax.random.split(jax.random.key(42), len(datasets))
        }
        if include_targets:
            batched['targets'] = jax.device_put(stack_dicts(targets_list))
        return batched

    def _level_index(self, target_levels):
        if target_levels is not None:
            level_idx = [np.argmin(np.abs(self.all_levels - l)) for l in target_levels]
        else:
            level_idx = list(range(len(self.all_levels)))
        return jnp.array(level_idx, dtype=jnp.int32)

    def _forward_args(self, datasets, target_levels=None, include_era5_label=False,
                      region_lon=None, region_lat=None):
        batched = self._prepare_inputs(datasets, include_targets=include_era5_label)
        B = int(batched['rng_keys'].shape[0])
        level_idx = self._level_index(target_levels)
        interp_idx = (
            self._compute_interp_indices(region_lon, region_lat)
            if region_lon is not None else None
        )
        if self._multi_gpu:
            n_devices = jax.local_device_count()
            if B % n_devices != 0:
                raise ValueError(f"batch size ({B}) must be divisible by device count ({n_devices})")
            b_local = B // n_devices
            reshape = lambda x: x.reshape((n_devices, b_local) + x.shape[1:])
            args = (
                jax.tree_util.tree_map(reshape, batched['inputs']),
                jax.tree_util.tree_map(reshape, batched['input_forcings']),
                reshape(batched['rng_keys']),
                jax.tree_util.tree_map(reshape, batched['all_forcings']),
                level_idx,
                interp_idx,
            )
        else:
            args = (
                batched['inputs'], batched['input_forcings'], batched['rng_keys'],
                batched['all_forcings'], level_idx, interp_idx,
            )
        return args, B, batched

    def compile(self, datasets, target_levels=None, include_era5_label=False,
                region_lon=None, region_lat=None):
        if self._compiled_fn is not None:
            return
        args, _, _ = self._forward_args(
            datasets,
            target_levels=target_levels,
            include_era5_label=include_era5_label,
            region_lon=region_lon,
            region_lat=region_lat,
        )
        self._compiled_fn = self._forward_fn.lower(*args).compile()

    def _run_forward_fn(self, args, batch_size):
        fn = self._compiled_fn if self._compiled_fn is not None else self._forward_fn
        jax_full, jax_phy = fn(*args)
        if self._multi_gpu:
            merge = lambda x: x.reshape((batch_size,) + x.shape[2:])
            jax_full = jax.tree_util.tree_map(merge, jax_full)
            jax_phy = jax.tree_util.tree_map(merge, jax_phy)
        return jax_full, jax_phy

    def forward(self, datasets, target_levels=None, include_era5_label=True,
                region_lon=None, region_lat=None):
        args, B, batched = self._forward_args(
            datasets,
            target_levels=target_levels,
            include_era5_label=include_era5_label,
            region_lon=region_lon,
            region_lat=region_lat,
        )
        jax_full, jax_phy = self._run_forward_fn(args, B)
        jax.block_until_ready(jax_full)

        t_full = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(jax_full))
        t_phy = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(jax_phy))

        if not include_era5_label:
            return t_full, t_phy

        level_idx, interp_idx = args[-2], args[-1]

        def process_target(targets, level_idx, interp_idx):
            return process_predictions(targets, level_idx, interp_idx)[0]

        if self._multi_gpu:
            n_devices = jax.local_device_count()
            b_local = B // n_devices
            reshape = lambda x: x.reshape((n_devices, b_local) + x.shape[1:])
            merge = lambda x: x.reshape((B,) + x.shape[2:])
            targets_p = jax.tree_util.tree_map(reshape, batched['targets'])
            jax_target = jax.pmap(process_target, in_axes=(0, None, None))(
                targets_p, level_idx, interp_idx
            )
            jax_target = jax.tree_util.tree_map(merge, jax_target)
        else:
            jax_target = jax.jit(process_target)(batched['targets'], level_idx, interp_idx)

        jax.block_until_ready(jax_target)
        t_target = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(jax_target))
        return t_full, t_phy, t_target
