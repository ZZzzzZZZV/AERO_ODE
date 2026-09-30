import torch

from script import config as cfg
from script.project_import import project_on_path


class SurfaceForecaster:
    def __init__(self, device: str = "cuda:0", time_chunk: int = 8, ngcm=None):
        with project_on_path(cfg.SURFACE_ROOT):
            from Generate_Forecast_Surface import Config
            from Generate_Forecast_Surface import SurfaceForecaster as WDForecaster

            config = Config()
            config.DEVICE = device
            config.TIME_CHUNK = time_chunk
            self._impl = WDForecaster(config, ngcm=ngcm)
        self.ngcm = self._impl.ngcm
        self.device = self._impl.device
        self.time_chunk = time_chunk

    def compile(self, input_list):
        impl = self._impl
        if getattr(impl.ngcm, "_compiled_fn", None) is None:
            impl.ngcm.compile(
                input_list,
                target_levels=list(impl.target_levels),
                include_era5_label=False,
                region_lon=impl.region_lon,
                region_lat=impl.region_lat,
            )
        width, height = impl.region_lat.shape
        dyn = torch.zeros(1, 74, width, height, device=impl.device)
        sta = impl.static_manager.get_static_input(1)
        zeros = torch.zeros(1, dtype=torch.long, device=impl.device)
        with torch.inference_mode():
            impl.diagnosis_model(dyn, sta, zeros, zeros, zeros)

    def predict(self, input_list, time_list):
        return self._impl.inference(input_list, time_list)
