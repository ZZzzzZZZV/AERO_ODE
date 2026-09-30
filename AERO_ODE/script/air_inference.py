import torch

from script import config as cfg
from script.project_import import project_on_path


class AirForecaster:
    def __init__(self, device: str = "cuda:0", time_chunk: int = 8, ngcm=None):
        with project_on_path(cfg.AIR_ROOT):
            from Generate_Forecast_wb import Config, Forecaster

            config = Config()
            config.DEVICE = device
            config.TIME_CHUNK = time_chunk
            self._impl = Forecaster(config, ngcm=ngcm)
        self.ngcm = self._impl.ngcm
        self.device = self._impl.device
        self.time_chunk = time_chunk

    def compile(self, input_list):
        impl = self._impl
        impl.ngcm.compile(
            input_list,
            target_levels=list(impl.target_levels),
            include_era5_label=False,
            region_lon=impl.region_lon,
            region_lat=impl.region_lat,
        )
        width, height = impl.region_lat.shape
        dyn = torch.zeros(1, 70, width, height, device=impl.device)
        sta = impl.static_manager.get_static_input(1)
        zeros = torch.zeros(1, dtype=torch.long, device=impl.device)
        with torch.inference_mode():
            impl.correction_model(dyn, sta, zeros, zeros, zeros)

    def predict(self, input_list, time_list):
        return self._impl.inference(input_list, time_list)
