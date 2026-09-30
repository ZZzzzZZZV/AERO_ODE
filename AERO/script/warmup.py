import torch


def warmup_models(air_model, surface_model, input_list, time_str):
    air_model.compile(input_list)
    surface_model.compile(input_list)
    air_model.predict(input_list, [time_str])
    surface_model.predict(input_list, [time_str])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
