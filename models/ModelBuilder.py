from models.unet import build_unet
from models.gated_scnn import build_gscnn
from models.fast_fcn import build_fast_fcn
from models.deeplabv3plus import build_deeplabv3plus

MODEL_DICT = {"unet": build_unet,
              "gated_scnn": build_gscnn,
              "fast_fcn": build_fast_fcn,
              "deeplabv3plus": build_deeplabv3plus}

def build_model(name, img_size, num_channels):
    if name in MODEL_DICT:
        return MODEL_DICT[name].build(img_size, num_channels)
    else:
        raise NotImplementedError("{} is not a supported model".format(name))

