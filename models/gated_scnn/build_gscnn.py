from models.gated_scnn.gated_shape_cnn.model import GSCNN

def build(img_size, output_channels):
    return GSCNN(n_classes=output_channels)

