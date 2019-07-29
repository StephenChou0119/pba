import tensorflow as tf


def build_mobilenetv2(inputs, num_classes, is_training):
    import models.mobilenet.mobilenet_v2 as mobilenet_v2
    if is_training:
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
            logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
    else:
        logits, endpoints = mobilenet_v2.mobilenet_v2_035(inputs, num_classes=num_classes)
    return logits


def build_efficientnet(inputs, num_classes, is_training):
    from models.efficientnet_builder import build_model
    logits,_ = build_model(inputs, 'efficientnet-b0', is_training,
                override_params={'num_classes': num_classes})
    return logits


build_model=build_efficientnet

rotate_max_degree = 30
posterize_max = 4
enhance_max = 1.8  # color contrast brightness sharpness
shear_x_max = 0.3
shear_y_max = 0.3
translate_x_max = 10
translate_y_max = 10
cutout_max_size = 20
