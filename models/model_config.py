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
    if is_training:
        build_model(inputs,'efficientnet-b0', True,
                                override_params={'num_classes': num_classes})
    else:
        build_model(inputs,'efficientnet-b0', False,
                                override_params={'num_classes': num_classes})


build_model=build_mobilenetv2
