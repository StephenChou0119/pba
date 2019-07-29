import tensorflow as tf
import models.custom_ops as ops

arg_scope = tf.contrib.framework.arg_scope


def setup_arg_scopes(is_training):
    """Sets up the argscopes that will be used when building an image model.

    Args:
      is_training: Is the model training or not.

    Returns:
      Arg scopes to be put around the model being constructed.
    """

    batch_norm_decay = 0.9
    batch_norm_epsilon = 1e-5
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        'scale': True,
        # collection containing the moving mean and moving variance.
        'is_training': is_training,
    }

    scopes = []

    scopes.append(arg_scope([ops.batch_norm], **batch_norm_params))
    return scopes


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
