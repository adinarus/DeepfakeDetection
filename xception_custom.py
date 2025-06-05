import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense,
    Dropout, Input, DepthwiseConv2D, MaxPooling2D, SeparableConv2D
)
#
# # ------------------------
# # XceptionBlock using built-in SeparableConv2D
# # ------------------------
# class XceptionBlock(tf.keras.layers.Layer):
#     def __init__(self, in_channels, out_channels, reps, strides=1,
#                  start_with_relu=True, grow_first=True, name_prefix='xblock'):
#         super().__init__(name=name_prefix)
#         self.rep = []
#         self.skip = None
#
#         if out_channels != in_channels or strides != 1:
#             self.skip = Conv2D(out_channels, kernel_size=1, strides=strides,
#                                padding='same', use_bias=False, name=f'{name_prefix}_skip_conv')
#             self.skip_bn = BatchNormalization(name=f'{name_prefix}_skip_bn')
#
#         filters = in_channels
#         idx = 0
#
#         if grow_first:
#             if start_with_relu:
#                 self.rep.append(ReLU(name=f'{name_prefix}_relu_{idx}'))
#                 idx += 1
#             self.rep.append(SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same',
#                                             use_bias=False, name=f'{name_prefix}_sepconv_{idx}'))
#             self.rep.append(BatchNormalization(name=f'{name_prefix}_bn_{idx}'))
#             filters = out_channels
#             idx += 1
#
#         for i in range(reps - 1):
#             self.rep.append(ReLU(name=f'{name_prefix}_relu_{idx}'))
#             self.rep.append(SeparableConv2D(filters, kernel_size=3, strides=1, padding='same',
#                                             use_bias=False, name=f'{name_prefix}_sepconv_{idx}'))
#             self.rep.append(BatchNormalization(name=f'{name_prefix}_bn_{idx}'))
#             idx += 1
#
#         if not grow_first:
#             self.rep.append(ReLU(name=f'{name_prefix}_relu_{idx}'))
#             self.rep.append(SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same',
#                                             use_bias=False, name=f'{name_prefix}_sepconv_{idx}'))
#             self.rep.append(BatchNormalization(name=f'{name_prefix}_bn_{idx}'))
#
#         if strides != 1:
#             self.rep.append(MaxPooling2D(pool_size=3, strides=strides, padding='same',
#                                          name=f'{name_prefix}_pool'))
#
#         self.rep = tf.keras.Sequential(self.rep, name=f'{name_prefix}_seq')
#
#     def call(self, inputs):
#         x = self.rep(inputs)
#         skip = inputs
#         if self.skip is not None:
#             skip = self.skip(skip)
#             skip = self.skip_bn(skip)
#         return layers.add([x, skip], name=f'{self.name}_add')
#
# # ------------------------
# # Xception model builder
# # ------------------------
# def build_xception_model(input_shape=(256, 256, 3), model_name='xception'):
#     inputs = Input(shape=input_shape, name=f'{model_name}_input')
#
#     x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False, name=f'{model_name}_block1_conv1')(inputs)
#     x = BatchNormalization(name=f'{model_name}_bn1')(x)
#     x = ReLU(name=f'{model_name}_relu1')(x)
#
#     x = Conv2D(64, 3, padding='same', use_bias=False, name=f'{model_name}_block1_conv2')(x)
#     x = BatchNormalization(name=f'{model_name}_bn2')(x)
#     x = ReLU(name=f'{model_name}_relu2')(x)
#
#     x = XceptionBlock(64, 128, reps=2, strides=2, start_with_relu=False, grow_first=True,
#                       name_prefix=f'{model_name}_block2')(x)
#     x = XceptionBlock(128, 256, reps=2, strides=2, name_prefix=f'{model_name}_block3')(x)
#     x = XceptionBlock(256, 728, reps=2, strides=2, name_prefix=f'{model_name}_block4')(x)
#
#     for i in range(8):
#         x = XceptionBlock(728, 728, reps=3, strides=1, name_prefix=f'{model_name}_block_mid_{i}')(x)
#
#     x = XceptionBlock(728, 1024, reps=2, strides=2, grow_first=False, name_prefix=f'{model_name}_block5')(x)
#
#     x = SeparableConv2D(1536, kernel_size=3, strides=1, padding='same', use_bias=False,
#                         name=f'{model_name}_sepconv1')(x)
#     x = BatchNormalization(name=f'{model_name}_bn3')(x)
#     x = ReLU(name=f'{model_name}_relu3')(x)
#
#     x = SeparableConv2D(2048, kernel_size=3, strides=1, padding='same', use_bias=False,
#                         name=f'{model_name}_sepconv2')(x)
#     x = BatchNormalization(name=f'{model_name}_bn4')(x)
#     x = ReLU(name=f'{model_name}_relu4')(x)
#
#     return Model(inputs, x, name=model_name)

##############################





def build_xception_model(input_shape=(256, 256, 3), model_name='xception'):
    """
    Custom Xception backbone for F3Net which supports multiple channel inputs used in FAD/LFS compared to the Keras native which only supports 3 channels.
    Layer renaming allows instantiation of multiple independent Xception branches (FAD and LFS are both used in MixBlock).
    """
    inputs = Input(shape=input_shape, name=f'{model_name}_input')

    x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False, name=f'{model_name}_block1_conv1')(inputs)
    x = BatchNormalization(name=f'{model_name}_bn1')(x)
    x = ReLU(name=f'{model_name}_relu1')(x)

    x = Conv2D(64, 3, padding='same', use_bias=False, name=f'{model_name}_block1_conv2')(x)
    x = BatchNormalization(name=f'{model_name}_bn2')(x)
    x = ReLU(name=f'{model_name}_relu2')(x)

    # Entry flow
    x = XceptionBlock(64, 128, downsample=True, name_prefix=f'{model_name}_block2')(x)
    x = XceptionBlock(128, 256, downsample=True, name_prefix=f'{model_name}_block3')(x)
    x = XceptionBlock(256, 728, downsample=True, name_prefix=f'{model_name}_block4')(x)

    # Middle flow (all blocks are same shape)
    x = XceptionBlock(728, 728, downsample=False, name_prefix=f'{model_name}_block5')(x)
    x = XceptionBlock(728, 728, downsample=False, name_prefix=f'{model_name}_block6')(x)
    x = XceptionBlock(728, 728, downsample=False, name_prefix=f'{model_name}_block7')(x)
    x = XceptionBlock(728, 728, downsample=False, name_prefix=f'{model_name}_block8')(x)
    x = XceptionBlock(728, 728, downsample=False, name_prefix=f'{model_name}_block9')(x)
    x = XceptionBlock(728, 728, downsample=False, name_prefix=f'{model_name}_block10')(x)
    x = XceptionBlock(728, 728, downsample=False, name_prefix=f'{model_name}_block11')(x)
    x = XceptionBlock(728, 728, downsample=False, name_prefix=f'{model_name}_block12')(x)

    # Exit flow
    x = XceptionBlock(728, 1024, downsample=True, name_prefix=f'{model_name}_block13')(x)

    x = SeparableConv2D(1536, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{model_name}_sepconv1')(x)
    x = BatchNormalization(name=f'{model_name}_bn3')(x)
    x = ReLU(name=f'{model_name}_relu3')(x)

    x = SeparableConv2D(2048, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{model_name}_sepconv2')(x)
    x = BatchNormalization(name=f'{model_name}_bn4')(x)
    x = ReLU(name=f'{model_name}_relu4')(x)

    return Model(inputs=inputs, outputs=x, name=model_name)


class XceptionBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, downsample=False, name_prefix='xblock'):
        super().__init__(name=name_prefix)
        self.downsample = downsample
        stride = 2 if downsample else 1

        self.sepconv1 = SeparableConv2D(out_channels, 3, padding='same', use_bias=False, strides=1, name=f'{name_prefix}_sepconv1')
        self.bn1 = BatchNormalization(name=f'{name_prefix}_bn1')

        self.sepconv2 = SeparableConv2D(out_channels, 3, padding='same', use_bias=False, strides=1, name=f'{name_prefix}_sepconv2')
        self.bn2 = BatchNormalization(name=f'{name_prefix}_bn2')

        self.sepconv3 = SeparableConv2D(out_channels, 3, padding='same', use_bias=False, strides=stride, name=f'{name_prefix}_sepconv3')
        self.bn3 = BatchNormalization(name=f'{name_prefix}_bn3')

        if in_channels != out_channels or downsample:
            self.skip = Conv2D(out_channels, 1, strides=stride, padding='same', use_bias=False, name=f'{name_prefix}_skip_conv')
            self.skip_bn = BatchNormalization(name=f'{name_prefix}_skip_bn')
        else:
            self.skip = None

        self.relu = ReLU()

    def call(self, x):
        shortcut = x

        x = self.relu(x)
        x = self.sepconv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.sepconv2(x)
        x = self.bn2(x)

        x = self.relu(x)
        x = self.sepconv3(x)
        x = self.bn3(x)

        if self.skip is not None:
            shortcut = self.skip(shortcut)
            shortcut = self.skip_bn(shortcut)

        return layers.add([x, shortcut], name=f'{self.name}_add')
#####################################################################################
#
# #
# # class XceptionBlock(tf.keras.layers.Layer):
# #     def __init__(self, in_channels, out_channels, reps, strides=1,
# #                  start_with_relu=True, grow_first=True, name_prefix='xblock'):
# #         super().__init__(name=name_prefix)
# #         self.rep = []
# #         self.skip = None
# #
# #         if out_channels != in_channels or strides != 1:
# #             self.skip = Conv2D(out_channels, kernel_size=1, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}_skip_conv')
# #             self.skip_bn = BatchNormalization(name=f'{name_prefix}_skip_bn')
# #
# #         filters = in_channels
# #         idx = 0
# #
# #         if grow_first:
# #             if start_with_relu:
# #                 self.rep.append(ReLU(name=f'{name_prefix}_relu_{idx}'))
# #                 idx += 1
# #             self.rep.append(SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name_prefix}_sepconv_{idx}'))
# #             self.rep.append(BatchNormalization(name=f'{name_prefix}_bn_{idx}'))
# #             filters = out_channels
# #             idx += 1
# #
# #         for i in range(reps - 1):
# #             self.rep.append(ReLU(name=f'{name_prefix}_relu_{idx}'))
# #             self.rep.append(SeparableConv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name_prefix}_sepconv_{idx}'))
# #             self.rep.append(BatchNormalization(name=f'{name_prefix}_bn_{idx}'))
# #             idx += 1
# #
# #         if not grow_first:
# #             self.rep.append(ReLU(name=f'{name_prefix}_relu_{idx}'))
# #             self.rep.append(SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{name_prefix}_sepconv_{idx}'))
# #             self.rep.append(BatchNormalization(name=f'{name_prefix}_bn_{idx}'))
# #
# #         if strides != 1:
# #             self.rep.append(MaxPooling2D(pool_size=3, strides=strides, padding='same', name=f'{name_prefix}_pool'))
# #
# #         self.rep = tf.keras.Sequential(self.rep, name=f'{name_prefix}_seq')
# #
# #     def call(self, inputs):
# #         x = self.rep(inputs)
# #         skip = inputs
# #         if self.skip is not None:
# #             skip = self.skip(skip)
# #             skip = self.skip_bn(skip)
# #         return layers.add([x, skip], name=f'{self.name}_add')