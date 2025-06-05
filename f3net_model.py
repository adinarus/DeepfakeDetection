import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Dropout, DepthwiseConv2D
from xception_custom import build_xception_model

def create_dct_matrix(N):
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                D[k, n] = np.sqrt(1 / N)
            else:
                D[k, n] = np.sqrt(2 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return D.astype(np.float32)

# compute frequency index, low values - top left, high values - bottom right
def generate_filter(start, end, size):

    mask = np.zeros((size, size), dtype=np.float32)


    for i in range(size):
        for j in range(size):
            freq_index = i + j

            # check if frequency is within band
            if start <= freq_index < end:
                mask[i, j] = 1.0  # include this frequency in the mask

    return mask

def sigma(x):
    return (1 - tf.exp(-x)) / (1 + tf.exp(-x))

class FrequencyFilter(tf.keras.layers.Layer):
    """
    A custom layer that applies filt_base + sigma(filt_learnable) to a DCT transformed image.
    """

    def __init__(self, size, band_start, band_end, name=None, use_learnable=True, norm=False):
        super().__init__(name=name)
        self.size = size
        self.band_start = band_start
        self.band_end = band_end
        self.use_learnable = use_learnable

        base_mask = generate_filter(band_start, band_end, size)
        self.f_base_np = base_mask
        self.filt_base = tf.constant(base_mask, dtype=tf.float32)

        if self.use_learnable:
            self.filt_learnable = self.add_weight(
                shape=(size, size),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                trainable=True,
                name = f'{self.name}_learnable'
            )
        else:
            self.filt_learnable = None

        self.norm = norm
        if norm:
            self.ft_num = tf.reduce_sum(self.filt_base)  # scalar tensor
        else:
            self.ft_num = None

    def call(self, x):
        if self.use_learnable:
            filt = self.filt_base + sigma(self.filt_learnable)
        else:
            filt = self.filt_base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt

        return y


class FAD_Head(tf.keras.layers.Layer):
    """
    Frequency-aware decomposition module.
    Uses internally constructed frequency filters for low, mid, high, all bands.
    """
    def __init__(self, img_size=256):
        super().__init__()
        self.img_size = img_size

        # Precompute DCT and IDCT matrices
        D = create_dct_matrix(img_size)
        self.D = tf.constant(D, dtype=tf.float32)
        self.D_T = tf.transpose(self.D)

        # Create frequency filters using band_start and band_end
        self.filters = self._init_filters()

    def _init_filters(self):
        size = self.img_size

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1

        return [
            FrequencyFilter(size, 0, 2 * size * 1 / 16, "low"),
            FrequencyFilter(size, 2 * size * 1 / 16, 2 * size * 1 / 8, "mid"),
            FrequencyFilter(size, 2 * size * 1 / 8, size * 2, "high"),
            FrequencyFilter(size, 0, size * 2, "all")
        ]

    def call(self, x):
        # x: (B, H, W, 3)
        outputs = []

        for c in range(3):  # R, G, B
            x_c = x[:, :, :, c] # select each channel, shape (B,H,W)
            x_freq = tf.matmul(self.D, tf.matmul(x_c, self.D_T))

            band_outputs = []
            for f_filter in self.filters:
                x_band = f_filter(x_freq)
                x_recon = tf.matmul(self.D_T, tf.matmul(x_band, self.D))
                band_outputs.append(x_recon)

            outputs.append(tf.stack(band_outputs, axis=-1))  # (B, H, W, nr_filters)

        return tf.concat(outputs, axis=-1)  # Final shape: (B, H, W, 3 * nr_filters)


class LFS_Head(tf.keras.layers.Layer):
    """
    Local Frequency Statistics Head

    Variable names:
    n  = batch size
    c  = number of channels
    h, w = image height and width
    s = patch size
    l  = number of patches per image
    m  = number of frequency bands (filters)
    """

    def __init__(self, img_size=256, window_size=10, bands=6):
        super().__init__()
        self.window_size = window_size
        self.bands = bands

        D = create_dct_matrix(self.window_size)
        self.D = tf.constant(D, dtype=tf.float32)
        self.D_T = tf.transpose(self.D)

        self.filters = []
        for i in range(self.bands):
            start = int(2.0 * self.window_size / self.bands * i)
            end = int(2.0 * self.window_size / self.bands * (i + 1))
            filt = FrequencyFilter( size=self.window_size, band_start=start, band_end=end, name=f"lfs_band_{i}", use_learnable=True, norm=True)
            self.filters.append(filt)

    def call(self, x):
        # grayscale = 0.299 * R + 0.587 * G + 0.114 * B
        x_gray = (0.299 * x[:,:,:,0] +0.587 * x[:,:,:,1] + 0.114 * x[:,:,:,2])
        x_gray = tf.expand_dims(x_gray, axis=-1)  # (n, h, w, 1)

        # rescale from [-1,1] to [0,255]
        x_gray = (x_gray + 1.0) * 127.5 # shape (N, H, W, 1)

        N = tf.shape(x_gray)[0]  # batch size
        S = self.window_size  # patch size
        M = self.bands  # number of frequency filters

        # sliding window patch extraction
        x_unfold = tf.image.extract_patches(
            images=x_gray,
            sizes=[1, S, S, 1],
            strides=[1, 2, 2, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )  # shape (N, H', W', S*S)

        # compute number of patches
        H_p = tf.shape(x_unfold)[1]  # H'
        W_p = tf.shape(x_unfold)[2]  # W'
        L = H_p * W_p  # total patches per image

        # reshape to (N * L, S, S)
        x_unfold_reshaped = tf.reshape(x_unfold, [N * L, S, S])

        # apply DCT per patch
        x_dct = tf.matmul(self.D, tf.matmul(x_unfold_reshaped, self.D_T))  # (N * L, S, S)

        # aply each of the M band filters
        y_list = []
        for i in range(M):
            y = tf.abs(x_dct)
            y = tf.math.log(y + 1e-15) / tf.math.log(tf.constant(10., dtype=tf.float32))
            y = self.filters[i](y)  # apply filters for each band
            y = tf.reduce_sum(y, axis=[1, 2])

            y = tf.reshape(y, [N, H_p, W_p])  # (N, H', W')
            y = tf.expand_dims(y, axis=1)  # (N, 1, H', W')
            y_list.append(y)

        # concat M band outputs along channel dimension: (N, H', W', M)
        out = tf.concat(y_list, axis=1)  # (N, M, H', W')
        return tf.transpose(out, perm=[0, 2, 3, 1])  # (N, H', W', M)

class F3Net(tf.keras.Model):
    def __init__(self, img_size=256, lfs_window_size=10, lfs_stride=2, lfs_bands=6, mode='FAD'):
        super(F3Net, self).__init__()
        self.mode = mode
        self.img_size = img_size
        self.lfs_window_size = lfs_window_size
        self.lfs_bands = lfs_bands
        self.lfs_stride = lfs_stride

        if mode in ['FAD', 'Both', 'Mix']:
            self.FAD_head = FAD_Head(img_size)
            self.init_xcep_FAD()

        if mode in ['LFS', 'Both', 'Mix']:
            self.LFS_head = LFS_Head(img_size, lfs_window_size, lfs_bands)
            self.init_xcep_LFS()

        if mode == 'Original':
            self.xcep = Xception(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))

        if mode == 'Mix':
            self.mix_block = MixBlock(2048, f"mix_block_{mode.lower()}")

        self.relu = ReLU()
        self.global_average_pooling = GlobalAveragePooling2D()
        self.dropout = Dropout(0.2)
        self.dense = Dense(1, activation='sigmoid')

    def init_xcep_FAD(self):
        base = Xception(include_top=False, weights='imagenet', input_shape=(self.img_size, self.img_size, 3))
        conv1_weights = base.get_layer('block1_conv1').get_weights()[0]
        new_weights = np.zeros((3, 3, 12, 32), dtype=np.float32)

        for i in range(4):
            new_weights[:, :, i * 3:(i + 1) * 3, :] = conv1_weights / 4.0 #(kernel_height, kernel_width, input_channels, output_channels)

        model = build_xception_model(input_shape=(self.img_size, self.img_size, 12), model_name='xception_fad')
        model.get_layer('xception_fad_block1_conv1').set_weights([new_weights])
        self.FAD_xcep = model

    def init_xcep_LFS(self):
        base = Xception(include_top=False, weights='imagenet', input_shape=(self.img_size, self.img_size, 3))
        conv1_weights = base.get_layer('block1_conv1').get_weights()[0]
        M = self.lfs_bands
        new_weights = np.zeros((3, 3, M, 32), dtype=np.float32)
        # divide the M input channels into blocks of size 3
        # each block is assigned the pretrained weights
        for i in range(M//3):
            new_weights[:, :, i * 3:(i + 1) * 3, :] = conv1_weights / (M // 3)

        model = build_xception_model(input_shape=(self.img_size//self.lfs_stride, self.img_size//self.lfs_stride, M), model_name='xception_lfs')
        model.get_layer('xception_lfs_block1_conv1').set_weights([new_weights])
        self.LFS_xcep = model

    def call(self, x):
        if self.mode == 'FAD':
            x = self.FAD_head(x)
            x = self.FAD_xcep(x)

        elif self.mode == 'LFS':
            x = self.LFS_head(x)
            x = self.LFS_xcep(x)

        elif self.mode == 'Original':
            x = self.xcep(x)

        elif self.mode == 'Both':
            x1 = self.FAD_head(x)
            x1 = self.FAD_xcep(x1)
            x2 = self.LFS_head(x)
            x2 = self.LFS_xcep(x2)
            x = tf.concat([x1, x2], axis=-1)


        elif self.mode == 'Mix':

            fea_FAD = self.FAD_head(x)
            fea_FAD = self.FAD_xcep(fea_FAD) # shape (B, H1, W1, C)

            fea_LFS = self.LFS_head(x)
            fea_LFS = self.LFS_xcep(fea_LFS) # shape (B, H2, W2, C)

            # resize LFS to match FAD shape
            target_h, target_w = tf.shape(fea_FAD)[1], tf.shape(fea_FAD)[2]
            fea_LFS = tf.image.resize(fea_LFS, size=(target_h, target_w), method='bilinear')

            y_FAD, y_LFS = self.mix_block(fea_FAD, fea_LFS)
            x = tf.concat([y_FAD, y_LFS], axis=-1)

        # final classifier layers
        x = self.relu(x)
        x = self.global_average_pooling(x)
        x = self.dropout(x)
        return self.dense(x)

class MixBlock(tf.keras.layers.Layer):
    def __init__(self, c_in, name='mix_block'):
        super().__init__(name=name)
        self.c_in = c_in

        # 1x1 conv for q and k
        self.q_fad = Conv2D(c_in, kernel_size=1)
        self.q_lfs = Conv2D(c_in, kernel_size=1)
        self.k_fad = Conv2D(c_in, kernel_size=1)
        self.k_lfs = Conv2D(c_in, kernel_size=1)

        self.fad_conv = DepthwiseConv2D(kernel_size=1)
        self.lfs_conv = DepthwiseConv2D(kernel_size=1)

        self.fad_bn = BatchNormalization()
        self.lfs_bn = BatchNormalization()

        self.fad_gamma = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='fad_gamma')
        self.lfs_gamma = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='lfs_gamma')

        self.relu = ReLU()
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x_fad, x_lfs):
        B, H, W, C = tf.shape(x_fad)[0], tf.shape(x_fad)[1], tf.shape(x_fad)[2], tf.shape(x_fad)[3]

        # Compute Q and K features
        q_fad = self.q_fad(x_fad)  # (B, H, W, C)
        q_lfs = self.q_lfs(x_lfs)
        k_fad = self.k_fad(x_fad)
        k_lfs = self.k_lfs(x_lfs)

        # transpose query to (B, W, H, C)
        q_fad = tf.transpose(q_fad, perm=[0, 2, 1, 3])
        q_lfs = tf.transpose(q_lfs, perm=[0, 2, 1, 3])
        q = tf.concat([q_fad, q_lfs], axis=2)  # spatial concat (B, W, 2H, C)
        q = tf.reshape(q, [B * C, W, 2 * H])

        # transpose keys to (B, H, W, C)
        k_fad = tf.transpose(k_fad, perm=[0, 1, 2, 3])
        k_lfs = tf.transpose(k_lfs, perm=[0, 1, 2, 3])
        k = tf.concat([k_fad, k_lfs], axis=1)  # spatial concat (B, 2H, W, C)
        k = tf.reshape(k, [B * C, 2 * H, W])

        # Attention map
        energy = tf.matmul(q, k)  # (B*C, W, W)
        attention = self.softmax(energy)
        attention = tf.reshape(attention, [B, C, W, W])
        attention = tf.transpose(attention, perm=[0, 2, 3, 1])  # from (B, C, W, W) to (B, W, W, C)

        att_lfs = x_lfs * attention * (tf.sigmoid(self.lfs_gamma) * 2.0 - 1.0)
        y_fad = x_fad + self.fad_bn(self.fad_conv(att_lfs))

        att_fad = x_fad * attention * (tf.sigmoid(self.fad_gamma) * 2.0 - 1.0)
        y_lfs = x_lfs + self.lfs_bn(self.lfs_conv(att_fad))

        return y_fad, y_lfs