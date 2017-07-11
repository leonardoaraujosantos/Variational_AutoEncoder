import tensorflow as tf
import model_util as util


class VAE_CNN(object):
    def __init__(self, img_size = 28, latent_size=20):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size * img_size], name='IMAGE_IN')
        self.__x_image = tf.reshape(self.__x, [-1, img_size, img_size, 1])

        with tf.name_scope('ENCODER'):
            ##### ENCODER
            # Calculating the convolution output:
            # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
            # H_out = 1 + (H_in+(2*pad)-K)/S
            # W_out = 1 + (W_in+(2*pad)-K)/S
            # CONV1: Input 28x28x1 after CONV 5x5 P:2 S:2 H_out: 1 + (28+4-5)/2 = 14, W_out= 1 + (28+4-5)/2 = 14
            self.__conv1 = util.conv2d(self.__x_image, 5, 5, 1, 16, 2, "conv1", pad='SAME',
                                       viewWeights=True, do_summary=False)
            self.__conv1_act = util.relu(self.__conv1, do_summary=False)

            # CONV2: Input 14x14x16 after CONV 5x5 P:0 S:2 H_out: 1 + (14+4-5)/2 = 7, W_out= 1 + (14+4-5)/2 = 7
            self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 16, 32, 2, "conv2", do_summary=False, pad='SAME')
            self.__conv2_act = util.relu(self.__conv2, do_summary=False)

        with tf.name_scope('LATENT'):
            # Reshape: Input 7x7x32 after [7x7x32]
            self.__enc_out = tf.reshape(self.__conv2_act, [tf.shape(self.__x)[0], 7 * 7 * 32])

            # Add linear ops for mean and variance
            self.__w_mean = util.linear_layer(self.__enc_out, 7 * 7 * 32, latent_size, "w_mean")
            self.__w_stddev = util.linear_layer(self.__enc_out, 7 * 7 * 32, latent_size, "w_stddev")

            self.__samples = tf.random_normal([tf.shape(self.__x)[0], latent_size], 0, 1, dtype=tf.float32)
            self.__guessed_z = self.__w_mean + (self.__w_stddev * self.__samples)

        with tf.name_scope('DECODER'):
            ##### DECODER (At this point we have 1x18x64
            # Kernel, output size, in_volume, out_volume, stride

            # Embedding variable based on the latent value (Tensorboard stuff)
            #self.__embedding = tf.Variable(tf.zeros_like(self.__guessed_z), name="test_embedding")
            # self.__assignment = self.__embedding.assign(self.__guessed_z)
            self.__embedding = tf.Variable(tf.zeros([50, latent_size]), name="test_embedding")
            self.__assignment = self.__embedding.assign(tf.reshape(self.__guessed_z, [tf.shape(self.__x)[0], latent_size]))

            # Linear layer
            self.__z_develop = util.linear_layer(self.__guessed_z, latent_size, 7 * 7 * 32,
                                                 'z_matrix', do_summary=True)
            self.__z_develop_act = util.relu(tf.reshape(self.__z_develop, [tf.shape(self.__x)[0], 7, 7, 32]),
                                             do_summary=False)

            self.__conv_t2_out = util.conv2d_transpose(self.__z_develop_act, (5, 5), (14, 14), 32,16, 2,
                                                       name="dconv1",do_summary=False, pad='SAME')
            self.__conv_t2_out_act = util.relu(self.__conv_t2_out, do_summary=False)

            self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (img_size, img_size), 16, 1, 2,
                                                       name="dconv2", do_summary=False, pad='SAME')

            # Model output
            self.__y = util.sigmoid(self.__conv_t1_out)
            self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], 28 * 28])


    @property
    def output(self):
        return self.__y

    @property
    def z_mean(self):
        return self.__w_mean

    @property
    def assignment(self):
        return self.__assignment

    @property
    def z_stddev(self):
        return self.__w_stddev

    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input(self):
        return self.__x

    @property
    def image_in(self):
        return self.__x_image




