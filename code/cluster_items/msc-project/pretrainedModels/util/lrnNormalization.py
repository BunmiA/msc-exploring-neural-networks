import tensorflow as tf

class MyLRNLayer(tf.keras.layers.Layer):
    def __init__(self, depth_radius=5,bias=1,alpha=1,beta=0.5, **kwargs):
        #         self.output_dim = output_dim
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        super(MyLRNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(None),
                                      initializer='uniform',
                                      trainable=False)
        super(MyLRNLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.nn.local_response_normalization(x,self.depth_radius,self.bias,self.alpha,self.beta)
