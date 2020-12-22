import tensorflow as tf
from tensorflow.keras import activations,initializers,regularizers,constraints
from tensorflow import keras
class MMoE(keras.layers.Layer):
    def __init__(self,units,num_experts,num_tasks,input_dimension):
        super(MMoE,self).__init__()
        self.expert_activation = activations.get('relu')
        self.gate_activation = activations.get('softmax')

        self.expert_kernel_initializer = initializers.get('VarianceScaling')
        self.gate_kernel_initializer = initializers.get('VarianceScaling')

        self.expert_bias_initializer = initializers.get('zeros')
        self.gate_bias_initializer = initializers.get('zeros')

        self.expert_kernels =self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, units, num_experts),
            initializer=self.expert_kernel_initializer,
            trainable=True

        )
        self.expert_bias = self.add_weight(
            name='expert_bias',
            shape=(units, num_experts),
            initializer=self.expert_bias_initializer,
            trainable=True

        )
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, num_experts),
            initializer=self.gate_kernel_initializer,
            trainable=True
        ) for i in range(num_tasks)]

        self.gate_bias = [self.add_weight(
            name='gate_bias_task_{}'.format(i),
            shape=(num_experts,),
            initializer=self.gate_bias_initializer,
            trainable=True
        ) for i in range(num_tasks)]

    def build(self, input_shape):
        input_dimension = input_shape[-1]
        self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: input_dimension})
        super(MMoE, self).build(input_shape)
    def call(self, inputs, **kwargs):
        """
        Method for the forward function of the layer.

        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        gate_outputs = []
        final_outputs = []

        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        expert_outputs = tf.einsum('ab,bcd->acd',inputs,self.expert_kernels)
        # Add the bias term to the expert weights if necessary

        expert_outputs += self.expert_bias
        expert_outputs = self.expert_activation(expert_outputs)

        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = tf.einsum("ab,bc->ac", (inputs, gate_kernel))
            # Add the bias term to the gate weights if necessary

            gate_output += self.gate_bias[index]
            gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        for gate_output in gate_outputs:
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * keras.backend.repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))

        return final_outputs
