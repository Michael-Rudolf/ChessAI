use std::f32::consts::E;
use rand::rng;
use rand::Rng;
use crate::utility::layer::Layer;

pub(crate) struct NeuronalLayer {
    inputs_count: usize,
    outputs_count: usize,
    biases: Vec<f32>,
    weights: Vec<f32>,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
    activation_function: ActivationFunction,
}

impl NeuronalLayer {
    pub(crate) fn clone(&self) -> NeuronalLayer {
        NeuronalLayer{ inputs_count: self.inputs_count, weights: self.weights.iter().map(|x| x.clone()).collect(), inputs: self.inputs.iter().map(|x| x.clone()).collect(), outputs: self.outputs.iter().map(|x| x.clone()).collect(), biases: self.biases.iter().map(|x| x.clone()).collect(), outputs_count: self.outputs_count, activation_function: self.activation_function.clone() }
    }
}

impl NeuronalLayer {
    pub fn new(input_count: usize, output_count: usize, function: ActivationFunction) -> NeuronalLayer {
        let mut biases: Vec<f32> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        for _ in 0..output_count {
            let mut random_number = rng().random_range(0..100) as f32 / 100.0;
            biases.push(random_number);
            for _ in 0..input_count {
                random_number = rng().random_range(0..100) as f32 / 100.0;
                weights.push(random_number);
            }
        }

        NeuronalLayer {inputs_count: input_count, outputs_count: output_count, biases, weights, inputs: Vec::new(), outputs: Vec::new(), activation_function: function}
    }

    pub(crate) fn get_weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    pub(crate) fn set_weight(&mut self, selector: usize, new_value: f32) {
        self.weights[selector] = new_value;
    }

    pub(crate) fn get_weight(&self, selector: usize) -> f32 {
        self.weights[selector]
    }

    pub(crate) fn get_biases(&self) -> Vec<f32> {
        self.biases.clone()
    }

    pub(crate) fn set_bias(&mut self, selector: usize, new_value: f32) {
        self.biases[selector] = new_value;
    }

    pub(crate) fn get_bias(&self, selector: usize) -> f32 {
        self.biases[selector]
    }
}

impl Layer for NeuronalLayer {
    fn set_inputs(&mut self, inputs: Vec<f32>) {
        self.inputs = inputs;
    }
    fn execute(&mut self) {
        let mut neurons: Vec<f32> = self.biases.clone();
        for i in 0..self.weights.len() {
            let input_neuron_position = i / self.outputs_count;
            let output_neuron_position = i % self.outputs_count;

            let input = self.inputs[input_neuron_position];
            neurons[output_neuron_position] += self.weights[input_neuron_position] * input;
        }
        for i in 0..neurons.len() {
            let initial_value = neurons[i];
            match self.activation_function {
                ActivationFunction::Sigmoid => {neurons[i] = 1.0/(1.0 + E.powf(-initial_value));},
                ActivationFunction::Linear => {},
            }
        }
        self.outputs = neurons.clone();
    }

    fn get_outputs(&self) -> Vec<f32> {
        self.outputs.clone()
    }



    fn output_count(&self) -> usize {self.outputs_count}

    fn input_count(&self) -> usize {self.inputs_count}

    fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights;
    }

    fn set_biases(&mut self, biases: Vec<f32>) {
        self.biases = biases;
    }
}


pub(crate) enum ActivationFunction {
    Sigmoid,
    Linear
}

impl Clone for ActivationFunction{
    fn clone(&self) -> ActivationFunction{
        match self {
            ActivationFunction::Sigmoid => ActivationFunction::Sigmoid,
            ActivationFunction::Linear => ActivationFunction::Linear,
        }
    }
}