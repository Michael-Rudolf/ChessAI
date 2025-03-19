use std::process::Output;
use crate::utility::layer::Layer;

pub struct Network {
    layers: Vec<dyn Layer>,
    output: Vec<f32>,
}

impl Network {
    pub fn add_layer(&mut self, layer: Layer) -> Network {
        self.layers.push(layer);
        return self;
    }

    pub fn execute(&mut self, inputs: Vec<f32>) {
        let mut last_output = inputs;
        for layer in &mut self.layers {
            (layer as &mut dyn Layer).set_inputs(last_output.clone());
            (layer as &mut dyn Layer).execute();
            last_output = (layer as &mut dyn Layer).get_outputs();
        }
    }

    pub fn get_outputs(&self) -> Vec<f32> {self.output.clone() }
}