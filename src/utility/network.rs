use std::f32::consts::E;
use std::process::Output;
use rand::{rng, Rng};
use crate::utility::layer::Layer;
use crate::utility::neuronal_layer::neuronal_layer::NeuronalLayer;
use crate::utility::training_data::TrainingData;

pub struct Network {
    layers: Vec<NeuronalLayer>,
    output: Vec<f32>,
    dataset: Vec<TrainingData>,
}


impl Network {

    pub fn new() -> Network {
        Network{layers: vec![], output: vec![], dataset: vec![] }
    }
    pub fn add_layer(&mut self, layer: NeuronalLayer) -> &mut Network {
        self.layers.push(layer);
        self
    }

    pub fn add_training_data(&mut self, data: TrainingData) -> &mut Network {
        self.dataset.push(data);
        self
    }

    pub fn execute(&mut self, inputs: Vec<f32>) {
        let mut last_output = inputs;
        for layer in &mut self.layers {
            layer.set_inputs(last_output.clone());
            layer.execute();
            last_output = (layer as &mut dyn Layer).get_outputs();
        }
        self.output = last_output;
    }

    pub fn make_mutated_version(&self, max_change: f32) -> Network{
        let mut mutant = self.clone();
        let min_change_pm = -1000.0 * max_change;//Minimum change in permil
        let max_change_pm = -min_change_pm;
        for i in 0..mutant.layers.len() {
            for j in 0..mutant.layers[i].get_biases().len(){
                let random = rng().random_range(min_change_pm..max_change_pm) / 1000.0;
                println!("Random: {}", random);
                let new_bias = self.layers[i].get_bias(j) + random;
                mutant.layers[i].set_bias(j, new_bias);
            }
            for j in 0..mutant.layers[i].get_weights().len(){
                let new_weight = self.layers[i].get_weight(j) + (rng().random_range(min_change_pm..max_change_pm) / 1000.0);
                mutant.layers[i].set_weight(j, new_weight);
            }
        }

        mutant
    }

    pub fn apply_mutant(&mut self, mutant: Network, success: f32){
        for layer in mutant.layers.iter().enumerate() {
            for bias in layer.1.get_biases().iter().enumerate(){
                let own_bias = self.layers[layer.0].get_bias(bias.0);
                let delta_bias = bias.1 - own_bias;
                let change = delta_bias * success;
                self.layers[layer.0].set_bias(bias.0, own_bias + change);
            }
            for weight in layer.1.get_weights().iter().enumerate(){
                let own_weight = self.layers[layer.0].get_weight(weight.0);
                let delta_weight = weight.1 - own_weight;
                let change = delta_weight * success;
                self.layers[layer.0].set_weight(weight.0, own_weight + change);
            }
        }

    }

    pub fn get_outputs(&self) -> Vec<f32> {self.output.clone() }

    pub fn perform_ppo_step(&mut self, probe_rate: f32, step_multiplier: f32, patch_size: usize){
        // First make a mutant
        let mut mutant = self.make_mutated_version(probe_rate);

        let mut success: f32 = 0.0;

        let mut mutant_success: f32 = 0.0;
        let mut self_success: f32 = 0.0;
        // Select patch_size elements from the training data set
        for i in 0..patch_size{
            // Select some data
            let selected_data_pos = i;//rand::random_range(0..self.dataset.len());
            let training_data = self.dataset[selected_data_pos].clone();
            println!("Training data at index {}, which has inputs: {:?}, outputs: {:?}", selected_data_pos, training_data.get_inputs(), training_data.get_outputs());
            // Execute the networks
            mutant.execute(training_data.get_inputs());
            self.execute(training_data.get_inputs());
            // Compare the mutant to the old network
            let mutant_result = mutant.get_outputs();
            let self_result = self.get_outputs();

            for i in 0..mutant_result.len() {
                let mutant_difference = (training_data.get_outputs()[i] - mutant_result[i]).abs();//1.0/(1.0 + E.powf(training_data.get_outputs()[i] - mutant_result[i] + 0.5));
                let self_difference = (training_data.get_outputs()[i] - self_result[i]).abs();//1.0/(1.0 + E.powf(training_data.get_outputs()[i] - self_result[i]));
                print!("difference self: {}, mutant: {}, expected result: {}, mutant: {}", self_difference, mutant_difference, training_data.get_outputs()[i], mutant_result[i]);

                mutant_success += mutant_difference;//(i as f32 * mutant_success + mutant_difference)/(i + 1) as f32;
                self_success += self_difference//(i as f32 * mutant_success + self_difference)/(i + 1) as f32;
            }
        }

        println!("success: {}, mutant: dif total: {}, self dif total: {}", success, mutant_success, self_success);
        // Now apply it

        if self_success > mutant_success {
            self.layers = mutant.layers.iter().map(|x| x.clone()).collect();
            println!("changed that stuff");
        }

        //self.apply_mutant(mutant, success * step_multiplier);
        //self.apply_mutant(mutant, success);
    }

    pub fn print_network_info(&self){
        for layer in self.layers.iter().enumerate() {
            println!("--- Layer {} ---", layer.0);
            for bias in layer.1.get_biases().iter().enumerate(){
                print!("Bias{}: {}, ", bias.0, bias.1 );
            }
            println!();
            for weight in layer.1.get_weights().iter().enumerate(){
                print!("Weight{}: {}, ", weight.0, weight.1 );
            }
            println!();
        }
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Network{layers: self.layers.iter().map(|x| (*x).clone()).collect(), output: self.output.clone(), dataset: self.dataset.clone()}
    }
}