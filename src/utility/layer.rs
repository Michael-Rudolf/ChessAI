pub trait Layer {
    fn set_inputs(&mut self, inputs: Vec<f32>);
    fn execute(&mut self);
    fn get_outputs(&self) -> Vec<f32>;
    fn output_count(&self) -> usize;
    fn input_count(&self) -> usize;
    fn set_weights(&mut self, weights: Vec<f32>);
    fn set_biases(&mut self, biases: Vec<f32>);
}