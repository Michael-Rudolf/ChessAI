pub trait Layer {
    fn set_inputs(&mut self, inputs: Vec<f32>);
    fn execute(&mut self);
    fn get_outputs(&self) -> Vec<f32>;
}