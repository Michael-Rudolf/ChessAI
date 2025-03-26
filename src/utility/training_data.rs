pub struct TrainingData {
    inputs: Vec<f32>,
    outputs: Vec<f32>,
    relevance: f32,
}

impl TrainingData {
    pub fn new(inputs: Vec<f32>, outputs: Vec<f32>) -> TrainingData {
        TrainingData{inputs, outputs, relevance: 1.0}
    }

    pub fn get_inputs(&self) -> Vec<f32> {
        self.inputs.iter().map(|x| x.clone()).collect()
    }

    pub fn get_outputs(&self) -> Vec<f32> {
        self.outputs.iter().map(|x| x.clone()).collect()
    }

    pub fn get_relevance(&self) -> f32 {
        self.relevance
    }
}

impl Clone for TrainingData {
    fn clone(&self) -> Self {
        TrainingData{inputs: self.inputs.clone(), outputs: self.outputs.clone(), relevance: self.relevance}
    }
}