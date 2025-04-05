use crate::utility::layer::Layer;
use crate::utility::network::Network;
use crate::utility::neuronal_layer::neuronal_layer::{ActivationFunction, NeuronalLayer};
use crate::utility::training_data::TrainingData;

pub(crate) fn test() {
    let mut network = Network::new();
    network
        .add_layer(NeuronalLayer::new(2, 4, ActivationFunction::Linear))
        .add_layer(NeuronalLayer::new(4, 3, ActivationFunction::Sigmoid))
        .add_layer(NeuronalLayer::new(3, 1, ActivationFunction::Sigmoid));


    network.execute(vec![0.1, 1.0]);
    println!("0^1: {:?}", network.get_outputs());
    network.execute(vec![1.0, 1.0]);
    println!("1^1: {:?}", network.get_outputs());
    network.execute(vec![1.0, 0.1]);
    println!("1^0: {:?}", network.get_outputs());
    network.execute(vec![0.1, 0.1]); //0
    println!("0^0: {:?}", network.get_outputs());


    for i in 0..2 {
        for j in 0..2 {
            let should_return_true = (i == 1 && j == 0) || (i == 0 && j == 1); //|| (i == 0 && j == 1);// || (i == 0 && j == 1);// j == 0);// || (i == 0 && j == 1);//i == 1;//j == 1 && i == 0;//(i == 1 && j == 0) || (i == 0 && j == 1);
            println!("should_return_true: {}, i: {}, j: {}", should_return_true, i, j);
            let something = if should_return_true { 1f32 } else { 0.1f32 };
            let training_data = TrainingData::new(vec![i as f32, j as f32], vec![something]);
            network.add_training_data(training_data);
        }
    }
    network
        .add_training_data(TrainingData::new(vec![1.0, 1.0], vec![0.0]));

    network
        .add_training_data(TrainingData::new(vec![1.0, 0.0], vec![1.0]));


    /**/for _ in 0..20000 {
        network.perform_ppo_step(0.1, 10.0, 6);
    }
    /*
            for _ in 0..8_000{
                network.perform_ppo_step(0.05, 10.0,4);
            }
            for _ in 0..50000{
                network.perform_ppo_step(0.01, 1.0,4);
            }*/

    network.print_network_info();

    network.execute(vec![0.1, 1.0]); //1
    println!("0^1: {:?}", network.get_outputs());
    network.execute(vec![1.0, 1.0]); //0
    println!("1^1: {:?}", network.get_outputs());
    network.execute(vec![1.0, 0.1]); //1
    println!("1^0: {:?}", network.get_outputs());
    network.execute(vec![0.1, 0.1]); //0
    println!("0^0: {:?}", network.get_outputs());
}