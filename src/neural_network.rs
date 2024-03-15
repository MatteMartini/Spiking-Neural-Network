
use rand::Rng;

use crate::Component;
use crate::ErrorType;
use crate::lif_neuron::Neuron;
use crate::neural_layer::NeuralLayer;

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::Arc;
use std::thread;

#[derive(Clone)]
pub struct NeuralNetwork<N: Neuron> {
    /// All the sorted layers of the neural network
    pub layers: Vec<NeuralLayer<N>>,
    pub errors_positions: HashSet<(usize, usize)>,

}

impl<N: Neuron> NeuralNetwork<N> {

    pub fn new(layer_sizes: Vec<usize>, input_weights: Vec<Vec<Vec<f64>>>, intra_weights: Vec<Vec<Vec<f64>>>, neuron: N) -> NeuralNetwork<N> {
        let mut layers = Vec::with_capacity(layer_sizes.len());
    
        // Iterate over layer_sizes to create NeuralLayer instances
        for (index, size) in layer_sizes.iter().enumerate() {
            let input = input_weights.get(index).expect("out of bounds");
            let intra = intra_weights.get(index).expect("out of bounds");
            // Create a new NeuralLayer with the current size, next size, and neuron
            let neural_layer = NeuralLayer::new(*size, input.to_vec(), intra.to_vec(), neuron.clone());
            
            // Push the created NeuralLayer into the layers vector
            layers.push(neural_layer);
        }
    
        // Create and return the NeuralNetwork with the populated layers vector
        NeuralNetwork { layers, errors_positions: HashSet::new() }
    }

    pub fn get_layer_mut(&mut self, index_layer: usize) -> Option<&mut NeuralLayer<N>> {
        self.layers.get_mut(index_layer)
    }

    // Get a reference to a specific neuron
    pub fn get_neuron_mut(&mut self, index_layer: usize, index_neuron: usize) -> Option<&mut N> {
        let l = self.get_layer_mut(index_layer).unwrap();
        let s = l.get_neuron_mut(index_neuron);
        return s;
    }

    pub fn update_neurons_parallel(&mut self, time_step: u128, spike_input: Vec<f64>, num_layers: usize) -> Vec<f64> {
        let mut layer_spikes = Vec::new();
    
        for layer_idx in 0..num_layers {
            if layer_idx == 0 {
                let mut spike_buffer = Vec::<f64>::new();
    
                for neuron_idx in 0..self.layers.get(layer_idx).unwrap().num_neurons() {
                    let layer= self.layers.get_mut(layer_idx).unwrap();
                    let spikes_clone = spike_input.clone();
    
                    let neuron_potential = layer.get_neuron_mut(neuron_idx)
                        .unwrap()
                        .handle_spike(*spikes_clone.get(neuron_idx).unwrap(), time_step) as f64;
    
                    spike_buffer.push(neuron_potential);
                    drop(layer);
                }

                layer_spikes = spike_buffer.clone();
    
            } 
            else 
            {
                // Aggiornamento degli altri layer
                let mut spike_buffer = Vec::<f64>::new();
    
                for current_neuron_idx in 0..self.layers.get(layer_idx).unwrap().num_neurons() {
                    let current_layer = self.layers.get(layer_idx).unwrap();
                    let layer_spikes_clone = layer_spikes.clone();
    
                    let mut weighted_sum = 0.0;

                    for other_neuron_idx in 0..current_layer.num_neurons() {
                        weighted_sum += layer_spikes_clone.get(other_neuron_idx).unwrap()
                            * current_layer.get_input_weight_value(current_neuron_idx, other_neuron_idx);
                    }

                    let current_layer = self.layers.get_mut(layer_idx).unwrap();

                    let neuron_potential = current_layer.get_neuron_mut(current_neuron_idx)
                        .unwrap()
                        .handle_spike(weighted_sum, time_step) as f64;

                    spike_buffer.push(neuron_potential);
                }
    
                layer_spikes = spike_buffer.clone();
            }
    
                let internal_spike_buffer = Arc::new(Mutex::new(vec![0.0; self.layers.get(layer_idx).unwrap().num_neurons()]));
                let mut vt =  Vec::new(); //vettore dei thread
    
                for current_neuron_idx in 0..self.layers.get(layer_idx).unwrap().num_neurons() {
                    let internal_spike_buffer_clone = Arc::clone(&internal_spike_buffer);
                    let current_layer = self.layers.get(layer_idx).unwrap().clone();
                    let layer_spikes_clone = layer_spikes.clone();  
                    vt.push(thread::spawn(move || {
                        for other_neuron_idx in 0..current_layer.num_neurons() {
                            internal_spike_buffer_clone.lock().unwrap()[other_neuron_idx] = 
                                layer_spikes_clone.get(current_neuron_idx).unwrap()
                                    * current_layer.get_intra_weight_value(current_neuron_idx, other_neuron_idx);
                        }
                    }));
                }
                
                for v in vt{ //aspettiamo le terminazioni dei thread
                    v.join().unwrap();
                }

                let internal_spike_len;
                {
                    internal_spike_len = internal_spike_buffer.lock().unwrap().len();
                }
    
                for neuron_idx in 0..internal_spike_len {
                    self.layers.get_mut(layer_idx).unwrap().neurons.get_mut(neuron_idx).unwrap()
                        .adjust_weight(*internal_spike_buffer.lock().unwrap().get(neuron_idx).unwrap());
                }
                drop(internal_spike_buffer);
            }
        
    
        return layer_spikes;
    }

//La funzione apply_error viene chiamata per applicare un errore a un componente specifico della rete neurale. Prende due argomenti:
//components: Un'opzione che rappresenta il componente della rete neurale a cui applicare l'errore. 
//Può essere Some(Component) se l'errore deve essere applicato a un componente specifico (come i pesi) o None se l'errore deve essere applicato casualmente a un componente e a un neurone.
//error_type: Il tipo di errore da applicare.
//La funzione genera un numero casuale per scegliere casualmente un layer o un neurone all'interno della rete neurale. 
//Se il parametro components è Some(Component::Weights), allora viene scelto casualmente un layer per applicare l'errore sui pesi utilizzando il metodo modify_weights_layer. 
//Se components contiene un'opzione diversa da Weights, viene scelto casualmente un layer e un neurone, e viene applicato l'errore al componente specificato del neurone utilizzando il metodo modify_parameters_neuron.

    pub fn apply_error(&mut self, components: Option<Component>, error_type: ErrorType) {
        let mut rng = rand::thread_rng();
        let num_layers = self.layers.len();
    
        match components {
            Some(Component::Weights) => {
                // Scelta casuale di un layer per i pesi
                let layer_index = rng.gen_range(0..num_layers);
                self.get_layer_mut(layer_index).unwrap().modify_weights_layer(&error_type);
            }
            Some(_) => {
                // Scelta casuale di un layer e un neurone per gli altri componenti
                let layer_index = rng.gen_range(0..num_layers);
                let neuron_index = rng.gen_range(0..self.layers[layer_index].num_neurons());
                if let Some(component) = components {
                    
                    let position = (layer_index, neuron_index);
                    self.get_neuron_mut(layer_index, neuron_index)
                        .unwrap()
                        .modify_parameters_neuron(component, &error_type);
                    self.errors_positions.insert(position);
                }
            }
            None => {}
        }
    }
    
}   
