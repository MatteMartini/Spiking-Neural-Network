use rand::Rng;

// neural_layer.rs
use crate::lif_neuron::Neuron;
use crate::{simulation_error::ErrorType, errors::modify_weight_based_on_error};  

#[derive(Clone)]
pub struct NeuralLayer<N: Neuron> {
    /// Lista di tutti i neuroni in questo strato
    pub(crate) neurons: Vec<N>,
    /// Matrice dei pesi in ingresso. Per il primo layer, questa deve essere una matrice diagonale quadrata.
    pub(crate) input_weights: Vec<Vec<f64>>,
    /// Matrice quadrata dei pesi intra-layer
    pub(crate) intra_weights: Vec<Vec<f64>>,
}

impl<N: Neuron> NeuralLayer<N> {
//utilizzata dentro la new della neural network. In neuron ci sono i parametri da mettere nel neurone
     pub fn new(layer_size: usize, input_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>, neuron: N) -> NeuralLayer<N> {
        let neurons = vec![neuron; layer_size];

        NeuralLayer {
            neurons,
            input_weights,
            intra_weights,
        }
    }

    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }
    
    pub fn get_neuron(&self, neuron: usize) -> Option<&N> {
        self.neurons.get(neuron)
    }

    pub fn get_neuron_mut(&mut self, index_neuron: usize) -> Option<&mut N> {
        self.neurons.get_mut(index_neuron)
    }

    pub fn get_neurons(&self) -> Vec<N>{
        self.neurons.clone()
    }

    pub fn get_neurons_mut(&mut self) -> Vec<N>{
        self.neurons.clone()
    }

    pub fn get_intra_weight_value(&self, from: usize, to: usize) -> &f64 {
        self.intra_weights.get(from).unwrap().get(to).unwrap()
    }

    pub fn get_input_weight_value(&self, from: usize, to: usize) -> &f64 {
        self.input_weights.get(from).unwrap().get(to).unwrap()
    }

    pub fn print_intra_weights(&self) { 
        println!("Intra-layer weights matrix:"); 
        for row in &self.intra_weights { 
            for &value in row { 
                print!("{:.14}  ", value); // Modifica il formato di stampa come preferisci 
            } 
            println!(); // Nuova riga per ogni riga della matrice 
        } 
    } 
 
    pub fn print_input_weights(&self) { 
        println!("Input-layer weights matrix:"); 
        for row in &self.input_weights { 
            for &value in row { 
                print!("{:.14}  ", value); // Modifica il formato di stampa come preferisci 
            } 
            println!(); // Nuova riga per ogni riga della matrice 
        } 
    } 
 
    //modifica weights, si specifica nel menu se modificare weights. Se si decide weights, si decide in modo casuale tra input e intra quale modificare
    pub fn modify_weights_layer(&mut self, error_type: &ErrorType) {  //in ErrorType c'è scritto se c'è un bitflip o uno stack a 0 o 1
        let mut rng = rand::thread_rng(); //restituisce un generatore di numeri casuali associato al thread corrente. ogni thread ha il suo generatore di numeri casuali separato
    
        // Decide whether to modify intra_weights or input_weights 
        let modify_intra_weights = rng.gen_bool(0.5); 
    
        if modify_intra_weights { 
            // Modify intra_weights 
            self.print_intra_weights(); 
            self.modify_random_intra_weight(error_type); 
            self.print_intra_weights(); 
        } else { 
            // Modify input_weights 
            self.print_input_weights(); 
            self.modify_random_input_weight(error_type); 
            self.print_input_weights(); 
        } 
    } 
    
    pub fn modify_random_intra_weight(&mut self, error_type: &ErrorType) { 
        let mut rng = rand::thread_rng(); 
    
        // Scegli casualmente gli indici della matrice intra_weights 
        let from_index = rng.gen_range(0..self.intra_weights.len()); 
        let to_index = rng.gen_range(0..self.intra_weights[from_index].len()); 
        let index_to_toggle = rand::thread_rng().gen_range(0..64); //si sceglie in modo casuale il bit su cui applicare l'errore

        println!{"{} {}",from_index,to_index}; 
        //funziona che fa la modifica effettiva, STA IN ERROR.RS
        modify_weight_based_on_error(&mut self.intra_weights[from_index][to_index], error_type,index_to_toggle); 
    } 
    
    
    pub fn modify_random_input_weight(&mut self, error_type: &ErrorType) { 
        let mut rng = rand::thread_rng(); 
    
        // Scegli casualmente gli indici della matrice intra_weights 
        let from_index = rng.gen_range(0..self.input_weights.len()); 
        let to_index = rng.gen_range(0..self.input_weights[from_index].len()); 
        let index_to_toggle = rand::thread_rng().gen_range(0..64);

        println!{"{} {}",from_index,to_index}; 
        modify_weight_based_on_error(&mut self.input_weights[from_index][to_index], error_type,index_to_toggle); 
    }
}
