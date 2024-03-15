use std::fmt::{self, Display};
use std::sync::{Mutex, Arc};

use rand::Rng;

use crate::LIFNeuron;
use crate::lif_neuron::ModifyNeuron;
use crate::neural_network::NeuralNetwork;
use crate::spike::{Spike, action_spike};

#[derive(Clone, Debug)]
pub struct SimulationError {
    pub components: Vec<String>, //vettore di stringhe che contiene i nomi dei componenti della rete neurale che sono soggetti agli errori durante la simulazione.
    pub error_type: ErrorType, //UN SOLO TIPO DI ERRORE CON SIMULATIONERROR
    pub occurrences: usize,  //Il numero di occorrenze di questa simulazione.
    pub spikes_len: usize,  // La lunghezza degli spikes utilizzati durante la simulazione
    pub output: Vec<Vec<f64>>, //Un vettore di vettori di numeri in virgola mobile a precisione singola (f64) che rappresenta l'output della simulazione senza errori.
    pub output_errors: Vec<Vec<f64>>, //Un vettore di vettori di numeri in virgola mobile a precisione singola (f64) che rappresenta l'output della simulazione con errori.
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ErrorType {
    StuckAt0,
    StuckAt1,
    BitFlip,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Component { //dice dove sta l'errore
    Threshold,
    ResetPotential,
    RestingPotential,
    MembranePotential,
    Tau,
    Weights,
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl SimulationError {
    pub fn new(components: Vec<Component>, error_type: &str, occurrences: usize, spikes_len: usize) -> Self {
        let components = components.into_iter().map(|c| c.to_string()).collect();
        let error_type = match error_type.to_lowercase().as_str() {
            "stuck-at-0" => ErrorType::StuckAt0,
            "stuck-at-1" => ErrorType::StuckAt1,
            "bit-flip" => ErrorType::BitFlip,
            _ => panic!("Invalid error type"),
        };

        Self {
            components,
            error_type,
            occurrences,
            spikes_len,
            output: Vec::new(),
            output_errors: Vec::new(),
        }
    }

    pub fn print_info(&self) {
        println!("Error Type: {:?}", self.error_type);
        println!("Occurrences: {}", self.occurrences);
        println!("Components:");
        for component in &self.components {
            println!("  - {}", component);
        }
    
        // Stampa il contenuto di self.output
        println!("Output:");
        for row in &self.output {
            println!("  {:?}", row);
        }
    
        let mut output_sim_counter = 0;

        // Stampa il contenuto di self.output_errors
        println!("Output Errors:");
           

        for row in &self.output_errors {
            

          // Stampa "output sim x" ogni 10 stampe di row
        if output_sim_counter % 10 == 0 {
            println!("\n\n\n");
            println!("Output simulazione: {}", output_sim_counter / 10 +1 ); 
        }   
            output_sim_counter += 1;

            println!("  {:?}", row);
        }
    }
        
    
    

    pub fn string_to_component(nome: &str) -> Option<Component> {
        match nome.to_lowercase().as_str() {
            "threshold" => Some(Component::Threshold),
            "resetpotential" => Some(Component::ResetPotential),
            "restingpotential" => Some(Component::RestingPotential),
            "membranepotential" => Some(Component::MembranePotential),
            "tau" => Some(Component::Tau),
            "weights" => Some(Component::Weights),
            _ => None,
        }
    }
//Questa funzione gestisce l'esecuzione della simulazione con errori per un numero specificato di occorrenze(INTESO COME NUMERO DI SIMULAZIONI). 
//Utilizza un meccanismo di multithreading per eseguire le simulazioni in parallelo. 
//Per ogni occorrenza, viene creato un thread che esegue la funzione run_simulation_with_error, e i risultati di ogni simulazione vengono raccolti in un buffer condiviso. 
//Una volta completate tutte le simulazioni, stampa i risultati.
    pub fn run_simulation_wrapper(
        &self,
        layer_sizes: Vec<usize>,
        num_layers: usize,
        input_weights: Vec<Vec<Vec<f64>>>,
        intra_weights: Vec<Vec<Vec<f64>>>,
        neuron_params: LIFNeuron,
        spikes: Vec<Vec<Spike>>,
    ){
        // Esegui run_simulation_error num_occurrences volte
        let buffer = Arc::new(Mutex::new(Vec::<Vec<Vec<f64>>>::new()));
        let mut vt = Vec::new(); //vettore dei thread
        for _ in 0..self.occurrences.clone() {
            let buffer_clone = Arc::clone(&buffer);
            let cloned_self = self.clone();
            let layer_sizes_clone = layer_sizes.clone();
            let num_layers_clone = num_layers.clone();
            let input_weights_clone = input_weights.clone();
            let intra_weights_clone = intra_weights.clone();
            let neuron_params_clone = neuron_params.clone();
            let spikes_clone = spikes.clone();
            let mut rng = rand::thread_rng();
            let comp_index = rng.gen_range(0..self.components.clone().len());
            let selected_component = SimulationError::string_to_component(&self.components[comp_index].clone());
            vt.push(std::thread::spawn(move || {
            let data = cloned_self.run_simulation_with_error(
                layer_sizes_clone,
                num_layers_clone,
                input_weights_clone,
                intra_weights_clone,
                neuron_params_clone,
                spikes_clone,
                selected_component,
            );
            buffer_clone.lock().unwrap().push(data.clone());
            }));
        }

        for v in vt{ //aspettiamo le terminazioni dei thread
            v.join().unwrap();
        }

        let buffer_lock = buffer.lock().unwrap();
        for (idx, data) in buffer_lock.iter().enumerate() {
            println!("Iterazione #{}: ",idx+1);
            for inner_vec in data.iter() {
                println!("{:?}", inner_vec);
            }
        }
    }
    

//Questa funzione effettua la simulazione effettiva con errori. Crea una rete neurale utilizzando i parametri specificati e applica l'errore specificato ai componenti della rete. 
//Successivamente, esegue la simulazione temporale utilizzando gli spikes forniti.
//Durante ogni passaggio temporale, verifica se ci sono spikes e aggiorna di conseguenza lo stato della rete neurale. Infine, restituisce l'output della simulazione.
    pub fn run_simulation_with_error(
        &self,
        layer_sizes: Vec<usize>,
        num_layers: usize,
        input_weights: Vec<Vec<Vec<f64>>>,
        intra_weights: Vec<Vec<Vec<f64>>>,
        neuron_params: LIFNeuron,
        spikes: Vec<Vec<Spike>>,
        component: Option<Component>,
    ) -> Vec<Vec<f64>>{
        // Creazione rete neurale
        let mut network = NeuralNetwork::new(layer_sizes, input_weights, intra_weights, neuron_params);
    
        let sorted_spike_array_for_nn = Spike::get_all_spikes(spikes.clone());
        let max_value = *sorted_spike_array_for_nn.iter().max().unwrap();
        let mut time = 0;
    
    
        network.apply_error(component, self.error_type);
        println!("Component modificata: {}, type_error: {}", component.unwrap(), self.error_type);
        let mut output = Vec::new();
        while time < max_value {
            // Incrementa il contatore
            time += 1;
            let mut s = vec![0.0, 0.0];
            if self.error_type != ErrorType::BitFlip && component != Some(Component::Weights) {
                if let Some(&(l_index, n_index)) = network.errors_positions.iter().next() {
                    network.get_neuron_mut(l_index, n_index).map(|neuron| neuron.apply_old_errors());
                }
                
            }
            if sorted_spike_array_for_nn.contains(&time) {
                s = action_spike(spikes.clone(), time);
            }
            
            // Ciclo sui neuroni per calcolo soglia
            s = network.update_neurons_parallel(time, s, num_layers);
            output.push(s.clone());
        }
    output
    }
    
}
