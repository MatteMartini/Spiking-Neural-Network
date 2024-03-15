// lif_neuron.rs
use std::io;


use rand::Rng;

use crate::{
    simulation_error::{Component, ErrorType},
    errors::modify_weight_based_on_error,
};

const RESET_POTENTIAL: f64 = 0.7;
const RESTING_POTENTIAL: f64 = 2.0;
const THRESHOLD: f64 = 2.5;
const TAU: f64 = 1.0;

pub trait Neuron: 'static + Clone + Send + ModifyNeuron {
    type ClassNeuron: 'static + Sized + Clone + Sync + Send;

    fn handle_spike(&mut self, sum: f64, current_spike_time: u128) -> u128;
    fn adjust_weight(&mut self, input: f64);
}


#[derive(Clone, Copy, Debug)]
pub struct Error {
    // struttura gestione errore
    pub flag: bool,  //indica se l'errore deve essere preso in considerazione o meno. potrebbe essere utilizzato per gestire la presenza o l'assenza di un errore senza dover rimuovere fisicamente l'oggetto Error dalla lista degli errori.
    pub error_type: ErrorType, //cioe se è stuck at x o bitflip
    pub index: Option<usize>,  //Un'opzione di usize che indica l'indice del bit o del componente del neurone interessato dall'errore
    pub component: Option<Component>,  // Aggiunto campo Component (che ti dice all'interno di un neurone quale campo è (reset_potential, soglia ecc))
}

#[derive(Clone, Debug)]
pub struct LIFNeuron {
    pub membrane_potential: f64,
    pub reset_potential: f64,
    pub resting_potential: f64,
    pub threshold: f64,
    pub tau: f64,
    pub last_spike_time: u128,
    pub errors: Vec<Error>,  // Cambiato nome da error a errors e usato Vec<Error>
}

impl LIFNeuron {
    pub fn new(reset_potential: f64, resting_potential: f64, threshold: f64, tau: f64) -> Self {
        Self {
            membrane_potential: resting_potential,
            reset_potential,
            resting_potential,
            threshold,
            tau,
            last_spike_time: 0,
            errors: Vec::new(),

        }
    }

    pub fn default() -> Self {
        LIFNeuron {
            membrane_potential: RESTING_POTENTIAL,
            reset_potential: RESET_POTENTIAL,
            resting_potential: RESTING_POTENTIAL,
            threshold: THRESHOLD,
            tau: TAU,
            last_spike_time: 0,
            errors: Vec::new(),

        }
    }

    // Funzione di supporto per leggere l'input utente
    fn read_user_input() -> f64 {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Errore durante la lettura dell'input");
        input.trim().parse().expect("Impossibile convertire l'input in f64")
    }

    // Metodo per chiedere all'utente di inserire i valori
    pub fn from_user_input() -> Self {
        println!("Inserisci i valori del neurone:");

        println!("Reset Potential:");
        let reset_potential: f64 = Self::read_user_input();

        println!("Resting Potential:");
        let resting_potential: f64 = Self::read_user_input();

        println!("Threshold:");
        let threshold: f64 = Self::read_user_input();

        println!("Tau:");
        let tau: f64 = Self::read_user_input();

        //Inserire errori(?)

        LIFNeuron {
            membrane_potential: resting_potential,
            reset_potential,
            resting_potential,
            threshold,
            tau,
            last_spike_time: 0,
            errors: Vec::new(),
        }
    }

    // Funzione per ottenere un riferimento mutabile alla struttura Error
    pub fn get_error_mut(&mut self) -> &mut Vec<Error> {
        &mut self.errors
    }

    // Funzione per modificare la struttura Error (NON USATA)
    pub fn modify_error(error: &mut Error, error_type: &ErrorType, index: Option<usize>, component: Option<Component>) {
        error.flag = true;
        error.error_type = *error_type;
        error.index = index;
        error.component = component;
    }

    // Funzione per verificare se un errore è già presente nella lista degli errori
    fn is_error_already_present(errors: &[Error], error_type: &ErrorType, component: Component) -> bool {
        errors.iter().any(|err| err.error_type == *error_type && err.component == Some(component))
    }


  pub fn print_neuron_parameters(&self) {
        println!("Neuron Parameters:");
        println!("Membrane Potential: {:.14}", self.membrane_potential);
        println!("Reset Potential: {:.14}", self.reset_potential);
        println!("Resting Potential: {:.14}", self.resting_potential);
        println!("Threshold: {:.14}", self.threshold);
        println!("Tau: {:.14}", self.tau);
        println!("Last Spike Time: {}", self.last_spike_time);
        println!("Errors: {:?}", self.errors);
    }
}

impl Neuron for LIFNeuron {
    type ClassNeuron = LIFNeuron;
    //funzione che aggiorna il valore della tensioone di soglia. La funzione handle_spike (di LIF_Neuron) viene utilizzata per simulare il comportamento di un neurone in risposta a un impulso. 
    //Verifica se la somma degli impulsi in ingresso è uguale a zero. Se lo è, restituisce subito a 0 (pruning). Calcola il nuovo potenziale di membrana del neurone.
    //Finisce verificando se il potenziale di membrana supera la soglia (threshold).

    fn handle_spike(&mut self, sum: f64, current_spike_time: u128) -> u128 {
        // Questo if implementa l'event-based condition
        if sum == 0.0 {
            return 0;
        }

        let delta_t = (current_spike_time - self.last_spike_time) as f64;
        self.last_spike_time = current_spike_time;

        // Calcola il nuovo potenziale di membrana
        let expo = (-delta_t / self.tau).exp();
        let intermediate = (self.membrane_potential - self.resting_potential) * expo;
        self.membrane_potential = self.resting_potential + intermediate + sum;

        // Comparatore di soglia
        if self.membrane_potential > self.threshold {
            self.membrane_potential = self.reset_potential;
            1
        } else {
            0
        }
    }

    //prepara il neurone a ricevere il prossimo impulso (aggiornamento peso intra)
    fn adjust_weight(&mut self, input: f64) {
        self.membrane_potential = self.membrane_potential + input;
    }
    
}

pub trait ModifyNeuron {
    fn modify_parameters_neuron(&mut self, component: Component, error_type: &ErrorType);
    fn apply_old_errors(&mut self);
}

impl ModifyNeuron for LIFNeuron{
    fn modify_parameters_neuron(&mut self, component: Component, error_type: &ErrorType) {
        
        // Verifica se l'errore è già presente nella lista
        if !Self::is_error_already_present(&self.errors, error_type, component) {
        let mut index: Option<usize> = None;
        let index_to_toggle = rand::thread_rng().gen_range(0..64);

        match component {
            Component::Threshold => { //anche se si chiama modify_weight è usata per modificare tutti i vari componenti del neurone
                index = modify_weight_based_on_error(&mut self.threshold, error_type,index_to_toggle);
            }
            Component::ResetPotential => {
                index = modify_weight_based_on_error(&mut self.reset_potential, error_type,index_to_toggle);
            }
            Component::RestingPotential => {
                index = modify_weight_based_on_error(&mut self.resting_potential, error_type,index_to_toggle);
            }
            Component::MembranePotential => {
                index = modify_weight_based_on_error(&mut self.membrane_potential, error_type,index_to_toggle);
            }
            Component::Tau => {
                index = modify_weight_based_on_error(&mut self.tau, error_type,index_to_toggle);
            }
            _ => {},
        }
        //LIFNeuron::modify_error(&mut self.errors[0], error_type, index, Some(component));
        //Se il tipo di errore non è BitFlip, aggiunge un nuovo errore alla lista degli errori associati al neurone. PERCHE DEVI RICORDARE SOLO GLI ERRRORI NON BITFLIP!
        //Questo nuovo errore contiene informazioni sull'errore (tipo e indice del bit o del componente modificato) e sul componente interessato.
        if *error_type != ErrorType::BitFlip {
            self.errors.push(Error {
            flag: true, //se è un Bitflip questo flag rimane a false e non verrà riapplicato
            error_type: *error_type,
            index: index,
            component: Some(component),
        });
        }}
    }

    //Questa funzione apply_old_errors si occupa di applicare gli errori memorizzati nella struttura LIFNeuron ai relativi componenti del neurone.
    fn apply_old_errors(&mut self) {
        let mut new_threshold = self.threshold;
        let mut new_reset_potential = self.reset_potential;
        let mut new_resting_potential = self.resting_potential;
        let mut new_membrane_potential = self.membrane_potential;
        let mut new_tau = self.tau;
    
        for error in &self.errors {
           // println!("applico l'errore \n");
            if error.flag {
                match error.component {
                    Some(component) => {
                        // Chiamata a modify_weight_based_on_error per ogni errore
                        match component {
                            Component::Threshold => {
                                modify_weight_based_on_error(&mut new_threshold, &error.error_type, self.errors[0].index.unwrap() );
                            }
                            Component::ResetPotential => {
                                modify_weight_based_on_error(&mut new_reset_potential, &error.error_type,self.errors[0].index.unwrap());
                            }
                            Component::RestingPotential => {
                                modify_weight_based_on_error(&mut new_resting_potential, &error.error_type,self.errors[0].index.unwrap());
                            }
                            Component::MembranePotential => {
                                modify_weight_based_on_error(&mut new_membrane_potential, &error.error_type,self.errors[0].index.unwrap());
                            }
                            Component::Tau => {
                                modify_weight_based_on_error(&mut new_tau, &error.error_type,self.errors[0].index.unwrap());
                            }
                            _ => {}
                        }
                    }
                    None => {}
                }
            }
        }
    
        // Aggiorna i campi della struttura con i nuovi valori
        self.threshold = new_threshold;
        self.reset_potential = new_reset_potential;
        self.resting_potential = new_resting_potential;
        self.membrane_potential = new_membrane_potential;
        self.tau = new_tau;
    }
    
}