
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Spike {
    // Stands for "time of the spike", and represents a timestamp of when the spike occurs
    pub spike_time: u128,
    // Index of the neuron this spike applies to inside its layer
    pub neuron_id: usize,
    // Index of the layer this spike applies
    pub layer_id: usize
}

impl Spike {
    // Create a new spike at time `ts` for neuron `neuron_id`
    pub fn new(spike_time: u128, neuron_id: usize, layer_id: usize) -> Spike {
        Spike {
            spike_time,
            neuron_id,
            layer_id
        }
    }

     // Get the spike time of the current spike
     pub fn get_spike_time(&self) -> u128 {
        self.spike_time
    }

    pub fn create_spike_vec(neuron_id: usize, layer_id: usize, ts_vec: Vec<u128>) -> Vec<Spike> {
        let mut spike_vec : Vec<Spike> = Vec::with_capacity(ts_vec.len()); //per tutti gli istanti nel vettore crei uno spike per un errore
        
        //Creating the Spikes array for a single Neuron
        for ts in ts_vec.into_iter() {
            spike_vec.push(Spike::new(ts, neuron_id, layer_id));
        }

        //Order the ts vector -> ordini gli spike in base al tempo in cui avvengono
        spike_vec.sort();

        spike_vec
    }

    //prende in input tutti gli spike della rete e restituisce un vettore con tutti gli istanti
    pub fn get_all_spikes(spikes: Vec<Vec<Spike>>) -> Vec<u128> {
        let mut res: Vec<u128> = Vec::new();

        for line in spikes {
            for spike in line {
                res.push(spike.get_spike_time());
            }
        }
        res.sort(); //ascending
    
        res
    }
}

//Questa funzione accetta un riferimento ad un vettore di Spike (spike_vec) e un tempo di picco (time).
//Itera attraverso ogni Spike nel vettore e controlla se il tempo di picco di quel Spike corrisponde al tempo specificato.
//Se trova un Spike con il tempo specificato, restituisce un riferimento ad esso all'interno di Some(spike). Se non trova corrispondenze, restituisce None.
pub fn contains_time<'a>(spike_vec: &'a [Spike], time: u128) -> Option<&'a Spike> {
    for spike in spike_vec.iter() {
        if spike.spike_time == time {
            return Some(spike);
        }
    }
    None
}

//Questa funzione accetta un vettore di vettori di Spike (spikes) e un tempo di picco (time). 
//Per ogni riga nel vettore esterno (cioè per ogni vettore di Spike), chiama la funzione contains_time per verificare se il tempo specificato esiste all'interno della riga corrente. 
//Se il tempo esiste, aggiunge 1.0 al vettore di output v, altrimenti aggiunge 0.0. Alla fine, restituisce il vettore v contenente i risultati (1.0 se il tempo è presente, altrimenti 0.0).
pub fn action_spike(spikes: Vec<Vec<Spike>>, time: u128) -> Vec<f64>{ 

    let mut v = vec![];
    for riga in spikes.iter() {
        match contains_time(&riga, time) {
            Some(_) => {
                v.push(1.0);
            }
            None => {
                v.push(0.0);
            }
        }
    }

    v
}