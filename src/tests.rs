// // tests/test.rs
use crate::{errors::{bit_flip, stuck_at_x}, ErrorType, SimulationError, Component, spike::Spike, lif_neuron::LIFNeuron};

#[test]
    fn test_create_spike_vec() { //verifica il corretto funzionamento del metodo create_spike_vec della struttura Spike
        // Chiamare la funzione da testare con i parametri desiderati
        let neuron_id = 1;
        let layer_id = 1;
        let ts_vec = vec![1, 5 ,7];
        let result = Spike::create_spike_vec(neuron_id, layer_id, ts_vec.clone());

        // Assert per verificare che il risultato sia quello atteso
        assert_eq!(result.len(), ts_vec.len());

        // Verifica che gli spike siano ordinati per tempo
        for i in 1..result.len() {
            assert!(result[i - 1].spike_time <= result[i].spike_time);
        }

        // Verifica che gli spike abbiano i valori corretti
        for (i, &time) in ts_vec.iter().enumerate() {
            assert_eq!(result[i].neuron_id, neuron_id);
            assert_eq!(result[i].layer_id, layer_id);
            assert_eq!(result[i].spike_time, time);
        }
    }

    #[test]
    fn test_get_all_spikes() { //Test sulla funzione get_all_spikes(). Viene verificato che il risultato sia un vettore ordinato contenente tutti i tempi degli spike.
        // Creare un esempio di vettore di vettori di Spike
        let spikes = vec![
            vec![
                Spike::new(1, 0, 1),
                Spike::new(3, 1, 1),
                Spike::new(5, 2, 1),
            ],
            vec![
                Spike::new(2, 0, 2),
                Spike::new(4, 1, 2),
                Spike::new(6, 2, 2),
            ],
        ];

        // Chiamare la funzione da testare
        let result = Spike::get_all_spikes(spikes);

        // Verificare che il risultato sia ordinato e contenga tutti i tempi degli spike
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

#[test]
fn test_bit_flip() {
    // Test case 1: Flip the least significant bit
    let mut value = 42.0;
    bit_flip(&mut value);
    println!("{}",value);
    assert_ne!(value, 42.0 ); // 42.0 + 2^(-52)

    // Test case 3: Flip the most significant bit
    let mut value = 42.0;
    bit_flip(&mut value);
    println!("{}",value);
    assert_ne!(value, 42.0); // -2^62 (due a causa della rappresentazione a complemento a due)

    // Test case 4: Index out of bounds (no modification should occur)
    let mut value = 42.0;
    bit_flip(&mut value);
    println!("{}",value);
    assert_ne!(value, 42.0); // Value should remain unchanged
}

#[test]
fn test_simulation_error() { //ll tipo Component è una enumerazione definita nel progetto che sembra rappresentare i vari componenti di un neurone o di un sistema di simulazione.
    let components_list = vec![
        Component::Threshold,
        Component::ResetPotential,
        Component::RestingPotential,
        Component::MembranePotential,
        Component::Tau,
        Component::Weights,
    ];

    let simulation_error = SimulationError::new(components_list, "bit-flip", 3,0);

    assert_eq!(simulation_error.occurrences, 3);
    assert_eq!(simulation_error.error_type, ErrorType::BitFlip);
    assert_eq!(simulation_error.components.len(), 6);
}