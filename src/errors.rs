use rand::Rng;
use crate::simulation_error::ErrorType; //ErrorType è IMPORTATO!!

//funzione che applica l'errore, riceve il valore da modificare e ne modifica un bit in base a quale ErrorType c'è
pub fn modify_weight_based_on_error(original_value: &mut f64, error_type: &ErrorType, index:usize) -> Option<usize> {
    match error_type {
        ErrorType::StuckAt0 => stuck_at_0(original_value,index),
        ErrorType::StuckAt1 => stuck_at_1(original_value, index),
        ErrorType::BitFlip => bit_flip(original_value),
    }
}

pub fn stuck_at_x(value: &mut f64, index: usize, new_bit: u8) -> Option<usize> {
    // Convert f64 to byte array
    let mut value_bytes: [u8; 8] = value.to_ne_bytes();

    // Ensure the index is within bounds
    if index < 64 {
        // Calculate the byte index and bit index within the byte
        let byte_index = index / 8;
        let bit_index = index % 8;

        // Create a mask with a 1 at the bit position
        let bit_mask = 1 << bit_index;

        // Clear the specific bit
        value_bytes[byte_index] &= !bit_mask;

        // Set the new bit
        value_bytes[byte_index] |= (new_bit & 1) << bit_index;

        // Convert byte array back to f64
        *value = f64::from_ne_bytes(value_bytes);
        println!("valore aggiornato {:.14} ", value);
        // Return the modified bit index
        Some(index)
    } else {
        None
    }
}

pub fn stuck_at_0(original_value: &mut f64, index: usize) -> Option<usize> {
    // Generate a random bit index to toggle (0 to 63)
    println!("valore {:.14} all'indice {}", original_value, index);
    // Toggle the bit using the byte array with the value 1
    stuck_at_x(original_value, index, 0)
}

pub fn stuck_at_1(original_value: &mut f64, index: usize) -> Option<usize> {
    // Generate a random bit index to toggle (0 to 63)
    println!("valore {:.14} all'indice {}", original_value, index);
    // Toggle the bit using the byte array with the value 1
    stuck_at_x(original_value, index, 1)
}

pub fn bit_flip(value: &mut f64) -> Option<usize> {
    // Generate a random bit index to toggle (0 to 63)
    let index = rand::thread_rng().gen_range(0..64);
    println!("valore {:.14} all'indice {}", value, index);

    // Convert f64 to u64
    let mut value_bits: u64 = unsafe { std::mem::transmute_copy(value) };

    // Toggle the bit using XOR with 1
    value_bits ^= 1u64 << index;

    // Convert back to f64
    *value = unsafe { std::mem::transmute_copy(&value_bits) };
    println!("valore aggiornato bitflip {:.14} ", value);
    Some(index)
}

