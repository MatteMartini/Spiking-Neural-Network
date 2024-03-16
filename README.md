# Spiking-Neural-Network
This Rust library can create and resolve spiking neural networks defined for any possible applicable model, thanks to the powerful extensibility achieved through Rust's type system: simply implement the Model trait for your personally defined custom model and be good to go!

By default, the Leaky Integrate and Fire model is provided in the lif submodule.

In addition, the possibility of introducing some malfunctions such as the alteration of certain bits within the network modules is being studied.

# Resilience study
The study of resilience is supported by the possibility, once the network is built, to investigate potential failures in different components of the network: connections between neurons, memory areas that maintain numerical information (such as weights, thresholds, potentials, etc.), or internal processing blocks within the neuron (such as adders, multipliers, or threshold comparators). The possible failures, limited to this project, are single-bit faults, meaning that only one bit among all those susceptible to failure "breaks". The breakdown corresponds to a very specific and classifiable functional behavior as follows: stuck-at-0 (the bit remains fixed at 0, even if the opposite is requested) or stuck-at-1 (the bit remains fixed at 1, even if the opposite is requested), transient bit-flip (the value of the bit is inverted). 

The temporal nature of the faults identifies how it should be modeled: in the first two cases, whenever the bit is used, the corresponding stuck-at-X is applied, forcing the bit to the value X for the entire duration of the inference (i.e., ensuring the value X at each update/write), while in the third case (transient bit-flip), this forcing occurs only at a specific moment in time, and any subsequent new writings do not undergo variations.





