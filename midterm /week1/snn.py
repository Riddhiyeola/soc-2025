import numpy as np

# Neuron Parameters
class LIFNeuron:
    def __init__(self, threshold, reset_value, decay_factor, refractory_period):
        self.threshold = threshold
        self.reset_value = reset_value
        self.decay_factor = decay_factor
        self.refractory_period = refractory_period
        self.membrane_potential = 0
        self.spike_time = -1
        self.refractory_end_time = -1

    def update(self, incoming_spikes, current_time):
        if current_time < self.refractory_end_time:
            return False
        
        self.membrane_potential *= self.decay_factor
        self.membrane_potential += np.sum(incoming_spikes)
        
        if self.membrane_potential >= self.threshold:
            self.spike_time = current_time
            self.membrane_potential = self.reset_value
            self.refractory_end_time = current_time + self.refractory_period
            return True
        return False

# Synapse Parameters
class Synapse:
    def __init__(self, weight):
        self.weight = weight

# Spike-Timing-Dependent Plasticity (STDP)
def stdp(pre_spike_time, post_spike_time, weight, learning_rate, tau_positive, tau_negative):
    if pre_spike_time > 0 and post_spike_time > 0:
        delta_t = post_spike_time - pre_spike_time
        if delta_t > 0:
            return weight + learning_rate * np.exp(-delta_t / tau_positive)
        else:
            return weight - learning_rate * np.exp(delta_t / tau_negative)
    return weight

# Simulation Parameters
time_steps = 100
input_size = 5
hidden_size = 3
output_size = 1

# Network Initialization
input_neurons = [LIFNeuron(threshold=1.0, reset_value=0.0, decay_factor=0.9, refractory_period=2) for _ in range(input_size)]
hidden_neurons = [LIFNeuron(threshold=1.0, reset_value=0.0, decay_factor=0.9, refractory_period=2) for _ in range(hidden_size)]
output_neurons = [LIFNeuron(threshold=1.0, reset_value=0.0, decay_factor=0.9, refractory_period=2) for _ in range(output_size)]

input_to_hidden_synapses = np.random.rand(input_size, hidden_size)
hidden_to_output_synapses = np.random.rand(hidden_size, output_size)

learning_rate = 0.01
tau_positive = 20
tau_negative = 20

# Spike Train Pattern to Detect
pattern = [1, 0, 1, 0, 1]

# Simulation Loop
for t in range(time_steps):
    # Generate input spike trains (random for this example)
    input_spikes = np.random.randint(0, 2, size=input_size)
    
    # Update input neurons
    hidden_spikes = np.zeros(hidden_size)
    for i, neuron in enumerate(input_neurons):
        if neuron.update(input_spikes[i] * input_to_hidden_synapses[i], t):
            hidden_spikes += input_to_hidden_synapses[i]
    
    # Update hidden neurons
    output_spikes = np.zeros(output_size)
    for j, neuron in enumerate(hidden_neurons):
        if neuron.update(hidden_spikes[j] * hidden_to_output_synapses[j], t):
            output_spikes += hidden_to_output_synapses[j]
    
    # Update output neurons
    for k, neuron in enumerate(output_neurons):
        neuron.update(output_spikes[k], t)
    
    # STDP Learning
    for i in range(input_size):
        for j in range(hidden_size):
            input_to_hidden_synapses[i, j] = stdp(input_neurons[i].spike_time, hidden_neurons[j].spike_time, input_to_hidden_synapses[i, j], learning_rate, tau_positive, tau_negative)
    for j in range(hidden_size):
        for k in range(output_size):
            hidden_to_output_synapses[j, k] = stdp(hidden_neurons[j].spike_time, output_neurons[k].spike_time, hidden_to_output_synapses[j, k], learning_rate, tau_positive, tau_negative)

    # Check if pattern is detected
    if all(neuron.spike_time == t for neuron, pat in zip(input_neurons, pattern) if pat == 1):
        print(f"Pattern detected at time step {t}")