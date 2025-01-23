
#include "ppo.h"

PPO* create_ppo(int capacity, int state_size, int action_size, int* layer_sizes, ActivationFunction* activation_functions, int num_layers, float gamma, float lambda, float epsilon, float init_std) {
    
    PPO* ppo = (PPO*)malloc(sizeof(PPO));
    ppo->buffer = create_trajectory_buffer(capacity, state_size, action_size);
    ppo->policy = create_gaussian_policy(layer_sizes, activation_functions, num_layers, init_std);
    ppo->V = create_neural_network(layer_sizes, activation_functions, num_layers);
    ppo->gamma = gamma;
    ppo->lambda = lambda;
    ppo->epsilon = epsilon;
    return ppo;
}

void free_ppo(PPO* ppo) {
    free_trajectory_buffer(ppo->buffer);
    free_gaussian_policy(ppo->policy);
    free_neural_network(ppo->V);
    free(ppo);
}
