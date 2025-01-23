
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


void collect_trajectories(TrajectoryBuffer* buffer, Env* env, GaussianPolicy* policy, int steps) {

    env->reset_env(buffer->buffer[buffer->idx].state);
    for (int i = 0; i < steps; i++) {
        // float action[buffer->action_size];
        sample_action(policy, buffer->buffer[buffer->idx].state, buffer->buffer[buffer->idx].action, &buffer->buffer[buffer->idx].logprob, 1);

        env->step_env(buffer->buffer[buffer->idx].action, buffer->buffer[buffer->idx].next_state, &buffer->buffer[buffer->idx].reward, &buffer->buffer[buffer->idx].terminated, &buffer->buffer[buffer->idx].truncated, buffer->action_size);
        
        int new_idx = (buffer->idx + 1) % buffer->capacity;

        if (buffer->buffer[buffer->idx].truncated || buffer->buffer[buffer->idx].terminated) {
            env->reset_env(buffer->buffer[new_idx].state);
        } else {
            memcpy(buffer->buffer[new_idx].state, buffer->buffer[buffer->idx].next_state, buffer->state_size * sizeof(float));
        }

        buffer->idx = new_idx;
    }
}