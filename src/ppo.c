
#include "ppo.h"

PPO* create_ppo(Env* env, ActivationFunction* activation_functions, int* layer_sizes, int num_layers, int buffer_size, float gamma, float lambda, float epsilon, float init_std) {

    PPO* ppo = (PPO*)malloc(sizeof(PPO));
    ppo->buffer = create_trajectory_buffer(buffer_size, env->state_size, env->action_size);
    ppo->policy = create_gaussian_policy(layer_sizes, activation_functions, num_layers, init_std);
    ppo->V = create_neural_network(layer_sizes, activation_functions, num_layers);
    ppo->env = env;
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

        if (i < steps - 1){
            if (buffer->buffer[buffer->idx].truncated || buffer->buffer[buffer->idx].terminated) {
                env->reset_env(buffer->buffer[new_idx].state);
            } else {
                memcpy(buffer->buffer[new_idx].state, buffer->buffer[buffer->idx].next_state, buffer->state_size * sizeof(float));
            }
        } else {
            if (!buffer->buffer[buffer->idx].terminated) {
                buffer->buffer[buffer->idx].truncated = 1;
            }

        }

        buffer->idx = new_idx;
        buffer->full = buffer->full || buffer->idx == 0;
    }
}


float policy_loss(float* adv, float* logprobs,  float* old_logprobs, int epsilon, int m) {
    


}


void compute_gae(NeuralNetwork* V, float* v_target, float* adv, float* state, float* reward, float* next_state, bool* terminated, bool* truncated, float gamma, float lambda, int m) {
    float v_next[m];
    forward_propagation(V, next_state, m);
    memcpy(v_next, V->output, m * sizeof(float));
    
    // Save intermediate outputs for grad computation
    float v[m];
    forward_propagation(V, state, m);
    memcpy(v, V->output, m * sizeof(float));

    float delta[m];

    for (int i = 0; i < m; i++) {
        delta[i] = reward[i] + gamma * v_next[i] * !terminated[i] - v[i];
    }

    for (int i = m - 1; i >= 0; i--) {
        adv[i] = delta[i] + gamma * lambda * !truncated[i] * adv[i + 1];
    }

    for (int i = 0; i < m; i++) {
        v_target[i] = v[i] + adv[i];
    }
}



void train_ppo(PPO* ppo, int epochs, int batch_size, int num_batches) {
    float states[batch_size * ppo->buffer->state_size];
    float actions[batch_size * ppo->buffer->action_size];
    float rewards[batch_size];
    float next_states[batch_size * ppo->buffer->state_size];
    bool terminated[batch_size];
    bool truncated[batch_size];
    float logprobs[batch_size];
    float logprobs_old[batch_size];

    float v_target[batch_size];
    float adv[batch_size];
    float v_loss_grad[batch_size];

    for (int i = 0; i < epochs; i++) {
        collect_trajectories(ppo->buffer, ppo->env, ppo->policy, batch_size * num_batches);

        for (int j = 0; j < num_batches; j++) {

            sample_batch(ppo->buffer, batch_size, states, actions, rewards, next_states, terminated, truncated, logprobs_old);

            // Compute advantages
            compute_gae(ppo->V, v_target, adv, states, rewards, next_states, terminated, truncated, ppo->gamma, ppo->lambda, batch_size);

            // Fit value function
            float v_loss = mean_squared_error(ppo->V->output, v_target, batch_size, 1);

            mean_squared_error_derivative(v_loss_grad, ppo->V->output, v_target, batch_size, 1);
            
            backward_propagation(ppo->V, v_loss_grad, batch_size);
            // UPDATE WEIGHTS


            // Compute policy loss
            // SETS VALUES FOR GRAD COMPUTATION
            compute_log_prob(ppo->policy, logprobs, states, actions, batch_size);

            // float policy_loss = policy_loss(ppo->policy, states, actions, adv, logprobs_old, batch_size);

            // Update Policy
        }
    }
}
