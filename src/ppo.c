#include "ppo.h"

PPO* create_ppo(Env* env, ActivationFunction* activation_functions, int* layer_sizes, int num_layers, int buffer_size, float gamma, float lambda, float epsilon, float ent_coeff, float init_std) {

    PPO* ppo = (PPO*)malloc(sizeof(PPO));
    ppo->buffer = create_trajectory_buffer(buffer_size, env->state_size, env->action_size);
    ppo->policy = create_gaussian_policy(layer_sizes, activation_functions, num_layers, init_std);
    ppo->V = create_neural_network(layer_sizes, activation_functions, num_layers);
    ppo->env = env;
    ppo->gamma = gamma;
    ppo->lambda = lambda;
    ppo->epsilon = epsilon;
    ppo->ent_coeff = ent_coeff;
    return ppo;
}

void free_ppo(PPO* ppo) {
    free_trajectory_buffer(ppo->buffer);
    free_gaussian_policy(ppo->policy);
    free_neural_network(ppo->V);
    free(ppo);
}

float clamp(float d, float min, float max) {
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

void collect_trajectories(TrajectoryBuffer* buffer, Env* env, GaussianPolicy* policy, int steps) {
    env->reset_env(buffer->state(buffer, buffer->idx));

    for (int i = 0; i < steps; i++) {
        sample_action(policy, buffer->state(buffer, buffer->idx), buffer->action(buffer, buffer->idx), buffer->logprob(buffer, buffer->idx), 1);

        env->step_env(buffer->action(buffer, buffer->idx), buffer->next_state(buffer, buffer->idx), buffer->reward(buffer, buffer->idx), buffer->terminated(buffer, buffer->idx), buffer->truncated(buffer, buffer->idx), buffer->action_size);

        int new_idx = (buffer->idx + 1) % buffer->capacity;

        if (i < steps - 1) {
            if (*buffer->truncated(buffer, buffer->idx) || *buffer->terminated(buffer, buffer->idx)) {
                env->reset_env(buffer->state(buffer, new_idx));
            } else {
                memcpy(buffer->state(buffer, new_idx), buffer->next_state(buffer, buffer->idx), buffer->state_size * sizeof(float));
            }
        } else {
            if (!*buffer->terminated(buffer, buffer->idx)) {
                *buffer->truncated(buffer, buffer->idx) = true;
            }
        }

        buffer->idx = new_idx;
        buffer->full = buffer->full || buffer->idx == 0;
    }
}


float policy_loss_and_grad(float* grad, float* adv, float* logprobs,  float* old_logprobs, float entropy, float ent_coeff, int epsilon, int m) {
    
    float loss = 0;

    // float ration[m];
    // float clipped_ratio[m];
    float ratio;
    float clipped_ratio;

    for (int i = 0; i < m; i++) {
        ratio = exp(logprobs[i] - old_logprobs[i]);

        bool adv_pos = adv[i] > 0;
        bool ratio_pos = ratio > 1 + epsilon;
        bool ratio_neg = ratio < 1 - epsilon;

        loss -= adv[i] * (adv_pos * (ratio_pos * (1 + epsilon) + !ratio_pos * ratio) + !adv_pos * (ratio_neg * (1 - epsilon) + !ratio_neg * ratio)) ;

        grad[i] = -(adv_pos * !ratio_pos * ratio + !adv_pos * !ratio_neg * ratio) * adv[i] / m;

        // clipped_ratio = clamp(ratio, 1 - epsilon, 1 + epsilon);

        // loss -= minf(ratio * adv[i], clipped_ratio * adv[i]);

        // grad[i] = -minf(1 / clipped_ratio, 1) * adv[i];
    }
    loss /= m;

    loss -= ent_coeff * entropy;

    return loss;

}


void compute_gae(NeuralNetwork* V, TrajectoryBuffer* buffer, float* v_target, float gamma, float lambda) {
    int limit = buffer->full ? buffer->capacity : buffer->idx;
    
    float v_next[limit];
    forward_propagation(V, buffer->next_state_p, limit);
    memcpy(v_next, V->output, limit * sizeof(float));
    
    // Save intermediate outputs for grad computation
    float v[limit];
    forward_propagation(V, buffer->state_p, limit);
    memcpy(v, V->output, limit * sizeof(float));

    float delta[limit];

    for (int i = 0; i < limit; i++) {
        delta[i] = *buffer->reward(buffer, i) + gamma * v_next[i] * !*buffer->terminated(buffer, i) - v[i];
    }

    // TODO Fix this
    for (int i = limit - 1; i >= 0; i--) {
        *buffer->advantage(buffer, i) = delta[i] + gamma * lambda * !*buffer->truncated(buffer, i) * *buffer->advantage(buffer, i + 1);
    }

    for (int i = 0; i < limit; i++) {
        v_target[i] = v[i] + *buffer->advantage(buffer, i);
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
    float adv[batch_size];
    float policy_loss_grad[batch_size];

    float v_target[batch_size * num_batches];
    float v_loss_grad[batch_size * num_batches];


    for (int i = 0; i < epochs; i++) {
        collect_trajectories(ppo->buffer, ppo->env, ppo->policy, batch_size * num_batches);

        // Compute advantages
        // TODO Use whole buffer not batch
        compute_gae(ppo->V, ppo->buffer, v_target, ppo->gamma, ppo->lambda);

        // Fit value function
        float v_loss = mean_squared_error(ppo->V->output, v_target, batch_size, 1);

        mean_squared_error_derivative(v_loss_grad, ppo->V->output, v_target, batch_size, 1);
        
        backward_propagation(ppo->V, v_loss_grad, batch_size);
        // UPDATE WEIGHTS


        for (int j = 0; j < num_batches; j++) {

            sample_batch(ppo->buffer, batch_size, states, actions, rewards, next_states, terminated, truncated, logprobs_old, adv);

            // Compute policy loss
            // SETS VALUES FOR GRAD COMPUTATION
            compute_log_prob(ppo->policy, logprobs, states, actions, batch_size);

            float entropy = compute_entropy(ppo->policy);

            float policy_loss = policy_loss_and_grad(policy_loss_grad, adv, logprobs, logprobs_old, entropy, ppo->ent_coeff, ppo->epsilon, batch_size);

            log_prob_backwards(ppo->policy, policy_loss_grad, batch_size);

            printf("Policy Loss: %f\n", policy_loss);
            printf("Value Loss: %f\n", v_loss);
          
            // Print weight gradients of policy mu network
            for (int l = 0; l < ppo->policy->mu->num_layers; l++) {
                printf("Policy Layer %d Weights: ", l);
                for (int m = 0; m < ppo->policy->mu->layers[l].input_size * ppo->policy->mu->layers[l].output_size; m++) {
                    printf(" %f", ppo->policy->mu->layers[l].grad_weights[m]);
                }
                printf("\n ------------------- \n");
            }

            // Update Policy
        }
    }
}
