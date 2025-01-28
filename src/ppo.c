#include "ppo.h"

PPO* create_ppo(Env* env, ActivationFunction* activation_functions, int* layer_sizes, int num_layers, int buffer_size, float lr_policy, float lr_v, float gamma, float lambda, float epsilon, float ent_coeff, float init_std) {

    PPO* ppo = (PPO*)malloc(sizeof(PPO));
    ppo->buffer = create_trajectory_buffer(buffer_size, env->state_size, env->action_size);
    ppo->policy = create_gaussian_policy(layer_sizes, activation_functions, num_layers, init_std);

    // TODO Set last layer to 1
    ppo->V = create_neural_network(layer_sizes, activation_functions, num_layers);
    ppo->adam_policy = create_adam_from_nn(ppo->policy->mu, 0.9, 0.999);
    ppo->adam_V = create_adam_from_nn(ppo->V, 0.9, 0.999);

    int length = 1;
    ppo->adam_entropy = create_adam(&ppo->policy->log_std, &ppo->policy->log_std_grad, &length, 1, 1,  0.9, 0.999);
    
    ppo->env = env;
    ppo->gamma = gamma;
    ppo->lambda = lambda;
    ppo->epsilon = epsilon;
    ppo->ent_coeff = ent_coeff;
    ppo->lr_policy = lr_policy;
    ppo->lr_V = lr_v;
    return ppo;
}

void free_ppo(PPO* ppo) {
    free_adam(ppo->adam_policy);
    free_adam(ppo->adam_V);
    free_adam(ppo->adam_entropy);
    free_trajectory_buffer(ppo->buffer);
    free_gaussian_policy(ppo->policy);
    free_neural_network(ppo->V);
    free(ppo);
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


float policy_loss_and_grad(float* grad_logprob, float* grad_entropy, float* adv, float* logprobs,  float* old_logprobs, float entropy, float ent_coeff, float epsilon, int m) {
    
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

        grad_logprob[i] = -(adv_pos * !ratio_pos * ratio + !adv_pos * !ratio_neg * ratio) * adv[i] / m;

        // printf("Grad: %f\n", grad_logprob[i]);

        // clipped_ratio = clamp(ratio, 1 - epsilon, 1 + epsilon);

        // loss -= minf(ratio * adv[i], clipped_ratio * adv[i]);

        // grad[i] = -minf(1 / clipped_ratio, 1) * adv[i];
    }
    loss /= m;

    loss -= ent_coeff * entropy;
    *grad_entropy = -ent_coeff;

    return loss;

}


void compute_gae(NeuralNetwork* V, TrajectoryBuffer* buffer, float gamma, float lambda) {
    int limit = buffer->full ? buffer->capacity : buffer->idx;
    
    float debug_rewars[limit];
    memcpy(debug_rewars, buffer->reward_p, limit * sizeof(float));

    float debug_actions[limit];
    memcpy(debug_actions, buffer->action_p, limit * sizeof(float));

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

    float sum = 0;
    for (int i = limit - 1; i >= 0; i--) {
        *buffer->advantage(buffer, i) = delta[i] + gamma * lambda * !(*buffer->truncated(buffer, i) || *buffer->terminated(buffer, i)) * *buffer->advantage(buffer, i + 1);

        sum += *buffer->advantage(buffer, i);
    }

    for (int i = 0; i < limit; i++) {
        *buffer->adv_target(buffer, i) = v[i] + *buffer->advantage(buffer, i);
    }

    float debug_targets[limit];
    memcpy(debug_targets, buffer->adv_target_p, limit * sizeof(float));

    float debug_adv[limit];
    memcpy(debug_adv, buffer->advantage_p, limit * sizeof(float));

    float mean = sum / limit;
    
    float std = 0;
    for (int i = 0; i < limit; i++) {
        std += pow(*buffer->advantage(buffer, i) - mean, 2);
    }
    std = sqrt(std / limit);

    for (int i = 0; i < limit; i++) {
        *buffer->advantage(buffer, i) = (*buffer->advantage(buffer, i) - mean) / (std + 1e-8);
    }
}



void train_ppo_epoch(PPO* ppo, int steps_per_epoch, int batch_size, int n_epochs_policy, int n_epochs_value) {
    float states[batch_size * ppo->buffer->state_size];
    float actions[batch_size * ppo->buffer->action_size];
    float logprobs[batch_size];
    float logprobs_old[batch_size];
    float adv[batch_size];
    float adv_target[batch_size];

    float policy_loss_grad[batch_size];
    float mu_grad[batch_size * ppo->buffer->action_size];
    float v_loss_grad[batch_size];

    float entropy_grad;

    int num_batches_policy = ceilf(ppo->buffer->capacity / batch_size) * n_epochs_policy;
    int num_batches_value = ceilf(ppo->buffer->capacity / batch_size) * n_epochs_value;

    printf("ENtropy: %f\n", compute_entropy(ppo->policy));

    for (int i = 0; i < steps_per_epoch / ppo->buffer->capacity; i++) {
        collect_trajectories(ppo->buffer, ppo->env, ppo->policy, ppo->buffer->capacity);

        // Compute advantages
        compute_gae(ppo->V, ppo->buffer, ppo->gamma, ppo->lambda);

        float v_loss = mean_squared_error(ppo->V->output, ppo->buffer->adv_target_p, ppo->buffer->capacity, 1);

        // printf("V loss: %f\n", v_loss);

        shuffle_buffer(ppo->buffer);

        // Fit value function
        for (int j = 0; j < num_batches_value; j++) {
            // sample_batch(ppo->buffer, batch_size, states, actions, logprobs_old, adv, adv_target);
            get_batch(ppo->buffer, j, batch_size, states, actions, logprobs_old, adv, adv_target);

            forward_propagation(ppo->V, states, batch_size);

            float v_loss = mean_squared_error(ppo->V->output, adv_target, batch_size, 1);

            // printf("loss: %f\n", v_loss);

            mean_squared_error_derivative(v_loss_grad, ppo->V->output, adv_target, batch_size, 1);
            
            backward_propagation(ppo->V, v_loss_grad, batch_size);

            adam_update(ppo->adam_V, ppo->lr_V);
        }

        // adam_update(ppo->adam_V, ppo->lr_V);

        // printf("VLOSS: %f\n", v_loss);

        for (int j = 0; j < num_batches_policy; j++) {

            // sample_batch(ppo->buffer, batch_size, states, actions, logprobs_old, adv, adv_target);
            get_batch(ppo->buffer, j, batch_size, states, actions, logprobs_old, adv, adv_target);

            // Compute policy loss
            // SETS VALUES FOR GRAD COMPUTATION
            compute_log_prob(ppo->policy, logprobs, states, actions, batch_size);

            float entropy = compute_entropy(ppo->policy);

            float policy_loss = policy_loss_and_grad(policy_loss_grad, &entropy_grad, adv, logprobs, logprobs_old, entropy, ppo->ent_coeff, ppo->epsilon, batch_size);

            // printf("Policy loss: %f\n", policy_loss);

            // printf("Policy loss: %f\n", policy_loss);

            log_prob_backwards(ppo->policy, policy_loss_grad, mu_grad, ppo->policy->log_std_grad, batch_size);

            backward_propagation(ppo->policy->mu, mu_grad, batch_size);

            for (int i = 0; i < ppo->policy->action_size; i++) {
                ppo->policy->log_std_grad[i] += entropy_grad;
            }

            adam_update(ppo->adam_entropy, ppo->lr_policy);

            adam_update(ppo->adam_policy, ppo->lr_policy);
        }
    }
}

void eval_ppo(PPO* ppo, int steps){
    // TrajectoryBuffer* buffer = create_trajectory_buffer(steps, ppo->env->state_size, ppo->env->action_size);
    reset_buffer(ppo->buffer);

    collect_trajectories(ppo->buffer, ppo->env, ppo->policy, steps);

    float episode_rewards = 0;
    int n_episodes = 0;
    float sum_rewards = 0;
    for (int i = 0; i < steps; i++) {
        episode_rewards += *ppo->buffer->reward(ppo->buffer, i);
        if (*ppo->buffer->terminated(ppo->buffer, i) || *ppo->buffer->truncated(ppo->buffer, i)) {
            n_episodes++;
            sum_rewards += episode_rewards;
            episode_rewards = 0;
        }
    }
    printf("Mean reward: %f %d\n", sum_rewards / n_episodes, n_episodes);
    reset_buffer(ppo->buffer);
}
