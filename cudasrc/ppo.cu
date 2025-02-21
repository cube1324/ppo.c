#include "ppo.h"

#include "cuda_helper.cuh"
#include "welford_var.cuh"
#include "constants.cuh"

PPO* create_ppo(char** activation_functions, int* layer_sizes, int num_layers, int buffer_size, float lr_policy, float lr_v, float lambda, float epsilon, float ent_coeff, float init_std) {

    PPO* ppo = (PPO*)malloc(sizeof(PPO));
    ppo->buffer = create_trajectory_buffer(buffer_size, layer_sizes[0], layer_sizes[num_layers - 1]);
    ppo->policy = create_gaussian_policy(layer_sizes, activation_functions, num_layers, init_std);

    int layer_sizes_v[num_layers];
    memcpy(layer_sizes_v, layer_sizes, (num_layers - 1) * sizeof(int));
    layer_sizes_v[num_layers - 1] = 1;
    
    ppo->V = create_neural_network(layer_sizes, activation_functions, num_layers);

    ppo->adam_policy = create_adam_from_nn(ppo->policy->mu, 0.9, 0.999);
    ppo->adam_V = create_adam_from_nn(ppo->V, 0.9, 0.999);

    ppo->adam_entropy = create_adam(&ppo->policy->log_std, &ppo->policy->log_std_grad, &ppo->policy->action_size, 1, ppo->policy->action_size,  0.9, 0.999);
    
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

    float ratio;
    float clipped_ratio;

    for (int i = 0; i < m; i++) {
        ratio = exp(logprobs[i] - old_logprobs[i]);

        bool adv_pos = adv[i] > 0;
        bool ratio_pos = ratio > 1 + epsilon;
        bool ratio_neg = ratio < 1 - epsilon;

        loss -= adv[i] * (adv_pos * (ratio_pos * (1 + epsilon) + !ratio_pos * ratio) + !adv_pos * (ratio_neg * (1 - epsilon) + !ratio_neg * ratio)) ;

        grad_logprob[i] = -(adv_pos * !ratio_pos + !adv_pos * !ratio_neg) * adv[i] * ratio / m;

    }
    loss /= m;

    loss -= ent_coeff * entropy;
    *grad_entropy = -ent_coeff;

    return loss;
}

__global__ void gae_compute_block_advantage_kernel(float* advantage, float* reward, float* v, float* v_next, bool* terminated, bool* truncated, bool* terminated_out, float gamma, float alpha, int n) {
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ bool shared_terminated[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        shared_sum[tid] = reward[idx] + gamma * v_next[idx] * !terminated[idx] - v[idx];
        shared_terminated[tid] = terminated[idx] || truncated[idx];
    } else {
        shared_sum[tid] = 0.0f;
        shared_terminated[tid] = 0;
    }

    __syncthreads();

    // TODO omptimize presum https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;

        if (tid + stride < blockDim.x && !shared_terminated[tid]) {
            shared_terminated[tid] = shared_terminated[tid + stride];
            temp = powf(alpha, stride) * shared_sum[tid + stride];
        }
        __syncthreads();
        
        shared_sum[tid] += temp;
        
        __syncthreads();
    }

    advantage[idx] = shared_sum[tid];
    terminated_out[idx] = shared_terminated[tid];
}

__global__ void gae_merge_kernel(float* advantage, bool* terminated, float* v, float* adv_target, float alpha, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int first_el_next_block = (blockIdx.x + 1) * blockDim.x;

    if (first_el_next_block < n) {
        if (!terminated[idx]) {
            // TODO This breaks if episode is longer than block size, for now assume this doesent happen
            advantage[idx] += powf(alpha, blockDim.x - threadIdx.x) * advantage[first_el_next_block];
        }
    }

    if (idx < n) {
        adv_target[idx] = v[idx] + advantage[idx];
    }
}

__global__ void normalize_advantage_kernel(float* advantage, float mean, float std, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        advantage[idx] = (advantage[idx] - mean) / (std + 1e-8);
    }
}


void compute_gae(NeuralNetwork* V, TrajectoryBuffer* buffer, float gamma, float lambda) {
    int limit = buffer->full ? buffer->capacity : buffer->idx;

    float v_next[limit];
    forward_propagation(V, buffer->next_state_p, limit);
    memcpy(v_next, V->output, limit * sizeof(float));

    float v[limit];
    forward_propagation(V, buffer->state_p, limit);
    memcpy(v, V->output, limit * sizeof(float));

    // float* v_next;
    // cudaErrorCheck(cudaMalloc(&v_next, limit * sizeof(float)));

    // forward_propagation(V, buffer->next_state_p, limit);
    // cudaErrorCheck(cudaMemcpy(v_next, V->output, limit * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // float* v;
    // cudaErrorCheck(cudaMalloc(&v, limit * sizeof(float)));

    // forward_propagation(V, buffer->state_p, limit);    
    // cudaErrorCheck(cudaMemcpy(v, V->output, limit * sizeof(float), cudaMemcpyDeviceToDevice));

    buffer_to_device(buffer);

    float* d_v;
    cudaErrorCheck(cudaMalloc(&d_v, limit * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(d_v, v, limit * sizeof(float), cudaMemcpyHostToDevice));

    float* d_v_next;
    cudaErrorCheck(cudaMalloc(&d_v_next, limit * sizeof(float)));
    cudaErrorCheck(cudaMemcpy(d_v_next, v_next, limit * sizeof(float), cudaMemcpyHostToDevice));

    bool* terminated_temp;
    cudaErrorCheck(cudaMalloc(&terminated_temp, limit * sizeof(bool)));

    const int n_blocks = (limit + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gae_compute_block_advantage_kernel<<<n_blocks, BLOCK_SIZE>>>(buffer->d_advantage_p, buffer->d_reward_p, d_v, d_v_next, buffer->d_terminated_p, buffer->d_truncated_p, terminated_temp, gamma, gamma * lambda, limit);

    gae_merge_kernel<<<n_blocks, BLOCK_SIZE>>>(buffer->d_advantage_p, terminated_temp, d_v, buffer->d_adv_target_p, gamma * lambda, limit);

    WelfordState block_states[n_blocks];
    WelfordState* d_block_states;
    cudaErrorCheck(cudaMalloc(&d_block_states, n_blocks * sizeof(WelfordState)));

    welford_var_kernel<<<n_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(WelfordState)>>>(buffer->d_advantage_p, limit, d_block_states);

    cudaErrorCheck(cudaMemcpy(block_states, d_block_states, n_blocks * sizeof(WelfordState), cudaMemcpyDeviceToHost));

    WelfordState state;

    welford_combine_blocks(block_states, n_blocks, &state);

    float mean2 = state.mean;
    float std2 = sqrt(state.m2 / state.n);

    normalize_advantage_kernel<<<n_blocks, BLOCK_SIZE>>>(buffer->d_advantage_p, mean2, std2, limit);

    buffer_to_host(buffer);

    cudaErrorCheck(cudaFree(d_v));
    cudaErrorCheck(cudaFree(d_v_next));
    cudaErrorCheck(cudaFree(terminated_temp));
}



void train_ppo_epoch(PPO* ppo, Env* env, int steps_per_epoch, int batch_size, int n_epochs_policy, int n_epochs_value) {
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

    int num_batches_policy = ceilf(ppo->buffer->capacity / batch_size);
    int num_batches_value = ceilf(ppo->buffer->capacity / batch_size);


    for (int i = 0; i < steps_per_epoch / ppo->buffer->capacity; i++) {
        collect_trajectories(ppo->buffer, env, ppo->policy, ppo->buffer->capacity);

        // Compute advantages
        compute_gae(ppo->V, ppo->buffer, env->gamma, ppo->lambda);

        float sum_v_loss = 0;
        for (int j = 0; j < n_epochs_value; j++) {
            shuffle_buffer(ppo->buffer);

            // Fit value function
            for (int k = 0; k < num_batches_value; k++) {
                get_batch(ppo->buffer, k, batch_size, states, actions, logprobs_old, adv, adv_target);

                forward_propagation(ppo->V, states, batch_size);

                float v_loss = mean_squared_error(ppo->V->output, adv_target, batch_size, 1);

                sum_v_loss += v_loss;

                mean_squared_error_derivative(v_loss_grad, ppo->V->output, adv_target, batch_size, 1);
                
                backward_propagation(ppo->V, v_loss_grad, batch_size);

                adam_update(ppo->adam_V, ppo->lr_V);
            }
        }

        for (int j = 0; j < n_epochs_policy; j++) {
            shuffle_buffer(ppo->buffer);

            for (int k = 0; k < num_batches_policy; k++) {

                get_batch(ppo->buffer, k, batch_size, states, actions, logprobs_old, adv, adv_target);

                // Compute policy loss
                // SETS VALUES FOR GRAD COMPUTATION
                compute_log_prob(ppo->policy, logprobs, states, actions, batch_size);

                float entropy = compute_entropy(ppo->policy);

                float policy_loss = policy_loss_and_grad(policy_loss_grad, &entropy_grad, adv, logprobs, logprobs_old, entropy, ppo->ent_coeff, ppo->epsilon, batch_size);

                log_prob_backwards(ppo->policy, policy_loss_grad, mu_grad, ppo->policy->log_std_grad, batch_size);

                backward_propagation(ppo->policy->mu, mu_grad, batch_size);

                for (int i = 0; i < ppo->policy->action_size; i++) {
                    ppo->policy->log_std_grad[i] += entropy_grad;
                }

                adam_update(ppo->adam_entropy, ppo->lr_policy);

                adam_update(ppo->adam_policy, ppo->lr_policy);
            }
        }

        // printf("Iteration %d V loss: %f Entropy: %f\n", i, sum_v_loss / (n_epochs_value * num_batches_value),  compute_entropy(ppo->policy));
    }
}

void eval_ppo(PPO* ppo, Env* env, int steps){
    // Uses same Buffer, so cannot use more eval steps than step_size
    reset_buffer(ppo->buffer);

    collect_trajectories(ppo->buffer, env, ppo->policy, steps);

    float rewards = *ppo->buffer->reward(ppo->buffer, steps - 1);
    float episode_J = *ppo->buffer->reward(ppo->buffer, steps - 1);
    int n_episodes = 1;
    float sum_J = 0;

    for (int i = steps - 2; i >= 0; i--) {
        rewards += *ppo->buffer->reward(ppo->buffer, i);
        episode_J = *ppo->buffer->reward(ppo->buffer, i) + env->gamma * episode_J;
        if (*ppo->buffer->terminated(ppo->buffer, i) || *ppo->buffer->truncated(ppo->buffer, i)) {
            n_episodes++;
            sum_J += episode_J;
            episode_J = 0;
        }
    }

    printf("J: %f R: %f Episodes: %d\n", sum_J / n_episodes, rewards / n_episodes, n_episodes);
    reset_buffer(ppo->buffer);
}

void save_ppo(PPO* ppo, const char* filename) {
    FILE* file = fopen(filename, "wb");

    fwrite(&ppo->lambda, sizeof(float), 1, file);
    fwrite(&ppo->epsilon, sizeof(float), 1, file);
    fwrite(&ppo->ent_coeff, sizeof(float), 1, file);
    fwrite(&ppo->lr_policy, sizeof(float), 1, file);
    fwrite(&ppo->lr_V, sizeof(float), 1, file);

    fwrite(&ppo->buffer->state_size, sizeof(int), 1, file);
    fwrite(&ppo->buffer->action_size, sizeof(int), 1, file);
    fwrite(&ppo->buffer->capacity, sizeof(int), 1, file);

    fwrite(ppo->policy->log_std, sizeof(float), ppo->policy->action_size, file);

    save_neural_network(ppo->policy->mu, file);
    save_neural_network(ppo->V, file);

    save_adam(ppo->adam_policy, file);
    save_adam(ppo->adam_V, file);
    save_adam(ppo->adam_entropy, file);

    fclose(file);
}


PPO* load_ppo(const char* filename) {
    FILE* file = fopen(filename, "rb");

    PPO* ppo = (PPO*)malloc(sizeof(PPO));

    fread(&ppo->lambda, sizeof(float), 1, file);
    fread(&ppo->epsilon, sizeof(float), 1, file);
    fread(&ppo->ent_coeff, sizeof(float), 1, file);
    fread(&ppo->lr_policy, sizeof(float), 1, file);
    fread(&ppo->lr_V, sizeof(float), 1, file);

    int state_size;
    int action_size;
    int capacity;

    fread(&state_size, sizeof(int), 1, file);
    fread(&action_size, sizeof(int), 1, file);
    fread(&capacity, sizeof(int), 1, file);

    ppo->buffer = create_trajectory_buffer(capacity, state_size, action_size);

    ppo->policy = (GaussianPolicy*)malloc(sizeof(GaussianPolicy));
    ppo->policy->state_size = state_size;
    ppo->policy->action_size = action_size;
    ppo->policy->log_std = (float*)malloc(action_size * sizeof(float));
    ppo->policy->log_std_grad = (float*)malloc(action_size * sizeof(float));

    fread(ppo->policy->log_std, sizeof(float), action_size, file);
    ppo->policy->mu = load_neural_network(file);
    
    ppo->policy->input_action = NULL;

    ppo->V = load_neural_network(file);

    ppo->adam_policy = load_adam_from_nn(file, ppo->policy->mu);
    ppo->adam_V = load_adam_from_nn(file, ppo->V);
    ppo->adam_entropy = load_adam(file, &ppo->policy->log_std, &ppo->policy->log_std_grad, &action_size);

    fclose(file);

    return ppo;
}

