#include "ppo.h"

#include "cuda_helper.h"
#include "welford_var.h"

PPO* create_ppo(char** activation_functions, int* layer_sizes, int num_layers, int buffer_size, float lr_policy, float lr_v, float lambda, float epsilon, float ent_coeff, float init_std, bool use_cuda) {

    PPO* ppo = (PPO*)malloc(sizeof(PPO));
    ppo->buffer = create_trajectory_buffer(buffer_size, layer_sizes[0], layer_sizes[num_layers - 1]);
    ppo->policy = create_gaussian_policy(layer_sizes, activation_functions, num_layers, init_std);

    int layer_sizes_v[num_layers];
    memcpy(layer_sizes_v, layer_sizes, (num_layers - 1) * sizeof(int));
    layer_sizes_v[num_layers - 1] = 1;
    
    ppo->V = create_neural_network(layer_sizes_v, activation_functions, num_layers);

    if (use_cuda){
        ppo->adam_policy = create_adam_from_nn_cuda(ppo->policy->mu, 0.9, 0.999);
        ppo->adam_V = create_adam_from_nn_cuda(ppo->V, 0.9, 0.999);
        ppo->adam_entropy = create_adam_cuda(&ppo->policy->d_log_std, &ppo->policy->d_log_std_grad, &ppo->policy->action_size, 1, ppo->policy->action_size,  0.9, 0.999);
    } else {
        ppo->adam_policy = create_adam_from_nn(ppo->policy->mu, 0.9, 0.999);
        ppo->adam_V = create_adam_from_nn(ppo->V, 0.9, 0.999);
        ppo->adam_entropy = create_adam(&ppo->policy->log_std, &ppo->policy->log_std_grad, &ppo->policy->action_size, 1, ppo->policy->action_size,  0.9, 0.999);
    }

    ppo->lambda = lambda;
    ppo->epsilon = epsilon;
    ppo->ent_coeff = ent_coeff;
    ppo->lr_policy = lr_policy;
    ppo->lr_V = lr_v;
    ppo->use_cuda = use_cuda;
    return ppo;
}

void free_ppo(PPO* ppo) {
    if (ppo->use_cuda) {
        free_adam_cuda(ppo->adam_policy);
        free_adam_cuda(ppo->adam_V);
        free_adam_cuda(ppo->adam_entropy);
    } else {
        free_adam(ppo->adam_policy);
        free_adam(ppo->adam_V);
        free_adam(ppo->adam_entropy);
    }
    free_trajectory_buffer(ppo->buffer, ppo->use_cuda);
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

__global__ void policy_loss_kernel(float* block_loss, float* grad_logprob, float* adv, float* logprobs, float* old_logprobs, float entropy, float ent_coeff, float epsilon, int m) {
    __shared__ float shared_sum[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < m){

        float ratio = exp(logprobs[idx] - old_logprobs[idx]);

        bool adv_pos = adv[idx] > 0;
        bool ratio_pos = ratio > 1 + epsilon;
        bool ratio_neg = ratio < 1 - epsilon;
        
        shared_sum[tid] = adv[idx] * (adv_pos * (ratio_pos * (1 + epsilon) + !ratio_pos * ratio) + !adv_pos * (ratio_neg * (1 - epsilon) + !ratio_neg * ratio));
    
        grad_logprob[idx] = -(adv_pos * !ratio_pos + !adv_pos * !ratio_neg) * adv[idx] * ratio / m; 
    }
    else{
        shared_sum[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_loss[blockIdx.x] = -shared_sum[0] / m - ent_coeff * entropy;
    }
}
    

float policy_loss_and_grad_cuda(float* grad_logprob, float* grad_entropy, float* adv, float* logprobs,  float* old_logprobs, float entropy, float ent_coeff, float epsilon, int m){
    int block_size = BLOCK_SIZE < m ? BLOCK_SIZE : m;
    int n_blocks = DIVUP(m, block_size);

    float* d_block_loss;
    cudaMalloc(&d_block_loss, sizeof(float) * n_blocks);

    policy_loss_kernel<<<n_blocks, block_size>>>(d_block_loss, grad_logprob, adv, logprobs, old_logprobs, entropy, ent_coeff, epsilon, m);

    float block_loss[BLOCK_SIZE];
    cudaMemcpy(block_loss, d_block_loss, sizeof(float) * n_blocks, cudaMemcpyDeviceToHost);
    
    
    cudaCheckErrors();

    float loss = 0.0;
    for (int i = 0; i < n_blocks; i++) {
        loss += block_loss[i];
    }

    *grad_entropy = -ent_coeff;

    return loss;
}

__global__ void gae_compute_block_advantage_kernel(float* advantage, float* reward, float* v, float* v_next, bool* terminated, bool* truncated, bool* terminated_out, float gamma, float alpha, int n) {
    volatile __shared__ float shared_sum[BLOCK_SIZE];
    volatile __shared__ bool shared_terminated[BLOCK_SIZE];

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

    float alpha_pow = alpha;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        bool temp_terminated = shared_terminated[tid];

        if (tid + stride < blockDim.x && !temp_terminated) {
            temp_terminated = shared_terminated[tid + stride];
            temp = alpha_pow * shared_sum[tid + stride];
        }
        __syncthreads();

        shared_sum[tid] += temp;
        shared_terminated[tid] = temp_terminated;

        __syncthreads();

        alpha_pow *= alpha_pow;
    }

    if (idx < n){
        advantage[idx] = shared_sum[tid];
        terminated_out[idx] = shared_terminated[tid];
    }
}

__global__ void gae_merge_kernel(float* advantage_out, float* advantage, bool* terminated, float* v, float* adv_target, float alpha, int num_blocks, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        advantage_out[idx] = advantage[idx];
    }

    int first_el_next_block = (blockIdx.x + 1) * blockDim.x;

    bool term = terminated[idx];

    if (first_el_next_block < n) {
        if (!term) {
            advantage_out[idx] += powf(alpha, blockDim.x - threadIdx.x) * advantage[first_el_next_block];
        }
    }

    for (int i = 1; i < num_blocks; i++){
        int first_el_ith_block = (blockIdx.x + i) * blockDim.x;

        int first_el_ith_next_block = (blockIdx.x + i + 1) * blockDim.x;




        if (first_el_ith_next_block < n) {
            term = term || terminated[first_el_ith_block]; 
            if (!term) {
                advantage_out[idx] += powf(alpha, (i + 1) * blockDim.x - threadIdx.x) * advantage[first_el_ith_next_block];
            }
        }
    }

    if (idx < n) {
        adv_target[idx] = v[idx] + advantage_out[idx];
    }
}

__global__ void normalize_advantage_kernel(float* advantage, float mean, float std, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        advantage[idx] = (advantage[idx] - mean) / (std + 1e-8);
    }
}

void compute_gae_cuda(NeuralNetwork* V, TrajectoryBuffer* buffer, float gamma, float lambda, int horizon){
    int limit = buffer->full ? buffer->capacity : buffer->idx;

    float* v_next;
    cudaMalloc(&v_next, limit * sizeof(float));

    forward_propagation_cuda(V, buffer->next_state_p, limit);
    cudaMemcpy(v_next, V->d_output, limit * sizeof(float), cudaMemcpyDeviceToDevice);
    
    float* v;
    cudaMalloc(&v, limit * sizeof(float));

    forward_propagation_cuda(V, buffer->state_p, limit);    
    cudaMemcpy(v, V->d_output, limit * sizeof(float), cudaMemcpyDeviceToDevice);

    bool* terminated_temp;
    cudaMalloc(&terminated_temp, limit * sizeof(bool));

    float* block_advantage;
    cudaMalloc(&block_advantage, limit * sizeof(float));

    cudaCheckErrors();

    const int n_blocks = DIVUP(limit, BLOCK_SIZE);

    gae_compute_block_advantage_kernel<<<n_blocks, BLOCK_SIZE>>>(block_advantage, buffer->d_reward_p, v, v_next, buffer->d_terminated_p, buffer->d_truncated_p, terminated_temp, gamma, gamma * lambda, limit);

    cudaCheckErrors();


    int num_blocks = horizon / BLOCK_SIZE;

    gae_merge_kernel<<<n_blocks, BLOCK_SIZE>>>(buffer->d_advantage_p, block_advantage, terminated_temp, v, buffer->d_adv_target_p, gamma * lambda, num_blocks, limit);



    cudaCheckErrors();

    WelfordState block_states[n_blocks];
    WelfordState* d_block_states;
    cudaMalloc(&d_block_states, n_blocks * sizeof(WelfordState));

    welford_var_kernel<<<n_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(WelfordState)>>>(buffer->d_advantage_p, limit, d_block_states);

    cudaCheckErrors();

    cudaMemcpy(block_states, d_block_states, n_blocks * sizeof(WelfordState), cudaMemcpyDeviceToHost);

    WelfordState state;

    welford_combine_blocks(block_states, n_blocks, &state);

    float mean2 = state.mean;
    float std2 = sqrt(state.m2 / state.n);

    normalize_advantage_kernel<<<n_blocks, BLOCK_SIZE>>>(buffer->d_advantage_p, mean2, std2, limit);

    cudaCheckErrors();

    cudaFree(v);
    cudaFree(v_next);
    cudaFree(terminated_temp);
}


void compute_gae(NeuralNetwork* V, TrajectoryBuffer* buffer, float gamma, float lambda) {
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

    float sum = 0;
    for (int i = limit - 1; i >= 0; i--) {
        *buffer->advantage(buffer, i) = delta[i] + gamma * lambda * !(*buffer->truncated(buffer, i) || *buffer->terminated(buffer, i)) * *buffer->advantage(buffer, i + 1);

        sum += *buffer->advantage(buffer, i);
    }

    for (int i = 0; i < limit; i++) {
        *buffer->adv_target(buffer, i) = v[i] + *buffer->advantage(buffer, i);
    }

    float mean = sum / limit;
    
    float std = 0;
    for (int i = 0; i < limit; i++) {
        std += pow(*buffer->advantage(buffer, i) - mean, 2);
        // printf("Advantage at index %d: %f\n", i, *buffer->advantage(buffer, i));
    }
    std = sqrt(std / limit);

    // printf("Mean: %f Std: %f\n", mean, std);

    for (int i = 0; i < limit; i++) {
        *buffer->advantage(buffer, i) = (*buffer->advantage(buffer, i) - mean) / (std + 1e-8);
    }
}



void _train_ppo_epoch(PPO* ppo, Env* env, int steps_per_epoch, int batch_size, int n_epochs_policy, int n_epochs_value) {
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


void _train_ppo_epoch_cuda(PPO* ppo, Env* env, int steps_per_epoch, int batch_size, int n_epochs_policy, int n_epochs_value) {
    float* states;
    float* actions;
    float* logprobs;
    float* logprobs_old;
    float* adv;
    float* adv_target;

    float* policy_loss_grad;
    float* mu_grad;
    float* v_loss_grad;

    cudaMalloc(&states, batch_size * ppo->buffer->state_size * sizeof(float));
    cudaMalloc(&actions, batch_size * ppo->buffer->action_size * sizeof(float));
    cudaMalloc(&logprobs, batch_size * sizeof(float));
    cudaMalloc(&logprobs_old, batch_size * sizeof(float));
    cudaMalloc(&adv, batch_size * sizeof(float));
    cudaMalloc(&adv_target, batch_size * sizeof(float));
    cudaMalloc(&policy_loss_grad, batch_size * sizeof(float));
    cudaMalloc(&mu_grad, batch_size * ppo->buffer->action_size * sizeof(float));
    cudaMalloc(&v_loss_grad, batch_size * sizeof(float));

    float entropy_grad;

    int num_batches_policy = ceilf(ppo->buffer->capacity / batch_size);
    int num_batches_value = ceilf(ppo->buffer->capacity / batch_size);


    for (int i = 0; i < steps_per_epoch / ppo->buffer->capacity; i++) {
        // buffer_to_host(ppo->buffer);

        collect_trajectories(ppo->buffer, env, ppo->policy, ppo->buffer->capacity);


        buffer_to_device(ppo->buffer);
        // Compute advantages
        compute_gae_cuda(ppo->V, ppo->buffer, env->gamma, ppo->lambda, env->horizon);


        float sum_v_loss = 0;
        for (int j = 0; j < n_epochs_value; j++) {
            shuffle_buffer_cuda(ppo->buffer);

            // Fit value function
            for (int k = 0; k < num_batches_value; k++) {
                get_batch_cuda(ppo->buffer, k, batch_size, states, actions, logprobs_old, adv, adv_target);

                forward_propagation_cuda(ppo->V, states, batch_size);

                float v_loss = mean_squared_error_cuda(ppo->V->d_output, adv_target, batch_size, 1);

                sum_v_loss += v_loss;

                mean_squared_error_derivative_cuda(v_loss_grad, ppo->V->d_output, adv_target, batch_size, 1);
                
                backward_propagation_cuda(ppo->V, v_loss_grad, batch_size);

                adam_update_cuda(ppo->adam_V, ppo->lr_V);
            }
        }

        for (int j = 0; j < n_epochs_policy; j++) {
            shuffle_buffer_cuda(ppo->buffer);

            for (int k = 0; k < num_batches_policy; k++) {

                get_batch_cuda(ppo->buffer, k, batch_size, states, actions, logprobs_old, adv, adv_target);

                compute_log_prob_cuda(ppo->policy, logprobs, states, actions, batch_size);

                float entropy = compute_entropy_cuda(ppo->policy);

                float policy_loss = policy_loss_and_grad_cuda(policy_loss_grad, &entropy_grad, adv, logprobs, logprobs_old, entropy, ppo->ent_coeff, ppo->epsilon, batch_size);

                log_prob_backwards_cuda(ppo->policy, policy_loss_grad, mu_grad, ppo->policy->d_log_std_grad, batch_size);

                backward_propagation_cuda(ppo->policy->mu, mu_grad, batch_size);

                adam_update_cuda(ppo->adam_entropy, ppo->lr_policy);

                adam_update_cuda(ppo->adam_policy, ppo->lr_policy);
            }
        }

        // printf("Iteration %d V loss: %f Entropy: %f\n", i, sum_v_loss / (n_epochs_value * num_batches_value),  compute_entropy(ppo->policy));
        buffer_to_host(ppo->buffer);
        policy_to_host(ppo->policy);
        nn_write_weights_to_host(ppo->V);
    }

    cudaFree(states);
    cudaFree(actions);
    cudaFree(logprobs);
    cudaFree(logprobs_old);
    cudaFree(adv);
    cudaFree(adv_target);
    cudaFree(policy_loss_grad);
    cudaFree(mu_grad);
    cudaFree(v_loss_grad);
}

void train_ppo_epoch(PPO* ppo, Env* env, int steps_per_epoch, int batch_size, int n_epochs_policy, int n_epochs_value) {
    if (ppo->use_cuda) {
        _train_ppo_epoch_cuda(ppo, env, steps_per_epoch, batch_size, n_epochs_policy, n_epochs_value);
    } else {
        _train_ppo_epoch(ppo, env, steps_per_epoch, batch_size, n_epochs_policy, n_epochs_value);
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

    save_policy(ppo->policy, file);

    save_neural_network(ppo->V, file);

    save_adam(ppo->adam_policy, file, ppo->use_cuda);
    save_adam(ppo->adam_V, file, ppo->use_cuda);
    save_adam(ppo->adam_entropy, file, ppo->use_cuda);

    fclose(file);
}


PPO* load_ppo(const char* filename, bool use_cuda) {
    FILE* file = fopen(filename, "rb");

    PPO* ppo = (PPO*)malloc(sizeof(PPO));

    ppo->use_cuda = use_cuda;

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

    ppo->policy = load_policy(file, state_size, action_size);

    ppo->V = load_neural_network(file);

    ppo->adam_policy = load_adam_from_nn(file, ppo->policy->mu, use_cuda);
    ppo->adam_V = load_adam_from_nn(file, ppo->V, use_cuda);
    if (use_cuda){
        ppo->adam_entropy = load_adam(file, &ppo->policy->d_log_std, &ppo->policy->d_log_std_grad, &action_size, use_cuda);
    } else {
        ppo->adam_entropy = load_adam(file, &ppo->policy->log_std, &ppo->policy->log_std_grad, &action_size, use_cuda);
    }
    
    fclose(file);

    return ppo;
}

