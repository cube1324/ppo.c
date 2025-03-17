
#include "policy.h"

#include "cuda_helper.h"

GaussianPolicy* create_gaussian_policy(int* layer_sizes, char** activation_functions, int num_layers, float init_std) {
    GaussianPolicy* policy = (GaussianPolicy*)malloc(sizeof(GaussianPolicy));
    policy->state_size = layer_sizes[0];
    policy->action_size = layer_sizes[num_layers - 1];
    
    policy->mu = create_neural_network(layer_sizes, activation_functions, num_layers);
    
    policy->log_std = (float*)malloc(policy->action_size * sizeof(float));
    policy->log_std_grad = (float*)malloc(policy->action_size * sizeof(float));

    cudaMalloc(&policy->d_log_std, sizeof(float) * policy->action_size);
    cudaMalloc(&policy->d_log_std_grad, sizeof(float) * policy->action_size);

    policy->input_action = NULL;
    policy->d_input_action = NULL;

    for (int i = 0; i < policy->action_size; i++) {
        policy->log_std[i] = logf(init_std);
    }

    cudaMemcpy(policy->d_log_std, policy->log_std, sizeof(float) * policy->action_size, cudaMemcpyHostToDevice);

    //cudaCheckErrors()

    return policy;
}

void free_gaussian_policy(GaussianPolicy* policy) {
    free_neural_network(policy->mu);
    free(policy->log_std);
    free(policy->log_std_grad);
    free(policy->input_action);

    cudaFree(policy->d_log_std);
    cudaFree(policy->d_log_std_grad);
    cudaFree(policy->d_input_action);
    free(policy);
}


void generate_gaussian_noise(float* out, int n) {
    // Box Muller Transform 
    if (n == 1) {
        out[0] = sqrtf(-2 * logf((float)rand() / RAND_MAX)) * cosf(2 * M_PI * (float)rand() / RAND_MAX);
        return;
    }

    for (int i = 0; i <= n / 2; i+= 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float r = sqrtf(-2 * logf(u1));
        float theta = 2 * M_PI * u2;
        out[i] = r * cosf(theta);
        out[i + 1] = r * sinf(theta);
    }

    if (n % 2 == 1) {
        out[n - 1] = sqrtf(-2 * logf((float)rand() / RAND_MAX)) * cosf(2 * M_PI * (float)rand() / RAND_MAX);
    }
}

float _compute_log_prob(float* mu, float* log_std, float* action, int action_size) {
    float logprob = -0.5 * action_size *  logf(2 * M_PI);

    for (int i = 0; i < action_size; i++){
        logprob -= log_std[i] + 0.5 * powf((action[i] - mu[i]) / expf(log_std[i]), 2);
    }
    return logprob;
}

void sample_action(GaussianPolicy* policy, float* state, float* action, float* log_prob, int m) {
    forward_propagation(policy->mu, state, m);

    float noise[m * policy->action_size];
    generate_gaussian_noise(noise, m * policy->action_size);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < policy->action_size; j++) {
            // DO i need to transpose output?
            action[i * policy->action_size + j] = policy->mu->output[i * policy->action_size + j] + noise[i * policy->action_size + j] * expf(policy->log_std[j]);
        }
        log_prob[i] = _compute_log_prob(policy->mu->output + i * policy->action_size, policy->log_std, action + i * policy->action_size, policy->action_size);
    }
}

void compute_log_prob(GaussianPolicy* policy, float* out, float* state, float* action, int m) {
    policy->input_action = action;

    forward_propagation(policy->mu, state, m);

    for (int i = 0; i < m; i++) {
        out[i] = _compute_log_prob(policy->mu->output + i * policy->action_size, policy->log_std, action + i * policy->action_size, policy->action_size);
    }
}

void log_prob_backwards(GaussianPolicy* policy, float* grad_in, float* grad_mu, float* grad_log_std, int m) {
    memset(grad_log_std, 0, policy->action_size * sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < policy->action_size; j++) {
            grad_mu[i * policy->action_size + j] = (policy->input_action[i * policy->action_size + j] - policy->mu->output[i * policy->action_size + j]) * expf(-2 * policy->log_std[j]) * grad_in[i * policy->action_size + j];

            grad_log_std[j] += (-1 + powf(policy->input_action[i * policy->action_size + j] - policy->mu->output[i * policy->action_size + j], 2) * expf(-2 * policy->log_std[j])) * grad_in[i * policy->action_size + j];
        }
    }
}

__global__ void compute_log_prob_kernel(float* mu, float* log_std, float* action, float* out, int action_size, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int transposed_idx = (idx % action_size) * m + idx / action_size;

    float temp_out = -0.5 * action_size *  logf(2 * M_PI);

    for (int i = 0; i < action_size; i++){
        temp_out -= log_std[idx % action_size] + 0.5 * powf((action[idx] - mu
        [transposed_idx]) / expf(log_std[idx % action_size]), 2);
    }
    out[idx] = temp_out;
}

void compute_log_prob_cuda(GaussianPolicy* policy, float* out, float* state, float* action, int m){
    policy->d_input_action = action;

    int block_size = BLOCK_SIZE < m ? BLOCK_SIZE : m;
    int n_blocks = DIVUP(m, block_size);

    forward_propagation_cuda(policy->mu, state, m);

    compute_log_prob_kernel<<<n_blocks, block_size>>>(policy->mu->d_output, policy->d_log_std, action, out, policy->action_size, m);

    //cudaDeviceSynchronize();
    //cudaCheckErrors()
}

__global__ void log_prob_backwards_kernel(
    float* grad_in, float* grad_mu, float* grad_log_std, float* input_action, float* mu, float* log_std, int action_size, int m) {
    // 2D grid layout for coalesced memory access
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= m || j >= action_size) return;

    const int index = i * action_size + j;
    const float diff = input_action[index] - mu[index];
    const float exp_neg2_log_std = expf(-2.0f * log_std[j]);

    // Direct computation for grad_mu (coalesced write)
    grad_mu[index] = diff * exp_neg2_log_std * grad_in[index];

    // Thread-safe accumulation for grad_log_std
    const float grad_term = (-1.0f + (diff * diff) * exp_neg2_log_std) * grad_in[index];
    atomicAdd(&grad_log_std[j], grad_term);
}

void log_prob_backwards_cuda(GaussianPolicy* policy, float* grad_in, float* grad_mu, float* grad_log_std, int m){
    cudaMemset(grad_log_std, 0, policy->action_size * sizeof(float));

    dim3 block_size(policy->action_size, BLOCK_SIZE / policy->action_size < m ? BLOCK_SIZE / policy->action_size : m);
    dim3 grid_size(1, DIVUP(m, block_size.y));

    log_prob_backwards_kernel<<<grid_size, block_size>>>(grad_in, grad_mu, grad_log_std, policy->d_input_action, policy->mu->d_output, policy->d_log_std, policy->action_size, m);

    //cudaDeviceSynchronize();
    //cudaCheckErrors()
}

float compute_entropy(GaussianPolicy* policy) {
    float entropy = policy->action_size * 0.5 * (1 + log(2 * M_PI));
    for (int j = 0; j < policy->action_size; j++) {
        entropy += policy->log_std[j];
    }

    return entropy;
}

float compute_entropy_cuda(GaussianPolicy* policy){
    float h_log_std[policy->action_size];

    cudaMemcpy(h_log_std, policy->d_log_std, sizeof(float) * policy->action_size, cudaMemcpyDeviceToHost);

    //cudaCheckErrors()

    float entropy = policy->action_size * 0.5 * (1 + log(2 * M_PI));
    for (int j = 0; j < policy->action_size; j++) {
        entropy += h_log_std[j];
    }

    return entropy;
}

void policy_to_host(GaussianPolicy* policy) {
    nn_write_weights_to_host(policy->mu);
    cudaMemcpy(policy->log_std, policy->d_log_std, sizeof(float) * policy->action_size, cudaMemcpyDeviceToHost);
}


void save_policy(GaussianPolicy* policy, FILE* file){
    fwrite(policy->log_std, sizeof(float), policy->action_size, file);

    save_neural_network(policy->mu, file);
}

GaussianPolicy* load_policy(FILE* file, int state_size, int action_size) {
    GaussianPolicy* policy = (GaussianPolicy*)malloc(sizeof(GaussianPolicy));
    policy->state_size = state_size;
    policy->action_size = action_size;
    policy->log_std = (float*)malloc(action_size * sizeof(float));
    policy->log_std_grad = (float*)malloc(action_size * sizeof(float));

    cudaMalloc(&policy->d_log_std, sizeof(float) * action_size);
    cudaMalloc(&policy->d_log_std_grad, sizeof(float) * action_size);

    fread(policy->log_std, sizeof(float), action_size, file);
    
    cudaMemcpy(policy->d_log_std, policy->log_std, sizeof(float) * action_size, cudaMemcpyHostToDevice);
    
    policy->mu = load_neural_network(file);
    
    policy->input_action = NULL;
    policy->d_input_action = NULL;

    return policy;
}