
#include "policy.h"


GaussianPolicy* create_gaussian_policy(int* layer_sizes, ActivationFunction* activation_functions, int num_layers, float init_std) {
    GaussianPolicy* policy = (GaussianPolicy*)malloc(sizeof(GaussianPolicy));
    policy->state_size = layer_sizes[0];
    policy->action_size = layer_sizes[num_layers - 1];
    policy->mu = create_neural_network(layer_sizes, activation_functions, num_layers);
    policy->log_std = (float*)malloc(policy->action_size * sizeof(float));
    policy->log_std_grad = (float*)malloc(policy->action_size * sizeof(float));
    policy->input_action = NULL;

    for (int i = 0; i < policy->action_size; i++) {
        policy->log_std[i] = logf(init_std);
    }

    return policy;
}

void free_gaussian_policy(GaussianPolicy* policy) {
    free_neural_network(policy->mu);
    free(policy->log_std);
    free(policy->log_std_grad);
    free(policy);
}


void generate_gaussian_noise(float* out, int n) {
    // Box Muller Transform 
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
    float logprob = -0.5 * action_size *  logf(2 * M_PI) ;

    for (int i = 0; i < action_size; i++){
        logprob -= log_std[i] + 0.5 * powf((action[i] - mu[i]) / expf(log_std[i]), 2);
    }
    return logprob;
}

void sample_action(GaussianPolicy* policy, float* state, float* action, float* log_prob, int m) {
    forward_propagation(policy->mu, state, m);

    float* noise = (float*)malloc(m * policy->action_size * sizeof(float));
    generate_gaussian_noise(noise, m * policy->action_size);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < policy->action_size; j++) {
            action[i * policy->action_size + j] = policy->mu->output[i * policy->action_size + j] + noise[i * policy->action_size + j] * expf(policy->log_std[j]);
        }
        log_prob[i] = _compute_log_prob(policy->mu->output + i * policy->action_size, policy->log_std, action + i * policy->action_size, policy->action_size);
    }
    free(noise);
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
            grad_mu[i * policy->action_size + j] = (policy->input_action[i * policy->action_size + j] - policy->mu->output[i * policy->action_size + j]) / expf(2 * policy->log_std[j]) * grad_in[i * policy->action_size + j];

            grad_log_std[j] += (-1 + powf(policy->input_action[i * policy->action_size + j] - policy->mu->output[i * policy->action_size + j], 2) / expf(2 * policy->log_std[j])) * grad_in[i * policy->action_size + j];
        }
    }
}

float compute_entropy(GaussianPolicy* policy) {
    float entropy = policy->action_size * 0.5 * (1 + log(2 * M_PI));
    for (int j = 0; j < policy->action_size; j++) {
        entropy += policy->log_std[j];
    }

    return entropy;
}

