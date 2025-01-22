
#include "policy.h"


GaussianPolicy* create_gaussian_policy(int* layer_sizes, ActivationFunction* activation_functions, int num_layers, int init_std) {
    GaussianPolicy* policy = (GaussianPolicy*)malloc(sizeof(GaussianPolicy));
    policy->state_size = layer_sizes[0];
    policy->action_size = layer_sizes[num_layers - 1];
    policy->mu = create_neural_network(layer_sizes, activation_functions, num_layers);
    policy->log_std = (float*)malloc(policy->action_size * sizeof(float));

    for (int i = 0; i < policy->action_size; i++) {
        policy->log_std[i] = log(init_std);
    }

    return policy;
}

void free_gaussian_policy(GaussianPolicy* policy) {
    free_neural_network(policy->mu);
    free(policy->log_std);
    free(policy->input_action);
    free(policy);
}


void generate_gaussian_noise(float* out, int n) {
    // Box Muller Transform 
    for (int i = 0; i <= n / 2; i+= 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float r = sqrt(-2 * log(u1));
        float theta = 2 * M_PI * u2;
        out[i] = r * cos(theta);
        out[i + 1] = r * sin(theta);
    }

    if (n % 2 == 1) {
        out[n - 1] = sqrt(-2 * log((float)rand() / RAND_MAX)) * cos(2 * M_PI * (float)rand() / RAND_MAX);
    }
}

float compute_log_prob(float mu, float log_std, float action) {
    return -0.5 * log(2 * M_PI) - log_std - 0.5 * pow((action - mu) / exp(log_std), 2);
}

void sample_action(GaussianPolicy* policy, float* state, float* action, float* log_prob, int m) {
    forward_propagation(policy->mu, state, m);

    float* noise = (float*)malloc(m * policy->action_size * sizeof(float));
    generate_gaussian_noise(noise, m * policy->action_size);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < policy->action_size; j++) {
            action[i * policy->action_size + j] = policy->mu->output[i * policy->action_size + j] + noise[i * policy->action_size + j] * exp(policy->log_std[j]);
            log_prob[i * policy->action_size + j] = compute_log_prob(policy->mu->output[i * policy->action_size + j], policy->log_std[j], action[i * policy->action_size + j]);
        }
    }
    free(noise);
}

void log_prob(GaussianPolicy* policy, float* out, float* state, float* action, int m) {
    free(policy->input_action);
    policy->input_action = action;

    forward_propagation(policy->mu, state, m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < policy->action_size; j++) {
            out[i * policy->action_size + j] = compute_log_prob(policy->mu->output[i * policy->action_size + j], policy->log_std[j], action[i * policy->action_size + j]);
        }
    }
}

void log_prob_backwards(GaussianPolicy* policy, float* grad_in, float* state, float* action, int m) {
    float* grad_out = (float*)malloc(m * policy->action_size * sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < policy->action_size; j++) {
            grad_out[i * policy->action_size + j] = (policy->input_action[i * policy->action_size + j] - policy->mu->output[i * policy->action_size + j]) / pow(exp(policy->log_std[j]), 2) * grad_in[i * policy->action_size + j];
        }
    }

    backward_propagation(policy->mu, grad_out, m);
}



