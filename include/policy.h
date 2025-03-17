#ifndef POLICY_H
#define POLICY_H

#include "neural_network.h"
#include <math.h>

#define M_PI 3.14159265358979323846

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    NeuralNetwork* mu;
    float* log_std;
    float* log_std_grad;
    float* d_log_std;
    float* d_log_std_grad;
    int state_size;
    int action_size;

    float* input_action;
    float* d_input_action;
} GaussianPolicy;

GaussianPolicy* create_gaussian_policy(int* layer_sizes, char** activation_functions, int num_layers, float init_std);
void free_gaussian_policy(GaussianPolicy* policy);

void sample_action(GaussianPolicy* policy, float* state, float* action, float* log_prob, int m);
void compute_log_prob(GaussianPolicy* policy, float* out, float* state, float* action, int m);
void log_prob_backwards(GaussianPolicy* policy, float* grad_in, float* grad_mu, float* grad_log_std, int m);

void compute_log_prob_cuda(GaussianPolicy* policy, float* out, float* state, float* action, int m);
void log_prob_backwards_cuda(GaussianPolicy* policy, float* grad_in, float* grad_mu, float* grad_log_std, int m);
float compute_entropy_cuda(GaussianPolicy* policy);

float compute_entropy(GaussianPolicy* policy);
void policy_to_host(GaussianPolicy* policy);

void save_policy(GaussianPolicy* policy, FILE* file);
GaussianPolicy* load_policy(FILE* file, int state_size, int action_size);

#ifdef __cplusplus
}
#endif

#endif // POLICY_H