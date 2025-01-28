#ifndef PPO_H
#define PPO_H

#include "trajectory_buffer.h"
#include "policy.h"
#include "neural_network.h"
#include "env.h"
#include "loss.h"
#include "adam.h"

typedef struct {
    TrajectoryBuffer* buffer;
    GaussianPolicy* policy;
    NeuralNetwork* V;
    Adam* adam_policy;
    Adam* adam_V;
    Adam* adam_entropy;
    Env* env;
    float gamma;
    float lambda;
    float epsilon;
    float ent_coeff;
    float lr_policy;
    float lr_V;
} PPO;

PPO* create_ppo(Env* env, ActivationFunction* activation_functions, int* layer_sizes, int num_layers, int buffer_size, float lr_policy, float lr_v, float gamma, float lambda, float epsilon, float ent_coeff, float init_std);

void free_ppo(PPO* ppo);

void collect_trajectories(TrajectoryBuffer* buffer, Env* env, GaussianPolicy* policy, int steps);
void compute_gae(NeuralNetwork* V, TrajectoryBuffer* buffer, float gamma, float lambda);

float policy_loss_and_grad(float* grad_logprob, float* grad_entropy, float* adv, float* logprobs,  float* old_logprobs, float entropy, float ent_coeff, float epsilon, int m);

void train_ppo_epoch(PPO* ppo, int steps_per_epoch, int batch_size, int n_epochs_policy, int n_epochs_value);
void eval_ppo(PPO* ppo, int steps);


#endif // PPO_H
