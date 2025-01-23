#ifndef PPO_H
#define PPO_H

#include "trajectory_buffer.h"
#include "policy.h"
#include "neural_network.h"
#include "env.h"
#include "loss.h"

typedef struct {
    TrajectoryBuffer* buffer;
    GaussianPolicy* policy;
    NeuralNetwork* V;
    Env* env;
    float gamma;
    float lambda;
    float epsilon;
} PPO;

PPO* create_ppo(Env* env, ActivationFunction* activation_functions, int* layer_sizes, int num_layers, int buffer_size, float gamma, float lambda, float epsilon, float init_std);

void free_ppo(PPO* ppo);

void collect_trajectories(TrajectoryBuffer* buffer, Env* env, GaussianPolicy* policy, int steps);
void compute_gae(NeuralNetwork* V, float* v_target, float* adv, float* state, float* reward, float* next_state, bool* terminated, bool* truncated, float gamma, float lambda, int m);

float policy_loss(float* adv, float* logprobs,  float* old_logprobs, int epsilon, int m);

void policy_loss_backward(GaussianPolicy* policy, float* grad_out, float* states, float* actions, float* adv, float* old_logprobs, int m);

void train_ppo(PPO* ppo, int epochs, int batch_size, int num_batches);


#endif // PPO_H
