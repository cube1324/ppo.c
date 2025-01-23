#ifndef PPO_H
#define PPO_H

#include "trajectory_buffer.h"
#include "policy.h"
#include "neural_network.h"
#include "env.h"

typedef struct {
    TrajectoryBuffer* buffer;
    GaussianPolicy* policy;
    NeuralNetwork* V;
    float gamma;
    float lambda;
    float epsilon;
} PPO;

PPO* create_ppo(int capacity, int state_size, int action_size, int* layer_sizes, ActivationFunction* activation_functions, int num_layers, float gamma, float lambda, float epsilon, float init_std);
void free_ppo(PPO* ppo);

void collect_trajectories(TrajectoryBuffer* buffer, Env* env, GaussianPolicy* policy, int steps);
void train_ppo(PPO* ppo, int epochs, int batch_size, int num_batches);


#endif // PPO_H
