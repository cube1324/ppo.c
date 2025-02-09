#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "ppo.h"
#include "gym_env.h"
#include "env.h"

void test_nn(){
    srand(0);

    int input_size = 2;
    int output_size = 1;
    int m = 4;


    int layer_sizes[] = {input_size, 32, 32, output_size};
    int num_layers = 4;
    float* input = malloc(m * input_size * sizeof(float));

    float temp_input[] = {0.0, 0.0,
                          0.0, 1.0,
                          1.0, 0.0,
                          1.0, 1.0};

    memcpy(input, temp_input, m * input_size * sizeof(float));

    float y_true[] = {0.0, 1.0, 1.0, 0.0};

    char* activation_functions[] = {"ReLU", "ReLU", "None"};

    NeuralNetwork* nn = create_neural_network(layer_sizes, activation_functions, num_layers);

    Adam* adam = create_adam_from_nn(nn, 0.9, 0.999);
    LossFunction* lossf = (LossFunction*)malloc(sizeof(LossFunction));
    lossf->loss = &mean_squared_error;
    lossf->loss_derivative = &mean_squared_error_derivative;

    for (int i = 0; i < 100; i++){
        forward_propagation(nn, input, m);

        printf("Out %f %f %f %f |  ", nn->output[0],  nn->output[1],  nn->output[2],  nn->output[3]);

        backward_pass(nn, lossf, y_true, m);

        adam_update(adam, 0.01);
    }
    
    // Free allocated memory
    free_neural_network(nn);
    free_adam(adam);
    free(lossf);
}

bool compare_arrays(float* arr1, float* arr2, int size) {
    for (int i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

bool compare_ppo(PPO* ppo, PPO* new_ppo) {
    if (ppo->lambda != new_ppo->lambda) return false;
    if (ppo->epsilon != new_ppo->epsilon) return false;
    if (ppo->ent_coeff != new_ppo->ent_coeff) return false;
    if (ppo->lr_policy != new_ppo->lr_policy) return false;
    if (ppo->lr_V != new_ppo->lr_V) return false;

    // Compare buffer
    if (ppo->buffer->capacity != new_ppo->buffer->capacity) return false;
    if (ppo->buffer->state_size != new_ppo->buffer->state_size) return false;
    if (ppo->buffer->action_size != new_ppo->buffer->action_size) return false;
    // if (!compare_arrays(ppo->buffer->state_p, new_ppo->buffer->state_p, ppo->buffer->capacity * ppo->buffer->state_size)) return false;
    // if (!compare_arrays(ppo->buffer->action_p, new_ppo->buffer->action_p, ppo->buffer->capacity * ppo->buffer->action_size)) return false;

    // Compare policy
    if (ppo->policy->state_size != new_ppo->policy->state_size) return false;
    if (ppo->policy->action_size != new_ppo->policy->action_size) return false;
    if (!compare_arrays(ppo->policy->log_std, new_ppo->policy->log_std, ppo->policy->action_size)) return false;

    // Compare neural network V
    for (int i = 0; i < ppo->V->num_layers; i++) {
        if (!compare_arrays(ppo->V->layers[i].weights, new_ppo->V->layers[i].weights, ppo->V->layers[i].input_size * ppo->V->layers[i].output_size)) return false;
        if (!compare_arrays(ppo->V->layers[i].biases, new_ppo->V->layers[i].biases, ppo->V->layers[i].output_size)) return false;
    }

    // Compare Adam optimizers
    if (ppo->adam_policy->beta1 != new_ppo->adam_policy->beta1) return false;
    if (ppo->adam_policy->beta2 != new_ppo->adam_policy->beta2) return false;
    if (ppo->adam_policy->time_step != new_ppo->adam_policy->time_step) return false;
    if (!compare_arrays(ppo->adam_policy->m, new_ppo->adam_policy->m, ppo->adam_policy->size)) return false;
    if (!compare_arrays(ppo->adam_policy->v, new_ppo->adam_policy->v, ppo->adam_policy->size)) return false;

    return true;
}

int main() {
    // test_nn();
    int seed = time(NULL);
    srand(seed);
    // srand(time(NULL));

    Env* env = create_gym_env(0, seed);
    // Env* env = create_simple_env(0, seed);

    int layer_sizes[] = {env->state_size, 32, 32, env->action_size};
    int num_layers = 4;
    
    char* activation_functions[] = {"relu", "relu", "none"};

    float lr = 3e-4;
    int batch_size = 64;
    float lambda = 0.95;
    float epsilon = 0.2;
    float ent_coeff = 0.0;
    float init_std = 1.0;
    int n_epochs_policy = 4;
    int n_epochs_value = 10;
    int steps_per_epoch = 30000;
    int steps_per_fit = 3000;
    int n_epochs = 10;

    PPO* ppo = create_ppo(activation_functions, layer_sizes, num_layers, steps_per_fit, lr, lr, lambda, epsilon, ent_coeff, init_std);

    eval_ppo(ppo, env, 3000);

    for (int i = 0; i < n_epochs; i++) {
        clock_t tic = clock();
        train_ppo_epoch(ppo, env, steps_per_epoch, batch_size, n_epochs_policy, n_epochs_value);
        clock_t toc = clock();
        printf("Epoch: %d Entropy: %f Time %fs ", i, compute_entropy(ppo->policy), (double)(toc - tic) / CLOCKS_PER_SEC);
        eval_ppo(ppo, env, 3000);
    }

    save_ppo(ppo, "ppo_model.bin");

    env->free_env();
    free(env);
    free_ppo(ppo);
    return 0;
}