#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "ppo.h"
#include "gym_env.h"
#include "env.h"

void ReLU(float* x, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

void ReLU_derivative(float* x, float* grad, int m,  int n) {
    for (int i = 0; i < m * n; i++) {
        grad[i] = x[i] > 0 ? grad[i] : 0;
    }
}

void test_nn(){
    srand(time(NULL));

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

    ActivationFunction activation_functions[] = {{&ReLU, &ReLU_derivative}, {&ReLU, &ReLU_derivative}, {NULL, NULL}};
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

int main() {
    // test_nn();
    srand(time(NULL));

    Env* env = create_gym_env(0);
    // Env* env = create_simple_env(0);

    int layer_sizes[] = {env->state_size, 32, 32, env->action_size};
    int num_layers = 4;
    
    ActivationFunction activation_functions[] = {{&ReLU, &ReLU_derivative}, {&ReLU, &ReLU_derivative}, {NULL, NULL}};

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
    int n_epochs = 100;

    PPO* ppo = create_ppo(activation_functions, layer_sizes, num_layers, steps_per_fit, lr, lr, lambda, epsilon, ent_coeff, init_std);

    eval_ppo(ppo, env, 3000);

    for (int i = 0; i < n_epochs; i++) {
        train_ppo_epoch(ppo, env, steps_per_epoch, batch_size, n_epochs_policy, n_epochs_value);
        eval_ppo(ppo, env, 3000);
    }

    env->free_env();
    free(env);
    free_ppo(ppo);
    return 0;
}