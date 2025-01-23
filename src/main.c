#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "mat_mul.h"
#include "neural_network.h"
#include "policy.h"
#include "gym_env.h"
#include "trajectory_buffer.h"
#include "ppo.h"

float mean_squared_error(float* y, float* y_true, int m, int n) {
    float loss = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            loss += pow(y_true[i * n + j] - y[i * n + j], 2);
        }
    }
    return loss / m;
}


void mean_squared_error_derivative(float* grad, float* y, float* y_true, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
        // float d_loss = 0.0;
            grad[i * n + j] = 2 * (y[i * n + j] - y_true[i * n + j]);
        }
        // grad[j] = d_loss / m;
    }
}

void ReLU(float* x, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

void ReLU_derivative(float* x, float* grad, int m,  int n) {
    for (int i = 0; i < m * n; i++) {
        grad[i] = grad[i] > 0 ? grad[i] : 0;
    }
}

void test_nn(){
    int input_size = 2;
    int output_size = 1;
    int m = 4;


    int layer_sizes[] = {input_size, 4, output_size};
    int num_layers = 3;
    float* input = malloc(m * input_size * sizeof(float));

    float temp_input[] = {0.0, 0.0,
                          0.0, 1.0,
                          1.0, 0.0,
                          1.0, 1.0};

    memcpy(input, temp_input, m * input_size * sizeof(float));

    float y_true[] = {0.0, 1.0, 1.0, 0.0};

    ActivationFunction activation_functions[] = {{&ReLU, &ReLU_derivative}, {NULL, NULL}};
    NeuralNetwork* nn = create_neural_network(layer_sizes, activation_functions, num_layers);

    LossFunction* lossf = (LossFunction*)malloc(sizeof(LossFunction));
    lossf->loss = &mean_squared_error;
    lossf->loss_derivative = &mean_squared_error_derivative;

    // forward_propagation(nn, input, m);

    // printf("Out %f %f %f %f | ", nn->output[0],  nn->output[1],  nn->output[2],  nn->output[3]);

    // backward_pass(nn, lossf, y_true, m);

    // for (int j = 0; j < 2; j++){
    //     printf("%d\n", j);
    //     for (int i = 0; i < nn->layers[j].input_size * nn->layers[j].output_size; i++){
    //         printf("%f  |  %f \n", nn->layers[j].weights[i], nn->layers[j].grad_weights[i]);
    //     }
    //     printf("------\n");
    //     for (int i = 0; i < nn->layers[j].output_size; i++){
    //         printf("%f  |  %f \n", nn->layers[j].biases[i], nn->layers[j].grad_biases[i]);
    //     }
    // }

    // float delta = 0.005;

    // for (int j = 0; j < 2; j++){
    //     for (int i = 0; i < nn->layers[j].input_size * nn->layers[j].output_size; i++){
    //         nn->layers[j].weights[i] += delta;
    //         forward_propagation(nn, input, m);
    //         float loss1 = lossf->loss(nn->output, y_true, m, nn->output_size);

    //         nn->layers[j].weights[i] -= 2 * delta;
    //         forward_propagation(nn, input, m);
    //         float loss2 = lossf->loss(nn->output, y_true, m, nn->output_size);

    //         nn->layers[j].weights[i] += delta;

    //         printf("%f %f | %f  |  %f \n", loss1, loss2, (loss1 - loss2) / (2 * delta), nn->layers[j].grad_weights[i]);
    //     }
    //     printf("------\n");
    // }



    for (int i = 0; i < 2000; i++){
        forward_propagation(nn, input, m);

        printf("Out %f %f %f %f |  ", nn->output[0],  nn->output[1],  nn->output[2],  nn->output[3]);

        backward_pass(nn, lossf, y_true, m);

        for (int j = 0; j < nn->num_layers - 1; j++) {
            for (int k = 0; k < nn->layers[j].input_size * nn->layers[j].output_size; k++){
                nn->layers[j].weights[k] -= 0.01 * nn->layers->grad_weights[k];
                // printf("%f ", nn->layers->grad_weights[k]);
            }

            for (int k = 0; k < nn->layers[j].output_size; k++){
                nn->layers[j].biases[k] -= 0.01 * nn->layers->grad_biases[k];
            }
        }
        // printf("\n ------------------- \n");
    }
    

    // Print the output
    printf("Output: [%f]\n", nn->output[0]);

    // Free allocated memory
    free_neural_network(nn);
    // free(lossf);
}


void test_policy() {
    srand(time(NULL)); 
    int state_size = 2;
    int action_size = 1;
    int m = 4;

    int layer_sizes[] = {state_size, 4, action_size};
    int num_layers = 3;
    ActivationFunction activation_functions[] = {{&ReLU, &ReLU_derivative}, {NULL, NULL}};
    GaussianPolicy* policy = create_gaussian_policy(layer_sizes, activation_functions, num_layers, 1);

    float out[5];
    generate_gaussian_noise(out, 5);

    printf("Noise: %f %f %f %f %f\n", out[0], out[1], out[2], out[3], out[4]);
}


int main() {
    // Example usage
    // test_policy();
    // test_nn();

    int state_size = 3;
    int action_size = 1;
    int m = 4;

    int layer_sizes[] = {state_size, 4, action_size};
    int num_layers = 3;
    ActivationFunction activation_functions[] = {{&ReLU, &ReLU_derivative}, {NULL, NULL}};

    GaussianPolicy* policy = create_gaussian_policy(layer_sizes, activation_functions, num_layers, 0.1);

    Env* env = create_gym_env(0);

    TrajectoryBuffer* buffer = create_trajectory_buffer(1000, state_size, action_size);

    collect_trajectories(buffer, env, policy, 10);

    for (int i = 0; i < 10; i++) {
        printf("%f %f %f | %f | %f | %d | %d\n", buffer->buffer[i].state[0], buffer->buffer[i].state[1], buffer->buffer[i].state[2], buffer->buffer[i].action[0], buffer->buffer[i].reward, buffer->buffer[i].terminated, buffer->buffer[i].truncated);
    }
    // float obs[3];

    // env->reset_env(obs);

    // for (int i = 0; i < 3; i++) {
    //     printf("%f ", obs[i]);
    // }

    // float reward;
    // bool terminated;
    // bool truncated;
    // float action[1] = {1.0};

    // env->step_env(action, obs, &reward, &terminated, &truncated, 1);
    // printf("\n");
    // for (int i = 0; i < 3; i++) {
    //     printf("%f ", obs[i]);
    // }

    // env->free_env();

    // free(env);


    return 0;
}