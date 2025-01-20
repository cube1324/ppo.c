#include <stdio.h>
#include <math.h>
#include "mat_mul.h"
#include "neural_network.h"
#include <string.h>


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


int main() {
    // Example usage
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

    LossFunction* lossf = (LossFunction*)malloc(sizeof(LossFunction));
    lossf->loss = &mean_squared_error;
    lossf->loss_derivative = &mean_squared_error_derivative;


    for (int i = 0; i < 50; i++){
        forward_propagation(nn, input, m);

        backward_propagation(nn, lossf, y_true, m);

        for (int j = 0; j < nn->num_layers - 1; j++) {
            for (int k = 0; k < nn->layers[j].input_size * nn->layers[j].output_size; k++){
                nn->layers[j].weights[k] -= 0.00001 * nn->layers->grad_weights[k];
            }

            for (int k = 0; k < nn->layers[j].output_size; k++){
                nn->layers[j].biases[k] -= 0.00001 * nn->layers->grad_biases[k];
            }
        }
    }
    

    // Print the output
    printf("Output: [%f %f]\n", nn->output[2], nn->output[3]);

    // Free allocated memory
    free_neural_network(nn);
    // free(lossf);

    return 0;
}