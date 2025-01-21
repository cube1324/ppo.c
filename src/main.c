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

    // backward_propagation(nn, lossf, y_true, m);

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

        backward_propagation(nn, lossf, y_true, m);

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

    return 0;
}