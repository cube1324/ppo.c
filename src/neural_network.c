
#include "neural_network.h"

NeuralNetwork* create_neural_network(int* layer_sizes, ActivationFunction* activation_functions, int num_layers) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layers = (Layer*)malloc((num_layers - 1) * sizeof(Layer));

    for (int i = 0; i < num_layers - 1; i++) {
        nn->layers[i].input_size = layer_sizes[i];
        nn->layers[i].output_size = layer_sizes[i + 1];
        nn->layers[i].weights = (float*)malloc(layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));
        nn->layers[i].biases = (float*)calloc(layer_sizes[i + 1], sizeof(float));
        nn->layers[i].grad_weights = (float*)calloc(layer_sizes[i] * layer_sizes[i + 1], sizeof(float));
        nn->layers[i].grad_biases = (float*)calloc(layer_sizes[i + 1], sizeof(float));
        nn->layers[i].activation_function = &activation_functions[i];
        nn->layers[i].input = NULL;

        // Initialize weights and biases, he init for hidden layers and xavier for output layer
        float scale = i == num_layers - 2 ? sqrtf(6.0 / (layer_sizes[i] + layer_sizes[i + 1])) : sqrtf(6.0 / layer_sizes[i]); 

        for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
            for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
                nn->layers[i].weights[j] = (2 * (float)rand() / RAND_MAX - 1) * scale;
            }
        }
    }

    nn->output_size = nn->layers[num_layers - 2].output_size;
    nn->output = NULL;

    return nn;
}

void forward_propagation(NeuralNetwork* nn, float* input, int m) {
    nn->layers[0].input = input;

    int last_idx = nn->num_layers - 2;

    for (int i = 0; i < last_idx; i++) {
        free(nn->layers[i + 1].input);
        nn->layers[i + 1].input = calloc(m * nn->layers[i + 1].input_size, sizeof(float));

        mat_mul(nn->layers[i + 1].input, nn->layers[i].input, nn->layers[i].weights, nn->layers[i].biases, m, nn->layers[i].input_size, nn->layers[i].output_size);

        if (nn->layers[i].activation_function->activation != NULL){
            nn->layers[i].activation_function->activation(nn->layers[i + 1].input, m, nn->layers[i + 1].input_size);
        }
    }
    free(nn->output);
    nn->output = calloc(m * nn->output_size, sizeof(float));

    mat_mul(nn->output, nn->layers[last_idx].input, nn->layers[last_idx].weights, nn->layers[last_idx].biases, m, nn->layers[last_idx].input_size, nn->output_size);

    if (nn->layers[last_idx].activation_function->activation != NULL){
        nn->layers[last_idx].activation_function->activation(nn->output, m, nn->output_size);
    }
}

void backward_pass(NeuralNetwork* nn, LossFunction* lossf, float* y_true, int m) {
    float loss = lossf->loss(nn->output, y_true, m, nn->output_size);
    printf("Loss: %f\n", loss);

    float loss_grad[m * nn->output_size];

    lossf->loss_derivative(loss_grad, nn->output, y_true, m, nn->output_size);
    backward_propagation(nn, loss_grad, m);
}

void backward_propagation(NeuralNetwork* nn, float* grad_in, int m) {
    // out = activation(x * w + b)
    
    float* layer_grad = calloc(m * nn->output_size, sizeof(float));

    memcpy(layer_grad, grad_in, m * nn->output_size * sizeof(float));

    if (nn->layers[nn->num_layers - 2].activation_function->activation_derivative != NULL) {
        nn->layers[nn->num_layers - 2].activation_function->activation_derivative(nn->output, layer_grad, m, nn->output_size);
    }

    for (int i = nn->num_layers - 2; i >= 0; i--) {

        float* temp_grad_x = calloc(m * nn->layers[i].input_size, sizeof(float));

        memset(nn->layers[i].grad_weights, 0.0, nn->layers[i].input_size * nn->layers[i].output_size * sizeof(float));
        memset(nn->layers[i].grad_biases, 0.0, nn->layers[i].output_size * sizeof(float));

        // Sum over m for derivative with respect to b
        for (int j = 0; j < nn->layers[i].output_size; j++){
            for (int k = 0; k < m; k++){
                nn->layers[i].grad_biases[j] += layer_grad[k * nn->layers[i].output_size + j];
            }
        }

        // Compute derivative for x and w
        mat_mul_backwards(temp_grad_x, nn->layers[i].grad_weights, layer_grad, nn->layers[i].input, nn->layers[i].weights, m, nn->layers[i].input_size, nn->layers[i].output_size);
        // x * w.T + b

        free(layer_grad);
        layer_grad = temp_grad_x;

        // Apply 
        if (i > 0 && nn->layers[i].activation_function->activation_derivative != NULL) {
            nn->layers[i - 1].activation_function->activation_derivative(nn->layers[i].input, layer_grad, m, nn->layers[i].input_size);
        }
    }
    free(layer_grad);
}

void free_neural_network(NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        free(nn->layers[i].weights);
        free(nn->layers[i].biases);
        free(nn->layers[i].grad_weights);
        free(nn->layers[i].grad_biases);
        free(nn->layers[i].input);
    }
    free(nn->layers);
    free(nn->output);
    free(nn);
}
