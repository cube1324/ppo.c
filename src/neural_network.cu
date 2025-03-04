
#include "neural_network.h"
#include "cuda_helper.h"


NeuralNetwork* create_neural_network(int* layer_sizes, char** activation_functions, int num_layers) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layers = (Layer*)malloc((num_layers - 1) * sizeof(Layer));
    
    nn->activation_functions = (char**)malloc((num_layers - 1) * sizeof(char*));
    for (int i = 0; i < num_layers - 1; i++) {
        nn->activation_functions[i] = strdup(activation_functions[i]);
    }


    for (int i = 0; i < num_layers - 1; i++) {
        nn->layers[i].input_size = layer_sizes[i];
        nn->layers[i].output_size = layer_sizes[i + 1];
        nn->layers[i].weights = (float*)malloc(layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));
        nn->layers[i].biases = (float*)malloc(layer_sizes[i + 1] * sizeof(float));
        nn->layers[i].grad_weights = (float*)calloc(layer_sizes[i] * layer_sizes[i + 1], sizeof(float));
        nn->layers[i].grad_biases = (float*)calloc(layer_sizes[i + 1], sizeof(float));
        nn->layers[i].activation_function = build_activation_function(activation_functions[i]);
        nn->layers[i].input = NULL;

        cudaMalloc(&nn->layers[i].d_weights, layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));
        cudaMalloc(&nn->layers[i].d_biases, layer_sizes[i + 1] * sizeof(float));
        cudaMalloc(&nn->layers[i].d_grad_weights, layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));
        cudaMalloc(&nn->layers[i].d_grad_biases, layer_sizes[i + 1] * sizeof(float));
        nn->layers[i].d_activation_function = build_activation_function_cuda(activation_functions[i]);
        nn->layers[i].d_input = NULL;

        cudaCheckErrors();


        // Initialize weights and biases, he init for hidden layers and xavier for output layer
        float gain = i == num_layers - 2 ? 1 : sqrtf(2.0); 

        float std = gain * sqrtf(2.0 / (layer_sizes[i] + layer_sizes[i + 1]));
        
        for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
            nn->layers[i].weights[j] = (2 * (float)rand() / RAND_MAX - 1) * sqrtf(3.0) * std;
        }

        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            nn->layers[i].biases[j] = (2 * (float)rand() / RAND_MAX - 1) * (1. / sqrtf(layer_sizes[i]));
        }
    }

    nn->output_size = nn->layers[num_layers - 2].output_size;
    nn->output = NULL;

    nn->d_output = NULL;

    nn_write_weights_to_device(nn);
    
    return nn;
}

void forward_propagation_cuda(NeuralNetwork* nn, float* input, int m) {
    cudaFree(nn->layers[0].d_input);

    cudaCheckErrors();

    cudaMalloc(&nn->layers[0].d_input, m * nn->layers[0].input_size * sizeof(float));
    cudaMemcpy(nn->layers[0].d_input, input, m * nn->layers[0].input_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaCheckErrors();

    int last_idx = nn->num_layers - 2;

    for (int i = 0; i < last_idx; i++) {
        cudaFree(nn->layers[i + 1].d_input);

        cudaMalloc(&nn->layers[i + 1].d_input, m * nn->layers[i + 1].input_size * sizeof(float));
        mat_mul_cuda(nn->layers[i + 1].d_input, nn->layers[i].d_input, nn->layers[i].d_weights, nn->layers[i].d_biases, m, nn->layers[i].input_size, nn->layers[i].output_size);

        cudaCheckErrors();

       
        if (nn->layers[i].d_activation_function->activation != NULL){
            nn->layers[i].d_activation_function->activation(nn->layers[i + 1].d_input, m, nn->layers[i + 1].input_size);
        }
    }
    cudaFree(nn->d_output);

    cudaCheckErrors();

    cudaMalloc(&nn->d_output, m * nn->output_size * sizeof(float));

    cudaCheckErrors();

    mat_mul_cuda(nn->d_output, nn->layers[last_idx].d_input, nn->layers[last_idx].d_weights, nn->layers[last_idx].d_biases, m, nn->layers[last_idx].input_size, nn->output_size);

    if (nn->layers[last_idx].d_activation_function->activation != NULL){
        nn->layers[last_idx].d_activation_function->activation(nn->d_output, m, nn->output_size);
    }

}


__global__ void sum_bias_gradients_kernel(float* grad_biases, const float* layer_grad, int output_size, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < output_size) {
        float sum = 0.0f;
        for (int k = 0; k < m; k++) {
            sum += layer_grad[k * output_size + j];
        }
        grad_biases[j] = sum;
    }
}


void backward_propagation_cuda(NeuralNetwork* nn, float* grad_in, int m) {
    // out = activation(x * w + b)

    float* layer_grad;
    cudaMalloc(&layer_grad, m * nn->output_size * sizeof(float));
    cudaMemcpy(layer_grad, grad_in, m * nn->output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaCheckErrors();

    if (nn->layers[nn->num_layers - 2].d_activation_function->activation_derivative != NULL) {
        nn->layers[nn->num_layers - 2].d_activation_function->activation_derivative(nn->d_output, layer_grad, m, nn->output_size);
    }

    cudaCheckErrors();

    for (int i = nn->num_layers - 2; i >= 0; i--) {

        float* temp_grad_x;
        cudaMalloc(&temp_grad_x, m * nn->layers[i].input_size * sizeof(float));
        
        cudaCheckErrors();
        cudaMemset(nn->layers[i].d_grad_weights, 0.0, nn->layers[i].input_size * nn->layers[i].output_size * sizeof(float));
        cudaMemset(nn->layers[i].d_grad_biases, 0.0, nn->layers[i].output_size * sizeof(float));
        
        cudaCheckErrors();

        // Sum over m for derivative with respect to b
        sum_bias_gradients_kernel<<<DIVUP(nn->layers[i].output_size, BLOCK_SIZE), BLOCK_SIZE>>>(nn->layers[i].d_grad_biases, layer_grad, nn->layers[i].output_size, m);

        cudaDeviceSynchronize();
        cudaCheckErrors();


        mat_mul_backwards_cuda(temp_grad_x, nn->layers[i].d_grad_weights, layer_grad, nn->layers[i].d_input, nn->layers[i].d_weights, m, nn->layers[i].input_size, nn->layers[i].output_size);
        // x * w.T + b

        cudaFree(layer_grad);
        layer_grad = temp_grad_x;

        if (i > 0 && nn->layers[i - 1].d_activation_function->activation_derivative != NULL) {
            nn->layers[i - 1].d_activation_function->activation_derivative(nn->layers[i].d_input, layer_grad, m, nn->layers[i].input_size);
        }
        cudaFree(nn->layers[i].d_input);
        nn->layers[i].d_input = NULL;

        cudaCheckErrors();

    }
    cudaFree(layer_grad);
}

void forward_propagation(NeuralNetwork* nn, float* input, int m) {
    free(nn->layers[0].input);
    nn->layers[0].input = (float*)malloc(m * nn->layers[0].input_size * sizeof(float));
    memcpy(nn->layers[0].input, input, m * nn->layers[0].input_size * sizeof(float));

    int last_idx = nn->num_layers - 2;

    for (int i = 0; i < last_idx; i++) {
        free(nn->layers[i + 1].input);

        nn->layers[i + 1].input = (float*)calloc(m * nn->layers[i + 1].input_size, sizeof(float));

        mat_mul(nn->layers[i + 1].input, nn->layers[i].input, nn->layers[i].weights, nn->layers[i].biases, m, nn->layers[i].input_size, nn->layers[i].output_size);

        if (nn->layers[i].activation_function->activation != NULL){
            nn->layers[i].activation_function->activation(nn->layers[i + 1].input, m, nn->layers[i + 1].input_size);
        }
    }
    free(nn->output);
    nn->output = (float*)calloc(m * nn->output_size, sizeof(float));

    mat_mul(nn->output, nn->layers[last_idx].input, nn->layers[last_idx].weights, nn->layers[last_idx].biases, m, nn->layers[last_idx].input_size, nn->output_size);

    if (nn->layers[last_idx].activation_function->activation != NULL){
        nn->layers[last_idx].activation_function->activation(nn->output, m, nn->output_size);
    }
}


void backward_propagation(NeuralNetwork* nn, float* grad_in, int m) {
    // out = activation(x * w + b)
    
    float* layer_grad = (float*)calloc(m * nn->output_size, sizeof(float));

    memcpy(layer_grad, grad_in, m * nn->output_size * sizeof(float));

    if (nn->layers[nn->num_layers - 2].activation_function->activation_derivative != NULL) {
        nn->layers[nn->num_layers - 2].activation_function->activation_derivative(nn->output, layer_grad, m, nn->output_size);
    }

    for (int i = nn->num_layers - 2; i >= 0; i--) {

        float* temp_grad_x = (float*)calloc(m * nn->layers[i].input_size, sizeof(float));

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

        if (i > 0 && nn->layers[i - 1].activation_function->activation_derivative != NULL) {
            nn->layers[i - 1].activation_function->activation_derivative(nn->layers[i].input, layer_grad, m, nn->layers[i].input_size);
        }
        free(nn->layers[i].input);
        nn->layers[i].input = NULL;
    }
    free(layer_grad);
}

void nn_write_weights_to_device(NeuralNetwork* nn){
    for (int i = 0; i < nn->num_layers - 1; i++) {
        cudaMemcpy(nn->layers[i].d_weights, nn->layers[i].weights, nn->layers[i].input_size * nn->layers[i].output_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(nn->layers[i].d_biases, nn->layers[i].biases, nn->layers[i].output_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaCheckErrors();
}

void nn_write_weights_to_host(NeuralNetwork* nn){
    for (int i = 0; i < nn->num_layers - 1; i++) {
        cudaMemcpy(nn->layers[i].weights, nn->layers[i].d_weights, nn->layers[i].input_size * nn->layers[i].output_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn->layers[i].biases, nn->layers[i].d_biases, nn->layers[i].output_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaCheckErrors();
}

void free_neural_network(NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        free(nn->layers[i].weights);
        free(nn->layers[i].biases);
        free(nn->layers[i].grad_weights);
        free(nn->layers[i].grad_biases);
        free(nn->layers[i].input);
        free(nn->layers[i].activation_function);

        cudaFree(nn->layers[i].d_weights);
        cudaFree(nn->layers[i].d_biases);
        cudaFree(nn->layers[i].d_grad_weights);
        cudaFree(nn->layers[i].d_grad_biases);
        cudaFree(nn->layers[i].d_input);

        free(nn->layers[i].d_activation_function);
    }

    for (int i = 0; i < nn->num_layers - 1; i++) {
        free(nn->activation_functions[i]);
    }
    free(nn->activation_functions);

    free(nn->layers);
    free(nn->output);
    free(nn);
}

void save_neural_network(NeuralNetwork* nn, FILE* file) {
    fwrite(&nn->num_layers, sizeof(int), 1, file);
    fwrite(&nn->output_size, sizeof(int), 1, file);

    for (int i = 0; i < nn->num_layers - 1; i++) {
        int length = strlen(nn->activation_functions[i]) + 1;
        fwrite(&length, sizeof(int), 1, file);
        fwrite(nn->activation_functions[i], sizeof(char), length, file);
    }

    for (int i = 0; i < nn->num_layers - 1; i++) {
        fwrite(&nn->layers[i].input_size, sizeof(int), 1, file);
        fwrite(&nn->layers[i].output_size, sizeof(int), 1, file);
        fwrite(nn->layers[i].weights, sizeof(float), nn->layers[i].input_size * nn->layers[i].output_size, file);
        fwrite(nn->layers[i].biases, sizeof(float), nn->layers[i].output_size, file);
    }
}


NeuralNetwork* load_neural_network(FILE* file) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

    fread(&nn->num_layers, sizeof(int), 1, file);
    fread(&nn->output_size, sizeof(int), 1, file);

    nn->layers = (Layer*)malloc((nn->num_layers - 1) * sizeof(Layer));

    nn->activation_functions = (char**)malloc((nn->num_layers - 1) * sizeof(char*));
    for (int i = 0; i < nn->num_layers - 1; i++) {
        int length;
        fread(&length, sizeof(int), 1, file);
        nn->activation_functions[i] = (char*)malloc(length * sizeof(char));
        fread(nn->activation_functions[i], sizeof(char), length, file);
    }

    for (int i = 0; i < nn->num_layers - 1; i++) {
        fread(&nn->layers[i].input_size, sizeof(int), 1, file);
        fread(&nn->layers[i].output_size, sizeof(int), 1, file);

        nn->layers[i].weights = (float*)malloc(nn->layers[i].input_size * nn->layers[i].output_size * sizeof(float));
        nn->layers[i].biases = (float*)malloc(nn->layers[i].output_size * sizeof(float));
        nn->layers[i].grad_weights = (float*)calloc(nn->layers[i].input_size * nn->layers[i].output_size, sizeof(float));
        nn->layers[i].grad_biases = (float*)calloc(nn->layers[i].output_size, sizeof(float));
        nn->layers[i].activation_function = build_activation_function(nn->activation_functions[i]);
        nn->layers[i].input = NULL;

        fread(nn->layers[i].weights, sizeof(float), nn->layers[i].input_size * nn->layers[i].output_size, file);
        fread(nn->layers[i].biases, sizeof(float), nn->layers[i].output_size, file);
    }

    nn->output = NULL;

    return nn;
}
