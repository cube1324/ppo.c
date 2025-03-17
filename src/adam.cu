
#include "adam.h"

#include "cuda_helper.h"

Adam* create_adam(float** weights, float** grad_weights, int* length, int num_layers, int size, float beta1, float beta2) {
    Adam* adam = (Adam*)malloc(sizeof(Adam));
    adam->m = (float*)calloc(size, sizeof(float));
    adam->v = (float*)calloc(size, sizeof(float));
    adam->weights = (float**)malloc(num_layers * sizeof(float*));
    adam->grad_weights = (float**)malloc(num_layers * sizeof(float*));
    adam->lengths = (int*)malloc(num_layers * sizeof(int));
    memcpy(adam->weights, weights, num_layers * sizeof(float*));
    memcpy(adam->grad_weights, grad_weights, num_layers * sizeof(float*));
    memcpy(adam->lengths, length, num_layers * sizeof(int));
    
    adam->size = size;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->time_step = 0;
    adam->num_layers = num_layers;
    return adam;
}

Adam* create_adam_from_nn(NeuralNetwork* nn, float beta1, float beta2) {
    float* weights[2 * nn->num_layers - 1];
    float* grad_weights[2 * nn->num_layers - 1];
    int length[2 * nn->num_layers - 1];

    int size = 0;
    for (int i = 0; i < nn->num_layers - 1; i++) {
        weights[i * 2] = nn->layers[i].weights;
        weights[i * 2 + 1] = nn->layers[i].biases;
        grad_weights[i * 2] = nn->layers[i].grad_weights;
        grad_weights[i * 2 + 1] = nn->layers[i].grad_biases;
        length[i * 2] = nn->layers[i].input_size * nn->layers[i].output_size;
        length[i * 2 + 1] = nn->layers[i].output_size;
        size += nn->layers[i].input_size * nn->layers[i].output_size;
        size += nn->layers[i].output_size;
    }
    return create_adam(weights, grad_weights, length, 2 * (nn->num_layers - 1), size, beta1, beta2);
}

void free_adam(Adam* adam) {
    free(adam->weights);
    free(adam->grad_weights);
    free(adam->lengths);
    free(adam->m);
    free(adam->v);
    free(adam);
}

void adam_update(Adam* adam, float lr) {
    adam->time_step += 1;

    float bias_correction1 = 1 - powf(adam->beta1, adam->time_step);
    float bias_correction2 = 1 - powf(adam->beta2, adam->time_step);

    float step_size = lr / bias_correction1;

    int current_idx = 0;
    for (int i = 0; i < adam->num_layers; i++){
        for (int j = 0; j < adam->lengths[i]; j++){
            adam->m[current_idx] = adam->beta1 * adam->m[current_idx] + (1 - adam->beta1) * adam->grad_weights[i][j];
            adam->v[current_idx] = adam->beta2 * adam->v[current_idx] + (1 - adam->beta2) * powf(adam->grad_weights[i][j], 2);
        
            float denom = sqrtf(adam->v[current_idx] / bias_correction2) + 1e-8;
            
            adam->weights[i][j] -= step_size * adam->m[current_idx] / denom;

            current_idx++;
        }
    }
}

Adam* create_adam_cuda(float** weights, float** grad_weights, int* length, int num_layers, int size, float beta1, float beta2){
    Adam* adam = (Adam*)malloc(sizeof(Adam));

    cudaMalloc(&adam->m, size * sizeof(float));
    cudaMalloc(&adam->v, size * sizeof(float));
    cudaMemset(adam->m, 0, size * sizeof(float));
    cudaMemset(adam->v, 0, size * sizeof(float));

    cudaMalloc(&adam->weights, num_layers * sizeof(float*));
    cudaMalloc(&adam->grad_weights, num_layers * sizeof(float*));
    cudaMalloc(&adam->lengths, num_layers * sizeof(int));

    int presum_length[num_layers];

    presum_length[0] = length[0];
    for (int i = 1; i < num_layers; i++) {
        presum_length[i] = presum_length[i - 1] + length[i];
    }

    cudaMemcpy(adam->weights, weights, num_layers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(adam->grad_weights, grad_weights, num_layers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(adam->lengths, presum_length, num_layers * sizeof(int), cudaMemcpyHostToDevice);

    cudaCheckErrors();
    
    adam->size = size;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->time_step = 0;
    adam->num_layers = num_layers;
    return adam;
}

Adam* create_adam_from_nn_cuda(NeuralNetwork* nn, float beta1, float beta2){
    float* weights[2 * nn->num_layers - 1];
    float* grad_weights[2 * nn->num_layers - 1];
    int length[2 * nn->num_layers - 1];

    int size = 0;
    for (int i = 0; i < nn->num_layers - 1; i++) {
        weights[i * 2] = nn->layers[i].d_weights;
        weights[i * 2 + 1] = nn->layers[i].d_biases;
        grad_weights[i * 2] = nn->layers[i].d_grad_weights;
        grad_weights[i * 2 + 1] = nn->layers[i].d_grad_biases;
        length[i * 2] = nn->layers[i].input_size * nn->layers[i].output_size;
        length[i * 2 + 1] = nn->layers[i].output_size;
        size += nn->layers[i].input_size * nn->layers[i].output_size;
        size += nn->layers[i].output_size;
    }
    return create_adam_cuda(weights, grad_weights, length, 2 * (nn->num_layers - 1), size, beta1, beta2);
}

void free_adam_cuda(Adam* adam){
    cudaFree(adam->m);
    cudaFree(adam->v);
    cudaFree(adam->weights);
    cudaFree(adam->grad_weights);
    cudaFree(adam->lengths);

    free(adam);
}

__global__ void adam_update_kernel(float* m, float* v, float** weights, float** grad_weights, int* presum_length, int num_layers, float beta1, float beta2, float bias_correction2, float step_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int layer = 0;
        while (idx >= presum_length[layer]) {
            layer++;
        }
        int residual_idx = idx - (layer == 0 ? 0 : presum_length[layer - 1]);

        m[idx] = beta1 * m[idx] + (1 - beta1) * grad_weights[layer][residual_idx];
        v[idx] = beta2 * v[idx] + (1 - beta2) * powf(grad_weights[layer][residual_idx], 2);
        float denom = sqrtf(v[idx] / bias_correction2) + 1e-8;
        weights[layer][residual_idx] -= step_size * m[idx] / denom;
    }
}

void adam_update_cuda(Adam* adam, float lr){
    adam->time_step += 1;

    float bias_correction1 = 1 - powf(adam->beta1, adam->time_step);
    float bias_correction2 = 1 - powf(adam->beta2, adam->time_step);

    float step_size = lr / bias_correction1;

    int block_size = BLOCK_SIZE < adam->size ? BLOCK_SIZE : adam->size;

    adam_update_kernel<<<DIVUP(adam->size, block_size), block_size>>>(adam->m, adam->v, adam->weights, adam->grad_weights, adam->lengths, adam->num_layers, adam->beta1, adam->beta2, bias_correction2, step_size, adam->size);

    // cudaDev
    // cudaCheckErrors();
}


void save_adam(Adam* adam, FILE* file, bool cuda) {
    fwrite(&adam->size, sizeof(int), 1, file);
    fwrite(&adam->time_step, sizeof(int), 1, file);
    fwrite(&adam->beta1, sizeof(float), 1, file);
    fwrite(&adam->beta2, sizeof(float), 1, file);
    fwrite(&adam->num_layers, sizeof(int), 1, file);
    if (cuda) {
        float m[adam->size];
        float v[adam->size];
        cudaMemcpy(m, adam->m, adam->size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(v, adam->v, adam->size * sizeof(float), cudaMemcpyDeviceToHost);        
        fwrite(m, sizeof(float), adam->size, file);
        fwrite(v, sizeof(float), adam->size, file);
    } else {
        fwrite(adam->m, sizeof(float), adam->size, file);
        fwrite(adam->v, sizeof(float), adam->size, file);
    }
}

Adam* load_adam(FILE* file, float** weights, float** grad_weights, int* length, bool cuda) {
    Adam* adam = (Adam*)malloc(sizeof(Adam));
    fread(&adam->size, sizeof(int), 1, file);
    fread(&adam->time_step, sizeof(int), 1, file);
    fread(&adam->beta1, sizeof(float), 1, file);
    fread(&adam->beta2, sizeof(float), 1, file);
    fread(&adam->num_layers, sizeof(int), 1, file);

    if (cuda) {
        float m[adam->size];
        float v[adam->size];
        fread(m, sizeof(float), adam->size, file);
        fread(v, sizeof(float), adam->size, file);

        cudaMalloc(&adam->m, adam->size * sizeof(float));
        cudaMalloc(&adam->v, adam->size * sizeof(float));
        cudaMemcpy(adam->m, m, adam->size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(adam->v, v, adam->size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&adam->weights, adam->num_layers * sizeof(float*));
        cudaMalloc(&adam->grad_weights, adam->num_layers * sizeof(float*));
        cudaMalloc(&adam->lengths, adam->num_layers * sizeof(int));

        int presum_length[adam->num_layers];
        presum_length[0] = length[0];
        for (int i = 1; i < adam->num_layers; i++) {
            presum_length[i] = presum_length[i - 1] + length[i];
        }
        cudaMemcpy(adam->weights, weights, adam->num_layers * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(adam->grad_weights, grad_weights, adam->num_layers * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(adam->lengths, presum_length, adam->num_layers * sizeof(int), cudaMemcpyHostToDevice);

        cudaCheckErrors();

    } else {
        adam->m = (float*)malloc(adam->size * sizeof(float));
        adam->v = (float*)malloc(adam->size * sizeof(float));
        fread(adam->m, sizeof(float), adam->size, file);
        fread(adam->v, sizeof(float), adam->size, file);

        adam->weights = (float**)malloc(adam->num_layers * sizeof(float*));
        adam->grad_weights = (float**)malloc(adam->num_layers * sizeof(float*));
        adam->lengths = (int*)malloc(adam->num_layers * sizeof(int));

        memcpy(adam->weights, weights, adam->num_layers * sizeof(float*));
        memcpy(adam->grad_weights, grad_weights, adam->num_layers * sizeof(float*));
        memcpy(adam->lengths, length, adam->num_layers * sizeof(int));
    }
    return adam;
}

Adam* load_adam_from_nn(FILE* file, NeuralNetwork* nn, bool cuda) {
    float* weights[2 * nn->num_layers - 1];
    float* grad_weights[2 * nn->num_layers - 1];
    int length[2 * nn->num_layers - 1];

    for (int i = 0; i < nn->num_layers - 1; i++) {
        if (cuda) {
            weights[i * 2] = nn->layers[i].d_weights;
            weights[i * 2 + 1] = nn->layers[i].d_biases;
            grad_weights[i * 2] = nn->layers[i].d_grad_weights;
            grad_weights[i * 2 + 1] = nn->layers[i].d_grad_biases;
        } else {
            weights[i * 2] = nn->layers[i].weights;
            weights[i * 2 + 1] = nn->layers[i].biases;
            grad_weights[i * 2] = nn->layers[i].grad_weights;
            grad_weights[i * 2 + 1] = nn->layers[i].grad_biases;
        }

        length[i * 2] = nn->layers[i].input_size * nn->layers[i].output_size;
        length[i * 2 + 1] = nn->layers[i].output_size;
    }
    return load_adam(file, weights, grad_weights, length, cuda);
}
