#ifndef WELFORD_VAR_HU
#define WELFORD_VAR_HU

#include <cuda_runtime.h>

typedef struct
{
    float mean;
    float m2;
    int n;
} WelfordState;

__global__ void welford_var_kernel(float *data, int n, WelfordState *state_out) {
    extern __shared__ WelfordState sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        sharedData[tid].mean = data[idx];
        sharedData[tid].m2 = 0.0f;
        sharedData[tid].n = 1;
    } else {
        sharedData[tid].mean = 0.0f;
        sharedData[tid].m2 = 0.0f;
        sharedData[tid].n = 0;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        WelfordState localState = sharedData[tid];
        if (tid < s) {
            WelfordState other = sharedData[tid + s];
            float delta = other.mean - localState.mean;
            int n = localState.n + other.n;
            if (n > 0){
                float mean = localState.mean + delta * other.n / n;
                float m2 = localState.m2 + other.m2 + delta * delta * localState.n * other.n / n;
                localState.mean = mean;
                localState.m2 = m2;
                localState.n = n;
            }
        }
        __syncthreads();
        sharedData[tid] = localState;
    }

    if (tid == 0) {
        state_out[blockIdx.x] = sharedData[0];
    }
}


void welford_combine_blocks(WelfordState* block_states, int n_blocks, WelfordState* state_out) {
    WelfordState state;
    state.mean = 0;
    state.m2 = 0;
    state.n = 0;
    for (int i = 0; i < n_blocks; i++) {
        WelfordState block_state = block_states[i];
        float delta = block_state.mean - state.mean;
        int n = state.n + block_state.n;
        float mean = state.mean + delta * block_state.n / n;
        float m2 = state.m2 + block_state.m2 + delta * delta * state.n * block_state.n / n;
        state.mean = mean;
        state.m2 = m2;
        state.n = n;
    }
    *state_out = state;
}



#endif // WELFORD_VAR_HU