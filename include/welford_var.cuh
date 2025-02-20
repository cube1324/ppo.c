#ifndef WELFORD_VAR_HU
#define WELFORD_VAR_HU

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

    WelfordState localState;
    if (idx < n) {
        localState.mean = data[idx];
        localState.m2 = 0;
        localState.n = 1;
    } else {
        localState.mean = 0;
        localState.m2 = 0;
        localState.n = 0;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            WelfordState other = sharedData[tid + s];
            WelfordState combined;
            float delta = other.mean - localState.mean;
            int n = localState.n + other.n;
            float mean = localState.mean + delta * other.n / n;
            float m2 = localState.m2 + other.m2 + delta * delta * localState.n * other.n / n;
            combined.mean = mean;
            combined.m2 = m2;
            combined.n = localState.n + other.n;
            localState = combined;
        }
        __syncthreads();
        sharedData[tid] = localState;
    }

    state_out[blockIdx.x] = localState;
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