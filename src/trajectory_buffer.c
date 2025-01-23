#include "trajectory_buffer.h"


TrajectoryBuffer* create_trajectory_buffer(int capacity, int state_size, int action_size) {
    TrajectoryBuffer* buffer = (TrajectoryBuffer*)malloc(sizeof(TrajectoryBuffer));
    buffer->buffer = (Transition*)malloc(capacity * sizeof(Transition));
    buffer->capacity = capacity;
    buffer->idx = 0;
    buffer->state_size = state_size;
    buffer->action_size = action_size;
    for (int i = 0; i < capacity; i++) {
        buffer->buffer[i].state = (float*)malloc(state_size * sizeof(float));
        buffer->buffer[i].action = (float*)malloc(action_size * sizeof(float));
        buffer->buffer[i].next_state = (float*)malloc(state_size * sizeof(float));
    }
    return buffer;
}


void free_trajectory_buffer(TrajectoryBuffer* buffer) {
    for (int i = 0; i < buffer->capacity; i++) {
        free(buffer->buffer[i].state);
        free(buffer->buffer[i].action);
        free(buffer->buffer[i].next_state);
    }
    free(buffer->buffer);
    free(buffer);
}

void sample_batch(TrajectoryBuffer* buffer, int batch_size, float* states, float* actions, float* rewards, float* next_states, bool* terminated, bool* truncated, float* logprobs) {
    int limit = buffer->full ? buffer->capacity : buffer->idx;
    
    for (int i = 0; i < batch_size; i++) {
        int idx = rand() % limit;

        for (int j = 0; j < buffer->state_size; j++) {
            states[i * buffer->state_size + j] = buffer->buffer[idx].state[j];
            next_states[i * buffer->state_size + j] = buffer->buffer[idx].next_state[j];
        }
        for (int j = 0; j < buffer->action_size; j++) {
            actions[i * buffer->action_size + j] = buffer->buffer[idx].action[j];
        }
        rewards[i] = buffer->buffer[idx].reward;
        terminated[i] = buffer->buffer[idx].terminated;
        truncated[i] = buffer->buffer[idx].truncated;
        logprobs[i] = buffer->buffer[idx].logprob;
    }
}