#include "trajectory_buffer.h"

float* get_action(TrajectoryBuffer* buffer, int idx) {
    return buffer->action_p + idx * buffer->action_size;
}

float* get_state(TrajectoryBuffer* buffer, int idx) {
    return buffer->state_p + idx * buffer->state_size;
}

float* get_next_state(TrajectoryBuffer* buffer, int idx) {
    return buffer->next_state_p + idx * buffer->state_size;
}

float* get_reward(TrajectoryBuffer* buffer, int idx) {
    return buffer->reward_p + idx;
}

float* get_logprob(TrajectoryBuffer* buffer, int idx) {
    return buffer->logprob_p + idx;
}

float* get_advantage(TrajectoryBuffer* buffer, int idx) {
    return buffer->advantage_p + idx;
}

bool* get_terminated(TrajectoryBuffer* buffer, int idx) {
    return buffer->terminated_p + idx;
}

bool* get_truncated(TrajectoryBuffer* buffer, int idx) {
    return buffer->truncated_p + idx;
}

TrajectoryBuffer* create_trajectory_buffer(int capacity, int state_size, int action_size) {
    TrajectoryBuffer* buffer = (TrajectoryBuffer*)malloc(sizeof(TrajectoryBuffer));
    buffer->capacity = capacity;
    buffer->idx = 0;
    buffer->state_size = state_size;
    buffer->action_size = action_size;
    buffer->full = false;
    buffer->action_p = (float*)malloc(capacity * action_size * sizeof(float));
    buffer->state_p = (float*)malloc(capacity * state_size * sizeof(float));
    buffer->next_state_p = (float*)malloc(capacity * state_size * sizeof(float));
    buffer->reward_p = (float*)malloc(capacity * sizeof(float));
    buffer->logprob_p = (float*)malloc(capacity * sizeof(float));
    buffer->advantage_p = (float*)malloc(capacity * sizeof(float));
    buffer->terminated_p = (bool*)malloc(capacity * sizeof(bool));
    buffer->truncated_p = (bool*)malloc(capacity * sizeof(bool));

    buffer->action = get_action;
    buffer->state = get_state;
    buffer->next_state = get_next_state;
    buffer->reward = get_reward;
    buffer->logprob = get_logprob;
    buffer->advantage = get_advantage;
    buffer->terminated = get_terminated;
    buffer->truncated = get_truncated;
    
    return buffer;
}


void free_trajectory_buffer(TrajectoryBuffer* buffer) {
    free(buffer->action);
    free(buffer->state);
    free(buffer->next_state);
    free(buffer->reward);
    free(buffer->logprob);
    free(buffer->advantage);
    free(buffer->terminated);
    free(buffer->truncated);
    free(buffer);
}

void sample_batch(TrajectoryBuffer* buffer, int batch_size, float* states, float* actions, float* rewards, float* next_states, bool* terminated, bool* truncated, float* logprobs, float* advantages) {
    int limit = buffer->full ? buffer->capacity : buffer->idx;
    
    for (int i = 0; i < batch_size; i++) {
        int idx = rand() % limit;

        for (int j = 0; j < buffer->state_size; j++) {
            states[i * buffer->state_size + j] = buffer->state(buffer, idx)[j];
            next_states[i * buffer->state_size + j] = buffer->next_state(buffer, idx)[j];
        }
        for (int j = 0; j < buffer->action_size; j++) {
            actions[i * buffer->action_size + j] = buffer->action(buffer, idx)[j];
        }
        rewards[i] = *buffer->reward(buffer, idx);
        logprobs[i] = *buffer->logprob(buffer, idx);
        advantages[i] = *buffer->advantage(buffer, idx);
        terminated[i] = *buffer->terminated(buffer, idx);
        truncated[i] = *buffer->truncated(buffer, idx);
    }
}