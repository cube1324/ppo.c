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

float* get_adv_target(TrajectoryBuffer* buffer, int idx) {
    return buffer->adv_target_p + idx;
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
    buffer->adv_target_p = (float*)malloc(capacity * sizeof(float));
    buffer->terminated_p = (bool*)malloc(capacity * sizeof(bool));
    buffer->truncated_p = (bool*)malloc(capacity * sizeof(bool));

    buffer->action = get_action;
    buffer->state = get_state;
    buffer->next_state = get_next_state;
    buffer->reward = get_reward;
    buffer->logprob = get_logprob;
    buffer->advantage = get_advantage;
    buffer->adv_target = get_adv_target;
    buffer->terminated = get_terminated;
    buffer->truncated = get_truncated;
    
    return buffer;
}


void free_trajectory_buffer(TrajectoryBuffer* buffer) {
    free(buffer->action_p);
    free(buffer->state_p);
    free(buffer->next_state_p);
    free(buffer->reward_p);
    free(buffer->logprob_p);
    free(buffer->advantage_p);
    free(buffer->adv_target_p);
    free(buffer->terminated_p);
    free(buffer->truncated_p);
    free(buffer);
}

void shuffle_buffer(TrajectoryBuffer* buffer){
    int limit = buffer->full ? buffer->capacity : buffer->idx;
    for (int i = 0; i < limit; i++) {
        int idx = rand() % limit;
        float* state = buffer->state(buffer, i);
        float* action = buffer->action(buffer, i);
        float* next_state = buffer->next_state(buffer, i);
        float* reward = buffer->reward(buffer, i);
        float* logprob = buffer->logprob(buffer, i);
        float* advantage = buffer->advantage(buffer, i);
        float* adv_target = buffer->adv_target(buffer, i);
        bool* terminated = buffer->terminated(buffer, i);
        bool* truncated = buffer->truncated(buffer, i);

        memcpy(buffer->state(buffer, i), buffer->state(buffer, idx), buffer->state_size * sizeof(float));
        memcpy(buffer->action(buffer, i), buffer->action(buffer, idx), buffer->action_size * sizeof(float));
        memcpy(buffer->next_state(buffer, i), buffer->next_state(buffer, idx), buffer->state_size * sizeof(float));
        *reward = *buffer->reward(buffer, idx);
        *logprob = *buffer->logprob(buffer, idx);
        *advantage = *buffer->advantage(buffer, idx);
        *adv_target = *buffer->adv_target(buffer, idx);
        *terminated = *buffer->terminated(buffer, idx);
        *truncated = *buffer->truncated(buffer, idx);

        memcpy(buffer->state(buffer, idx), state, buffer->state_size * sizeof(float));
        memcpy(buffer->action(buffer, idx), action, buffer->action_size * sizeof(float));
        memcpy(buffer->next_state(buffer, idx), next_state, buffer->state_size * sizeof(float));
        *buffer->reward(buffer, idx) = *reward;
        *buffer->logprob(buffer, idx) = *logprob;
        *buffer->advantage(buffer, idx) = *advantage;
        *buffer->adv_target(buffer, idx) = *adv_target;
        *buffer->terminated(buffer, idx) = *terminated;
        *buffer->truncated(buffer, idx) = *truncated;
    }
}

void get_batch(TrajectoryBuffer* buffer, int batch_idx, int batch_size, float* states, float* actions, float* logprobs, float* advantages, float* adv_targets) {
    int limit = buffer->full ? buffer->capacity : buffer->idx;

    int offset = batch_idx * batch_size;

    for (int i = 0; i < batch_size; i++) {
        int idx = (offset + i) % limit;
        for (int j = 0; j < buffer->state_size; j++) {
            states[i * buffer->state_size + j] = buffer->state(buffer, idx)[j];
        }
        for (int j = 0; j < buffer->action_size; j++) {
            actions[i * buffer->action_size + j] = buffer->action(buffer, idx)[j];
        }
        logprobs[i] = *buffer->logprob(buffer, idx);
        advantages[i] = *buffer->advantage(buffer, idx);
        adv_targets[i] = *buffer->adv_target(buffer, idx);
    }
}

void sample_batch(TrajectoryBuffer* buffer, int batch_size, float* states, float* actions, float* logprobs, float* advantages, float* adv_targets) {
    int limit = buffer->full ? buffer->capacity : buffer->idx;
    
    for (int i = 0; i < batch_size; i++) {
        int idx = rand() % limit;

        for (int j = 0; j < buffer->state_size; j++) {
            states[i * buffer->state_size + j] = buffer->state(buffer, idx)[j];
        }
        for (int j = 0; j < buffer->action_size; j++) {
            actions[i * buffer->action_size + j] = buffer->action(buffer, idx)[j];
        }
        logprobs[i] = *buffer->logprob(buffer, idx);
        advantages[i] = *buffer->advantage(buffer, idx);
        adv_targets[i] = *buffer->adv_target(buffer, idx);
    }
}

void reset_buffer(TrajectoryBuffer* buffer) {
    buffer->idx = 0;
    buffer->full = false;
}