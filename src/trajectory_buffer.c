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

    buffer->random_idx = NULL;

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
    free(buffer->random_idx);
    free(buffer);
}

void shuffle_buffer(TrajectoryBuffer* buffer){
    int limit = buffer->full ? buffer->capacity : buffer->idx;

    free(buffer->random_idx);
    buffer->random_idx = (int*)malloc(limit * sizeof(int));

    for (int i = 0; i < limit; i++) {
        buffer->random_idx[i] = i;
    }

    for (int i = 0; i < limit; i++) {
        int j = rand() % limit;
        int temp = buffer->random_idx[i];
        buffer->random_idx[i] = buffer->random_idx[j];
        buffer->random_idx[j] = temp;
    }
}

void get_batch(TrajectoryBuffer* buffer, int batch_idx, int batch_size, float* states, float* actions, float* logprobs, float* advantages, float* adv_targets) {
    int limit = buffer->full ? buffer->capacity : buffer->idx;

    int offset = batch_idx * batch_size;

    for (int i = 0; i < batch_size; i++) {
        int list_idx = (offset + i) % limit;\
        int idx = buffer->random_idx[list_idx];
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