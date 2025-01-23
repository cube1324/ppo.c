#ifndef TRAJECTORY_BUFFER_H
#define TRAJECTORY_BUFFER_H

#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float* state;
    float* action;
    float* next_state;
    float logprob;
    float reward;
    bool terminated;
    bool truncated;
} Transition;


typedef struct {
    Transition* buffer;
    int state_size;
    int action_size;
    int capacity;
    int idx;
    bool full;
} TrajectoryBuffer;


TrajectoryBuffer* create_trajectory_buffer(int capacity, int state_size, int action_size);
void free_trajectory_buffer(TrajectoryBuffer* buffer);
void sample_batch(TrajectoryBuffer* buffer, int batch_size, float* states, float* actions, float* rewards, float* next_states, bool* terminated, bool* truncated, float* logprobs);


#endif // TRAJECTORY_BUFFER_H