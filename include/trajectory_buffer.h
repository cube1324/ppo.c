#ifndef TRAJECTORY_BUFFER_H
#define TRAJECTORY_BUFFER_H

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct TrajectoryBuffer TrajectoryBuffer;

struct TrajectoryBuffer
{
    float* state_p;;
    float* action_p;
    float* next_state_p;
    float* reward_p;
    float* logprob_p;
    float* advantage_p;
    float* adv_target_p;
    bool* terminated_p;
    bool* truncated_p;
    int state_size;
    int action_size;
    int capacity;
    int idx;
    bool full;
    float* (*state)(TrajectoryBuffer* buffer, int idx);
    float* (*action)(TrajectoryBuffer* buffer, int idx);
    float* (*next_state)(TrajectoryBuffer* buffer, int idx);
    float* (*reward)(TrajectoryBuffer* buffer, int idx);
    float* (*logprob)(TrajectoryBuffer* buffer, int idx);
    float* (*advantage)(TrajectoryBuffer* buffer, int idx);
    float* (*adv_target)(TrajectoryBuffer* buffer, int idx);
    bool* (*terminated)(TrajectoryBuffer* buffer, int idx);
    bool* (*truncated)(TrajectoryBuffer* buffer, int idx);
};



TrajectoryBuffer* create_trajectory_buffer(int capacity, int state_size, int action_size);
void free_trajectory_buffer(TrajectoryBuffer* buffer);
void sample_batch(TrajectoryBuffer* buffer, int batch_size, float* states, float* actions, float* logprobs, float* advantages, float* adv_targets);

void shuffle_buffer(TrajectoryBuffer* buffer);
void get_batch(TrajectoryBuffer* buffer, int batch_idx, int batch_size, float* states, float* actions, float* logprobs, float* advantages, float* adv_targets);
void reset_buffer(TrajectoryBuffer* buffer);

#endif // TRAJECTORY_BUFFER_H