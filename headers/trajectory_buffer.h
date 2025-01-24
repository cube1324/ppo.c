#ifndef TRAJECTORY_BUFFER_H
#define TRAJECTORY_BUFFER_H

#include <stdbool.h>
#include <stdlib.h>
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
    bool* (*terminated)(TrajectoryBuffer* buffer, int idx);
    bool* (*truncated)(TrajectoryBuffer* buffer, int idx);
};


// typedef struct {
//     float* state_p;;
//     float* action_p;
//     float* next_state_p;
//     float* reward_p;
//     float* logprob_p;
//     float* advantage_p;
//     bool* terminated_p;
//     bool* truncated_p;
//     int state_size;
//     int action_size;
//     int capacity;
//     int idx;
//     bool full;
//     float* (*state)(int idx);
//     float* (*action)(int idx);
//     float* (*next_state)(int idx);
//     float* (*reward)(int idx);
//     float* (*logprob)(int idx);
//     float* (*advantage)(int idx);
//     bool* (*terminated)(int idx);
//     bool* (*truncated)(int idx);
// } TrajectoryBuffer;


TrajectoryBuffer* create_trajectory_buffer(int capacity, int state_size, int action_size);
void free_trajectory_buffer(TrajectoryBuffer* buffer);
void sample_batch(TrajectoryBuffer* buffer, int batch_size, float* states, float* actions, float* rewards, float* next_states, bool* terminated, bool* truncated, float* logprobs, float* advantages);

// float* get_state(TrajectoryBuffer* buffer, int idx);
// float* get_next_state(TrajectoryBuffer* buffer, int idx);
// float* get_reward(TrajectoryBuffer* buffer, int idx);
// float* get_logprob(TrajectoryBuffer* buffer, int idx);
// float* get_advantage(TrajectoryBuffer* buffer, int idx);
// bool* get_terminated(TrajectoryBuffer* buffer, int idx);
// bool* get_truncated(TrajectoryBuffer* buffer, int idx);

#endif // TRAJECTORY_BUFFER_H