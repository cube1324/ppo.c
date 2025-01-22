#ifndef TRAJECTORY_BUFFER_H
#define TRAJECTORY_BUFFER_H

typedef struct {
    float* state;
    float* action;
    float* next_state;
    float* logprob;
    float reward;
    bool done;
} Transition;


typedef struct {
    Transition* buffer;
    int capacity;
    int idx;
} TrajectoryBuffer;


TrajectoryBuffer* create_trajectory_buffer(int capacity, int state_size, int action_size);
void free_trajectory_buffer(TrajectoryBuffer* buffer);
void add_transition(TrajectoryBuffer* buffer, float* state, float* action, float reward, float* next_state, float done, float* logprob);
void sample_batch(TrajectoryBuffer* buffer, int batch_size, float* states, float* actions, float* rewards, float* next_states, float* dones, float* logprobs);


#endif // TRAJECTORY_BUFFER_H