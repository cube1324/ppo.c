#ifndef ENV_H
#define ENV_H

#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    void (*free_env)();
    void (*reset_env)(float* state);
    void (*step_env)(float* action, float* obs, float* reward, bool* terminated, bool* truncated, int action_size);
    int state_size;
    int action_size;
} Env;


Env* create_simple_env(int seed);

#endif // ENV_H