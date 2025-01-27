#include "env.h"

#include <math.h>


float state = 0;
int step = 0;

void reset_simple_env(float* obs) {
    state = 0;
    step = 0;
    obs[0] = 0;
}

void step_simple_env(float* action, float* obs, float* reward, bool* terminated, bool* truncated, int action_size) {
    state += fmaxf(fminf(action[0], 1), -1);
    obs[0] = state;
    step += 1;
    
    if (state >= 5) {
        reward[0] = 1;
        terminated[0] = true;
        truncated[0] = false;
    } else if (step >= 15) {
        reward[0] = 0;
        terminated[0] = false;
        truncated[0] = true;
    } else {
        reward[0] = 0;
        terminated[0] = false;
        truncated[0] = false;
    }
}

void free_simple_env() {
    // Nothing to free
}



Env* create_simple_env(int seed) {
    Env* env = (Env*)malloc(sizeof(Env));
    env->state_size = 1;
    env->action_size = 1;
    env->reset_env = reset_simple_env;
    env->step_env = step_simple_env;
    env->free_env = free_simple_env;
    return env;
}

