#ifndef GYM_ENV_H
#define GYM_ENV_H

#include "env.h"
#include <Python.h>
#include <stdbool.h>

Env* create_gym_env(int id);


#endif // GYM_ENV_H