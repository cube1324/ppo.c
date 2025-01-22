#ifndef GYM_ENV_H
#define GYM_ENV_H

#include <Python.h>
#include <stdbool.h>


PyObject* init_env(int id);
void free_env(PyObject* pModule);
void reset_env(PyObject* pModule, float* state);
void step_env(PyObject* pModule, float* action, float* obs, float* reward, bool* terminated, bool* truncated, int action_size);


#endif // GYM_ENV_H