#include "gym_env.h"

PyObject* pModule;

void init_env(int id, int seed, int* action_size, int* state_size, int* horizon) {

    Py_Initialize();

    // Add the directory containing gym_env.py to the Python path
    PyObject *sys_path = PySys_GetObject("path");
    PyObject *path = PyUnicode_DecodeFSDefault("../scripts");
    PyList_Append(sys_path, path);
    Py_DECREF(path);

    PyObject *pName = PyUnicode_DecodeFSDefault("gym_env");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "init_env");
        if (PyCallable_Check(pFunc)) {
            PyObject *pArgs = PyTuple_Pack(2, PyLong_FromLong(id), PyLong_FromLong(seed));
            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);

            if (PyTuple_Check(pValue) && PyTuple_Size(pValue) == 3) {
                *action_size = PyLong_AsLong(PyTuple_GetItem(pValue, 0));
                *state_size = PyLong_AsLong(PyTuple_GetItem(pValue, 1));
                *horizon = PyLong_AsLong(PyTuple_GetItem(pValue, 2));
            }
            Py_DECREF(pValue);

            Py_DECREF(pArgs);

        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc);
    } else {
        PyErr_Print();
    }
}

void free_env() {
    Py_DECREF(pModule);
    Py_Finalize();
}

void reset_env(float* state) {
    PyObject *pFunc = PyObject_GetAttrString(pModule, "reset_env");
    if (PyCallable_Check(pFunc)) {
        PyObject *pArgs = PyTuple_Pack(0);
        PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);

        if (PyList_Check(pValue)) {
            for (int i = 0; i < PyList_Size(pValue); i++) {
                state[i] = PyFloat_AsDouble(PyList_GetItem(pValue, i));
            }
        }
        Py_DECREF(pValue);
    } else {
        PyErr_Print();
    }
    Py_DECREF(pFunc);
}

void step_env(float* action, float* obs, float* reward, bool* terminated, bool* truncated, int action_size) {
    PyObject *pFunc = PyObject_GetAttrString(pModule, "step_env");
    if (PyCallable_Check(pFunc)) {
        PyObject *pArgs = PyTuple_Pack(1, PyList_New(action_size));
        for (int i = 0; i < action_size; i++) {
            PyList_SetItem(PyTuple_GetItem(pArgs, 0), i, PyFloat_FromDouble(action[i]));
        }
        PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);

        if (PyTuple_Check(pValue)) {
            PyObject *pObs = PyTuple_GetItem(pValue, 0);
            PyObject *pReward = PyTuple_GetItem(pValue, 1);
            PyObject *pTerminated = PyTuple_GetItem(pValue, 2);
            PyObject *pTruncated = PyTuple_GetItem(pValue, 3);

            if (PyList_Check(pObs)) {
            for (int i = 0; i < PyList_Size(pObs); i++) {
                obs[i] = PyFloat_AsDouble(PyList_GetItem(pObs, i));
            }
            }

            *reward = PyFloat_AsDouble(pReward);
            *terminated = PyObject_IsTrue(pTerminated);
            *truncated = PyObject_IsTrue(pTruncated);

        }
        Py_DECREF(pValue);
    } else {
        PyErr_Print();
    }
    Py_DECREF(pFunc);
}


Env* create_gym_env(int id, int seed) {
    Env* env = (Env*)malloc(sizeof(Env));
    init_env(id, seed, &env->action_size, &env->state_size, &env->horizon);
    env->free_env = &free_env;
    env->reset_env = &reset_env;
    env->step_env = &step_env;
    env->gamma = 0.99;
    return env;
}