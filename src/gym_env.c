#include "gym_env.h"

PyObject* init_env(int id) {

    Py_Initialize();

    // Add the directory containing gym_env.py to the Python path
    PyObject *sys_path = PySys_GetObject("path");
    PyObject *path = PyUnicode_DecodeFSDefault(".");
    PyList_Append(sys_path, path);
    Py_DECREF(path);

    PyObject *pName = PyUnicode_DecodeFSDefault("gym_env");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "init_env");
        if (PyCallable_Check(pFunc)) {
            PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(id));
            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc);
    } else {
        PyErr_Print();
    }
    return pModule;
}

void free_env(PyObject* pModule) {
    Py_DECREF(pModule);
    Py_Finalize();
}

void reset_env(PyObject* pModule, float* state) {
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

void step_env(PyObject* pModule, float* action, float* obs, float* reward, bool* terminated, bool* truncated, int action_size) {
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

            Py_DECREF(pObs);
            Py_DECREF(pReward);
            Py_DECREF(pTerminated);
            Py_DECREF(pTruncated);
        }
        Py_DECREF(pValue);
    } else {
        PyErr_Print();
    }
    Py_DECREF(pFunc);
}