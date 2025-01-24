#include "loss.h"

float mean_squared_error(float* y, float* y_true, int m, int n) {
    float loss = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            loss += pow(y_true[i * n + j] - y[i * n + j], 2);
        }
    }
    return loss / (m * n);
}


void mean_squared_error_derivative(float* grad, float* y, float* y_true, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
        // float d_loss = 0.0;
            grad[i * n + j] = 2 * (y[i * n + j] - y_true[i * n + j]) / (m * n);
        }
    }
}