#include "nn_model.h"
#include <math.h>

void nn_normalize(const float* raw_input, float* normalized_out) {
    for (int i = 0; i < NN_INPUT_SIZE; i++) {
        float range = feature_max[i] - feature_min[i];
        float v = (range > 1e-6f) ? (raw_input[i] - feature_min[i]) / range : 0.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        normalized_out[i] = v;
    }
}

// w is row-major [out_size x in_size], matching how train_model.py exports it.
static void dense(const float* in, int in_size, const float* w, const float* b,
                   float* out, int out_size) {
    for (int o = 0; o < out_size; o++) {
        float acc = b[o];
        const float* wrow = w + (size_t)o * in_size;
        for (int i = 0; i < in_size; i++) acc += wrow[i] * in[i];
        out[o] = acc;
    }
}

static void relu_inplace(float* v, int n) {
    for (int i = 0; i < n; i++)
        if (v[i] < 0.0f) v[i] = 0.0f;
}

static void softmax_inplace(float* v, int n) {
    float m = v[0];
    for (int i = 1; i < n; i++)
        if (v[i] > m) m = v[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i] = expf(v[i] - m);
        sum += v[i];
    }
    for (int i = 0; i < n; i++) v[i] /= sum;
}

void nn_forward(const float* normalized_input, float* output_probs) {
    static float buf_a[NN_MAX_WIDTH];
    static float buf_b[NN_MAX_WIDTH];

    const float* cur = normalized_input;
    int cur_size = NN_INPUT_SIZE;
    float* out_buf = buf_a;

    for (int l = 0; l < NN_NUM_LAYERS; l++) {
        int out_size = nn_layer_sizes[l];
        out_buf = (l % 2 == 0) ? buf_a : buf_b;
        dense(cur, cur_size, nn_weights[l], nn_biases[l], out_buf, out_size);
        bool is_last = (l == NN_NUM_LAYERS - 1);
        if (!is_last) relu_inplace(out_buf, out_size);
        cur = out_buf;
        cur_size = out_size;
    }

    softmax_inplace(out_buf, cur_size); // cur_size == NUM_GESTURES here
    for (int i = 0; i < cur_size; i++) output_probs[i] = out_buf[i];
}

int nn_argmax(const float* probs, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (probs[i] > probs[best]) best = i;
    return best;
}
