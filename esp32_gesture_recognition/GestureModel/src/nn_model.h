#pragma once

// Generic forward pass for a small feedforward classifier:
//   input -> [Dense + ReLU] x (N-1) -> Dense -> softmax
//
// The layer count/sizes/weights come entirely from model_weights.h, which
// train_model.py generates. Put the generated file directly in this
// library's src folder (next to this header, i.e.
// .../libraries/GestureModel/src/model_weights.h) - Arduino's compiler only
// reliably finds a quote-included header if it's alongside the file that
// includes it, not just anywhere in the sketch. This file and nn_model.cpp
// never need to change when you add gestures, add sensors, or resize the
// network - only the generated header does.
//
// model_weights.h is expected to define:
//   NN_INPUT_SIZE, NN_NUM_LAYERS, NN_MAX_WIDTH, NUM_GESTURES
//   nn_layer_sizes[NN_NUM_LAYERS]
//   nn_weights[NN_NUM_LAYERS], nn_biases[NN_NUM_LAYERS]
//   class_labels[NUM_GESTURES]
//   feature_min[NN_INPUT_SIZE], feature_max[NN_INPUT_SIZE]

#include "model_weights.h"

// Scales raw sensor readings into the [0,1] range the model was trained on,
// using the min/max captured from the training dataset.
void nn_normalize(const float* raw_input, float* normalized_out);

// Runs the forward pass. output_probs must have room for NUM_GESTURES floats.
void nn_forward(const float* normalized_input, float* output_probs);

// Returns the index of the highest-probability class in probs[0..n).
int nn_argmax(const float* probs, int n);
