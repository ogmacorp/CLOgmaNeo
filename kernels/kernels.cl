float sigmoid(float x) {
    return tanh(x * 0.5f) * 0.5f + 0.5f;
}

__kernel void accum_activation(
    __global const int* visible_states,
    __global const float* weights,
    __global float* activations,
    int3 visible_size,
    int3 hidden_size,
    int radius,
    int diam,
    float2 hToV
) {
    int2 hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
    int hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

    // Project
    int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * hToV.x, (hidden_column_pos.y + 0.5f) * hToV.y);

    // Bounds
    int2 field_lower_bound = visible_center - radius;
    
    int2 iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    int2 iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

    // For all hidden cells
    for (int c = 0; c < hidden_size.z; c++) {
        int hidden_cell_index = c + hidden_size.z * hidden_column_index;

        float sum = 0.0f;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 visible_column_pos = (int2)(ix, iy);

                int visible_column_index = iy + visible_size.y * ix;

                int2 offset = visible_column_pos - field_lower_bound;

                int visible_state = visible_states[visible_column_index];

                int wi = visible_state + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                sum += weights[wi];
            }

        activations[hidden_cell_index] += sum;
    }
}

__kernel void inhibit_activations(
    __global const float* activations,
    __global int* states,
    int3 size
) {
    int2 column_pos = (int2)(get_global_id(0), get_global_id(1));
    int column_index = column_pos.y + size.y * column_pos.x;

    int max_index = 0;
    float max_activation = -999999.0f;

    for (int c = 0; c < size.z; c++) {
        float activation = activations[c + size.z * column_index];

        if (activation > max_activation) {
            max_activation = activation;
            max_index = c;
        }
    }

    states[column_index] = max_index;
}

__kernel void encoder_learn(
    __global const int* visible_states,
    __global const int* hidden_states,
    __global float* weights,
    __global float* reconstruction,
    int3 visible_size,
    int3 hidden_size,
    int radius,
    int2 reverse_radii,
    int diam,
    float2 hToV,
    float2 vToH,
    float lr
) {
    int2 visible_column_pos = (int2)(get_global_id(0), get_global_id(1));
    int visible_column_index = visible_column_pos.y + visible_size.y * visible_column_pos.x;

    int target_state = visible_states[visible_column_index];

    // Project
    int2 hidden_center = (int2)((visible_column_pos.x + 0.5f) * vToH.x, (visible_column_pos.y + 0.5f) * vToH.y);

    // Bounds
    int2 field_lower_bound = hidden_center - reverse_radii;
    
    int2 iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    int2 iter_upper_bound = (int2)(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));

    int max_index = 0;
    float max_activation = -999999.0f;

    for (int c = 0; c < visible_size.z; c++) {
        int visible_cell_index = c + visible_size.z * visible_column_index;

        float sum = 0.0f;
        int count = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 hidden_column_pos = (int2)(ix, iy);

                int hidden_column_index = iy + hidden_size.y * ix;

                int hidden_cell_index = hidden_states[hidden_column_index] + hidden_size.z * hidden_column_index;

                // Project
                int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * hToV.x, (hidden_column_pos.y + 0.5f) * hToV.y);

                // Bounds check
                if (visible_column_pos.x >= visible_center.x - radius && visible_column_pos.x <= visible_center.x + radius &&
                    visible_column_pos.y >= visible_center.y - radius && visible_column_pos.y <= visible_center.y + radius)
                {
                    int2 offset = (int2)(visible_column_pos.x - visible_center.x + radius, visible_column_pos.y - visible_center.y + radius);

                    int wi = c + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                    sum += weights[wi];
                    count++;
                }
            }

        sum /= max(1, count);

        reconstruction[visible_cell_index] = sum;

        if (sum > max_activation) {
            max_activation = sum;
            max_index = c;
        }
    }

    if (max_index != target_state) {
        for (int c = 0; c < visible_size.z; c++) {
            int visible_cell_index = c + visible_size.z * visible_column_index;

            float delta = lr * ((c == target_state) - sigmoid(reconstruction[visible_cell_index]));

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int2 hidden_column_pos = (int2)(ix, iy);

                    int hidden_column_index = iy + hidden_size.y * ix;

                    int hidden_cell_index = hidden_states[hidden_column_index] + hidden_size.z * hidden_column_index;

                    // Project
                    int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * hToV.x, (hidden_column_pos.y + 0.5f) * hToV.y);

                    // Bounds check
                    if (visible_column_pos.x >= visible_center.x - radius && visible_column_pos.x <= visible_center.x + radius &&
                        visible_column_pos.y >= visible_center.y - radius && visible_column_pos.y <= visible_center.y + radius)
                    {
                        int2 offset = (int2)(visible_column_pos.x - visible_center.x + radius, visible_column_pos.y - visible_center.y + radius);

                        int wi = c + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                        weights[wi] += delta;
                    }
                }
        }
    }
}

__kernel void decoder_learn(
    __global const int* visible_states,
    __global const int* target_hidden_states,
    __global const float* activations,
    __global float* weights,
    int3 visible_size,
    int3 hidden_size,
    int radius,
    int diam,
    float2 hToV,
    float lr
) {
    int2 hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
    int hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

    // Project
    int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * hToV.x, (hidden_column_pos.y + 0.5f) * hToV.y);

    // Bounds
    int2 field_lower_bound = visible_center - radius;
    
    int2 iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    int2 iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

    int target_state = target_hidden_states[hidden_column_index];

    // For all hidden cells
    for (int c = 0; c < hidden_size.z; c++) {
        int hidden_cell_index = c + hidden_size.z * hidden_column_index;

        float delta = lr * ((c == target_state) - sigmoid(activations[hidden_cell_index]));

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 visible_column_pos = (int2)(ix, iy);

                int visible_column_index = iy + visible_size.y * ix;

                int2 offset = visible_column_pos - field_lower_bound;

                int visible_state = visible_states[visible_column_index];

                int wi = visible_state + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                weights[wi] += delta;
            }
    }
}
