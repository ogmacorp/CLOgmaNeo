__inline float sigmoid(float x) {
    return tanh(x * 0.5f) * 0.5f + 0.5f;
}

// --- Core SPH ---

__kernel void accum_activation(
    __global const int* visible_states,
    __global const float* weights,
    __global float* activations,
    int4 visible_size,
    int3 hidden_size,
    int radius,
    int diam,
    float2 hToV,
    int history_pos
) {
    __local int2 hidden_column_pos;
    __local int hidden_column_index;

    // Project
    __local int2 visible_center;

    // Bounds
    __local int2 field_lower_bound;
    
    __local int2 iter_lower_bound;
    __local int2 iter_upper_bound;

    __local int count;

    __local int num_visible_columns;

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
        hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

        // Project
        visible_center = (int2)((hidden_column_pos.x + 0.5f) * hToV.x, (hidden_column_pos.y + 0.5f) * hToV.y);

        // Bounds
        field_lower_bound = visible_center - radius;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

        count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * visible_size.w;

        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gc = get_global_id(2);

    int hidden_cell_index = gc + hidden_size.z * hidden_column_index;

    float sum = 0.0f;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos + t) % visible_size.w;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                int wi = t + visible_size.w * (visible_state + wi_start);

                sum += weights[wi];
            }
        }

    sum /= count;

    activations[hidden_cell_index] += sum;
}

__kernel void inhibit_activations(
    __global float* activations,
    __global int* states,
    int4 size,
    float scale
) {
    int2 column_pos = (int2)(get_global_id(0), get_global_id(1));
    int column_index = column_pos.y + size.y * column_pos.x;

    int slice = get_global_id(2);

    int max_index = 0;
    float max_activation = -999999.0f;

    for (int c = 0; c < size.z; c++) {
        int cell_index = slice + size.w * (c + size.z * column_index);

        activations[cell_index] *= scale;

        float activation = activations[cell_index];

        if (activation > max_activation) {
            max_activation = activation;
            max_index = c;
        }
    }

    states[slice + size.w * column_index] = max_index;
}

__kernel void encoder_learn(
    __global const int* visible_states,
    __global const int* hidden_states,
    __global float* weights,
    __global float* reconstruction,
    int4 visible_size,
    int3 hidden_size,
    int radius,
    int2 reverse_radii,
    int diam,
    float2 hToV,
    float2 vToH,
    int history_pos,
    float lr
) {
    __local int2 visible_column_pos;
    __local int visible_column_index;

    // Project
    __local int2 hidden_center;

    // Bounds
    __local int2 field_lower_bound;
    
    __local int2 iter_lower_bound;
    __local int2 iter_upper_bound;

    __local int max_index;
    __local float max_activation;

    __local int num_visible_columns;

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        visible_column_pos = (int2)(get_global_id(0), get_global_id(1));
        visible_column_index = visible_column_pos.y + visible_size.y * visible_column_pos.x;

        // Project
        hidden_center = (int2)((visible_column_pos.x + 0.5f) * vToH.x, (visible_column_pos.y + 0.5f) * vToH.y);

        // Bounds
        field_lower_bound = hidden_center - reverse_radii;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));

        max_index = 0;
        max_activation = -999999.0f;

        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gc = get_global_id(2) % visible_size.w;
    int t = get_global_id(2) / visible_size.w;

    int slice = (history_pos + t) % visible_size.w;

    int target_state = visible_states[visible_column_index + num_visible_columns * slice];

    int temporal_visible_cell_index = slice + visible_size.w * (gc + visible_size.z * visible_column_index);

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

                int wi = t + visible_size.w * (gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index)));

                sum += weights[wi];
                count++;
            }
        }

    sum /= max(1, count);

    reconstruction[temporal_visible_cell_index] = sum;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (get_local_id(2) == 0) {
        for (int c = 0; c < visible_size.z; c++) {
            float recon = reconstruction[slice + visible_size.w * (c + visible_size.z * visible_column_index)];

            if (recon > max_activation) {
                max_activation = recon;
                max_index = c;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (max_index != target_state) {
        float delta = lr * ((gc == target_state) - sigmoid(sum));

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

                    int wi = t + visible_size.w * (gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index)));

                    weights[wi] += delta;
                }
            }
    }
}

__kernel void decoder_learn(
    __global const int* visible_states,
    __global const int* target_hidden_states,
    __global const float* activations,
    __global float* weights,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 hToV,
    int history_pos,
    float lr
) {
    __local int2 hidden_column_pos;
    __local int hidden_column_index;

    // Project
    __local int2 visible_center;

    // Bounds
    __local int2 field_lower_bound;
    
    __local int2 iter_lower_bound;
    __local int2 iter_upper_bound;

    __local int target_state;
    
    __local int num_visible_columns;

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
        hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

        // Project
        visible_center = (int2)((hidden_column_pos.x + 0.5f) * hToV.x, (hidden_column_pos.y + 0.5f) * hToV.y);

        // Bounds
        field_lower_bound = visible_center - radius;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

        target_state = target_hidden_states[hidden_column_index];

        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gc = get_global_id(2);

    // For all hidden cells
    int hidden_cell_index = gc + hidden_size.z * hidden_column_index;

    float delta = lr * ((gc == target_state) - sigmoid(activations[hidden_cell_index]));

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos + t) % visible_size.w;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                int wi = t + visible_size.w * (visible_state + wi_start);

                weights[wi] += delta;
            }
        }
}
