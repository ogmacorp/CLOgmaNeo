// ----------------------------------------------------------------------------
//  CLOgmaNeo
//  Copyright(c) 2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of CLOgmaNeo is licensed to you under the terms described
//  in the CLOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// --- Core SPH ---

__kernel void accum_sparse_activations(
    __global const int* visible_states,
    __global const float* weights,
    __global float* activations,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
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
        visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

        // Bounds
        field_lower_bound = visible_center - radius;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

        count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * visible_size.w;

        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gt = get_global_id(2) / hidden_size.z;
    int gc = get_global_id(2) % hidden_size.z;

    int hidden_cell_index = gt + hidden_size.w * (gc + hidden_size.z * hidden_column_index);

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

__kernel void accum_dense_activations(
    __global const float* visible_states,
    __global const float* weights,
    __global float* activations,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
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
    __local int num_visible_cells;

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
        hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

        // Project
        visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

        // Bounds
        field_lower_bound = visible_center - radius;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

        count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * visible_size.w;

        num_visible_columns = visible_size.x * visible_size.y;
        num_visible_cells = num_visible_columns * visible_size.z;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gt = get_global_id(2) / hidden_size.z;
    int gc = get_global_id(2) % hidden_size.z;

    int hidden_cell_index = gt + hidden_size.w * (gc + hidden_size.z * hidden_column_index);

    float sum = 0.0f;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos + t) % visible_size.w;

                for (int c = 0; c < visible_size.z; c++) {
                    int visible_cell_index = c + visible_size.z * visible_column_index;

                    float visible_state = visible_states[visible_cell_index + num_visible_cells * slice];

                    int wi = t + visible_size.w * (c + wi_start);

                    sum += weights[wi] * visible_state;
                }
            }
        }

    sum /= count;

    activations[hidden_cell_index] += sum;
}

__kernel void sparse_activations(
    __global float* activations,
    __global int* states,
    int4 size,
    float scale
) {
    int2 column_pos = (int2)(get_global_id(0), get_global_id(1));
    int column_index = column_pos.y + size.y * column_pos.x;

    int gt = get_global_id(2);

    int max_index = 0;
    float max_activation = -999999.0f;

    for (int c = 0; c < size.z; c++) {
        int cell_index = gt + size.w * (c + size.z * column_index);

        activations[cell_index] *= scale;

        float activation = activations[cell_index];

        if (activation > max_activation) {
            max_activation = activation;
            max_index = c;
        }
    }

    states[column_index + gt * size.x * size.y] = max_index;
}

__kernel void dense_tanh_activations(
    __global float* activations,
    int4 size,
    float scale
) {
    int2 column_pos = (int2)(get_global_id(0), get_global_id(1));
    int column_index = column_pos.y + size.y * column_pos.x;

    int gt = get_global_id(2) / size.z;
    int gc = get_global_id(2) % size.z;

    int cell_index = gt + size.w * (gc + size.z * column_index);

    activations[cell_index] = max(0.0f, tanh(activations[cell_index] * scale));
}

__kernel void encoder_learn(
    __global const int* visible_states,
    __global const float* activations,
    __global const float* errors,
    __global float* weights,
    int4 visible_size,
    int3 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    int history_pos,
    float lr,
    float reg
) {
    __local int2 hidden_column_pos;
    __local int hidden_column_index;

    // Project
    __local int2 visible_center;

    // Bounds
    __local int2 field_lower_bound;
    
    __local int2 iter_lower_bound;
    __local int2 iter_upper_bound;

    __local int num_visible_columns;

    __local int num_non_zero_activations;

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
        hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

        // Project
        visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

        // Bounds
        field_lower_bound = visible_center - radius;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

        num_visible_columns = visible_size.x * visible_size.y;

        num_non_zero_activations = 0;

        // Find number of non-zero activations
        for (int c = 0; c < hidden_size.z; c++) {
            int hidden_cell_index = c + hidden_size.z * hidden_column_index;

            if (activations[hidden_cell_index] > 0.0f)
                num_non_zero_activations++;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gc = get_global_id(2);

    int hidden_cell_index = gc + hidden_size.z * hidden_column_index;

    float activation = activations[hidden_cell_index];

    if (activation == 0.0f)
        return;

    float delta = lr * (errors[hidden_cell_index] * (1.0f - activation * activation) - reg * (num_non_zero_activations > 1));

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

__kernel void decoder_sparse_learn(
    __global const int* visible_states,
    __global const int* target_hidden_states,
    __global const float* activations,
    __global float* weights,
    int3 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    int target_pos,
    int target_temporal_horizon,
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

    __local int num_hidden_columns;
    __local int num_visible_columns;

    __local int target_state;

    int gt = get_global_id(2) / hidden_size.z;
    int gc = get_global_id(2) % hidden_size.z;

    int gslice = (target_pos + gt) % target_temporal_horizon;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
        hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

        // Project
        visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

        // Bounds
        field_lower_bound = visible_center - radius;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

        num_hidden_columns = hidden_size.x * hidden_size.y;
        num_visible_columns = visible_size.x * visible_size.y;

        target_state = target_hidden_states[hidden_column_index + num_hidden_columns * gslice];
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int hidden_cell_index = gt + hidden_size.w * (gc + hidden_size.z * hidden_column_index);

    float delta = lr * ((gc == target_state) - activations[hidden_cell_index]);

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            int visible_state = visible_states[visible_column_index];

            int wi = visible_state + wi_start;

            weights[wi] += delta;
        }
}

__kernel void decoder_dense_learn(
    __global const float* visible_states,
    __global const int* target_hidden_states,
    __global const float* activations,
    __global float* weights,
    int3 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    int target_pos,
    int target_temporal_horizon,
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

    __local int num_hidden_columns;
    __local int num_visible_columns;
    __local int num_visible_cells;

    __local int target_state;

    int gt = get_global_id(2) / hidden_size.z;
    int gc = get_global_id(2) % hidden_size.z;

    int gslice = (target_pos + gt) % target_temporal_horizon;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
        hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

        // Project
        visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

        // Bounds
        field_lower_bound = visible_center - radius;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

        num_hidden_columns = hidden_size.x * hidden_size.y;
        num_visible_columns = visible_size.x * visible_size.y;
        num_visible_cells = num_visible_columns * visible_size.z;

        target_state = target_hidden_states[hidden_column_index + num_hidden_columns * gslice];
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int hidden_cell_index = gt + hidden_size.w * (gc + hidden_size.z * hidden_column_index);

    float delta = lr * ((gc == target_state) - activations[hidden_cell_index]);

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            for (int c = 0; c < visible_size.z; c++) {
                int visible_cell_index = c + visible_size.z * visible_column_index;

                float visible_state = visible_states[visible_cell_index];

                int wi = c + wi_start;

                weights[wi] += delta * visible_state;
            }
        }
}

__kernel void decoder_generate_errors(
    __global const int* target_hidden_states,
    __global const float* activations,
    __global const float* weights,
    __global float* errors,
    int3 visible_size,
    int4 hidden_size,
    int radius,
    int2 reverse_radii,
    int diam,
    float2 h_to_v,
    float2 v_to_h,
    int target_pos,
    int target_temporal_horizon
) {
    __local int2 visible_column_pos;
    __local int visible_column_index;

    // Project
    __local int2 hidden_center;

    // Bounds
    __local int2 field_lower_bound;
    
    __local int2 iter_lower_bound;
    __local int2 iter_upper_bound;

    __local int num_hidden_columns;
    __local int num_visible_columns;

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        visible_column_pos = (int2)(get_global_id(0), get_global_id(1));
        visible_column_index = visible_column_pos.y + visible_size.y * visible_column_pos.x;

        // Project
        hidden_center = (int2)((visible_column_pos.x + 0.5f) * v_to_h.x, (visible_column_pos.y + 0.5f) * v_to_h.y);

        // Bounds
        field_lower_bound = hidden_center - reverse_radii;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));

        num_hidden_columns = hidden_size.x * hidden_size.y;
        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gc = get_global_id(2);

    int visible_cell_index = gc + visible_size.z * visible_column_index;

    float sum = 0.0f;
    int count = 0;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 hidden_column_pos = (int2)(ix, iy);

            int hidden_column_index = iy + hidden_size.y * ix;

            // Project
            int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

            // Bounds check
            if (visible_column_pos.x >= visible_center.x - radius && visible_column_pos.x <= visible_center.x + radius &&
                visible_column_pos.y >= visible_center.y - radius && visible_column_pos.y <= visible_center.y + radius)
            {
                int2 offset = (int2)(visible_column_pos.x - visible_center.x + radius, visible_column_pos.y - visible_center.y + radius);

                for (int t = 0; t < hidden_size.w; t++) {
                    int slice = (target_pos + t) % target_temporal_horizon;

                    int target_hidden_state = target_hidden_states[hidden_column_index + slice * num_hidden_columns];

                    for (int c = 0; c < hidden_size.z; c++) {
                        int hidden_cell_index = t + hidden_size.w * (c + hidden_size.z * hidden_column_index);

                        int wi = gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                        sum += weights[wi] * ((c == target_hidden_state) - activations[hidden_cell_index]);
                    }

                }

                count += hidden_size.w;
            }
        }

    sum /= max(1, count);

    errors[visible_cell_index] += sum;
}
