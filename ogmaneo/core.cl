// ----------------------------------------------------------------------------
//  CLOgmaNeo
//  Copyright(c) 2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of CLOgmaNeo is licensed to you under the terms described
//  in the CLOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// --- Helpers ---

__inline float sigmoid(float x) {
    return tanh(x * 0.5f) * 0.5f + 0.5f;
}

// --- Core SPH ---

__kernel void accum_activations(
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

            int wi_start = visible_size.w * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos + t) % visible_size.w;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                float visible_value = (float)visible_state / (float)(visible_size.z - 1);

                int wi = t + wi_start;

                float delta = visible_value - weights[wi];

                sum -= delta * delta;
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

__kernel void inhibit_activations_second_stage(
    __global const int* states,
    __global const float* activations,
    __global char* peaks,
    int4 size,
    int radius
) {
    int2 column_pos = (int2)(get_global_id(0), get_global_id(1));
    int column_index = column_pos.y + size.y * column_pos.x;

    int gt = get_global_id(2);

    int state = states[column_index + gt * size.x * size.y];
    int cell_index = gt + size.w * (state + size.z * column_index);

    float max_activation = activations[cell_index];

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 other_column_pos = (int2)(column_pos.x + dx, column_pos.y + dy);

            if (other_column_pos.x < 0 || other_column_pos.x >= size.x || other_column_pos.y < 0 || other_column_pos.y >= size.y)
                continue;

            int other_column_index = other_column_pos.y + size.y * other_column_pos.x;

            int other_state = states[other_column_index + gt * size.x * size.y];

            int other_cell_index = gt + size.w * (other_state + size.z * other_column_index);

            float other_max_activation = activations[other_cell_index];
            
            if (other_max_activation > max_activation) {
                peaks[column_index + gt * size.x * size.y] = 0;

                return;
            }
        }

    peaks[column_index + gt * size.x * size.y] = 1;
}

__kernel void encoder_learn(
    __global const int* visible_states,
    __global const int* hidden_states,
    __global const char* hidden_peaks,
    __global const float* hidden_rates,
    __global float* weights,
    int4 visible_size,
    int3 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
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

    __local int count;

    __local int num_visible_columns;

    __local int hidden_state;

    // Pre-compute for work group
    if (get_local_id(2) == 0) {
        hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
        hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

        // Second inhibition stage
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int2 other_column_pos = (int2)(column_pos.x + dx, column_pos.y + dy);

                if (other_column_pos.x < 0 || other_column_pos.x >= size.x || other_column_pos.y < 0 || other_column_pos.y >= size.y)
                    continue;

                int other_column_index = other_column_pos.y + size.y * other_column_pos.x;

                int other_peak = peaks[other_column_index + gt * size.x * size.y];

                if (other_peak)
                    goto learn;
            }

        return;

learn:
        // Project
        visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

        // Bounds
        field_lower_bound = visible_center - radius;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

        count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * visible_size.w;

        num_visible_columns = visible_size.x * visible_size.y;

        hidden_state = hidden_states[hidden_column_index + num_hidden_columns * gslice];
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gt = get_global_id(2) / 3;
    int dc = get_global_id(2) % 3 - 1;

    int gc = hidden_state + dc;

    if (gc < 0 || gc >= hidden_size.z)
        return;

    int hidden_cell_index = gt + hidden_size.w * (gc + hidden_size.z * hidden_column_index);

    float hidden_rate = hidden_rates[hidden_cell_index];

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.w * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos + t) % visible_size.w;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                float visible_value = (float)visible_state / (float)(visible_size.z - 1);

                int wi = t + wi_start;

                float delta = visible_value - weights[wi];

                weights[wi] += hidden_rate * delta;
            }
        }

    hidden_rates[hidden_cell_index] -= lr * hidden_rate;
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
    float2 h_to_v,
    int history_pos,
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

        // Update weights upwards
        int hidden_cell_index = gt + hidden_size.w * (target_state + hidden_size.z * hidden_column_index);

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

                    weights[wi] += lr * (1.0f - weights[wi]);
                }
            }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int hidden_cell_index = gt + hidden_size.w * (gc + hidden_size.z * hidden_column_index);

    // Determine lowest weight
    float min_weight = 1.0f;

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

                min_weight = min(min_weight, weights[wi]);
            }
        }

    // Shift
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

                weights[wi] -= min_weight;
            }
        }
}
