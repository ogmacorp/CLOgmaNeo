// ----------------------------------------------------------------------------
//  CLOgmaNeo
//  Copyright(c) 2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of CLOgmaNeo is licensed to you under the terms described
//  in the CLOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// --- Helpers ---

float sigmoid(float x) {
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
    int history_pos,
    float importance
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

    activations[hidden_cell_index] += sum * importance;
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

__kernel void encoder_accum_usages(
    __global const int* hidden_states,
    __global const unsigned char* usages,
    __global float* usage_sums,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    float importance
) {
    int2 hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
    int hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

    // Project
    int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

    // Bounds
    int2 field_lower_bound = visible_center - radius;
    
    int2 iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    int2 iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

    int count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1) * visible_size.z * visible_size.w;

    int hidden_cell_index = hidden_states[hidden_column_index] + hidden_size.z * hidden_column_index;

    int sum = 0;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            for (int c = 0; c < visible_size.z; c++) {
                for (int t = 0; t < visible_size.w; t++) {
                    int wi = t + visible_size.w * (c + wi_start);

                    sum += usages[wi];
                }
            }
        }

    usage_sums[hidden_column_index] += (sum / 255.0f) / count * importance;
}

__kernel void encoder_activate_gates(
    __global const float* usage_sums,
    __global float* gates,
    int4 hidden_size,
    float scale,
    float gcurve
) {
    int2 hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
    int hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

    gates[hidden_column_index] = exp(-gcurve * usage_sums[hidden_column_index] * scale);
}

__kernel void encoder_learn(
    __global const int* visible_states,
    __global const int* hidden_states,
    __global const float* hidden_gates,
    __global float* weights,
    __global unsigned char* usages,
    __global float* reconstruction,
    int4 visible_size,
    int3 hidden_size,
    int radius,
    int2 reverse_radii,
    int diam,
    float2 h_to_v,
    float2 v_to_h,
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
        hidden_center = (int2)((visible_column_pos.x + 0.5f) * v_to_h.x, (visible_column_pos.y + 0.5f) * v_to_h.y);

        // Bounds
        field_lower_bound = hidden_center - reverse_radii;
        
        iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
        iter_upper_bound = (int2)(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));

        max_index = 0;
        max_activation = -999999.0f;

        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gt = get_global_id(2) / visible_size.z;
    int gc = get_global_id(2) % visible_size.z;

    int gslice = (history_pos + gt) % visible_size.w;

    int target_state = visible_states[visible_column_index + num_visible_columns * gslice];

    int temporal_visible_cell_index = gt + visible_size.w * (gc + visible_size.z * visible_column_index);

    float sum = 0.0f;
    int count = 0;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 hidden_column_pos = (int2)(ix, iy);

            int hidden_column_index = iy + hidden_size.y * ix;

            int hidden_cell_index = hidden_states[hidden_column_index] + hidden_size.z * hidden_column_index;

            // Project
            int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

            // Bounds check
            if (visible_column_pos.x >= visible_center.x - radius && visible_column_pos.x <= visible_center.x + radius &&
                visible_column_pos.y >= visible_center.y - radius && visible_column_pos.y <= visible_center.y + radius)
            {
                int2 offset = (int2)(visible_column_pos.x - visible_center.x + radius, visible_column_pos.y - visible_center.y + radius);

                int wi = gt + visible_size.w * (gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index)));

                sum += weights[wi];
                count++;
            }
        }

    sum /= max(1, count);

    reconstruction[temporal_visible_cell_index] = sum;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (get_local_id(2) == 0) {
        for (int c = 0; c < visible_size.z; c++) {
            float recon = reconstruction[gt + visible_size.w * (c + visible_size.z * visible_column_index)];

            if (recon > max_activation) {
                max_activation = recon;
                max_index = c;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (max_index != target_state) {
        float delta = lr * ((gc == target_state) - exp(sum - 1.0f));

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 hidden_column_pos = (int2)(ix, iy);

                int hidden_column_index = iy + hidden_size.y * ix;

                int hidden_cell_index = hidden_states[hidden_column_index] + hidden_size.z * hidden_column_index;

                // Project
                int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

                // Bounds check
                if (visible_column_pos.x >= visible_center.x - radius && visible_column_pos.x <= visible_center.x + radius &&
                    visible_column_pos.y >= visible_center.y - radius && visible_column_pos.y <= visible_center.y + radius)
                {
                    int2 offset = (int2)(visible_column_pos.x - visible_center.x + radius, visible_column_pos.y - visible_center.y + radius);

                    int wi = gt + visible_size.w * (gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index)));

                    weights[wi] += delta * hidden_gates[hidden_column_index];
                }
            }
    }

    if (gc == target_state) {
        // usage update
        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 hidden_column_pos = (int2)(ix, iy);

                int hidden_column_index = iy + hidden_size.y * ix;

                int hidden_cell_index = hidden_states[hidden_column_index] + hidden_size.z * hidden_column_index;

                // Project
                int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

                // Bounds check
                if (visible_column_pos.x >= visible_center.x - radius && visible_column_pos.x <= visible_center.x + radius &&
                    visible_column_pos.y >= visible_center.y - radius && visible_column_pos.y <= visible_center.y + radius)
                {
                    int2 offset = (int2)(visible_column_pos.x - visible_center.x + radius, visible_column_pos.y - visible_center.y + radius);

                    int wi = gt + visible_size.w * (gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index)));

                    usages[wi] = min(255, usages[wi] + 1);
                }
            }
    }
}

__kernel void decoder_activate_gates(
    __global const int* visible_states,
    __global const unsigned char* usages,
    __global float* visible_gates,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int2 reverse_radii,
    int diam,
    float2 h_to_v,
    float2 v_to_h,
    int history_pos,
    float gcurve
) {
    int2 visible_column_pos = (int2)(get_global_id(0), get_global_id(1));
    int visible_column_index = visible_column_pos.y + visible_size.y * visible_column_pos.x;

    // Project
    int2 hidden_center = (int2)((visible_column_pos.x + 0.5f) * v_to_h.x, (visible_column_pos.y + 0.5f) * v_to_h.y);

    // Bounds
    int2 field_lower_bound = hidden_center - reverse_radii;
    
    int2 iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    int2 iter_upper_bound = (int2)(min(hidden_size.x - 1, hidden_center.x + reverse_radii.x), min(hidden_size.y - 1, hidden_center.y + reverse_radii.y));

    int num_visible_columns = visible_size.x * visible_size.y;

    int gt = get_global_id(2);

    int gslice = (history_pos + gt) % visible_size.w;

    int target_state = visible_states[visible_column_index + num_visible_columns * gslice];

    int temporal_visible_column_index = gt + visible_size.w * visible_column_index;

    int sum = 0;
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

                for (int c = 0; c < hidden_size.z; c++) {
                    int hidden_cells_start = hidden_size.w * (c + hidden_size.z * hidden_column_index);

                    for (int t = 0; t < hidden_size.w; t++) {
                        int hidden_cell_index = t + hidden_cells_start;

                        int wi = gt + visible_size.w * (target_state + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index)));

                        sum += usages[wi];
                    }
                }

                count++;
            }
        }

    visible_gates[temporal_visible_column_index] = exp(-gcurve * (sum / 255.0f) / (count * hidden_size.z * hidden_size.w));
}

__kernel void decoder_learn(
    __global const int* visible_states,
    __global const int* hidden_states,
    __global const int* target_hidden_states,
    __global const float* visible_gates,
    __global float* activations,
    __global float* weights,
    __global unsigned char* usages,
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
    __local int hidden_state;

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

        int hidden_state_index = hidden_column_index + num_hidden_columns * gslice;

        target_state = target_hidden_states[hidden_state_index];
        hidden_state = hidden_states[hidden_state_index];
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (gc == hidden_state) {
        int hidden_cell_index_max = gt + hidden_size.w * (hidden_state + hidden_size.z * hidden_column_index);

        float max_activation = activations[hidden_cell_index_max];

        float total_activation = 0.0f;

        for (int c = 0; c < hidden_size.z; c++) {
            int hidden_cell_index = gt + hidden_size.w * (c + hidden_size.z * hidden_column_index);

            activations[hidden_cell_index] = exp(activations[hidden_cell_index] - max_activation);

            total_activation += activations[hidden_cell_index];
        }

        float scale = 1.0f / max(0.0001f, total_activation);

        for (int c = 0; c < hidden_size.z; c++) {
            int hidden_cell_index = gt + hidden_size.w * (c + hidden_size.z * hidden_column_index);

            activations[hidden_cell_index] *= scale;
        }
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

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos + t) % visible_size.w;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                int wi = t + visible_size.w * (visible_state + wi_start);

                weights[wi] += delta * visible_gates[t + visible_size.w * visible_column_index];
            }
        }

    if (gc == target_state) {
        // update usages
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

                    usages[wi] = min(255, usages[wi] + 1);
                }
            }
    }
}
