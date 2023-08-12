// ----------------------------------------------------------------------------
//  CLOgmaNeo
//  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of CLOgmaNeo is licensed to you under the terms described
//  in the CLOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// --- Core SPH ---

__kernel void decoder_activate(
    __global const int* visible_states,
    __global const int* visible_states_prev,
    __global const float* visible_gates,
    __global const int* target_hidden_states,
    __global float* weights,
    __global float* activations,
    __global int* hidden_states,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    int history_pos,
    int target_pos,
    int target_temporal_horizon,
    float importance,
    uchar inhibit,
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

    __local int num_hidden_columns;
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

        num_hidden_columns = hidden_size.x * hidden_size.y;
        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gt = get_global_id(2) / hidden_size.z;
    int gc = get_global_id(2) % hidden_size.z;

    int gslice = (target_pos + gt) % target_temporal_horizon;

    int hidden_state_index = hidden_column_index + num_hidden_columns * gslice;

    int target_state = target_hidden_states[hidden_state_index];

    int hidden_cell_index = gt + hidden_size.w * (gc + hidden_size.z * hidden_column_index);

    if (lr != 0.0f) {
        float delta = lr * ((gc == target_state) - activations[hidden_cell_index]);

        for (int t = 0; t < visible_size.w; t++) {
            int slice = (history_pos + t) % visible_size.w;

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int2 visible_column_pos = (int2)(ix, iy);

                    int visible_column_index = iy + visible_size.y * ix;

                    int2 offset = visible_column_pos - field_lower_bound;

                    int visible_state_prev = visible_states_prev[visible_column_index + num_visible_columns * slice];

                    int wi = gc + hidden_size.z * (gt + hidden_size.w * (t + visible_size.w * (visible_state_prev + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index)))));

                    weights[wi] += delta * visible_gates[t + visible_size.w * visible_column_index];
                }
        }
    }

    float sum = 0.0f;

    for (int t = 0; t < visible_size.w; t++) {
        int slice = (history_pos + t) % visible_size.w;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 visible_column_pos = (int2)(ix, iy);

                int visible_column_index = iy + visible_size.y * ix;

                int2 offset = visible_column_pos - field_lower_bound;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                int wi = gc + hidden_size.z * (gt + hidden_size.w * (t + visible_size.w * (visible_state + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index)))));

                sum += weights[wi];
            }
    }

    sum /= count;

    activations[hidden_cell_index] += sum * importance;

    if (inhibit) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (gc == 0) {
            int max_index = 0;
            float max_activation = -999999.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                float activation = activations[gt + hidden_size.w * (c + hidden_size.z * hidden_column_index)];

                if (activation > max_activation) {
                    max_activation = activation;
                    max_index = c;
                }
            }

            hidden_states[hidden_column_index + gt * hidden_size.x * hidden_size.y] = max_index;

            float total_activation = 0.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                int hidden_cell_index_scan = gt + hidden_size.w * (c + hidden_size.z * hidden_column_index);

                activations[hidden_cell_index_scan] = exp(activations[hidden_cell_index_scan] - max_activation);

                total_activation += activations[hidden_cell_index_scan];
            }

            float total_inv = 1.0f / max(0.0001f, total_activation);

            for (int c = 0; c < hidden_size.z; c++) {
                int hidden_cell_index_scan = gt + hidden_size.w * (c + hidden_size.z * hidden_column_index);

                activations[hidden_cell_index_scan] *= total_inv;
            }
        }
    }
}

__kernel void encoder_activate(
    __global const int* visible_states,
    __global const float* weights,
    __global float* activations,
    __global int* hidden_states,
    __global int* hidden_usages,
    __global float* hidden_gates,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    int history_pos,
    float importance,
    uchar inhibit,
    uchar gate_update,
    float gcurve
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

    barrier(CLK_LOCAL_MEM_FENCE);

    int gc = get_global_id(2);

    int hidden_cell_index = gc + hidden_size.z * hidden_column_index;

    float sum = 0.0f;

    for (int t = 0; t < visible_size.w; t++) {
        int slice = (history_pos + t) % visible_size.w;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 visible_column_pos = (int2)(ix, iy);

                int visible_column_index = iy + visible_size.y * ix;

                int2 offset = visible_column_pos - field_lower_bound;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                int wi = gc + hidden_size.z * (t + visible_size.w * (visible_state + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index))));

                sum += weights[wi];
            }
    }

    sum /= count;

    activations[hidden_cell_index] += sum * importance;

    if (inhibit) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (gc == 0) {
            int max_index = 0;
            float max_activation = -999999.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                float activation = activations[c + hidden_size.z * hidden_column_index];

                if (activation > max_activation) {
                    max_activation = activation;
                    max_index = c;
                }
            }

            hidden_states[hidden_column_index] = max_index;

            if (gate_update) {
                int hidden_cell_index_max = max_index + hidden_size.z * hidden_column_index;

                hidden_gates[hidden_column_index] = exp(-hidden_usages[hidden_cell_index_max] * gcurve);
                hidden_usages[hidden_cell_index_max] = min(999999, hidden_usages[hidden_cell_index_max] + 1);
            }
        }
    }
}

__kernel void update_gates(
    __global const int* states,
    __global int* usages,
    __global float* gates,
    int4 size,
    float gcurve
) {
    int2 column_pos = (int2)(get_global_id(0), get_global_id(1));
    int column_index = column_pos.y + size.y * column_pos.x;

    int gt = get_global_id(2) / size.z;
    int gc = get_global_id(2) % size.z;

    int state = states[column_index];

    int temporal_column_index = gt + size.w * column_index;

    int cell_index = state + size.z * temporal_column_index;

    gates[temporal_column_index] = exp(-usages[cell_index] * gcurve);
    usages[cell_index] = min(999999, usages[cell_index] + 1);
}

__kernel void encoder_learn(
    __global const int* visible_states,
    __global const int* hidden_states,
    __global const float* hidden_gates,
    __global float* weights,
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

    barrier(CLK_LOCAL_MEM_FENCE);

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

            int hidden_state = hidden_states[hidden_column_index];

            // Project
            int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

            // Bounds check
            if (visible_column_pos.x >= visible_center.x - radius && visible_column_pos.x <= visible_center.x + radius &&
                visible_column_pos.y >= visible_center.y - radius && visible_column_pos.y <= visible_center.y + radius)
            {
                int2 offset = (int2)(visible_column_pos.x - visible_center.x + radius, visible_column_pos.y - visible_center.y + radius);

                int wi = hidden_state + hidden_size.z * (gt + visible_size.w * (gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index))));

                sum += weights[wi];
                count++;
            }
        }

    sum /= max(1, count);

    reconstruction[temporal_visible_cell_index] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(2) == 0) {
        for (int c = 0; c < visible_size.z; c++) {
            float recon = reconstruction[gt + visible_size.w * (c + visible_size.z * visible_column_index)];

            if (recon > max_activation) {
                max_activation = recon;
                max_index = c;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (max_index != target_state) {
        float delta = lr * ((gc == target_state) - exp(min(0.0f, sum - 1.0f)));

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 hidden_column_pos = (int2)(ix, iy);

                int hidden_column_index = iy + hidden_size.y * ix;

                int hidden_state = hidden_states[hidden_column_index];

                // Project
                int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

                // Bounds check
                if (visible_column_pos.x >= visible_center.x - radius && visible_column_pos.x <= visible_center.x + radius &&
                    visible_column_pos.y >= visible_center.y - radius && visible_column_pos.y <= visible_center.y + radius)
                {
                    int2 offset = (int2)(visible_column_pos.x - visible_center.x + radius, visible_column_pos.y - visible_center.y + radius);

                    int wi = hidden_state + hidden_size.z * (gt + visible_size.w * (gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index))));

                    weights[wi] += delta * hidden_gates[hidden_column_index];
                }
            }
    }
}
