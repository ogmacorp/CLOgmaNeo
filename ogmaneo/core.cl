// ----------------------------------------------------------------------------
//  CLOgmaNeo
//  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of CLOgmaNeo is licensed to you under the terms described
//  in the CLOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// --- Core SPH ---

__kernel void assign_slice(
    __global const int* src,
    __global int* dst,
    int dst_offset
) {
    int index = get_global_id(0);

    dst[dst_offset + index] = src[index]; 
}

__kernel void stack_slices(
    __global const int* src1,
    __global const int* src2,
    __global int* dst,
    int stride,
    int offset
) {
    int index = get_global_id(0);

    dst[index] = src1[index]; 
    dst[stride + index] = src2[offset + index]; 
}

__kernel void decoder_activate(
    __global const int* visible_states,
    __global const int* visible_states_prev,
    __global const int* target_hidden_states,
    __global const float* dendrite_activations_prev,
    __global const float* hidden_activations_prev,
    __global float* weights,
    __global float* dendrite_activations,
    __global float* hidden_activations,
    __global int* hidden_states,
    int4 visible_size,
    int4 hidden_size,
    int num_dendrites_per_cell,
    int radius,
    int diam,
    float2 h_to_v,
    int history_pos,
    int history_pos_prev,
    int target_pos,
    int target_temporal_horizon,
    float importance,
    uchar finish,
    float lr,
    float leak
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

    __local int half_num_dendrites_per_cell;

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

        half_num_dendrites_per_cell = num_dendrites_per_cell / 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gt = get_global_id(2) / hidden_size.z;
    int gc = get_global_id(2) % hidden_size.z;

    int gslice = (target_pos + gt) % target_temporal_horizon;

    int target_state = target_hidden_states[hidden_column_index + num_hidden_columns * gslice];

    int hidden_cells_start = hidden_size.z * (gt + hidden_size.w * hidden_column_index);

    int hidden_cell_index = gc + hidden_cells_start;

    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

    if (lr != 0.0f) {
        float hidden_delta = lr * ((gc == target_state) - hidden_activations_prev[hidden_cell_index]);

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            float dendrite_delta = hidden_delta * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f) * ((dendrite_activations_prev[dendrite_index] > 0.0f) * (1.0f - leak) + leak);

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos_prev + t) % visible_size.w;
                int visible_columns_start = num_visible_columns * slice;

                for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                    for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                        int2 visible_column_pos = (int2)(ix, iy);

                        int visible_column_index = iy + visible_size.y * ix;

                        int2 offset = visible_column_pos - field_lower_bound;

                        int visible_state_prev = visible_states_prev[visible_column_index + visible_columns_start];

                        int wi = gc + hidden_size.z * (gt + hidden_size.w * (visible_state_prev + visible_size.z * (t + visible_size.w * (offset.y + diam * (offset.x + diam * (di + num_dendrites_per_cell * hidden_column_index))))));

                        weights[wi] += dendrite_delta;
                    }
            }
        }
    }

    for (int di = 0; di < num_dendrites_per_cell; di++) {
        int dendrite_index = di + dendrites_start;

        float sum = 0.0f;

        for (int t = 0; t < visible_size.w; t++) {
            int slice = (history_pos + t) % visible_size.w;
            int visible_columns_start = num_visible_columns * slice;

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int2 visible_column_pos = (int2)(ix, iy);

                    int visible_column_index = iy + visible_size.y * ix;

                    int2 offset = visible_column_pos - field_lower_bound;

                    int visible_state = visible_states[visible_column_index + visible_columns_start];

                    int wi = gc + hidden_size.z * (gt + hidden_size.w * (visible_state + visible_size.z * (t + visible_size.w * (offset.y + diam * (offset.x + diam * (di + num_dendrites_per_cell * hidden_column_index))))));

                    sum += weights[wi];
                }
        }

        sum *= sqrt(1.0f / count);

        dendrite_activations[dendrite_index] += sum * importance;
    }

    if (finish) {
        float activation = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            dendrite_activations[dendrite_index] = max(dendrite_activations[dendrite_index], dendrite_activations[dendrite_index] * leak);

            activation += dendrite_activations[dendrite_index] * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
        }
        
        activation *= sqrt(1.0f / num_dendrites_per_cell);

        hidden_activations[hidden_cell_index] = activation;

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (gc == 0) {
            int max_index = 0;
            float max_activation = -999999.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                float activation = hidden_activations[c + hidden_cells_start];

                if (activation > max_activation) {
                    max_activation = activation;
                    max_index = c;
                }
            }

            hidden_states[hidden_column_index + gt * num_hidden_columns] = max_index;

            float total_activation = 0.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                int hidden_cell_index_scan = c + hidden_cells_start;

                float activation = hidden_activations[hidden_cell_index_scan];

                activation = exp(activation - max_activation);

                hidden_activations[hidden_cell_index_scan] = activation;

                total_activation += activation;
            }

            float total_inv = 1.0f / max(0.0001f, total_activation);

            for (int c = 0; c < hidden_size.z; c++)
                hidden_activations[c + hidden_cells_start] *= total_inv;
        }
    }
}

__kernel void encoder_activate(
    __global const int* visible_states,
    __global const float* weights,
    __global float* activations,
    __global int* hidden_states,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    int history_pos,
    float importance,
    uchar finish
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

    int hidden_cells_start = hidden_size.z * hidden_column_index;

    int hidden_cell_index = gc + hidden_cells_start;

    float sum = 0.0f;

    for (int t = 0; t < visible_size.w; t++) {
        int slice = (history_pos + t) % visible_size.w;
        int visible_columns_start = num_visible_columns * slice;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 visible_column_pos = (int2)(ix, iy);

                int visible_column_index = iy + visible_size.y * ix;

                int2 offset = visible_column_pos - field_lower_bound;

                int visible_state = visible_states[visible_column_index + visible_columns_start];

                int wi = visible_state + visible_size.z * (t + visible_size.w * (offset.y + diam * (offset.x + diam * (gc + hidden_size.z * hidden_column_index))));

                sum += weights[wi];
            }
    }

    sum *= sqrt(1.0f / count);

    activations[hidden_cell_index] += sum * importance;

    if (finish) {
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (gc == 0) {
            int max_index = 0;
            float max_activation = -999999.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                float activation = activations[c + hidden_cells_start];

                if (activation > max_activation) {
                    max_activation = activation;
                    max_index = c;
                }
            }

            hidden_states[hidden_column_index] = max_index;
        }
    }
}

__kernel void encoder_learn(
    __global const int* visible_states,
    __global const int* hidden_states,
    __global float* weights,
    __global float* reconstruction,
    int4 visible_size,
    int4 hidden_size,
    int radius,
    int2 reverse_radii,
    int diam,
    float2 h_to_v,
    float2 v_to_h,
    int history_pos,
    float lr,
    int early_stop_cells
) {
    __local int2 visible_column_pos;
    __local int visible_column_index;

    // Project
    __local int2 hidden_center;

    // Bounds
    __local int2 field_lower_bound;
    
    __local int2 iter_lower_bound;
    __local int2 iter_upper_bound;

    __local int num_visible_columns;

    __local int num_higher;

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

        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gt = get_global_id(2) / visible_size.z;
    int gc = get_global_id(2) % visible_size.z;

    int gslice = (history_pos + gt) % visible_size.w;

    int target_state = visible_states[visible_column_index + num_visible_columns * gslice];

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

                int wi = gc + visible_size.z * (gt + visible_size.w * (offset.y + diam * (offset.x + diam * (hidden_state + hidden_size.z * hidden_column_index))));

                sum += weights[wi];
                count++;
            }
        }

    sum /= max(1, count);

    int visible_cells_start = visible_size.z * (gt + visible_size.w * visible_column_index);

    reconstruction[gc + visible_cells_start] = sum;

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (get_local_id(2) == 0) {
        num_higher = 0;
        float target_recon = reconstruction[target_state + visible_cells_start];

        for (int c = 0; c < visible_size.z; c++) {
            float recon = reconstruction[c + visible_cells_start];

            if (recon > target_recon)
                num_higher++;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (num_higher >= early_stop_cells) {
        float delta = lr * ((gc == target_state) - exp(sum - 1.0f));

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

                    int wi = gc + visible_size.z * (gt + visible_size.w * (offset.y + diam * (offset.x + diam * (hidden_state + hidden_size.z * hidden_column_index))));

                    weights[wi] += delta;
                }
            }
    }
}
