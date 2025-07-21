// ----------------------------------------------------------------------------
//  CLOgmaNeo
//  Copyright(c) 2023-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of CLOgmaNeo is licensed to you under the terms described
//  in the CLOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// --- Core SPH ---

__constant float softplus_limit = 4.0f;
__constant float byte_inv = 1.0f / 255.0f;
__constant float limit_max = 999999.0f;
__constant float limit_min = -999999.0f;
__constant float limit_small = 0.00001f;

float softplus(
    float x
) {
    float in_range = (x < softplus_limit);

    return log(1.0f + exp(x * in_range)) * in_range + x * (1.0f - in_range);
}

float sigmoid(
    float x
) {
    return tanh(x * 0.5f) * 0.5f + 0.5f;
}

__kernel void decoder_activate(
    __global const int* visible_states,
    __global const unsigned char* weights,
    __global float* dendrite_activations,
    __global float* hidden_activations,
    __global int* hidden_states,
    int3 visible_size,
    int3 hidden_size,
    int num_dendrites_per_cell,
    int radius,
    int diam,
    float2 h_to_v,
    float importance,
    uchar finish,
    float scale
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
    __local float dendrite_scale;

    __local int num_hidden_columns;
    __local int num_visible_columns;

    __local int half_num_dendrites_per_cell;

    __local float activation_scale;

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

        count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        dendrite_scale = sqrt(1.0f / count) * byte_inv;

        num_hidden_columns = hidden_size.x * hidden_size.y;
        num_visible_columns = visible_size.x * visible_size.y;

        half_num_dendrites_per_cell = num_dendrites_per_cell / 2;

        activation_scale = sqrt(1.0f / num_dendrites_per_cell) * scale;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gc = get_global_id(2);

    int hidden_cells_start = hidden_size.z * hidden_column_index;

    int hidden_cell_index = gc + hidden_cells_start;

    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

    for (int di = 0; di < num_dendrites_per_cell; di++) {
        int dendrite_index = di + dendrites_start;

        int sum = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 visible_column_pos = (int2)(ix, iy);

                int visible_column_index = iy + visible_size.y * ix;

                int2 offset = visible_column_pos - field_lower_bound;

                int visible_state = visible_states[visible_column_index];

                int wi = gc + hidden_size.z * (visible_state + visible_size.z * (offset.y + diam * (offset.x + diam * (di + num_dendrites_per_cell * hidden_column_index))));

                sum += weights[wi] - 127;
            }

        dendrite_activations[dendrite_index] += sum * dendrite_scale * importance;
    }

    if (finish) {
        float activation = 0.0f;

        for (int di = 0; di < num_dendrites_per_cell; di++) {
            int dendrite_index = di + dendrites_start;

            activation += softplus(dendrite_activations[dendrite_index]) * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f);
        }
        
        activation *= activation_scale;

        hidden_activations[hidden_cell_index] = activation;

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (get_local_id(2) == 0) {
            int max_index = 0;
            float max_activation = limit_min;

            for (int c = 0; c < hidden_size.z; c++) {
                float activation = hidden_activations[c + hidden_cells_start];

                if (activation > max_activation) {
                    max_activation = activation;
                    max_index = c;
                }
            }

            hidden_states[hidden_column_index] = max_index;

            float total_activation = 0.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                int hidden_cell_index_scan = c + hidden_cells_start;

                float activation = hidden_activations[hidden_cell_index_scan];

                activation = exp(activation - max_activation);

                hidden_activations[hidden_cell_index_scan] = activation;

                total_activation += activation;
            }

            float total_inv = 1.0f / max(limit_small, total_activation);

            for (int c = 0; c < hidden_size.z; c++)
                hidden_activations[c + hidden_cells_start] *= total_inv;
        }
    }
}

__kernel void decoder_learn(
    __global const int* visible_states,
    __global const int* target_hidden_states,
    __global const float* dendrite_activations,
    __global const float* hidden_activations,
    __global unsigned char* weights,
    int3 visible_size,
    int3 hidden_size,
    int num_dendrites_per_cell,
    int radius,
    int diam,
    float2 h_to_v,
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

        num_hidden_columns = hidden_size.x * hidden_size.y;
        num_visible_columns = visible_size.x * visible_size.y;

        half_num_dendrites_per_cell = num_dendrites_per_cell / 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gc = get_global_id(2);

    int target_state = target_hidden_states[hidden_column_index];

    int hidden_cells_start = hidden_size.z * hidden_column_index;

    int hidden_cell_index = gc + hidden_cells_start;

    int dendrites_start = num_dendrites_per_cell * hidden_cell_index;

    float hidden_delta = lr * 255.0f * ((gc == target_state) - hidden_activations[hidden_cell_index]);

    for (int di = 0; di < num_dendrites_per_cell; di++) {
        int dendrite_index = di + dendrites_start;

        int dendrite_delta = round(hidden_delta * ((di >= half_num_dendrites_per_cell) * 2.0f - 1.0f) * sigmoid(dendrite_activations[dendrite_index]));

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 visible_column_pos = (int2)(ix, iy);

                int visible_column_index = iy + visible_size.y * ix;

                int2 offset = visible_column_pos - field_lower_bound;

                int visible_state = visible_states[visible_column_index];

                int wi = gc + hidden_size.z * (visible_state + visible_size.z * (offset.y + diam * (offset.x + diam * (di + num_dendrites_per_cell * hidden_column_index))));

                weights[wi] = clamp(weights[wi] + dendrite_delta, 0, 255);
            }
    }
}

__kernel void encoder_activate(
    __global const int* visible_states,
    __global const unsigned char* weights,
    __global const int* weight_totals,
    __global const unsigned char* committed_flags,
    __global float* accums,
    __global float* counts_except,
    __global float* counts_all,
    __global float* weight_totals_all,
    __global int* hidden_states,
    __global unsigned char* learn_flags,
    __global float* comparisons,
    int3 visible_size,
    int3 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    float importance,
    uchar finish,
    float choice,
    float vigilance
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

        count = (iter_upper_bound.x - iter_lower_bound.x + 1) * (iter_upper_bound.y - iter_lower_bound.y + 1);

        num_visible_columns = visible_size.x * visible_size.y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gc = get_global_id(2);

    int hidden_cells_start = hidden_size.z * hidden_column_index;

    int hidden_cell_index = gc + hidden_cells_start;

    int sum = 0;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int visible_state = visible_states[visible_column_index];

            int wi = gc + hidden_size.z * (offset.y + diam * (offset.x + diam * (visible_state + visible_size.z * hidden_column_index)));

            sum += weights[wi];
        }

    accums[hidden_cell_index] += importance * sum * byte_inv;
    counts_except[hidden_cell_index] += importance * count * (visible_size.z - 1);
    counts_all[hidden_cell_index] += importance * count * visible_size.z;
    weight_totals_all[hidden_cell_index] += importance * weight_totals[hidden_cell_index] * byte_inv;

    if (finish) {
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (gc == 0) {
            int max_index = -1;
            float max_activation = 0.0f;

            int max_complete_index = 0;
            float max_complete_activation = 0.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                int hidden_cell_index = c + hidden_cells_start;

                float accum = accums[hidden_cell_index];
                float total_all = weight_totals_all[hidden_cell_index];
                float count_except = counts_except[hidden_cell_index];

                float complemented = accum - total_all + count_except;
                float match = complemented / count_except; 
                float activation = accum / (choice + counts_all[hidden_cell_index] - total_all);

                if ((!committed_flags[hidden_cell_index] || match >= vigilance) && activation > max_activation) {
                    max_activation = activation;
                    max_index = c;
                }

                if (activation > max_complete_activation) {
                    max_complete_activation = activation;
                    max_complete_index = c;
                }
            }

            hidden_states[hidden_column_index] = (max_index == -1 ? max_complete_index : max_index);
            learn_flags[hidden_column_index] = (max_index != -1);
            comparisons[hidden_column_index] = (max_index == -1 ? 0.0f : max_complete_activation);
        }
    }
}

__kernel void encoder_learn(
    __global const int* visible_states,
    __global const int* hidden_states,
    __global const unsigned char* learn_flags,
    __global const float* comparisons,
    __global unsigned char* weights,
    __global unsigned char* committed_flags,
    __global int* weight_totals,
    int3 visible_size,
    int3 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    float active_ratio,
    int l_radius,
    float lr
) {
    // Pre-compute
    int2 hidden_column_pos = (int2)(get_global_id(0), get_global_id(1));
    int hidden_column_index = hidden_column_pos.y + hidden_size.y * hidden_column_pos.x;

    if (!learn_flags[hidden_column_index])
        return;

    // Project
    int2 visible_center = (int2)((hidden_column_pos.x + 0.5f) * h_to_v.x, (hidden_column_pos.y + 0.5f) * h_to_v.y);

    // Bounds
    int2 field_lower_bound = visible_center - radius;
    
    int2 iter_lower_bound = (int2)(max(0, field_lower_bound.x), max(0, field_lower_bound.y));
    int2 iter_upper_bound = (int2)(min(visible_size.x - 1, visible_center.x + radius), min(visible_size.y - 1, visible_center.y + radius));

    int num_visible_columns = visible_size.x * visible_size.y;

    int hidden_cells_start = hidden_size.z * hidden_column_index;

    int hidden_state = hidden_states[hidden_column_index];

    int hidden_cell_index = hidden_state + hidden_cells_start;

    float comparison = comparisons[hidden_column_index];

    int num_higher = 0;
    int l_count = 1; // start at 1 since self is skipped

    for (int dcx = -l_radius; dcx <= l_radius; dcx++)
        for (int dcy = -l_radius; dcy <= l_radius; dcy++) {
            if (dcx == 0 && dcy == 0)
                continue;

            int2 other_hidden_column_pos = (int2)(hidden_column_pos.x + dcx, hidden_column_pos.y + dcy);

            if (other_hidden_column_pos.x >= 0 && other_hidden_column_pos.y >= 0 && other_hidden_column_pos.x < hidden_size.x && other_hidden_column_pos.y < hidden_size.y) {
                int other_hidden_column_index = other_hidden_column_pos.y + hidden_size.y * other_hidden_column_pos.x;

                if (comparisons[other_hidden_column_index] >= comparison)
                    num_higher++;

                l_count++;
            }
        }

    float ratio = (float)(num_higher) / l_count;

    if (ratio <= active_ratio) {
        int weight_total_delta = 0;

        for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
            for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                int2 visible_column_pos = (int2)(ix, iy);

                int visible_column_index = iy + visible_size.y * ix;

                int2 offset = visible_column_pos - field_lower_bound;

                int visible_state = visible_states[visible_column_index];

                int wi = hidden_state + hidden_size.z * (offset.y + diam * (offset.x + diam * (visible_state + visible_size.z * hidden_column_index)));

                unsigned char weight_old = weights[wi];

                weights[wi] = min(255, weights[wi] + (int)ceil(lr * (255.0f - weights[wi])));

                weight_total_delta += weights[wi] - weight_old;
            }

        weight_totals[hidden_cell_index] += weight_total_delta;
    }
}
