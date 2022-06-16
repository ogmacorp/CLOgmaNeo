// ----------------------------------------------------------------------------
//  CLOgmaNeo
//  Copyright(c) 2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of CLOgmaNeo is licensed to you under the terms described
//  in the CLOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// --- Weight lookup ---

__constant int weight_lookup_table_resolution = 64;
__constant float weight_lookup_table_scale = weight_lookup_table_resolution - 1;

__constant float weight_lookup_table[weight_lookup_table_resolution + 1] = {
    -4.158883, -3.465736, -3.060271, -2.772589, -2.549445, -2.367124, -2.212973, -2.079442, 
    -1.961659, -1.856298, -1.760988, -1.673976, -1.593934, -1.519826, -1.450833, -1.386294, 
    -1.325670, -1.268511, -1.214444, -1.163151, -1.114361, -1.067841, -1.023389, -0.980829, 
    -0.940007, -0.900787, -0.863046, -0.826679, -0.791587, -0.757686, -0.724896, -0.693147, 
    -0.662376, -0.632523, -0.603535, -0.575364, -0.547965, -0.521297, -0.495321, -0.470004, 
    -0.445311, -0.421213, -0.397683, -0.374693, -0.352221, -0.330242, -0.308735, -0.287682, 
    -0.267063, -0.246860, -0.227057, -0.207639, -0.188591, -0.169899, -0.151550, -0.133531, 
    -0.115832, -0.098440, -0.081346, -0.064539, -0.048009, -0.031749, -0.015748, 0.000000,
    0.000000 // Dummy
};

// --- Helpers ---

__inline float sigmoid(float x) {
    return tanh(x * 0.5f) * 0.5f + 0.5f;
}

__inline float weight_lookup(float w) {
    w *= weight_lookup_table_scale;

    int index = (int)w;
    float interp = w - index;

    return weight_lookup_table[index] * (1.0f - interp) + weight_lookup_table[index + 1] * interp;
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

__kernel void accum_dendritic_activations(
    __global const int* visible_states,
    __global const float* weights,
    __global float* activations,
    int4 visible_size,
    int4 hidden_size,
    int num_dendrites,
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

    __local int num_dendrites_per_column;

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
        
        num_dendrites_per_column = hidden_size.z * num_dendrites;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int gt = get_global_id(2) / num_dendrites_per_column;
    int gd = get_global_id(2) % num_dendrites_per_column;

    int hidden_dendrite_index = gt + hidden_size.w * (gd + num_dendrites_per_column * hidden_column_index);

    float sum = 0.0f;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.z * (offset.y + diam * (offset.x + diam * hidden_dendrite_index));

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos + t) % visible_size.w;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                int wi = t + visible_size.w * (visible_state + wi_start);

                sum += weight_lookup(weights[wi]);
            }
        }

    sum /= count;

    activations[hidden_dendrite_index] += sum;
}

__kernel void inhibit_dendritic_activations(
    __global float* activations,
    __global int* states,
    int4 size,
    int num_dendrites,
    float scale
) {
    int2 column_pos = (int2)(get_global_id(0), get_global_id(1));
    int column_index = column_pos.y + size.y * column_pos.x;

    int num_dendrites_per_column = size.z * num_dendrites;

    int gt = get_global_id(2);

    int max_index = 0;
    float max_activation = -999999.0f;

    for (int c = 0; c < num_dendrites_per_column; c++) {
        int dendrite_index = gt + size.w * (c + num_dendrites_per_column * column_index);

        activations[dendrite_index] *= scale;

        float activation = activations[dendrite_index];

        if (activation > max_activation) {
            max_activation = activation;
            max_index = c;
        }
    }

    states[column_index + gt * size.x * size.y] = max_index / num_dendrites;
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
        float delta = lr * ((gc == target_state) - exp(sum));

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
    int num_dendrites,
    int radius,
    int diam,
    float2 h_to_v,
    int history_pos,
    int target_pos,
    int target_temporal_horizon,
    float lr,
    float boost
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

    __local int max_dendrite_index;
    __local float max_dendrite_activation;

    int gt = get_global_id(2) / num_dendrites;
    int gdi = get_global_id(2) % num_dendrites;

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

        max_dendrite_index = 0;
        max_dendrite_activation = -999999.0f;

        for (int di = 0; di < num_dendrites; di++) {
            int hidden_dendrite_index = gt + hidden_size.w * (di + num_dendrites * (target_state + hidden_size.z * hidden_column_index));

            float activation = activations[hidden_dendrite_index];

            if (activation > max_dendrite_activation) {
                max_dendrite_activation = activation;
                max_dendrite_index = di;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    int hidden_dendrite_index = gt + hidden_size.w * (gdi + num_dendrites * (target_state + hidden_size.z * hidden_column_index));

    float rate = (gdi == max_dendrite_index ? lr : boost);

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int wi_start = visible_size.z * (offset.y + diam * (offset.x + diam * hidden_dendrite_index));

            for (int t = 0; t < visible_size.w; t++) {
                int slice = (history_pos + t) % visible_size.w;

                int visible_state = visible_states[visible_column_index + num_visible_columns * slice];

                for (int c = 0; c < visible_size.z; c++) {
                    int wi = t + visible_size.w * (c + wi_start);

                    weights[wi] += rate * ((float)(c == visible_state) - weights[wi]);
                }
            }
        }
}
