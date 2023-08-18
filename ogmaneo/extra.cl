// ----------------------------------------------------------------------------
//  CLOgmaNeo
//  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of CLOgmaNeo is licensed to you under the terms described
//  in the CLOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// --- Image Encoder ---

__kernel void image_enc_activate(
    __global const float* visible_states,
    __global float* protos,
    __global float* activations,
    __global int* hidden_states,
    __global float* hidden_rates,
    int3 visible_size,
    int3 hidden_size,
    int radius,
    int diam,
    float2 h_to_v,
    uchar inhibit,
    float lr,
    float falloff
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
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gc = get_global_id(2);

    int hidden_cell_index = gc + hidden_size.z * hidden_column_index;

    float sum = 0.0f;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int2 visible_column_pos = (int2)(ix, iy);

            int visible_column_index = iy + visible_size.y * ix;

            int2 offset = visible_column_pos - field_lower_bound;

            int visible_states_start = visible_size.z * visible_column_index;

            for (int c = 0; c < visible_size.z; c++) {
                int wi = gc + hidden_size.z * (c + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index)));

                float w = protos[wi];

                float delta = visible_states[c + visible_states_start] - protos[wi];

                sum -= delta * delta;
            }
        }

    activations[hidden_cell_index] += sum;

    if (inhibit) {
        barrier(CLK_GLOBAL_MEM_FENCE);

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

            float total_activation = 0.0f;

            for (int c = 0; c < hidden_size.z; c++) {
                int hidden_cell_index_scan = c + hidden_size.z * hidden_column_index;

                activations[hidden_cell_index_scan] = exp(activations[hidden_cell_index_scan] - max_activation);

                total_activation += activations[hidden_cell_index_scan];
            }

            float total_inv = 1.0f / max(0.0001f, total_activation);

            for (int c = 0; c < hidden_size.z; c++) {
                int hidden_cell_index_scan = c + hidden_size.z * hidden_column_index;

                activations[hidden_cell_index_scan] *= total_inv;
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lr != 0.0f) {
            float diff = gc - hidden_states[hidden_column_index];

            float strength = exp(-falloff * diff * diff / max(0.0001f, hidden_rates[hidden_cell_index])) * hidden_rates[hidden_cell_index];

            for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
                for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
                    int2 visible_column_pos = (int2)(ix, iy);

                    int visible_column_index = iy + visible_size.y * ix;

                    int2 offset = visible_column_pos - field_lower_bound;

                    int visible_states_start = visible_size.z * visible_column_index;

                    for (int c = 0; c < visible_size.z; c++) {
                        int wi = gc + hidden_size.z * (c + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_column_index)));

                        protos[wi] += strength * (visible_states[c + visible_states_start] - protos[wi]);
                    }
                }

            hidden_rates[hidden_cell_index] -= lr * strength;
        }
    }
}

__kernel void image_enc_learn_weights(
    __global const float* visible_states,
    __global const int* hidden_states,
    __global float* weights,
    int3 visible_size,
    int3 hidden_size,
    int radius,
    int2 reverse_radii,
    int diam,
    float2 h_to_v,
    float2 v_to_h,
    float rr
) {
    __local int2 visible_column_pos;
    __local int visible_column_index;

    // Project
    __local int2 hidden_center;

    // Bounds
    __local int2 field_lower_bound;
    
    __local int2 iter_lower_bound;
    __local int2 iter_upper_bound;

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
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gc = get_global_id(2);

    int visible_cell_index = gc + visible_size.z * visible_column_index;

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

                int wi = gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                sum += weights[wi];
                count++;
            }
        }

    sum /= max(1, count);

    float delta = rr * (visible_states[visible_cell_index] - sum);

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

                int wi = gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                weights[wi] += delta;
            }
        }
}

__kernel void image_enc_reconstruct(
    __global const int* hidden_states,
    __global const float* weights,
    __global float* reconstruction,
    int3 visible_size,
    int3 hidden_size,
    int radius,
    int2 reverse_radii,
    int diam,
    float2 h_to_v,
    float2 v_to_h
) {
    __local int2 visible_column_pos;
    __local int visible_column_index;

    // Project
    __local int2 hidden_center;

    // Bounds
    __local int2 field_lower_bound;
    
    __local int2 iter_lower_bound;
    __local int2 iter_upper_bound;

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
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gc = get_global_id(2);

    int visible_cell_index = gc + visible_size.z * visible_column_index;

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

                int wi = gc + visible_size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

                sum += weights[wi];
                count++;
            }
        }

    sum /= max(1, count);

    reconstruction[visible_cell_index] = sum;
}

