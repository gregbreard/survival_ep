
// Simple kernel code that adds vector inputs

/**
 * Takes two vectors as input and adds them.
 */
kernel void vectorAdd(global float* input1, global float* input2, global float* output) {
    size_t i = get_global_id(0);
    output[i] = input1[i] + input2[i];
} // end vectorAdd