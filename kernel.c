#ifndef CLK_LOCAL_MEM_FENCE
#define kernel
#define global
#endif

// cap is number of bits in digit
void kernel simple_add(global int* input, global int* output, const int length, const int cap, const int bits) {
    global int* temp;

    // global index is digit which this thread storing

    const int global_index = get_global_id(0);
    const int numOfThreads = 1 << cap;
    const int maxDigit = numOfThreads;
    const int localLength = length / numOfThreads;
    const int localStart = localLength * global_index;
    const int localEnd = global_index == numOfThreads - 1 ?
                         length : localLength + localStart;
    // we going from least significant digit position to most
    for (int binPos = 0; binPos <= bits; binPos += cap) {
        int offset = 0;
        // get offset
        for (int i = 0; i < length; i++) {
            const int n = input[i] >> binPos;
            const int digit = n % maxDigit;
            if (digit < global_index) {
                offset++;
            }
        }

        // fill output array
        int localPos = 0;
        for (int i = 0; i < length; i++) {
            const int n = input[i] >> binPos;
            const int digit = n % maxDigit;
            if (digit == global_index) {
                output[offset + localPos] = input[i];
                localPos++;
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        if (global_index == 0) {
            temp = input;
            input = output;
            output = temp;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    if (global_index == 0) {
        for (int i = 0; i < length; ++i) {
            input[i] = output[i];
        }
    }
}