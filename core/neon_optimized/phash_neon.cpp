#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

// Fast DCT implementation using standard C++
void fast_dct_8x8(const float* input, float* output) {
    // Constants for DCT computation
    const float c1 = 0.9808f;  // cos(pi/16)
    const float c2 = 0.9239f;  // cos(pi/8)
    const float c3 = 0.8315f;  // cos(3*pi/16)
    const float c4 = 0.7071f;  // cos(pi/4)
    const float c5 = 0.5556f;  // cos(5*pi/16)
    const float c6 = 0.3827f;  // cos(3*pi/8)
    const float c7 = 0.1951f;  // cos(7*pi/16)
    
    float temp[64]; // Temporary buffer
    
    // First pass - process rows
    for (int i = 0; i < 8; i++) {
        const float* row = input + i * 8;
        float* temp_row = temp + i * 8;
        
        // Stage 1 - butterfly operations
        float a0 = row[0] + row[7];
        float a1 = row[1] + row[6];
        float a2 = row[2] + row[5];
        float a3 = row[3] + row[4];
        float a4 = row[3] - row[4];
        float a5 = row[2] - row[5];
        float a6 = row[1] - row[6];
        float a7 = row[0] - row[7];
        
        // Stage 2
        float b0 = a0 + a3;
        float b1 = a1 + a2;
        float b2 = a1 - a2;
        float b3 = a0 - a3;
        float b4 = a4;
        float b5 = a5 * c4 + a6 * c4;
        float b6 = a6 * c4 - a5 * c4;
        float b7 = a7;
        
        // Stage 3
        float c0 = b0 + b1;
        float c1 = b0 - b1;
        float c2 = b2 * c4 + b3;
        float c3 = b3 - b2 * c4;
        float c4b = b4 + b5;
        float c5b = b4 - b5;
        float c6b = b7 - b6;
        float c7b = b6 + b7;
        
        // Stage 4 - scaling
        temp_row[0] = c0 * 0.125f;
        temp_row[4] = c1 * 0.125f;
        temp_row[2] = c2 * 0.125f;
        temp_row[6] = c3 * 0.125f;
        temp_row[1] = c4b * 0.125f;
        temp_row[5] = c5b * 0.125f;
        temp_row[3] = c6b * 0.125f;
        temp_row[7] = c7b * 0.125f;
    }
    
    // Second pass - process columns
    for (int i = 0; i < 8; i++) {
        // Column data
        float a0 = temp[i] + temp[56 + i];
        float a1 = temp[8 + i] + temp[48 + i];
        float a2 = temp[16 + i] + temp[40 + i];
        float a3 = temp[24 + i] + temp[32 + i];
        float a4 = temp[24 + i] - temp[32 + i];
        float a5 = temp[16 + i] - temp[40 + i];
        float a6 = temp[8 + i] - temp[48 + i];
        float a7 = temp[i] - temp[56 + i];
        
        // Stage 2
        float b0 = a0 + a3;
        float b1 = a1 + a2;
        float b2 = a1 - a2;
        float b3 = a0 - a3;
        float b4 = a4;
        float b5 = a5 * c4 + a6 * c4;
        float b6 = a6 * c4 - a5 * c4;
        float b7 = a7;
        
        // Stage 3
        float c0 = b0 + b1;
        float c1 = b0 - b1;
        float c2 = b2 * c4 + b3;
        float c3 = b3 - b2 * c4;
        float c4b = b4 + b5;
        float c5b = b4 - b5;
        float c6b = b7 - b6;
        float c7b = b6 + b7;
        
        // Output with scaling
        output[i] = c0 * 0.125f;
        output[8 + i] = c4b * 0.125f;
        output[16 + i] = c2 * 0.125f;
        output[24 + i] = c6b * 0.125f;
        output[32 + i] = c1 * 0.125f;
        output[40 + i] = c5b * 0.125f;
        output[48 + i] = c3 * 0.125f;
        output[56 + i] = c7b * 0.125f;
    }
}

extern "C" {

__declspec(dllexport) void neon_phash(const uint8_t* input, uint8_t* output, int size) {
    // Convert 8x8 uint8 image to float
    float input_float[64];
    for (int i = 0; i < 64; i++) {
        input_float[i] = static_cast<float>(input[i]);
    }
    
    // Process DCT
    float dct_output[64];
    fast_dct_8x8(input_float, dct_output);
    
    // Use the low frequency 8x8 portion for perceptual hash
    float low_freq[64];
    memcpy(low_freq, dct_output, 64 * sizeof(float));
    
    // Calculate median using partial selection
    float sorted_copy[64];
    memcpy(sorted_copy, low_freq, 64 * sizeof(float));
    
    std::nth_element(sorted_copy, sorted_copy + 32, sorted_copy + 64);
    float median = sorted_copy[32];
    
    // Generate binary hash (64 bits = 8 bytes)
    memset(output, 0, 8);
    
    for (int i = 0; i < 64; i++) {
        if (low_freq[i] > median) {
            output[i / 8] |= (1 << (7 - (i % 8)));
        }
    }
}

// Optimized histogram function
__declspec(dllexport) void neon_histogram(const uint8_t* data, int width, int height, uint32_t* histogram) {
    // Clear histogram
    memset(histogram, 0, 256 * sizeof(uint32_t));
    
    // Process pixels in larger blocks for better cache usage
    const int total_pixels = width * height;
    const int block_size = 256; // Process 256 pixels at a time
    
    for (int offset = 0; offset < total_pixels; offset += block_size) {
        int block_end = std::min(offset + block_size, total_pixels);
        
        // Process one block
        for (int i = offset; i < block_end; i++) {
            histogram[data[i]]++;
        }
    }
}

}