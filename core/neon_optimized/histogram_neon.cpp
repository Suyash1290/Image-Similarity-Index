#include "NEON_2_SSE.h"
#include <cstdint>
#include <cstring>
#include <cmath>

#define EXPORT_DLL __declspec(dllexport)

// Function to compute histogram of an image
static void compute_histogram(const uint8_t* data, int total_pixels, uint32_t* hist) {
    // Zero out the histogram
    memset(hist, 0, 256 * sizeof(uint32_t));
    
    int i = 0;
    
    // Process 16 pixels at a time using NEON
    for (; i <= total_pixels - 16; i += 16) {
        uint8x16_t pixels = vld1q_u8(data + i);
        
        // Extract pixel values and update histogram
        uint8_t p[16];
        vst1q_u8(p, pixels);  // Store vector to array
        
        // Unrolled loop for performance
        hist[p[0]]++; hist[p[1]]++; hist[p[2]]++; hist[p[3]]++;
        hist[p[4]]++; hist[p[5]]++; hist[p[6]]++; hist[p[7]]++;
        hist[p[8]]++; hist[p[9]]++; hist[p[10]]++; hist[p[11]]++;
        hist[p[12]]++; hist[p[13]]++; hist[p[14]]++; hist[p[15]]++;
    }
    
    // Process leftover pixels
    for (; i < total_pixels; i++) {
        hist[data[i]]++;
    }
}

// Function to calculate the cosine similarity between two histograms
static float calculate_cosine_similarity(const uint32_t* hist1, const uint32_t* hist2) {
    float dot_product = 0.0f;
    float magnitude1 = 0.0f;
    float magnitude2 = 0.0f;
    
    // NEON acceleration for computing dot product and magnitudes
    float32x4_t dot_vec = vdupq_n_f32(0);
    float32x4_t mag1_vec = vdupq_n_f32(0);
    float32x4_t mag2_vec = vdupq_n_f32(0);
    
    // Process 4 histogram bins at a time
    for (int i = 0; i < 256; i += 4) {
        // Load 4 values from each histogram
        float32x4_t h1 = vcvtq_f32_u32(vld1q_u32(&hist1[i]));
        float32x4_t h2 = vcvtq_f32_u32(vld1q_u32(&hist2[i]));
        
        // Update dot product: dot += h1 * h2
        dot_vec = vmlaq_f32(dot_vec, h1, h2);
        
        // Update magnitudes: mag1 += h1^2, mag2 += h2^2
        mag1_vec = vmlaq_f32(mag1_vec, h1, h1);
        mag2_vec = vmlaq_f32(mag2_vec, h2, h2);
    }
    
    // Horizontal sum of vectors to get final values
    float32x2_t dot_vec2 = vadd_f32(vget_low_f32(dot_vec), vget_high_f32(dot_vec));
    float32x2_t mag1_vec2 = vadd_f32(vget_low_f32(mag1_vec), vget_high_f32(mag1_vec));
    float32x2_t mag2_vec2 = vadd_f32(vget_low_f32(mag2_vec), vget_high_f32(mag2_vec));
    
    dot_product = vget_lane_f32(vpadd_f32(dot_vec2, dot_vec2), 0);
    magnitude1 = vget_lane_f32(vpadd_f32(mag1_vec2, mag1_vec2), 0);
    magnitude2 = vget_lane_f32(vpadd_f32(mag2_vec2, mag2_vec2), 0);
    
    // Compute the cosine similarity
    float magnitude = sqrt(magnitude1) * sqrt(magnitude2);
    if (magnitude < 1e-10) {
        return 0.0f; // Avoid division by zero
    }
    
    return dot_product / magnitude;
}

extern "C" {
// Function to compute cosine similarity between two images
EXPORT_DLL float neon_image_cosine_similarity(const uint8_t* image1, const uint8_t* image2, 
                                  int width, int height) {
    uint32_t hist1[256] = {0};
    uint32_t hist2[256] = {0};
    int total_pixels = width * height;
    
    // Compute histograms for both images
    compute_histogram(image1, total_pixels, hist1);
    compute_histogram(image2, total_pixels, hist2);
    
    // Calculate cosine similarity between the histograms
    return calculate_cosine_similarity(hist1, hist2);
}

// Original histogram function kept for backward compatibility
EXPORT_DLL void neon_histogram(const uint8_t* data, int width, int height, uint32_t* hist) {
    int total_pixels = width * height;
    compute_histogram(data, total_pixels, hist);
}
}