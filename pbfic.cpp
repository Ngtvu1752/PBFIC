#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
using namespace cv;
using namespace std;

struct PBFIC_Sample{
    Vec3f color;

    Mat J_k;
    Mat W_k;
};

inline float getColorDistSq(const Vec3f& c1, const Vec3f& c2){
    float db = c1[0] - c2[0];
    float dg = c1[1] - c2[1];
    float dr = c1[2] - c2[2];
    return db*db + dg*dg + dr*dr;
}

// Gaussian weight
// exp( - ||Ip - Iq||^2 / (2 * sigma_r^2) ) e^
inline float getRangeWeight(const Vec3f& Ip, const Vec3f& Iq, float sigma_r){
    float colorDistSq = getColorDistSq(Ip, Iq);
    return exp( - colorDistSq / (2.0f * sigma_r * sigma_r) );
}

//  ADAPTIVE SAMPLING 
// Input: Original image (src), Target number of samples (target_samples)
// Output: List of selected colors (Vec3f)
vector<Vec3f> adaptiveSampling(const Mat& src, int target_samples) {
    vector<Vec3f> samples;
    
    RNG rng(time(0));
    float rs = 50.0f; 
    float decay_rate = 0.90f; 
    
    // Maximum number of attempts before deciding to reduce radius
    int max_attempts_per_radius = 200; 
    int current_attempts = 0;

    while (samples.size() < target_samples) {
        
        int y = rng.uniform(0, src.rows);
        int x = rng.uniform(0, src.cols);
        Vec3f candidate_color = src.at<Vec3f>(y, x);

        // Step 2: Check Poisson Disk condition
        // Distance to ALL selected samples must be > 2 * rs
        // Compare squared distances: distSq > (2*rs)^2
        float min_dist_threshold_sq = (2.0f * rs) * (2.0f * rs);
        
        bool accepted = true;
        for (const auto& existing_sample : samples) {
            float d_sq = getColorDistSq(candidate_color, existing_sample);
            if (d_sq <= min_dist_threshold_sq) {
                accepted = false;
                break;
            }
        }

        if (accepted) {
            samples.push_back(candidate_color);
            current_attempts = 0; 
            // cout << "Sample " << samples.size() << " found. rs=" << rs << endl;
        } else {
            current_attempts++;
        }

        // Reduce radius if stuck
        if (current_attempts > max_attempts_per_radius) {
            rs *= decay_rate;
            current_attempts = 0;
            if (rs < 1.0f) rs = 1.0f; 
        }
    }
    return samples;
}


inline Vec3d getBlockSum3(const Mat& SAT, int r0, int c0, int r1, int c1) {
    Vec3d A = SAT.at<Vec3d>(r0, c0);
    Vec3d B = SAT.at<Vec3d>(r0, c1 + 1);
    Vec3d C = SAT.at<Vec3d>(r1 + 1, c0);
    Vec3d D = SAT.at<Vec3d>(r1 + 1, c1 + 1);

    return D - B - C + A;
}

// Phiên bản overload cho ảnh 1 kênh (dùng cho W_k)
inline double getBlockSum1(const Mat& SAT, int r0, int c0, int r1, int c1) {
    double A = SAT.at<double>(r0, c0);
    double B = SAT.at<double>(r0, c1 + 1);
    double C = SAT.at<double>(r1 + 1, c0);
    double D = SAT.at<double>(r1 + 1, c1 + 1);
    return D - B - C + A;
}

// FILTERING O(1)
void computePBFICComponent(const Mat& src, PBFIC_Sample& sample, int radius, float sigma_r) {
    int rows = src.rows;
    int cols = src.cols;

    // (Intermediate Images) 
    // temp_num: (I(q) * w)
    // temp_den: (w)
    Mat temp_num(rows, cols, CV_64FC3);
    Mat temp_den(rows, cols, CV_64F);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            Vec3f pixel_val = src.at<Vec3f>(y, x);
            
            float w = getRangeWeight(pixel_val, sample.color, sigma_r);
            temp_num.at<Vec3d>(y, x) = (Vec3d)pixel_val * w; 
            temp_den.at<double>(y, x) = (double)w;
        }
    }

    // (Integral Images) 
    Mat SAT_num, SAT_den;
    
    integral(temp_num, SAT_num, CV_64F);
    integral(temp_den, SAT_den, CV_64F);

    // BOX FILTER O(1) 
    sample.J_k = Mat(rows, cols, CV_32FC3);
    sample.W_k = Mat(rows, cols, CV_32F);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int r0 = max(0, y - radius);
            int c0 = max(0, x - radius);
            int r1 = min(rows - 1, y + radius);
            int c1 = min(cols - 1, x + radius);

            Vec3d sum_num = getBlockSum3(SAT_num, r0, c0, r1, c1);
            double sum_den = getBlockSum1(SAT_den, r0, c0, r1, c1);

            sample.J_k.at<Vec3f>(y, x) = (Vec3f)sum_num;
            sample.W_k.at<float>(y, x) = (float)sum_den;
        }
    }
}

// Sort distance and keep track of original indices
struct DistIndex {
    float distSq; 
    int index; 
    
    bool operator<(const DistIndex& other) const {
        return distSq < other.distSq;
    }
};

// INTERPOLATION WITH WEIGHTING
// Input: Original image (src), List of filtered components (components)
// Output: Result image (dst)
void applyPBFICInterpolation(const Mat& src, const vector<PBFIC_Sample>& components, Mat& dst) {
    int rows = src.rows;
    int cols = src.cols;
    int K = components.size();
    
    dst = Mat::zeros(rows, cols, CV_32FC3);

    // Iterate over each pixel of the original image
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            Vec3f current_color = src.at<Vec3f>(y, x);
            
            vector<DistIndex> distances(K);
            for (int i = 0; i < K; i++) {
                distances[i].distSq = getColorDistSq(current_color, components[i].color);
                distances[i].index = i;
            }

            int top_k = min(4, K); 
            partial_sort(distances.begin(), distances.begin() + top_k, distances.end());

            Vec3f numerator_sum(0, 0, 0); 
            float denominator_sum = 0.0f;
            
            float d_min = sqrt(distances[0].distSq);
            if (d_min < 1e-5) d_min = 1e-5;

            for (int i = 0; i < top_k; i++) {
                int idx = distances[i].index;
                float d_i = sqrt(distances[i].distSq);
                
                // omega = exp( - d_i / (2 * d_min) )
                float omega = exp(-d_i / (2.0f * d_min));
                
                Vec3f J = components[idx].J_k.at<Vec3f>(y, x);
                float W = components[idx].W_k.at<float>(y, x);
                
                if (W > 1e-5) {
                    Vec3f filtered_value = J / W; // J^k / W^k
                    
                    numerator_sum += omega * filtered_value;
                    denominator_sum += omega;
                }
            }

            if (denominator_sum > 1e-5) {
                dst.at<Vec3f>(y, x) = numerator_sum / denominator_sum;
            } else {
                dst.at<Vec3f>(y, x) = current_color;
            }
        }
    }
}

// Input is an 8-bit CV_8U image
double getPSNR(const Mat& I1, const Mat& I2) {
    Mat s1;
    absdiff(I1, I2, s1);     
    s1.convertTo(s1, CV_32F); 
    s1 = s1.mul(s1); 
    Scalar s = sum(s1); 
    
    double sse = s.val[0] + s.val[1] + s.val[2]; 

    if(sse <= 1e-10) 
        return 0; 
    else {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

int main() {
    string inputWindow = "Input Image";
    string outputWindow = "Output PBFIC";
    
    Mat img_in = imread("flower.png");
    if (img_in.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Convert float (0.0 - 255.0) 
    Mat src;
    img_in.convertTo(src, CV_32FC3);

    int num_samples = 20;  
    int radius = 10;        
    float sigma_r = 30.0f; 

    cout << "--- PBFIC START ---" << endl;
    
    // ==========================================
    // PHASE 1: ADAPTIVE SAMPLING
    // ==========================================
    auto start_total = chrono::high_resolution_clock::now();
    cout << "1. Sampling colors..." << endl;
    vector<Vec3f> sample_colors = adaptiveSampling(src, num_samples);
    cout << "   Selected " << sample_colors.size() << " samples." << endl;

    vector<PBFIC_Sample> components(sample_colors.size());
    for(size_t i=0; i<sample_colors.size(); i++) {
        components[i].color = sample_colors[i];
    }

    // ==========================================
    // PHASE 2: O(1) FILTERING
    // ==========================================
    cout << "2. Filtering components (O(1))..." << endl;
    // #pragma omp parallel for
    auto start_time = chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    for (int i = 0; i < components.size(); i++) {
        computePBFICComponent(src, components[i], radius, sigma_r);
    }
    auto end_time = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> duration = end_time - start_time;
    
    cout << "   -> Filtering took: " << duration.count() << " ms" << endl;
    cout << "   -> Radius used: " << radius << endl;


    // ==========================================
    // PHASE 3: INTERPOLATION
    // ==========================================
    cout << "3. Interpolating final result..." << endl;
    Mat result;
    applyPBFICInterpolation(src, components, result);
    auto end_total = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> total_duration = end_total - start_total;
    cout << "   -> Total filtering + interpolation took: " << total_duration.count() << " ms" << endl;

    cout << "Done!" << endl;
    
    // Convert back to 8-bit for display
    Mat final_out;
    result.convertTo(final_out, CV_8UC3);

    imwrite("output_pbfic_result.jpg", final_out);
    
    // Display (if running on a machine with GUI)
    // imshow(inputWindow, img_in);
    // imshow(outputWindow, final_out);
    // waitKey(0);


    cout << "--- EXPERIMENT 5.2: QUALITY (PSNR) ---" << endl;

    Mat src_8u;
    if (src.depth() == CV_32F) {
        src.convertTo(src_8u, CV_8U);
    } else {
        src_8u = src.clone();
    }

    // 2. Create reference image using OpenCV library
    // Note: OpenCV uses Gaussian Spatial, while we use Box Spatial.
    // To be consistent, we set sigmaSpace = radius.
    Mat reference_img;
    cv::bilateralFilter(src_8u, reference_img, -1, sigma_r, radius);

    double psnr_value = getPSNR(reference_img, final_out);

    cout << "PSNR value: " << psnr_value << " dB" << endl;
    
    imwrite("reference_opencv.jpg", reference_img);
    return 0;
}