#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cfloat>

// Error checking
#define GPU_CHECK(call) do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(status) << std::endl; \
        exit(1); \
    } \
} while(0)

// Timer implementation for benchmarking
#ifdef __MACH__
#include <mach/mach.h>
#include <mach/mach_time.h>
static double getTime() {
    static mach_timebase_info_data_t info;
    static double conversion = 0.0;
    if (conversion == 0.0) {
        mach_timebase_info(&info);
        conversion = (info.numer / info.denom) / 1e9;
    }
    return conversion * mach_absolute_time();
}
#else
#include <ctime>
static double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

__global__ void assignClusters(
    const float* points, const float* medoids, int* assignments, int* changed, const int N, const int K, const int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float minDist = FLT_MAX;
    int bestCluster = -1;

    for (int k = 0; k < K; k++) {
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = points[idx * D + d] - medoids[k * D + d];
            dist += diff * diff;
        }
        if (dist < minDist) {
            minDist = dist;
            bestCluster = k;
        }
    }

    if (assignments[idx] != bestCluster) {
        assignments[idx] = bestCluster;
        *changed = 1;
    }
}

__global__ void updateMedoids(
    const float* points, const int* assignments, float* medoids, const int N, const int K, const int D) {
    const int k = blockIdx.x;
    if (k >= K) return;

    __shared__ float bestDist;
    __shared__ int bestPoint;

    if (threadIdx.x == 0) {
        bestDist = FLT_MAX;
        bestPoint = -1;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (assignments[i] != k) continue;

        float totalDist = 0.0f;
        for (int j = 0; j < N; j++) {
            if (assignments[j] != k) continue;

            float dist = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = points[i * D + d] - points[j * D + d];
                dist += diff * diff;
            }
            totalDist += sqrtf(dist);
        }

        atomicMin((int*)&bestDist, __float_as_int(totalDist));
        if (__float_as_int(totalDist) == (int)bestDist) {
            bestPoint = i;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && bestPoint >= 0) {
        for (int d = 0; d < D; d++) {
            medoids[k * D + d] = points[bestPoint * D + d];
        }
    }
}

class FastKMedoids {
    int N, D, K;
    std::vector<float> points;
    std::vector<float> medoids;
    std::vector<int> assignments;

public:
    FastKMedoids(const char* filename, int k) : K(k) {
        std::ifstream fin(filename);
        fin >> N >> D;
        points.resize(N * D);
        for (float& x : points) fin >> x;
        medoids.assign(points.begin(), points.begin() + K * D);
        assignments.assign(N, -1);
    }

    void run(int threadsPerBlock) {
        float *d_points, *d_medoids;
        int *d_assignments, *d_changed;

        GPU_CHECK(cudaMalloc(&d_points, N * D * sizeof(float)));
        GPU_CHECK(cudaMalloc(&d_medoids, K * D * sizeof(float)));
        GPU_CHECK(cudaMalloc(&d_assignments, N * sizeof(int)));
        GPU_CHECK(cudaMalloc(&d_changed, sizeof(int)));

        GPU_CHECK(cudaMemcpy(d_points, points.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_medoids, medoids.data(), K * D * sizeof(float), cudaMemcpyHostToDevice));

        int pointBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        const int maxIter = 20;
        double startTime = getTime();

        for (int iter = 0; iter < maxIter; iter++) {
            int changed = 0;
            GPU_CHECK(cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice));
            assignClusters<<<pointBlocks, threadsPerBlock>>>(d_points, d_medoids, d_assignments, d_changed, N, K, D);
            GPU_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            if (!changed && iter > 0) break;
            updateMedoids<<<K, threadsPerBlock>>>(d_points, d_assignments, d_medoids, N, K, D);
        }

        double endTime = getTime();
        printf("k-medoids clustering time: %.4fs\n", endTime - startTime);

        GPU_CHECK(cudaMemcpy(assignments.data(), d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(medoids.data(), d_medoids, K * D * sizeof(float), cudaMemcpyDeviceToHost));

        // 메도이드 출력
        std::ofstream medoids_file("medoids.txt");
        for (int k = 0; k < K; k++) {
            for (int d = 0; d < D; d++) {
                medoids_file << medoids[k * D + d] << " ";
            }
            medoids_file << std::endl;
        }
        medoids_file.close();

        // 클러스터 할당 출력
        std::ofstream clusters_file("clusters.txt");
        for (int i = 0; i < N; i++) {
            clusters_file << assignments[i] << std::endl;
        }
        clusters_file.close();

        cudaFree(d_points);
        cudaFree(d_medoids);
        cudaFree(d_assignments);
        cudaFree(d_changed);
    }
};

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <K> <num_blocks> <threads_per_block>\n";
        return 1;
    }

    try {
        FastKMedoids clustering(argv[1], std::stoi(argv[2]));
        clustering.run(std::stoi(argv[4]));
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
