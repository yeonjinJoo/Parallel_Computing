#include <iostream>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <omp.h>

#ifdef __MACH__
    #include <mach/mach.h>
    #include <mach/mach_time.h>
#endif

using namespace std;

int K, N, D, threadNum;
int* clusters, * medoidsIndex;
double** points;

int maxIter = 20;

double euclidean_distance(int firstIndex, int secondIndex){
    double sum = 0;
    // iterate for dimension
    for(int i = 0; i < D; i++){
        sum += pow(points[firstIndex][i] - points[secondIndex][i], 2);
    }
    return sqrt(sum);
}

void assign_medoids(int pointIndex){
    int minCluster = 0;
    double minDistance = __DBL_MAX__;
    double distance = 0;

    for(int i = 0; i < K; i++){
        distance = euclidean_distance(pointIndex, medoidsIndex[i]);
        if( distance < minDistance){
            minDistance = distance;
            minCluster = i;
        }
    }
    // update clusters info
    clusters[pointIndex] = minCluster;
}

int update_medoids(int z){
    int newMedoidIdx = -1;
    double minDistance = __DBL_MAX__;
    double total;

    for(int i = 0; i < N; i++){
        if(clusters[i] == z){
            total = 0;
            for(int j = 0; j < N; j++){
                if(clusters[j] == z){
                    total += euclidean_distance(i, j);
                }
            }
            if(total < minDistance){
                minDistance = total;
                newMedoidIdx = i;
            }
        }
    }

    // Check if medoid has changed
    if(medoidsIndex[z] != newMedoidIdx){
        medoidsIndex[z] = newMedoidIdx;
        return 1;  // Indicate that the medoid has changed
    }
    return 0;
}

/**
 * @brief Return the number of seconds since an unspecified time (e.g., Unix
 *        epoch). This is accomplished with a high-resolution monotonic timer,
 *        suitable for performance timing.
 *
 * @return The number of seconds.
 */
static inline double monotonic_seconds() {
#ifdef __MACH__
  // macOS 시스템에서는 mach_absolute_time을 사용
    static mach_timebase_info_data_t info;
    static double seconds_per_unit = 0.0;

    if (seconds_per_unit == 0.0) {
        mach_timebase_info(&info);
        seconds_per_unit = (info.numer / info.denom) / 1e9;
    }
    return seconds_per_unit * mach_absolute_time();
#else
    // Linux 시스템에서는 clock_gettime을 사용
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

int main(int argc, char* argv[]) {
    string inputFilePath = argv[1];
    K = atoi(argv[2]);
    threadNum = atoi(argv[3]);

    medoidsIndex = new int[K];

    // initialize medoids
    for(int i = 0; i < K; i++){
        medoidsIndex[i] = i;
    }

    // open the input file
    ifstream input_file(inputFilePath);
    if (!input_file.is_open()) {
        cerr << "can not open the file." << endl;
        return 1;
    }

    input_file >> N;
    input_file >> D;

    // points 값 넣기
    points = new double*[N];
    clusters = new int[N];
    for(int i = 0; i < N; i++){
        points[i] = new double[D];
        for(int j = 0; j < D; j++){
            input_file >> points[i][j];
        }
    }

    int isChangedGlobally = 0;

    // 타이머 사용 예시
    double timeStart = monotonic_seconds();

    for(int iter = 0; iter < maxIter; iter++){
        isChangedGlobally = 0;

        // 1. assign medoids to every points (parallel)
        #pragma omp parallel for num_threads(threadNum)
        for(int i = 0; i < N; i++){
            assign_medoids(i);
        }

        // 2. find new medoids (parallel)
        #pragma omp parallel for num_threads(threadNum) reduction(|:isChangedGlobally)
        for(int z = 0; z < K; z++){
            isChangedGlobally |= update_medoids(z);
        }

        // 3. Check medoids is changed or not
        if(isChangedGlobally == 0){
            break;
        }
    }

    double timeEnd = monotonic_seconds();

    ofstream clusterFile;
    ofstream medoidsFile;

    clusterFile.open("clusters.txt");
    medoidsFile.open("medoids.txt");

    if(clusterFile.is_open()){
        for(int i = 0; i < N; i++){
            clusterFile << clusters[i] << "\n";
        }
    }

    if(medoidsFile.is_open()){
        for(int i = 0; i < K; i++){
            for(int j = 0; j < D; j++){
                medoidsFile << points[medoidsIndex[i]][j] << " ";
            }
            medoidsFile << "\n";
        }
    }

    cout << "Elapsed time: " << timeEnd - timeStart << " seconds." << endl;

    return 0;
}
