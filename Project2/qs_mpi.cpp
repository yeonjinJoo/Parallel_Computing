#include <iostream>
#include <ctime>  
#include <iostream>
#include <fstream>
#include <string>
#include <pthread.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <random>
#include <fstream>
#include <mpi.h>
#ifdef __MACH__
    #include <mach/mach.h>
    #include <mach/mach_time.h>
#endif

using namespace std;


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

static void print_time(double const seconds){
    printf("Sort Time: %0.04fs\n", seconds);
}

void generate_numbers(int rank, int chunkSize, vector<int> & numbers){
    random_device rd;
    mt19937 gen(rd() + rank);
    uniform_int_distribution<> dist(0, INT32_MAX);
    for(int i = 0; i < chunkSize; i++){
        numbers.push_back(dist(gen));
    }
}

int random_select(int rank, vector<int> & numbers){
    srand(static_cast<unsigned int>(time(0))); // 시드 초기화
    int index = rand() % numbers.size();
    return numbers[index];
}

int median_select(vector<int> & pivots){
    sort(pivots.begin(), pivots.end());
    
    int mid = pivots.size() / 2;
    return pivots[mid];
}

   // p_num과 group, numbers는 호출할 때마다 계속 변경 필요

void parallel_quicksort(int rank, vector<int> & numbers, int p_num, MPI_Comm group){
    // 배정된 process 하나일 땐 이렇게. break point.
    if(p_num == 1 || numbers.size() == 0){
        if(numbers.size()!= 0){
            sort(numbers.begin(), numbers.end());
        }
        return;
    }

    // 1. local pivot 고르고 group끼리 전체 pivot 주고받기   // p_num도 첨엔 전체지만 계속 바뀜. 줄어들것
    int localPivot = random_select(rank, numbers);

    vector<int> pivots(p_num); // group 내 피봇들
    MPI_Allgather(&localPivot, 1, MPI_INT, pivots.data(), 1, MPI_INT, group); // 계속 2그룹씩으로 나누니까. group끼리 데이터 주고받기

    int globalPivot = median_select(pivots);

    
    // 2. 데이터 피봇 기준으로 분할
    vector<int> less, greater;
    for(int num : numbers){
        if(num <= globalPivot){
            less.push_back(num);
        }
        else{
            greater.push_back(num);
        }
    }

    // 3. 프로세스 2그룹으로 분리
    int half = p_num / 2;
    MPI_Comm new_group;
    if(rank < half){ // less 담당
        MPI_Comm_split(group, 0, rank, &new_group);

        // 1) greater을 상위로 보내기
        int greaterSize = greater.size();
        MPI_Send(&greaterSize, 1, MPI_INT, rank + half, 0, group);
        MPI_Send(greater.data(), greaterSize, MPI_INT, rank + half, 1, group);

        // 2) less를 상위에서 받기 _ 근데 odd인데 half 바로 전인 경우 2번받아야함
        int incomingSize;
        int extraReceive = (p_num % 2 == 1 && rank == (half - 1)) ? 2 : 1;
        numbers = less;

        for(int i = 0; i < extraReceive; i++){
            MPI_Recv(&incomingSize, 1, MPI_INT, rank + half + i, 0, group, MPI_STATUS_IGNORE);
            vector<int> incomingData(incomingSize);
            MPI_Recv(incomingData.data(), incomingSize, MPI_INT, rank + half + i, 1, group, MPI_STATUS_IGNORE);

            // 3) 원래 값이랑 합치기
            numbers.insert(numbers.end(), incomingData.begin(), incomingData.end());
        }
        
        // 4) 재귀
        int new_rank;
        MPI_Comm_rank(new_group, &new_rank);
        parallel_quicksort(new_rank, numbers, half, new_group);
    }
    else{ // greater 담당
        MPI_Comm_split(group, 1, rank, &new_group);

        // 1) greater를 하위에서 받기
        int incomingSize;
        numbers = greater;
        int receive = (p_num % 2 == 1 && rank == (p_num - 1)) ? 0 : 1;
        if(receive == 1){
            MPI_Recv(&incomingSize, 1, MPI_INT, rank - half, 0, group, MPI_STATUS_IGNORE);
            vector<int> incomingData(incomingSize);
            MPI_Recv(incomingData.data(), incomingSize, MPI_INT, rank - half, 1, group, MPI_STATUS_IGNORE);

            // 2) 원래 값이랑 합치기
            numbers.insert(numbers.end(), incomingData.begin(), incomingData.end());
        }

        // 3) less를 하위로 보내기 _ 근데 rank-half가 greater 담당 그룹이면 1 빼준다
        int count = ((rank - half) == half) ? 1 : 0;
        int lessSize = less.size();
        MPI_Send(&lessSize, 1, MPI_INT, rank - half - count, 0, group);
        MPI_Send(less.data(), lessSize, MPI_INT, rank - half - count, 1, group);

        // 4) 재귀
        int new_rank;
        MPI_Comm_rank(new_group, &new_rank);
        parallel_quicksort(new_rank, numbers, p_num - half, new_group);
    }
    MPI_Comm_free(&new_group);
}

void redistribute_data(vector<int> &numbers, int rank, int p_num, int N) {
    int localSize = numbers.size();
    vector<int> allSizes(p_num, 0);
    
    // 1. Rank 0이 각 Rank의 데이터 크기를 수집
    MPI_Gather(&localSize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> allData;
    vector<int> index(p_num, 0);

    if (rank == 0) {
        // 2. Rank 0에서 전체 데이터를 위한 공간 준비
        int totalSize = accumulate(allSizes.begin(), allSizes.end(), 0);
        allData.resize(totalSize);

        for (int i = 1; i < p_num; ++i) {
            index[i] = index[i - 1] + allSizes[i - 1]; // 데이터 시작 위치 계산
        }
    }

    // 3. Rank 0이 모든 데이터를 수집
    MPI_Gatherv(numbers.data(), localSize, MPI_INT, 
                allData.data(), allSizes.data(), index.data(), MPI_INT, 
                0, MPI_COMM_WORLD);

    // 4. 데이터를 다시 각 Rank로 분배
    vector<int> finalSizes(p_num, N / p_num); // 각 Rank에 N / p_num 만큼 분배
    vector<int> finalIndex(p_num, 0);

    if (rank == 0) {
        for (int i = 1; i < p_num; ++i) {
            finalIndex[i] = finalIndex[i - 1] + finalSizes[i - 1]; // 시작 위치 계산
        }
    }

    vector<int> finalNumbers(N / p_num); // 각 Rank가 받을 데이터
    MPI_Scatterv(allData.data(), finalSizes.data(), finalIndex.data(), MPI_INT, 
                finalNumbers.data(), N / p_num, MPI_INT, 
                0, MPI_COMM_WORLD);

    // 기존 numbers를 받은 데이터로 교체
    numbers = finalNumbers;
}

void write_file(const string& filename, vector<int> & data, int N){
    ofstream outfile(filename);
    if(!outfile){
        cerr << "Cannot open file " << filename << "\n";
        return;
    }

    outfile << N << "\n";
    for(int num : data){
        outfile << num << "\n";
    }
    outfile.close();
}


int main(int argc, char** argv){
    MPI_Init(&argc, &argv); // MPI initialize

    int rank, p_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // process ID
    MPI_Comm_size(MPI_COMM_WORLD, &p_num); // total process num

    int N = atoi(argv[1]);
    string outputPath = argv[2];

    int chunkSize = N / p_num;

    vector<int> numbers; // vector for random generated numbers

    // 1) generate random number for ranks
    generate_numbers(rank, chunkSize, numbers);

    double timeStart = monotonic_seconds();

    // 2) start parallel quicksort
    parallel_quicksort(rank, numbers, p_num, MPI_COMM_WORLD);
    // 3) redistribute data to ensure that every worker has N/P data
    redistribute_data(numbers, rank, p_num, N);

    double timeEnd = monotonic_seconds();
    // 4) check the time
    double localSortTime = timeEnd - timeStart;
    double maxSortTime;
    MPI_Reduce(&localSortTime, &maxSortTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        print_time(maxSortTime);
    }

    // 5) write to file
    vector<int> finalData(rank == 0 ? N : 0);
    MPI_Gather(numbers.data(), chunkSize, MPI_INT, finalData.data(), chunkSize, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0){
        write_file(outputPath, finalData, N);
    }

    MPI_Finalize();
    return 0;
}