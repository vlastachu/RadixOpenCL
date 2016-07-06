#include <functional>
#include "main.h"
using namespace std;

#define arraySize 1 << 17

void singleThreadRadixSort(const int* input, int* output, int length, int cap, int bits) {
    int buffer[length];
    int maxDigit = 1 << cap;
    for (int i = 0; i < length; i++)
        buffer[i] = input[i];

    // we going from least significant digit position to most
    for (int binPos = 0; binPos <= bits; binPos += cap) {
        int offset = 0;
        for (int digit = 0; digit < maxDigit; digit++) {
            for (int i = 0; i < length; i++) {
                const int n = buffer[i] >> binPos;
                const int input_digit = n % maxDigit;
                if (input_digit == digit) {
                    output[offset] = buffer[i];
                    offset++;
                }
            }
        }
        for (int i = 0; i < length; i++)
            buffer[i] = output[i];
    }
}

double countTime(function<void()> fn) {
    auto start = chrono::system_clock::now();
    fn();
    auto end = chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    return elapsed_seconds.count();
}

// print vector
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    if ( !v.empty() ) {
        out << '[';
        std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, "\t"));
        out << "\b\b]";
    }
    return out;
}

vector<int> generateRandomVector(int size) {
    std::vector<int> v(size);
    std::iota(v.begin(), v.end(), 0);

    std::shuffle(v.begin(), v.end(), std::mt19937{std::random_device{}()});
    return v;
}

vector<int> generateRandomVectorAllPositiveIntegers(int size) {
    std::vector<int> v(size);
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(0, std::numeric_limits<int>::max()); // guaranteed unbiased
    for (int i = 0; i < size; ++i) {
        v.push_back(uni(rng));
    }
    return v;
}

int main() {
    // get platform
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    cl_int result;

    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    // actually program
    cl::Context context(default_device);
    cl::Program::Sources sources;

    ifstream t("/home/vlastachu/dev/opencl/test/myopencl/kernel.c");
    std::string kernel_code((std::istreambuf_iterator<char>(t)),
                    std::istreambuf_iterator<char>());
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(context,sources, &result);
    if (result  != CL_SUCCESS) cerr << getErrorString(result) << endl;
    if(program.build({default_device}) != CL_SUCCESS) {
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    cl::Buffer buffer_input(context, CL_MEM_READ_WRITE, sizeof(int)*arraySize);
    cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, sizeof(int)*arraySize);
    cl::Buffer buffer_args(context, CL_MEM_READ_WRITE, sizeof(int)*3);

    vector<int> randomInput = generateRandomVectorAllPositiveIntegers(arraySize);

    int* input = &randomInput[0];

//    std::cout << "\n input: \n";
//    for(int i=0;i<arraySize;i++){
//        std::cout << input[i] << " ";
//    }

    int output[arraySize] = {};
    int cap = 8, bits = 31;
    unsigned int numOfThreads = 1 << cap;

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);

    //write arrays A and B to the device
    check(queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, sizeof(int)*arraySize, input));
    check(queue.enqueueWriteBuffer(buffer_output, CL_TRUE, 0, sizeof(int)*arraySize, output));


    int _arraySize = arraySize;
    cl::Kernel simple_add(program, "simple_add");
    check(simple_add.setArg(0, buffer_input));
    check(simple_add.setArg(1, buffer_output));
    check(simple_add.setArg(2, arraySize));
    // set cap dinamically
    check(simple_add.setArg(4, bits));



//    for(int i=0;i<arraySize;i++){
//        std::cout << output[i] << " ";
//    }
    //simple_add(buffer_A, buffer_B, buffer_C);

    //read result C from the device to array C

    vector<double> singleThreadResults, manyThreadsResults;
    for (cap = 1; cap <= bits; cap++){
        if (singleThreadResults.empty() || singleThreadResults.back() < 3) {
            singleThreadResults.push_back(countTime([&] {
                singleThreadRadixSort(input, output, arraySize, cap, bits);
            }));
        }

        manyThreadsResults.push_back(countTime([&]{
            simple_add.setArg(3, cap);
            queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(1 << cap));
            queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(int)*arraySize, output);
        }));
    }

    std::cout << "\n result (1 thread): \n" << singleThreadResults;
    std::cout << "\n result: \n" << manyThreadsResults;
//    for(int i=0;i<arraySize;i++){
//        std::cout << output[i] << " ";
//    }
    return 0;
}