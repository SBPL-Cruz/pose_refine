#include "renderer.h"
#include <chrono>

#ifdef CUDA_ON
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#endif

using namespace cv;

static std::string prefix = "/home/jessy/pose_refine/test/";

namespace helper {
cv::Mat view_dep(cv::Mat dep){
    cv::Mat map = dep;
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
    return falseColorsMap;
};

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
}

int main(int argc, char const *argv[])
{
    const int width = 640; const int height = 480;

    cuda_renderer::Model model(prefix+"test.ply");

    Mat K = (Mat_<float>(3,3) << 572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0);
    auto proj = cuda_renderer::compute_proj(K, width, height);

    Mat R_ren = (Mat_<float>(3,3) << 1, 0, 0.00000000, 0,
                 1, 0, 0, 0,1);
    Mat t_ren = (Mat_<float>(3,1) << 10, 0, 25);

    cuda_renderer::Model::mat4x4 mat4;
    mat4.init_from_cv(R_ren, t_ren);

    std::vector<cuda_renderer::Model::mat4x4> mat4_v(1, mat4);
    for(int i = -50; i <50; i ++){
        for(int j = -1; j <10; j ++){
            t_ren = (Mat_<float>(3,1) << i, j, 25);
            mat4.init_from_cv(R_ren, t_ren);
            // std::cout<<mat4.a3;
            mat4_v.push_back(mat4);
        }
        
    }
    
    std::cout << "test render nums: " << mat4_v.size() << std::endl;
    std::cout << "---------------------------------\n" << std::endl;
    helper::Timer timer;

#ifdef CUDA_ON
    {  // gpu need sometime to warm up
        cudaFree(0);
//        cudaSetDevice(0);
    }

    if(true){   //render test
        std::cout << "\nrendering test" << std::endl;
        std::cout << "-----------------------\n" << std::endl;
        timer.reset();
        // std::cout<<model.tris[0].color.v1;
        // std::vector<int> result_cpu = cuda_renderer::render_cpu(model.tris, mat4_v, width, height, proj);
        // timer.out("cpu render");

        std::vector<int> result_gpu = cuda_renderer::render_cuda(model.tris, mat4_v, width, height, proj);
        timer.out("gpu render");

        // auto result_gpu_keep_in =
        //         cuda_renderer::render_cuda_keep_in_gpu(model.tris, mat4_v, width, height, proj);
        // timer.out("gpu_keep_in render");

        // std::vector<int> result_gpu_back_to_host(result_gpu_keep_in.size());
        // thrust::copy(result_gpu_keep_in.begin_thr(), result_gpu_keep_in.end_thr(), result_gpu_back_to_host.begin());
        // timer.out("gpu_keep_in back to host");

        // gpu cpu check
        // std::vector<int> result_diff(result_cpu.size());
        // for(size_t i=0; i<result_cpu.size(); i++){
        //     result_diff[i] = std::abs(result_cpu[i] - result_gpu[i]);
        // }
        // assert(std::accumulate(result_diff.begin(), result_diff.end(), 0) == 0 &&
        //        "rendering results, cpu should be same as gpu");

        // //gpu gpu_keep_in_check
        // for(size_t i=0; i<result_gpu_back_to_host.size(); i++){
        //     result_diff[i] = std::abs(result_gpu_back_to_host[i] - result_gpu[i]);
        // }
        // assert(std::accumulate(result_diff.begin(), result_diff.end(), 0) == 0 &&
        //        "rendering results, gpu keep in should be same as gpu");

        // just show first 1
        cv::Mat color_mat = cv::Mat(height, width, CV_8UC3);;
        for(int n = 0 ; n < 1; n ++){
            for(int i = 0; i < height; i ++){
                for(int j = 0; j <width; j ++){
                    int index = n*width*height+(i*width+j);
                    int red = (result_gpu[index] >> 16) & 0xFF;
                    int green = (result_gpu[index] >> 8) & 0xFF;
                    int blue = result_gpu[index] & 0xFF;
                    // std::cout<<red<<","<<green<<","<<blue<<std::endl;
                    color_mat.at<Vec3b>(i, j) = Vec3b(blue, green,red);
                }
            }
            std::string name;
            name = std::to_string(n);
            cv::imshow(name, color_mat);
        }
        
        imwrite( "/home/jessy/pose_refine/test/Image.jpg", color_mat );
        cv::waitKey(0);

        cv::Mat depth = cv::Mat(height, width, CV_32SC1, result_gpu.data());
        cv::FileStorage file("some_name.txt", cv::FileStorage::WRITE);
        file << "matName" << color_mat;
        //std::cout << "M = "<< std::endl << " "  << depth << std::endl << std::endl;
        // cv::Mat depth1 = cv::Mat(height, width, CV_32SC1, result_gpu.data()+99*height*width);
        // cv::imshow("gpu_mask1", depth1>0);
        // cv::imshow("gpu_depth1", helper::view_dep(depth1));
        // cv::imshow("gpu_mask", depth>0);
        // cv::imshow("gpu_depth", helper::view_dep(depth));
        // cv::waitKey(0);
    }

    if(true){   //roi render test
        std::cout << "\nroi test" << std::endl;
        std::cout << "-----------------------\n" << std::endl;
        timer.reset();

        //roi: topleft x, y, width, height
        cuda_renderer::Model::ROI roi = {160, 80, 320, 240};

        std::vector<int> result_cpu = cuda_renderer::render_cpu(model.tris, mat4_v, width, height, proj, roi);
        timer.out("cpu roi render");

        std::vector<int> result_gpu = cuda_renderer::render_cuda(model.tris, mat4_v, width, height, proj, roi);
        timer.out("gpu roi render");

        auto result_gpu_keep_in =
                cuda_renderer::render_cuda_keep_in_gpu(model.tris, mat4_v, width, height, proj, roi);
        timer.out("gpu_keep_in roi render");

        std::vector<int> result_gpu_back_to_host(result_gpu_keep_in.size());
        thrust::copy(result_gpu_keep_in.begin_thr(), result_gpu_keep_in.end_thr(), result_gpu_back_to_host.begin());
        timer.out("gpu_keep_in roi back to host");

        std::vector<int> result_diff(result_cpu.size());
        for(size_t i=0; i<result_cpu.size(); i++){
            result_diff[i] = std::abs(result_cpu[i] - result_gpu[i]);
        }
        assert(std::accumulate(result_diff.begin(), result_diff.end(), 0) == 0 &&
               "rendering results, cpu should be same as gpu");

        for(size_t i=0; i<result_cpu.size(); i++){
            result_diff[i] = std::abs(result_gpu_back_to_host[i] - result_gpu[i]);
        }
        assert(std::accumulate(result_diff.begin(), result_diff.end(), 0) == 0 &&
               "rendering results, gpu keep in should be same as gpu");

        // just show first 1
        cv::Mat depth = cv::Mat(roi.height, roi.width, CV_32SC1, result_cpu.data());
        cv::Mat depth_2 = cv::Mat(height, width, CV_32SC1, result_cpu.data() + height*width);
        cv::imshow("gpu_mask_roi", depth>0);
        cv::imshow("gpu_depth_roi", helper::view_dep(depth));
        cv::imshow("gpu_mask_roi", depth_2>0);
        cv::imshow("gpu_depth_roi", helper::view_dep(depth_2));
        cv::waitKey(0);
    }
#else
    std::vector<int> result_cpu = cuda_renderer::render_cpu(model.tris, mat4_v, width, height, proj);
    timer.out("cpu render");
    // just show first 1
    cv::Mat depth = cv::Mat(height, width, CV_32SC1, result_cpu.data());

    cv::imshow("mask", depth>0);
    cv::imshow("depth", helper::view_dep(depth));
    cv::waitKey(0);
#endif

    return 0;
}
