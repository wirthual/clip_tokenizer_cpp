#ifndef IMAGE_PREPROCESSOR_H
#define IMAGE_PREPROCESSOR_H

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

class ImagePreprocessor {
public:
    explicit ImagePreprocessor(int n_px);

    std::vector<float> preprocess(const std::string& image_path) const;
    std::vector<float> preprocess(const cv::Mat& image) const;

private:
    cv::Mat resize_shorter_side_bicubic(const cv::Mat& input) const;
    cv::Mat center_crop(const cv::Mat& input) const;
    static cv::Mat convert_to_rgb(const cv::Mat& input);

    int n_px_;
    static constexpr std::array<float, 3> kMean = {0.48145466f, 0.4578275f, 0.40821073f};
    static constexpr std::array<float, 3> kStd = {0.26862954f, 0.26130258f, 0.27577711f};
};

#endif