#include <cmath>
#include <stdexcept>

#include "image_preprocessor.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

ImagePreprocessor::ImagePreprocessor(int n_px) : n_px_(n_px) {
	if (n_px_ <= 0) {
		throw std::invalid_argument("n_px must be greater than 0");
	}
}

std::vector<float> ImagePreprocessor::preprocess(const std::string& image_path) const {
	const cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
	if (image.empty()) {
		throw std::runtime_error("Failed to read image: " + image_path);
	}
	return preprocess(image);
}

std::vector<float> ImagePreprocessor::preprocess(const cv::Mat& image) const {
	if (image.empty()) {
		throw std::invalid_argument("Input image is empty");
	}

	cv::Mat resized = resize_shorter_side_bicubic(image);
	cv::Mat cropped = center_crop(resized);
	cv::Mat rgb = convert_to_rgb(cropped);

	cv::Mat float_rgb;
	rgb.convertTo(float_rgb, CV_32FC3, 1.0 / 255.0);

	std::vector<cv::Mat> channels(3);
	cv::split(float_rgb, channels);

	for (int channel_index = 0; channel_index < 3; ++channel_index) {
		channels[channel_index] = (channels[channel_index] - kMean[channel_index]) / kStd[channel_index];
	}

	const int tensor_size = 3 * n_px_ * n_px_;
	std::vector<float> tensor;
	tensor.reserve(tensor_size);

	for (int channel_index = 0; channel_index < 3; ++channel_index) {
		const float* begin = channels[channel_index].ptr<float>(0);
		const float* end = begin + channels[channel_index].total();
		tensor.insert(
			tensor.end(),
			begin,
			end
		);
	}

	return tensor;
}

cv::Mat ImagePreprocessor::resize_shorter_side_bicubic(const cv::Mat& input) const {
	const int width = input.cols;
	const int height = input.rows;

	if (width <= 0 || height <= 0) {
		throw std::invalid_argument("Input image has invalid dimensions");
	}

	int new_width = 0;
	int new_height = 0;

	if (width < height) {
		new_width = n_px_;
		new_height = static_cast<int>(std::round(static_cast<double>(height) * n_px_ / width));
	} else {
		new_height = n_px_;
		new_width = static_cast<int>(std::round(static_cast<double>(width) * n_px_ / height));
	}

	cv::Mat resized;
	cv::resize(input, resized, cv::Size(new_width, new_height), 0.0, 0.0, cv::INTER_CUBIC);
	return resized;
}

cv::Mat ImagePreprocessor::center_crop(const cv::Mat& input) const {
	if (input.cols < n_px_ || input.rows < n_px_) {
		throw std::runtime_error("Image is smaller than center crop size");
	}

	const int x_offset = (input.cols - n_px_) / 2;
	const int y_offset = (input.rows - n_px_) / 2;
	const cv::Rect crop_roi(x_offset, y_offset, n_px_, n_px_);
	return input(crop_roi).clone();
}

cv::Mat ImagePreprocessor::convert_to_rgb(const cv::Mat& input) {
	cv::Mat rgb;
	if (input.channels() == 3) {
		cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
	} else if (input.channels() == 4) {
		cv::cvtColor(input, rgb, cv::COLOR_BGRA2RGB);
	} else if (input.channels() == 1) {
		cv::cvtColor(input, rgb, cv::COLOR_GRAY2RGB);
	} else {
		throw std::runtime_error("Unsupported channel count in input image");
	}
	return rgb;
}
