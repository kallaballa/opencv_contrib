// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/source.hpp"
#include "opencv2/v4d/v4d.hpp"

namespace cv {
namespace v4d {

cv::Ptr<Source> Source::makeVaSource(cv::Ptr<V4D> window, const string& inputFilename, const int vaDeviceIndex) {
    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, vaDeviceIndex, cv::CAP_PROP_HW_ACCELERATION,
            cv::VIDEO_ACCELERATION_VAAPI, cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    float fps = capture->get(cv::CAP_PROP_FPS);
    CV_LOG_INFO(nullptr, "Using a VA source");

    window->sourceCtx()->copyContext();

    return new Source([=](cv::UMat& frame) {
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}

cv::Ptr<Source> Source::makeAnyHWSource(cv::Ptr<V4D> window, const string& inputFilename) {
	cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY
	});

	float fps = capture->get(cv::CAP_PROP_FPS);

    window->sourceCtx()->copyContext();

    return new Source([=](cv::UMat& frame) {
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}

cv::Ptr<Source> Source::make(cv::Ptr<V4D> window, const string& inputFilename) {
#ifdef HAVE_VA
	if (is_intel_va_supported()) {
        return makeVaSource(window, inputFilename, 0);
    } else
#endif
    {
        try {
            return makeAnyHWSource(window, inputFilename);
        } catch(...) {
            CV_LOG_INFO(nullptr, "Failed to create hardware source");
        }
    }

    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG);
    float fps = capture->get(cv::CAP_PROP_FPS);

    return new Source([=](cv::UMat& frame) {
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}

Source::Source(std::function<bool(cv::UMat&)> generator, float fps) :
        generator_(generator), fps_(fps) {
}

Source::Source() :
        open_(false), fps_(0) {
}

Source::~Source() {
}

bool Source::isOpen() {
	std::lock_guard<std::mutex> guard(mtx_);
    return generator_ && open_;
}

float Source::fps() {
    return fps_;
}

std::pair<uint64_t, cv::UMat> Source::operator()() {
	std::lock_guard<std::mutex> guard(mtx_);
	static thread_local cv::UMat frame;
	open_ = generator_(frame);
	//first frame has the sequence number 1!
	return {++count_, frame};
}
} /* namespace v4d */
} /* namespace kb */
