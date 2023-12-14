// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/sink.hpp"
#include "opencv2/v4d/v4d.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace v4d {

cv::Ptr<Sink> Sink::makeVaSink(cv::Ptr<V4D> window, const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize, const int vaDeviceIndex) {
    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG,
            fourcc, fps, frameSize, {
                    cv::VIDEOWRITER_PROP_HW_DEVICE, vaDeviceIndex,
                    cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
                    cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });

    window->sourceCtx()->copyContext();

    cerr << "Using a VA sink" << endl;
    if(writer->isOpened()) {
		return new Sink([=](const uint64_t& seq, const cv::UMat& frame) {
			CV_UNUSED(seq);
	        //FIXME cache it
            cv::UMat converted;
            cv::resize(frame, converted, frameSize);
            cvtColor(converted, converted, cv::COLOR_BGRA2RGB);
            (*writer) << converted;
			return writer->isOpened();
		});
    } else {
    	return new Sink();
    }
}

cv::Ptr<Sink> Sink::makeAnyHWSink(const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize) {
    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG,
            fourcc, fps, frameSize, { cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY });

    if(writer->isOpened()) {
        return new Sink([=](const uint64_t& seq, const cv::UMat& frame) {
        	CV_UNUSED(seq);
            cv::UMat converted;
            cv::UMat context_corrected;
            frame.copyTo(context_corrected);
            cv::resize(context_corrected, converted, frameSize);
            cvtColor(converted, converted, cv::COLOR_BGRA2RGB);
            (*writer) << converted;
            return writer->isOpened();
        });
    } else {
        return new Sink();
    }
}

cv::Ptr<Sink> Sink::make(cv::Ptr<V4D> window, const string& outputFilename, const float fps, const cv::Size& frameSize) {
    int fourcc = 0;
    //FIXME find a cleverer way to guess a decent codec
    if(getGlVendor() == "NVIDIA Corporation") {
        fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    } else {
        fourcc = cv::VideoWriter::fourcc('V', 'P', '9', '0');
    }
    return make(window, outputFilename, fps, frameSize, fourcc);
}

cv::Ptr<Sink> Sink::make(cv::Ptr<V4D> window, const string& outputFilename, const float fps,
        const cv::Size& frameSize, int fourcc) {
    if (isIntelVaSupported()) {
        return makeVaSink(window, outputFilename, fourcc, fps, frameSize, 0);
    } else {
        try {
            return makeAnyHWSink(outputFilename, fourcc, fps, frameSize);
        } catch(...) {
            cerr << "Failed creating hardware source" << endl;
        }
    }

    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG,
            fourcc, fps, frameSize);

    if(writer->isOpened()) {
		return new Sink([=](const uint64_t& seq, const cv::UMat& frame) {
			CV_UNUSED(seq);
            cv::UMat converted;
            cv::resize(frame, converted, frameSize);
            cvtColor(converted, converted, cv::COLOR_BGRA2RGB);
            (*writer) << converted;
			return writer->isOpened();
		});
    } else {
    	return new Sink();
    }
}

Sink::Sink(std::function<bool(const uint64_t&, const cv::UMat&)> consumer) :
        consumer_(consumer) {
}

Sink::Sink() {

}
Sink::~Sink() {
}

bool Sink::isReady() {
	std::lock_guard<std::mutex> lock(mtx_);
    if (consumer_)
        return true;
    else
        return false;
}

bool Sink::isOpen() {
	std::lock_guard<std::mutex> lock(mtx_);
    return open_;
}

void Sink::operator()(const uint64_t& seq, const cv::UMat& frame) {
	std::lock_guard<std::mutex> lock(mtx_);
	if(seq == nextSeq_) {
		uint64_t currentSeq = seq;
		cv::UMat currentFrame = frame;
		buffer_[seq] = frame;
		do {
			open_ = consumer_(currentSeq, currentFrame);
			++nextSeq_;
			buffer_.erase(buffer_.begin());
			if(buffer_.empty())
				break;
			auto pair = (*buffer_.begin());
			currentSeq = pair.first;
			currentFrame = pair.second;
		} while(currentSeq == nextSeq_);
	} else {
		buffer_[seq] = frame;
	}
	if(buffer_.size() > 2048) {
		CV_LOG_WARNING(nullptr, "Buffer overrun in sink.");
		buffer_.clear();
	}
}
} /* namespace v4d */
} /* namespace kb */
