// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_SINK_HPP_
#define SRC_OPENCV_V4D_SINK_HPP_

#include <functional>
#include <map>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/mat.hpp>
#include <mutex>
#include <string>

using std::string;
namespace cv {
namespace v4d {

class V4D;
/*!
 * A Sink object represents a way to write data produced by V4D (e.g. a video-file).
 */
class CV_EXPORTS Sink {
    std::mutex mtx_;
    bool open_ = true;
    //0 is an illegal sequence number
    uint64_t nextSeq_ = 1;
    std::map<uint64_t, cv::UMat> buffer_;
    std::function<bool(const uint64_t&, const cv::UMat&)> consumer_;
public:
    /*!
     * Constructs the Sink object from a consumer functor.
     * @param consumer A function object that consumes a UMat frame (e.g. writes it to a video file).
     */
    CV_EXPORTS Sink(std::function<bool(const uint64_t&, const cv::UMat&)> consumer);
    /*!
     * Constucts a null Sink that is never open or ready
     */
    CV_EXPORTS Sink();
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~Sink();
    /*!
     * Signals if the sink is ready to consume data.
     * @return true if the sink is ready.
     */
    CV_EXPORTS bool isReady();
    /*!
     * Determines if the sink is open.
     * @return true if the sink is open.
     */
    CV_EXPORTS bool isOpen();
    /*!
     * The sink operator. It accepts a UMat frame to pass to the consumer
     * @param frame The frame to pass to the consumer. (e.g. VideoWriter)
     */
    CV_EXPORTS void operator()(const uint64_t& seq, const cv::UMat& frame);


    static cv::Ptr<Sink> make(cv::Ptr<V4D> window, const string& outputFilename, const float fps, const cv::Size& frameSize);
    static cv::Ptr<Sink> make(cv::Ptr<V4D> window, const string& outputFilename, const float fps,const cv::Size& frameSize, int fourcc);
private:
    static cv::Ptr<Sink> makeVaSink(cv::Ptr<V4D> window, const string& outputFilename, const int fourcc, const float fps, const cv::Size& frameSize, const int vaDeviceIndex);
    static cv::Ptr<Sink> makeAnyHWSink(const string& outputFilename, const int fourcc, const float fps, const cv::Size& frameSize);
};

} /* namespace v4d */
} /* namespace kb */

#endif /* SRC_OPENCV_V4D_SINK_HPP_ */
