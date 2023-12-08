#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_RESEQUENCE_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_RESEQUENCE_HPP_

#include <functional>
#include <set>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/mat.hpp>
#include <mutex>
#include <semaphore>
#include <condition_variable>

namespace cv {
namespace v4d {



class CV_EXPORTS Resequence {
	bool finish_ = false;
	std::mutex putMtx_;
	std::mutex waitMtx_;
	std::condition_variable cv_;
    uint64_t nextSeq_ = 0;
public:
    CV_EXPORTS Resequence() {
    }

    CV_EXPORTS virtual ~Resequence() {}
    CV_EXPORTS void finish();
    CV_EXPORTS void notify();
    CV_EXPORTS void waitFor(const uint64_t& seq);
};

} /* namespace v4d */
} /* namespace kb */



#endif /* MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_RESEQUENCE_HPP_ */
