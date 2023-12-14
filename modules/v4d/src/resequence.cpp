#include "../include/opencv2/v4d/detail/resequence.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace v4d {
	void Resequence::finish() {
		std::lock_guard lock(mtx_);
		finish_ = true;
		cv_.notify_all();
	}

	void Resequence::waitFor(const uint64_t& seq, std::function<void(uint64_t)> completion) {
		while(true) {
			{
				std::lock_guard lock(mtx_);
				if(finish_)
					break;

				if(seq == nextSeq_) {
					++nextSeq_;
					completion(seq);
					cv_.notify_all();
					break;
				}
			}

			std::unique_lock<std::mutex> lock(mtx_);
			cv_.wait(lock, [this, seq](){ return seq == nextSeq_ || finish_;});
		}
    }
} /* namespace v4d */
} /* namespace cv */
