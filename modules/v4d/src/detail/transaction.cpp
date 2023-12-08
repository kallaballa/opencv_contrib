#include "../../include/opencv2/v4d/detail/transaction.hpp"

#include <tuple>
#include <functional>
#include <utility>
#include <type_traits>
#include <opencv2/core.hpp>

namespace cv {
namespace v4d {
Transaction::Transaction() : btype_(BranchType::NONE) {

}

bool Transaction::isBranch() {
	return btype_ != BranchType::NONE;
}

void Transaction::setBranchType(BranchType::Enum btype) {
	btype_ = btype;
}

BranchType::Enum Transaction::getBranchType() {
	return btype_;
}

void Transaction::setContext(cv::Ptr<cv::v4d::detail::V4DContext> ctx) {
	ctx_ = ctx;
}

cv::Ptr<cv::v4d::detail::V4DContext> Transaction::getContext() {
	return ctx_;
}

}
}
