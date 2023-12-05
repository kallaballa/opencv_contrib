#include "../../include/opencv2/v4d/detail/transaction.hpp"
#include "../../include/opencv2/v4d/v4d.hpp"

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

void Transaction::setBranchType(BranchType btype) {
	btype_ = btype;
}

BranchType Transaction::getBranchType() {
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
