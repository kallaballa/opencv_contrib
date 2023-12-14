#ifndef MODULES_V4D_SRC_BACKEND_HPP_
#define MODULES_V4D_SRC_BACKEND_HPP_

#include "context.hpp"

#include <tuple>
#include <functional>
#include <utility>
#include <type_traits>
#include <opencv2/core.hpp>

namespace cv {
namespace v4d {

struct BranchType {
	enum Enum {
	NONE = 0,
	SINGLE = 1,
	PARALLEL = 2,
	ONCE = 4,
	PARALLEL_ONCE = 8
	};
};

class CV_EXPORTS Transaction {
private:
	cv::Ptr<cv::v4d::detail::V4DContext> ctx_;
	BranchType::Enum btype_;
	cv::Rect viewport_;
public:
	CV_EXPORTS Transaction();
	CV_EXPORTS virtual ~Transaction() {}
	CV_EXPORTS virtual void perform() = 0;
	CV_EXPORTS virtual bool ran() = 0;
	CV_EXPORTS virtual bool enabled() = 0;
	CV_EXPORTS virtual bool isPredicate() = 0;
	CV_EXPORTS bool isBranch();
	CV_EXPORTS void setBranchType(BranchType::Enum btype);
	CV_EXPORTS BranchType::Enum getBranchType();
	CV_EXPORTS void setViewport(const cv::Rect& vp);
	CV_EXPORTS cv::Rect getViewport();


	CV_EXPORTS void setContext(cv::Ptr<cv::v4d::detail::V4DContext> ctx);
	CV_EXPORTS cv::Ptr<cv::v4d::detail::V4DContext> getContext();
};

namespace detail {

template <typename F, typename... Ts>
class TransactionImpl : public Transaction
{
    static_assert(sizeof...(Ts) == 0 || (!(std::is_rvalue_reference_v<Ts> && ...)));
private:
    F f;
    std::tuple<Ts...> args;
    bool ran_;
public:
    template <typename FwdF, typename... FwdTs,
        typename = std::enable_if_t<sizeof...(Ts) == 0 || ((std::is_convertible_v<FwdTs&&, Ts> && ...))>>
		TransactionImpl(FwdF&& func, FwdTs&&... fwdArgs)
        : f(std::forward<FwdF>(func)),
          args{std::forward_as_tuple(fwdArgs...)},
		  ran_(false)
    {}

    virtual ~TransactionImpl() override
	{}

    virtual bool ran() override {
    	return ran_;
    }

    virtual void perform() override
    {
        std::apply(f, args);
        ran_ = true;
    }


    template<bool b>
    typename std::enable_if<b, bool>::type enabled() {
    	bool res = std::apply(f, args);
    	ran_ = true;
    	return res;
    }

    template<bool b>
    typename std::enable_if<!b, bool>::type enabled() {
    	return false;
    }

    virtual bool enabled() override {
    	return enabled<std::is_same_v<std::remove_cv_t<typename decltype(f)::result_type>, bool>>();
    }

    template<bool b>
    typename std::enable_if<b, bool>::type isPredicate() {
    	return true;
    }

    template<bool b>
    typename std::enable_if<!b, bool>::type isPredicate() {
    	return false;
    }

    virtual bool isPredicate() override {
    	return isPredicate<std::is_same_v<std::remove_cv_t<typename decltype(f)::result_type>, bool>>();
    }
};
}

template <typename F, typename... Args>
cv::Ptr<Transaction> make_transaction(F f, Args&&... args) {
    return cv::Ptr<Transaction>(dynamic_cast<Transaction*>(new detail::TransactionImpl<std::decay_t<F>, std::remove_cv_t<Args>...>
        (std::forward<F>(f), std::forward<Args>(args)...)));
}


//template <typename F, typename Tfb, typename... Args>
//cv::Ptr<Transaction> make_transaction(F f, Tfb&& fb, Args&&... args) {
//	return cv::Ptr<Transaction>(dynamic_cast<Transaction*>(new detail::TransactionImpl<std::decay_t<F>, std::remove_cv_t<Tfb>, std::remove_cv_t<Args>...>
//        (std::forward<F>(f), std::forward<Tfb>(fb), std::forward<Args>(args)...)));
//}


}
}

#endif /* MODULES_V4D_SRC_BACKEND_HPP_ */
