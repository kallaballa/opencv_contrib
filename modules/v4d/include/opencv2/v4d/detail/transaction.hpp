#ifndef MODULES_V4D_SRC_BACKEND_HPP_
#define MODULES_V4D_SRC_BACKEND_HPP_

#include "context.hpp"
#include "../util.hpp"
#include <tuple>
#include <iostream>
#include <utility>
#include <type_traits>
#include <opencv2/core.hpp>

namespace cv {
namespace v4d {
namespace detail {
template<auto V1, decltype(V1) V2, typename T>
struct values_equal : std::bool_constant<V1 == V2>
{
    using type = T;
};

// default_type<T>::value is always true
template<typename T>
struct default_type : std::true_type
{
    using type = T;
};

class EdgeBase {

};


template<typename T, bool Tcopy, bool Tread, typename Tbase = void>
class Edge : public EdgeBase {
public:
	using copy_t = std::integral_constant<bool, Tcopy>;
	using read_t = std::integral_constant<bool, Tread>;
	using temp_t = values_equal<std::is_same<Tbase, void>::value, false, std::true_type>;
private:
	template <typename, typename = void>
	struct has_deref_t : std::false_type {};

	template <typename Tval>
	struct has_deref_t<Tval, std::void_t<decltype(&Tval::operator*)>> : std::is_same<std::true_type, std::true_type>
	{};

	template <typename, typename = void>
	struct has_arrow_t : std::false_type {};

	template <typename Tval>
	struct has_arrow_t<Tval, std::void_t<decltype(&Tval::operator->)>> : std::is_same<std::true_type, std::true_type>
	{};

	template <typename, typename = void>
	struct has_get_t : std::false_type {};

	template <typename Tval>
	struct has_get_t<Tval, std::void_t<decltype(&Tval::get)>> : std::is_same<std::true_type, std::true_type>
	{};

	using ispointer_t = std::is_pointer<T>;

	using issmart_t = typename std::conjunction<
			has_deref_t<T>,
			has_arrow_t<T>,
			has_get_t<T>,
			values_equal<ispointer_t::value, false, std::true_type>
			>::type;

	static_assert((!ispointer_t::value && !issmart_t::value) || !copy_t::value, "You are trying to explicitly copy the value of a (smart) pointer.");
	static_assert((!temp_t::value) || (!copy_t::value && read_t::value), "Internal error: Trying to form a copy or write edge to a temporary.");

	using without_ptr_t = typename std::remove_pointer<T>::type;
 	using without_ptr_and_cont_t = typename std::remove_const<without_ptr_t>::type;
	using internal_ptr_t = typename std::disjunction<
			values_equal<temp_t::value, true, Tbase*>,
			values_equal<read_t::value, true, const without_ptr_and_cont_t*>,
			default_type<without_ptr_and_cont_t*>
			>::type;

	using holder_t = typename std::disjunction<
			values_equal<issmart_t::value, true, T>,
			default_type<nullptr_t>
			>::type;

	internal_ptr_t ptr_ = nullptr;
	internal_ptr_t copyPtr_ = nullptr;
	holder_t holder_ = nullptr;

	template<typename Tptr>
	static auto get_ptr(internal_ptr_t t) {
		return reinterpret_cast<Tptr*>(t);
	}

	template<typename Tplan, typename Tval>
	static void check(Tplan& plan, Tval& val) {
		const char* planPtr = reinterpret_cast<const char*>(&plan);
		const char* tPtr = reinterpret_cast<const char*>(&val);
		ptrdiff_t actualTypeSize = plan.getActualTypeSize();
		ptrdiff_t planSize = sizeof(Tplan);
		ptrdiff_t diff = tPtr - planPtr;

		if(diff < 0 || diff > (actualTypeSize + planSize)) {
			throw std::runtime_error("Variable of type " + demangle(typeid(T).name()) + " not a member of plan. Maybe it is a shared variable and you forgot to register it?");
		}
	}
public:
	using value_t = typename std::disjunction<
			values_equal<temp_t::value, true, T>,
			values_equal<read_t::value, true, const T&>,
			default_type<T&>
			>::type;

	template<typename Tplan>
	static Edge make(Tplan& plan, value_t t, const bool doCheck = true) {
		Edge e;
		if(doCheck)
			check(plan, t);

		e.set(t);
		return e;
	}

	void set(value_t t) {
		if constexpr(temp_t::value) {
			holder_ = t;
			ptr_ = holder_.get();
		} else {
			ptr_ = &t;
		}

		if constexpr(copy_t::value && !read_t::value) {
			copyPtr_ = new typename std::remove_pointer<decltype(ptr())>::type();
		}
	}

	auto* ptr() const {
		return get_ptr<typename std::remove_pointer<decltype(ptr_)>::type>(ptr_);
	}

	size_t id() const {
		return reinterpret_cast<size_t>(ptr_);
	}

	auto& get_value() const {
    	if constexpr(std::disjunction<issmart_t, ispointer_t>::value) {
       		return *ptr();
    	} else {
    		static_assert(temp_t::value, "Internal Error: Found a temporary that is not a smart pointer.");
    		if constexpr(!copy_t::value || read_t::value) {
        		return *ptr();
        	} else if constexpr(std::conjunction<copy_t, read_t>::value) {
        		return Global::copy(*ptr());
        	} else {
            	Global::safe_copy(*ptr(), *copyPtr_);
            	return *copyPtr_;
        	}
    	}
	}

    void copy_back() {
    	if constexpr(copy_t::value && !read_t::value) {
    		if constexpr(!std::disjunction<issmart_t, ispointer_t, temp_t>::value) {
        		Global::safe_copy(*copyPtr_, *ptr_);
        	}
    	}
    }
};
}
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
	std::function<cv::Ptr<cv::v4d::detail::V4DContext>()> ctxCallback_;
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


	CV_EXPORTS void setContextCallback(std::function<cv::Ptr<cv::v4d::detail::V4DContext>()> ctx);
	CV_EXPORTS std::function<cv::Ptr<cv::v4d::detail::V4DContext>()> getContextCallback();
};

namespace detail {

template <typename F, typename... Ts>
class TransactionImpl : public Transaction
{
    static_assert(sizeof...(Ts) == 0 || (!(std::is_rvalue_reference_v<Ts> && ...)));
private:
    F f;
    std::tuple<Ts...> args_;
    bool ran_;
public:
    template <typename FwdF, typename... FwdTs,
        typename = std::enable_if_t<sizeof...(Ts) == 0 || ((std::is_convertible_v<FwdTs&&, Ts> && ...))>>
		TransactionImpl(FwdF func, FwdTs... fwdArgs)
        : f(func),
          args_{std::make_tuple(fwdArgs...)},
		  ran_(false)
    {}

    virtual ~TransactionImpl() override
	{}

    virtual bool ran() override {
    	return ran_;
    }

    template <typename _Fn, typename _Tuple, size_t... _Idx>
    void
    performImpl(_Fn&& __f, _Tuple&& __t, std::index_sequence<_Idx...>) {
    	std::invoke(std::forward<_Fn>(__f),
    			std::get<_Idx>(std::forward<_Tuple>(__t)).get_value()...);
        (std::get<_Idx>(std::forward<_Tuple>(__t)).copy_back(),...);
    }

    template <typename _Fn, typename _Tuple, size_t... _Idx>
    constexpr decltype(auto)
    performImplRet(_Fn&& __f, _Tuple&& __t, std::index_sequence<_Idx...>) {
    	auto res = std::invoke(std::forward<_Fn>(__f),
    			std::get<_Idx>(std::forward<_Tuple>(__t)).get_value()...);
        (std::get<_Idx>(std::forward<_Tuple>(__t)).copy_back(),...);
        return res;
    }

    virtual void perform() override {
        using _Indices= std::make_index_sequence<std::tuple_size<decltype(args_)>::value>;
        performImpl(std::forward<F>(f),
  			       std::forward<decltype(args_)>(args_),
  			       _Indices{});
        ran_ = true;
    }

    template<bool b>
    typename std::enable_if<b, bool>::type enabled() {
        using _Indices= std::make_index_sequence<std::tuple_size<decltype(args_)>::value>;
        bool res = performImplRet(std::forward<F>(f),
  			       std::forward<decltype(args_)>(args_),
  			       _Indices{});
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
cv::Ptr<Transaction> make_transaction(F f, Args... args) {
    return cv::Ptr<Transaction>(dynamic_cast<Transaction*>(new detail::TransactionImpl<std::decay_t<F>, std::remove_cv_t<Args>...>
        (f, args...)));
}
}
}

#endif /* MODULES_V4D_SRC_BACKEND_HPP_ */
