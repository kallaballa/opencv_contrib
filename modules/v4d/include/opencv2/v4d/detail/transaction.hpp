#ifndef MODULES_V4D_SRC_BACKEND_HPP_
#define MODULES_V4D_SRC_BACKEND_HPP_

#include "context.hpp"
#include "../util.hpp"
#include <tuple>
#include <set>
#include <string>
#include <utility>
#include <type_traits>
#include <opencv2/core/types.hpp>

namespace cv {
namespace v4d {
namespace detail {
template<auto V1, decltype(V1) V2, typename T>
struct values_equal : std::bool_constant<V1 == V2>
{
    using type = T;
};

template<typename T>
struct default_type : std::true_type
{
    using type = T;
};

template<typename T, bool Tcopy, bool Tread, bool Tshared = false, typename Tbase = void>
class Edge {
public:
	using copy_t = std::integral_constant<bool, Tcopy>;
	using read_t = std::integral_constant<bool, Tread>;
	using shared_t = std::integral_constant<bool, Tshared>;
	using temp_t = values_equal<std::is_same<Tbase, void>::value, false, std::true_type>;
	using lockie_t = values_equal<!copy_t::value && shared_t::value, true, std::true_type>;
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
	static_assert(shared_t::value || !(copy_t::value && !read_t::value), "Internal error: Trying to form  copy-write edge on a non-shared variable.");
	static_assert(!lockie_t::value || !copy_t::value, "Internal error: Trying to form a copy edge on a to be locked variable.");

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

	template<typename Tplan, typename Tvar>
	static void check(Tplan& plan, Tvar& var) {
		const char* planPtr = reinterpret_cast<const char*>(&plan);
		const char* varPtr = reinterpret_cast<const char*>(&var);
		off_t parentOffset = plan.getParentOffset();
		off_t parentActualSize = plan.getParentActualTypeSize();
		off_t actualTypeSize = plan.getActualTypeSize();
		off_t varOffset = off_t (varPtr);
		off_t planOffset = off_t (planPtr);
		off_t planSize = sizeof(Tplan);

		CV_Assert((parentOffset == 0  && parentActualSize == 0) || (parentOffset > 0 && parentActualSize > 0));
		CV_Assert(actualTypeSize > planSize);
		CV_Assert(parentActualSize == 0 || parentActualSize > actualTypeSize);

		off_t parentLowerBound = parentOffset;
		off_t parentUpperBound = parentOffset + parentActualSize;
		off_t lowerBound = planOffset;
		off_t upperBound = planOffset + actualTypeSize;

		if(! ((varOffset >= lowerBound && varOffset <= upperBound) || (parentOffset > 0 && varOffset >= parentLowerBound && varOffset <= parentUpperBound))) {
			throw std::runtime_error("Variable of type " + demangle(typeid(T).name()) + " not a member of plan. Maybe it is a shared variable and you forgot to register it?");
		}
	}
public:
	using pass_t = typename std::disjunction<
			values_equal<temp_t::value, true, T>,
			values_equal<read_t::value, true, const T&>,
			default_type<T&>
			>::type;

	using value_t = typename std::disjunction<
			values_equal<temp_t::value, true, Tbase>,
			values_equal<read_t::value, true, const T>,
			default_type<T>
			>::type;

	using ref_t = typename std::add_lvalue_reference<value_t>::type;

	template<typename Tplan>
	static Edge make(Tplan& plan, pass_t t, const bool doCheck = true) {
		Edge e;
		if(doCheck)
			check(plan, t);

		e.set(t);
		return e;
	}

	void set(pass_t t) {
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

	auto& ref() const {
		if constexpr(!copy_t::value) {
			return *ptr();
		} else {
    		if constexpr(shared_t::value) {
    			Global::instance().safe_copy(*ptr(), *copyPtr_);
				return *copyPtr_;
			} else {
				return *ptr();
			}
		}

		CV_Assert(false);
   		return *ptr();
	}

    void copyBack() {
    	if constexpr(shared_t::value && copy_t::value && !read_t::value) {
			Global::instance().safe_copy(*copyPtr_, *ptr_);
		}
    }

    std::mutex& getMutex() {
    	static_assert(lockie_t::value, "Internal Error: Trying to get mutex from a non-lockie edge");
    	//FIXME thows if the mutex ptr can't be found.
    	//write a no check/no throw alternative because this situation should be impossible
    	return *Global::instance().getMutexPtr(*ptr());
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
public:
	CV_EXPORTS Transaction();
	CV_EXPORTS virtual ~Transaction() {}
	CV_EXPORTS virtual constexpr bool isPredicate() = 0;

	CV_EXPORTS virtual bool hasLockies() = 0;
	CV_EXPORTS virtual bool hasCopyBacks()  = 0;
	CV_EXPORTS virtual void perform() = 0;
	CV_EXPORTS virtual bool performPredicate() = 0;
	CV_EXPORTS virtual bool ran() = 0;

	CV_EXPORTS bool isBranch();
	CV_EXPORTS void setBranchType(BranchType::Enum btype);
	CV_EXPORTS BranchType::Enum getBranchType();
	CV_EXPORTS void setContextCallback(std::function<cv::Ptr<cv::v4d::detail::V4DContext>()> ctx);
	CV_EXPORTS std::function<cv::Ptr<cv::v4d::detail::V4DContext>()> getContextCallback();
};

namespace detail {

template <typename T>
auto filter_lockie(T& t) {
	if constexpr(T::lockie_t::value) {
		return std::make_tuple(t);
	} else {
		return std::tuple<>();
	}
}

template<typename Tuple, size_t... _Idx>
auto filter_lockies(Tuple t, std::index_sequence<_Idx...>)
{
	return std::tuple_cat(filter_lockie(std::get<_Idx>(t))...);
}

template<typename Tuple, size_t ... I>
auto make_scoped_lock_ptr(Tuple t, std::index_sequence<I ...>)
{
	using lock_t = std::scoped_lock<std::remove_reference_t<decltype(std::get<I>(t).getMutex())>...>;
	cv::Ptr<lock_t> ptr = new lock_t(std::get<I>(t).getMutex()...);
	return ptr;
}

template<typename Tuple>
auto make_scoped_lock_ptr(Tuple t)
{
	static constexpr auto size = std::tuple_size<Tuple>::value;
	return make_scoped_lock_ptr(t, std::make_index_sequence<size>{});
}

template <typename F, typename... Ts>
class TransactionImpl : public Transaction
{
    static_assert(sizeof...(Ts) == 0 || (!(std::is_rvalue_reference_v<Ts> && ...)));
private:
    F f;
    std::tuple<Ts...> args_;
    bool ran_;
    using predicate_t = std::is_same<std::remove_cv_t<typename decltype(f)::result_type>, bool>;
	static constexpr bool hasLockies_ = (std::remove_reference_t<Ts>::lockie_t::value || ...);
	static constexpr bool hasCopyBacks_ = (
			(
					std::remove_reference_t<Ts>::copy_t::value
					&& !std::remove_reference_t<Ts>::read_t::value
			) && ...);
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
    performImpl(_Fn&& __f, _Tuple&& __t, std::index_sequence<_Idx...> seq) {
    	if constexpr(hasLockies_) {
    		auto scopedLock = make_scoped_lock_ptr(filter_lockies(__t, seq));
        	std::invoke(std::forward<_Fn>(__f),
        			std::get<_Idx>(std::forward<_Tuple>(__t)).ref()...);

        	if constexpr(hasCopyBacks_) {
        		(std::get<_Idx>(std::forward<_Tuple>(__t)).copyBack(),...);
        	}
    	} else {
    		std::invoke(std::forward<_Fn>(__f),
    			std::get<_Idx>(std::forward<_Tuple>(__t)).ref()...);

    		if constexpr(hasCopyBacks_) {
        		(std::get<_Idx>(std::forward<_Tuple>(__t)).copyBack(),...);
        	}
    	}
    }

    template <typename _Fn, typename _Tuple, size_t... _Idx>
    constexpr decltype(auto)
    performImplRet(_Fn&& __f, _Tuple&& __t, std::index_sequence<_Idx...> seq) {
    	bool res = false;
    	if constexpr(hasLockies_) {
    		auto scopedLock = make_scoped_lock_ptr(filter_lockies(__t, seq));

    		res = std::invoke(std::forward<_Fn>(__f),
        			std::get<_Idx>(std::forward<_Tuple>(__t)).ref()...);

        	if constexpr(hasCopyBacks_) {
        		(std::get<_Idx>(std::forward<_Tuple>(__t)).copyBack(),...);
        	}
    	} else {
        	res = std::invoke(std::forward<_Fn>(__f),
        			std::get<_Idx>(std::forward<_Tuple>(__t)).ref()...);

    		if constexpr(hasCopyBacks_) {
        		(std::get<_Idx>(std::forward<_Tuple>(__t)).copyBack(),...);
        	}
    	}

        return res;
    }

    virtual void perform() override {
        using _Indices= std::make_index_sequence<std::tuple_size<decltype(args_)>::value>;
        performImpl(std::forward<F>(f),
  			       std::forward<decltype(args_)>(args_),
  			       _Indices{});
        ran_ = true;
    }

    virtual bool performPredicate() override {
    	if constexpr(predicate_t::value) {
            using _Indices= std::make_index_sequence<std::tuple_size<decltype(args_)>::value>;
            bool res = performImplRet(std::forward<F>(f),
      			       std::forward<decltype(args_)>(args_),
      			       _Indices{});
        	ran_ = true;
        	return res;
    	} else {
    		CV_Error(cv::Error::StsInternal, "Internal Error: Trying to check a non-predicate");
    	}
    	return false;
    }

    virtual bool isPredicate() override {
    	return predicate_t::value;
    }

    virtual bool hasLockies() override {
    	return hasLockies_;
    }

    virtual bool hasCopyBacks() override {
    	return hasCopyBacks_;
    }
};
}

struct Node {
	string name_;
	std::set<long> read_deps_;
	std::set<long> write_deps_;
	cv::Ptr<Transaction> tx_  = nullptr;
	bool initialized() {
		return tx_;
	}
};

template <typename F, typename... Args>
cv::Ptr<Transaction> make_transaction(F f, Args... args) {
    return cv::Ptr<Transaction>(new detail::TransactionImpl<std::decay_t<F>, std::remove_cv_t<Args>...>
        (f, args...));
}
}
}

#endif /* MODULES_V4D_SRC_BACKEND_HPP_ */
