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

class EdgeBase {};

template<typename T, bool Tcopy, bool Tread, bool Tshared = false, typename Tbase = void>
class Edge : public EdgeBase {
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

public:

	using value_type_t = T;
	using copy_t = std::integral_constant<bool, Tcopy>;
	using read_t = std::integral_constant<bool, Tread>;
	using shared_t = std::integral_constant<bool, Tshared>;
	using temp_t = values_equal<std::is_same<Tbase, void>::value, false, std::true_type>;
	using lockie_t = values_equal<!copy_t::value && shared_t::value, true, std::true_type>;
	using func_t = values_equal<
					!std::is_same<typename element_t<value_type_t>::type, std::false_type>::value
				    && !std::is_same<typename return_t<typename element_t<value_type_t>::type>::type, std::false_type>::value, true, std::true_type>;
private:
	using ispointer_t = values_equal<std::is_pointer<T>::value || std::is_array<T>::value, true, std::true_type>;

	using issmart_t = typename std::conjunction<
			has_deref_t<value_type_t>,
			has_arrow_t<value_type_t>,
			has_get_t<value_type_t>,
			values_equal<ispointer_t::value, false, std::true_type>
			>::type;

	using func_ret_t = typename return_t<typename element_t<value_type_t>::type>::type;

 	using orig_t = typename std::disjunction<
			values_equal<temp_t::value, true, Tbase>,
			default_type<value_type_t>
			>::type;

 	using base_maybe_const_t = typename std::remove_pointer<typename std::remove_extent<orig_t>::type>::type;
 	using base_t = typename std::remove_const<base_maybe_const_t>::type;


 	using iswriteable_func_t = typename std::conjunction<
			values_equal<std::is_reference<func_ret_t>::value, true, std::true_type>,
 			values_equal<std::is_const<func_ret_t>::value, false, std::true_type>
			>::type;
// 	static_assert(!read_t::value || !iswriteable_func_t::value, "You are trying to write to the return value of a function which is not writable and/or not a reference");
 	static_assert((!ispointer_t::value) || !copy_t::value, "You are trying to explicitly copy the value of a (smart) pointer.");
	static_assert((!temp_t::value) || (!copy_t::value && read_t::value), "Internal error: Trying to form a copy or write edge to a temporary.");
	static_assert(shared_t::value || !(copy_t::value && !read_t::value), "Internal error: Trying to form  copy-write edge on a non-shared variable.");
	static_assert(!lockie_t::value || !copy_t::value, "Internal error: Trying to form a copy edge on a to be locked variable.");

 	using internal_base_t = typename std::disjunction<
			values_equal<func_t::value, true, func_ret_t>,
			values_equal<temp_t::value, true, const base_t>,
			values_equal<issmart_t::value, true, base_t>,
			values_equal<read_t::value, true, const base_t>,
			default_type<base_maybe_const_t>
			>::type;

 	using internal_base_ptr_t = typename std::disjunction<
 			values_equal<func_t::value, true, internal_base_t*>,
 			values_equal<temp_t::value, true, internal_base_t*>,
			values_equal<issmart_t::value, true, internal_base_t>,
			default_type<internal_base_t*>
 			>::type;

 	using internal_copy_ptr_t = typename std::disjunction<
			values_equal<func_t::value, true, func_ret_t*>,
 			values_equal<temp_t::value, true, base_t*>,
			values_equal<issmart_t::value, true, base_t>,
			default_type<base_t*>
 			>::type;

 	using holder_t = typename std::disjunction<
			values_equal<func_t::value || issmart_t::value || temp_t::value, true, typename std::remove_const<value_type_t>::type>,
			default_type<nullptr_t>
			>::type;

 	internal_base_ptr_t ptr_ = nullptr;
 	internal_copy_ptr_t copyPtr_ = nullptr;
	holder_t holder_ = nullptr;
public:
	using pass_t = typename std::disjunction<
			values_equal<func_t::value || temp_t::value || issmart_t::value, true, holder_t>,
			values_equal<ispointer_t::value, true, internal_base_ptr_t>,
			default_type<value_type_t&>
			>::type;

	using ref_t = typename std::disjunction<
			values_equal<(!temp_t::value && !func_t::value && issmart_t::value), true, internal_base_ptr_t&>,
			values_equal<ispointer_t::value, true, internal_base_ptr_t>,
			default_type<internal_base_t&>
			>::type;

	static Edge make(pass_t t) {
		Edge e;
		e.set(t);
		return e;
	}

	void set(pass_t t) {
		if constexpr(temp_t::value || issmart_t::value || func_t::value) {
			holder_ = t;
		}

		if constexpr(temp_t::value){
			ptr_ = holder_.get();
		} else if constexpr(func_t::value && read_t::value) {
			ptr_ = new internal_base_t();
		} else if constexpr(ispointer_t::value || issmart_t::value) {
			ptr_ = t;
		} else {
			ptr_ = &t;
		}


		if constexpr(copy_t::value) {
			copyPtr_ = new typename std::remove_pointer<internal_copy_ptr_t>::type();
		}
	}

	internal_base_ptr_t ptr() const {
		if constexpr(func_t::value) {
			if constexpr(!read_t::value) {
				ptr_ = &holder_->operator()();
			} else {
				*ptr_ = holder_->operator()();
			}
		}

		return ptr_;
	}

	size_t id() const {
		if constexpr(issmart_t::value && !temp_t::value && !func_t::value) {
			return reinterpret_cast<size_t>(ptr_.get());
		} else {
			return reinterpret_cast<size_t>(ptr_);
		}
	}


	ref_t ref() {
		if constexpr(!copy_t::value) {
			if constexpr(ispointer_t::value || (!temp_t::value && !func_t::value && issmart_t::value)) {
				return ptr_;
			} else {
				return *ptr();
			}
		} else {
			if constexpr(issmart_t::value || ispointer_t::value){
				if constexpr(shared_t::value) {
					Global::instance().safe_copy(*ptr()->get(), *copyPtr_.get());
					return copyPtr_;
				} else {
					Global::instance().copy(*ptr()->get(), *copyPtr_.get());
					return copyPtr_;
				}
			} else {
				if constexpr(shared_t::value) {
					Global::instance().safe_copy(*ptr(), *copyPtr_);
					return *copyPtr_;
				} else {
					Global::instance().copy(*ptr(), *copyPtr_);
					return *copyPtr_;
				}
			}
		}

		CV_Assert(false);
		if constexpr(!temp_t::value && !func_t::value && issmart_t::value) {
			return holder_;
		} else if constexpr(ispointer_t::value){
			return copyPtr_;
		} else {
			return *copyPtr_;
		}
	}

    void copyBack() {
    	if constexpr(!read_t::value && (copy_t::value || iswriteable_func_t::value)) {
    		if constexpr(shared_t::value) {
    			Global::instance().safe_copy(*copyPtr_, *ptr_);
    		} else {
    			Global::instance().copy(*copyPtr_, *ptr_);
    		}
    	}
    }

    std::mutex& getMutex() {
    	static_assert(lockie_t::value, "Internal Error: Trying to get mutex from a non-lockie edge");
    	//uses the no check variant because this should never fail due to previous checks.
    	return *Global::instance().getMutexPtr(*ptr(), true);
    }

    bool tryLock() {
    	return getMutex().try_lock();
    }

    bool unlock() {
    	getMutex().unlock();
    	return true;
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
	CV_EXPORTS virtual void perform(const bool& countContention = false) = 0;
	CV_EXPORTS virtual bool performPredicate(const bool& countContention = false) = 0;
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
auto filter_lockies(Tuple t, std::index_sequence<_Idx...>) {
	return std::tuple_cat(filter_lockie(std::get<_Idx>(t))...);
}

template<bool TcountContention = false, typename Tuple, size_t ... I>
auto make_scoped_lock_ptr(Tuple t, std::index_sequence<I ...>) {
	if constexpr(TcountContention) {
		size_t cnt = 0;
		(((std::get<I>(t).tryLock() && std::get<I>(t).unlock()) || ++cnt), ...);

		Global::instance().apply<size_t>(Global::Keys::LOCK_CONTENTION_CNT, [cnt](size_t& v){
			v += cnt;
			return v;
		});
	}
	using lock_t = std::scoped_lock<std::remove_reference_t<decltype(std::get<I>(t).getMutex())>...>;
	cv::Ptr<lock_t> ptr = new lock_t(std::get<I>(t).getMutex()...);
	return ptr;
}

template<bool TcountContention = false, typename Tuple>
auto make_scoped_lock_ptr(Tuple t) {
	static constexpr auto size = std::tuple_size<Tuple>::value;
	return make_scoped_lock_ptr<TcountContention>(t, std::make_index_sequence<size>{});
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

    template <bool TcountContention = false, typename _Fn, typename _Tuple, size_t... _Idx>
    void
    performImpl(_Fn&& __f, _Tuple&& __t, std::index_sequence<_Idx...> seq) {
    	if constexpr(hasLockies_) {
    		auto scopedLock = make_scoped_lock_ptr<TcountContention>(filter_lockies(__t, seq));
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

    template <bool TcountContention = false, typename _Fn, typename _Tuple, size_t... _Idx>
    constexpr decltype(auto)
    performImplRet(_Fn&& __f, _Tuple&& __t, std::index_sequence<_Idx...> seq) {
    	bool res = false;
    	if constexpr(hasLockies_) {
    		auto scopedLock = make_scoped_lock_ptr<TcountContention>(filter_lockies(__t, seq));

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

    virtual void perform(const bool& countContention = false) override {
        using _Indices= std::make_index_sequence<std::tuple_size<decltype(args_)>::value>;
        if(countContention) {
        	performImpl<true>(std::forward<F>(f),
  			       std::forward<decltype(args_)>(args_),
  			       _Indices{});
        } else {
			performImpl<false>(std::forward<F>(f),
				   std::forward<decltype(args_)>(args_),
				   _Indices{});
        }
        ran_ = true;
    }

    virtual bool performPredicate(const bool& countContention = false) override {
    	if constexpr(predicate_t::value) {
            using _Indices= std::make_index_sequence<std::tuple_size<decltype(args_)>::value>;
            bool res = false;
            if(countContention) {
            	res = performImplRet<true>(std::forward<F>(f),
      			       std::forward<decltype(args_)>(args_),
      			       _Indices{});
            } else {
            	res = performImplRet<false>(std::forward<F>(f),
      			       std::forward<decltype(args_)>(args_),
      			       _Indices{});
            }
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
