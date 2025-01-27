// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_UTIL_HPP_
#define SRC_OPENCV_V4D_UTIL_HPP_

#include "source.hpp"
#include "sink.hpp"
#include "detail/framebuffercontext.hpp"
#include <filesystem>
#include <string>
#include <iostream>
#include <array>

#include "threadsafeanymap.hpp"
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#endif
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <unistd.h>
#include <mutex>
#include <functional>
#include <iostream>
#include <cmath>
#include <thread>
#include <latch>
#include <deque>

using std::cout;
using std::endl;

namespace cv {
namespace v4d {

#define _OLM_(r,c,f, ...) static_cast<r (c::*)(__VA_ARGS__)>(f)
#define _OLMC_(r,c,f, ...) static_cast<r (c::*)(__VA_ARGS__) const>(f)

#define _OL_(r,f, ...) static_cast<r (*)(__VA_ARGS__)>(f)
#define _OLC_(r,f, ...) static_cast<r (*)(__VA_ARGS__) const>(f)

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

template <typename, typename = void>
struct has_call_operator_t : std::false_type {};

template <typename T>
struct has_call_operator_t<T, std::void_t<decltype(&T::operator())>> : std::is_same<std::true_type, std::true_type>
{};

template <typename, typename = void>
struct has_return_type_t : std::false_type {};

template <typename T>
struct has_return_type_t<T, std::void_t<decltype(&T::return_type)>> : std::is_same<std::true_type, std::true_type>
{};

template < template <typename...> class Template, typename T >
struct is_specialization_of : std::false_type {};

template < template <typename...> class Template, typename... Args >
struct is_specialization_of< Template, Template<Args...> > : std::true_type {};

template<typename T>
struct is_callable : public std::disjunction<std::disjunction<
									has_call_operator_t<T>,
									std::is_pointer<T>>,
									std::is_member_function_pointer<T>,
									std::is_function<T>> {
};

//https://stackoverflow.com/a/27885283/1884837



template<class T>
struct function_traits : function_traits<decltype(&T::operator())> {
};


template<>
struct function_traits<std::false_type> : std::false_type {
	using result_type = std::false_type;
};

// partial specialization for function type
template<class R, class... Args>
struct function_traits<R(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};

// partial specialization for function pointer
template<class R, class... Args>
struct function_traits<R (*)(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};

// partial specialization for std::function
template<class R, class... Args>
struct function_traits<std::function<R(Args...)>> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};

// partial specialization for pointer-to-member-function (i.e., operator()'s)
template<class T, class R, class... Args>
struct function_traits<R (T::*)(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};

template<class T, class R, class... Args>
struct function_traits<R (T::*)(Args...) const> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
	static const bool value = true;
};


template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
struct fun_ptr_helper
{
public:
    typedef std::function<_Res(_ArgTypes...)> function_type;

    static void bind(function_type&& f)
    { instance().fn_.swap(f); }

    static void bind(const function_type& f)
    { instance().fn_=f; }

    static _Res invoke(_ArgTypes... args)
    { return instance().fn_(args...); }

    typedef decltype(&fun_ptr_helper::invoke) pointer_type;
    static pointer_type ptr()
    { return &invoke; }

private:
    static fun_ptr_helper& instance()
    {
        static fun_ptr_helper inst_;
        return inst_;
    }

    fun_ptr_helper() {}

    function_type fn_;
};

template <typename, typename = void>
struct element_t : std::false_type {
	using type = std::false_type ;
};

template <typename Tptr>
struct element_t<Tptr, std::void_t<decltype(&Tptr::get)>> : std::is_same<std::true_type, std::true_type>
{
	using type = std::remove_pointer_t<typename Tptr::element_type>;
};

template <typename, typename = void>
struct return_t : std::false_type {
	using type = std::false_type ;
};

template <typename Tfn>
struct return_t<Tfn, std::void_t<decltype(&Tfn::operator())>> : std::is_same<std::true_type, std::true_type>
{
	using type = typename function_traits<Tfn>::result_type;
};

//template <typename T>
//struct CallableTraits {
//	using return_t = typename detail::return_t<T>::type;
//	using data_t = std::false_type;
//	using object_t = std::false_type;
//	using args_t = std::false_type;
//};

template <typename T>
struct CallableTraits {
	using return_type_t = typename detail::return_t<T>::type;
	using member_t = std::false_type;
	using object_t = std::false_type;
	using args_t = std::false_type;
};

template <typename Return, typename Object>
struct CallableTraits<Return Object::*>
{
	using return_type_t = Return;
    using member_t = std::true_type;
    using object_t = Object;
    using args_t = std::false_type;
};

template <typename Return, typename Object, typename... Args>
struct CallableTraits<Return (Object::*)(Args...)>
{
    using return_type_t = Return;
    using member_t = std::true_type;
    using object_t = Object;
    using args_t = std::tuple<Args...>;
};

template <typename Return, typename... Args>
struct CallableTraits<Return (*)(Args...)>
{
    using return_type_t = Return;
    using member_t = std::false_type;
    using object_t = std::false_type;
    using args_t = std::tuple<Args...>;
};

template <typename Return, typename... Args>
struct CallableTraits<Return(Args...)>
{
    using member_t = std::false_type;
    using return_type_t =  Return;
    using object_t = std::false_type;
    using args_t = std::tuple<Args...>;
};

template <size_t offset, size_t len, class tuple, size_t ... idx>
auto sub_tuple(tuple&& t, std::index_sequence<idx...>) {
    static_assert(offset + len <= std::tuple_size<typename std::remove_reference<tuple>::type>::value, "sub tuple is out of bounds!");
    return std::make_tuple(std::get<idx + offset>(t)...);
}

template <size_t offset, size_t len, class tuple>
auto sub_tuple(tuple&& t) {
	return sub_tuple<offset, len, tuple>(std::forward<tuple>(t), std::make_index_sequence<len>());
}


template<typename Tfn, typename Tret = typename CallableTraits<Tfn>::return_t, typename ... Args>
struct AssignableMemData {
	Tfn fn_;
	std::tuple<Args...> args_;
	AssignableMemData(Tfn fn, Args ... args) : fn_(fn), args_(args...) {

	}

	void operator=(Tret v) {
		std::get<0>(args_).*fn_ = v;
	}

	operator Tret() {
		return fn_(std::get<0>(args_));
	}
};

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
typename fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::pointer_type
get_fn_ptr(const std::function<_Res(_ArgTypes...)>& f)
{
    fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::bind(f);
    return fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::ptr();
}

template<typename T>
std::function<typename std::enable_if<std::is_function<T>::value, T>::type>
make_function(T *t)
{
    return {t};
}

//https://stackoverflow.com/a/33047781/1884837
class Lambda {
    template<typename T>
    static const void* fn(const void* new_fn = nullptr) {
        CV_Assert(new_fn);
    	return new_fn;
    }
	template<typename Tret, typename T>
    static Tret lambda_ptr_exec() {
        return (Tret) (*(T*)fn<T>());
    }
public:
    template<typename Tret = void, typename Tfp = Tret(*)(), typename T>
    static Tfp ptr(T& t) {
        fn<T>(&t);
        return (Tfp) lambda_ptr_exec<Tret, T>;
    }
};


template<bool read, typename Tfn, typename ... Args>
struct edgefun_t {
	edgefun_t(Tfn fn, Args ... args) {}
	using return_type_t = typename CallableTraits<Tfn>::return_type_t;
	static_assert(!std::is_same<return_type_t, std::false_type>::value, "Invalid callable passed");
	using type = typename std::disjunction<
				default_type<std::function<return_type_t(typename Args::ref_t ...)>>
				>::type;
};


}
using std::string;
class V4D;
class Plan;

CV_EXPORTS size_t cnz(const cv::UMat& m);

inline uint64_t get_epoch_nanos() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

class SharedVariables {
	std::mutex sharedVarsMtx_;
	std::mutex safeVarsMtx_;
	std::map<size_t, std::pair<size_t, cv::Ptr<std::mutex>>> sharedVars_;
	std::map<size_t, std::pair<size_t, cv::Ptr<std::mutex>>> safeVars_;
	typedef typename std::map<size_t, std::pair<size_t, cv::Ptr<std::mutex>>>::iterator SharedVarsIter;

	template<typename T>
	std::pair<size_t, size_t> findSharedParent(const T& shared) {
		off_t varOffset = reinterpret_cast<size_t>(&shared);
		off_t registeredOffset = 0;
		off_t registeredSize = 0;
		for(const auto& p : sharedVars_) {
			registeredOffset = p.first;
			if(varOffset > registeredOffset) {
				registeredSize = p.second.first;
				if(varOffset < (registeredOffset + registeredSize)) {
					return {registeredOffset, registeredSize};
				}
			}
		}
		return {0,0};
	}

public:
	template<typename Tplan, typename Tvar>
	static bool isPlanMember(Tplan& plan, Tvar& var) {
		const char* planPtr = reinterpret_cast<const char*>(&plan);
		const char* varPtr = reinterpret_cast<const char*>(&var);
		off_t parentOffset = plan.getParentOffset();
		off_t parentActualSize = plan.getParentActualTypeSize();
		off_t actualTypeSize = plan.getActualTypeSize();
		off_t varOffset = off_t (varPtr);
		off_t planOffset = off_t (planPtr);
		off_t planSize = sizeof(Tplan);

		CV_Assert((parentOffset == 0  && parentActualSize == 0) || (parentOffset > 0 && parentActualSize > 0));

		off_t parentLowerBound = parentOffset;
		off_t parentUpperBound = parentOffset + parentActualSize;
		off_t lowerBound = planOffset;
		off_t upperBound = planOffset + actualTypeSize;

		if(! ((varOffset >= lowerBound && varOffset <= upperBound) || (parentOffset > 0 && varOffset >= parentLowerBound && varOffset <= parentUpperBound))) {
			return false;
		}
		return true;
	}


	template<typename T>
	void makeSharedVar(const T& candidate) {
		{
			std::lock_guard<std::mutex> guard(safeVarsMtx_);
			CV_Assert(safeVars_.find(reinterpret_cast<size_t>(&candidate)) == safeVars_.end());
		}

		std::lock_guard<std::mutex> guard(sharedVarsMtx_);
		if(sharedVars_.find(reinterpret_cast<size_t>(&candidate)) != sharedVars_.end()) {
			return;
		} else  {
			auto parent = findSharedParent(candidate);
			if(parent.first != 0) {
				auto it = sharedVars_.find(parent.first);
				CV_Assert(it != sharedVars_.end());
				sharedVars_.insert({reinterpret_cast<size_t>(&candidate), std::make_pair(sizeof(T), (*it).second.second)});
			} else {
				sharedVars_.insert({reinterpret_cast<size_t>(&candidate), std::make_pair(sizeof(T), cv::makePtr<std::mutex>())});
			}
		}
	}

	template<typename Tplan, typename T, bool Tcheck = true>
	bool checkShared(Tplan& plan, const T& candidate) {
		{
			std::lock_guard<std::mutex> guard(safeVarsMtx_);
			if(safeVars_.find(reinterpret_cast<size_t>(&candidate)) != safeVars_.end()) {
				return false;
			}
		}

		std::lock_guard<std::mutex> guard(sharedVarsMtx_);
		if(sharedVars_.find(reinterpret_cast<size_t>(&candidate)) != sharedVars_.end()) {
			return true;
		} else if(!isPlanMember(plan, candidate)) {
			auto parent = findSharedParent(candidate);
			if(parent.first != 0) {
				auto it = sharedVars_.find(parent.first);
				CV_Assert(it != sharedVars_.end());
				sharedVars_.insert({reinterpret_cast<size_t>(&candidate), std::make_pair(sizeof(T), (*it).second.second)});
			} else {
				sharedVars_.insert({reinterpret_cast<size_t>(&candidate), std::make_pair(sizeof(T), cv::makePtr<std::mutex>())});
			}

			return true;
		}
		return false;
	}

	template<typename T>
	void registerSafe(const T& safe) {
		std::lock_guard<std::mutex> guard(safeVarsMtx_);
		auto it = safeVars_.find(reinterpret_cast<size_t>(&safe));
		if(it == safeVars_.end()) {
			safeVars_.insert({reinterpret_cast<size_t>(&safe), std::make_pair(sizeof(T), cv::makePtr<std::mutex>())});
		}
	}

	template<typename T>
	void safe_copy(const T& from, T& to) {
		std::mutex* mtx = getMutexPtr(from, false);
		mtx = mtx == nullptr ? getMutexPtr(to, false) : mtx;
		if(mtx == nullptr)
			throw std::runtime_error("Internal error: Trying to safe copy non-shared variables.");

		std::lock_guard<std::mutex> guard(*mtx);
		to = copy(from);
	}

	template<typename T>
	static void copy(const T& from, T& to) {
			to = copy_construct(from);
	}

	static void copy(const cv::UMat& from, cv::UMat& to) {
		if(from.empty())
			return;
		to = from.clone();
	}

	template<typename T>
	static T safe_copy(const T& from) {
		T to;
		safe_copy(from, to);
		return to;
	}

	template<typename T>
	static T copy(const T& from) {
		T to;
		copy(from, to);
		return to;
	}

	template<typename T>
	static T copy_construct(const T& t) {
		return t;
	}


	template<typename T>
	std::mutex* getMutexPtr(const T& shared, bool check = true) {
		SharedVarsIter it, end;
		cv::Ptr<std::mutex> mtx = nullptr;
			std::lock_guard<std::mutex> guard(sharedVarsMtx_);
			it = sharedVars_.find(reinterpret_cast<size_t>(&shared));
			end = sharedVars_.end();
			if(it != end) {
				mtx = (*it).second.second;
			}

			if(check && !mtx)
				throw std::runtime_error("You are trying to lock a non-shared variable");

			return mtx.get();
	}

	template<typename T>
	void lock(const T& shared) {
		getMutexPtr(shared)->lock();
	}

	template<typename T>
	void unlock(const T& shared) {
		getMutexPtr(shared)->unlock();
	}

	template<typename T>
	bool tryLock(const T& shared) {
		return getMutexPtr(shared)->try_lock();
	}
};

CV_EXPORTS class Global : public SharedVariables {
public:
	struct Keys {
		enum Enum {
			FRAME_COUNT,
			RUN_COUNT,
			START_TIME,
			FPS,
			WORKERS_READY,
			WORKERS_STARTED,
			FRAMEBUFFER_INDEX,
			LOCKING,
			DISPLAY_READY,
			LOCK_CONTENTION_CNT,
			LOCK_CONTENTION_RATE,
			PLAN_CNT,
			WORKER_CNT
		};
	};
private:
	CV_EXPORTS ThreadSafeAnyMap<Keys::Enum> map_;
	CV_EXPORTS static Global* instance_;
	std::mutex threadIDMtx_;
	const std::thread::id defaultThreadID_;
	std::thread::id mainThreadID_;
	bool isFirstRun_ = true;
	std::set<string> once_;
	std::mutex nodeLockMtx_;
	std::map<string, std::pair<std::thread::id, cv::Ptr<std::mutex>>> nodeLockMap_;

	CV_EXPORTS cv::Ptr<std::mutex> getNodeLockInternal(const string& name, const bool owned = true) {
		auto it = nodeLockMap_.find(name);
		if(owned) {
			if(it == nodeLockMap_.end()) {
				auto mtxPtr = cv::makePtr<std::mutex>();
				nodeLockMap_[name] = {std::this_thread::get_id(), mtxPtr};
				return mtxPtr;
			} else {
				auto entry = *it;
				if(entry.second.first == std::this_thread::get_id()) {
					return entry.second.second;
				}
			}
		} else {
			if(it != nodeLockMap_.end()) {
				auto entry = *it;
				if(entry.second.first != std::this_thread::get_id()) {
					return entry.second.second;
				}
			}
		}
		return nullptr;
	}

	CV_EXPORTS bool invalidateNodeLockInternal(const string& name) {
		auto it = nodeLockMap_.find(name);
		if(it != nodeLockMap_.end()) {
			auto& entry = *it;
			entry.second.second = nullptr;
			return true;
		}
		return false;
	}

	Global() {
		create<false, size_t>(Keys::FRAME_COUNT, 0);
		create<false, size_t>(Keys::RUN_COUNT, 0);
		create<false, uint64_t>(Keys::START_TIME, get_epoch_nanos());
		create<false, double>(Keys::FPS, 0);
		create<false, size_t>(Keys::WORKERS_READY, 0);
		create<false, size_t>(Keys::WORKERS_STARTED, 0);
		create<false, size_t>(Keys::FRAMEBUFFER_INDEX, 0);
		create<false, bool>(Keys::LOCKING, false);
		create<false, bool>(Keys::DISPLAY_READY, false);
		create<false, size_t>(Keys::LOCK_CONTENTION_CNT, 0);
		create<false, double>(Keys::LOCK_CONTENTION_RATE, 0.0);
		create<false, size_t>(Keys::PLAN_CNT, 0);
		create<false, size_t>(Keys::WORKER_CNT, 0);
	}
public:
	template <typename V> const V& get(Keys::Enum k) {
		return map_.get<V>(k);
	}

	template <typename V> void set(Keys::Enum k, V v) {
		map_.set(k, v);
	}

	template <bool Tread, typename V> void create(Keys::Enum k, V v, const std::function<void(const V& val)>& cb = std::function<void(const V& val)>()) {
		map_.create<Tread>(k, v, cb);
	}

	template <typename V> V apply(Keys::Enum k, std::function<V(V&)> f) {
		return map_.apply(k, f);
	}

	CV_EXPORTS void setMainID(const std::thread::id& id) {
		std::lock_guard<std::mutex> lock(threadIDMtx_);
		mainThreadID_ = id;
    }

	CV_EXPORTS bool isMain() {
		std::lock_guard<std::mutex> lock(threadIDMtx_);
		return (mainThreadID_ == defaultThreadID_ || mainThreadID_ == std::this_thread::get_id());
	}

	CV_EXPORTS bool isFirstRun() {
		static std::mutex mtx;
		std::lock_guard<std::mutex> lock(mtx);
    	bool f = isFirstRun_;
    	isFirstRun_ = false;
		return f;
    }

	CV_EXPORTS static Global& instance() {
		static std::mutex mtx;
		std::lock_guard guard(mtx);
		if(instance_ == nullptr) {
			instance_ = new Global();
		}
		return *instance_;
	}

	CV_EXPORTS cv::Ptr<std::mutex> tryGetNodeLock(const string& name) {
		std::lock_guard guard(nodeLockMtx_);
		return getNodeLockInternal(name, false);
	}

	CV_EXPORTS bool lockNode(const string& name) {
		std::lock_guard guard(nodeLockMtx_);
		auto lock = getNodeLockInternal(name);
		if(lock) {
			lock->lock();
			return true;
		} else {
			return false;
		}
	}

	CV_EXPORTS bool tryUnlockNode(const string& name) {
		std::lock_guard guard(nodeLockMtx_);
		auto lock = getNodeLockInternal(name);
		if(lock) {
			lock->unlock();
			invalidateNodeLockInternal(name);
			return true;
		} else {
			return false;
		}
	}

	CV_EXPORTS size_t countNodeLocks() {
		std::lock_guard guard(nodeLockMtx_);
		size_t cnt = 0;
		for(auto entry : nodeLockMap_) {
			if(entry.second.first == std::this_thread::get_id() && entry.second.second) {
				++cnt;
			}
		}
		return cnt;
	}

	CV_EXPORTS bool once(string name) {
	    static std::mutex mtx;
		std::lock_guard<std::mutex> lock(mtx);
		string stem = name.substr(0, name.find_last_of("-"));

		auto it = once_.find(stem);
		if(it != once_.end()) {
			return false;
		} else {
			once_.insert(stem);
			return true;
		}
	}
};


class RunState {
	CV_EXPORTS static thread_local RunState* instance_;
public:
	struct Keys {
		enum Enum {
			WORKER_INDEX,
		};
	};
private:
	ThreadSafeAnyMap<Keys::Enum> map_;
public:
	RunState() {
		create<false, size_t>(Keys::WORKER_INDEX, 0);
	}

	static RunState& instance() {
		static std::mutex mtx;
		std::lock_guard guard(mtx);
		if(instance_ == nullptr) {
			instance_ = new RunState();
		}
		return *instance_;
	}

	template <typename V> const V& get(Keys::Enum k) {
		return map_.get<V>(k);
	}

	template <typename V> void set(Keys::Enum k, V v) {
		map_.set(k, v);
	}

	template <bool Tread, typename V> void create(Keys::Enum k, V v, const std::function<void(const V& val)>& cb = std::function<void(const V& val)>()) {
		map_.create<Tread>(k, v, cb);
	}

	template <typename V> V apply(Keys::Enum k, std::function<V(V&)> f) {
		return map_.apply(k, f);
	}
};



CV_EXPORTS void copy_cross(const cv::UMat& src, cv::UMat& dst);


template<typename T>
constexpr int matrix_depth() {
	if constexpr(std::is_same_v<T, uchar>) {
		return CV_8U;
	} else if constexpr(std::is_same_v<T, short>){
		return CV_16S;
	} else if constexpr(std::is_same_v<T, ushort>){
		return CV_16U;
	} else if constexpr(std::is_same_v<T, int>){
		return CV_32S;
	} else if constexpr(std::is_same_v<T, float>){
		return CV_32F;
	} else if constexpr(std::is_same_v<T, double>){
		return CV_64F;
	} else if constexpr(true) {
#if !defined(__APPLE__)
		static_assert(false, "Type not supported for operation.");
#endif
		return 0;
	}
}

template<bool Tround> double doRound(double t) {
	if constexpr(Tround) {
		return std::round(t);
	} else {
		return t;
	}
}

/*!
 * Convenience function to color convert from all Vec_ and Scalar_ variants
 * @param src The color to convert
 * @param code The color converions code
 * @return The color converted scalar
 */
template<int Tcode = -1, typename Tsrc, typename Tdst = Vec<typename Tsrc::value_type, Tsrc::channels>, bool Tround = std::is_floating_point_v<typename Tsrc::value_type> && std::is_integral_v<typename Tdst::value_type>>
Tdst convert_pix(const Tsrc &src, double alpha = 1.0, double beta = 0.0) {
	constexpr int srcCn = Tsrc::channels;
	constexpr int dstCn = Tdst::channels;

	using srcv_t = typename Tsrc::value_type;
	using dstv_t = typename Tdst::value_type;
	using src_internal_t = Vec<srcv_t, srcCn>;
	using intermediate_t = Vec<srcv_t, dstCn>;
	using dst_internal_t = Vec<dstv_t, dstCn>;
	static_assert((srcCn == 3 || srcCn == 4) && (dstCn == 3 || dstCn == 4), "Only 3 or 4 (src/dst) channels supported");
	constexpr int srcType = CV_MAKETYPE(
			matrix_depth<typename src_internal_t::value_type>(),
			src_internal_t::channels);
	constexpr int intermediateType = CV_MAKETYPE(
			matrix_depth<typename src_internal_t::value_type>(), dstCn);
	constexpr int dstType = CV_MAKETYPE(
			matrix_depth<typename dst_internal_t::value_type>(), dstCn);

	std::array<src_internal_t, 1> srcArr;
	if constexpr (srcCn == 3) {
		srcArr[0] = src_internal_t(src[0], src[1], src[2]);
	} else {
		srcArr[0] = src_internal_t(src[0], src[1], src[2], src[3]);
	}

	cv::Mat intermediateMat(cv::Size(1, 1), intermediateType);

	if constexpr (dstCn == srcCn) {
		intermediateMat = srcArr[0];
	} else if constexpr (srcCn == 3) {
		intermediateMat = intermediate_t(srcArr[0][0], srcArr[0][1],
				srcArr[0][2]);
	} else if constexpr (srcCn == 4) {
		intermediateMat = intermediate_t(srcArr[0][0], srcArr[0][1],
				srcArr[0][2], srcArr[0][3]);
	}

	if constexpr (Tcode >= 0) {
		cvtColor(srcArr, intermediateMat, Tcode);
	}

	std::array<dst_internal_t, 1> dstArr;
	if constexpr (!std::is_same<srcv_t, dstv_t>::value) {
		//will just copy if types match
		if constexpr (dstCn == srcCn) {
			intermediateMat.convertTo(dstArr, dstType);
		} else if constexpr (dstCn == 3) {
			cvtColor(intermediateMat, intermediateMat, cv::COLOR_BGRA2BGR);
			intermediateMat.convertTo(dstArr, dstType);
		} else if constexpr (dstCn == 4) {
			cvtColor(intermediateMat, intermediateMat, cv::COLOR_BGR2BGRA);
			intermediateMat.convertTo(dstArr, dstType);
		}
	} else {
		if constexpr (dstCn == srcCn) {
			dstArr[0] = intermediateMat.at<src_internal_t>(0.0);
		} else if constexpr (dstCn == 3) {
			auto im = intermediateMat.at<src_internal_t>(0.0);
			dstArr[0] = dst_internal_t(im[0], im[1], im[2]);
		} else if constexpr (dstCn == 4) {
			auto im = intermediateMat.at<src_internal_t>(0.0);
			if (intermediateMat.depth() == CV_32F
					|| intermediateMat.depth() == CV_64F) {
				dstArr[0] = dst_internal_t(im[0], im[1], im[2], 255.0);
			} else {
				dstv_t a = std::numeric_limits<dstv_t>::max();
				dstArr[0] = dst_internal_t(im[0], im[1], im[2], a);
			}
		}
	}

	Tdst dst;

	if constexpr (dstCn == 3) {
		dst = Tdst(dstArr[0][0], dstArr[0][1], dstArr[0][2]);
	} else if constexpr (dstCn == 4) {
		dst = Tdst(dstArr[0][0], dstArr[0][1], dstArr[0][2], dstArr[0][3]);
	}

	if (alpha != 1.0) {
		if constexpr (dstCn == 3) {
			dst[0] = doRound<Tround>(dst[0] * alpha);
			dst[1] = doRound<Tround>(dst[1] * alpha);
			dst[2] = doRound<Tround>(dst[2] * alpha);
		} else if constexpr (dstCn == 4) {
			dst[0] = doRound<Tround>(dst[0] * alpha);
			dst[1] = doRound<Tround>(dst[1] * alpha);
			dst[2] = doRound<Tround>(dst[2] * alpha);
			dst[3] = doRound<Tround>(dst[3] * alpha);
		}
	}

	if (beta != 0.0) {
		if constexpr (dstCn == 3) {
			dst[0] = doRound<Tround>(dst[0] + beta);
			dst[1] = doRound<Tround>(dst[1] + beta);
			dst[2] = doRound<Tround>(dst[2] + beta);
		} else if constexpr (dstCn == 4) {
			dst[0] = doRound<Tround>(dst[0] + beta);
			dst[1] = doRound<Tround>(dst[1] + beta);
			dst[2] = doRound<Tround>(dst[2] + beta);
			dst[3] = doRound<Tround>(dst[3] + beta);
		}
	}
	return dst;
}

inline double seconds() {
	return cv::getTickCount() / cv::getTickFrequency();
}
/*!
 * Convenience function to check for OpenGL errors. Should only be used via the macro #GL_CHECK.
 * @param file The file path of the error.
 * @param line The file line of the error.
 * @param expression The expression that failed.
 */
CV_EXPORTS void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression);
/*!
 * Convenience macro to check for OpenGL errors.
 */
#ifndef NDEBUG
#define GL_CHECK(expr)                            \
    expr;                                        \
    cv::v4d::gl_check_error(__FILE__, __LINE__, #expr);
#else
#define GL_CHECK(expr)                            \
    expr;
#endif
CV_EXPORTS void init_shaders(unsigned int handles[3], const string vShader, const string fShader, const string outputAttributeName);
CV_EXPORTS void init_fragment_shader(unsigned int handles[2], const char* fshader);
/*!
 * Returns the OpenGL vendor string
 * @return a string object with the OpenGL vendor information
 */
CV_EXPORTS std::string get_gl_vendor();
/*!
 * Returns the OpenGL Version information.
 * @return a string object with the OpenGL version information
 */
CV_EXPORTS std::string get_gl_info();
/*!
 * Returns the OpenCL Version information.
 * @return a string object with the OpenCL version information
 */
CV_EXPORTS std::string get_cl_info();
/*!
 * Determines if Intel VAAPI is supported
 * @return true if it is supported
 */
CV_EXPORTS bool is_intel_va_supported();
/*!
 * Determines if cl_khr_gl_sharing is supported
 * @return true if it is supported
 */
CV_EXPORTS bool is_clgl_sharing_supported();
/*!
 * Tells the application if it's alright to keep on running.
 * Note: If you use this mechanism signal handlers are installed
 * @return true if the program should keep on running
 */
CV_EXPORTS bool keep_running();

CV_EXPORTS void request_finish();

CV_EXPORTS float aspect_preserving_scale(const cv::Size& scaled, const cv::Size& unscaled);
CV_EXPORTS void resize_preserving_aspect_ratio(const cv::UMat& src, cv::UMat& output, const cv::Size& dstSize, const cv::Scalar& bgcolor = {0,0,0,255});

}
}

#endif /* SRC_OPENCV_V4D_UTIL_HPP_ */
