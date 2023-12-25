#ifndef SRC_OPENCV_THREADSAFEMAP_HPP_
#define SRC_OPENCV_THREADSAFEMAP_HPP_

#include <any>
#include <vector>
#include <mutex>
#include <shared_mutex>

namespace cv {
namespace v4d {

class Value : public std::any {
public:
	std::function<void(Value& val)> callback_;
	const bool read_;

	Value(const bool& read) : read_(read) {
	}

    any& operator=(const any& rhs) {
    	return std::any::operator =(rhs);
    }

    /// Store a copy of @p __rhs as the contained object.
    template <typename T>
    any& operator=(T&& __rhs) {
    	std::any::operator =(any(std::forward<T>(__rhs)));
    	return *this;
    }
};

template<typename K>
class AnyPropertyMap {
	static_assert(std::is_enum<K>::value);
private:
    std::vector<Value> properties_;

    void check_write(K key) {
    	CV_Assert(properties_.size() > key);
    	if(properties_[key].read_) {
    		CV_Error(cv::Error::StsBadArg, "You are trying to set a read only property");
    	}
    }

public:
    template<bool Tread, typename V>
    void create(K key, const V& value, std::function<void(const V& val)> cb) {
    	CV_Assert(properties_.size() == key);
    	CV_Assert(!Tread || (Tread && !cb));
    	if constexpr(Tread) {
    		Value val(Tread);
    		val.callback_ = [](const Value& val){};
    		val = value;
    		properties_.emplace_back(val);
    	} else {
    		if(!cb)
    			cb = [](const V& v){};

    		properties_.emplace_back(Value(Tread));
    		properties_[key] = value;
    		Value& val = properties_[key];
    		val.callback_ = [cb](const Value& val){ cb(std::any_cast<V>(val)); };
    	}
    }

    template<typename V>
    void set(K key, const V& value, bool fire = true) {
    	check_write(key);
    	V& p = *std::any_cast<V>(&properties_[key]);
    	V oldVal = p;
    	p = value;
    	if(value != oldVal)
    		properties_[key].callback_(properties_[key]);
    }

    template<typename V>
    const V& get(K key) const {
        return *ptr<V>(key);
    }

    template<typename V> auto apply(K key, std::function<V(V&)> func) {
    	check_write(key);
    	V ret = func(*std::any_cast<V>(&properties_[key]));
        return ret;
    }

    // A method to get a pointer to the value for a given key
    template<typename V>
    const V* ptr(K key) const {
    	CV_Assert(properties_.size() > key);
    	const V* p = std::any_cast<V>(&properties_[key]);
    	CV_Assert(p != nullptr);
    	return p;
    }
};

template<typename K>
class ThreadSafeAnyMap : public AnyPropertyMap<K> {
private:
	AnyPropertyMap<K> mutexes_;
    cv::Ptr<std::shared_mutex> mapMutexPtr_ = cv::makePtr<std::shared_mutex>();
    using parent_t = AnyPropertyMap<K>;
public:
    template<bool Tread, typename V>
   void create(K key, const V& value, const std::function<void(const V& val)>& cb = std::function<void(const V& val)>()) {
    	std::function<void(const cv::Ptr<std::shared_mutex>&)> emptyFn;
    	mutexes_.template create<Tread>(key, cv::makePtr<std::shared_mutex>(), emptyFn);
    	parent_t::template create<Tread>(key, value, cb);
    }

    template<typename V>
    void set(K key, const V& value, bool fire = true) {
        std::unique_lock<std::shared_mutex> map_lock(*mapMutexPtr_);
        std::unique_lock<std::shared_mutex> key_lock(*mutexes_.template get<cv::Ptr<std::shared_mutex>>(key));
        parent_t::set(key, value, fire);
    }

    template<typename V>
    const V& get(K key) const {
        std::shared_lock<std::shared_mutex> map_lock(*mapMutexPtr_);
        std::shared_lock<std::shared_mutex> key_lock(*mutexes_.template get<cv::Ptr<std::shared_mutex>>(key));
        return parent_t::template get<V>(key);
    }

    template<typename V> V apply(K key, std::function<V(V&)> func) {
        std::shared_lock<std::shared_mutex> map_lock(*mapMutexPtr_);
        std::unique_lock<std::shared_mutex> key_lock(*mutexes_.template get<cv::Ptr<std::shared_mutex>>(key));
        return parent_t::template apply<V>(key, func);
    }

    // A method to get a pointer to the value for a given key
    // Note: This function is not thread-safe
    template<typename V>
    V* ptr(K key) const {
        return parent_t::template ptr<V>(key);
    }
};

}
}

#endif // SRC_OPENCV_THREADSAFEMAP_HPP_
