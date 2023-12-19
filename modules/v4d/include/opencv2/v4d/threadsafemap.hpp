#ifndef SRC_OPENCV_THREADSAFEMAP_HPP_
#define SRC_OPENCV_THREADSAFEMAP_HPP_

#include <any>
#include <concepts>
#include <mutex>
#include <unordered_map>
#include <shared_mutex>

#include <any>
#include <concepts>
#include <mutex>
#include <unordered_map>
#include <shared_mutex>

namespace cv {
namespace v4d {

template<typename K>
class AnyHashMap {
private:
    std::unordered_map<K, std::any> map;
public:
    template<typename V>
    void set(K key, V value) {
    	map[key] = value;
    }

    bool has(K key) {
        return map.find(key) != map.end();
    }

    template<typename V>
    V get(K key) {
        return std::any_cast<V>(map[key]);
    }

    template<typename V> V on(K key, std::function<V(V&)> func) {
        if (!has(key)) {
            CV_Error(Error::StsError, "Key not found in map");
        }

        std::any& value = map[key];
        V ret = func(*std::any_cast<V>(&value));
        return ret;
    }

    // A method to get a pointer to the value for a given key
    // Note: This function is not thread-safe
    template<typename V>
    V* ptr(K key) {
        return std::any_cast<V>(&map[key]);
    }
};

template<typename K>
class ThreadSafeMap : public AnyHashMap<K> {
private:
    std::unordered_map<K, std::shared_mutex> mutexes;
    std::shared_mutex map_mutex;
    using parent_t = AnyHashMap<K>;
public:
    template<typename V>
    void set(K key, V value) {
        std::unique_lock<std::shared_mutex> map_lock(map_mutex);
        std::unique_lock<std::shared_mutex> key_lock(mutexes[key]);
        parent_t::set(key, value);
    }

    template<typename V>
    V get(K key) {
        std::shared_lock<std::shared_mutex> map_lock(map_mutex);
        if (!parent_t::has(key)) {
            CV_Error(Error::StsError, "Key not found in map");
        }

        std::shared_lock<std::shared_mutex> key_lock(mutexes[key]);
        return parent_t::template get<V>(key);
    }

    template<typename V> V on(K key, std::function<V(V&)> func) {
        std::shared_lock<std::shared_mutex> map_lock(map_mutex);

        if (!parent_t::has(key)) {
            CV_Error(Error::StsError, "Key not found in map");
        }

        std::unique_lock<std::shared_mutex> key_lock(mutexes[key]);
        return parent_t::template on<V>(key, func);
    }

    // A method to get a pointer to the value for a given key
    // Note: This function is not thread-safe
    template<typename V>
    V* ptr(K key) {
        return parent_t::template ptr<V>(key);
    }
};

}
}

#endif // SRC_OPENCV_THREADSAFEMAP_HPP_
