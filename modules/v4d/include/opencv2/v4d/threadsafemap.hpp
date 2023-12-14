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

//template<typename T>
//concept Hashable = requires(T a) {
//    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
//};
//
//template<typename T>
//concept Mappable = requires(T a) {
//    { std::any_cast<T>(std::any{}) } -> std::same_as<T>;
//};

template<typename K>
class ThreadSafeMap {
private:
    std::unordered_map<K, std::any> map;
    std::unordered_map<K, std::shared_mutex> mutexes;
    std::shared_mutex map_mutex;

public:
    template<typename V>
    void set(K key, V value) {
        std::unique_lock<std::shared_mutex> map_lock(map_mutex);

        if (map.find(key) == map.end()) {
            std::unique_lock<std::shared_mutex> key_lock(mutexes[key]);
            map[key] = value;
            mutexes[key];
        } else {
            std::unique_lock<std::shared_mutex> key_lock(mutexes[key]);
            map[key] = value;
        }
    }

    template<typename V>
    V get(K key) {
        std::shared_lock<std::shared_mutex> map_lock(map_mutex);
        if (map.find(key) == map.end()) {
            CV_Error(Error::StsError, "Key not found in map");
        }

        std::shared_lock<std::shared_mutex> key_lock(mutexes[key]);
        return std::any_cast<V>(map[key]);
    }

    template<typename V> V on(K key, std::function<V(V&)> func) {
        std::shared_lock<std::shared_mutex> map_lock(map_mutex);

        if (map.find(key) == map.end()) {
            CV_Error(Error::StsError, "Key not found in map");
        }

        std::unique_lock<std::shared_mutex> key_lock(mutexes[key]);

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

}
}

#endif // SRC_OPENCV_THREADSAFEMAP_HPP_
