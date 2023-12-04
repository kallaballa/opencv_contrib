// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_


#include <GLFW/glfw3.h>
#include <queue>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <functional>
#include <map>
#include <cmath>
#include <cassert>

namespace cv {
namespace v4d {
namespace event {

typedef std::function<bool(GLFWwindow *, int, int, int, int)> KeyCallback;
typedef std::function<bool(GLFWwindow*, int, int, int)> MouseButtonCallback;
typedef std::function<bool(GLFWwindow*, double, double)> ScrollCallback;
typedef std::function<bool(GLFWwindow*, double, double)> CursorPosCallback;
typedef std::function<bool(GLFWwindow*, int, int)> WindowSizeCallback;
typedef std::function<bool(GLFWwindow*, int, int)> WindowPosCallback;
typedef std::function<bool(GLFWwindow*, int)> WindowFocusCallback;
typedef std::function<bool(GLFWwindow*)> WindowCloseCallback;

class Event {
public:
	enum class Class {
		KEYBOARD, MOUSE, JOYSTICK, WINDOW
	};

	virtual ~Event() = default;

	Class getClass() const {
		return class_;
	}

protected:
	Event(Class c) :
			class_(c) {
	}


private:
	Class class_;
};

class Keyboard: public Event {
public:
	enum class Key {
	    A,
	    B,
	    C,
	    D,
	    E,
	    F,
	    G,
	    H,
	    I,
	    J,
	    K,
	    L,
	    M,
	    N,
	    O,
	    P,
	    Q,
	    R,
	    S,
	    T,
	    U,
	    V,
	    W,
	    X,
	    Y,
	    Z,
	    N0,
	    N1,
	    N2,
	    N3,
	    N4,
	    N5,
	    N6,
	    N7,
	    N8,
	    N9,
	    SPACE,
	    ENTER,
	    BACKSPACE,
	    TAB,
	    ESCAPE,
	    UP,
	    DOWN,
	    LEFT,
	    RIGHT,
	    HOME,
	    END,
	    PAGE_UP,
	    PAGE_DOWN,
	    INSERT,
	    DELETE,
	    F1,
	    F2,
	    F3,
	    F4,
	    F5,
	    F6,
	    F7,
	    F8,
	    F9,
	    F10,
	    F11,
	    F12,
	    APOSTROPHE,
	    COMMA,
	    MINUS,
	    PERIOD,
	    SLASH,
	    SEMICOLON,
	    EQUAL,
	    LEFT_BRACKET,
	    BACKSLASH,
	    RIGHT_BRACKET,
	    GRAVE_ACCENT,
	    WORLD_1,
	    WORLD_2,
	    CAPS_LOCK,
	    SCROLL_LOCK,
	    NUM_LOCK,
	    PRINT_SCREEN,
	    PAUSE,
	    F13,
	    F14,
	    F15,
	    F16,
	    F17,
	    F18,
	    F19,
	    F20,
	    F21,
	    F22,
	    F23,
	    F24,
	    F25,
	    KP_0,
	    KP_1,
	    KP_2,
	    KP_3,
	    KP_4,
	    KP_5,
	    KP_6,
	    KP_7,
	    KP_8,
	    KP_9,
	    KP_DECIMAL,
	    KP_DIVIDE,
	    KP_MULTIPLY,
	    KP_SUBTRACT,
	    KP_ADD,
	    KP_ENTER,
	    KP_EQUAL,
	    LEFT_SHIFT,
	    LEFT_CONTROL,
	    LEFT_ALT,
	    LEFT_SUPER,
	    RIGHT_SHIFT,
	    RIGHT_CONTROL,
	    RIGHT_ALT,
	    RIGHT_SUPER,
	    MENU
	};

	enum class Type {
		PRESS, RELEASE, REPEAT, HOLD
	};

	Keyboard(Type type, Key key) :
			Event(Class::KEYBOARD), key_(key), type_(type) {
	}

	Key key() const {
		return key_;
	}

	Type type() const {
		return type_;
	}

	bool is(Key a) const {
		return key() == a;
	}

	bool is(Type t) const {
		return type() == t;
	}
private:
	Key key_;
	Type type_;
};



template<typename Tpoint>
class Mouse: public Event {
public:
	enum class Button {
		NONE,
		LEFT,
		RIGHT,
		MIDDLE,
		N4,
		N5,
		N6,
		N7,
		N8
	};

	enum class Type {
		PRESS,
		RELEASE,
		MOVE,
		SCROLL,
		DRAG,
		HOVER_ENTER,
		HOVER_EXIT,
		DOUBLE_CLICK
	};

	Mouse(Type type, Tpoint position) :
			Event(Class::MOUSE), button_(Button::NONE), type_(type), position_(
					position) {
	}

	Mouse(Type type, Tpoint position, Tpoint data) :
			Event(Class::MOUSE), button_(Button::NONE), type_(type), position_(
					position), data_(data) {
	}

	Mouse(Type type, Button button, Tpoint position) :
			Event(Class::MOUSE), button_(button), type_(type), position_(
					position) {
	}

	Mouse(Type type, Button button, Tpoint position, Tpoint data) :
			Event(Class::MOUSE), button_(button), type_(type), position_(
					position), data_(data) {
	}

	Button button() const {
		return button_;
	}

	Type type() const {
		return type_;
	}

	Tpoint position() const {
		return position_;
	}

	Tpoint data() const {
		return data_;
	}

	bool is(Button a) const {
		return button() == a;
	}

	bool is(Type t) const {
		return type() == t;
	}
private:
	Button button_;
	Type type_;
	Tpoint position_;
	Tpoint data_;
};


class Joystick: public Event {
public:
	enum class Type {
		PRESS, RELEASE, MOVE
	};

	enum class Button {
		NONE,
		A,
		B,
		X,
		Y,
		LB,
		RB,
		BACK,
		START,
		GUIDE,
		LEFT_THUMB,
		RIGHT_THUMB,
		DPAD_UP,
		DPAD_RIGHT,
		DPAD_DOWN,
		DPAD_LEFT
	};

	enum class Axis {
		NONE,
		LEFT_X,
		LEFT_Y,
		RIGHT_X,
		RIGHT_Y,
		LEFT_TRIGGER,
		RIGHT_TRIGGER
	};

	Joystick(Type type, int joystick, Button button) :
			Event(Class::JOYSTICK), type_(type), joystick_(joystick), button_(button), state_(
					type == Type::PRESS), axis_(Axis::NONE), init_(0), value_(
					0), delta_(0) {
	}

	Joystick(int joystick, Axis axis, float init, float value, float delta) :
			Event(Class::JOYSTICK), type_(Type::MOVE), joystick_(joystick), button_(
					Button::NONE), state_(false), axis_(axis), init_(init), value_(value), delta_(delta) {
	}

	bool is(Button a) const {
		return button() == a;
	}

	bool is(Axis a) const {
		return axis() == a;
	}

	bool is(Type t) const {
		return type() == t;
	}

	Type type() const {
		return type_;
	}

	int joystick() const {
		return joystick_;
	}

	Button button() const {
		return button_;
	}

	bool state() const {
		return state_;
	}

	Axis axis() const {
		return axis_;
	}

	float init() const {
		return init_;
	}

	float value() const {
		return value_;
	}

	float delta() const {
		return delta_;
	}

	float abs() const {
		return value_ - init_;
	}

	bool active() const {
		if(is(Axis::LEFT_TRIGGER) || is(Axis::RIGHT_TRIGGER)) {
			return value() > -1;
		} else {
			return std::fabs(abs()) > 0.1;
		}
	}

private:
	Type type_;
	int joystick_;
	Button button_;
	bool state_;
	Axis axis_;
	float init_;
	float value_;
	float delta_;
};

template<typename Tpoint>
class Window: public Event {
public:
	enum class Type {
		RESIZE, MOVE, FOCUS, UNFOCUS, CLOSE
	};

	Window(Type type, Tpoint data) :
			Event(Class::WINDOW), type_(type), data_(data) {
	}

	Window(Type type) :
			Event(Class::WINDOW), type_(type) {
	}

	Type type() const {
		return type_;
	}

	Tpoint data() const {
		return data_;
	}

private:
	Type type_;
	Tpoint data_;
};


namespace detail {


inline static std::mutex queue_access_mtx;
inline static std::map<std::thread::id, size_t> thread_id_map;

constexpr float AXIS_NO_VALUE = std::numeric_limits<float>::max();
inline static float prev_axis_value[6] = {AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE};
inline static float init_axis_value[6] = {AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE};
inline GLFWwindow* main_window = nullptr;

static void set_main_glfw_window(GLFWwindow* win) {
	assert(win);
	main_window = win;
}

static GLFWwindow* get_main_glfw_window() {
	assert(main_window);
	return main_window;
}

constexpr Keyboard::Key v4d_key(int glfw_key) {
	switch (glfw_key) {
	case GLFW_KEY_A:
		return Keyboard::Key::A;
	case GLFW_KEY_B:
		return Keyboard::Key::B;
	case GLFW_KEY_C:
		return Keyboard::Key::C;
	case GLFW_KEY_D:
		return Keyboard::Key::D;
	case GLFW_KEY_E:
		return Keyboard::Key::E;
	case GLFW_KEY_F:
		return Keyboard::Key::F;
	case GLFW_KEY_G:
		return Keyboard::Key::G;
	case GLFW_KEY_H:
		return Keyboard::Key::H;
	case GLFW_KEY_I:
		return Keyboard::Key::I;
	case GLFW_KEY_J:
		return Keyboard::Key::J;
	case GLFW_KEY_K:
		return Keyboard::Key::K;
	case GLFW_KEY_L:
		return Keyboard::Key::L;
	case GLFW_KEY_M:
		return Keyboard::Key::M;
	case GLFW_KEY_N:
		return Keyboard::Key::N;
	case GLFW_KEY_O:
		return Keyboard::Key::O;
	case GLFW_KEY_P:
		return Keyboard::Key::P;
	case GLFW_KEY_Q:
		return Keyboard::Key::Q;
	case GLFW_KEY_R:
		return Keyboard::Key::R;
	case GLFW_KEY_S:
		return Keyboard::Key::S;
	case GLFW_KEY_T:
		return Keyboard::Key::T;
	case GLFW_KEY_U:
		return Keyboard::Key::U;
	case GLFW_KEY_V:
		return Keyboard::Key::V;
	case GLFW_KEY_W:
		return Keyboard::Key::W;
	case GLFW_KEY_X:
		return Keyboard::Key::X;
	case GLFW_KEY_Y:
		return Keyboard::Key::Y;
	case GLFW_KEY_Z:
		return Keyboard::Key::Z;
	case GLFW_KEY_0:
		return Keyboard::Key::N0;
	case GLFW_KEY_1:
		return Keyboard::Key::N1;
	case GLFW_KEY_2:
		return Keyboard::Key::N2;
	case GLFW_KEY_3:
		return Keyboard::Key::N3;
	case GLFW_KEY_4:
		return Keyboard::Key::N4;
	case GLFW_KEY_5:
		return Keyboard::Key::N5;
	case GLFW_KEY_6:
		return Keyboard::Key::N6;
	case GLFW_KEY_7:
		return Keyboard::Key::N7;
	case GLFW_KEY_8:
		return Keyboard::Key::N8;
	case GLFW_KEY_9:
		return Keyboard::Key::N9;
	case GLFW_KEY_SPACE:
		return Keyboard::Key::SPACE;
	case GLFW_KEY_ENTER:
		return Keyboard::Key::ENTER;
	case GLFW_KEY_BACKSPACE:
		return Keyboard::Key::BACKSPACE;
	case GLFW_KEY_TAB:
		return Keyboard::Key::TAB;
	case GLFW_KEY_ESCAPE:
		return Keyboard::Key::ESCAPE;
	case GLFW_KEY_UP:
		return Keyboard::Key::UP;
	case GLFW_KEY_DOWN:
		return Keyboard::Key::DOWN;
	case GLFW_KEY_LEFT:
		return Keyboard::Key::LEFT;
	case GLFW_KEY_RIGHT:
		return Keyboard::Key::RIGHT;
	case GLFW_KEY_HOME:
		return Keyboard::Key::HOME;
	case GLFW_KEY_END:
		return Keyboard::Key::END;
	case GLFW_KEY_PAGE_UP:
		return Keyboard::Key::PAGE_UP;
	case GLFW_KEY_PAGE_DOWN:
		return Keyboard::Key::PAGE_DOWN;
	case GLFW_KEY_INSERT:
		return Keyboard::Key::INSERT;
	case GLFW_KEY_DELETE:
		return Keyboard::Key::DELETE;
	case GLFW_KEY_F1:
		return Keyboard::Key::F1;
	case GLFW_KEY_F2:
		return Keyboard::Key::F2;
	case GLFW_KEY_F3:
		return Keyboard::Key::F3;
	case GLFW_KEY_F4:
		return Keyboard::Key::F4;
	case GLFW_KEY_F5:
		return Keyboard::Key::F5;
	case GLFW_KEY_F6:
		return Keyboard::Key::F6;
	case GLFW_KEY_F7:
		return Keyboard::Key::F7;
	case GLFW_KEY_F8:
		return Keyboard::Key::F8;
	case GLFW_KEY_F9:
		return Keyboard::Key::F9;
	case GLFW_KEY_F10:
		return Keyboard::Key::F10;
	case GLFW_KEY_F11:
		return Keyboard::Key::F11;
	case GLFW_KEY_F12:
		return Keyboard::Key::F12;
    case GLFW_KEY_APOSTROPHE:
        return Keyboard::Key::APOSTROPHE;
    case GLFW_KEY_COMMA:
        return Keyboard::Key::COMMA;
    case GLFW_KEY_MINUS:
        return Keyboard::Key::MINUS;
    case GLFW_KEY_PERIOD:
        return Keyboard::Key::PERIOD;
    case GLFW_KEY_SLASH:
        return Keyboard::Key::SLASH;
    case GLFW_KEY_SEMICOLON:
        return Keyboard::Key::SEMICOLON;
    case GLFW_KEY_EQUAL:
        return Keyboard::Key::EQUAL;
    case GLFW_KEY_LEFT_BRACKET:
        return Keyboard::Key::LEFT_BRACKET;
    case GLFW_KEY_BACKSLASH:
        return Keyboard::Key::BACKSLASH;
    case GLFW_KEY_RIGHT_BRACKET:
        return Keyboard::Key::RIGHT_BRACKET;
    case GLFW_KEY_GRAVE_ACCENT:
        return Keyboard::Key::GRAVE_ACCENT;
    case GLFW_KEY_WORLD_1:
        return Keyboard::Key::WORLD_1;
    case GLFW_KEY_WORLD_2:
        return Keyboard::Key::WORLD_2;
    case GLFW_KEY_CAPS_LOCK:
        return Keyboard::Key::CAPS_LOCK;
    case GLFW_KEY_SCROLL_LOCK:
        return Keyboard::Key::SCROLL_LOCK;
    case GLFW_KEY_NUM_LOCK:
        return Keyboard::Key::NUM_LOCK;
    case GLFW_KEY_PRINT_SCREEN:
        return Keyboard::Key::PRINT_SCREEN;
    case GLFW_KEY_PAUSE:
        return Keyboard::Key::PAUSE;
    case GLFW_KEY_KP_0:
        return Keyboard::Key::KP_0;
    case GLFW_KEY_KP_1:
        return Keyboard::Key::KP_1;
    case GLFW_KEY_KP_2:
        return Keyboard::Key::KP_2;
    case GLFW_KEY_KP_3:
        return Keyboard::Key::KP_3;
    case GLFW_KEY_KP_4:
        return Keyboard::Key::KP_4;
    case GLFW_KEY_KP_5:
        return Keyboard::Key::KP_5;
    case GLFW_KEY_KP_6:
        return Keyboard::Key::KP_6;
    case GLFW_KEY_KP_7:
        return Keyboard::Key::KP_7;
    case GLFW_KEY_KP_8:
        return Keyboard::Key::KP_8;
    case GLFW_KEY_KP_9:
        return Keyboard::Key::KP_9;
    case GLFW_KEY_KP_DECIMAL:
        return Keyboard::Key::KP_DECIMAL;
    case GLFW_KEY_KP_DIVIDE:
        return Keyboard::Key::KP_DIVIDE;
    case GLFW_KEY_KP_MULTIPLY:
        return Keyboard::Key::KP_MULTIPLY;
    case GLFW_KEY_KP_SUBTRACT:
        return Keyboard::Key::KP_SUBTRACT;
    case GLFW_KEY_KP_ADD:
        return Keyboard::Key::KP_ADD;
    case GLFW_KEY_KP_ENTER:
        return Keyboard::Key::KP_ENTER;
    case GLFW_KEY_KP_EQUAL:
        return Keyboard::Key::KP_EQUAL;
    case GLFW_KEY_LEFT_SHIFT:
        return Keyboard::Key::LEFT_SHIFT;
    case GLFW_KEY_LEFT_CONTROL:
        return Keyboard::Key::LEFT_CONTROL;
    case GLFW_KEY_LEFT_ALT:
        return Keyboard::Key::LEFT_ALT;
    case GLFW_KEY_LEFT_SUPER:
        return Keyboard::Key::LEFT_SUPER;
    case GLFW_KEY_RIGHT_SHIFT:
        return Keyboard::Key::RIGHT_SHIFT;
    case GLFW_KEY_RIGHT_CONTROL:
        return Keyboard::Key::RIGHT_CONTROL;
    case GLFW_KEY_RIGHT_ALT:
        return Keyboard::Key::RIGHT_ALT;
    case GLFW_KEY_RIGHT_SUPER:
        return Keyboard::Key::RIGHT_SUPER;
    case GLFW_KEY_MENU:
        return Keyboard::Key::MENU;
	default:
		throw std::runtime_error(
				"Invalid key: " + std::to_string(glfw_key)
						+ ". Please ensure the key is within the valid range.");
	}
}

template<typename Tpoint>
constexpr typename Mouse<Tpoint>::Button v4d_mouse_button(int glfw_button) {
	switch (glfw_button) {
	case GLFW_MOUSE_BUTTON_LEFT:
		return Mouse<Tpoint>::Button::LEFT;
	case GLFW_MOUSE_BUTTON_RIGHT:
		return Mouse<Tpoint>::Button::RIGHT;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		return Mouse<Tpoint>::Button::MIDDLE;
	case GLFW_MOUSE_BUTTON_4:
		return Mouse<Tpoint>::Button::N4;
	case GLFW_MOUSE_BUTTON_5:
		return Mouse<Tpoint>::Button::N5;
	case GLFW_MOUSE_BUTTON_6:
		return Mouse<Tpoint>::Button::N6;
	case GLFW_MOUSE_BUTTON_7:
		return Mouse<Tpoint>::Button::N7;
	case GLFW_MOUSE_BUTTON_8:
		return Mouse<Tpoint>::Button::N8;
	default:
		throw std::runtime_error(
				"Invalid mouse button: " + std::to_string(glfw_button)
						+ ". Please ensure the button is within the valid range.");
	}
}

constexpr Joystick::Button v4d_joystick_button(int glfw_button) {
	switch (glfw_button) {
	case GLFW_GAMEPAD_BUTTON_A:
		return Joystick::Button::A;
	case GLFW_GAMEPAD_BUTTON_B:
		return Joystick::Button::B;
	case GLFW_GAMEPAD_BUTTON_X:
		return Joystick::Button::X;
	case GLFW_GAMEPAD_BUTTON_Y:
		return Joystick::Button::Y;
	case GLFW_GAMEPAD_BUTTON_LEFT_BUMPER:
		return Joystick::Button::LB;
	case GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER:
		return Joystick::Button::RB;
	case GLFW_GAMEPAD_BUTTON_BACK:
		return Joystick::Button::BACK;
	case GLFW_GAMEPAD_BUTTON_START:
		return Joystick::Button::START;
	case GLFW_GAMEPAD_BUTTON_GUIDE:
		return Joystick::Button::GUIDE;
	case GLFW_GAMEPAD_BUTTON_LEFT_THUMB:
		return Joystick::Button::LEFT_THUMB;
	case GLFW_GAMEPAD_BUTTON_RIGHT_THUMB:
		return Joystick::Button::RIGHT_THUMB;
	case GLFW_GAMEPAD_BUTTON_DPAD_UP:
		return Joystick::Button::DPAD_UP;
	case GLFW_GAMEPAD_BUTTON_DPAD_RIGHT:
		return Joystick::Button::DPAD_RIGHT;
	case GLFW_GAMEPAD_BUTTON_DPAD_DOWN:
		return Joystick::Button::DPAD_DOWN;
	case GLFW_GAMEPAD_BUTTON_DPAD_LEFT:
		return Joystick::Button::DPAD_LEFT;
	default:
		throw std::runtime_error(
				"Invalid joystick button: " + std::to_string(glfw_button)
						+ ". Please ensure the button is within the valid range.");
	}
}

constexpr Joystick::Axis v4d_joystick_axis(int glfw_axis) {
	switch (glfw_axis) {
	case GLFW_GAMEPAD_AXIS_LEFT_X:
		return Joystick::Axis::LEFT_X;
	case GLFW_GAMEPAD_AXIS_LEFT_Y:
		return Joystick::Axis::LEFT_Y;
	case GLFW_GAMEPAD_AXIS_RIGHT_X:
		return Joystick::Axis::RIGHT_X;
	case GLFW_GAMEPAD_AXIS_RIGHT_Y:
		return Joystick::Axis::RIGHT_Y;
	case GLFW_GAMEPAD_AXIS_LEFT_TRIGGER:
		return Joystick::Axis::LEFT_TRIGGER;
	case GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER:
		return Joystick::Axis::RIGHT_TRIGGER;
	default:
		throw std::runtime_error(
				"Invalid joystick axis: " + std::to_string(glfw_axis)
						+ ". Please ensure the axis is within the valid range.");
	}
}

constexpr Keyboard::Type v4d_keyboard_event_type(int glfw_state) {
	Keyboard::Type type;
	switch (glfw_state) {
	case GLFW_PRESS:
		type = Keyboard::Type::PRESS;
		break;
	case GLFW_RELEASE:
		type = Keyboard::Type::RELEASE;
		break;
	case GLFW_REPEAT:
		type = Keyboard::Type::REPEAT;
		break;
	default:
		throw std::runtime_error(
				"Invalid state: " + std::to_string(glfw_state)
						+ ". Please ensure the state is within the valid range.");
	}
	return type;
}

template<typename Tpoint>
constexpr typename Mouse<Tpoint>::Type v4d_mouse_event_type(int glfw_state) {
	typename Mouse<Tpoint>::Type type;
	switch (glfw_state) {
	case GLFW_PRESS:
		type = Mouse<Tpoint>::Type::PRESS;
		break;
	case GLFW_RELEASE:
		type = Mouse<Tpoint>::Type::RELEASE;
		break;
	default:
		throw std::runtime_error(
				"Invalid state: " + std::to_string(glfw_state)
						+ ". Please ensure the action is within the valid range.");
	}
	return type;
}

constexpr Joystick::Type v4d_joystick_event_type(int glfw_state) {
	Joystick::Type type;
	switch (glfw_state) {
	case GLFW_PRESS:
		type = Joystick::Type::PRESS;
		break;
	case GLFW_RELEASE:
		type = Joystick::Type::RELEASE;
		break;
	default:
		throw std::runtime_error(
				"Invalid state: " + std::to_string(glfw_state)
						+ ". Please ensure the state is within the valid range.");
	}
	return type;
}


class EventQueue: public std::deque<std::shared_ptr<Event>> {
	std::mutex mutex_;
	size_t capacity_ = 100;
	typedef std::deque<std::shared_ptr<Event>> parent_t;
public:
	EventQueue() {
		assert(capacity_ > 0);
	}

	void push(const std::shared_ptr<Event>& event) {
		std::unique_lock < std::mutex > lock(mutex_);
		if(size() == capacity_) {
			parent_t::pop_front();
		}
		parent_t::push_back(event);
	}


	template <typename Tevent>
	bool has() {
		std::unique_lock<std::mutex> lock(mutex_);
		for(size_t i = 0; i < size(); ++i) {
			std::shared_ptr<Tevent> found = std::dynamic_pointer_cast<Tevent>(parent_t::operator[](i));
			if(found) {
				return true;
			}
		}

		return false;
	}

	template <typename Tevent>
	bool has(std::function<bool(std::shared_ptr<Tevent>)> fn) {
		std::unique_lock<std::mutex> lock(mutex_);
		for(size_t i = 0; i < size(); ++i) {
			std::shared_ptr<Tevent> found = std::dynamic_pointer_cast<Tevent>(parent_t::operator[](i));
			if(found && fn(found)) {
				return true;
			}
		}

		return false;
	}

	template <typename Tevent>
	void get(std::vector<std::shared_ptr<Tevent>>& result) {
		std::unique_lock<std::mutex> lock(mutex_);
		for(size_t i = 0; i < size(); ++i) {
			std::shared_ptr<Tevent> found = std::dynamic_pointer_cast<Tevent>(parent_t::operator[](i));
			if(found) {
				result.push_back(found);
			}
		}
	}

	template <typename Tevent>
	void get(std::vector<std::shared_ptr<Tevent>>& result, std::function<bool(std::shared_ptr<Tevent>)> fn) {
		std::unique_lock<std::mutex> lock(mutex_);
		for(size_t i = 0; i < size(); ++i) {
			std::shared_ptr<Tevent> found = std::dynamic_pointer_cast<Tevent>(parent_t::operator[](i));
			if(found && fn(found)) {
				result.push_back(found);
			}
		}
	}


	template <typename Tevent>
	bool has(const typename Tevent::Type& t) {
		std::unique_lock<std::mutex> lock(mutex_);
		for(size_t i = 0; i < size(); ++i) {
			std::shared_ptr<Tevent> found = std::dynamic_pointer_cast<Tevent>(parent_t::operator[](i));
			if(found && found->type() == t) {
				return true;
			}
		}

		return false;
	}

	template <typename Tevent>
	bool has(const typename Tevent::Type& t, std::function<bool(std::shared_ptr<Tevent>)> fn) {
		std::unique_lock<std::mutex> lock(mutex_);
		for(size_t i = 0; i < size(); ++i) {
			std::shared_ptr<Tevent> found = std::dynamic_pointer_cast<Tevent>(parent_t::operator[](i));
			if(found && found->type() == t && fn(found)) {
				return true;
			}
		}

		return false;
	}

	template <typename Tevent>
	void get(const typename Tevent::Type& t, std::vector<std::shared_ptr<Tevent>>& result) {
		std::unique_lock<std::mutex> lock(mutex_);
		for(size_t i = 0; i < size(); ++i) {
			std::shared_ptr<Tevent> found = std::dynamic_pointer_cast<Tevent>(parent_t::operator[](i));
			if(found && found->type() == t) {
				result.push_back(found);
			}
		}
	}

	template <typename Tevent>
	void get(const typename Tevent::Type& t, std::vector<std::shared_ptr<Tevent>>& result, std::function<bool(std::shared_ptr<Tevent>)> fn) {
		std::unique_lock<std::mutex> lock(mutex_);
		for(size_t i = 0; i < size(); ++i) {
			std::shared_ptr<Tevent> found = std::dynamic_pointer_cast<Tevent>(parent_t::operator[](i));
			if(found && found->type() == t &&  fn(found)) {
				result.push_back(found);
			}
		}
	}

	bool empty() {
		std::unique_lock < std::mutex > lock(mutex_);
		return parent_t::empty();
	}

	void clear() {
		std::unique_lock < std::mutex > lock(mutex_);
		parent_t empty_queue;
		std::swap(*this, empty_queue);
	}
};

inline static std::vector<EventQueue*> queue_vector;

static EventQueue& queue() {
	std::lock_guard<std::mutex> guard(queue_access_mtx);
	auto it = thread_id_map.find(std::this_thread::get_id());
	size_t id = 0;
	if(it != thread_id_map.end()) {
		id = (*it).second;
	} else if(!thread_id_map.empty()) {
		id = (*std::prev(thread_id_map.end())).second + 1;
	}

	std::vector<EventQueue*>& qs = queue_vector;
	if(qs.size() <= id) {
		qs.push_back(new EventQueue());
	}
	return *qs[id];
}

static void push(const std::shared_ptr<Event>& event) {
	std::lock_guard<std::mutex> guard(queue_access_mtx);
	auto& qs = queue_vector;

	for(size_t i = 0; i < qs.size(); ++i) {
		qs[i]->push(event);
	}
}

static GLFWgamepadstate last_state;
static void poll_joystick_events() {
	for(size_t i = 0; i >= GLFW_GAMEPAD_BUTTON_LAST; ++i) {
		last_state.buttons[i] = GLFW_RELEASE;
	}
	for (int jid = 0; jid <= GLFW_JOYSTICK_LAST; jid++) {
		if (glfwJoystickPresent(jid) && glfwJoystickIsGamepad(jid)) {
			GLFWgamepadstate state;
			if (glfwGetGamepadState(jid, &state)) {
				for (int button = 0; button <= GLFW_GAMEPAD_BUTTON_LAST;
						button++) {
					Joystick::Button jsButton = v4d_joystick_button(
							button);
					bool bounced = (last_state.buttons[button] == state.buttons[button]) && ((state.buttons[button] == GLFW_RELEASE));
					if(!bounced) {
						Joystick::Type type = v4d_joystick_event_type(state.buttons[button]);
						std::shared_ptr< Joystick> event = std::make_shared<Joystick>(type, jid, jsButton);
						push(event);
						last_state = state;
					}
				}
				for (int axis = 0; axis <= GLFW_GAMEPAD_AXIS_LAST; axis++) {
					Joystick::Axis v4d_axis = v4d_joystick_axis(axis);
					float& init = init_axis_value[axis];
					float& prev = prev_axis_value[axis];

					float value = state.axes[axis];
					float delta = 0;
					if(init == AXIS_NO_VALUE) {
						init = value;
					}

					if(prev != AXIS_NO_VALUE) {
						delta = value - prev;
					}
					prev_axis_value[axis] = value;
					std::shared_ptr<Joystick> event = std::make_shared<Joystick>(jid, v4d_axis, init, value, delta);
					if(event->active())
						push(event);
				}
			}
		}
	}
}

inline static KeyCallback keyboardCallback;
inline static MouseButtonCallback mouseButtonCallback;
inline static ScrollCallback scrollCallback;
inline static CursorPosCallback cursorPosCallback;
inline static WindowSizeCallback windowSizeCallback;
inline static WindowPosCallback windowPosCallback;
inline static WindowFocusCallback windowFocusCallback;
inline static WindowCloseCallback windowCloseCallback;
}

template<typename Tpoint>
static void init(
		KeyCallback keyboardCallback = KeyCallback(),
		MouseButtonCallback mouseButtonCallback = MouseButtonCallback(),
		ScrollCallback scrollCallback = ScrollCallback(),
		CursorPosCallback cursorPosCallback = CursorPosCallback(),
		WindowSizeCallback windowSizeCallback = WindowSizeCallback(),
		WindowPosCallback windowPosCallback = WindowPosCallback(),
		WindowFocusCallback windowFocusCallback = WindowFocusCallback(),
		WindowCloseCallback windowCloseCallback = WindowCloseCallback()) {
	GLFWwindow* win = glfwGetCurrentContext();
	assert(win);
	detail::set_main_glfw_window(win);
	detail::keyboardCallback = keyboardCallback;
	detail::mouseButtonCallback = mouseButtonCallback;
	detail::scrollCallback = scrollCallback;
	detail::cursorPosCallback = cursorPosCallback;
	detail::windowSizeCallback = windowSizeCallback;
	detail::windowPosCallback = windowPosCallback;
	detail::windowFocusCallback = windowFocusCallback;
	detail::windowCloseCallback = windowCloseCallback;

	std::vector<detail::EventQueue*> *queues = new std::vector<detail::EventQueue*>();
	glfwSetKeyCallback(win,
			[](GLFWwindow *window, int key, int scancode, int action,
					int mods) {
				if (!detail::keyboardCallback || detail::keyboardCallback(window, key, scancode, action, mods)) {
					if (key != GLFW_KEY_UNKNOWN) {
						Keyboard::Key v4d_key = detail::v4d_key(key);
						Keyboard::Type type = detail::v4d_keyboard_event_type(action);
						std::shared_ptr<Keyboard> event = std::make_shared<Keyboard>(type, v4d_key);
						detail::push(event);
					}
				}
			});

	glfwSetMouseButtonCallback(win,
			[](GLFWwindow *window, int button, int action, int mods) {
				if (!detail::mouseButtonCallback || detail::mouseButtonCallback(window, button, action, mods)) {
					if (button != GLFW_MOUSE_BUTTON_LAST) {
						typename Mouse<Tpoint>::Button mouseButton = detail::v4d_mouse_button<Tpoint>(button);
						typename Mouse<Tpoint>::Type type = detail::v4d_mouse_event_type<Tpoint>(action);

						double x, y;
						glfwGetCursorPos(window, &x, &y);
						Tpoint position(x, y);
						std::shared_ptr<Mouse<Tpoint>> event = std::make_shared<Mouse<Tpoint>>(type, mouseButton, position);
						detail::push(event);
					}
				}
			});
	glfwSetScrollCallback(win,
			[](GLFWwindow *window, double xoffset, double yoffset) {
			if (!detail::scrollCallback || detail::scrollCallback(window, xoffset, yoffset)) {
					Tpoint offset(xoffset, yoffset);
					double x, y;
					glfwGetCursorPos(window, &x, &y);
					Tpoint position(x, y);
					std::shared_ptr<Mouse<Tpoint>> event = std::make_shared<Mouse<Tpoint>>(Mouse<Tpoint>::Type::SCROLL, position, offset);
					detail::push(event);
				}
			});

	glfwSetCursorPosCallback(win,
			[](GLFWwindow *window, double xpos, double ypos) {
				static Tpoint prevMousePos;

				if (!detail::cursorPosCallback || detail::cursorPosCallback(window, xpos, ypos)) {
					double x, y;
					glfwGetCursorPos(window, &x, &y);
					Tpoint position(x, y);
					bool pressed = false;
					for (int button = 0; button <= GLFW_MOUSE_BUTTON_LAST;
							button++) {
						if (glfwGetMouseButton(window, button) == GLFW_PRESS) {
							pressed = true;
							break;
						}
					}
					if (pressed) {
						Tpoint delta = position - prevMousePos;
						for (int button = 0; button <= GLFW_MOUSE_BUTTON_LAST;
								button++) {
							if (glfwGetMouseButton(window, button) == GLFW_PRESS) {
								typename Mouse<Tpoint>::Button v4d_button = detail::v4d_mouse_button<Tpoint>(button);
								std::shared_ptr<Mouse<Tpoint>> event = std::make_shared <Mouse<Tpoint>>(Mouse<Tpoint>::Type::DRAG, v4d_button, position, delta);
								detail::push(event);
							}
						}
						prevMousePos = position;
					} else {
						std::shared_ptr<Mouse<Tpoint>> event = std::make_shared<Mouse<Tpoint>> (Mouse<Tpoint>::Type::MOVE, position);
						detail::push(event);
					}
				}
			});

	glfwSetWindowSizeCallback(win,
			[](GLFWwindow *window, int width, int height) {
				if(!detail::windowSizeCallback || detail::windowSizeCallback(window, width, height)) {
					Tpoint sz(width, height);
					std::shared_ptr<Window<Tpoint>> event = std::make_shared<Window<Tpoint>>(Window<Tpoint>::Type::RESIZE, sz);
					detail::push(event);
				}
			});

	glfwSetWindowPosCallback(win,
			[](GLFWwindow *window, int xpos, int ypos) {
				if(!detail::windowPosCallback || detail::windowPosCallback(window, xpos, ypos)) {
					Tpoint position(xpos, ypos);
					std::shared_ptr<Window<Tpoint>> event = std::make_shared<Window<Tpoint>>(Window<Tpoint>::Type::MOVE, position);
					detail::push(event);
				}
			});

	glfwSetWindowFocusCallback(win, [](GLFWwindow *window, int focused) {
		if(!detail::windowFocusCallback || detail::windowFocusCallback(window, focused)) {
			typename Window<Tpoint>::Type type = focused
					? Window<Tpoint>::Type::FOCUS
							: Window<Tpoint>::Type::UNFOCUS;

			std::shared_ptr<Window<Tpoint>> event = std::make_shared<Window<Tpoint>>(type);
			detail::push(event);
		}
	});

	glfwSetWindowCloseCallback(win, [](GLFWwindow *window) {
		if(!detail::windowCloseCallback || detail::windowCloseCallback(window)) {
			std::shared_ptr<Window<Tpoint>> event = std::make_shared<Window<Tpoint>>(Window<Tpoint>::Type::CLOSE);
			detail::push(event);
		}
	});
}


template<typename Tevent>
static bool has(){
	return detail::queue().has<Tevent>();
}

template<typename Tevent>
static std::vector<std::shared_ptr<Tevent>> get(){
	std::vector<std::shared_ptr<Tevent>> result;
	detail::queue().get<Tevent>(result);
	return result;
}

template<typename Tevent>
static bool has(const typename Tevent::Type& t){
	return detail::queue().has(t);
}

template<typename Tevent>
static std::vector<std::shared_ptr<Tevent>> get(const typename Tevent::Type& t){
	std::vector<std::shared_ptr<Tevent>> result;
	detail::queue().get(t, result);
	return result;
}

static std::vector<std::shared_ptr<Keyboard>> get(const Keyboard::Type& t, const Keyboard::Key& k){
	std::vector<std::shared_ptr<Keyboard>> result;
	detail::queue().get<Keyboard>(t, result, [k](std::shared_ptr<Keyboard> ev){ return ev->key() == k; });
	return result;
}

template<typename Tpoint>
static std::vector<std::shared_ptr<Mouse<Tpoint>>> get(const typename Mouse<Tpoint>::Type& t, const typename Mouse<Tpoint>::Button& b){
	std::vector<std::shared_ptr<Mouse<Tpoint>>> result;
	detail::queue().get<Mouse<Tpoint>>(t, result, [b](std::shared_ptr<Mouse<Tpoint>> ev){ return ev->button() == b; });
	return result;
}

static std::vector<std::shared_ptr<Joystick>> get(const Joystick::Type& t, const Joystick::Button& b){
	std::vector<std::shared_ptr<Joystick>> result;
	detail::queue().get<Joystick>(t, result, [b](std::shared_ptr<Joystick> ev){ return ev->button() == b; });
	return result;
}

static std::vector<std::shared_ptr<Joystick>> get(const Joystick::Type& t, const Joystick::Axis& a){
	std::vector<std::shared_ptr<Joystick>> result;
	detail::queue().get<Joystick>(t, result, [a](std::shared_ptr<Joystick> ev){ return ev->axis() == a; });
	return result;
}

static bool has(const Keyboard::Type& t, const Keyboard::Key& k){
	return detail::queue().has<Keyboard>(t, [k](std::shared_ptr<Keyboard> ev){ return ev->key() == k; });
}

template<typename Tpoint>
static bool has(const typename Mouse<Tpoint>::Type& t, const typename Mouse<Tpoint>::Button& b){
	return detail::queue().has<Mouse<Tpoint>>(t, [b](std::shared_ptr<Mouse<Tpoint>> ev){ return ev->button() == b; });
}

static bool has(const Joystick::Type& t, const Joystick::Button& b){
	return detail::queue().has<Joystick>(t, [b](std::shared_ptr<Joystick> ev){ return ev->button() == b; });
}

static bool has(const Joystick::Type& t, const Joystick::Axis& a){
	return detail::queue().has<Joystick>(t, [a](std::shared_ptr<Joystick> ev){ return ev->axis() == a; });
}

template<typename T>
static bool has(const typename T::Type& t, std::function<bool(const T&)> fn){
	return detail::queue().has<T>(t, [fn](std::shared_ptr<T> ev){ return fn(*ev.get()); });
}


template<typename T>
static std::vector<std::shared_ptr<T>> get(const typename T::Type& t, std::function<bool(const T&)> fn){
	return detail::queue().get<T>(t, [fn](std::shared_ptr<T> ev){ return fn(*ev.get()); });
}

static void poll() {
	if(detail::get_main_glfw_window()) {
		detail::queue().clear();
		glfwPollEvents();
		detail::poll_joystick_events();
	}
}

}
}
}
#endif  // MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_
