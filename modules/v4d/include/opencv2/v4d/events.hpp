// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_


#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>
#include <queue>
#include <vector>
#include <map>
#include <variant>
#include <memory>
#include <thread>
#include <mutex>
#include <iostream>
#include <functional>
#include <type_traits>
#include "imgui_impl_glfw.h"

namespace cv {
namespace v4d {
namespace event {

class Event {
public:
	enum class Class {
		KEYBOARD, MOUSE, JOYSTICK, WINDOW
	};

	virtual ~Event() = default;

	Class getClass() const {
		return type_;
	}

protected:
	Event(Class type) :
			type_(type) {
	}

private:
	Class type_;
};

class Keyboard: public Event {
public:
	enum class Key {
	    KEY_A,
	    KEY_B,
	    KEY_C,
	    KEY_D,
	    KEY_E,
	    KEY_F,
	    KEY_G,
	    KEY_H,
	    KEY_I,
	    KEY_J,
	    KEY_K,
	    KEY_L,
	    KEY_M,
	    KEY_N,
	    KEY_O,
	    KEY_P,
	    KEY_Q,
	    KEY_R,
	    KEY_S,
	    KEY_T,
	    KEY_U,
	    KEY_V,
	    KEY_W,
	    KEY_X,
	    KEY_Y,
	    KEY_Z,
	    KEY_0,
	    KEY_1,
	    KEY_2,
	    KEY_3,
	    KEY_4,
	    KEY_5,
	    KEY_6,
	    KEY_7,
	    KEY_8,
	    KEY_9,
	    KEY_SPACE,
	    KEY_ENTER,
	    KEY_BACKSPACE,
	    KEY_TAB,
	    KEY_ESCAPE,
	    KEY_UP,
	    KEY_DOWN,
	    KEY_LEFT,
	    KEY_RIGHT,
	    KEY_HOME,
	    KEY_END,
	    KEY_PAGE_UP,
	    KEY_PAGE_DOWN,
	    KEY_INSERT,
	    KEY_DELETE,
	    KEY_F1,
	    KEY_F2,
	    KEY_F3,
	    KEY_F4,
	    KEY_F5,
	    KEY_F6,
	    KEY_F7,
	    KEY_F8,
	    KEY_F9,
	    KEY_F10,
	    KEY_F11,
	    KEY_F12,
	    KEY_APOSTROPHE,
	    KEY_COMMA,
	    KEY_MINUS,
	    KEY_PERIOD,
	    KEY_SLASH,
	    KEY_SEMICOLON,
	    KEY_EQUAL,
	    KEY_LEFT_BRACKET,
	    KEY_BACKSLASH,
	    KEY_RIGHT_BRACKET,
	    KEY_GRAVE_ACCENT,
	    KEY_WORLD_1,
	    KEY_WORLD_2,
	    KEY_CAPS_LOCK,
	    KEY_SCROLL_LOCK,
	    KEY_NUM_LOCK,
	    KEY_PRINT_SCREEN,
	    KEY_PAUSE,
	    KEY_F13,
	    KEY_F14,
	    KEY_F15,
	    KEY_F16,
	    KEY_F17,
	    KEY_F18,
	    KEY_F19,
	    KEY_F20,
	    KEY_F21,
	    KEY_F22,
	    KEY_F23,
	    KEY_F24,
	    KEY_F25,
	    KEY_KP_0,
	    KEY_KP_1,
	    KEY_KP_2,
	    KEY_KP_3,
	    KEY_KP_4,
	    KEY_KP_5,
	    KEY_KP_6,
	    KEY_KP_7,
	    KEY_KP_8,
	    KEY_KP_9,
	    KEY_KP_DECIMAL,
	    KEY_KP_DIVIDE,
	    KEY_KP_MULTIPLY,
	    KEY_KP_SUBTRACT,
	    KEY_KP_ADD,
	    KEY_KP_ENTER,
	    KEY_KP_EQUAL,
	    KEY_LEFT_SHIFT,
	    KEY_LEFT_CONTROL,
	    KEY_LEFT_ALT,
	    KEY_LEFT_SUPER,
	    KEY_RIGHT_SHIFT,
	    KEY_RIGHT_CONTROL,
	    KEY_RIGHT_ALT,
	    KEY_RIGHT_SUPER,
	    KEY_MENU
	};

	enum class Type {
		NONE, PRESS, RELEASE, REPEAT, HOLD
	};

	Keyboard(Key key, Type type) :
			Event(Class::KEYBOARD), key_(key), type_(type) {
	}

	Key key() const {
		return key_;
	}

	Type type() const {
		return type_;
	}

private:
	Key key_;
	Type type_;
};

class Mouse: public Event {
public:
	enum class Button {
		NONE,
		LEFT,
		RIGHT,
		MIDDLE,
		BUTTON_4,
		BUTTON_5,
		BUTTON_6,
		BUTTON_7,
		BUTTON_8
	};

	enum class Type {
		NONE,
		PRESS,
		RELEASE,
		MOVE,
		SCROLL,
		DRAG_START,
		DRAG,
		DRAG_END,
		HOVER_ENTER,
		HOVER_EXIT,
		DOUBLE_CLICK
	};

	Mouse(Button button, Type type, cv::Point2d position) :
			Event(Class::MOUSE), button_(button), type_(type), position_(
					position) {
	}

	Button button() const {
		return button_;
	}

	Type type() const {
		return type_;
	}

	cv::Point2d position() const {
		return position_;
	}

private:
	Button button_;
	Type type_;
	cv::Point2d position_;
};

class Joystick: public Event {
public:
	enum class Type {
		NONE, BUTTON_PRESS, BUTTON_RELEASE, AXIS_MOVE
	};

	enum class Button {
		BUTTON_NONE,
		BUTTON_A,
		BUTTON_B,
		BUTTON_X,
		BUTTON_Y,
		BUTTON_LB,
		BUTTON_RB,
		BUTTON_BACK,
		BUTTON_START,
		BUTTON_GUIDE,
		BUTTON_LEFT_THUMB,
		BUTTON_RIGHT_THUMB,
		BUTTON_DPAD_UP,
		BUTTON_DPAD_RIGHT,
		BUTTON_DPAD_DOWN,
		BUTTON_DPAD_LEFT
	};

	enum class Axis {
		AXIS_NONE,
		AXIS_LEFT_X,
		AXIS_LEFT_Y,
		AXIS_RIGHT_X,
		AXIS_RIGHT_Y,
		AXIS_LEFT_TRIGGER,
		AXIS_RIGHT_TRIGGER
	};

	Joystick(int joystick, Button button, Type type) :
			Event(Class::JOYSTICK), type_(type), joystick_(joystick), button_(button), state_(
					type == Type::BUTTON_PRESS), axis_(Axis::AXIS_NONE), init_(0), value_(
					0), delta_(0) {
	}

	Joystick(int joystick, Axis axis, float init, float value, float delta) :
			Event(Class::JOYSTICK), type_(Type::AXIS_MOVE), joystick_(joystick), button_(
					Button::BUTTON_NONE), state_(false), axis_(axis), init_(init), value_(value), delta_(delta) {
	}

	Type type() {
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

class Window: public Event {
public:
	enum class Type {
		NONE, RESIZE, MOVE, FOCUS, UNFOCUS, CLOSE
	};

	Window(Type type, cv::Size size) :
			Event(Class::WINDOW), type_(type), size_(size) {
	}
	Window(Type type, cv::Point position) :
			Event(Class::WINDOW), type_(type), position_(position) {
	}

	Window(Type type) :
			Event(Class::WINDOW), type_(type) {
	}

	Type type() const {
		return type_;
	}

	cv::Size get_size() const {
		return size_;
	}

	cv::Point get_position() const {
		return position_;
	}

private:
	Type type_;
	cv::Size size_;
	cv::Point position_;
};

class MouseDrag: public Mouse {
public:
	MouseDrag(Button button, cv::Point2d position, cv::Point2d delta) :
			Mouse(button, Type::DRAG, position), delta_(delta) {
	}

	cv::Point2d delta() const {
		return delta_;
	}

private:
	cv::Point2d delta_;
};

class MouseDragStartEvent: public Mouse {
public:
	MouseDragStartEvent(Button button, cv::Point2d position) :
			Mouse(button, Type::DRAG_START, position) {
	}
};

class MouseDragEndEvent: public Mouse {
public:
	MouseDragEndEvent(Button button, cv::Point2d position) :
			Mouse(button, Type::DRAG_END, position) {
	}
};

class MouseHoverEnterEvent: public Mouse {
public:
	MouseHoverEnterEvent(cv::Point2d position) :
			Mouse(Button::NONE, Type::HOVER_ENTER, position) {
	}
};

class MouseHoverExitEvent: public Mouse {
public:
	MouseHoverExitEvent(cv::Point2d position) :
			Mouse(Button::NONE, Type::HOVER_EXIT, position) {
	}
};

class MouseDoubleClickEvent: public Mouse {
public:
	MouseDoubleClickEvent(Button button, cv::Point2d position) :
			Mouse(button, Type::DOUBLE_CLICK, position) {
	}
};

class MouseMoveEvent: public Mouse {
public:
	MouseMoveEvent(cv::Point2d position, cv::Point2d delta) :
			Mouse(Button::NONE, Type::MOVE, position), delta_(delta) {
	}

	MouseMoveEvent(cv::Point2d position) :
			Mouse(Button::NONE, Type::RELEASE, position), delta_(delta_) {
	}
	cv::Point2d delta() const {
		return delta_;
	}

private:
	cv::Point2d delta_;
};

class MouseScrollEvent: public Mouse {
public:
	MouseScrollEvent(cv::Point2d position, cv::Point2d offset) :
			Mouse(Button::NONE, Type::SCROLL, position), offset_(
					offset) {
	}

	cv::Point2d offset() const {
		return offset_;
	}

private:
	cv::Point2d offset_;
};

namespace detail {


constexpr float AXIS_NO_VALUE = std::numeric_limits<float>::max();
inline static Point2d prev_mouse_pos;
inline static float prev_axis_value[6] = {AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE};
inline static float init_axis_value[6] = {AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE, AXIS_NO_VALUE};

void set_main_glfw_window(GLFWwindow* win);
GLFWwindow* get_main_glfw_window();

constexpr Keyboard::Key v4d_key(int glfw_key) {
	switch (glfw_key) {
	case GLFW_KEY_A:
		return Keyboard::Key::KEY_A;
	case GLFW_KEY_B:
		return Keyboard::Key::KEY_B;
	case GLFW_KEY_C:
		return Keyboard::Key::KEY_C;
	case GLFW_KEY_D:
		return Keyboard::Key::KEY_D;
	case GLFW_KEY_E:
		return Keyboard::Key::KEY_E;
	case GLFW_KEY_F:
		return Keyboard::Key::KEY_F;
	case GLFW_KEY_G:
		return Keyboard::Key::KEY_G;
	case GLFW_KEY_H:
		return Keyboard::Key::KEY_H;
	case GLFW_KEY_I:
		return Keyboard::Key::KEY_I;
	case GLFW_KEY_J:
		return Keyboard::Key::KEY_J;
	case GLFW_KEY_K:
		return Keyboard::Key::KEY_K;
	case GLFW_KEY_L:
		return Keyboard::Key::KEY_L;
	case GLFW_KEY_M:
		return Keyboard::Key::KEY_M;
	case GLFW_KEY_N:
		return Keyboard::Key::KEY_N;
	case GLFW_KEY_O:
		return Keyboard::Key::KEY_O;
	case GLFW_KEY_P:
		return Keyboard::Key::KEY_P;
	case GLFW_KEY_Q:
		return Keyboard::Key::KEY_Q;
	case GLFW_KEY_R:
		return Keyboard::Key::KEY_R;
	case GLFW_KEY_S:
		return Keyboard::Key::KEY_S;
	case GLFW_KEY_T:
		return Keyboard::Key::KEY_T;
	case GLFW_KEY_U:
		return Keyboard::Key::KEY_U;
	case GLFW_KEY_V:
		return Keyboard::Key::KEY_V;
	case GLFW_KEY_W:
		return Keyboard::Key::KEY_W;
	case GLFW_KEY_X:
		return Keyboard::Key::KEY_X;
	case GLFW_KEY_Y:
		return Keyboard::Key::KEY_Y;
	case GLFW_KEY_Z:
		return Keyboard::Key::KEY_Z;
	case GLFW_KEY_0:
		return Keyboard::Key::KEY_0;
	case GLFW_KEY_1:
		return Keyboard::Key::KEY_1;
	case GLFW_KEY_2:
		return Keyboard::Key::KEY_2;
	case GLFW_KEY_3:
		return Keyboard::Key::KEY_3;
	case GLFW_KEY_4:
		return Keyboard::Key::KEY_4;
	case GLFW_KEY_5:
		return Keyboard::Key::KEY_5;
	case GLFW_KEY_6:
		return Keyboard::Key::KEY_6;
	case GLFW_KEY_7:
		return Keyboard::Key::KEY_7;
	case GLFW_KEY_8:
		return Keyboard::Key::KEY_8;
	case GLFW_KEY_9:
		return Keyboard::Key::KEY_9;
	case GLFW_KEY_SPACE:
		return Keyboard::Key::KEY_SPACE;
	case GLFW_KEY_ENTER:
		return Keyboard::Key::KEY_ENTER;
	case GLFW_KEY_BACKSPACE:
		return Keyboard::Key::KEY_BACKSPACE;
	case GLFW_KEY_TAB:
		return Keyboard::Key::KEY_TAB;
	case GLFW_KEY_ESCAPE:
		return Keyboard::Key::KEY_ESCAPE;
	case GLFW_KEY_UP:
		return Keyboard::Key::KEY_UP;
	case GLFW_KEY_DOWN:
		return Keyboard::Key::KEY_DOWN;
	case GLFW_KEY_LEFT:
		return Keyboard::Key::KEY_LEFT;
	case GLFW_KEY_RIGHT:
		return Keyboard::Key::KEY_RIGHT;
	case GLFW_KEY_HOME:
		return Keyboard::Key::KEY_HOME;
	case GLFW_KEY_END:
		return Keyboard::Key::KEY_END;
	case GLFW_KEY_PAGE_UP:
		return Keyboard::Key::KEY_PAGE_UP;
	case GLFW_KEY_PAGE_DOWN:
		return Keyboard::Key::KEY_PAGE_DOWN;
	case GLFW_KEY_INSERT:
		return Keyboard::Key::KEY_INSERT;
	case GLFW_KEY_DELETE:
		return Keyboard::Key::KEY_DELETE;
	case GLFW_KEY_F1:
		return Keyboard::Key::KEY_F1;
	case GLFW_KEY_F2:
		return Keyboard::Key::KEY_F2;
	case GLFW_KEY_F3:
		return Keyboard::Key::KEY_F3;
	case GLFW_KEY_F4:
		return Keyboard::Key::KEY_F4;
	case GLFW_KEY_F5:
		return Keyboard::Key::KEY_F5;
	case GLFW_KEY_F6:
		return Keyboard::Key::KEY_F6;
	case GLFW_KEY_F7:
		return Keyboard::Key::KEY_F7;
	case GLFW_KEY_F8:
		return Keyboard::Key::KEY_F8;
	case GLFW_KEY_F9:
		return Keyboard::Key::KEY_F9;
	case GLFW_KEY_F10:
		return Keyboard::Key::KEY_F10;
	case GLFW_KEY_F11:
		return Keyboard::Key::KEY_F11;
	case GLFW_KEY_F12:
		return Keyboard::Key::KEY_F12;
    case GLFW_KEY_APOSTROPHE:
        return Keyboard::Key::KEY_APOSTROPHE;
    case GLFW_KEY_COMMA:
        return Keyboard::Key::KEY_COMMA;
    case GLFW_KEY_MINUS:
        return Keyboard::Key::KEY_MINUS;
    case GLFW_KEY_PERIOD:
        return Keyboard::Key::KEY_PERIOD;
    case GLFW_KEY_SLASH:
        return Keyboard::Key::KEY_SLASH;
    case GLFW_KEY_SEMICOLON:
        return Keyboard::Key::KEY_SEMICOLON;
    case GLFW_KEY_EQUAL:
        return Keyboard::Key::KEY_EQUAL;
    case GLFW_KEY_LEFT_BRACKET:
        return Keyboard::Key::KEY_LEFT_BRACKET;
    case GLFW_KEY_BACKSLASH:
        return Keyboard::Key::KEY_BACKSLASH;
    case GLFW_KEY_RIGHT_BRACKET:
        return Keyboard::Key::KEY_RIGHT_BRACKET;
    case GLFW_KEY_GRAVE_ACCENT:
        return Keyboard::Key::KEY_GRAVE_ACCENT;
    case GLFW_KEY_WORLD_1:
        return Keyboard::Key::KEY_WORLD_1;
    case GLFW_KEY_WORLD_2:
        return Keyboard::Key::KEY_WORLD_2;
    case GLFW_KEY_CAPS_LOCK:
        return Keyboard::Key::KEY_CAPS_LOCK;
    case GLFW_KEY_SCROLL_LOCK:
        return Keyboard::Key::KEY_SCROLL_LOCK;
    case GLFW_KEY_NUM_LOCK:
        return Keyboard::Key::KEY_NUM_LOCK;
    case GLFW_KEY_PRINT_SCREEN:
        return Keyboard::Key::KEY_PRINT_SCREEN;
    case GLFW_KEY_PAUSE:
        return Keyboard::Key::KEY_PAUSE;
	default:
		throw std::runtime_error(
				"Invalid key: " + std::to_string(glfw_key)
						+ ". Please ensure the key is within the valid range.");
	}
}

constexpr Mouse::Button v4d_mouse_button(int glfw_button) {
	switch (glfw_button) {
	case GLFW_MOUSE_BUTTON_LEFT:
		return Mouse::Button::LEFT;
	case GLFW_MOUSE_BUTTON_RIGHT:
		return Mouse::Button::RIGHT;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		return Mouse::Button::MIDDLE;
	case GLFW_MOUSE_BUTTON_4:
		return Mouse::Button::BUTTON_4;
	case GLFW_MOUSE_BUTTON_5:
		return Mouse::Button::BUTTON_5;
	case GLFW_MOUSE_BUTTON_6:
		return Mouse::Button::BUTTON_6;
	case GLFW_MOUSE_BUTTON_7:
		return Mouse::Button::BUTTON_7;
	case GLFW_MOUSE_BUTTON_8:
		return Mouse::Button::BUTTON_8;
	default:
		throw std::runtime_error(
				"Invalid mouse button: " + std::to_string(glfw_button)
						+ ". Please ensure the button is within the valid range.");
	}
}

constexpr Joystick::Button v4d_joystick_button(int glfw_button) {
	switch (glfw_button) {
	case GLFW_GAMEPAD_BUTTON_A:
		return Joystick::Button::BUTTON_A;
	case GLFW_GAMEPAD_BUTTON_B:
		return Joystick::Button::BUTTON_B;
	case GLFW_GAMEPAD_BUTTON_X:
		return Joystick::Button::BUTTON_X;
	case GLFW_GAMEPAD_BUTTON_Y:
		return Joystick::Button::BUTTON_Y;
	case GLFW_GAMEPAD_BUTTON_LEFT_BUMPER:
		return Joystick::Button::BUTTON_LB;
	case GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER:
		return Joystick::Button::BUTTON_RB;
	case GLFW_GAMEPAD_BUTTON_BACK:
		return Joystick::Button::BUTTON_BACK;
	case GLFW_GAMEPAD_BUTTON_START:
		return Joystick::Button::BUTTON_START;
	case GLFW_GAMEPAD_BUTTON_GUIDE:
		return Joystick::Button::BUTTON_GUIDE;
	case GLFW_GAMEPAD_BUTTON_LEFT_THUMB:
		return Joystick::Button::BUTTON_LEFT_THUMB;
	case GLFW_GAMEPAD_BUTTON_RIGHT_THUMB:
		return Joystick::Button::BUTTON_RIGHT_THUMB;
	case GLFW_GAMEPAD_BUTTON_DPAD_UP:
		return Joystick::Button::BUTTON_DPAD_UP;
	case GLFW_GAMEPAD_BUTTON_DPAD_RIGHT:
		return Joystick::Button::BUTTON_DPAD_RIGHT;
	case GLFW_GAMEPAD_BUTTON_DPAD_DOWN:
		return Joystick::Button::BUTTON_DPAD_DOWN;
	case GLFW_GAMEPAD_BUTTON_DPAD_LEFT:
		return Joystick::Button::BUTTON_DPAD_LEFT;
	default:
		throw std::runtime_error(
				"Invalid joystick button: " + std::to_string(glfw_button)
						+ ". Please ensure the button is within the valid range.");
	}
}

constexpr Joystick::Axis v4d_joystick_axis(int glfw_axis) {
	switch (glfw_axis) {
	case GLFW_GAMEPAD_AXIS_LEFT_X:
		return Joystick::Axis::AXIS_LEFT_X;
	case GLFW_GAMEPAD_AXIS_LEFT_Y:
		return Joystick::Axis::AXIS_LEFT_Y;
	case GLFW_GAMEPAD_AXIS_RIGHT_X:
		return Joystick::Axis::AXIS_RIGHT_X;
	case GLFW_GAMEPAD_AXIS_RIGHT_Y:
		return Joystick::Axis::AXIS_RIGHT_Y;
	case GLFW_GAMEPAD_AXIS_LEFT_TRIGGER:
		return Joystick::Axis::AXIS_LEFT_TRIGGER;
	case GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER:
		return Joystick::Axis::AXIS_RIGHT_TRIGGER;
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

constexpr Mouse::Type v4d_mouse_event_type(int glfw_state) {
	Mouse::Type type;
	switch (glfw_state) {
	case GLFW_PRESS:
		type = Mouse::Type::PRESS;
		break;
	case GLFW_RELEASE:
		type = Mouse::Type::RELEASE;
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
		type = Joystick::Type::BUTTON_PRESS;
		break;
	case GLFW_RELEASE:
		type = Joystick::Type::BUTTON_RELEASE;
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
	size_t capacity_;
	typedef std::deque<std::shared_ptr<Event>> parent_t;
public:
	EventQueue(const size_t& capacity) : capacity_(capacity)  {
		CV_Assert(capacity > 0);
	}

	void push(const std::shared_ptr<Event>& event) {
		std::unique_lock < std::mutex > lock(mutex_);
		if(size() == capacity_)
			parent_t::pop_front();

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

static void poll_joystick_events(EventQueue *queue) {
	if (queue == nullptr)
		throw std::runtime_error("The event-queue is not initialized");

	for (int jid = 0; jid <= GLFW_JOYSTICK_LAST; jid++) {
		if (glfwJoystickPresent(jid) && glfwJoystickIsGamepad(jid)) {
			GLFWgamepadstate state;
			if (glfwGetGamepadState(jid, &state)) {
				for (int button = 0; button <= GLFW_GAMEPAD_BUTTON_LAST;
						button++) {
					Joystick::Button jsButton = v4d_joystick_button(
							button);
					Joystick::Type type = v4d_joystick_event_type(state.buttons[button]);
					std::shared_ptr< Joystick> event = std::make_shared<Joystick>(jid, jsButton, type);
					queue->push(event);
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
					queue->push(event);
				}
			}
		}
	}
}

static EventQueue* queue() {
	auto* win = get_main_glfw_window();
	CV_Assert(win);
	return reinterpret_cast<EventQueue*>(glfwGetWindowUserPointer(win));
}

}

static void init(GLFWwindow *win) {
	CV_Assert(win);
	detail::set_main_glfw_window(win);
	detail::EventQueue *queue = new detail::EventQueue(10);

	glfwSetWindowUserPointer(win, queue);
	glfwSetKeyCallback(win,
			[](GLFWwindow *window, int key, int scancode, int action,
					int mods) {
				ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
				if (!ImGui::GetIO().WantCaptureKeyboard) {
					auto *queue = detail::queue();
					if (key != GLFW_KEY_UNKNOWN) {
						Keyboard::Key v4d_key = detail::v4d_key(key);
						Keyboard::Type type = detail::v4d_keyboard_event_type(action);
						std::shared_ptr<Keyboard> event = std::make_shared<Keyboard>(v4d_key, type);
						queue->push(event);
					}
				}
			});

	glfwSetMouseButtonCallback(win,
			[](GLFWwindow *window, int button, int action, int mods) {
				auto *queue = detail::queue();
				ImGui_ImplGlfw_MouseButtonCallback(window, button, action,
						mods);

				if (!ImGui::GetIO().WantCaptureMouse) {
					if (button != GLFW_MOUSE_BUTTON_LAST) {
						Mouse::Button mouseButton = detail::v4d_mouse_button(button);
						Mouse::Type type = detail::v4d_mouse_event_type(action);

						double x, y;
						glfwGetCursorPos(window, &x, &y);
						cv::Point2d position(x, y);
						std::shared_ptr<Mouse> event = std::make_shared<Mouse>(mouseButton, type, position);
						queue->push(event);
					}
				}
			});

	glfwSetWindowSizeCallback(win,
			[](GLFWwindow *window, int width, int height) {
				auto *queue = detail::queue();
				cv::Size size(width, height);
				std::shared_ptr<Window> event = std::make_shared<Window>(Window::Type::RESIZE, size);
				queue->push(event);
			});

	glfwSetWindowPosCallback(win,
			[](GLFWwindow *window, int xpos, int ypos) {
				auto *queue = detail::queue();
				cv::Point position(xpos, ypos);
				std::shared_ptr<Window> event = std::make_shared<Window>(Window::Type::MOVE, position);
				queue->push(event);
			});

	glfwSetWindowFocusCallback(win, [](GLFWwindow *window, int focused) {
		auto *queue = detail::queue();
		Window::Type type = focused ? Window::Type::FOCUS :
	Window::Type::UNFOCUS;
		std::shared_ptr<Window> event = std::make_shared<Window>(type);
		queue->push(event);
	});

	glfwSetWindowCloseCallback(win, [](GLFWwindow *window) {
		auto *queue = detail::queue();
		std::shared_ptr<Window> event = std::make_shared<Window>(Window::Type::CLOSE);
		queue->push(event);
	});

	glfwSetScrollCallback(win,
			[](GLFWwindow *window, double xoffset, double yoffset) {
				auto *queue = detail::queue();
				cv::Point2d offset(xoffset, yoffset);
				double x, y;
				glfwGetCursorPos(window, &x, &y);
				cv::Point2d position(x, y);
				std::shared_ptr<MouseScrollEvent> event = std::make_shared<MouseScrollEvent>(position, offset);
				queue->push(event);
				ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
			});

	glfwSetCursorPosCallback(win,
			[](GLFWwindow *window, double xpos, double ypos) {
				ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
				if (!ImGui::GetIO().WantCaptureMouse) {
					auto *queue = detail::queue();
					double x, y;
					glfwGetCursorPos(window, &x, &y);
					cv::Point2d position(x, y);
					bool pressed = false;
					for (int button = 0; button <= GLFW_MOUSE_BUTTON_LAST;
							button++) {
						if (glfwGetMouseButton(window, button) == GLFW_PRESS) {
							pressed = true;
							break;
						}
					}
					if (pressed) {
						cv::Point2d delta = position - detail::prev_mouse_pos;
						for (int button = 0; button <= GLFW_MOUSE_BUTTON_LAST;
								button++) {
							if (glfwGetMouseButton(window, button) == GLFW_PRESS) {
								Mouse::Button v4d_button =
										detail::v4d_mouse_button(button);
								std::shared_ptr < MouseDrag > event =
										std::make_shared < MouseDrag
												> (v4d_button, position, delta);
								queue->push(event);
							}
						}
						detail::prev_mouse_pos = position;
					} else {
						std::shared_ptr < MouseMoveEvent > event = std::make_shared
								< MouseMoveEvent > (position);
						queue->push(event);
					}
				}

			});
}


template<typename Tevent>
static bool has(){
	auto *queue = detail::queue();
	CV_Assert(queue);
	return queue->has<Tevent>();
}

template<typename Tevent>
static std::vector<std::shared_ptr<Tevent>> get(){
	std::vector<std::shared_ptr<Tevent>> result;
	auto *queue = detail::queue();
	CV_Assert(queue);
	queue->get<Tevent>(result);
	return result;
}

template<typename Tevent>
static bool has(const typename Tevent::Type& t){
	auto *queue = detail::queue();
	CV_Assert(queue);
	return queue->has(t);
}

template<typename Tevent>
static std::vector<std::shared_ptr<Tevent>> get(const typename Tevent::Type& t){
	std::vector<std::shared_ptr<Tevent>> result;
	auto *queue = detail::queue();
	CV_Assert(queue);
	queue->get(t, result);
	return result;
}

static std::vector<std::shared_ptr<Keyboard>> get(const Keyboard::Type& t, const Keyboard::Key& k){
	std::vector<std::shared_ptr<Keyboard>> result;
	auto *queue = detail::queue();
	CV_Assert(queue);
	queue->get<Keyboard>(t, result, [k](std::shared_ptr<Keyboard> ev){ return ev->key() == k; });
	return result;
}

static std::vector<std::shared_ptr<Mouse>> get(const Mouse::Type& t, const Mouse::Button& b){
	std::vector<std::shared_ptr<Mouse>> result;
	auto *queue = detail::queue();
	CV_Assert(queue);
	queue->get<Mouse>(t, result, [b](std::shared_ptr<Mouse> ev){ return ev->button() == b; });
	return result;
}

static std::vector<std::shared_ptr<Joystick>> get(const Joystick::Type& t, const Joystick::Button& b){
	std::vector<std::shared_ptr<Joystick>> result;
	auto *queue = detail::queue();
	CV_Assert(queue);
	queue->get<Joystick>(t, result, [b](std::shared_ptr<Joystick> ev){ return ev->button() == b; });
	return result;
}

static std::vector<std::shared_ptr<Joystick>> get(const Joystick::Type& t, const Joystick::Axis& a){
	std::vector<std::shared_ptr<Joystick>> result;
	auto *queue = detail::queue();
	CV_Assert(queue);
	queue->get<Joystick>(t, result, [a](std::shared_ptr<Joystick> ev){ return ev->axis() == a; });
	return result;
}

static bool has(const Keyboard::Type& t, const Keyboard::Key& k){
	auto *queue = detail::queue();
	CV_Assert(queue);
	return queue->has<Keyboard>(t, [k](std::shared_ptr<Keyboard> ev){ return ev->key() == k; });
}

static bool has(const Mouse::Type& t, const Mouse::Button& b){
	auto *queue = detail::queue();
	CV_Assert(queue);
	return queue->has<Mouse>(t, [b](std::shared_ptr<Mouse> ev){ return ev->button() == b; });
}

static bool has(const Joystick::Type& t, const Joystick::Button& b){
	auto *queue = detail::queue();
	CV_Assert(queue);
	return queue->has<Joystick>(t, [b](std::shared_ptr<Joystick> ev){ return ev->button() == b; });
}

static bool has(const Joystick::Type& t, const Joystick::Axis& a){
	auto *queue = detail::queue();
	CV_Assert(queue);
	return queue->has<Joystick>(t, [a](std::shared_ptr<Joystick> ev){ return ev->axis() == a; });
}

static void poll() {
	if(detail::get_main_glfw_window()) {
		auto *queue = detail::queue();
		queue->clear();
		glfwPollEvents();
		poll_joystick_events(queue);
	}
}

}
}
}
#endif  // MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_
