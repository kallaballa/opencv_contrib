// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <opencv2/v4d/scene.hpp>

using namespace cv::v4d;


class Camera
{
	// Default camera values
	constexpr static float ROLL        =  0.0f;
	constexpr static float YAW         = -90.0f;
	constexpr static float PITCH       =  0.0f;
	constexpr static float SPEED       =  2.5f;
	constexpr static float ZOOM        =  45.0f;
public:
    // camera Attributes
    cv::Vec3f position_;
    cv::Vec3f front_;
    cv::Vec3f up_;
    cv::Vec3f right_;
    cv::Vec3f worldUp_;
    cv::Vec3f lookAt_;
    // euler Angles
    float roll_;
    float yaw_;
    float pitch_;
    // camera options
    float movementSpeed_;
    float zoom_;

    // constructor with vectors
    Camera(cv::Vec3f position = cv::Vec3f(0.0f, 0.0f, 0.0f), cv::Vec3f up = cv::Vec3f(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH, float roll = ROLL) : front_(cv::Vec3f(0.0f, 0.0f, -1.0f)), movementSpeed_(SPEED),  zoom_(ZOOM)
    {
        position_ = position;
        worldUp_ = up;
        yaw_ = yaw;
        pitch_ = pitch;
        roll_ = roll;
        updateCameraVectors();
    }
    // constructor with scalar values
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch, float roll) : front_(cv::Vec3f(0.0f, 0.0f, -1.0f)), movementSpeed_(SPEED), zoom_(ZOOM)
    {
        position_ = cv::Vec3f(posX, posY, posZ);
        worldUp_ = cv::Vec3f(upX, upY, upZ);
        yaw_ = yaw;
        pitch_ = pitch;
        roll_ = roll;
        updateCameraVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    cv::Matx44f getViewMatrix()
    {
        return gl::lookAt(position_, lookAt_, up_);
    }

    void circle(float xamount, float yamount, float zamount) {
        // Convert degrees to radians
        float xrad = radians(xamount);
        float yrad = radians(yamount);
        float zrad = radians(zamount);

        // Create rotation matrices for each axis
        cv::Matx33f rotx = cv::Matx33f(
            1, 0, 0,
            0, cos(xrad), -sin(xrad),
            0, sin(xrad), cos(xrad)
        );
        cv::Matx33f roty = cv::Matx33f(
            cos(yrad), 0, sin(yrad),
            0, 1, 0,
            -sin(yrad), 0, cos(yrad)
        );
        cv::Matx33f rotz = cv::Matx33f(
            cos(zrad), -sin(zrad), 0,
            sin(zrad), cos(zrad), 0,
            0, 0, 1
        );

        // Apply the rotations to the position vector
        cv::Vec3f pos = position_ - lookAt_;
        cv::Matx31f posmat = cv::Matx31f(pos[0], pos[1], pos[2]);
        posmat = rotx * roty * rotz * posmat;
        position_ = cv::Vec3f(posmat(0), posmat(1), posmat(2)) + lookAt_;

        // Update the camera vectors
        updateCameraVectors();
    }

    void advance(float amount) {
    	position_ += front_ * (movementSpeed_ * amount);
    	updateLookAt();
    }

    void strafe(float amount) {
    	position_ += right_ * (movementSpeed_ * amount);
    	updateLookAt();
    }

    void roll(float amount) {
        roll_ += amount;
        updateCameraVectors();
    }

    void yaw(float amount) {
    	yaw_ += amount;
        updateCameraVectors();
    }

    void pitch(float amount, bool constraint = false) {
    	pitch_ += amount;
        updateCameraVectors();
        if (constraint) {
            if (pitch_ > 89.0f)
                pitch_ = 89.0f;
            if (pitch_ < -89.0f)
                pitch_ = -89.0f;
        }
    }

    void zoom(float amount) {
        zoom_ -= (float)amount;
        if (zoom_ < 1.0f)
            zoom_ = 1.0f;
        if (zoom_ > 45.0f)
            zoom_ = 45.0f;

    }
private:
    float radians(float degrees) {
    	return degrees * CV_PI / 180.0f;
    }

    cv::Vec3f normalize(const cv::Vec3f& v) {
        float norm = cv::norm(v);
        if (norm != 0.0f) {
            return v / norm;
        } else {
            return v;
        }
    }

    void updateCameraVectors()
    {
        // calculate the new Front vector
        cv::Vec3f front;
        front[0] = cos(radians(yaw_)) * cos(radians(pitch_));
        front[1] = sin(radians(pitch_));
        front[2] = sin(radians(yaw_)) * cos(radians(pitch_));
        front_ = normalize(front);
        // also re-calculate the Right and Up vector
        right_ = normalize(front_.cross(worldUp_));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        up_    = normalize(right_.cross(front_));

        // Create a rotation matrix for rolling
        cv::Matx33f rot = cv::Matx33f(
            cos(radians(roll_)), -sin(radians(roll_)), 0,
            sin(radians(roll_)), cos(radians(roll_)), 0,
            0, 0, 1
        );
        cv::Matx31f up = cv::Matx31f(up_[0], up_[1], up_[2]);
        cv::Matx31f right = cv::Matx31f(right_[0], right_[1], right_[2]);

        // Apply the rotation to the up and right vectors
        up = rot * up;
        right = rot * right;

        // Update the up and right vectors
        up_ = cv::Vec3f(up(0), up(1), up(2));
        right_ = cv::Vec3f(right(0), right(1), right(2));
        updateLookAt();
    }

    void updateLookAt() {
        lookAt_ = position_ + front_;
    }
};

class SceneDemoPlan : public Plan {
	const string filename_ = "Avocado.glb";
	gl::Scene scene_;
	Camera camera_;
	struct Transform {
		cv::Vec3f translate_;
		cv::Vec3f rotation_;
		cv::Vec3f scale_;
	    cv::Matx44f projection_;
		cv::Matx44f view_;
		cv::Matx44f model_;
	} transform_;


public:
	using Plan::Plan;

	void setup(cv::Ptr<V4D> window) override {
		//Executed once before infer is called
		window->gl([](const cv::Rect& viewport, gl::Scene& scene, const string& filename, Transform& transform) {
			CV_Assert(scene.load(filename));
			float scale = scene.autoScale();
			//initial center of scene deduced from bounding box
		    cv::Vec3f center = scene.autoCenter();
		    transform.rotation_ = {0, 0, 0};
		    transform.translate_ = {-center[0], -center[1], -center[2]};
		    transform.scale_ = { scale, scale, scale };
	        //gl::perspective works analogous to glm::perspective
	        transform.projection_ = gl::perspective(45.0f * (CV_PI/180), float(viewport.width) / viewport.height, 0.1f, 100.0f);
		    //gl::modelView works analogous to glm::modelView
		    transform.model_ = gl::modelView(transform.translate_, transform.rotation_, transform.scale_);
		}, viewport(), scene_, filename_, transform_);
	}

	void infer(cv::Ptr<V4D> window) override {
	    //Executed on every frame
		window->gl(0,[](const int32_t& ctx, const cv::Rect& viewport, gl::Scene& scene, Transform& transform, Camera& camera){
	        using namespace cv::v4d::event;
	        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	        //retrieve current joystick events
	        auto jsEvents = event::get<Joystick>();

	        // Process joystick events
	        for (const auto& jsEvent : jsEvents) {
	            if (jsEvent->type() == Joystick::Type::BUTTON_PRESS && jsEvent->button() == Joystick::Button::BUTTON_A) {
	                // Use BUTTON_A to cycle through render modes
	                int m = (static_cast<int>(scene.getMode()) + 1) % 3;
	                scene.setMode(static_cast<gl::Scene::RenderMode>(m));
	            } else if (jsEvent->type() == Joystick::Type::AXIS_MOVE && std::fabs(jsEvent->abs()) > 0.1) {
	                if (jsEvent->axis() == Joystick::Axis::AXIS_LEFT_X) {
	                	cerr << "LEFT_X" << jsEvent->abs() << endl;
	                	camera.strafe(jsEvent->abs() / 100.0f);
	                } else if (jsEvent->axis() == Joystick::Axis::AXIS_LEFT_Y && std::fabs(jsEvent->abs()) > 0.1) {
	                	cerr << "LEFT_Y" << jsEvent->abs() << endl;
	                    camera.advance(-jsEvent->abs() / 100.0f);
	                } else if (jsEvent->axis() == Joystick::Axis::AXIS_RIGHT_X && std::fabs(jsEvent->abs()) > 0.1) {
	                	cerr << "RIGHT_X" << jsEvent->abs() << endl;
	                	camera.yaw(jsEvent->abs() / 10.0f);
	                } else if (jsEvent->axis() == Joystick::Axis::AXIS_RIGHT_Y && std::fabs(jsEvent->abs()) > 0.1) {
	                	cerr << "RIGHT_Y" << jsEvent->abs() << endl;
	                	camera.pitch(jsEvent->abs() / 10.0f);
	                } else if (jsEvent->axis() == Joystick::Axis::AXIS_LEFT_TRIGGER && jsEvent->value() > -1) {
	                	cerr << "LEFT_TRIGGER" << jsEvent->value()  << endl;
	                	camera.roll(-((jsEvent->value() + 1) / 2.0) / 10.0f);
	                } else if (jsEvent->axis() == Joystick::Axis::AXIS_RIGHT_TRIGGER && jsEvent->value() > -1) {
	                	cerr << "RIGHT_TRIGGER" << jsEvent->value() << endl;
	                	camera.roll(((jsEvent->value() + 1) / 2.0) / 10.0f);
	                }
	            }
	        }

	        // Use the camera's view matrix
	        transform.view_ = camera.getViewMatrix();

	        scene.render(viewport, transform.projection_, transform.view_, transform.model_);
	    }, viewport(), scene_, transform_, camera_);
	    window->write();
	}

};


int main() {
    cv::Ptr<V4D> window = V4D::make(cv::Size(1280, 720), "Scene Demo", IMGUI);
	cv::Ptr<SceneDemoPlan> plan = new SceneDemoPlan(cv::Size(1280, 720));

    auto sink = makeWriterSink(window, "scene-demo.mkv", 60, plan->size());
    window->setSink(sink);
    window->run(plan, 0);

    return 0;
}
