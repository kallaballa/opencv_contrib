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
	constexpr static float SPEED       =  0;
	constexpr static float ZOOM        =  0.0f;
	//only used in space ship mode
	constexpr static float MASS		   =  1000.0f;
public:
    // camera Attributes
    cv::Vec3f position_;
    cv::Vec3f front_;
    cv::Vec3f up_;
    cv::Vec3f right_;
    cv::Vec3f worldUp_;
    // euler Angles
    float roll_;
    float yaw_;
    float pitch_;
    // camera options
    float zoom_;

    //crude inertia physics
    bool enableInertia_ = false;
    float advanceSpeed_ = 0;
    float strafeSpeed_ = 0;
    float rollSpeed_ = 0;
    float pitchSpeed_ = 0;
    float yawSpeed_ = 0;

    float speedOverride_ = 1.0;
    float sensitivity_ = 1.0;
    float mass_;

    // constructor with vectors
    Camera(cv::Vec3f position = cv::Vec3f(0.0f, 0.0f, 3.0f), cv::Vec3f up = cv::Vec3f(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH, float roll = ROLL, float mass = MASS) : front_(cv::Vec3f(0.0f, 0.0f, -1.0f)), zoom_(ZOOM), mass_(mass)
    {
        position_ = position;
        worldUp_ = up;
        yaw_ = yaw;
        pitch_ = pitch;
        roll_ = roll;
		right_ = normalize(front_.cross(worldUp_));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        up_    = normalize(right_.cross(front_));
    }

    void reset() {
		advanceSpeed_ = 0;
    	strafeSpeed_ = 0;
		rollSpeed_ = 0;
    	pitchSpeed_ = 0;
    	yawSpeed_ = 0;
    	zoom_ = ZOOM;
        yaw_ = YAW;
        pitch_ = PITCH;
        roll_ = ROLL;
        mass_ = MASS;
        worldUp_ = cv::Vec3f(0.0f, 1.0f, 0.0f);
    	position_ = cv::Vec3f(0.0f, 0.0f, 3.0f);
        front_ = cv::Vec3f(0.0f, 0.0f, -1.0f);
		right_ = normalize(front_.cross(worldUp_));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        up_    = normalize(right_.cross(front_));
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
	cv::Matx44f getViewMatrix() {
		cv::Vec3f zoomed  = position_ + (front_ * -zoom_);
		return gl::lookAt(zoomed, zoomed + front_, up_);
	}

	void enableInertia(bool inertia) {
		enableInertia_ = inertia;
	}

	bool getSpaceShipMode() {
		return enableInertia_;
	}

    void circle(float xamount, float yamount) {
        float xrad = radians(xamount);
        float yrad = radians(yamount);

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

        // Apply the rotations to the position vector
        cv::Vec3f pos = position_;
        cv::Vec3f f = front_;
        cv::Matx31f posmat = cv::Matx31f(pos[0], pos[1], pos[2]);
        cv::Matx31f fmat = cv::Matx31f(f[0], f[1], f[2]);
        posmat = rotx * roty * posmat;
        fmat = rotx * roty * fmat;
        position_ = cv::Vec3f(posmat(0), posmat(1), posmat(2));
        front_ = normalize(cv::Vec3f(fmat(0), fmat(1), fmat(2)));
        right_ = normalize(front_.cross(worldUp_));
        up_    = normalize(right_.cross(front_));
        updateZ();
    }

    void advance(float impulse) {
    	if(enableInertia_)
    		accelerate(advanceSpeed_, enableInertia_ ? mass_ : 1, impulse, 1.0 / Global::fps());
    	else
    		advanceSpeed_ = impulse > 0 ? 30 : -30;

    	auto amount = (advanceSpeed_ * speedOverride_) * (std::fabs(impulse) * sensitivity_);
    	position_ += front_ * amount;
    }

    void strafe(float impulse) {
    	if(enableInertia_)
    		accelerate(strafeSpeed_, enableInertia_ ? mass_ : 1, impulse, 1.0 / (Global::fps() + 1));
    	else
    		strafeSpeed_ = impulse > 0 ? 30 : -30;
    	auto amount = (strafeSpeed_ * speedOverride_) * (std::fabs(impulse) * sensitivity_);
    	position_ += right_ * amount;
    }

    void roll(float impulse) {
    	if(enableInertia_)
    		accelerate(rollSpeed_, enableInertia_ ? mass_ : 1, impulse, 1.0 / Global::fps());
    	else
    		rollSpeed_ = impulse > 0 ? 30 : -30;

    	auto amount = (rollSpeed_ * speedOverride_) * (std::fabs(impulse) * sensitivity_);

    	if(enableInertia_)
    		roll_ = amount;
    	else
    		roll_ = amount;
        updateZ();
    }

    void yaw(float impulse) {
    	if(enableInertia_)
    		accelerate(yawSpeed_, enableInertia_ ? mass_ : 1, impulse, 1.0 / Global::fps());
    	else
    		yawSpeed_ = impulse > 0 ? 30 : -30;

    	auto amount = (yawSpeed_ * speedOverride_) * (std::fabs(impulse) * sensitivity_);
    	yaw_ += amount;
        updateXY();
        updateZ();
    }

    void pitch(float impulse, bool constraint = false) {
    	if(enableInertia_)
    		accelerate(pitchSpeed_, enableInertia_ ? mass_ : 1, impulse, 1.0 / Global::fps());
    	else
    		pitchSpeed_ = impulse > 0 ? 30 : -30;

    	auto amount = (pitchSpeed_ * speedOverride_) * (std::fabs(impulse) * sensitivity_);
    	pitch_ += amount;

        if (constraint) {
            if (pitch_ > 89.0f)
                pitch_ = 89.0f;
            if (pitch_ < -89.0f)
                pitch_ = -89.0f;
        }
        updateXY();
        updateZ();
    }

    void zoom(float amount) {
        zoom_ += (float)amount;
    }

    void setSpeedOverride(float factor) {
    	if(factor < 0)
    		return;

    	speedOverride_ = factor;
    }

    float getSpeedOverride() {
		return speedOverride_;
	}

    void setSensitivity(float factor) {
    	if(factor < 0)
    		return;

    	sensitivity_ = factor;
    }

    float getSensitivity() {
		return sensitivity_;
	}

    void update() {
    	if(enableInertia_ ) {
	        cerr << "before:" << rollSpeed_ << endl;

			decelerate(advanceSpeed_, enableInertia_ ? mass_ : 1, 1.0 / (Global::fps() + 1));
			decelerate(strafeSpeed_, enableInertia_ ? mass_ : 1, 1.0 / (Global::fps() + 1));
			decelerate(pitchSpeed_, enableInertia_ ? mass_ : 1, 1.0 / (Global::fps() + 1));
			decelerate(yawSpeed_, enableInertia_ ? mass_ : 1, 1.0 / (Global::fps() + 1));
			decelerate(rollSpeed_, enableInertia_ ? mass_ : 1, 1.0 / (Global::fps() + 1));
			position_ += front_ * (advanceSpeed_ * speedOverride_ * sensitivity_);
			position_ += right_ * (strafeSpeed_ * speedOverride_ * sensitivity_);
			pitch_ += (pitchSpeed_ * speedOverride_ * sensitivity_);
			yaw_ += (yawSpeed_ * speedOverride_ * sensitivity_);
			roll_ += (rollSpeed_ * speedOverride_ * sensitivity_);
			updateXY();
	        updateZ();
	        cerr << "after:" << rollSpeed_ << endl;
    	}
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

    void accelerate(float& speed, const float mass, const float impulse, const float timeStep) {
    	float scaled = impulse / (mass * timeStep);
    	speed += scaled;
    }



    void decelerate(float& speed, const float mass, const float timeStep) {
        float force =  std::fabs(speed / mass;

        speed  += (speed * (force * timeStep));
    }

    void updateZ() {
        // Create a rotation matrix for rolling
        cv::Matx33f rot = cv::Matx33f(
            cos(radians(roll_)), -sin(radians(roll_)), 0,
            sin(radians(roll_)), cos(radians(roll_)), 0,
            0, 0, 1
        );

        cv::Matx31f up = cv::Matx31f(up_[0], up_[1], up_[2]);

        // Apply the rotation to the up and right vectors
        up = rot * up;

        // Update the up and right vectors
        up_ = normalize(cv::Vec3f(up(0), up(1), up(2)));
        right_ = normalize(up_.cross(front_));
    }

    void updateXY() {
        cv::Vec3f front;
        front[0] = cos(radians(yaw_)) * cos(radians(pitch_));
        front[1] = sin(radians(pitch_));
        front[2] = sin(radians(yaw_)) * cos(radians(pitch_));
        front_ = normalize(front);
    	// also re-calculate the Right and Up vector
        right_ = normalize(front_.cross(worldUp_));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    }
};

class SceneDemoPlan : public Plan {
	const string filename_ = "Avocado.glb";
	inline static Camera camera_;
	inline static struct Params {
		gl::Scene::RenderMode renderMode_;;
		bool inertia_;
		bool fly_;
		float senitivity_;
		float speed_;
	} params_ = {gl::Scene::RenderMode::DEFAULT, true, true, 0.001, 1.0};

	inline static struct Transform {
		cv::Vec3f translate_;
		cv::Vec3f rotation_;
		cv::Vec3f scale_;
	    cv::Matx44f projection_;
		cv::Matx44f view_;
		cv::Matx44f model_;
	} transform_;

	gl::Scene scene_;
public:
	using Plan::Plan;
	SceneDemoPlan(const cv::Rect& viewport) : Plan(viewport) {
		Global::registerShared(camera_);
		Global::registerShared(params_);
		Global::registerShared(transform_);
	}

	void setup(cv::Ptr<V4D> window) override {
		window->gl([](const cv::Rect& viewport, gl::Scene& scene, const string& filename) {
			CV_Assert(scene.load(filename));
		}, viewport(), scene_, filename_);

		window->once([](const cv::Rect& viewport, const gl::Scene& scene, Transform& transform) {
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
		}, viewport(), scene_, transform_);
	}


	void infer(cv::Ptr<V4D> window) override {
		window->branch(0, always_);
		{
			window->plain([](Transform& transform, Camera& camera, Params& params){
				using namespace cv::v4d::event;
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				//retrieve current joystick events
				auto joystick = event::get<Joystick>();

				for (const auto& ev : joystick) {
					if (ev->is(Joystick::Type::RELEASE)) {
						// Use BUTTON_A to cycle through render modes
						if(ev->is(Joystick::Button::B))
							params.renderMode_ = (static_cast<gl::Scene::RenderMode>((static_cast<int>(params.renderMode_) + 1) % 3));
						else if(ev->is(Joystick::Button::Y)) {
							camera.reset();
						} else if(ev->is(Joystick::Button::BACK)) {
							params.fly_ = !params.fly_;
							camera.reset();
						} else if(ev->is(Joystick::Button::X)) {
							params.inertia_ = !params.inertia_;
						} else if(ev->is(Joystick::Button::LB)) {
							params.speed_ = params.speed_ + 0.1;
						} else if(ev->is(Joystick::Button::RB)) {
							params.speed_ = params.speed_ - 0.1;
						}
					} else if (ev->is(Joystick::Type::MOVE) && ev->active()) {
						switch(ev->axis()) {
						case Joystick::Axis::LEFT_X:
							if(params.fly_)
								camera.strafe(-ev->abs());
							else
								camera.circle(0, ev->abs());
							break;
						case Joystick::Axis::LEFT_Y:
							if(params.fly_)
								camera.advance(-ev->abs());
							else
								camera.circle(ev->abs(), 0);
							break;
						case Joystick::Axis::RIGHT_X:
							if(params.fly_)
								camera.yaw(ev->abs());
							break;
						case Joystick::Axis::RIGHT_Y:
							if(params.fly_)
								camera.pitch(ev->abs());
							else
								camera.zoom(ev->abs());
							break;
						case Joystick::Axis::LEFT_TRIGGER:
							camera.roll(ev->abs());
							break;
						case Joystick::Axis::RIGHT_TRIGGER:
							camera.roll(-ev->abs());
							break;
						}
					}
				}
				camera.enableInertia(params.inertia_);
				camera.setSpeedOverride(params.speed_);
				camera.setSensitivity(params.senitivity_);
				camera.update();
				// Use the camera's view matrix
				transform.view_ = camera.getViewMatrix();
			}, transform_, camera_, params_);
		}
		window->branch(always_); {
			window->gl([](const cv::Rect& viewport, gl::Scene& scene, Transform& transform, Params& params){
				scene.setMode(params.renderMode_);
				scene.render(viewport, transform.projection_, transform.view_, transform.model_);
			}, viewport(), scene_, transform_, params_);
		}
	    window->write();
	}

};


int main() {
    cv::Ptr<V4D> window = V4D::make(cv::Size(1280, 720), "Scene Demo", IMGUI);
	cv::Ptr<SceneDemoPlan> plan = new SceneDemoPlan(cv::Rect(0,0, 1280, 720));

    auto sink = makeWriterSink(window, "scene-demo.mkv", 60, plan->size());
    window->setSink(sink);
    window->run(plan, 0);

    return 0;
}
