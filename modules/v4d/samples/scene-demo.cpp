// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <opencv2/v4d/scene.hpp>

using namespace cv::v4d;

//static double radians(double degrees) {
//	return degrees * CV_PI / 180.0f;
//}
//
//static double degrees(double radians) {
//	return radians * (180.0f / CV_PI);
//}
//
//static void calculate_vectors_euler(double yaw, double pitch, double roll, cv::Vec3d &front, cv::Vec3d &right, cv::Vec3d &up) {
//	double yawRad = yaw * CV_PI / 180.0f;
//	double rollRad = roll * CV_PI / 180.0f;
//	double pitchRad = pitch * CV_PI / 180.0f;
//
//	 // Calculate rotation matrices for each axis
//	    cv::Matx33d rotYaw = cv::Matx33d(
//	        cos(yaw), 0, sin(yaw),
//	        0, 1, 0,
//	        -sin(yaw), 0, cos(yaw)
//	    );
//
//	    cv::Matx33d rotPitch = cv::Matx33d(
//	        1, 0, 0,
//	        0, cos(pitch), -sin(pitch),
//	        0, sin(pitch), cos(pitch)
//	    );
//
//	    cv::Matx33d rotRoll = cv::Matx33d(
//	        cos(roll), -sin(roll), 0,
//	        sin(roll), cos(roll), 0,
//	        0, 0, 1
//	    );
//
//		 //Calculate the front vector
//	    front = rotYaw * rotPitch * rotRoll * cv::Vec3d(0, 0, -1);
//
//	    // Calculate the up vector
//	    up = rotYaw * rotPitch * rotRoll * cv::Vec3d(0, 1, 0);
//
//	    // Calculate the right vector as the cross product of the up and front vectors
//	    right = up.cross(front);
//}
//
//static void accelerate(double& speed, const double mass, const double impulse, const double timeStep) {
//    double impulseForce = impulse / mass * timeStep;
//    double netForce = mass + impulseForce;
//    double acceleration = netForce / mass;
//    double deltaV = acceleration * timeStep;
//    if(speed == 0) {
//    	speed += deltaV;
//    } else if(speed * impulse < 0) {
//        if(speed < 0) {
//            speed += deltaV;
//        } else {
//            speed -= deltaV;
//        }
//    } else {
//    	if(speed < 0) {
//			speed -= deltaV;
//		} else {
//			speed += deltaV;
//		}
//    }
//}
//
//static void decelerate(double& speed, const double mass, const double timeStep, const double friction = 1.1) {
//	assert(friction >= 1.0);
//	if(speed == 0)
//		return;
//    double frictionForce = friction * mass;
//    double netForce = mass - frictionForce;
//    double acceleration = netForce / mass;
//    double deltaV = acceleration * timeStep;
//    if(speed < 0) {
//    	speed -= deltaV;
//    	speed = speed >= 0 ? 0 : speed;
//    } else {
//    	speed += deltaV;
//    	speed = speed < 0 ? 0 : speed;
//	}
//}
//
//class Camera
//{
//public:
//	//default values
//	constexpr static double DEFAULT_YAW = 0;
//	constexpr static double DEFAULT_PITCH = 0;
//	constexpr static double DEFAULT_ROLL = 0;
//	constexpr static double DEFAULT_MASS = 1;
//private:
//	//name speed indexes
//	constexpr static size_t ADVANCE = 0;
//	constexpr static size_t STRAFE = 1;
//	constexpr static size_t YAW = 2;
//	constexpr static size_t PITCH = 3;
//	constexpr static size_t ROLL = 4;
//
//	cv::Vec3d init_position_;
//	cv::Vec3d position_;
//    cv::Vec3d front_;
//    cv::Vec3d up_;
//    cv::Vec3d right_;
//    cv::Vec3d worldUp_;
//
//    double roll_;
//    double yaw_;
//    double pitch_;
//    double zoom_ = 0;
//
//    //crude inertia physics
//    std::vector<double> speeds_ = { 0, 0, 0, 0, 0};
//
//    double speedOverride_ = 1.0;
//    double sensitivity_ = 1.0;
//    double friction_ = 0.0;
//    double mass_;
//public:
//    Camera(cv::Vec3d position = cv::Vec3d(0.0f, 0.0f, 3.0f),
//    		cv::Vec3d front = cv::Vec3d(0.0f, 0.0f, -1.0f),
//    		cv::Vec3d worldUp = cv::Vec3d(0.0f, 1.0f, 0.0f),
//			double yaw = DEFAULT_YAW,
//			double pitch = DEFAULT_PITCH,
//			double roll = DEFAULT_ROLL,
//			double mass = DEFAULT_MASS) :
//				init_position_(position),
//				position_(position),
//				worldUp_(worldUp),
//				yaw_(yaw),
//				pitch_(pitch),
//				roll_(roll),
//				mass_(mass),
//				front_(front) {
//    	updateVectors();
//    }
//
//   void reset() {
//    	position_ = init_position_;
//    	speeds_ = { 0, 0, 0, 0, 0 };
//    	zoom_ = 0;
//        yaw_ = DEFAULT_YAW;
//        pitch_ = DEFAULT_PITCH;
//        roll_ = DEFAULT_ROLL;
//        updateVectors();
//    }
//
//    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
//	cv::Matx44d getViewMatrix() {
//		cv::Vec3d zoomed  = position_ + (front_ * -zoom_);
//		return gl::lookAt(zoomed, zoomed + front_, up_);
//	}
//
//	cv::Matx33d getRotationMatrix() {
//	    cv::Matx33f rotXMat(
//	            1.0, 0.0, 0.0,
//	            0.0, cos(yaw_), -sin(yaw_),
//	            0.0, sin(yaw_), cos(yaw_)
//	    );
//
//	    cv::Matx33f rotYMat(
//	            cos(pitch_), 0.0, sin(pitch_),
//	            0.0, 1.0, 0.0,
//	            -sin(pitch_), 0.0,cos(pitch_)
//	            );
//
//	    cv::Matx33f rotZMat(
//	            cos(roll_), -sin(roll_), 0.0,
//	            sin(roll_), cos(roll_), 0.0,
//	            0.0, 0.0, 1.0
//	    		);
//	    return rotXMat * rotYMat * rotZMat;
//	}
//
//	cv::Vec3d position() {
//		return position_;
//	}
//
//	cv::Vec3d direction() {
//		return front_;
//	}
//
//	void setMass(double mass) {
//		mass_ = mass;
//	}
//
//	double getMass() {
//		return mass_;
//	}
//
//	double hasInertia() {
//		return mass_ > 1;
//	}
//
//	void circle(double xDegree, double yDegree, double rollDegree, cv::Vec3d around) {
//	    double xrad = radians(xDegree * sensitivity_ * 70.0);
//	    double yrad = radians(yDegree * sensitivity_ * 70.0);
//	    double zRoll = radians(rollDegree * sensitivity_ * 20.0);
//
//	    cv::Vec2d rotationVector(xrad, yrad);
//	    cv::Vec3d rotatedPosition = gl::rotate3D(position_, around, rotationVector);
//	    cv::Vec3d toAround = around - position_;
//	    position_ = rotatedPosition;
//	    roll_ += zRoll;
//	    updateVectors();
//	    //override the front_ and right_ vector just calculate with the once fixed at around
//	    front_ = cv::normalize(toAround);
//	    right_ = up_.cross(front_);
//	}
//
//    void advance(double impulse) {
//    	impulse /= 300;
//    	if(hasInertia())
//    		accelerate(speeds_[ADVANCE], mass_, impulse, 1.0 / Global::fps());
//    	else {
//    		if(impulse >= 0)
//    			speeds_[ADVANCE] = 0.25;
//    		else
//    			speeds_[ADVANCE] = -0.25;
//    		impulse = 1.0;
//    	}
//
//    	auto amount = (speeds_[ADVANCE] * speedOverride_ * sensitivity_);
//    	position_ += front_ * amount;
//    }
//
//    void strafe(double impulse) {
//    	impulse /= 300;
//    	if(hasInertia())
//    		accelerate(speeds_[STRAFE], mass_, impulse, 1.0 / Global::fps());
//    	else {
//    		if(impulse >= 0)
//    			speeds_[STRAFE] = 0.25;
//    		else
//    			speeds_[STRAFE] = -0.25;
//    		impulse = 1.0;
//    	}
//
//    	auto amount = (speeds_[STRAFE] * speedOverride_ * sensitivity_);
//    	position_ += right_ * amount;
//    }
//
//    void roll(double impulse) {
//    	impulse /= 300.0;
//    	if(hasInertia())
//    		accelerate(speeds_[ROLL], mass_, impulse, 1.0 / Global::fps());
//    	else {
//    		if(impulse >= 0)
//    			speeds_[ROLL] = 0.25;
//    		else
//    			speeds_[ROLL] = -0.25;
//    		impulse = 1.0;
//    	}
//
//    	double amount = (speeds_[ROLL] * speedOverride_ * sensitivity_);
//    	roll_ += amount;
//        updateVectors();
//    }
//
//    void yaw(double impulse) {
//    	impulse /= 300.0;
//    	if(hasInertia())
//    		accelerate(speeds_[YAW], mass_, impulse, 1.0 / Global::fps());
//    	else {
//    		if(impulse >= 0)
//    			speeds_[YAW] = 0.25;
//    		else
//    			speeds_[YAW] = -0.25;
//    		impulse = 1.0;
//    	}
//
//    	double amount = (speeds_[YAW] * speedOverride_ * sensitivity_);
//    	yaw_ += amount;
//        updateVectors();
//    }
//
//    void pitch(double impulse, bool constraint = true) {
//    	impulse /= 300.0;
//    	if(hasInertia())
//    		accelerate(speeds_[PITCH], mass_, impulse, 1.0 / Global::fps());
//    	else {
//    		if(impulse >= 0)
//    			speeds_[PITCH] = 0.25;
//    		else
//    			speeds_[PITCH] = -0.25;
//    		impulse = 1.0;
//    	}
//
//    	double amount = (speeds_[PITCH] * speedOverride_ * sensitivity_);
//    	pitch_ += amount;
//
//        if (constraint) {
//            if (pitch_ > 89.0f)
//                pitch_ = 89.0f;
//            if (pitch_ < -89.0f)
//                pitch_ = -89.0f;
//        }
//
//        updateVectors();
//    }
//
//    void zoom(double amount) {
//        zoom_ += amount * sensitivity_;
//    }
//
//    void setSpeedOverride(double factor) {
//    	if(factor < 0)
//    		return;
//
//    	speedOverride_ = factor;
//    }
//
//    double getSpeedOverride() {
//		return speedOverride_;
//	}
//
//    void setSensitivity(double factor) {
//    	if(factor < 0)
//    		return;
//
//    	sensitivity_ = factor;
//    }
//
//    double getSensitivity() {
//		return sensitivity_;
//	}
//
//    void setFriction(double factor) {
//    	if(factor < 0)
//    		return;
//
//    	friction_ = factor;
//    }
//
//    double getFriction() {
//		return friction_;
//	}
//
//    void update() {
//    	if(hasInertia() ) {
//    		for(size_t i = 0; i < speeds_.size(); ++i) {
//    			if(friction_ > 1.0) {
//    				decelerate(speeds_[i], mass_, 1.0/Global::fps(), friction_);
//    			}
//    			switch (i) {
//				case ADVANCE:
//	    			position_ += front_ * (speeds_[i] * speedOverride_ * sensitivity_);
//					break;
//				case STRAFE:
//					position_ += right_ * (speeds_[i] * speedOverride_ * sensitivity_);
//					break;
//				case ROLL:
//					roll_ += (speeds_[i] * speedOverride_ * sensitivity_);
//					break;
//				case PITCH:
//					pitch_ += (speeds_[i] * speedOverride_ * sensitivity_);
//					break;
//				case YAW:
//					yaw_ += (speeds_[i] * speedOverride_ * sensitivity_);
//					break;
//				default:
//					assert(false);
//					break;
//				}
//    		}
//    		updateVectors();
//    	}
//    }
//private:
//    void updateVectors() {
//    	updateVectorsEuler();
//    }
//
//    void updateVectorsEuler() {
//    	calculate_vectors_euler(yaw_, pitch_, roll_, front_, right_, up_);
//    }
//};

class SceneDemoPlan : public Plan {
public:
	string filename_;
private:
//	inline static Camera camera_;
//	inline static struct Params {
//		gl::Scene::RenderMode renderMode_;;
//		double mass_;
//		bool fly_;
//		double sensitivity_;
//		double friction_;
//		double speed_;
//	} params_ = {gl::Scene::RenderMode::DEFAULT, Camera::DEFAULT_MASS, false, 0.01, 0.0, 1.0};
//
//	inline static struct Transform {
//		cv::Vec3d position_;
//		cv::Vec3d direction_;
//		cv::Matx33f cameraRotation_;
//		cv::Vec3d rotation_;
//		cv::Vec3d center_;
//		cv::Vec3d translate_;
//		cv::Vec3d scale_;
//	    cv::Matx44d projection_;
//		cv::Matx44d view_;
//		cv::Matx44d model_;
//
//	} transform_;
//
//	gl::Scene scene_;
public:
//	SceneDemoPlan(const cv::Rect& viewport, const string& filename) : Plan(viewport), filename_(filename)
//	, scene_(viewport)
//	{
//		Global::registerShared(camera_);
//		Global::registerShared(params_);
//		Global::registerShared(transform_);
//	}

	void setup() override {
//		window->ext([](gl::Scene& scene, const string& filename) {
//			CV_Assert(scene.load(filename));
//		}, scene_, filename_);
//		window->branch(BranchType::ONCE, always_);
//		{
//		window->plain([](const cv::Rect& viewport, const gl::Scene& scene, Transform& transform) {
//			double scale = scene.autoScale();
//			//initial center of scene deduced from bounding box
//			transform.center_ = scene.autoCenter();
//		    transform.rotation_ = {0, 0, 0};
//		    transform.translate_ = {-transform.center_[0], -transform.center_[1], -transform.center_[2]};
//		    transform.scale_ = { scale, scale, scale };
//	        //gl::perspective works analogous to glm::perspective
//	        transform.projection_ = gl::perspective(45.0f * (CV_PI/180), double(viewport.width) / viewport.height, 0.1f, 100.0f);
//		    //gl::modelView works analogous to glm::modelView
//		    transform.model_ = gl::modelView(transform.translate_, transform.rotation_, transform.scale_);
//		}, viewport(), scene_, transform_);
//		}
//		window->endbranch(BranchType::ONCE, always_);
	}


	void infer() override {
//		window->branch(0, always_);
//		{
//			window->plain([](Transform& transform, Camera& camera, Params& params){
//				using namespace cv::v4d::event;
//				//retrieve current joystick events
//				auto joystick = event::get<Joystick>();
//
//				for (const auto& ev : joystick) {
//					if (ev->is(Joystick::Type::RELEASE)) {
//						// Use BUTTON_A to cycle through render modes
//						if(ev->is(Joystick::Button::A))
//							params.renderMode_ = (static_cast<gl::Scene::RenderMode>((static_cast<int>(params.renderMode_) + 1) % 3));
//						else if(ev->is(Joystick::Button::Y)) {
//							camera.reset();
//						} else if(ev->is(Joystick::Button::BACK)) {
//							params.fly_ = !params.fly_;
//							camera.reset();
//						} else if(ev->is(Joystick::Button::X)) {
//							if(params.mass_ == Camera::DEFAULT_MASS)
//								params.mass_ = 1000;
//							else
//								params.mass_ = Camera::DEFAULT_MASS;
//						} else if(ev->is(Joystick::Button::B)) {
//							if(params.friction_ == 0.0)
//								params.friction_ = 1.1;
//							else
//								params.friction_ = 0.0;
//						} else if(ev->is(Joystick::Button::LB)) {
//							params.speed_ = params.speed_ + 0.1;
//						} else if(ev->is(Joystick::Button::RB)) {
//							params.speed_ = params.speed_ - 0.1;
//						}
//					} else if (ev->is(Joystick::Type::MOVE) && ev->active()) {
//						switch(ev->axis()) {
//						case Joystick::Axis::LEFT_X:
//							if(params.fly_)
//								camera.strafe(-ev->abs());
//							else
//								camera.circle(0, ev->abs(), 0, transform.center_);
//							break;
//						case Joystick::Axis::LEFT_Y:
//							if(params.fly_)
//								camera.advance(-ev->abs());
//							else
//								camera.circle(ev->abs(), 0, 0, transform.center_);
//							break;
//						case Joystick::Axis::RIGHT_X:
//							if(params.fly_)
//								camera.yaw(-ev->abs());
//							break;
//						case Joystick::Axis::RIGHT_Y:
//							if(params.fly_)
//								camera.pitch(ev->abs());
//							else
//								camera.zoom(ev->abs());
//							break;
//						case Joystick::Axis::LEFT_TRIGGER:
//							if(params.fly_)
//								camera.roll(-ev->abs());
//							else
//								camera.circle(0, 0, -ev->abs(), transform.center_);
//							break;
//						case Joystick::Axis::RIGHT_TRIGGER:
//							if(params.fly_)
//								camera.roll(ev->abs());
//							else
//								camera.circle(0, 0, ev->abs(), transform.center_);
//							break;
//						}
//					}
//				}
//
//				camera.setMass(params.mass_);
//				camera.setFriction(params.friction_);
//				camera.setSpeedOverride(params.speed_);
//				camera.setSensitivity(params.sensitivity_);
//				camera.update();
//
//				transform.position_ = camera.position();
//				transform.cameraRotation_ = camera.getRotationMatrix();
//				transform.view_ = camera.getViewMatrix();
//			}, transform_, camera_, params_);
//		}
//		window->endbranch(0, always_);
//		window->branch(always_);
//		{
//			window->ext([](gl::Scene& scene, Transform& transform, Params& params){
//				scene.setMode(params.renderMode_);
//				scene.render(transform.position_, transform.direction_, transform.cameraRotation_, transform.projection_, transform.view_, transform.model_);
//			}, scene_, transform_, params_);
//			window->write();
//		}
//		window->endbranch(always_);
	}
};


int main(int argc, char** argv) {
	CV_Assert(argc == 2);
//	string filename = argv[1];
//	cv::Ptr<SceneDemoPlan> plan = new SceneDemoPlan(cv::Rect(0,0, 1920, 1080), filename);
//	cv::Ptr<V4D> window = V4D::make(plan->size(), "Scene Demo", AllocateFlags::IMGUI);
//	window->setFullscreen(true);
//    auto sink = Sink::make(window, "scene-demo.mkv", 60, plan->size());
//    window->setSink(sink);
//    window->run(plan, 0, filename);

    return 0;
}
