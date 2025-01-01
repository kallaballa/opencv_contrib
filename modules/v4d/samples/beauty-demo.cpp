// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>
#include <opencv2/v4d/v4d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <string>
#include <utility>

using std::vector;
using std::string;

/*!
 * Data structure holding the points for all face landmarks
 */
struct FaceFeatures {
    cv::Rect faceRect_;
    vector<vector<cv::Point2f>> allFeatures_;
    vector<cv::Point2f> allPoints_;
	double scale_ = 1;

	FaceFeatures() {
	}

    FaceFeatures(const cv::Rect& faceRect, const vector<cv::Point2f>& shapes, const double& scale) :
    	faceRect_(cv::Rect(faceRect.x * scale, faceRect.y * scale, faceRect.width * scale, faceRect.height * scale)),
		scale_(scale) {
    	vector<cv::Point2f> chin;
        vector<cv::Point2f> topNose;
        vector<cv::Point2f> bottomNose;
        vector<cv::Point2f> leftEyebrow;
        vector<cv::Point2f> rightEyebrow;
        vector<cv::Point2f> leftEye;
        vector<cv::Point2f> rightEye;
        vector<cv::Point2f> outerLips;
        vector<cv::Point2f> insideLips;

    	/** Copy and scale all features **/
        size_t i = 0;
        // Around Chin. Ear to Ear
        for (i = 0; i <= 16; ++i)
            chin.push_back(shapes[i] * scale);
        // left eyebrow
        for (; i <= 21; ++i)
            leftEyebrow.push_back(shapes[i] * scale);
        // Right eyebrow
        for (; i <= 26; ++i)
            rightEyebrow.push_back(shapes[i] * scale);
        // Line on top of nose
        for (; i <= 30; ++i)
            topNose.push_back(shapes[i] * scale);
        // Bottom part of the nose
        for (; i <= 35; ++i)
            bottomNose.push_back(shapes[i] * scale);
        // Left eye
        for (; i <= 41; ++i)
            leftEye.push_back(shapes[i] * scale);
        // Right eye
        for (; i <= 47; ++i)
            rightEye.push_back(shapes[i] * scale);
        // Lips outer part
        for (; i <= 59; ++i)
            outerLips.push_back(shapes[i] * scale);
        // Lips inside part
        for (; i <= 67; ++i)
            insideLips.push_back(shapes[i] * scale);

        allPoints_.insert(allPoints_.begin(), chin.begin(), chin.end());
        allPoints_.insert(allPoints_.begin(), topNose.begin(), topNose.end());
        allPoints_.insert(allPoints_.begin(), bottomNose.begin(), bottomNose.end());
        allPoints_.insert(allPoints_.begin(), leftEyebrow.begin(), leftEyebrow.end());
        allPoints_.insert(allPoints_.begin(), rightEyebrow.begin(), rightEyebrow.end());
        allPoints_.insert(allPoints_.begin(), leftEye.begin(), leftEye.end());
        allPoints_.insert(allPoints_.begin(), rightEye.begin(), rightEye.end());
        allPoints_.insert(allPoints_.begin(), outerLips.begin(), outerLips.end());
        allPoints_.insert(allPoints_.begin(), insideLips.begin(), insideLips.end());

        allFeatures_ = {chin,
                topNose,
                bottomNose,
                leftEyebrow,
                rightEyebrow,
                leftEye,
                rightEye,
                outerLips,
                insideLips};
    }

    //Concatenates all feature points
    const vector<cv::Point2f>& points() const {
        return allPoints_;
    }

    //Returns all feature points in fixed order
    const vector<vector<cv::Point2f>>& features() const {
        return allFeatures_;
    }

    size_t empty() const {
        return points().empty();
    }

    //based on the detected FaceFeatures it guesses a decent face oval and draws a mask for it.
    void drawFaceOval() const {
        using namespace cv::v4d::nvg;
        cv::RotatedRect rotRect = cv::fitEllipse(points());

        beginPath();
        fillColor(cv::Scalar(255, 255, 255, 255));
        ellipse(rotRect.center.x, rotRect.center.y * 0.875, rotRect.size.width / 2, rotRect.size.height / 1.75);
        rotate(rotRect.angle);
        fill();
    }

    void drawFaceOvalMask() const {
    	cv::v4d::nvg::clearScreen();
    	drawFaceOval();
    }

    void drawEyes() const {
        using namespace cv::v4d::nvg;
        vector<vector<cv::Point2f>> ff = features();
        for (size_t j = 5; j < 7; ++j) {
            beginPath();
            fillColor(cv::Scalar(255, 255, 255, 255));
            moveTo(ff[j][0].x, ff[j][0].y);
            for (size_t k = 1; k < ff[j].size(); ++k) {
                lineTo(ff[j][k].x, ff[j][k].y);
            }
            closePath();
            fill();
        }
    }

    void drawLips() const {
        using namespace cv::v4d::nvg;
        vector<vector<cv::Point2f>> ff = features();
        for (size_t j = 7; j < 8; ++j) {
            beginPath();
            fillColor(cv::Scalar(255, 255, 255, 255));
            moveTo(ff[j][0].x, ff[j][0].y);
            for (size_t k = 1; k < ff[j].size(); ++k) {
                lineTo(ff[j][k].x, ff[j][k].y);
            }
            closePath();
            fill();
        }

	    beginPath();
	    fillColor(cv::Scalar(0, 0, 0, 255));
	    moveTo(ff[8][0].x, ff[8][0].y);
	    for (size_t k = 1; k < ff[8].size(); ++k) {
	        lineTo(ff[8][k].x, ff[8][k].y);
	    }
	    closePath();
	    fill();
    }
    //Draws a mask consisting of eyes and lips areas (deduced from FaceFeatures)
    void drawEyesAndLipsMask() const {
    	cv::v4d::nvg::clearScreen();
        drawEyes();
        drawLips();
    }
};


class FaceFeatureExtractor {
	const cv::Size sz_;
	const float scale_;

	cv::Ptr<cv::FaceDetectorYN> detector_;
	cv::Ptr<cv::face::Facemark> facemark_ = cv::face::createFacemarkLBF();

	std::vector<std::vector<cv::Point2f>> shapes_;
	std::vector<cv::Rect> faceRects_;
	cv::Mat faces_;

public:
	FaceFeatureExtractor(const cv::Size& inputSize, const float& inputScale) : sz_(inputSize), scale_(inputScale) {
    	detector_ = cv::FaceDetectorYN::create("modules/v4d/assets/models/face_detection_yunet_2023mar.onnx", "", inputSize, 0.9, 0.3, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL);
    	facemark_->loadModel("modules/v4d/assets/models/lbfmodel.yaml");
	}

	bool extract(const cv::UMat& inputFrame, FaceFeatures& outputFeatures) {
		shapes_.clear();
		faceRects_.clear();
		//Detect faces in the down-scaled image
		detector_->detect(inputFrame, faces_);

		//Only add the first face
		if(!faces_.empty())
			faceRects_.push_back(cv::Rect(int(faces_.at<float>(0, 0)),
										 int(faces_.at<float>(0, 1)),
										 int(faces_.at<float>(0, 2)),
										 int(faces_.at<float>(0, 3))));

		//find landmarks if faces have been detected
		bool found = !faceRects_.empty() && facemark_->fit(inputFrame, faceRects_, shapes_);
		if(found)
			outputFeatures = FaceFeatures(faceRects_[0], shapes_[0], scale_);
		return found;
	}
};

//adjusts the saturation of a UMat
static void adjust_saturation(const cv::UMat &srcBGR, cv::UMat &dstBGR, float factor, std::vector<cv::UMat>& channel) {
	cv::UMat tmp;
	cvtColor(srcBGR, tmp, cv::COLOR_BGR2HLS);
    split(tmp, channel);
    cv::multiply(channel[2], factor, channel[2]);
    merge(channel, tmp);
    cvtColor(tmp, dstBGR, cv::COLOR_HLS2BGR);
}

using namespace cv::v4d;
using namespace cv::v4d::event;

class FaceFeatureMasksPlan;
class BeautyFilterPlan;
class BeautyDemoPlan : public Plan {
public:
	struct Params {
		//Saturation boost factor for eyes and lips
		float eyesAndLipsSaturation_ = 1.85f;
		//Saturation boost factor for skin
		float skinSaturation_ = 1.35f;
		//Contrast factor skin
		float skinContrast_ = 0.75f;
		//Show input and output side by side
		bool sideBySide_ = false;
		//Scale the video to the window size
		bool stretch_ = true;
		//Show the window in fullscreen mode
		bool fullscreen_ = false;
		//Enable or disable the effect
		bool enabled_ = true;

		size_t frame_cnt = 0;

		enum State {
			ON,
			OFF,
			NOT_DETECTED
		} state_ = ON;
	};

	struct Frames {
		//BGR
		cv::UMat orig_, stitched_, down_, faceOval_, eyesAndLips_, skin_;
		//the frame holding the stitched image if detection went through

		//in split mode the left and right half of the screen
		cv::UMat lhalf_;
		cv::UMat rhalf_;

		//the frame holding the final composed image
		cv::UMat result_;

		//GREY
		cv::UMat faceSkinMaskGrey_, eyesAndLipsMaskGrey_, backgroundMaskGrey_;
	};

	FaceFeatures features_;
private:
	//Key spaces of different state machines of V4D
	using G_ = Global::Keys;
	using S_ = RunState::Keys;
	using K_ = V4D::Keys;
	using M_ = Mouse::Type;

	float scale_ = 1;
	cv::Size size_;
	const cv::Size downSize_ = { 640, 360 };

	static Params params_;
	Frames frames_;
	cv::Ptr<FaceFeatureExtractor> extractor_;

	//think of properties as data-pinholes into one of the v4d runtime's state-machines. Properties are in fact a special kind of edge.
	Property<cv::Rect> vp_ = P<cv::Rect>(K_::VIEWPORT);
	Property<size_t> numWorkers_ = P<size_t>(G_::WORKERS_STARTED);
	Property<size_t> workerIndex_ = P<size_t>(S_::WORKER_INDEX);

	//A special kind of edge used to signal user input events
	Event<Mouse> pressEvents_ = E<Mouse>(M_::PRESS);

	static void prepare_frames(const cv::UMat& framebuffer, const cv::Size& downSize, Frames& frames) {
		cvtColor(framebuffer, frames.orig_, cv::COLOR_RGBA2BGR);
		cv::resize(frames.orig_, frames.down_, downSize);
		frames.orig_.copyTo(frames.stitched_);
	}

	static void compose_result(const cv::Rect& vp, const cv::UMat& src, Frames& frames, const Params& params) {
		if (params.sideBySide_) {
			//create side-by-side view with a result
			cv::resize(frames.orig_, frames.lhalf_, cv::Size(0, 0), 0.5, 0.5);
			cv::resize(src, frames.rhalf_, cv::Size(0, 0), 0.5, 0.5);

			frames.result_ = cv::Scalar::all(0);
			frames.lhalf_.copyTo(frames.result_(cv::Rect(0, vp.height / 2.0, frames.lhalf_.size().width, frames.lhalf_.size().height)));
			frames.rhalf_.copyTo(frames.result_(cv::Rect(vp.width / 2.0, vp.height / 2.0, frames.lhalf_.size().width, frames.lhalf_.size().height)));
		} else {
			src.copyTo(frames.result_);
		}
	}

	//sub-plans
	cv::Ptr<FaceFeatureMasksPlan> prepareFeatureMasksPlan_;
	cv::Ptr<BeautyFilterPlan> beautyFilterPlan_;
public:
	BeautyDemoPlan() {
		//construct sub-plans only in the constructor
		prepareFeatureMasksPlan_ = _sub<FaceFeatureMasksPlan>(this, features_, frames_);
		beautyFilterPlan_ = _sub<BeautyFilterPlan>(this, params_, frames_);
	}

	//at the moment gui is an exception from the rule that a Plan only implements the graph, because it runs on the display thread. in the future it should implement its own graph which would run in concurrent to the main algorithm - locking shared state where neccessary
	void gui() override {
		imgui([](Params& params){
			using namespace ImGui;
			Begin("Effect");
			Text("Display");
			Checkbox("Side by side", &params.sideBySide_);
			Checkbox("Stetch", &params.stretch_);

			if(Button("Fullscreen")) {
				params.fullscreen_ = !params.fullscreen_;
			};

			Text("Face Skin");
			SliderFloat("Saturation", &params.skinSaturation_, 0.0f, 10.0f);
			SliderFloat("Contrast", &params.skinContrast_, 0.0f, 2.0f);
			Text("Eyes and Lips");
			SliderFloat("Saturation ", &params.eyesAndLipsSaturation_, 0.0f, 10.0f);
			End();

			ImVec4 color;
			string text;
			switch(params.state_) {
				case Params::ON:
					text = "On";
					color = ImVec4(0.25, 1.0, 0.25, 1.0);
					break;
				case Params::OFF:
					text = "Off";
					color = ImVec4(0.25, 0.25, 1.0, 1.0);
					break;
				case Params::NOT_DETECTED:
					color = ImVec4(1.0, 0.25, 0.25, 1.0);
					text ="Not detected";
					break;
				default:
					CV_Assert(false);
			}

			Begin("Status");
			TextColored(color, text.c_str());
			End();
		}, params_);
	}

	void setup() override {
		//emits a node the performs and assignment. F creates and edge that reads the result of a funcion call-
		assign(RW(size_), F(&cv::Rect::size, vp_));
		assign(RW(scale_), F(aspect_preserving_scale, R(size_), R(downSize_)));
		//emits a node that calls a contructor
		construct(RW(extractor_), R(downSize_), R(scale_));
	}

	void infer() override {
		//emits a node setting the states for "fullscreen" and "stretching" during execution of the graph reading values from the shared data by copying it.
		set(_(K_::FULLSCREEN, CS(params_.fullscreen_)),
			_(K_::STRETCHING, CS(params_.stretch_)));

		//create a node the will capture video
		capture();
		fb(prepare_frames, R(downSize_), RW(frames_));

		// a branch is basically a graph node that decides what graph node to run next.
		branch(
				//edge-calls result in edge-objects which support many operators.
				RWS(params_.enabled_) = IF(
											///query mouse release events
											F(&Mouse::List::empty, pressEvents_),
											CS(params_.enabled_),
											!CS(params_.enabled_)
										))
			//every numWorkers_ frames redect the face features.
			->branch(++RWS(params_.frame_cnt) % numWorkers_ == workerIndex_)
				->branch(!(F(&FaceFeatureExtractor::extract, RW(extractor_), R(frames_.down_), RW(features_))));
					//Set a shared state that will be displayed on-screen.
					assign(RWS(params_.state_), V(Params::NOT_DETECTED))
				->endBranch()
			->endBranch()
			->branch(!(F(&FaceFeatures::empty, R(features_))));
				assign(RWS(params_.state_), V(Params::ON))
				//run inference on the sub-plans which will emit their own nodes
				->subInfer(prepareFeatureMasksPlan_)
				->subInfer(beautyFilterPlan_)
			->endBranch()
		->elseBranch()
			->assign(RWS(params_.state_), V(Params::OFF))
		->endBranch();

		plain(compose_result, vp_, R(frames_.stitched_), RW(frames_), CS(params_))
		->fb<1>(cv::cvtColor, R(frames_.result_), V(cv::COLOR_BGR2RGBA), V(0), V(cv::ALGO_HINT_DEFAULT));
	}
};

//A sub-plan the provides face features
class FaceFeatureMasksPlan : public Plan {
	const FaceFeatures& inputFeatures_;
	BeautyDemoPlan::Frames& inputOutputFrames_;
public:
	FaceFeatureMasksPlan(const FaceFeatures& inputFeatures, BeautyDemoPlan::Frames& inputOutputFrames) :
		inputFeatures_(inputFeatures), inputOutputFrames_(inputOutputFrames) {
	}

	static void prepare_masks(BeautyDemoPlan::Frames& frames) {
		//Create the skin mask
		cv::subtract(frames.faceOval_, frames.eyesAndLipsMaskGrey_, frames.faceSkinMaskGrey_);
		//Create the background mask
		cv::bitwise_not(frames.faceOval_, frames.backgroundMaskGrey_);
	}

	void infer() override {
		//context-call provides a nanovg context to the node emitteed
		nvg(&FaceFeatures::drawFaceOvalMask, R(inputFeatures_))
		//context-call provides a cv::UMat representation of the framebuffer to the node emitteed
		->fb(cv::cvtColor, RW(inputOutputFrames_.faceOval_), V(cv::COLOR_BGRA2GRAY), V(0), V(cv::ALGO_HINT_DEFAULT))
		->nvg(&FaceFeatures::drawEyesAndLipsMask, R(inputFeatures_))
		->fb(cv::cvtColor, RW(inputOutputFrames_.eyesAndLipsMaskGrey_), V(cv::COLOR_BGRA2GRAY), V(0), V(cv::ALGO_HINT_DEFAULT))
		//
		->plain(prepare_masks, RW(inputOutputFrames_));
	}
};

//a sub-plan implementing the actual beauty-filter.
class BeautyFilterPlan : public Plan {
	const BeautyDemoPlan::Params& inputParams_;
	BeautyDemoPlan::Frames& inputOutputFrames_;

	//Blender (used to put the different face parts back together)
	cv::Ptr<cv::detail::MultiBandBlender> blender_ = new cv::detail::MultiBandBlender(true, 5);
	std::vector<cv::UMat> channels_;
	cv::UMat stitchedFloat_;

	static void adjust_face_features(BeautyDemoPlan::Frames& frames, std::vector<cv::UMat>& channels, const BeautyDemoPlan::Params& params) {
		cv::UMat tmp;
		//boost saturation of eyes and lips
		adjust_saturation(frames.orig_,  frames.eyesAndLips_, params.eyesAndLipsSaturation_, channels);
		//reduce skin contrast
		multiply(frames.orig_, cv::Scalar::all(params.skinContrast_), frames.skin_);
		//fix skin brightness
		add(frames.skin_, cv::Scalar::all((1.0 - params.skinContrast_) / 2.0) * 255.0, tmp);
		//boost skin saturation
		adjust_saturation(tmp, frames.skin_, params.skinSaturation_, channels);
	}

	static void stitch_face(cv::Ptr<cv::detail::MultiBandBlender>& bl, BeautyDemoPlan::Frames& frames, cv::UMat& stitchedFloat) {
		CV_Assert(!frames.skin_.empty());
		CV_Assert(!frames.eyesAndLips_.empty());
		//piece it all together
		bl->prepare(cv::Rect(0, 0, frames.skin_.cols, frames.skin_.rows));
		bl->feed(frames.skin_, frames.faceSkinMaskGrey_, cv::Point(0, 0));
		bl->feed(frames.orig_, frames.backgroundMaskGrey_, cv::Point(0, 0));
		bl->feed(frames.eyesAndLips_, frames.eyesAndLipsMaskGrey_, cv::Point(0, 0));
		bl->blend(stitchedFloat, cv::UMat());
		CV_Assert(!stitchedFloat.empty());
		stitchedFloat.convertTo(frames.stitched_, CV_8U, 1.0);
	}
public:
	BeautyFilterPlan(const BeautyDemoPlan::Params& intputParams, BeautyDemoPlan::Frames& inputOutputFrames) :
		inputParams_(intputParams), inputOutputFrames_(inputOutputFrames) {
	}

	void infer() override {
		plain(adjust_face_features, RW(inputOutputFrames_), RW(channels_), CS(inputParams_))
		->plain(stitch_face, RW(blender_), RW(inputOutputFrames_), RW(stitchedFloat_));
	}
};
//Shared data
BeautyDemoPlan::Params BeautyDemoPlan::params_;

int main(int argc, char **argv) {
	if (argc != 2) {
        std::cerr << "Usage: beauty-demo <input-video-file>" << std::endl;
        exit(1);
    }

	cv::Rect viewport(0, 0, 1920, 1080);
	cv::Ptr<V4D> runtime = V4D::init(viewport, "Beautification Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI, ConfigFlags::DEFAULT, DebugFlags::DEFAULT);
	//V4D provides a source, sink system which is use mostly but not exclusively used with video data.
	auto src = Source::make(runtime, argv[1]);
    runtime->setSource(src);
    Plan::run<BeautyDemoPlan>(0);

    return 0;
}
