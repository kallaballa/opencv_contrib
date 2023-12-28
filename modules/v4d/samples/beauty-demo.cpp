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



static void extract_face_features(const float& scale, cv::Ptr<cv::FaceDetectorYN>& detector, cv::Ptr<cv::face::Facemark>& facemark, const cv::UMat& down, FaceFeatures& features, bool& found) {
	std::vector<std::vector<cv::Point2f>> shapes;
	std::vector<cv::Rect> faceRects;
	cv::Mat faces;
	//Detect faces in the down-scaled image
	detector->detect(down, faces);
	//Only add the first face
	if(!faces.empty())
		faceRects.push_back(cv::Rect(int(faces.at<float>(0, 0)),
 									 int(faces.at<float>(0, 1)),
									 int(faces.at<float>(0, 2)),
									 int(faces.at<float>(0, 3))));

	//find landmarks if faces have been detected
	found = !faceRects.empty() && facemark->fit(down, faceRects, shapes);
	if(found)
		features = FaceFeatures(faceRects[0], shapes[0], scale);
}


//adjusts the saturation of a UMat
static void adjust_saturation(const cv::UMat &srcBGR, cv::UMat &dstBGR, float factor, std::vector<cv::UMat>& channel) {
	cv::UMat tmp;
	cvtColor(srcBGR, tmp, cv::COLOR_BGR2HLS);
    split(tmp, channel);
    cv::multiply(channel[2], factor, channel[2]);
    merge(channel, tmp);
    cvtColor(tmp, dstBGR, cv::COLOR_HLS2BGR);
}

static void present(cv::UMat& framebuffer, const cv::UMat& result) {
	cvtColor(result, framebuffer, cv::COLOR_BGR2BGRA);
}

using namespace cv::v4d;

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

	//results of face detection and facemark
	struct Face {
		bool found_ = false;
		FaceFeatures features_;
	};
private:
	using K = V4D::Keys;

	Frames frames_;
	Face face_;
	cv::Size downSize_;
	Params& params_;

	cv::Ptr<cv::face::Facemark> facemark_ = cv::face::createFacemarkLBF();
	//Face detector
	cv::Ptr<cv::FaceDetectorYN> detector_;

	Property<cv::Rect> vp_ = GET<cv::Rect>(V4D::Keys::VIEWPORT);


	static void prepare_frames(const cv::UMat& framebuffer, const cv::Size& downSize, Frames& frames) {
		cvtColor(framebuffer, frames.orig_, cv::COLOR_BGRA2BGR);
		cv::resize(frames.orig_, frames.down_, downSize);
		frames.orig_.copyTo(frames.stitched_);
	}

	static bool is_enabled(Params& params) {
		using namespace cv::v4d::event;
		if(consume(Mouse::Type::PRESS)) {
			params.enabled_ = !params.enabled_;
		}

		return params.enabled_;
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

	static void set_state(Params& params, Params::State& state) {
		params.state_ = state;
	}

	cv::Ptr<FaceFeatureMasksPlan> prepareFeatureMasksPlan_;
	cv::Ptr<BeautyFilterPlan> beautyFilterPlan_;
public:
	BeautyDemoPlan(Params& params) : params_(params) {
		_shared(params_);
		prepareFeatureMasksPlan_ = _sub<FaceFeatureMasksPlan>(this, face_.features_, frames_, params_);
		beautyFilterPlan_ = _sub<BeautyFilterPlan>(this, face_.features_, frames_, params_);
	}

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
		plain([](const cv::Rect& vp, cv::Size& downSize, cv::Ptr<cv::face::Facemark>& facemark, cv::Ptr<cv::FaceDetectorYN>& detector) {
	    	int w = vp.width;
	    	int h = vp.height;
	    	downSize = { std::min(w, std::max(640, int(round(w / 2.0)))), std::min(h, std::max(360, int(round(h / 2.0)))) };
	    	detector = cv::FaceDetectorYN::create("modules/v4d/assets/models/face_detection_yunet_2023mar.onnx", "", downSize, 0.9, 0.3, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL);
	    	facemark->loadModel("modules/v4d/assets/models/lbfmodel.yaml");
		}, vp_, RW(downSize_), RW(facemark_), RW(detector_));
	}

	void infer() override {
		set(K::FULLSCREEN, m_(&Params::fullscreen_), R_SC(params_));
		set(K::STRETCHING, m_(&Params::stretch_), R_SC(params_));

		capture()
		->fb(prepare_frames, R(downSize_), RW(frames_));

		branch(is_enabled, RW_S(params_))
			->plain([](const cv::Size& downSize, auto& detector, auto& facemark, const Frames& frames, Face& face){
				extract_face_features(frames.orig_.size().width / float(downSize.width), detector, facemark, frames.down_, face.features_, face.found_);
			}, R(downSize_), RW(detector_), RW(facemark_), R(frames_), RW(face_))
			->branch(isTrue_, R(face_.found_))
				->subInfer(prepareFeatureMasksPlan_)
				->subInfer(beautyFilterPlan_)
				->plain(set_state, RW_S(params_), VAL(Params::ON))
			->elseBranch()
				->plain(set_state, RW_S(params_), VAL(Params::NOT_DETECTED))
			->endBranch()
		->elseBranch()
			->plain(set_state, RW_S(params_), VAL(Params::OFF))
		->endBranch();

		plain(compose_result, vp_, R(frames_.stitched_), RW(frames_), R_SC(params_))
		->fb(present, R(frames_.result_));
	}
};


class FaceFeatureMasksPlan : public Plan {
	FaceFeatures& baseFeatures_;
	BeautyDemoPlan::Frames& baseFrames_;
	BeautyDemoPlan::Params& baseParams_;
public:
	FaceFeatureMasksPlan(FaceFeatures& baseFeatures, BeautyDemoPlan::Frames& baseFrames, BeautyDemoPlan::Params& baseParams)
	: baseFeatures_(baseFeatures), baseFrames_(baseFrames), baseParams_(baseParams) {
	}

	static void prepare_masks(BeautyDemoPlan::Frames& frames, const BeautyDemoPlan::Params& params) {
		//Create the skin mask
		cv::subtract(frames.faceOval_, frames.eyesAndLipsMaskGrey_, frames.faceSkinMaskGrey_);
		//Create the background mask
		cv::bitwise_not(frames.faceOval_, frames.backgroundMaskGrey_);
	}

	void infer() override {
		nvg(m_(&FaceFeatures::drawFaceOvalMask), R(baseFeatures_))
		->fb([](const cv::UMat& framebuffer, cv::UMat& faceOval) {
			cvtColor(framebuffer, faceOval, cv::COLOR_BGRA2GRAY);
		}, RW(baseFrames_.faceOval_))
		->nvg(m_(&FaceFeatures::drawEyesAndLipsMask), R(baseFeatures_))
		->fb([](const cv::UMat &framebuffer, cv::UMat& eyesAndLipsMaskGrey) {
			cvtColor(framebuffer, eyesAndLipsMaskGrey, cv::COLOR_BGRA2GRAY);
		}, RW(baseFrames_.eyesAndLipsMaskGrey_))
		->plain(prepare_masks, RW(baseFrames_), R_SC(baseParams_));
	}
};

class BeautyFilterPlan : public Plan {
	FaceFeatures& baseFeatures_;
	BeautyDemoPlan::Frames& baseFrames_;
	BeautyDemoPlan::Params& baseParams_;
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
	BeautyFilterPlan(FaceFeatures& baseFeatures, BeautyDemoPlan::Frames& baseFrames, BeautyDemoPlan::Params& baseParams)
	: baseFeatures_(baseFeatures), baseFrames_(baseFrames), baseParams_(baseParams) {
	}

	void infer() override {
		plain(adjust_face_features, RW(baseFrames_), RW(channels_), R_SC(baseParams_))
		->plain(stitch_face, RW(blender_), RW(baseFrames_), RW(stitchedFloat_));
	}
};

int main(int argc, char **argv) {
	if (argc != 2) {
        std::cerr << "Usage: beauty-demo <input-video-file>" << std::endl;
        exit(1);
    }

	cv::Rect viewport(0, 0, 1280, 720);
	BeautyDemoPlan::Params params;
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Beautification Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI, ConfigFlags::DEFAULT, DebugFlags::LOWER_WORKER_PRIORITY | DebugFlags::PRINT_CONTROL_FLOW);
    auto src = Source::make(runtime, argv[1]);
    runtime->setSource(src);
    Plan::run<BeautyDemoPlan>(0, params);

    return 0;
}
