// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <limits>

using std::string;
using std::vector;
using std::istringstream;

using namespace cv::v4d;

class FontDemoPlan : public Plan {
	inline const static cv::Scalar_<float> INITIAL_COLOR = cv::v4d::colorConvert(cv::Scalar(0.15 * 180.0, 128, 255, 255), cv::COLOR_HLS2RGB);
	static struct Params {
		float minStarSize_ = 0.5f;
		float maxStarSize_ = 1.5f;
		int minStarCount_ = 1000;
		int maxStarCount_ = 3000;
		float starAlpha_ = 0.3f;

		float fontSize_ = 0.0f;
		cv::Scalar_<float> textColor_ = INITIAL_COLOR / 255.0;
		float warpRatio_ = 1.0f/3.0f;
		bool updateStars_ = true;
		bool updatePerspective_ = true;
		double timeOffset_ = 0.0f;
	} params_;

    //BGRA
	inline static cv::UMat stars_;
	cv::UMat warped_;
	//transformation matrix
    inline static cv::Mat tm_;

    static struct TextVars {
    	//the text to display
    	vector<string> lines_;
    	//Total number of lines in the text
    	int32_t numLines_ = 0;
    	//Height of the text in pixels
    	double textHeight_ = 0;
    } textVars_;

    //y-value of the current line
    double y_ = 0;

    double translateY_ = 0;

    cv::RNG rng_ = cv::getTickCount();
public:
	using Plan::Plan;

	FontDemoPlan(const cv::Rect& vp) : Plan(vp) {
		Global::registerShared(params_);
		Global::registerShared(textVars_);
		Global::registerShared(tm_);
		Global::registerShared(stars_);
	}

	void gui(cv::Ptr<V4D> window) override {
        window->imgui([](cv::Ptr<V4D> win, Params& params) {
        	CV_UNUSED(win);
        	using namespace ImGui;
            Begin("Effect");
            Text("Text Crawl");
            SliderFloat("Font Size", &params.fontSize_, 1.0f, 100.0f);
            if(SliderFloat("Warp Ratio", &params.warpRatio_, 0.1f, 1.0f))
                params.updatePerspective_ = true;
            ColorPicker4("Text Color", params.textColor_.val);
            Text("Stars");

            if(SliderFloat("Min Star Size", &params.minStarSize_, 0.5f, 1.0f))
                params.updateStars_ = true;
            if(SliderFloat("Max Star Size", &params.maxStarSize_, 1.0f, 10.0f))
            	params.updateStars_ = true;
            if(SliderInt("Min Star Count", &params.minStarCount_, 1, 1000))
            	params.updateStars_ = true;
            if(SliderInt("Max Star Count", &params.maxStarCount_, 1000, 5000))
            	params.updateStars_ = true;
            if(SliderFloat("Min Star Alpha", &params.starAlpha_, 0.2f, 1.0f))
            	params.updateStars_ = true;
            End();
        }, params_);
    }

    void setup(cv::Ptr<V4D> window) override {
		window->branch(BranchType::ONCE, always_)
			->plain([](const cv::Size& sz, TextVars& textVars, Params& params) {
				//The text to display
				string txt = cv::getBuildInformation();
				//Save the text to a vector
				std::istringstream iss(txt);

				double fontSize = hypot(sz.width, sz.height) / 60.0;
				for (std::string line; std::getline(iss, line); ) {
					textVars.lines_.push_back(line);
				}
				textVars.numLines_ = textVars.lines_.size();
				textVars.textHeight_ = (textVars.numLines_ * fontSize);
				params.fontSize_ = fontSize;
			}, R(size()), RW_C(textVars_), RW_C(params_))
		->endBranch();
    }

    void infer(cv::Ptr<V4D> window) override {
		window->branch(BranchType::SINGLE, [](const Params params) {
			return params.updateStars_;
		}, R_C(params_))
			->nvg([](const cv::Size& sz, cv::RNG& rng, const Params params) {
				using namespace cv::v4d::nvg;
				clear();

				//draw stars
				int numStars = rng.uniform(params.minStarCount_, params.maxStarCount_);
				for(int i = 0; i < numStars; ++i) {
					beginPath();
					const auto size = rng.uniform(params.minStarSize_, params.maxStarSize_);
					strokeWidth(size);
					strokeColor(cv::Scalar(255, 255, 255, params.starAlpha_ * 255.0f));
					circle(rng.uniform(0, sz.width) , rng.uniform(0, sz.height), size / 2.0);
					stroke();
				}
			}, R(size()), RW(rng_), R_C(params_))
			->fb([](const cv::UMat& framebuffer, cv::UMat& stars, Params& params) {
				params.updateStars_ = false;

				Global::Scope scope(stars);
				framebuffer.copyTo(stars);
			}, RW_S(stars_), RW_C(params_))
		->endBranch();

		window->branch(BranchType::SINGLE, [](const Params params){
			return params.updatePerspective_;
		}, R_C(params_))
			->plain([](const cv::Size& sz, cv::Mat& tm, Params& params) {
				//Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
				vector<cv::Point2f> quad1 = {cv::Point2f(0,0),cv::Point2f(sz.width,0),
						cv::Point2f(sz.width,sz.height),cv::Point2f(0,sz.height)};
				float l = (sz.width - (sz.width * params.warpRatio_)) / 2.0;
				float r = sz.width - l;

				vector<cv::Point2f> quad2 = {cv::Point2f(l, 0.0f),cv::Point2f(r, 0.0f),
						cv::Point2f(sz.width,sz.height), cv::Point2f(0,sz.height)};
				tm = cv::getPerspectiveTransform(quad1.data(), quad2.data());
				params.updatePerspective_ = false;
			}, R(size()), RW_C(tm_), RW_C(params_))
		->endBranch();

		window->nvg([](const cv::Size& sz, double& translateY, double& y, const TextVars textVars, const Params params) {
			double time = (cv::getTickCount() / cv::getTickFrequency()) - params.timeOffset_;
			//How many pixels to translate the text up.
			translateY = double(sz.height) - round(time * 20.0);
			using namespace cv::v4d::nvg;
			clear();
			fontSize(params.fontSize_);
			fontFace("sans-bold");
			fillColor(params.textColor_ * 255);
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

			/** only draw lines that are visible **/
			translate(0, translateY);

			for (size_t i = 0; i < textVars.lines_.size(); ++i) {
				y = (i * params.fontSize_);
				if (y + translateY < textVars.textHeight_ && y + translateY + params.fontSize_ > 0) {
					text(sz.width / 2.0, y, textVars.lines_[i].c_str(), textVars.lines_[i].c_str() + textVars.lines_[i].size());
				}
			}
		}, R(size()), RW(translateY_), RW(y_), R_C(textVars_), R_C(params_));

		window->fb([](cv::UMat& framebuffer, cv::UMat& warped, cv::UMat& stars, cv::Mat tm) {
			cv::warpPerspective(framebuffer, warped, tm, framebuffer.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

			Global::Scope scope(stars);
			cv::add(stars.clone(), warped, framebuffer);
		},  RW(warped_), RW_S(stars_), RW_C(tm_));

		window->plain([](const double& translateY, const TextVars textVars, Params& params) {
			if(-translateY > textVars.textHeight_) {
				//reset the timeOffset once the text is out of the picture
				params.timeOffset_ = cv::getTickCount() / cv::getTickFrequency();
			}
		}, R(translateY_), R_C(textVars_), RW_C(params_));
    }
};

FontDemoPlan::Params FontDemoPlan::params_;
FontDemoPlan::TextVars FontDemoPlan::textVars_;

int main() {
	cv::Rect viewport(0, 0, 1280, 720);
	cv::Ptr<V4D> window = V4D::make(viewport.size(), "Font Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI);

	window->run<FontDemoPlan>(0, viewport);
    return 0;
}
