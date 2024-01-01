// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include <sstream>
#include <limits>


using std::string;
using std::vector;
using std::istringstream;

using namespace cv::v4d;

class FontDemoPlan : public Plan {
	inline static const cv::Scalar_<float> INITIAL_COLOR = cv::v4d::convert_pix(cv::Scalar(0.15 * 180.0, 128, 255, 255), cv::COLOR_HLS2RGB);
	static struct Params {
		float minStarSize_ = 0.5f;
		float maxStarSize_ = 1.5f;
		int minStarCount_ = 1000;
		int maxStarCount_ = 3000;
		float starAlpha_ = 0.3f;

		float fontSize_ = 0.0f;
		cv::Scalar_<float> textColor_ = INITIAL_COLOR / 255.0;
		float warpRatio_ = 1.0f;
		bool updateStars_ = true;
		bool updatePerspective_ = true;
		double timeOffset_ = 0.0f;
	} params_;

    //BGRA
	inline static cv::UMat stars_;
	cv::UMat warped_, text_;
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

    Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void gui() override {
        imgui([](Params& params) {
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

    void setup() override {
		branch(BranchType::ONCE, always_)
			->plain([](const cv::Rect& vp, TextVars& textVars, Params& params) {
				//The text to display
				string txt = cv::getBuildInformation();
				//Save the text to a vector
				std::istringstream iss(txt);

				double fontSize = hypot(vp.width, vp.height) / 60.0;
				for (std::string line; std::getline(iss, line); ) {
					textVars.lines_.push_back(line);
				}
				textVars.numLines_ = textVars.lines_.size();
				textVars.textHeight_ = (textVars.numLines_ * fontSize * params.warpRatio_);
				params.fontSize_ = fontSize;
			}, vp_, RW_S(textVars_), RW_S(params_))
		->endBranch();
    }

    void infer() override {
    	branch([](const Params& params) {
			return params.updateStars_;
		}, R_SC(params_))
			->nvg([](const cv::Rect& vp, cv::RNG& rng, const Params& params) {
				using namespace cv::v4d::nvg;
				clearScreen();
				//draw stars
				int numStars = rng.uniform(params.minStarCount_, params.maxStarCount_);
				for(int i = 0; i < numStars; ++i) {
					beginPath();
					const auto size = rng.uniform(params.minStarSize_, params.maxStarSize_);
					strokeWidth(size);
					strokeColor(cv::Scalar(255, 255, 255, params.starAlpha_ * 255.0f));
					circle(rng.uniform(0, vp.width) , rng.uniform(0, vp.height), size / 2.0);
					stroke();
				}
			}, vp_, RW(rng_), R_SC(params_))
			->fb([](const cv::UMat& framebuffer, cv::UMat& stars, Params& params) {
				params.updateStars_ = false;
				framebuffer.copyTo(stars);
			}, RW_S(stars_), RW_S(params_))
		->endBranch();

		branch([](const Params& params){
			return params.updatePerspective_;
		}, R_SC(params_))
			->plain([](const cv::Rect& vp, cv::Mat& tm, Params& params) {
				//Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
				vector<cv::Point2f> quad1 = { cv::Point2f(0, 0),
						cv::Point2f(vp.width, 0), cv::Point2f(vp.width,
								vp.height), cv::Point2f(0, vp.height) };
				float l = (vp.width - (vp.width * params.warpRatio_)) / 2.0;
				float r = vp.width - l;

				vector<cv::Point2f> quad2 = { cv::Point2f(l, 0.0f),
						cv::Point2f(r, 0.0f), cv::Point2f(vp.width,
								vp.height), cv::Point2f(0, vp.height) };
				tm = cv::getPerspectiveTransform(quad1, quad2);
				params.updatePerspective_ = false;
			}, vp_, RW_S(tm_), RW_S(params_))
		->endBranch();

		nvg([](const cv::Rect& vp, double& translateY, double& y, const TextVars& textVars, const Params& params) {
			double time = (cv::getTickCount() / cv::getTickFrequency()) - params.timeOffset_;
			//How many pixels to translate the text up.
			translateY = (double(vp.height) - round(time * 30.0));
			using namespace cv::v4d::nvg;
			clearScreen();
			fontSize(params.fontSize_);
			fontFace("sans-bold");
			fillColor(params.textColor_ * 255);
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

			/** only draw lines that are visible **/
			translate(0, translateY);

			for (size_t i = 0; i < textVars.lines_.size(); ++i) {
				y = (i * params.fontSize_);
				if (y + translateY < textVars.textHeight_ && y + translateY + params.fontSize_ > 0) {
					text(vp.width / 2.0, y, textVars.lines_[i].c_str(), textVars.lines_[i].c_str() + textVars.lines_[i].size());
				}
			}
		}, vp_, RW(translateY_), RW(y_), R_SC(textVars_), R_SC(params_));

		fb([](const cv::UMat& framebuffer, cv::UMat& text) {
			framebuffer.copyTo(text);
		}, RW(text_));

		clear();

		fb([](cv::UMat& framebuffer, const cv::UMat& text, cv::UMat& warped, const cv::UMat& stars, const cv::Mat& tm) {
			cv::warpPerspective(text, warped, tm, text.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
			cv::UMat blur;
			cv::UMat mask;
			cv::boxFilter(warped, blur, -1, cv::Size(5, 5), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
			cv::threshold(blur, mask, 1, 255, cv::THRESH_BINARY);
			cvtColor(mask, mask, cv::COLOR_BGRA2GRAY);
			cv::UMat s = stars.clone();
			s.setTo(cv::Scalar::all(0), mask);
			cv::add(s, warped, framebuffer);
		},  R(text_), RW(warped_), R_SC(stars_), R_SC(tm_));

		plain([](const double& translateY, const TextVars& textVars, Params& params) {
			if(-translateY > textVars.textHeight_) {
				//reset the timeOffset once the text is out of the picture
				params.timeOffset_ = cv::getTickCount() / cv::getTickFrequency();
			}
		}, R(translateY_), R_SC(textVars_), RW_S(params_));
    }
};

FontDemoPlan::Params FontDemoPlan::params_;
FontDemoPlan::TextVars FontDemoPlan::textVars_;

int main() {
	cv::Rect viewport(0, 0, 1280, 720);
	cv::Ptr<V4D> runtime = V4D::init(viewport, cv::Size(2560, 1440),  "Font Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
	Plan::run<FontDemoPlan>(0);
    return 0;
}
