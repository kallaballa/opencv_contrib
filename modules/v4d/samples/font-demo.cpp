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

inline static const cv::Scalar_<float> INITIAL_COLOR = cv::v4d::convert_pix<cv::COLOR_HLS2RGB_FULL, cv::Vec3b, cv::Scalar_<float>>(cv::Vec3b(35, 127, 255), 1.0/255.0);

struct TextRenderer {
	cv::Scalar_<float> color_ = INITIAL_COLOR;
	//the text to display
 	vector<string> lines_;
 	//Height of the text in pixels
 	double height_ = 0;
    //y offset of the current line
     double lineOffsetY_ = 0;

     double textOffsetY_ = 0;

     float fontSize_;

     cv::UMat rendering_;
public:
    TextRenderer(const float& fontSize = 1) : fontSize_(fontSize) {
		//The text to display
		string txt = cv::getBuildInformation();
		//Save the text to a vector
		std::istringstream iss(txt);

		for (std::string line; std::getline(iss, line); ) {
			lines_.push_back(line);
		}
		height_ = (lines_.size() * fontSize);
	}

    void draw(const cv::Rect& vp, const double& timeOffset) {
		double time = seconds() - timeOffset;
		//How many pixels to translate the text up.
		textOffsetY_ = (double(vp.height) - round(time * 30.0));
		using namespace cv::v4d::nvg;
		clearScreen();
		fontSize(fontSize_);
		fontFace("sans-bold");
		fillColor(convert_pix<cv::COLOR_BGR2RGBA>(color_, 255.0));
		textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

		/** only draw lines that are visible **/
		translate(0, textOffsetY_);

		for (size_t i = 0; i < lines_.size(); ++i) {
			lineOffsetY_ = (i * fontSize_);
			if (lineOffsetY_ + textOffsetY_ < vp.height && lineOffsetY_ + textOffsetY_ + fontSize_ > 0) {
				text(vp.width / 2.0, lineOffsetY_, lines_[i].c_str(), lines_[i].c_str() + lines_[i].size());
			}
		}
    }
};

class StarsRenderer {
    cv::RNG rng_ = cv::getTickCount();
public:
    cv::UMat rendering_;
    float minStarSize_ = 0.75f;
	float maxStarSize_ = 2.0f;
	int minStarCount_ = 1000;
	int maxStarCount_ = 2000;
	float maxStarAlpha_ = 0.25f;
	bool update_ = true;

	void draw(const cv::Rect& vp) {
		using namespace cv::v4d::nvg;
		clearScreen();
		//draw stars
		int numStars = rng_.uniform(minStarCount_, maxStarCount_);
		for(int i = 0; i < numStars; ++i) {
			beginPath();
			const auto size = rng_.uniform(minStarSize_, maxStarSize_);
			const auto alpha = rng_.uniform(0.05f, maxStarAlpha_);
			strokeWidth(size);
			strokeColor(cv::Scalar(255, 255, 255, alpha * 255.0f));
			circle(rng_.uniform(0, vp.width) , rng_.uniform(0, vp.height), size / 2.0);
			stroke();
		}
		update_ = false;
	}
};

struct Warp {
	cv::Mat transMatrix_;
public:
	cv::UMat rendering_;
	float warpRatio_ = 0.33f;
	bool update_ = true;

	void calculate(const cv::Rect& vp) {
		//Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
		vector<cv::Point2f> quad1 = { cv::Point2f(0, 0),
				cv::Point2f(vp.width, 0), cv::Point2f(vp.width,
						vp.height), cv::Point2f(0, vp.height) };
		float l = (vp.width - (vp.width * warpRatio_)) / 2.0;
		float r = vp.width - l;

		vector<cv::Point2f> quad2 = { cv::Point2f(l, 0.0f),
				cv::Point2f(r, 0.0f), cv::Point2f(vp.width,
						vp.height), cv::Point2f(0, vp.height) };
		transMatrix_ = cv::getPerspectiveTransform(quad1, quad2);
	}

	void perform(const cv::UMat& textRendering, const cv::UMat& starsRendering, cv::UMat& result) {
		cv::UMat tmp;
		cv::UMat blur_;
		cv::UMat mask_;
		cv::warpPerspective(textRendering.clone(), tmp, transMatrix_.clone(), textRendering.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
		rendering_ = tmp.clone();
		cv::boxFilter(tmp, blur_, -1, cv::Size(5, 5), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
		cv::threshold(blur_, mask_, 1, 255, cv::THRESH_BINARY);
		cvtColor(mask_, mask_, cv::COLOR_BGRA2GRAY);
		cv::UMat s = starsRendering.clone();
		s.setTo(cv::Scalar::all(0), mask_);
		cv::add(s, tmp, result);
	}
};

class FontDemoPlan : public Plan {
	static TextRenderer text_;
	static StarsRenderer stars_;
	static Warp warp_;
	static double timeOffset_;

    Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	void gui() override {
        imgui([](TextRenderer& text, StarsRenderer& stars, Warp& warp) {
        	using namespace ImGui;
            Begin("Effect");
            Text("Text Crawl");
            SliderFloat("Font Size", &text.fontSize_, 1.0f, 100.0f);
            if(SliderFloat("Warp Ratio", &warp.warpRatio_, 0.1f, 1.0f))
                warp_.update_ = true;
            ColorPicker4("Text Color", text.color_.val);
            Text("Stars");
            if(SliderFloat("Min Star Size", &stars.minStarSize_, 0.0f, 1.0f))
            	stars.update_ = true;
            if(SliderFloat("Max Star Size", &stars.maxStarSize_, 0.0f, 10.0f))
            	stars.update_ = true;
            if(SliderInt("Min Star Count", &stars.minStarCount_, 1, 1000))
            	stars.update_ = true;
            if(SliderInt("Max Star Count", &stars.maxStarCount_, 2, 5000))
            	stars.update_ = true;
            if(SliderFloat("Max Star Alpha", &stars.maxStarAlpha_, 0.0f, 1.0f))
            	stars.update_ = true;
            End();
        }, text_, stars_, warp_);
    }

    void setup() override {


		branch(BranchType::ONCE, always_)
				->assign(RW(timeOffset_), F(seconds))
				->construct(RW(text_), F(hypot,
											F(&cv::Size::width, F(&cv::Rect::size, vp_)),
											F(&cv::Size::height, F(&cv::Rect::size, vp_))
										) / V(60.0))
		->endBranch();
    }

    void infer() override {
    	constexpr auto copyToMemFn = static_cast<void(*)(const cv::UMat&, cv::UMat&)>(&SharedVariables::copy);

		branch(CS(warp_.update_))
			->plain(&Warp::calculate, RWS(warp_), vp_)
		->endBranch();

    	branch(CS(stars_.update_))
			->nvg(&StarsRenderer::draw, RWS(stars_), vp_)
			->fb(copyToMemFn, RWS(stars_.rendering_))
		->endBranch();

		nvg(&TextRenderer::draw, RWS(text_), vp_, CS(timeOffset_));
		fb(copyToMemFn, RWS(text_.rendering_));

		clear();

		fb<3>(&Warp::perform, RWS(warp_), RS(text_.rendering_), RS(stars_.rendering_));

		branch(-CS(text_.textOffsetY_) > CS(text_.height_))
			->assign(RWS(timeOffset_), F(seconds))
		->endBranch();
    }
};

TextRenderer FontDemoPlan::text_;
StarsRenderer FontDemoPlan::stars_;
Warp FontDemoPlan::warp_;
double FontDemoPlan::timeOffset_ = 0.0f;

int main() {
	cv::Rect viewport(0, 0, 1920, 1080);
	cv::Ptr<V4D> runtime = V4D::init(viewport, viewport.size(),  "Font Demo", AllocateFlags::NANOVG | AllocateFlags::IMGUI);
	Plan::run<FontDemoPlan>(0);
    return 0;
}
