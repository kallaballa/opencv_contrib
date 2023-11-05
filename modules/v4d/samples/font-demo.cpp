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
	static struct Params {
		const cv::Scalar_<float> INITIAL_COLOR = cv::v4d::colorConvert(cv::Scalar(0.15 * 180.0, 128, 255, 255), cv::COLOR_HLS2RGB);
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
	} params_;

    //BGRA
	inline static cv::UMat stars_;
	cv::UMat warped_;
	//transformation matrix
    inline static cv::Mat tm_;

    //the text to display
	inline static vector<string> lines_;
    //global frame count
    inline static uint32_t global_cnt_ = 0;
    //Total number of lines in the text
    inline static int32_t numLines_ = 0;
    //Height of the text in pixels
    inline static int32_t textHeight_ = 0;
    //the sequence number of the current frame
    uint32_t seqNum_ = 0;
    //y-value of the current line
    int32_t y_ = 0;

    int32_t translateY_ = 0;

    cv::RNG rng_ = cv::getTickCount();
public:
	using Plan::Plan;

    void gui(cv::Ptr<V4D> window) override {
        window->imgui([](cv::Ptr<V4D> win, ImGuiContext* ctx, Params& params){
        	CV_UNUSED(win);
        	using namespace ImGui;
            SetCurrentContext(ctx);
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
        //The text to display
        string txt = cv::getBuildInformation();
        //Save the text to a vector
        std::istringstream iss(txt);

        for (std::string line; std::getline(iss, line); ) {
            lines_.push_back(line);
        }
        params_.fontSize_ = hypot(size().width, size().height) / 60.0;
        numLines_ = lines_.size();
        textHeight_ = (numLines_ * params_.fontSize_);
    }

    void infer(cv::Ptr<V4D> window) override {
		window->branch(0, isTrue_, params_.updateStars_);
		{
			window->nvg([](const cv::Size& sz, cv::RNG& rng, const Params& params) {
				using namespace cv::v4d::nvg;
				clear();

				//draw stars
				int numStars = rng.uniform(params.minStarCount_, params.maxStarCount_);
				for(int i = 0; i < numStars; ++i) {
					beginPath();
					const auto& size = rng.uniform(params.minStarSize_, params.maxStarSize_);
					strokeWidth(size);
					strokeColor(cv::Scalar(255, 255, 255, params.starAlpha_ * 255.0f));
					circle(rng.uniform(0, sz.width) , rng.uniform(0, sz.height), size / 2.0);
					stroke();
				}
			}, size(), rng_, params_);

			window->fb([](const cv::UMat& framebuffer, const cv::Rect& viewport, cv::UMat& f, Params& params){
				framebuffer(viewport).copyTo(f);
				params.updateStars_ = false;
			}, viewport(), stars_, params_);
		}
		window->endbranch(0, isTrue_, params_.updateStars_);

		window->branch(0, isTrue_, params_.updatePerspective_);
		{
			window->single([](const cv::Size& sz, cv::Mat& tm, Params& params){
				//Derive the transformation matrix tm for the pseudo 3D effect from quad1 and quad2.
				vector<cv::Point2f> quad1 = {cv::Point2f(0,0),cv::Point2f(sz.width,0),
						cv::Point2f(sz.width,sz.height),cv::Point2f(0,sz.height)};
				float l = (sz.width - (sz.width * params.warpRatio_)) / 2.0;
				float r = sz.width - l;

				vector<cv::Point2f> quad2 = {cv::Point2f(l, 0.0f),cv::Point2f(r, 0.0f),
						cv::Point2f(sz.width,sz.height), cv::Point2f(0,sz.height)};
				tm = cv::getPerspectiveTransform(quad1, quad2);
				params.updatePerspective_ = false;
			}, size(), tm_, params_);
		}
		window->endbranch(0, isTrue_, params_.updatePerspective_);

		window->branch(always_);
		{
			window->nvg([](const cv::Size& sz, int32_t& ty, const int32_t& seqNum, int32_t& y, const int32_t& textHeight, const std::vector<std::string> lines, const Params& params) {
				//How many pixels to translate the text up.
		    	ty = sz.height - seqNum;
				using namespace cv::v4d::nvg;
				clear();
				fontSize(params.fontSize_);
				fontFace("sans-bold");
				fillColor(params.textColor_ * 255);
				textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);

				/** only draw lines that are visible **/
				translate(0, ty);

				for (size_t i = 0; i < lines.size(); ++i) {
					y = (i * params.fontSize_);
					if (y + ty < textHeight && y + ty + params.fontSize_ > 0) {
						text(sz.width / 2.0, y, lines[i].c_str(), lines[i].c_str() + lines[i].size());
					}
				}
			}, size(), translateY_, seqNum_, y_, textHeight_, lines_, params_);

			window->fb([](cv::UMat& framebuffer, const cv::Rect& viewport, cv::UMat& w, cv::UMat& s, cv::Mat& t) {
				if(s.empty() || t.empty())
					return;
				cv::warpPerspective(framebuffer(viewport), w, t, viewport.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
				cv::add(s.clone(), w, framebuffer(viewport));
			}, viewport(), warped_, stars_, tm_);

			window->write();

			window->single([](const int32_t& translateY, const int32_t& textHeight, uint32_t& gcnt, uint32_t& seqNum) {
				if(-translateY > textHeight) {
					//reset the scroll once the text is out of the picture
					gcnt = 0;
				}
				++gcnt;
				//Wrap the cnt around if it becomes to big.
				if(gcnt > std::numeric_limits<uint32_t>().max() / 2.0)
					gcnt = 0;
				seqNum = gcnt;
			}, translateY_, textHeight_, global_cnt_, seqNum_);
		}
		window->endbranch(always_);
    }
};

FontDemoPlan::Params FontDemoPlan::params_;

int main() {
	cv::Ptr<FontDemoPlan> plan = new FontDemoPlan(cv::Size(1280, 720));
	cv::Ptr<V4D> window = V4D::make(plan->size(), "Font Demo", ALL);

	auto sink = makeWriterSink(window, "font-demo.mkv", 60, plan->size());
	window->setSink(sink);
	window->run(plan);
    return 0;
}
