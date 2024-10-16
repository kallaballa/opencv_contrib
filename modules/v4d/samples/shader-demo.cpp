// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

using namespace cv::v4d;

//easing function for the bungee zoom
static float easeInOutQuint(float x) {
	return x < 0.5f ? 16.0f * x * x * x * x * x : 1.0f - std::pow(-2.0f * x + 2.0f, 5.0f) / 2.0f;
}

struct Camera2D {
	double startTime_ = seconds();
	double autoZoomSeconds_;

	//center x coordinate
	float centerX_ = -0.466f;
    //center y coordinate
    float centerY_ = 0.57052f;
    float currentZoom_ = 0.0;
    bool zoomIn_ = true;


    Camera2D(): Camera2D(15.0) {
    }

    Camera2D(double autoZoomSeconds) : autoZoomSeconds_(autoZoomSeconds) {
    }

    void updateAutoZoom(const int& maxIterations) {
    	double diff = seconds() - startTime_;
    	double progress = std::min(diff / autoZoomSeconds_, 1.0);

    	if(!zoomIn_)
			progress = 1.0 - progress;

		currentZoom_ = maxIterations * easeInOutQuint(progress);
		if (zoomIn_ && diff >= autoZoomSeconds_) {
			zoomIn_ = false;
			startTime_ = seconds();
		} else if (!zoomIn_ && diff >= autoZoomSeconds_) {
			zoomIn_ = true;
			startTime_ = seconds();
		}
    }
};

class MandelbrotScene {
    // vertex position, color
    constexpr static float vertices_[12] = {
        //    x      y      z
        -1.0f, -1.0f, -0.0f, 1.0f, 1.0f, -0.0f, -1.0f, 1.0f, -0.0f, 1.0f, -1.0f, -0.0f };

    constexpr static unsigned int indices_[6] = {
        //  2---,1
        //  | .' |
        //  0'---3
        0, 1, 2, 0, 3, 1 };

	struct Handles {
        /* GL uniform handles */
        GLint baseColorHdl_;
        GLint contrastBoostHdl_;
        GLint maxIterationsHdl_;
        GLint centerXHdl_;
        GLint centerYHdl_;
        GLint currentZoomHdl_;
        GLint viewportHdl_;

        /* Shader program handle */
        GLuint shaderHdl_;

        /* Object handles */
        GLuint vao_;
        GLuint vbo_, ebo_;
    } handles;

    //Load objects and buffers
    void loadBuffers() {
        glGenVertexArrays(1, &handles.vao_);
        glBindVertexArray(handles.vao_);

        glGenBuffers(1, &handles.vbo_);
        glGenBuffers(1, &handles.ebo_);

        glBindBuffer(GL_ARRAY_BUFFER, handles.vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_), vertices_, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handles.ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_), indices_, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    //mandelbrot shader code adapted from my own project: https://github.com/kallaballa/FractalDive#after
    GLuint loadShaders() {
        const string vert = R"(
        in vec4 position;

        void main()
        {
            gl_Position = vec4(position.xyz, 1.0);
        })";

        const string frag = R"(
        precision highp float;

        out vec4 outColor;

        uniform vec4 base_color;
        uniform int contrast_boost;
        uniform int max_iterations;
        uniform float current_zoom;
        uniform float center_y;
        uniform float center_x;

        uniform vec4 viewport;

        int get_iterations()
        {
            float pointr = (((((gl_FragCoord.x - viewport[0]) / (viewport[2])) - 0.5f) * 2.0)) * current_zoom + center_x;
            float pointi = (((((gl_FragCoord.y - viewport[1])  / (viewport[3])) - 0.5f) * 2.0)) * current_zoom + center_y;
            const float four = 4.0f;

            int iterations = 0;
            float zi = 0.0f;
            float zr = 0.0f;
            float zrsqr = 0.0f;
            float zisqr = 0.0f;

            while (iterations < max_iterations && zrsqr + zisqr < four) {
            //equals following line as a consequence of binomial expansion: zi = (((zr + zi)*(zr + zi)) - zrsqr) - zisqr
                zi = (zr + zr) * zi;

                zi += pointi;
                zr = (zrsqr - zisqr) + pointr;

                zrsqr = zr * zr;
                zisqr = zi * zi;
                ++iterations;
            }
            return iterations;
        }

        void mandelbrot()
        {
            int iter = get_iterations();
            if (iter < max_iterations) {
                float iterations = float(iter) / float(max_iterations);
                float cb = float(contrast_boost);
                float logBase;
                if(iter % 2 == 0)
					logBase = 25.0f;
				else
					logBase = 50.0f;
                
				float logDiv = log2(logBase);
				float colorBoost = iterations * cb;
				outColor = vec4(log2((logBase - 1.0f) * base_color[0] * colorBoost + 1.0f)/logDiv, 
								log2((logBase - 1.0f) * base_color[1] * colorBoost + 1.0f)/logDiv, 
								log2((logBase - 1.0f) * base_color[2] * colorBoost + 1.0f)/logDiv, 
								base_color[3]);
            } else {
                outColor = vec4(0,0,0,0);
            }
        }

        void main()
        {
            mandelbrot();
        })";
        unsigned int handles[3];
        cv::v4d::init_shaders(handles, vert.c_str(), frag.c_str(), "fragColor");
        return handles[0];
    }

public:
    /* Mandelbrot fractal rendering settings */
    struct Settings {
        // Red, green, blue and alpha. All from 0.0f to 1.0f
        cv::Scalar_<float> baseColor_{0.2f, 0.6f, 1.0f, 0.8f};
        //contrast boost
        int contrastBoost_ = 30; //0.0-255
        //max fractal iterations
        int maxIterations_ = 15000;

        bool autoZoom_ = true;
    };

    //Initialize shaders, objects, buffers and uniforms
    void init() {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        loadBuffers();
        handles.shaderHdl_ = loadShaders();
        handles.baseColorHdl_ = glGetUniformLocation(handles.shaderHdl_, "base_color");
        handles.contrastBoostHdl_ = glGetUniformLocation(handles.shaderHdl_, "contrast_boost");
        handles.maxIterationsHdl_ = glGetUniformLocation(handles.shaderHdl_, "max_iterations");
        handles.currentZoomHdl_ = glGetUniformLocation(handles.shaderHdl_, "current_zoom");
        handles.centerXHdl_ = glGetUniformLocation(handles.shaderHdl_, "center_x");
        handles.centerYHdl_ = glGetUniformLocation(handles.shaderHdl_, "center_y");
        handles.viewportHdl_ = glGetUniformLocation(handles.shaderHdl_, "viewport");
    }

    //Free OpenGL resources
    void destroy() const {
        glDeleteShader(handles.shaderHdl_);
        glDeleteBuffers(1, &handles.vbo_);
        glDeleteBuffers(1, &handles.ebo_);
        glDeleteVertexArrays(1, &handles.vao_);
    }

    //Render the mandelbrot fractal on top of a video
    void render(const cv::Rect& vp, const Settings& settings, const Camera2D& camera) const {
    	glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		glUseProgram(handles.shaderHdl_);
		glUniform4f(handles.baseColorHdl_, settings.baseColor_.val[0], settings.baseColor_.val[1], settings.baseColor_.val[2], settings.baseColor_.val[3]);
		glUniform1i(handles.contrastBoostHdl_, settings.contrastBoost_);
		glUniform1i(handles.maxIterationsHdl_, settings.maxIterations_);
		glUniform1f(handles.centerXHdl_, (camera.centerX_));
		glUniform1f(handles.centerYHdl_, (camera.centerY_));
		glUniform1f(handles.currentZoomHdl_, 1.0f / camera.currentZoom_);

		float vpArr[4] = {vp.x, vp.y, vp.width, vp.height};
		glUniform4fv(handles.viewportHdl_, 1, vpArr);
        glBindVertexArray(handles.vao_);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
};

using namespace cv::v4d::event;

class ShaderDemoPlan : public Plan {
	using K = V4D::Keys;
	using M = Mouse::Type;

	static struct Params {
    	Camera2D camera_;
    	MandelbrotScene::Settings settings_;
    } params_;

    int autoZoomSeconds_;
    double scale_ = 1.0;
    MandelbrotScene scene_;

    Property<cv::Rect> vp_ = P<cv::Rect>(K::VIEWPORT);
    Property<cv::Size> fbSz_ = P<cv::Size>(K::FRAMEBUFFER_SIZE);
    Property<cv::Size> winSz_ = P<cv::Size>(K::WINDOW_SIZE);

    Event<Mouse> release_ = E<Mouse>(M::RELEASE);
    Event<Mouse> scroll_ = E<Mouse>(M::SCROLL);

    static bool process_events(const cv::Rect& vp, const cv::Size& fbSize, const cv::Size& winSz, const Mouse::List& scrollEvents, const Mouse::List& releaseEvents, const double& scale, Params& params) {
    	if(!scrollEvents.empty() || !releaseEvents.empty()) {
			double borderX = ((fabs((winSz.width / scale) - fbSize.width) / 2.0));
			double borderY = ((fabs((winSz.height / scale) - fbSize.height) / 2.0));

			for(auto re : releaseEvents) {
				cv::Point2d pos = re->position() / scale;
				pos.x -= borderX;
				pos.y -= borderY;
				if(vp.contains(pos)) {
					params.camera_.centerX_ += ((pos.x / fbSize.width) - 0.5) / (params.camera_.currentZoom_ / 2.0);
					params.camera_.centerY_ += ((pos.y / fbSize.height) - 0.5) / (params.camera_.currentZoom_ / 2.0);
					params.settings_.autoZoom_ = false;
				}
			}

			for(auto se : scrollEvents) {
				if(vp.contains(se->position() / scale)) {
					params.camera_.currentZoom_ += (params.camera_.currentZoom_ / params.settings_.maxIterations_) * (params.settings_.maxIterations_ / 3.0) * se->data().y;
					params.camera_.currentZoom_ = std::min(params.camera_.currentZoom_, float(params.settings_.maxIterations_));
					params.settings_.autoZoom_ = false;
				}
			}
		}
//		std::cerr << "RET: " << !params.settings_.autoZoom_ << std::endl;
		return params.settings_.autoZoom_;
    }
public:
    ShaderDemoPlan(int autoZoomSeconds) : autoZoomSeconds_(autoZoomSeconds) {
    }

    void gui() override {
        imgui([](Params& params) {
            using namespace ImGui;
            Begin("Fractal");
            Text("Navigation");
            if(SliderInt("Iterations", &params.settings_.maxIterations_, 3, 100000))
            	params.settings_.autoZoom_ = false;
            if(DragFloat("Zoom", &params.camera_.currentZoom_, 0.01f * params.camera_.currentZoom_, 0.02f, params.settings_.maxIterations_))
                params.settings_.autoZoom_ = false;
            if(DragFloat("X", &params.camera_.centerX_, 0.001f / params.camera_.currentZoom_, -1.0f, 1.0f, "%.12f"))
            	params.settings_.autoZoom_ = false;
            if(DragFloat("Y", &params.camera_.centerY_, 0.001f / params.camera_.currentZoom_, -1.0f, 1.0f, "%.12f"))
            	params.settings_.autoZoom_ = false;
            Checkbox("Auto Zoom", &params.settings_.autoZoom_);
			Text("Color");
            ColorPicker4("Color", params.settings_.baseColor_.val);
            SliderInt("Contrast boost", &params.settings_.contrastBoost_, 1, 255);
            End();
        }, params_);
    }

    void setup() override {
    	branch(BranchType::ONCE, always_)
    		->assign(RWS(params_.camera_), V(Camera2D(autoZoomSeconds_)))
    	->endBranch();
    	gl(&MandelbrotScene::init, RW(scene_));
    }

    void infer() override {
    	assign(RW(scale_), F(aspect_preserving_scale, winSz_, fbSz_));

    	capture();

    	branch(process_events, vp_, fbSz_, winSz_, scroll_, release_, R(scale_), RWS(params_))
        	->plain(&Camera2D::updateAutoZoom, RWS(params_.camera_), R(params_.settings_.maxIterations_))
		->endBranch();

        gl(&MandelbrotScene::render, R(scene_), vp_, CS(params_.settings_), CS(params_.camera_));
    }

    void teardown() override {
    	gl(&MandelbrotScene::destroy, R(scene_));
    }
};

ShaderDemoPlan::Params ShaderDemoPlan::params_;

int main(int argc, char** argv) {
    if (argc != 2) {
		std::cerr << "Usage: shader-demo <video-file>" << std::endl;
        exit(1);
    }

    cv::Rect viewport(0, 0, 1280, 720);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Mandelbrot Shader Demo", AllocateFlags::IMGUI, ConfigFlags::DEFAULT);//, DebugFlags::PRINT_CONTROL_FLOW);
	auto src = Source::make(runtime, argv[1]);
	runtime->setSource(src);

	//0 extra workers, 15 seconds auto zoom
	Plan::run<ShaderDemoPlan>(0, 30);

	return 0;
}
