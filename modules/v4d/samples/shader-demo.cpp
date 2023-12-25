// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

using namespace cv::v4d;

class ShaderDemoPlan : public Plan {
private:
    // vertex position, color
    constexpr static float vertices[12] = {
        //    x      y      z
        -1.0f, -1.0f, -0.0f, 1.0f, 1.0f, -0.0f, -1.0f, 1.0f, -0.0f, 1.0f, -1.0f, -0.0f };

    constexpr static unsigned int indices[6] = {
        //  2---,1
        //  | .' |
        //  0'---3
        0, 1, 2, 0, 3, 1 };

    static struct Params {
        /* Mandelbrot control parameters */
        // Red, green, blue and alpha. All from 0.0f to 1.0f
        float baseColorVal_[4] = {0.2, 0.6, 1.0, 0.8};
        //contrast boost
        int contrastBoost_ = 255; //0.0-255
        //max fractal iterations
        int maxIterations_ = 10000;
        //center x coordinate
        float centerX_ = -0.466;
        //center y coordinate
        float centerY_ = 0.57052;
        float zoomFactor_ = 1.0;
        float currentZoom_ = 4.0;
        bool zoomIn = true;
        float zoomIncr_ = -currentZoom_ / 1000;
        bool manualNavigation_ = false;
    } params_;

    struct Handles {
        /* GL uniform handles */
        GLint baseColorHdl_;
        GLint contrastBoostHdl_;
        GLint maxIterationsHdl_;
        GLint centerXHdl_;
        GLint centerYHdl_;
        GLint currentZoomHdl_;
        GLint resolutionHdl_;

        /* Shader program handle */
        GLuint shaderHdl_;

        /* Object handles */
        GLuint vao_;
        GLuint vbo_, ebo_;
    } handles_;

    Property<cv::Rect> vp_ = GET<cv::Rect>(V4D::Keys::VIEWPORT);

    //easing function for the bungee zoom
    static float easeInOutQuint(float x) {
        return x < 0.5f ? 16.0f * x * x * x * x * x : 1.0f - std::pow(-2.0f * x + 2.0f, 5.0f) / 2.0f;
    }

    //Load objects and buffers
    static void load_buffers(Handles& handles) {
        GL_CHECK(glGenVertexArrays(1, &handles.vao_));
        GL_CHECK(glBindVertexArray(handles.vao_));

        GL_CHECK(glGenBuffers(1, &handles.vbo_));
        GL_CHECK(glGenBuffers(1, &handles.ebo_));

        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, handles.vbo_));
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));

        GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handles.ebo_));
        GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW));

        GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0));
        GL_CHECK(glEnableVertexAttribArray(0));

        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CHECK(glBindVertexArray(0));
    }

    //mandelbrot shader code adapted from my own project: https://github.com/kallaballa/FractalDive#after
    static GLuint load_shaders() {
        #if !defined(OPENCV_V4D_USE_ES3)
        const string shaderVersion = "330";
        #else
        const string shaderVersion = "300 es";
        #endif

        const string vert =
        "    #version " + shaderVersion
        + R"(
        in vec4 position;

        void main()
        {
            gl_Position = vec4(position.xyz, 1.0);
        })";

        const string frag =
                "    #version " + shaderVersion
                        + R"(
        precision highp float;

        out vec4 outColor;

        uniform vec4 base_color;
        uniform int contrast_boost;
        uniform int max_iterations;
        uniform float current_zoom;
        uniform float center_y;
        uniform float center_x;

        uniform vec2 resolution;

        int get_iterations()
        {
            float pointr = (((gl_FragCoord.x / resolution[0]) - 0.5f) * current_zoom + center_x);
            float pointi = (((gl_FragCoord.y / resolution[1]) - 0.5f) * current_zoom + center_y);
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

    static void update_params(Params& params) {
		//bungee zoom
		if (params.currentZoom_ >= 3) {
			params.zoomIn = true;
		} else if (params.currentZoom_ < 0.05) {
			params.zoomIn = false;
		}

		params.zoomIncr_ = (params.currentZoom_ / 100);
		if(params.zoomIn)
			params.zoomIncr_ = -std::fabs(params.zoomIncr_);

		if (!params.manualNavigation_) {
			params.currentZoom_ += params.zoomIncr_;
		} else {
			params.currentZoom_ = 1.0 / pow(params.zoomFactor_, 5.0f);
		}
    }

    //Initialize shaders, objects, buffers and uniforms
    static void init_scene(Handles& handles) {
        GL_CHECK(glEnable(GL_BLEND));
        GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        handles.shaderHdl_ = load_shaders();
        load_buffers(handles);

        handles.baseColorHdl_ = glGetUniformLocation(handles.shaderHdl_, "base_color");
        handles.contrastBoostHdl_ = glGetUniformLocation(handles.shaderHdl_, "contrast_boost");
        handles.maxIterationsHdl_ = glGetUniformLocation(handles.shaderHdl_, "max_iterations");
        handles.currentZoomHdl_ = glGetUniformLocation(handles.shaderHdl_, "current_zoom");
        handles.centerXHdl_ = glGetUniformLocation(handles.shaderHdl_, "center_x");
        handles.centerYHdl_ = glGetUniformLocation(handles.shaderHdl_, "center_y");
        handles.resolutionHdl_ = glGetUniformLocation(handles.shaderHdl_, "resolution");
    }

    //Free OpenGL resources
    static void destroy_scene(const Handles& handles) {
        glDeleteShader(handles.shaderHdl_);
        glDeleteBuffers(1, &handles.vbo_);
        glDeleteBuffers(1, &handles.ebo_);
        glDeleteVertexArrays(1, &handles.vao_);
    }

    //Render the mandelbrot fractal on top of a video
    static void render_scene(const cv::Rect& vp, const Params& params, const Handles& handles) {
    	glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		GL_CHECK(glUseProgram(handles.shaderHdl_));
		GL_CHECK(glUniform4f(handles.baseColorHdl_, params.baseColorVal_[0], params.baseColorVal_[1], params.baseColorVal_[2], params.baseColorVal_[3]));
		GL_CHECK(glUniform1i(handles.contrastBoostHdl_, params.contrastBoost_));
		GL_CHECK(glUniform1i(handles.maxIterationsHdl_, params.maxIterations_));
		GL_CHECK(glUniform1f(handles.centerYHdl_, params.centerY_));
		GL_CHECK(glUniform1f(handles.centerXHdl_, params.centerX_));
		if(params.manualNavigation_) {
			GL_CHECK(glUniform1f(handles.currentZoomHdl_, params.currentZoom_));
		}
		else {
			GL_CHECK(glUniform1f(handles.currentZoomHdl_, easeInOutQuint(params.currentZoom_)));
		}
		float res[2] = {float(vp.width), float(vp.height)};
		GL_CHECK(glUniform2fv(handles.resolutionHdl_, 1, res));
        GL_CHECK(glBindVertexArray(handles.vao_));
        GL_CHECK(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0));
    }
public:
	ShaderDemoPlan() {
		_shared(params_);
	}

	void gui() override {
        imgui([](Params& params) {
            using namespace ImGui;
            Begin("Fractal");
            Text("Navigation");
            SliderInt("Iterations", &params.maxIterations_, 3, 100000);
            DragFloat("X", &params.centerX_, 0.000001, -1.0f, 1.0f);
            DragFloat("Y", &params.centerY_, 0.000001, -1.0f, 1.0f);
            if(SliderFloat("Zoom", &params.zoomFactor_, 0.0001f, 10.0f))
                params.manualNavigation_ = true;
            Text("Color");
            ColorPicker4("Color", params.baseColorVal_);
            SliderInt("Contrast boost", &params.contrastBoost_, 1, 255);
            End();
        }, params_);
    }

    void setup() override {
    	gl([](Handles& handles) {
			init_scene(handles);
		}, RW(handles_));
    }

    void infer() override {
        capture();

        plain([](Params& params) {
			update_params(params);
		}, RW_S(params_));

		gl([](const cv::Rect& vp, const Params params, const Handles& handles) {
        	render_scene(vp, params, handles);
		}, vp_, R_C(params_), R(handles_));

        write();
    }

    void teardown() override {
		gl([](const Handles& handles) {
			destroy_scene(handles);
		}, R(handles_));
    }
};

ShaderDemoPlan::Params ShaderDemoPlan::params_;

int main(int argc, char** argv) {
    if (argc != 2) {
		std::cerr << "Usage: shader-demo <video-file>" << std::endl;
        exit(1);
    }

    cv::Rect viewport(0, 0, 1280, 720);
    cv::Ptr<V4D> runtime = V4D::init(viewport, "Mandelbrot Shader Demo", AllocateFlags::IMGUI);
	auto src = Source::make(runtime, argv[1]);
	auto sink = Sink::make(runtime, "shader-demo.mkv", src->fps(), viewport.size());
	runtime->setSource(src);
	runtime->setSink(sink);

	Plan::run<ShaderDemoPlan>(0);

	return 0;
}
