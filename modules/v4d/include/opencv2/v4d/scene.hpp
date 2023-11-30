// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef MODULES_V4D_SRC_SCENE_HPP_
#define MODULES_V4D_SRC_SCENE_HPP_

#include "v4d.hpp"
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <string>

namespace cv {
namespace v4d {
namespace import {

struct BoundingBox {
	cv::Vec3f min_;
	cv::Vec3f max_;
	cv::Vec3f center_;
	cv::Vec3f size_;
	cv::Vec3f span_;
};

class AssimpScene {
    Assimp::Importer importer_;
	const aiScene* scene_;
	BoundingBox bbox_;
	double autoScale_;
public:

AssimpScene(const std::string filename);
AssimpScene(std::vector<cv::Point3f>& vertices);
~AssimpScene();

BoundingBox boundingBox();
float autoScale();
const aiScene* scene() const;
cv::Mat_<float> verticesAsMat();
std::vector<cv::Point3f> verticesAsVector();
};

} // namespace assimp

namespace gl {

cv::Vec3f rotate3D(const cv::Vec3f& point, const cv::Vec3f& center, const cv::Vec2f& rotation);
cv::Matx44f perspective(float fov, float aspect, float zNear, float zFar);
cv::Matx44f lookAt(cv::Vec3f eye, cv::Vec3f center, cv::Vec3f up);
cv::Matx44f modelView(const cv::Vec3f& translation, const cv::Vec3f& rotationVec, const cv::Vec3f& scaleVec);

class Scene {
public:
	enum RenderMode {
		DEFAULT = 0,
		WIREFRAME = 1,
		POINTCLOUD = 2,
	};
private:
	const cv::Rect& viewport_;
	std::vector<float> gridVertices_;
	Mat volume3DData_;
	cv::v4d::import::AssimpScene* assimp_ = nullptr;
	RenderMode mode_ = DEFAULT;
	GLuint sceneFBO_ = 0;
	GLuint sceneTexture_ = 0;
	GLuint sharedDepthTexture_ = 0;
	GLuint volumeTexture_ = 0;
	GLuint shaderHandles_[3] = {0, 0, 0};
	GLuint hdrHandles_[3] = {0, 0, 0};
	GLuint volumeHandles_[3] = {0, 0, 0};
	cv::Vec3f lightPos_ = {1.2f, 1.0f, 2.0f};

    const string modelVertexSource_ = R"(
 	    #version 300 es
 	    layout(location = 0) in vec3 aPos;
		layout(location = 1) in vec3 aNormal;

 	    out vec3 fragPos;
		out vec3 Normal;
 	    uniform mat4 model;
 	    uniform mat4 view;
 	    uniform mat4 projection;
 	    void main() {

    		vec4 worldPos = model * vec4(aPos, 1.0);
    		fragPos = worldPos.xyz;
		    Normal = mat3(transpose(inverse(model))) * aNormal;
		    gl_Position = projection * view * worldPos;
			gl_PointSize = 3.0;  // Set the size_ of the points
 	    }
 	)";

 	const string modelFragmentSource_ = R"(
		#version 300 es
		
		#define RENDER_MODE_WIREFRAME 1
		#define RENDER_MODE_POINTCLOUD 2
		
		#define AMBIENT_COLOR vec3(0.75, 0.75, 0.75)
		#define DIFFUSE_COLOR vec3(0.6, 0.6, 0.6)
		#define SPECULAR_COLOR vec3(0.5, 0.5, 0.5)
		
		precision highp float;
		
		in vec3 fragPos;
		in vec3 Normal;
		out vec4 fragColor;
		
		uniform vec3 lightPos;
		uniform vec3 viewPos;
		uniform int renderMode;
		
		void main() {
			vec4 attuned;
			if (renderMode == RENDER_MODE_WIREFRAME) {
				attuned = vec4(1.0, 0.0, 0.0, 1.0);
			} else if (renderMode == RENDER_MODE_POINTCLOUD) {
				float distance = length(fragPos - viewPos);
				float attenuation = pow(1.0 / distance, 16.0);
				vec3 color = vec3(0.8, 0.8, 0.8);
				attuned = vec4(color, attenuation);
			} else {
				attuned = vec4(0.6, 0.6, 0.6, 1.0);
			}
		
			vec3 ambient = 0.7 * attuned.xyz * AMBIENT_COLOR;
			vec3 lightDir = normalize(lightPos - fragPos);

			float diff = max(dot(normalize(Normal), lightDir), 0.0);
			vec3 diffuse = diff * attuned.xyz * DIFFUSE_COLOR;
			float viewDist = length(viewPos - fragPos);
			vec3 viewDir = normalize(viewPos - fragPos);
			vec3 reflectDir = reflect(-lightDir, normalize(Normal));
			float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
			vec3 specular = spec * SPECULAR_COLOR;
		
			// Combine ambient, diffuse, and specular components
			vec3 finalColor = (ambient + diffuse + specular) * (1.0/viewDist);
		
		//    // Add checkers
		//    float checkers = mod(floor(fragPos.x) + floor(fragPos.y) + floor(fragPos.z), 2.0);
		//    finalColor *= vec3(checkers);
		
			fragColor = vec4(finalColor / 5.0, 1.0);
		}
	)";

 	const string hdrVertexSource = R"(
		#version 300 es
		precision highp float;
		
		layout (location = 0) in vec2 aPos;
		layout (location = 1) in vec2 aTexCoords;
		
		out vec2 TexCoords;
		
		void main()
		{
			gl_Position = vec4(aPos, 0.0, 1.0);
			TexCoords = aTexCoords;
		}
	)";



 	const string hdrFragmentSource_ = R"(
		#version 300 es
		precision highp float;
		
		in vec2 TexCoords;
		uniform sampler2D hdrBuffer;
		uniform sampler2D depthBuffer;		
		out vec4 fragColor;

		uniform vec3 cameraPosition;

		vec4 permute(vec4 x) {
			return mod(((x*34.0)+1.0)*x, 289.0);
		}
		
		float snoise(vec3 v) { 
			const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
			const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
		
			// First corner
			vec3 i  = floor(v + dot(v, C.yyy));
			vec3 x0 =   v - i + dot(i, C.xxx);
		
			// Other corners
			vec3 g = step(x0.yzx, x0.xyz);
			vec3 l = 1.0 - g;
			vec3 i1 = min(g.xyz, l.zxy);
			vec3 i2 = max(g.xyz, l.zxy);
		
			//  x0 = x0 - 0. + 0.0 * C
			vec3 x1 = x0 - i1 + 1.0 * C.xxx;
			vec3 x2 = x0 - i2 + 2.0 * C.xxx;
			vec3 x3 = x0 - 1. + 3.0 * C.xxx;
		
			// Permutations
			i = mod(i, 289.0); 
			vec4 p = permute( permute( permute( 
						vec4(i.z, i1.z, i2.z, 1.0 ))
					  + vec4(i.y, i1.y, i2.y, 1.0 )) 
					  + vec4(i.x, i1.x, i2.x, 1.0 ));
		
			// Gradients
			float n_ = 1.0/7.0; // N=7
			vec3  ns = n_ * D.wyz - D.xzx;
		
			vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
		
			vec4 x_ = floor(j * ns.z);
			vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
		
			vec4 x = x_ *ns.x + ns.yyyy;
			vec4 y = y_ *ns.x + ns.yyyy;
			vec4 h = 1.0 - abs(x) - abs(y);
		
			vec4 b0 = vec4( x.xy, y.xy );
			vec4 b1 = vec4( x.zw, y.zw );
		
			vec4 s0 = floor(b0)*2.0 + 1.0;
			vec4 s1 = floor(b1)*2.0 + 1.0;
			vec4 sh = -step(h, vec4(0.0));
		
			vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
			vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
		
			vec3 p0 = vec3(a0.xy,h.x);
			vec3 p1 = vec3(a0.zw,h.y);
			vec3 p2 = vec3(a1.xy,h.z);
			vec3 p3 = vec3(a1.zw,h.w);
		
			// Normalise gradients
			vec4 norm = inversesqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
			p0 *= norm.x;
			p1 *= norm.y;
			p2 *= norm.z;
			p3 *= norm.w;
		
			// Mix final noise value
			vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
			m = m * m;
			return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
											dot(p2,x2), dot(p3,x3) ) );
		}
		
		
		void main()
		{
			const float gamma = 2.2;
			vec3 hdrColor = texture(hdrBuffer, TexCoords).rgb;
			float depth = texture(depthBuffer, TexCoords).r; // Sample the depth value
		
			// Check if the fragment is at the maximum depth (i.e., it's background)
			if(depth < 1.0)
			{
				hdrColor *= depth; // Modulate the color by depth
			}
		
/*			// Add the nebula effect

			float nebulaIntensity = snoise(normalize(vec3(pow(cameraPosition.x, 2.0), pow(cameraPosition.x, 2.0), pow(cameraPosition.x, 2.0)) + vec3(TexCoords, 0)));
			vec3 nebulaColor = vec3(nebulaIntensity); // The nebula color is determined by the noise function
			hdrColor = mix(hdrColor, nebulaColor, 0.5); // Blend the nebula color with the HDR color
*/
			vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
			mapped = pow(mapped, vec3(1.0 / gamma));
			fragColor = vec4(mapped, 1.0);
		}
	)";

 	const string fogFragmentSource_ = R"(
		#version 300 es
		precision highp float;
		
		in vec2 TexCoords;
		uniform sampler2D colorBuffer;
		uniform sampler2D depthBuffer;		
		out vec4 fragColor;
		
		void main()
		{
			vec3 color = texture(colorBuffer, TexCoords).rgb;
			float depth = texture(depthBuffer, TexCoords).r; // Sample the depth value
		
			// Fog parameters
			float fogNear = 0.1;
			float fogFar = 1.0;
			vec3 fogColor = vec3(0.5); // Gray fog
		
			// Compute the fog factor
			float fogFactor = (fogFar - depth) / (fogFar - fogNear);
			fogFactor = clamp(fogFactor, 0.0, 1.0);
		
			// Mix the color with the fog color based on the fog factor
			vec3 finalColor = mix(fogColor, color, fogFactor);
		
			fragColor = vec4(finalColor, 1.0);
		}
		)";

 	const string volumeVertexSource_ = R"(
		#version 300 es
		precision highp float;
		layout(location = 0) in vec4 position;

		out vec3 TexCoord;
		void main() {
			gl_Position = vec4(position.xyz, 1.0);
			TexCoord = position.xyz * 0.5 + 0.5;
		}

	)";


 	const string volumeFragmentSource_ = R"(
		#version 300 es
		precision highp float;

		uniform sampler3D textureID;
		uniform float nearPlane;
		uniform float farPlane;
		uniform vec2 resolution;
		in vec3 TexCoord;

		out vec4 fragColor;
		void main() {
			// Calculate the depth value from the 3D texture.
			float depth = texture(textureID, TexCoord).z / nearPlane + 1.0;
			
			// Raycasting: calculate the distance along the ray and clamp it between near plane and far plane.
			float distance = (gl_FragCoord.x - resolution.x) * depth / gl_FragCoord.y;
			if(distance < -3.0 || distance > 3.0){
				fragColor = vec4(1.0, 1.0, 1.0, 1.0);
			} else {
				// Scale the depth value to be in range [0, 1] and convert it to a color using an RGB-to-YUV conversion.
				vec3 yuv = vec3(depth * 0.5 + 0.5, depth * 0.5 - 0.25, depth);
				
				// Output the Y component as the alpha channel and UV components as color channels.
				fragColor = vec4(yuv.y, yuv.z, distance / (farPlane - nearPlane), yuv.x);
			}
		}

 	)";

 	const string volumeFragmentSource2_ = R"(
/*
		#version 300 es
		precision highp float;
		
		uniform sampler3D noiseTex;
		uniform vec3 camPos;
		uniform mat3 camRot;
		uniform float time;

		in vec3 TexCoord; //UNUSED	
		out vec4 fragColor;
		vec3 bboxMin = vec3(-10.0, -10.0, -10.0);
		vec3 bboxMax = vec3(10.0, 10.0, 10.0);
		float getNoise(vec3 pos) {
			vec3 normalizedPos = (pos - bboxMin) / (bboxMax - bboxMax);
			return texture(noiseTex, normalizedPos).x;
		}
		
		float fbm(vec3 pos) {
			float total = 0.0;
			float persistence = 0.5;
			float t = time * 0.05;
			for(int i = 0; i < 6; i++) {
				total += getNoise(pos + vec3(t)) * pow(persistence, float(i));
				pos *= 2.0;
			}
			return total;
		}
		
		void main() {
			vec3 dir = normalize(camRot * (gl_FragCoord.xyz - camPos));
			vec3 pos = camPos;
			vec4 color = vec4(0.0);
			for(int i = 0; i < 64; i++) {
				float density = fbm(pos * 0.1);
				color += vec4(vec3(density), 1.0) * 0.02;
				pos += dir * 0.1;
			}
			fragColor = color;
		}
*/
#version 300 es
precision highp float;

uniform sampler3D noiseTex;
uniform vec3 camPos;
uniform mat3 camRot;
uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 invPersMatrix;

in vec3 TexCoord;    
out vec4 fragColor;

float getNoise(vec3 pos) {
//	vec3 normalizedPos = (pos - vec3(-10, -10, -10)) / vec3(20, 20, 20);
	return texture(noiseTex, pos * 0.5 + 0.5).x;
}
	float fbm(vec3 pos) {
		float total = 0.0;
		float persistence = 0.5;
		float t = time * 0.05;
		for(int i = 0; i < 6; i++) {
			total += getNoise(pos + vec3(t)) * pow(persistence, float(i));
			pos *= 2.0;
		}
		return total;
	}

float fbm2(vec3 pos) {
    float total = 0.0;
    float persistence = 0.5;
    for(int i = 0; i < 6; i++) {
        total += getNoise(pos) * pow(persistence, float(i));
        pos *= 2.0;
    }
    return total;
}


float fbm3(vec3 pos) {
    float total = 0.0;
    float amplitude = 1.0;
	float t = 0.0; //time * 0.001;
	
    for (int i = 0; i < 4; i++) {
        total += getNoise(pos) * amplitude;
        pos *= 2.0;
        amplitude *= 0.5;
    }
    return total;
}
	float fbm4(vec3 pos) {
		float total = 0.0;
		float persistence = 0.5;
		float t = 0.0; //time * 0.05;
		for(int i = 0; i < 6; i++) {
			total += getNoise(pos) * pow(persistence, float(i));
			pos *= 2.0;
		}
		return total;
	}

vec3 getColor(float density) {
    vec3 color1 = vec3(0.5, 0.0, 0.5); // purple
    vec3 color2 = vec3(1.0, 0.5, 0.0); // orange
    return mix(color1, color2, density);
}

void main() {
    vec4 ndc = vec4(
        (gl_FragCoord.x / 1280.0 - 0.5) * 2.0,
        (gl_FragCoord.y / 720.0 - 0.5) * 2.0,
        (gl_FragCoord.z - 0.5) * 2.0,
        1.0);
		vec4 clip = invPersMatrix * ndc;
    	vec3 vertex = (clip / clip.w).xyz;

      vec3 dir = normalize(vertex - camPos);
	  float dist = length(camPos);
	  vec3 pos = camPos;
	  vec4 color = vec4(0.0);
      for(int i = 0; i < 256; i++) {
        float density = pow(getNoise(pos), 8.0);
		float r = pow(getNoise(pos.yzx), 4.0);
		float g = pow(getNoise(pos.zxy), 8.0);
		float b = pow(getNoise(pos.xzy), 2.0);
/*		float y = 0.299 * density + 0.587 * density + 0.114 * density;
		float u = 0.492 * (density-y); 
		float v = 0.877 * (density-y);*/
        color += vec4(r, g, b, density * dist * 0.33) * 0.05;
        pos = ((camPos / 40.0) + ((dir / 40.0) * (float(i) + 1.0) * 0.1) / 2.0);
	  }

	  vec3 normPos = camPos / 10.0;
	  vec3 normDir = dir / 10.0;
	  float v = pow(getNoise((normPos + normDir) / 2.0) , 8.0);
//	  float v = getNoise(dir);
//      fragColor = vec4(v, v, v, 0.5);
//      fragColor = vec4(dir * 0.5 + 0.5, 1.0);
	  fragColor = vec4(color.xyz, 0.35 - min(1.0 / (1.0 + dist), 0.9999)); 
	  gl_FragDepth = 0.9 + (min(1.0 / ((color.x + color.y + color.z) / 2.0), 0.9999) / 10.0);
}


)";

 	void creatVolumeTexture(Mat& textureData);
 	void createSceneObjects();
public:
	Scene(const cv::Rect& viewport);
	virtual ~Scene();
	void reset();
	bool load(const std::vector<Point3f>& points);
	bool load(const std::string& filename);
	void render(const cv::Vec3f& cameraPosition, const cv::Matx33f& cameraRotation, const cv::Matx44f& projection, const cv::Matx44f& view, const cv::Matx44f& modelView);

	float autoScale() const {
		return assimp_->autoScale();
	}

	cv::Vec3f autoCenter() const {
		return assimp_->boundingBox().center_;
	}

	RenderMode getMode() const {
		return mode_;
	}

	void setMode(RenderMode mode) {
		mode_ = mode;
	}

	cv::Vec3f lightPosition() const {
		return lightPos_;
	}

	void setLightPosition(cv::Vec3f pos) {
		lightPos_ = pos;
	}

};

} /* namespace gl */
} /* namespace v4d */
} /* namespace cv */

#endif /* MODULES_V4D_SRC_SCENE_HPP_ */
