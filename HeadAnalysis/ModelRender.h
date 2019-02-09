#pragma once

#include <OpenCVLinker.h>
#include <OpenGLLinker.h>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace glm;
using namespace std::experimental::filesystem;

#define DEPTH_INFINITY 1.0f

class ModelRender
{
	struct Camera
	{
		int Width;
		int Height;
		float FocalLength;
		mat4 ToClipCoordinate;

		Camera() : Width( 0 ), Height( 0 ), FocalLength( 0.0f ) {}
	};

	struct Object
	{
		GLenum DrawMode;
		GLuint VAO, VBO;
		GLsizei VerticesCount;
		GLuint TextureID;
		vector<GLfloat> DataBuffer;

		Object() : DrawMode( 0 ), VAO( 0 ), VBO( 0 ), VerticesCount( 0 ), TextureID( 0 ) {}
	};

	struct Shader
	{
		GLuint Program;
		GLint MVPLocation;
		GLint TextureLocation;

		Shader() : Program( 0 ), MVPLocation( 0 ), TextureLocation( 0 ) {}
	};

	GLFWwindow* Window;

	Camera MainCamera;
	Object FaceObject;
	Shader FaceShader;
	GLuint FrameBufferObject;
	GLuint ColorBufferObject;
	GLuint DepthBufferObject;

	bool readObjectFile(vector<vec3>& vertices, vector<vec2>& textures, const path& file_path) const;
	void prepareTexture2DFromFile(const string& file_name) const;
	void prepareTexture(const int& n_bytes_per_vertex, const string& texture_file_name);
	void prepareVertexBuffer(const int& n_bytes_per_vertex);
	void setFaceObject();
	void setFaceShader();
	void setCamera(
		const int& width, 
		const int& height, 
		const float& focal_length,
		const Vec3f& euler_angle_in_degree
	);
	void setRenderBuffer();
	void drawFaceModel();
	void captureFaceImage(Mat& face, Mat& depth) const;


public:
	ModelRender();
	~ModelRender();

	// rotation order is x(pitch) -> y(yaw) -> z(roll)
	void getModelImage(
		Mat& face,
		Mat& depth,
		const int& width, 
		const int& height, 
		const float& focal_length,
		const Vec3f& euler_angle_in_degree
	);

	void getClipToCameraMatrix(Matx44f& pixel_to_clip) const;
};