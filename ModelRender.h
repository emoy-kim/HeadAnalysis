/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is a free software; it can be freely used, changed and redistributed.
 * If you use any version of the code, please reference the code.
 * 
 */

#pragma once

#include <glad/glad.h>
#include <glfw3.h>
#include <glm.hpp>
#include <common.hpp>
#include <gtc/type_ptr.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <gtx/quaternion.hpp>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>

#include "ProjectPath.h"

constexpr float DEPTH_INFINITY = 1.0f;

class ModelRender
{
public:
   ModelRender(const ModelRender&) = delete;
   ModelRender(const ModelRender&&) = delete;
   ModelRender& operator=(const ModelRender&) = delete;
   ModelRender& operator=(const ModelRender&&) = delete;

   ModelRender();
   ~ModelRender();

   // rotation order is x(pitch) -> y(yaw) -> z(roll)
   void getModelImage(
      cv::Mat& face,
      cv::Mat& depth,
      int width, 
      int height, 
      float focal_length,
      const cv::Vec3f& euler_angle_in_degree
   );

   void getClipToCameraMatrix(cv::Matx44f& pixel_to_clip) const;

private:
   struct Camera
   {
      int Width;
      int Height;
      float FocalLength;
      glm::mat4 ToClipCoordinate;

      Camera() : Width( 0 ), Height( 0 ), FocalLength( 0.0f ), ToClipCoordinate() {}
   };

   struct Object
   {
      GLenum DrawMode;
      GLuint VAO, VBO;
      GLsizei VerticesCount;
      GLuint TextureID;
      std::vector<GLfloat> DataBuffer;

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
   GLuint ColorTexture;
   GLuint DepthTexture;

   bool readObjectFile(
      std::vector<glm::vec3>& vertices, 
      std::vector<glm::vec2>& textures, 
      const std::filesystem::path& file_path
   ) const;
   void prepareTexture(const std::string& texture_file_name);
   void prepareVertexBuffer(int n_bytes_per_vertex);
   void setFaceObject();
   void setFaceShader();
   void setCamera(
      int width, 
      int height, 
      float focal_length,
      const cv::Vec3f& euler_angle_in_degree
   );
   void setRenderBuffer();
   void drawFaceModel();
   void captureFaceImage(cv::Mat& face, cv::Mat& depth) const;
};