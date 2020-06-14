#include "ModelRender.h"

ModelRender::ModelRender() : Window( nullptr ), FrameBufferObject( 0 ), ColorTexture( 0 ), DepthTexture( 0 )
{
   if (!glfwInit()) {
      std::cout << "Could not initialize OpenGL...\n";
      return;
   }
   glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
   glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 6 );
   glfwWindowHint( GLFW_DOUBLEBUFFER, GLFW_TRUE );
   glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

   Window = glfwCreateWindow( 500, 500, "3D Face Model", nullptr, nullptr );
   glfwHideWindow( Window );
   glfwMakeContextCurrent( Window );

   if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      std::cout << "Failed to initialize GLAD\n";
      return;
   }
   
   glEnable( GL_DEPTH_TEST );
   glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );

   setFaceObject();
   setFaceShader();
}

ModelRender::~ModelRender()
{
   glDeleteProgram( FaceShader.Program );
   glDeleteVertexArrays( 1, &FaceObject.VAO );
   glDeleteBuffers( 1, &FaceObject.VBO );
   glDeleteFramebuffers( 1, &FrameBufferObject );
   glDeleteTextures( 1, &ColorTexture );
   glDeleteTextures( 1, &DepthTexture );

   glfwTerminate();
}

bool ModelRender::readObjectFile(
   std::vector<glm::vec3>& vertices, 
   std::vector<glm::vec2>& textures, 
   const std::filesystem::path& file_path
) const
{
   std::ifstream file(file_path);
   if (!file.is_open()) {
      std::cout << "The object file is not correct.\n";
      return false;
   }

   std::vector<glm::vec3> vertex_buffer;
   std::vector<glm::vec2> texture_buffer;
   std::vector<int> vertex_indices, texture_indices;
   while (!file.eof()) {
      std::string word;
      file >> word;
      
      if (word == "v") {
         glm::vec3 vertex;
         file >> vertex.x >> vertex.y >> vertex.z;
         vertex_buffer.emplace_back( vertex );
      }
      else if (word == "vt") {
         glm::vec2 uv;
         file >> uv.x >> uv.y;
         uv.y = -uv.y;
         texture_buffer.emplace_back( uv );
      }
      else if (word == "f") {
         char c;
         vertex_indices.emplace_back(); file >> vertex_indices.back(); file >> c;
         texture_indices.emplace_back(); file >> texture_indices.back(); file >> c;
         vertex_indices.emplace_back(); file >> vertex_indices.back(); file >> c;
         texture_indices.emplace_back(); file >> texture_indices.back(); file >> c;
         vertex_indices.emplace_back(); file >> vertex_indices.back(); file >> c;
         texture_indices.emplace_back(); file >> texture_indices.back(); file >> c;
      }
      else getline( file, word );
   }

   for (size_t i = 0; i < vertex_indices.size(); ++i) {
      vertices.emplace_back( vertex_buffer[vertex_indices[i] - 1] );
      textures.emplace_back( texture_buffer[texture_indices[i] - 1] );
   }
   return true;
}

void ModelRender::prepareVertexBuffer(int n_bytes_per_vertex)
{
   glCreateBuffers( 1, &FaceObject.VBO );
   glNamedBufferStorage( 
      FaceObject.VBO, 
      sizeof( GLfloat ) * FaceObject.DataBuffer.size(), 
      FaceObject.DataBuffer.data(), 
      GL_DYNAMIC_STORAGE_BIT 
   );

   glCreateVertexArrays( 1, &FaceObject.VAO );
   glVertexArrayVertexBuffer( FaceObject.VAO, 0, FaceObject.VBO, 0, n_bytes_per_vertex );
   glVertexArrayAttribFormat( FaceObject.VAO, 0, 3, GL_FLOAT, GL_FALSE, 0 );
   glEnableVertexArrayAttrib( FaceObject.VAO, 0 );
   glVertexArrayAttribBinding( FaceObject.VAO, 0, 0 );
}

void ModelRender::prepareTexture(const std::string& texture_file_name)
{
   GLuint texture_id = 0;
   glCreateTextures( GL_TEXTURE_2D, 1, &texture_id );

   const cv::Mat texture = cv::imread( texture_file_name );
   const int width = texture.cols;
   const int height = texture.rows;
   glTextureStorage2D( texture_id, 1, GL_RGBA8, width, height );
   glTextureSubImage2D( texture_id, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, texture.data );

   glTextureParameteri( texture_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_S, GL_REPEAT );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_T, GL_REPEAT );
   glGenerateTextureMipmap( texture_id );

   FaceObject.TextureID = texture_id;

   glVertexArrayAttribFormat( FaceObject.VAO, 2, 2, GL_FLOAT, GL_FALSE, 3 * sizeof( GLfloat ) );
   glEnableVertexArrayAttrib( FaceObject.VAO, 2 );
   glVertexArrayAttribBinding( FaceObject.VAO, 2, 0 );
}

void ModelRender::setFaceObject()
{
   std::vector<glm::vec3> face_vertices;
   std::vector<glm::vec2> face_textures;
   const std::string model_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples/model";
   readObjectFile( face_vertices, face_textures, model_directory_path + "/female_average.obj" );
   
   FaceObject.DrawMode = GL_TRIANGLES;
   for (uint i = 0; i < face_vertices.size(); ++i) {
      FaceObject.DataBuffer.push_back( face_vertices[i].x );
      FaceObject.DataBuffer.push_back( face_vertices[i].y );
      FaceObject.DataBuffer.push_back( face_vertices[i].z );
      FaceObject.DataBuffer.push_back( face_textures[i].s );
      FaceObject.DataBuffer.push_back( face_textures[i].t );
      FaceObject.VerticesCount++;
   }
   const int n_bytes_per_vertex = 5 * sizeof(GLfloat);
   prepareVertexBuffer( n_bytes_per_vertex );
   prepareTexture( model_directory_path + "/female_average.jpg" );
}

void ModelRender::setFaceShader()
{
   const GLchar* const vertex_source = {
      "#version 460                                                         \n"
      "uniform mat4 ModelViewProjectionMatrix;                              \n"
      "layout (location = 0) in vec3 v_position;                            \n"
      "layout (location = 2) in vec2 v_tex_coord;                           \n"
      "out vec2 tex_coord;                                                  \n"
      "void main(void) {                                                    \n"
      "  tex_coord = v_tex_coord;                                           \n"
      "	gl_Position =  ModelViewProjectionMatrix * vec4(v_position, 1.0f); \n"
      "}                                                                    \n"
   };
   const GLchar* const fragment_source = {
      "#version 460                                             \n"
      "layout (binding = 0) uniform sampler2D BaseTexture;      \n"
      "in vec2 tex_coord;                                       \n"
      "layout (location = 0) out vec4 final_color;              \n"
      "void main(void) {                                        \n"
      "	final_color = texture( BaseTexture, tex_coord );       \n"
      "}                                                        \n"
   };

   const GLuint vertex_shader = glCreateShader( GL_VERTEX_SHADER );
   const GLuint fragment_shader = glCreateShader( GL_FRAGMENT_SHADER );
   
   glShaderSource( vertex_shader, 1, &vertex_source, nullptr );
   glShaderSource( fragment_shader, 1, &fragment_source, nullptr );
   glCompileShader( vertex_shader );
   glCompileShader( fragment_shader );

   FaceShader.Program = glCreateProgram();
   glAttachShader( FaceShader.Program, vertex_shader );
   glAttachShader( FaceShader.Program, fragment_shader );
   glLinkProgram( FaceShader.Program );

   FaceShader.MVPLocation = glGetUniformLocation( FaceShader.Program, "ModelViewProjectionMatrix" );
   FaceShader.TextureLocation = glGetUniformLocation( FaceShader.Program, "BaseTexture" );

   glDeleteShader( vertex_shader );
   glDeleteShader( fragment_shader );
}

void ModelRender::setCamera(
   int width, 
   int height, 
   float focal_length,
   const cv::Vec3f& euler_angle_in_degree
)
{
   MainCamera.Width = width;
   MainCamera.Height = height;
   MainCamera.FocalLength = focal_length;

   const float pitch_angle = glm::radians( euler_angle_in_degree(0) );
   const float yaw_angle = glm::radians( euler_angle_in_degree(1) );
   const float roll_angle = glm::radians( euler_angle_in_degree(2) );
   const glm::mat4 pitching = rotate( glm::mat4(1.0f), pitch_angle, glm::vec3(1.0f, 0.0f, 0.0f) );
   const glm::mat4 yawing = rotate( glm::mat4(1.0f), yaw_angle, glm::vec3(0.0f, 1.0f, 0.0f) );
   const glm::mat4 rolling = rotate( glm::mat4(1.0f), roll_angle, glm::vec3(0.0f, 0.0f, 1.0f) );
   const glm::mat4 to_front = inverse( pitching ) * inverse( yawing ) * inverse( rolling );

   const glm::vec3 viewing_point(0.0f, 10.0f, 0.0f);
   const glm::vec3 up_vector(0.0f, 1.0f, 0.0f);
   const glm::vec3 initial_position(0.0f, 20.0f, 250.0f);
   const glm::mat4 view_matrix = lookAt( initial_position, viewing_point, up_vector ) * to_front;

   const auto fovy = 2.0f * atan( static_cast<float>(height) / (2.0f * focal_length) );
   const auto aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
   const glm::mat4 projection_matrix = glm::perspective( fovy, aspect_ratio, 150.0f, 500.0f );

   MainCamera.ToClipCoordinate = projection_matrix * view_matrix;
}

void ModelRender::setRenderBuffer()
{
   if (ColorTexture != 0) {
      glDeleteTextures( 1, &ColorTexture );
      ColorTexture = 0;
   }
   if (DepthTexture != 0) {
      glDeleteTextures( 1, &DepthTexture );
      DepthTexture = 0;
   }
   if (FrameBufferObject != 0) {
      glDeleteFramebuffers( 1, &FrameBufferObject );  
   }

   glCreateTextures( GL_TEXTURE_2D, 1, &ColorTexture );
   glTextureStorage2D( ColorTexture, 1, GL_RGB8, MainCamera.Width, MainCamera.Height );
   glTextureParameteri( ColorTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( ColorTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glGenerateTextureMipmap( ColorTexture );

   glCreateTextures( GL_TEXTURE_2D, 1, &DepthTexture );
   glTextureStorage2D( DepthTexture, 1, GL_DEPTH_COMPONENT32F, MainCamera.Width, MainCamera.Height );
   glTextureParameteri( DepthTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( DepthTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glGenerateTextureMipmap( DepthTexture );

   glCreateFramebuffers( 1, &FrameBufferObject );
   glNamedFramebufferTexture( FrameBufferObject, GL_COLOR_ATTACHMENT0, ColorTexture, 0 );
   glNamedFramebufferTexture( FrameBufferObject, GL_DEPTH_ATTACHMENT, DepthTexture, 0 );
}

void ModelRender::drawFaceModel()
{
   glViewport( 0, 0, MainCamera.Width, MainCamera.Height );
   glUseProgram( FaceShader.Program );
   glUniformMatrix4fv( FaceShader.MVPLocation, 1, GL_FALSE, &MainCamera.ToClipCoordinate[0][0] );
   glBindTextureUnit( 0, FaceObject.TextureID );

   glBindVertexArray( FaceObject.VAO );
   glDrawArrays( FaceObject.DrawMode, 0, FaceObject.VerticesCount );
   glBindVertexArray( 0 );
}

void ModelRender::captureFaceImage(cv::Mat& face, cv::Mat& depth) const
{
   face.create( MainCamera.Height, MainCamera.Width, CV_8UC3 );
   glPixelStorei( GL_PACK_ALIGNMENT, face.step & 3 ? 1 : 4 );
   glReadBuffer( GL_COLOR_ATTACHMENT0 );
   glReadPixels( 0, 0, face.cols, face.rows, GL_BGR, GL_UNSIGNED_BYTE, face.data );
   flip( face, face, 0 );
   
   depth.create( MainCamera.Height, MainCamera.Width, CV_32FC1 );
   glPixelStorei( GL_PACK_ALIGNMENT, depth.step & 3 ? 1 : 4 );
   glReadBuffer( GL_DEPTH_ATTACHMENT );
   glReadPixels( 0, 0, depth.cols, depth.rows, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data );
   flip( depth, depth, 0 );
}

void ModelRender::getModelImage(
   cv::Mat& face,
   cv::Mat& depth,
   int width, 
   int height, 
   float focal_length,
   const cv::Vec3f& euler_angle_in_degree
)
{
   setCamera( width, height, focal_length, euler_angle_in_degree );
   setRenderBuffer();

   glBindFramebuffer( GL_FRAMEBUFFER, FrameBufferObject );
   glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
   
   drawFaceModel();

   captureFaceImage( face, depth );
}

void ModelRender::getClipToCameraMatrix(cv::Matx44f& pixel_to_clip) const
{
   const glm::mat4 m = inverse( MainCamera.ToClipCoordinate );
   pixel_to_clip(0, 0) = m[0][0]; pixel_to_clip(0, 1) = m[1][0]; pixel_to_clip(0, 2) = m[2][0]; pixel_to_clip(0, 3) = m[3][0];
   pixel_to_clip(1, 0) = m[0][1]; pixel_to_clip(1, 1) = m[1][1]; pixel_to_clip(1, 2) = m[2][1]; pixel_to_clip(1, 3) = m[3][1];
   pixel_to_clip(2, 0) = m[0][2]; pixel_to_clip(2, 1) = m[1][2]; pixel_to_clip(2, 2) = m[2][2]; pixel_to_clip(2, 3) = m[3][2];
   pixel_to_clip(3, 0) = m[0][3]; pixel_to_clip(3, 1) = m[1][3]; pixel_to_clip(3, 2) = m[2][3]; pixel_to_clip(3, 3) = m[3][3];
}