#include "ModelRender.h"

ModelRender::ModelRender() : FrameBufferObject( 0 ), ColorBufferObject( 0 ), DepthBufferObject( 0 )
{
	if (!glfwInit()) {
		cout << "Cannot Initialize OpenGL..." << endl;
		return;
	}
	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 2 );
	glfwWindowHint( GLFW_DOUBLEBUFFER, GLFW_TRUE );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

	Window = glfwCreateWindow( 500, 500, "3D Face Model", nullptr, nullptr );
	glfwHideWindow( Window );
	glfwMakeContextCurrent( Window );
	glewExperimental = true;
	glewInit();
	
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
	glDeleteRenderbuffers( 1, &ColorBufferObject );
	glDeleteRenderbuffers( 1, &DepthBufferObject );

	glfwTerminate();
}

bool ModelRender::readObjectFile(vector<vec3>& vertices, vector<vec2>& textures, const path& file_path) const
{
	ifstream file(file_path);
	if (!file.is_open()) {
		cout << "The object file is not correct." << endl;
		return false;
	}

	vector<vec3> vertex_buffer;
	vector<vec2> texture_buffer;
	vector<int> vertex_indices, texture_indices;
	while (!file.eof()) {
		string word;
		file >> word;
		
		if (word == "v") {
			vec3 vertex;
			file >> vertex.x >> vertex.y >> vertex.z;
			vertex_buffer.emplace_back( vertex );
		}
		else if (word == "vt") {
			vec2 uv;
			file >> uv.x >> uv.y;
			//uv.y = -uv.y;
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

	for (uint i = 0; i < vertex_indices.size(); ++i) {
		vertices.emplace_back( vertex_buffer[vertex_indices[i] - 1] );
		textures.emplace_back( texture_buffer[texture_indices[i] - 1] );
	}
	return true;
}

void ModelRender::prepareVertexBuffer(const int& n_bytes_per_vertex)
{
	glGenBuffers( 1, &FaceObject.VBO );
	glBindBuffer( GL_ARRAY_BUFFER, FaceObject.VBO );
	glBufferData( GL_ARRAY_BUFFER, sizeof(GLfloat) * FaceObject.DataBuffer.size(), FaceObject.DataBuffer.data(), GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	glGenVertexArrays( 1, &FaceObject.VAO );
	glBindVertexArray( FaceObject.VAO );
	glBindBuffer( GL_ARRAY_BUFFER, FaceObject.VBO );
	glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, n_bytes_per_vertex, reinterpret_cast<GLvoid *>(0) );
	glEnableVertexAttribArray( 0 );
}

void ModelRender::prepareTexture2DFromFile(const string& file_name) const
{
	const FREE_IMAGE_FORMAT format = FreeImage_GetFileType( file_name.c_str(), 0 );
	FIBITMAP* texture = FreeImage_Load( format, file_name.c_str() );
	const int n_bits_per_pixel = FreeImage_GetBPP( texture );
	
	FIBITMAP* texture_32bit;
	if (n_bits_per_pixel == 32) texture_32bit = texture;
	else texture_32bit = FreeImage_ConvertTo32Bits( texture );
	
	const int width = FreeImage_GetWidth( texture_32bit );
	const int height = FreeImage_GetHeight( texture_32bit );
	GLvoid* data = FreeImage_GetBits( texture_32bit );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, data );
	
	FreeImage_Unload( texture_32bit );
	if (n_bits_per_pixel != 32) {
		FreeImage_Unload( texture );
	}
}

void ModelRender::prepareTexture(const int& n_bytes_per_vertex, const string& texture_file_name)
{
	glEnable( GL_TEXTURE_2D );

	glGenTextures( 1, &FaceObject.TextureID );
	glActiveTexture( GL_TEXTURE0 + FaceObject.TextureID );
	glBindTexture( GL_TEXTURE_2D, FaceObject.TextureID );

	prepareTexture2DFromFile( texture_file_name );
	
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );

	glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, n_bytes_per_vertex, reinterpret_cast<GLvoid *>( 3 * sizeof(GLfloat) ) );
	glEnableVertexAttribArray( 2 );
}

void ModelRender::setFaceObject()
{
	vector<vec3> face_vertices;
	vector<vec2> face_textures;
	readObjectFile( face_vertices, face_textures, "Model/female_average.obj" );
	
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
	prepareTexture( n_bytes_per_vertex, "Model/female_average.jpg" );
}

void ModelRender::setFaceShader()
{
	const GLchar* const vertex_source = {
		"#version 330                                             \n"
		"uniform mat4 ModelViewProjectionMatrix;                  \n"
		"layout (location = 0) in vec4 v_position;                \n"
		"layout (location = 2) in vec2 v_tex_coord;               \n"
		"out vec2 tex_coord;                                      \n"
		"void main(void) {                                        \n"
		"  tex_coord = v_tex_coord;                               \n"
		"	gl_Position =  ModelViewProjectionMatrix * v_position; \n"
		"}                                                        \n"
	};
	const GLchar* const fragment_source = {
		"#version 330                                             \n"
		"uniform sampler2D BaseTexture;                           \n"
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
	const int& width, 
	const int& height, 
	const float& focal_length,
	const Vec3f& euler_angle_in_degree
)
{
	MainCamera.Width = width;
	MainCamera.Height = height;
	MainCamera.FocalLength = focal_length;

	const float pitch_angle = radians( euler_angle_in_degree(0) );
	const float yaw_angle = radians( euler_angle_in_degree(1) );
	const float roll_angle = radians( euler_angle_in_degree(2) );
	const mat4 pitching = glm::rotate( mat4(1.0f), pitch_angle, vec3(1.0f, 0.0f, 0.0f) );
	const mat4 yawing = glm::rotate( mat4(1.0f), yaw_angle, vec3(0.0f, 1.0f, 0.0f) );
	const mat4 rolling = glm::rotate( mat4(1.0f), roll_angle, vec3(0.0f, 0.0f, 1.0f) );
	const mat4 to_front = inverse( pitching ) * inverse( yawing ) * inverse( rolling );

	const vec3 viewing_point(0.0f, 10.0f, 0.0f);
	const vec3 up_vector(0.0f, 1.0f, 0.0f);
	const vec3 initial_position(0.0f, 20.0f, 250.0f);
	const mat4 view_matrix = lookAt( initial_position, viewing_point, up_vector ) * to_front;

	const auto fovy = 2.0f * atan( static_cast<float>(height) / (2.0f * focal_length) );
	const auto aspect_ratio = static_cast<float>(width) / height;
	const mat4 projection_matrix = perspective( fovy, aspect_ratio, 150.0f, 500.0f );

	MainCamera.ToClipCoordinate = projection_matrix * view_matrix;
}

void ModelRender::setRenderBuffer()
{
	if (FrameBufferObject != 0) {
		glDeleteFramebuffers( 1, &FrameBufferObject );
		glDeleteRenderbuffers( 1, &FrameBufferObject );
	}
	glGenFramebuffers( 1, &FrameBufferObject );
	glBindFramebuffer( GL_FRAMEBUFFER, FrameBufferObject );
	
	glGenRenderbuffers( 1, &ColorBufferObject );
	glBindRenderbuffer( GL_RENDERBUFFER, ColorBufferObject );
	glRenderbufferStorage( GL_RENDERBUFFER, GL_RGB, MainCamera.Width, MainCamera.Height );

	glGenRenderbuffers( 1, &DepthBufferObject );
	glBindRenderbuffer( GL_RENDERBUFFER, DepthBufferObject );
	glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, MainCamera.Width, MainCamera.Height );

	glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ColorBufferObject );
	glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, DepthBufferObject );
}

void ModelRender::drawFaceModel()
{
	glBindFramebuffer( GL_FRAMEBUFFER, FrameBufferObject );

	glUseProgram( FaceShader.Program );
	glUniformMatrix4fv( FaceShader.MVPLocation, 1, GL_FALSE, &MainCamera.ToClipCoordinate[0][0] );
	glUniform1i( FaceShader.TextureLocation, FaceObject.TextureID );

	glBindVertexArray( FaceObject.VAO );
	glDrawArrays( FaceObject.DrawMode, 0, FaceObject.VerticesCount );
	glBindVertexArray( 0 );
}

void ModelRender::captureFaceImage(Mat& face, Mat& depth) const
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
	Mat& face,
	Mat& depth,
	const int& width, 
	const int& height, 
	const float& focal_length,
	const Vec3f& euler_angle_in_degree
)
{
	setCamera( width, height, focal_length, euler_angle_in_degree );
	setRenderBuffer();

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	glViewport( 0, 0, width, height );
	drawFaceModel();

	captureFaceImage( face, depth );
}

void ModelRender::getClipToCameraMatrix(Matx44f& pixel_to_clip) const
{
	const mat4 m = inverse( MainCamera.ToClipCoordinate );
	pixel_to_clip(0, 0) = m[0][0]; pixel_to_clip(0, 1) = m[1][0]; pixel_to_clip(0, 2) = m[2][0]; pixel_to_clip(0, 3) = m[3][0];
	pixel_to_clip(1, 0) = m[0][1]; pixel_to_clip(1, 1) = m[1][1]; pixel_to_clip(1, 2) = m[2][1]; pixel_to_clip(1, 3) = m[3][1];
	pixel_to_clip(2, 0) = m[0][2]; pixel_to_clip(2, 1) = m[1][2]; pixel_to_clip(2, 2) = m[2][2]; pixel_to_clip(2, 3) = m[3][2];
	pixel_to_clip(3, 0) = m[0][3]; pixel_to_clip(3, 1) = m[1][3]; pixel_to_clip(3, 2) = m[2][3]; pixel_to_clip(3, 3) = m[3][3];
}