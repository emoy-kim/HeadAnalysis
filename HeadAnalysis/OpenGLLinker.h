#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <GL/FreeImage.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#pragma comment(lib,"glfw3dll.lib")
#pragma comment(lib,"freeglut.lib")
#pragma comment(lib,"FreeImage.lib")

#ifdef _DEBUG
#pragma comment(lib,"glew32d.lib")
#else
#pragma comment(lib,"glew32.lib")

#endif