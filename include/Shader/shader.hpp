#pragma once
#include <glad/glad.h>
#include <string>

class Shader {
public:
    GLuint ID;

    Shader(const char* vertexPath, const char* fragmentPath);
    ~Shader();

    void use();
    void setInt(const std::string& name, int value);

    void deleteProgram();

private:
    std::string loadFile(const char* path);
};