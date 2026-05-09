#include <Shader/shader.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

std::string Shader::loadFile(const char* path) {
    std::ifstream file(path);
    std::stringstream buffer;

    if(!file.is_open())
    {
        std::cerr << "Failed to open shader: " << path << std::endl;
        return "";
    }

    buffer << file.rdbuf();
    return buffer.str();
}

static GLuint compile(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);

    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if(!success) {
        char info[512];
        glGetShaderInfoLog(shader,512,nullptr,info);
        std::cerr << "Shader compile error:\n" << info << std::endl;
    }

    return shader;
}

Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    std::string vStr = loadFile(vertexPath);
    std::string fStr = loadFile(fragmentPath);

    const char* vSrc = vStr.c_str();
    const char* fSrc = fStr.c_str();

    GLuint vs = compile(GL_VERTEX_SHADER, vSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fSrc);

    ID = glCreateProgram();

    glAttachShader(ID, vs);
    glAttachShader(ID, fs);

    glLinkProgram(ID);

    int success;
    glGetProgramiv(ID, GL_LINK_STATUS, &success);

    if(!success) {
        char info[512];
        glGetProgramInfoLog(ID,512,nullptr,info);
        std::cerr << "Program link error:\n" << info << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
}

Shader::~Shader() {
    deleteProgram();
}

void Shader::deleteProgram() {
    if (ID != 0) {
        glDeleteProgram(ID);
        ID = 0;
    }
}

void Shader::use() {
    glUseProgram(ID);
}

void Shader::setInt(const std::string& name, int value) {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}