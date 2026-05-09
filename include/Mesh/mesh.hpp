#pragma once
#include <glad/glad.h>
#include <vector>

class Mesh {
public:
    GLuint VAO;
    GLuint VBO;

    Mesh(const std::vector<float>& vertices);
    ~Mesh();

    void draw();
    void cleanup();

private:
    size_t vertexCount;
};