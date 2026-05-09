#version 410 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uTex;

vec3 colormap(float x) {
    if (x > 0.0) return mix(vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0), x);
    else return mix(vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 1.0), -x);
}

void main() {
    float val = texture(uTex, TexCoord).r;

    val = val * 80.0;
    val = clamp(val, -1.0, 1.0);
    val = sign(val) * pow(abs(val), 0.5);

    vec3 color = colormap(val);

    FragColor = vec4(color, 1.0);
}