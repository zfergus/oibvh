#version 430 core

out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
// uniform vec3 lightColor;
// uniform vec3 objectColor;
uniform bool wireframe;
void main()
{
    vec3 norm = normalize(Normal);
    vec3 hardCodedLightPos = vec3(3.0, 3.0, 3.0);
    vec3 lightDir = normalize(hardCodedLightPos - FragPos);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);  // white
    vec3 objectColor = vec3(0.8, 0.8, 0.8); // grey
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    if (!wireframe)
    {
        FragColor = vec4(result, 1.0);
    }
    else
    {
        FragColor = vec4(vec3(0), 1.0);
    }
}