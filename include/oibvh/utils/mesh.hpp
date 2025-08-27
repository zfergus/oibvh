/*********************************************************************
 * @file       mesh.h
 * @brief      Header file for Mesh class
 * @details
 * @author     hhhcbw
 * @date       2023-11-20
 *********************************************************************/
#pragma once

#include "utils.hpp"

#include <glm/glm.hpp>

#include <vector>

#define MAX_BONE_INFLUENCE 4

struct Vertex
{
    /**
     * @brief Position coordinate of vertex
     */
    glm::vec3 m_position;
    /**
     * @brief Normal vector of vertex
     */
    glm::vec3 m_normal;
    /**
     * @brief Texture coordinate of vertex
     */
    glm::vec2 m_texCoords;
    /**
     * @brief Tangent vector of vertex
     */
    glm::vec3 m_tangent;
    /**
     * @brief Bitangent vector of vertex
     */
    glm::vec3 m_bitangent;
    /**
     * @brief Bone indexes which will influence this vertex
     */
    int m_boneIds[MAX_BONE_INFLUENCE];
    /**
     * @brief Weights from each bone
     */
    float m_weights[MAX_BONE_INFLUENCE];
};

class Mesh
{
public:
    // Mesh() = delete;

    /**
     * @brief       Constructor for Mesh class
     * @param[in]   vertices    Vertices data of mesh for vertex array buffer
     * @param[in]   indices     Indices of vertex data in mesh for element array buffer
     */
    Mesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices)
        : m_vertices(vertices), m_indices(indices), m_verticesCount(vertices.size()), m_facesCount(indices.size() / 3)
    {
        setupAABB();
    }

    /**
     * @brief        Copy constructor for Mesh class
     * @param[in]    other            Other mesh to copy
     */
    Mesh(const Mesh& other)
        : m_vertices(other.m_vertices)
        , m_indices(other.m_indices)
        , m_verticesCount(other.m_verticesCount)
        , m_facesCount(other.m_facesCount)
        , m_aabb(other.m_aabb)
    {
    }

    /**
     * @brief Count of vertices in mesh
     */
    unsigned int m_verticesCount;
    /**
     * @brief Count of faces in mesh
     */
    unsigned int m_facesCount;
    /**
     * @brief Vertices data of mesh for vertex array buffer
     */
    std::vector<Vertex> m_vertices;
    /**
     * @brief Indices of vertex data in mesh for element array buffer
     */
    std::vector<unsigned int> m_indices;
    /**
     * @brief AABB bounding box of mesh
     */
    aabb_box_t m_aabb;

protected:
    /**
     * @brief       Set aabb bounding box of mesh
     * @return      void
     */
    void setupAABB()
    {
        for (const auto& vertex : m_vertices)
        {
            m_aabb.m_maximum = glm::max(vertex.m_position, m_aabb.m_maximum);
            m_aabb.m_minimum = glm::min(vertex.m_position, m_aabb.m_minimum);
        }
    }
};