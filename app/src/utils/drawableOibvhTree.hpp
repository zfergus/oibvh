#pragma once

#include "shader.hpp"

#include <oibvh/cuda/oibvhTree.hpp>

class DrawableOibvhTree : public OibvhTree
{
public:
    DrawableOibvhTree() = delete;

    /**
     * @brief      Constructor for oibvhTree class
     * @param[in]  mesh        Mesh to build oibvh tree
     */
    DrawableOibvhTree(const std::shared_ptr<Mesh> mesh);

    /**
     * @brief      Copy constructor for oibvhTree class
     * @param[in]  other            Other oibvhTree to copy
     * @param[in]  mesh             Mesh to build oibvh tree
     */
    DrawableOibvhTree(const std::shared_ptr<DrawableOibvhTree> other, const std::shared_ptr<Mesh> mesh);

    /**
     * @brief      Destructor for oibvhTree class
     */
    ~DrawableOibvhTree();

    /**
     * @brief         Draw mesh and bounding box(drawBox = true)
     * @param[in]     shader        shader to use on box
     * @return        void
     */
    void draw(const Shader& shader);

    /**
     * @brief   Refit aabb bounding box of node in oibvh tree
     * @return  void
     */
    virtual void refit();

private:
    /**
     * @brief  Set up environment for building oibvh tree through gpu
     * @return void
     */
    void setup();

private:
    /**
     * @brief  Convert bvh data to vertex array for rendering
     * @return void
     */
    void convertToVertexArray();

    /**
     * @brief   Have converted bvh data to vertex array for rendering or not
     */
    bool m_convertDone;
    /**
     * @brief Vertex arrays object id
     */
    unsigned int m_vertexArrayObj;
    /**
     * @brief Vertex buffer object id
     */
    unsigned int m_vertexBufferObj;
    /**
     * @brief Element buffer object id
     */
    unsigned int m_elementBufferObj;
    /**
     * @brief Vertex array for bvh
     */
    std::vector<glm::vec3> m_vertices;
    /**
     * @brief Indices of vertex array for bvh
     */
    std::vector<unsigned int> m_indices;

    friend class Scene;
};