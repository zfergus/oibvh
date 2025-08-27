#include "drawableOibvhTree.hpp"

#include <glad/glad.h> // holds all OpenGL type declarations

DrawableOibvhTree::DrawableOibvhTree(const std::shared_ptr<Mesh> mesh) : OibvhTree(mesh), m_convertDone(false)
{
    setup();
}

DrawableOibvhTree::DrawableOibvhTree(const std::shared_ptr<DrawableOibvhTree> other, const std::shared_ptr<Mesh> mesh)
    : OibvhTree(other, mesh)
    , m_convertDone(other->m_convertDone)
    , m_vertices(other->m_vertices)
    , m_indices(other->m_indices)
{
    setup();
}

DrawableOibvhTree::~DrawableOibvhTree()
{
    glDeleteVertexArrays(1, &m_vertexArrayObj);
    glDeleteBuffers(1, &m_vertexBufferObj);
    glDeleteBuffers(1, &m_elementBufferObj);
}

void DrawableOibvhTree::draw(const Shader& shader)
{
    if (!m_convertDone)
    {
        convertToVertexArray();
    }

    glBindVertexArray(m_vertexArrayObj);
    glDrawElements(GL_LINES, static_cast<unsigned int>(m_indices.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0U);
}

namespace
{

/**
 * @brief      Make cube vertices array and indices for aabb bounding box
 * @param[in]  xSize              Half of x size of aabb bounding box
 * @param[in]  ySize              Half of y size of aabb bounding box
 * @param[in]  zSize              Half of z size of aabb bounding box
 * @param[in]  cubeVertices       Vertices array of cube
 * @param[in]  cubeIndices        Indices array of cube
 * @return     void
 */
void makeCube(const float xSize,
              const float ySize,
              const float zSize,
              std::vector<glm::vec3>& cubeVertices,
              std::vector<unsigned int>& cubeIndices)
{
    // front quad
    cubeVertices.push_back(glm::vec3(-xSize, -ySize, zSize)); // 0
    cubeVertices.push_back(glm::vec3(xSize, -ySize, zSize));  // 1
    cubeVertices.push_back(glm::vec3(xSize, ySize, zSize));   // 2
    cubeVertices.push_back(glm::vec3(-xSize, ySize, zSize));  // 3

    // back quad
    cubeVertices.push_back(glm::vec3(-xSize, -ySize, -zSize)); // 4
    cubeVertices.push_back(glm::vec3(xSize, -ySize, -zSize));  // 5
    cubeVertices.push_back(glm::vec3(xSize, ySize, -zSize));   // 6
    cubeVertices.push_back(glm::vec3(-xSize, ySize, -zSize));  // 7

    // front
    cubeIndices.push_back(0U);
    cubeIndices.push_back(1U);
    /**/
    cubeIndices.push_back(1U);
    cubeIndices.push_back(2U);
    /**/
    cubeIndices.push_back(2U);
    cubeIndices.push_back(3U);
    /**/
    cubeIndices.push_back(3U);
    cubeIndices.push_back(0U);
    // back
    cubeIndices.push_back(4U);
    cubeIndices.push_back(5U);
    /**/
    cubeIndices.push_back(5U);
    cubeIndices.push_back(6U);
    /**/
    cubeIndices.push_back(6U);
    cubeIndices.push_back(7U);
    /**/
    cubeIndices.push_back(7U);
    cubeIndices.push_back(4U);
    // side lines
    cubeIndices.push_back(1U);
    cubeIndices.push_back(5U);
    /**/
    cubeIndices.push_back(0U);
    cubeIndices.push_back(4U);
    /**/
    cubeIndices.push_back(3U);
    cubeIndices.push_back(7U);
    /**/
    cubeIndices.push_back(2U);
    cubeIndices.push_back(6U);
}

} // namespace

void DrawableOibvhTree::convertToVertexArray()
{
    assert(m_buildDone);

    const int primitiveCount = m_faces.size();
    const int numNodesInTree = m_aabbTree.size();
    const int internalNodes = numNodesInTree - primitiveCount;
    const int renderNodes = std::min(internalNodes, 256);
    m_vertices.clear();
    m_indices.clear();

    for (int i = 0; i < renderNodes; i++)
    {
        aabb_box_t aabb = m_aabbTree[i];

        // make cube
        std::vector<glm::vec3> cubeVertices;
        std::vector<unsigned int> cubeIndices;

        makeCube(0.5f * (aabb.m_maximum.x - aabb.m_minimum.x),
                 0.5f * (aabb.m_maximum.y - aabb.m_minimum.y),
                 0.5 * (aabb.m_maximum.z - aabb.m_minimum.z),
                 cubeVertices,
                 cubeIndices);

        int backFaceLowerLeftVertexIndex = 4;
        glm::vec3 backFaceLowerLeftVertex = cubeVertices[backFaceLowerLeftVertexIndex];
        glm::vec3 diff = aabb.m_minimum - backFaceLowerLeftVertex;

        // shift the bounding box to its real position
        for (int j = 0; j < (int)cubeVertices.size(); ++j)
        {
            glm::vec3 pos = cubeVertices[j] + diff;
            m_vertices.push_back(pos);
        }

        // offset indices
        for (int j = 0; j < (int)cubeIndices.size(); ++j)
        {
            cubeIndices[j] += cubeVertices.size() * i;
        }

        m_indices.insert(m_indices.end(), cubeIndices.begin(), cubeIndices.end());
    }

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(glm::vec3), m_vertices.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elementBufferObj);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int), m_indices.data(), GL_STREAM_DRAW);
    glBindVertexArray(0);

    // convert done
    m_convertDone = true;
}

void DrawableOibvhTree::setup()
{
    glGenVertexArrays(1U, &m_vertexArrayObj);
    glGenBuffers(1U, &m_vertexBufferObj);
    glGenBuffers(1U, &m_elementBufferObj);

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_vertices.size(), m_vertices.data(), GL_STREAM_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elementBufferObj);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * m_indices.size(), m_indices.data(), GL_STREAM_DRAW);

    glEnableVertexAttribArray(0U);
    glVertexAttribPointer(0U, 3U, GL_FLOAT, GL_FALSE, 0, (void*)0); // position

    glBindVertexArray(0U);
}

void DrawableOibvhTree::refit()
{
    OibvhTree::refit();
    m_convertDone = false;
}