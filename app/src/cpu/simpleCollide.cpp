#include "simpleCollide.hpp"

#include <glad/glad.h> // holds all OpenGL type declarations

#include <stack>

#define OUTPUT_TIMES 1

typedef struct simple_bvtt_node
{
    std::shared_ptr<simple_bvh_node_t> m_simpleBvhNode;
    unsigned int m_simpleBvhId;
} simple_bvtt_node_t;

typedef struct simple_bvtt_pair_node
{
    simple_bvtt_node_t m_simpleBvttNode[2];
    unsigned int depth;
} simple_bvtt_pair_node_t;

SimpleCollide::SimpleCollide() : m_primCount(0U), m_vertexCount(0U), m_outputTimes(0U), m_convertDone(false)
{
    glGenVertexArrays(1U, &m_vertexArrayObj);
    glGenBuffers(1U, &m_vertexBufferObj);

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_vertices.size(), m_vertices.data(), GL_STREAM_DRAW);

    glEnableVertexAttribArray(0U);
    glVertexAttribPointer(0U, 3U, GL_FLOAT, GL_FALSE, 0, (void*)0); // position

    glBindVertexArray(0U);
}

SimpleCollide::~SimpleCollide()
{
    glDeleteVertexArrays(1, &m_vertexArrayObj);
    glDeleteBuffers(1, &m_vertexBufferObj);
}

void SimpleCollide::addSimpleBVH(const std::shared_ptr<SimpleBVH>& simpleBvh)
{
    assert(simpleBvh->m_buildDone);
    m_simpleBvhs.push_back(simpleBvh);
    m_primCount += simpleBvh->m_faces.size();
    m_vertexCount += simpleBvh->m_positions.size();
}

namespace
{
bool project6(const glm::vec3& ax,
              const glm::vec3& p1,
              const glm::vec3& p2,
              const glm::vec3& p3,
              const glm::vec3& q1,
              const glm::vec3& q2,
              const glm::vec3& q3)
{
    float P1 = glm::dot(ax, p1);
    float P2 = glm::dot(ax, p2);
    float P3 = glm::dot(ax, p3);
    float Q1 = glm::dot(ax, q1);
    float Q2 = glm::dot(ax, q2);
    float Q3 = glm::dot(ax, q3);
    float mx1 = std::fmaxf(std::fmaxf(P1, P2), P3);
    float mn1 = std::fminf(std::fminf(P1, P2), P3);
    float mx2 = std::fmaxf(std::fmaxf(Q1, Q2), Q3);
    float mn2 = std::fminf(std::fminf(Q1, Q2), Q3);

    if (mn1 > mx2)
        return false;
    if (mn2 > mx1)
        return false;
    return true;
}

bool triangleIntersect(const glm::vec3& P1,
                       const glm::vec3& P2,
                       const glm::vec3& P3,
                       const glm::vec3& Q1,
                       const glm::vec3& Q2,
                       const glm::vec3& Q3)
{
    glm::vec3 p1(0);
    glm::vec3 p2 = P2 - P1;
    glm::vec3 p3 = P3 - P1;
    glm::vec3 q1 = Q1 - P1;
    glm::vec3 q2 = Q2 - P1;
    glm::vec3 q3 = Q3 - P1;
    glm::vec3 e1 = p2 - p1;
    glm::vec3 e2 = p3 - p2;
    glm::vec3 e3 = p1 - p3;
    glm::vec3 f1 = q2 - q1;
    glm::vec3 f2 = q3 - q2;
    glm::vec3 f3 = q1 - q3;
    glm::vec3 n1 = glm::cross(e1, e2);
    glm::vec3 m1 = glm::cross(f1, f2);
    glm::vec3 g1 = glm::cross(e1, n1);
    glm::vec3 g2 = glm::cross(e2, n1);
    glm::vec3 g3 = glm::cross(e3, n1);
    glm::vec3 h1 = glm::cross(f1, m1);
    glm::vec3 h2 = glm::cross(f2, m1);
    glm::vec3 h3 = glm::cross(f3, m1);
    glm::vec3 ef11 = glm::cross(e1, f1);
    glm::vec3 ef12 = glm::cross(e1, f2);
    glm::vec3 ef13 = glm::cross(e1, f3);
    glm::vec3 ef21 = glm::cross(e2, f1);
    glm::vec3 ef22 = glm::cross(e2, f2);
    glm::vec3 ef23 = glm::cross(e2, f3);
    glm::vec3 ef31 = glm::cross(e3, f1);
    glm::vec3 ef32 = glm::cross(e3, f2);
    glm::vec3 ef33 = glm::cross(e3, f3);
    // now begin the series of tests
    if (!project6(n1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(m1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef11, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef12, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef13, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef21, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef22, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef23, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef31, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef32, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef33, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g3, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h3, p1, p2, p3, q1, q2, q3))
        return false;
    return true;
}
} // namespace

void SimpleCollide::detect(bool printInformation)
{
    if (printInformation && m_outputTimes < OUTPUT_TIMES)
    {
        std::cout << "---Simple collide detecting---" << std::endl;
    }
    m_intTriPairs.clear();
    std::stack<simple_bvtt_pair_node> simpleBvttNodePairs;
    for (unsigned int i = 0; i < m_simpleBvhs.size(); i++)
        for (unsigned int j = i + 1; j < m_simpleBvhs.size(); j++)
        {
            simpleBvttNodePairs.push(simple_bvtt_pair_node{
                simple_bvtt_node_t{m_simpleBvhs[i]->m_root, i}, simple_bvtt_node_t{m_simpleBvhs[j]->m_root, j}, 0u});
        }

    unsigned int potentialIntTriPairCount = 0U;
    int a[19] = {0};
    while (!simpleBvttNodePairs.empty())
    {
        const auto simpleBvttNodePair = simpleBvttNodePairs.top();
        const auto depth = simpleBvttNodePair.depth;
        a[depth]++;
        simpleBvttNodePairs.pop();
        const auto simpleBvttNodeA = simpleBvttNodePair.m_simpleBvttNode[0];
        const auto simpleBvttNodeB = simpleBvttNodePair.m_simpleBvttNode[1];
        const auto simpleBvhNodeA = simpleBvttNodeA.m_simpleBvhNode;
        const auto simpleBvhNodeB = simpleBvttNodeB.m_simpleBvhNode;
        if (simpleBvhNodeA->m_aabb.overlap(simpleBvhNodeB->m_aabb)) // aabbs overlap
        {
            const unsigned int simpleBvhIdA = simpleBvttNodeA.m_simpleBvhId;
            const unsigned int simpleBvhIdB = simpleBvttNodeB.m_simpleBvhId;
            if (simpleBvhNodeA->m_triId != -1 && simpleBvhNodeB->m_triId != -1) // both nodes are leaf nodes
            {
                potentialIntTriPairCount++;
                const glm::uvec3 faceA = m_simpleBvhs[simpleBvhIdA]->m_faces[simpleBvhNodeA->m_triId];
                const glm::uvec3 faceB = m_simpleBvhs[simpleBvhIdB]->m_faces[simpleBvhNodeB->m_triId];
                const auto P1 = m_simpleBvhs[simpleBvhIdA]->m_positions[faceA.x];
                const auto P2 = m_simpleBvhs[simpleBvhIdA]->m_positions[faceA.y];
                const auto P3 = m_simpleBvhs[simpleBvhIdA]->m_positions[faceA.z];
                const auto Q1 = m_simpleBvhs[simpleBvhIdB]->m_positions[faceB.x];
                const auto Q2 = m_simpleBvhs[simpleBvhIdB]->m_positions[faceB.y];
                const auto Q3 = m_simpleBvhs[simpleBvhIdB]->m_positions[faceB.z];
                if (triangleIntersect(P1, P2, P3, Q1, Q2, Q3))
                {
                    assert(simpleBvhNodeA->m_triId > 0);
                    assert(simpleBvhNodeB->m_triId > 0);
                    // triangles intersect
                    m_intTriPairs.push_back(int_tri_pair_node_t{simpleBvhIdA,
                                                                simpleBvhIdB,
                                                                (unsigned int)simpleBvhNodeA->m_triId,
                                                                (unsigned int)simpleBvhNodeB->m_triId});
                }
                continue;
            }

            simple_bvtt_node_t newSimpleBvttNodeAs[2];
            simple_bvtt_node_t newSimpleBvttNodeBs[2];
            unsigned int newSimpleBvttNodeCountA = 0U;
            unsigned int newSimpleBvttNodeCountB = 0U;
            if (simpleBvhNodeA->m_triId != -1) // A node is leaf node
            {
                newSimpleBvttNodeAs[newSimpleBvttNodeCountA++] = simple_bvtt_node_t{simpleBvhNodeA, simpleBvhIdA};
            }
            else // A node is internal node
            {
                newSimpleBvttNodeAs[newSimpleBvttNodeCountA++] =
                    simple_bvtt_node_t{simpleBvhNodeA->m_left, simpleBvhIdA};
                if (simpleBvhNodeA->m_right != nullptr)
                {
                    newSimpleBvttNodeAs[newSimpleBvttNodeCountA++] =
                        simple_bvtt_node_t{simpleBvhNodeA->m_right, simpleBvhIdA};
                }
            }

            if (simpleBvhNodeB->m_triId != -1) // B node is leaf node
            {
                newSimpleBvttNodeBs[newSimpleBvttNodeCountB++] = simple_bvtt_node_t{simpleBvhNodeB, simpleBvhIdB};
            }
            else // B node is internal node
            {
                newSimpleBvttNodeBs[newSimpleBvttNodeCountB++] =
                    simple_bvtt_node_t{simpleBvhNodeB->m_left, simpleBvhIdB};
                if (simpleBvhNodeB->m_right != nullptr)
                {
                    newSimpleBvttNodeBs[newSimpleBvttNodeCountB++] =
                        simple_bvtt_node_t{simpleBvhNodeB->m_right, simpleBvhIdB};
                }
            }

            for (unsigned int i = 0; i < newSimpleBvttNodeCountA; i++)
                for (unsigned int j = 0; j < newSimpleBvttNodeCountB; j++)
                {
                    simpleBvttNodePairs.push(
                        simple_bvtt_pair_node_t{newSimpleBvttNodeAs[i], newSimpleBvttNodeBs[j], depth + 1});
                }
        }
    }

    if (printInformation && m_outputTimes < OUTPUT_TIMES)
    {
        std::cout << "potential intersect triangle pairs count: " << potentialIntTriPairCount << std::endl;
        std::cout << "Intersect triangle pairs count: " << m_intTriPairs.size() << std::endl;
        for (int i = 0; i < 19; i++)
        {
            std::cout << "bvtt depth" << i << ": " << a[i] << std::endl;
        }
        std::cout << std::endl;
    }

    if (printInformation)
    {
        m_outputTimes++;
    }
    m_convertDone = false;
}

void SimpleCollide::draw()
{
    if (!m_convertDone)
    {
        convertToVertexArray();
        m_convertDone = true;
    }

    glBindVertexArray(m_vertexArrayObj);
    glDrawArrays(GL_TRIANGLES, 0, m_vertices.size());
    glBindVertexArray(0U);
}

bool SimpleCollide::check(const unsigned int intTriPairsCount) const
{
    if (intTriPairsCount != m_intTriPairs.size())
    {
        return false;
    }

    return true;
}

unsigned int SimpleCollide::getIntTriPairCount() const
{
    return m_intTriPairs.size();
}

void SimpleCollide::convertToVertexArray()
{
    m_vertices.resize(m_intTriPairs.size() * 6);
    for (int i = 0; i < m_intTriPairs.size(); i++)
    {
        const auto intTriPair = m_intTriPairs[i];
        const auto bvhA = m_simpleBvhs[intTriPair.m_bvhIndex[0]];
        const auto bvhB = m_simpleBvhs[intTriPair.m_bvhIndex[1]];
        const auto triangleA = bvhA->m_faces[intTriPair.m_triIndex[0]];
        const auto triangleB = bvhB->m_faces[intTriPair.m_triIndex[1]];
        m_vertices[i * 6] = bvhA->m_positions[triangleA.x];
        m_vertices[i * 6 + 1] = bvhA->m_positions[triangleA.y];
        m_vertices[i * 6 + 2] = bvhA->m_positions[triangleA.z];
        m_vertices[i * 6 + 3] = bvhB->m_positions[triangleB.x];
        m_vertices[i * 6 + 4] = bvhB->m_positions[triangleB.y];
        m_vertices[i * 6 + 5] = bvhB->m_positions[triangleB.z];
    }

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(glm::vec3), m_vertices.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_convertDone = true;
}
