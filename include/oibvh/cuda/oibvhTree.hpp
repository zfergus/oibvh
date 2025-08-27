#pragma once
#include <oibvh/utils/mesh.hpp>
#include <oibvh/utils/utils.hpp>

#include <glm/glm.hpp>

#include <memory>
#include <vector>

typedef struct scheduling_param
{
    /**
     * @brief  Level of entry
     */
    unsigned int m_entryLevel;

    /**
     * @brief  Total number of real nodes at the corresponding entry level
     */
    unsigned int m_realCount;

    /**
     * @brief  Total number of threads at the corresponding entry level
     */
    unsigned int m_threads;

    /**
     * @brief  Number of threads per group
     */
    unsigned int m_threadsPerGroup;

    scheduling_param(unsigned int entryLevel,
                     unsigned int realCount,
                     unsigned int threads,
                     unsigned int threadsPerGroup)
        : m_entryLevel(entryLevel), m_realCount(realCount), m_threads(threads), m_threadsPerGroup(threadsPerGroup)
    {
    }
} s_param_t;

class OibvhTree
{
public:
    OibvhTree() = delete;

    /**
     * @brief      Constructor for oibvhTree class
     * @param[in]  mesh        Mesh to build oibvh tree
     */
    OibvhTree(const std::shared_ptr<Mesh> mesh);

    /**
     * @brief      Copy constructor for oibvhTree class
     * @param[in]  other            Other oibvhTree to copy
     * @param[in]  mesh             Mesh to build oibvh tree
     */
    OibvhTree(const std::shared_ptr<OibvhTree> other, const std::shared_ptr<Mesh> mesh);

    /**
     * @brief      Destructor for oibvhTree class
     */
    virtual ~OibvhTree();

    /**
     * @brief   Build oibvh tree through gpu
     * @return  void
     */
    void build();

    /**
     * @brief   Refit aabb bounding box of node in oibvh tree
     * @return  void
     */
    virtual void refit();

    /**
     * @brief        Get depth of oibvh tree
     * @return       Depth of oibvh tree
     */
    unsigned int getDepth() const;

    /**
     * @brief        Get primitive count in oibvh tree
     * @return       Primitive count in oibvh tree
     */
    unsigned int getPrimCount() const;

private:
    /**
     * @brief  Set up environment for building oibvh tree through gpu
     * @return void
     */
    void setup();

    /**
     * @brief       Scheduling parameters for building oibvh tree
     * @param[in]   entryLevel            Level of entry
     * @param[in]   realCount             Total number of real nodes at the corresponding entry level
     * @param[in]   threadsPerGroup       Number of threads per group
     * @return      void
     */
    void schedulingParameters(const unsigned int entryLevel,
                              const unsigned int realCount,
                              const unsigned int threadsPerGroup);

protected:
    /**
     * @brief  Mesh to build oibvh tree
     */
    std::shared_ptr<Mesh> m_mesh;
    /**
     * @brief   AABB bvh tree of mesh
     */
    std::vector<aabb_box_t> m_aabbTree;
    /**
     * @brief   Faces of mesh with three vertices' indices
     */
    std::vector<glm::uvec3> m_faces;
    /**
     * @brief   Position of vertices in mesh
     */
    std::vector<glm::vec3> m_positions;
    /**
     * @brief   Have build bvh done or not
     */
    bool m_buildDone;
    /**
     * @brief Parameters of scheduling kernel
     */
    std::vector<s_param_t> m_scheduleParams;

    glm::vec3* m_devicePositions;
    glm::uvec3* m_deviceFaces;
    aabb_box_t* m_deviceAabbs;
};