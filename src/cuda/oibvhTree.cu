#include <oibvh/cuda/oibvh.cuh>
#include <oibvh/cuda/oibvhTree.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <fstream>

OibvhTree::OibvhTree(const std::shared_ptr<Mesh> mesh) : m_mesh(mesh), m_buildDone(false)
{
    deviceMalloc(&m_devicePositions, 10000000);
    deviceMalloc(&m_deviceFaces, 10000000);
    deviceMalloc(&m_deviceAabbs, 10000000);
    setup();
}

OibvhTree::OibvhTree(const std::shared_ptr<OibvhTree> other, const std::shared_ptr<Mesh> mesh)
    : m_mesh(mesh)
    , m_buildDone(other->m_buildDone)
    , m_aabbTree(other->m_aabbTree)
    , m_faces(other->m_faces)
    , m_positions(other->m_positions)
    , m_scheduleParams(other->m_scheduleParams)
{
    deviceMalloc(&m_devicePositions, 10000000);
    deviceMalloc(&m_deviceFaces, 10000000);
    deviceMalloc(&m_deviceAabbs, 10000000);

    setup();
}

OibvhTree::~OibvhTree()
{
    cudaFree(m_devicePositions);
    cudaFree(m_deviceFaces);
    cudaFree(m_deviceAabbs);
}

unsigned int OibvhTree::getDepth() const
{
    return ilog2(m_aabbTree.size());
}

unsigned int OibvhTree::getPrimCount() const
{
    return m_faces.size();
}

void OibvhTree::schedulingParameters(const unsigned int entryLevel,
                                     const unsigned int realCount,
                                     const unsigned int threadsPerGroup)
{
    m_scheduleParams.clear();

    unsigned int l = entryLevel;
    unsigned int r = realCount;
    unsigned int g = std::min(threadsPerGroup, next_power_of_two(r));
    unsigned int t = (r + g - 1) / g * g;

    unsigned int rLast, gLast, tLast;
    while (1)
    {
        rLast = r;
        tLast = t;
        gLast = g;
        m_scheduleParams.push_back({l, rLast, tLast, gLast});

        if (l >= ilog2(gLast) + 1)
            l = l - ilog2(gLast) - 1;
        else
            break;

        r = tLast / gLast;
        r = (r + 1) / 2;
        g = std::min(gLast, next_power_of_two(r));
        t = (r + g - 1) / g * g;
    }
}

void OibvhTree::setup()
{
    if (!m_buildDone)
    {
        std::cout << "---Set up oibvh tree---" << std::endl;
        for (int i = 0; i < m_mesh->m_facesCount; i++)
        {
            m_faces.push_back(
                glm::uvec3(m_mesh->m_indices[i * 3], m_mesh->m_indices[i * 3 + 1], m_mesh->m_indices[i * 3 + 2]));
        }
        for (auto vertex : m_mesh->m_vertices)
        {
            m_positions.push_back(vertex.m_position);
        }
        std::cout << "faces count: " << m_faces.size() << std::endl;
        std::cout << "vertices count: " << m_positions.size() << std::endl;
        std::cout << std::endl;
    }
}

void OibvhTree::refit()
{
    // std::cout << "---Refit---" << std::endl;
    for (int i = 0; i < m_mesh->m_verticesCount; i++)
    {
        m_positions[i] = m_mesh->m_vertices[i].m_position;
    }
    float elapsed_ms = 0.0f;
    const unsigned int primitive_count = m_faces.size();
    const unsigned int vertex_count = m_positions.size();
    const unsigned int oibvh_size = oibvh_get_size(primitive_count);
    const unsigned int oibvh_internal_node_count = oibvh_size - primitive_count;
    glm::vec3* d_positions = m_devicePositions;
    glm::uvec3* d_faces = m_deviceFaces;
    aabb_box_t* d_aabbs = m_deviceAabbs;
    deviceMemcpy(d_positions, m_positions.data(), vertex_count);
    deviceMemcpy(d_faces, m_faces.data(), primitive_count);

    elapsed_ms = kernelLaunch([&]() {
        dim3 blockSize = dim3(256);
        int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        calculate_aabb_kernel<<<gridSize, blockSize>>>(
            d_faces, d_positions, primitive_count, d_aabbs + oibvh_internal_node_count);
    });
    // std::cout << "Refit: AABBs calculation took: " << elapsed_ms << "ms" << std::endl;

    for (int k = 0; k < m_scheduleParams.size(); k++)
    {
        elapsed_ms = kernelLaunch([&]() {
            dim3 blockSize = dim3(m_scheduleParams[k].m_threadsPerGroup);
            dim3 gridSize = dim3(m_scheduleParams[k].m_threads / m_scheduleParams[k].m_threadsPerGroup);
            oibvh_tree_construction_kernel<<<gridSize, blockSize>>>(m_scheduleParams[k].m_entryLevel,
                                                                    m_scheduleParams[k].m_realCount,
                                                                    primitive_count,
                                                                    m_scheduleParams[k].m_threadsPerGroup,
                                                                    d_aabbs);
        });
        // std::cout << "  oibvh refit construct kernel took: " << elapsed_ms << "ms" << std::endl;
    }

    hostMemcpy(m_aabbTree.data(), d_aabbs, oibvh_size);
}

void OibvhTree::build()
{
    std::cout << "---Build oibvh tree---" << std::endl;
    int dev;
    float elapsed_ms = 0.0f;
    cudaGetDevice(&dev);
    std::cout << "device id: " << dev << std::endl;
    const unsigned int primitive_count = m_faces.size();
    const unsigned int vertex_count = m_positions.size();
    const unsigned int oibvh_size = oibvh_get_size(primitive_count);

    const unsigned int oibvh_internal_node_count = oibvh_size - primitive_count;
    glm::vec3* d_positions;
    glm::uvec3* d_faces;
    aabb_box_t* d_aabbs;
    unsigned int* d_mortons;
    deviceMalloc(&d_positions, vertex_count);
    deviceMalloc(&d_faces, primitive_count);
    deviceMalloc(&d_aabbs, oibvh_size);
    deviceMalloc(&d_mortons, primitive_count);
    deviceMemcpy(d_positions, m_positions.data(), vertex_count);
    deviceMemcpy(d_faces, m_faces.data(), primitive_count);

    elapsed_ms = kernelLaunch([&]() {
        dim3 blockSize = dim3(256);
        int bx = (primitive_count + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        calculate_aabb_and_morton_kernel<<<gridSize, blockSize>>>(
            d_faces, d_positions, primitive_count, m_mesh->m_aabb, d_aabbs + oibvh_internal_node_count, d_mortons);
    });
    std::cout << "AABBs and mortons calculation took: " << elapsed_ms << "ms" << std::endl;

#if 0
     // check result
     aabb_box_t* temp_aabbs;
     hostMalloc(&temp_aabbs, primitive_count);
     hostMemcpy(temp_aabbs, d_aabbs + oibvh_internal_node_count, primitive_count);
     aabb_box_t aabb;
     std::cout << sizeof(aabb_box_t) << std::endl;
     aabb.minimum = glm::vec3(1e10);
     aabb.maximum = glm::vec3(-1e10);
     for (int i = 0; i < primitive_count; i++)
    {
        aabb.maximum = glm::max(aabb.maximum, temp_aabbs[i].maximum);
        aabb.minimum = glm::min(aabb.minimum, temp_aabbs[i].minimum);
    }
     aabb == m_mesh->m_aabb ? std::cout << "aabb is correct" << std::endl : std::cout << "aabb is wrong" << std::endl;
     delete[] temp_aabbs;
#endif

    unsigned int* d_mortons_copy;
    deviceMalloc(&d_mortons_copy, primitive_count);
    cudaMemcpy(d_mortons_copy, d_mortons, primitive_count * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    thrust::device_ptr<unsigned int> d_mortons_copy_ptr(d_mortons_copy);
    thrust::device_ptr<unsigned int> d_mortons_ptr(d_mortons);
    thrust::device_ptr<glm::uvec3> d_faces_ptr(d_faces);
    thrust::device_ptr<aabb_box_t> d_aabbs_leaf_ptr(d_aabbs + oibvh_internal_node_count);
    elapsed_ms = kernelLaunch([&]() {
        thrust::stable_sort_by_key(d_mortons_ptr, d_mortons_ptr + primitive_count, d_faces_ptr);
        thrust::stable_sort_by_key(d_mortons_copy_ptr, d_mortons_copy_ptr + primitive_count, d_aabbs_leaf_ptr);
    });
    std::cout << "Sorting took: " << elapsed_ms << "ms" << std::endl;
    cudaFree(d_mortons_copy);

#if 0
    // print result
    aabb_box_t* temp_aabbs;
    hostMalloc(&temp_aabbs, primitive_count);
    hostMemcpy(temp_aabbs, d_aabbs + oibvh_internal_node_count, primitive_count);
    glm::uvec3* temp_faces;
    hostMalloc(&temp_faces, primitive_count);
    hostMemcpy(temp_faces, d_faces, primitive_count);
    for (int i = 0; i < 100; i++)
    {
        std::cout << temp_aabbs[i].minimum << "," << temp_aabbs[i].maximum << std::endl;
        std::cout << m_positions[temp_faces[i].x] << "," << m_positions[temp_faces[i].y] << "," << m_positions[temp_faces[i].z]
                  << std::endl;
    }
#endif

    const unsigned int primitiveCountNextPower2 = next_power_of_two(primitive_count);
    const unsigned int tHeight = ilog2(primitiveCountNextPower2) + 1;
    const unsigned int tLeafLev = tHeight - 1;
    unsigned int entryLevel = tLeafLev - 1;
    const unsigned int virtualLeafCount = primitiveCountNextPower2 - primitive_count;
    unsigned int entryLevelSize = oibvh_level_real_node_count(entryLevel, tLeafLev, virtualLeafCount);

    schedulingParameters(entryLevel, entryLevelSize, THREADS_PER_BLOCK);

#if 0
    // print result
    std::cout << "m_scheduleParams: " << std::endl;
    for (auto param : m_scheduleParams)
    {
        std::cout << param.m_entryLevel << "," << param.m_realCount << "," << param.m_threadsPerGroup << ","
                  << param.m_threads << std::endl;
    }
#endif

    std::cout << "kernel count: " << m_scheduleParams.size() << std::endl;

    for (int k = 0; k < m_scheduleParams.size(); k++)
    {
        std::cout << "kernel" << k << std::endl;
        std::cout << "  entry level: " << m_scheduleParams[k].m_entryLevel << std::endl;
        std::cout << "  real nodes: " << m_scheduleParams[k].m_realCount << std::endl;
        std::cout << "  total threads: " << m_scheduleParams[k].m_threads << std::endl;
        std::cout << "  group size: " << m_scheduleParams[k].m_threadsPerGroup << std::endl;
        std::cout << "  group count: " << m_scheduleParams[k].m_threads / m_scheduleParams[k].m_threadsPerGroup
                  << std::endl;
        elapsed_ms = kernelLaunch([&]() {
            dim3 blockSize = dim3(m_scheduleParams[k].m_threadsPerGroup);
            dim3 gridSize = dim3(m_scheduleParams[k].m_threads / m_scheduleParams[k].m_threadsPerGroup);
            oibvh_tree_construction_kernel2<<<gridSize, blockSize>>>(m_scheduleParams[k].m_entryLevel,
                                                                     m_scheduleParams[k].m_realCount,
                                                                     primitive_count,
                                                                     m_scheduleParams[k].m_threadsPerGroup,
                                                                     d_aabbs);
        });
        std::cout << "  oibvh contruct kernel took: " << elapsed_ms << "ms" << std::endl;
    }
    std::cout << "count of node in oibvh tree: " << oibvh_size << std::endl;

#if 0
    // log result
    aabb_box_t* temp_aabbs;
    hostMalloc(&temp_aabbs, oibvh_size);
    hostMemcpy(temp_aabbs, d_aabbs, oibvh_size);
    std::ofstream outfile;
    outfile.open("..//..//..//logs//bvh_log.txt");
    for (int i = 0; i < oibvh_size; i++)
    {
        outfile << temp_aabbs[i] << std::endl;
    }
    // std::cout << m_mesh->m_aabb.minimum << "," << m_mesh->m_aabb.maximum << std::endl;
#endif

    // copy result to host
    m_aabbTree.resize(oibvh_size);
    hostMemcpy(m_aabbTree.data(), d_aabbs, oibvh_size);
    hostMemcpy(m_faces.data(), d_faces, primitive_count);

    // free memory
    cudaFree(d_positions);
    cudaFree(d_faces);
    cudaFree(d_aabbs);
    cudaFree(d_mortons);

    // build done
    m_buildDone = true;
    std::cout << std::endl;
}