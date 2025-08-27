#include <oibvh/cuda/collide.cuh>
#include <oibvh/cuda/oibvh.cuh>
#include <oibvh/cuda/utils.cuh>

#include <device_launch_parameters.h>

#define min(x, y) (x < y ? x : y)
#define LOCALMEMSIZE 2048

__device__ inline bool overlap(const aabb_box_t& aabb1, const aabb_box_t& aabb2)
{
    return (aabb1.m_minimum.x <= aabb2.m_maximum.x && aabb1.m_maximum.x >= aabb2.m_minimum.x) &&
        (aabb1.m_minimum.y <= aabb2.m_maximum.y && aabb1.m_maximum.y >= aabb2.m_minimum.y) &&
        (aabb1.m_minimum.z <= aabb2.m_maximum.z && aabb1.m_maximum.z >= aabb2.m_minimum.z);
}

__device__ inline void read_information(unsigned int* sharedAabbOffsets,
                                        unsigned int* sharedPrimOffsets,
                                        unsigned int* sharedPrimCount,
                                        unsigned int aabbIndex,
                                        unsigned int layoutLength,
                                        unsigned int& aabbOffset,
                                        unsigned int& primOffset,
                                        unsigned int& primitiveCount)
{
    int l = 0;
    int r = layoutLength - 1;
    int m;
    int idxLayout;
    while (l <= r)
    {
        m = (l + r) / 2;
        if (sharedAabbOffsets[m] <= aabbIndex)
        {
            l = m + 1;
            idxLayout = m;
        }
        else
            r = m - 1;
    }
    aabbOffset = sharedAabbOffsets[idxLayout];
    primOffset = sharedPrimOffsets[idxLayout];
    primitiveCount = sharedPrimCount[idxLayout];
}

__device__ inline void read_information(unsigned int* sharedPrimOffsets,
                                        unsigned int* sharedVertexOffsets,
                                        unsigned int primIndex,
                                        unsigned int layoutLength,
                                        unsigned int& bvhIndex,
                                        unsigned int& primOffset,
                                        unsigned int& vertexOffset)
{
    int l = 0;
    int r = layoutLength - 1;
    int m;
    int idxLayout;
    while (l <= r)
    {
        m = (l + r) / 2;
        if (sharedPrimOffsets[m] <= primIndex)
        {
            l = m + 1;
            idxLayout = m;
        }
        else
            r = m - 1;
    }
    bvhIndex = idxLayout;
    primOffset = sharedPrimOffsets[idxLayout];
    vertexOffset = sharedVertexOffsets[idxLayout];
}

__global__ void traversal_kernel(bvtt_node_t* src,
                                 bvtt_node_t* dst,
                                 aabb_box_t* aabbs,
                                 tri_pair_node_t* triPairs,
                                 unsigned int* aabbOffsets,
                                 unsigned int* primOffsets,
                                 unsigned int* primCounts,
                                 unsigned int* nextBvttSize,
                                 unsigned int* triPairCount,
                                 unsigned int layoutLength,
                                 unsigned int bvttSize,
                                 unsigned int expandLevels)
{
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localIndex = threadIdx.x;
    __shared__ unsigned int sharedAabbOffsets[256];
    __shared__ unsigned int sharedPrimOffsets[256];
    __shared__ unsigned int sharedPrimCounts[256];
    if (localIndex < layoutLength)
    {

        sharedAabbOffsets[localIndex] = aabbOffsets[localIndex];
        sharedPrimOffsets[localIndex] = primOffsets[localIndex];
        sharedPrimCounts[localIndex] = primCounts[localIndex];
    }
    __syncthreads();

    unsigned int aabbOffsetA, aabbOffsetB;
    unsigned int startImplicitIdxA, startImplicitIdxB;
    unsigned int expandNodeCountA = 0;
    unsigned int expandNodeCountB = 0;
    unsigned int realIndexA, realIndexB;
    unsigned int virtualCountA, virtualCountB;
    unsigned int levelA, levelB;
    unsigned int leafLevA, leafLevB;
    unsigned int numData = 0;

    if (globalIndex < bvttSize)
    {
        bvtt_node_t node = src[globalIndex];
        const unsigned int aabbIndexA = node.m_aabbIndex[0];
        const unsigned int aabbIndexB = node.m_aabbIndex[1];
        aabb_box_t aabbA = aabbs[aabbIndexA];
        aabb_box_t aabbB = aabbs[aabbIndexB];

        if (overlap(aabbA, aabbB)) // AABB overlap
        {
            // expand bvtt node
            unsigned int primOffsetA, primitiveCountA; // A oibvh tree
            unsigned int primOffsetB, primitiveCountB; // B oibvh tree
            read_information(sharedAabbOffsets,
                             sharedPrimOffsets,
                             sharedPrimCounts,
                             aabbIndexA,
                             layoutLength,
                             aabbOffsetA,
                             primOffsetA,
                             primitiveCountA);
            read_information(sharedAabbOffsets,
                             sharedPrimOffsets,
                             sharedPrimCounts,
                             aabbIndexB,
                             layoutLength,
                             aabbOffsetB,
                             primOffsetB,
                             primitiveCountB);
            const unsigned int primCountNextPower2A = next_power_of_two(primitiveCountA);
            const unsigned int primCountNextPower2B = next_power_of_two(primitiveCountB);
            virtualCountA = primCountNextPower2A - primitiveCountA;
            virtualCountB = primCountNextPower2B - primitiveCountB;
            leafLevA = ilog2(primCountNextPower2A);
            leafLevB = ilog2(primCountNextPower2B);
            realIndexA = aabbIndexA - aabbOffsetA;
            realIndexB = aabbIndexB - aabbOffsetB;
            const unsigned int implicitIndexA = oibvh_real_to_implicit(realIndexA, leafLevA, virtualCountA);
            const unsigned int implicitIndexB = oibvh_real_to_implicit(realIndexB, leafLevB, virtualCountB);
            levelA = ilog2(implicitIndexA + 1);
            levelB = ilog2(implicitIndexB + 1);

            if (levelA == leafLevA && levelB == leafLevB) // a and b are both at leaf node
            {
                const unsigned int primIndexA = primOffsetA + implicitIndexA + 1 - (1 << leafLevA);
                const unsigned int primIndexB = primOffsetB + implicitIndexB + 1 - (1 << leafLevB);
                const tri_pair_node_t triPair{primIndexA, primIndexB};
                const unsigned int triPairIndex = atomicAdd(triPairCount, 1u);
                triPairs[triPairIndex] = triPair;
            }
            else
            {
                const unsigned int nextLevelA = min(leafLevA, levelA + expandLevels);
                const unsigned int nextLevelB = min(leafLevB, levelB + expandLevels);
                startImplicitIdxA = oibvh_get_most_left_descendant_implicitIdx(implicitIndexA, nextLevelA - levelA);
                startImplicitIdxB = oibvh_get_most_left_descendant_implicitIdx(implicitIndexB, nextLevelB - levelB);
                const unsigned int mostRightValidImplicitIdxA =
                    oibvh_get_most_right_valid_implicitIdx(nextLevelA, leafLevA, virtualCountA);
                const unsigned int mostRightValidImplicitIdxB =
                    oibvh_get_most_right_valid_implicitIdx(nextLevelB, leafLevB, virtualCountB);
                expandNodeCountA =
                    min((1 << (nextLevelA - levelA)), mostRightValidImplicitIdxA - startImplicitIdxA + 1);
                expandNodeCountB =
                    min((1 << (nextLevelB - levelB)), mostRightValidImplicitIdxB - startImplicitIdxB + 1);
            }
            numData = expandNodeCountA * expandNodeCountB;
        }
    }

    __shared__ bvtt_node_t bvttNodes[LOCALMEMSIZE];
    __shared__ unsigned int C;
    __shared__ unsigned int baseOffset;
    C = 0;
    int times = 0;
    int limitNumData = blockDim.x * (1 << (2 * expandLevels));
    unsigned int c;
    unsigned int checkpoint = 0;
    int indexOffsetA = 0;
    int indexOffsetB = 0;
    __syncthreads();

    do
    {
        if (numData > 0)
        {
            c = atomicAdd(&C, numData) - checkpoint;
            if (c < LOCALMEMSIZE)
            {
                unsigned int r = LOCALMEMSIZE - c; // remaining space
                unsigned int w = min(numData, r);  // amount written
                // write bvtt nodes to share memory
                for (int cnt = 0; cnt < w; cnt++)
                {
                    realIndexA = oibvh_implicit_to_real(startImplicitIdxA + indexOffsetA, leafLevA, virtualCountA);
                    realIndexB = oibvh_implicit_to_real(startImplicitIdxB + indexOffsetB, leafLevB, virtualCountB);
                    bvttNodes[c + cnt] = bvtt_node_t{realIndexA + aabbOffsetA, realIndexB + aabbOffsetB};
                    indexOffsetB++;
                    if (indexOffsetB % expandNodeCountB == 0)
                    {
                        indexOffsetB = 0;
                        indexOffsetA++;
                    }
                }
                numData = numData - w;
            }
        }
        __syncthreads();
        unsigned int s = min(C - checkpoint, LOCALMEMSIZE);
        if (s > 0)
        {
            checkpoint = C;
            if (localIndex == 0)
            {
                baseOffset = atomicAdd(nextBvttSize, s);
            }
            __syncthreads();
            // write bvtt node from share memory to global memory
            for (int i = 0; i < (LOCALMEMSIZE + blockDim.x - 1) / blockDim.x; i++)
            {
                if (i * blockDim.x + localIndex >= s)
                {
                    break;
                }
                dst[baseOffset + i * blockDim.x + localIndex] = bvttNodes[i * blockDim.x + localIndex];
            }
            __syncthreads();
        }
        else
        {
            break;
        }
        times++;
    } while (times * LOCALMEMSIZE < limitNumData);
}

namespace
{

// Projects the vertices of two triangles onto the given axis and checks for overlap.
// Returns true if the projections overlap (i.e., no separating axis found).
__device__ bool project6(const glm::vec3& ax,
                         const glm::vec3& p1,
                         const glm::vec3& p2,
                         const glm::vec3& p3,
                         const glm::vec3& q1,
                         const glm::vec3& q2,
                         const glm::vec3& q3)
{
    // Project triangle A vertices onto axis
    float P1 = dot(ax, p1);
    float P2 = dot(ax, p2);
    float P3 = dot(ax, p3);

    // Project triangle B vertices onto axis
    float Q1 = dot(ax, q1);
    float Q2 = dot(ax, q2);
    float Q3 = dot(ax, q3);

    // Find min/max projections for both triangles
    float mx1 = fmax(fmax(P1, P2), P3);
    float mn1 = fmin(fmin(P1, P2), P3);
    float mx2 = fmax(fmax(Q1, Q2), Q3);
    float mn2 = fmin(fmin(Q1, Q2), Q3);

    // Check for overlap in projections
    return (mn1 <= mx2) && (mn2 <= mx1);
}

} // namespace

__device__ bool triangles_intersect(const glm::vec3& P1,
                                    const glm::vec3& P2,
                                    const glm::vec3& P3,
                                    const glm::vec3& Q1,
                                    const glm::vec3& Q2,
                                    const glm::vec3& Q3)
{
    // One triangle is (p1,p2,p3).  Other is (q1,q2,q3).
    // Edges are (e1,e2,e3) and (f1,f2,f3).
    // Normals are n1 and m1
    // Outwards are (g1,g2,g3) and (h1,h2,h3).
    //
    // We assume that the triangle vertices are in the same coordinate system.
    //
    // First thing we do is establish a new c.s. so that p1 is at (0,0,0).

    glm::vec3 p1, p2, p3;
    glm::vec3 q1, q2, q3;
    glm::vec3 e1, e2, e3;
    glm::vec3 f1, f2, f3;
    glm::vec3 g1, g2, g3;
    glm::vec3 h1, h2, h3;
    glm::vec3 n1, m1;

    glm::vec3 ef11, ef12, ef13;
    glm::vec3 ef21, ef22, ef23;
    glm::vec3 ef31, ef32, ef33;

    p1 = glm::vec3(0, 0, 0);
    p2 = P2 - P1;
    p3 = P3 - P1;

    q1 = Q1 - P1;
    q2 = Q2 - P1;
    q3 = Q3 - P1;

    e1 = p2 - p1;
    e2 = p3 - p2;
    e3 = p1 - p3;

    f1 = q2 - q1;
    f2 = q3 - q2;
    f3 = q1 - q3;

    n1 = cross(e1, e2);
    m1 = cross(f1, f2);

    g1 = cross(e1, n1);
    g2 = cross(e2, n1);
    g3 = cross(e3, n1);
    h1 = cross(f1, m1);
    h2 = cross(f2, m1);
    h3 = cross(f3, m1);

    ef11 = cross(e1, f1);
    ef12 = cross(e1, f2);
    ef13 = cross(e1, f3);
    ef21 = cross(e2, f1);
    ef22 = cross(e2, f2);
    ef23 = cross(e2, f3);
    ef31 = cross(e3, f1);
    ef32 = cross(e3, f2);
    ef33 = cross(e3, f3);

    // now begin the series of tests

    return project6(n1, p1, p2, p3, q1, q2, q3) && project6(m1, p1, p2, p3, q1, q2, q3) &&
        project6(ef11, p1, p2, p3, q1, q2, q3) && project6(ef12, p1, p2, p3, q1, q2, q3) &&
        project6(ef13, p1, p2, p3, q1, q2, q3) && project6(ef21, p1, p2, p3, q1, q2, q3) &&
        project6(ef22, p1, p2, p3, q1, q2, q3) && project6(ef23, p1, p2, p3, q1, q2, q3) &&
        project6(ef31, p1, p2, p3, q1, q2, q3) && project6(ef32, p1, p2, p3, q1, q2, q3) &&
        project6(ef33, p1, p2, p3, q1, q2, q3) && project6(g1, p1, p2, p3, q1, q2, q3) &&
        project6(g2, p1, p2, p3, q1, q2, q3) && project6(g3, p1, p2, p3, q1, q2, q3) &&
        project6(h1, p1, p2, p3, q1, q2, q3) && project6(h2, p1, p2, p3, q1, q2, q3) &&
        project6(h3, p1, p2, p3, q1, q2, q3);
}

__global__ void triangle_intersect_kernel(tri_pair_node_t* triPairs,
                                          glm::uvec3* primitives,
                                          glm::vec3* vertices,
                                          unsigned int* primOffsets,
                                          unsigned int* vertexOffsets,
                                          int_tri_pair_node_t* intTriPairs,
                                          unsigned int* intTriPairCount,
                                          unsigned int layoutLength,
                                          unsigned int triPairCount)
{
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localIndex = threadIdx.x;
    __shared__ unsigned int sharedPrimOffsets[256];
    __shared__ unsigned int sharedVertexOffsets[256];
    if (localIndex < layoutLength)
    {
        sharedPrimOffsets[localIndex] = primOffsets[localIndex];
        sharedVertexOffsets[localIndex] = vertexOffsets[localIndex];
    }
    __syncthreads();
    if (globalIndex >= triPairCount)
        return;

    tri_pair_node_t node = triPairs[globalIndex];
    unsigned int triIndexA = node.m_triIndex[0];
    unsigned int triIndexB = node.m_triIndex[1];
    unsigned int bvhIndexA, bvhIndexB, primOffsetA, primOffsetB, vertexOffsetA, vertexOffsetB;
    read_information(
        sharedPrimOffsets, sharedVertexOffsets, triIndexA, layoutLength, bvhIndexA, primOffsetA, vertexOffsetA);
    read_information(
        sharedPrimOffsets, sharedVertexOffsets, triIndexB, layoutLength, bvhIndexB, primOffsetB, vertexOffsetB);
    glm::uvec3 triangleA = primitives[triIndexA];
    glm::uvec3 triangleB = primitives[triIndexB];

    glm::vec3 triVerticesA[3], triVerticesB[3];
    for (int i = 0; i < 3; i++)
    {
        triVerticesA[i] = vertices[vertexOffsetA + triangleA[i]];
        triVerticesB[i] = vertices[vertexOffsetB + triangleB[i]];
    }

    // triangle intersect
    if (triangles_intersect(
            triVerticesA[0], triVerticesA[1], triVerticesA[2], triVerticesB[0], triVerticesB[1], triVerticesB[2]))
    {
        unsigned int intTriPairOffset = atomicAdd(intTriPairCount, 1u);
        int_tri_pair_node_t intTriPair{bvhIndexA, bvhIndexB, triIndexA - primOffsetA, triIndexB - primOffsetB};
        intTriPairs[intTriPairOffset] = intTriPair;
    }
}