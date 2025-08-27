#pragma once

#include <oibvh/cuda/oibvhTree.hpp>

#include <cuda_runtime.h>

typedef struct bvtt_node
{
    unsigned int m_aabbIndex[2];
} bvtt_node_t;

__global__ void traversal_kernel(bvtt_node_t* src,
                                 bvtt_node_t* dst,
                                 aabb_box_t* aabbs,
                                 tri_pair_node_t* triPairs,
                                 unsigned int* bvhOffsets,
                                 unsigned int* primOffsets,
                                 unsigned int* primCounts,
                                 unsigned int* nextBvttSize,
                                 unsigned int* triPairCount,
                                 unsigned int layoutLength,
                                 unsigned int bvttSize,
                                 unsigned int expandLevels);

__global__ void triangle_intersect_kernel(tri_pair_node_t* triPairs,
                                          glm::uvec3* primitives,
                                          glm::vec3* vertices,
                                          unsigned int* primOffsets,
                                          unsigned int* vertexOffsets,
                                          int_tri_pair_node_t* intTriPairs,
                                          unsigned int* intTriPairCount,
                                          unsigned int layoutLength,
                                          unsigned int triPairCount);