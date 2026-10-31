
#pragma once

#include <cstddef>
#include <vector>

#include "foundation/status.h"

// Spherical interpolation mesh over the measurement directions, ported from the
// reference sofa_hrir.FullHrirField: the convex hull of the unit directions gives
// the containing spherical triangle, its barycentric weights give the local
// interpolation, and the spherical Voronoi cells give the centre weighting.
namespace joc::hrtf {

class SphericalMesh {
public:
    // `directions` is [count][3]; entries need not be unit length.  Coincident
    // directions are merged (first occurrence wins) and their duplicates must be
    // identical, as the reference requires.
    Status build(const std::vector<double>& directions, std::size_t count);

    std::size_t vertex_count() const { return directions_.size() / 3; }
    std::size_t face_count() const { return faces_.size() / 3; }
    const std::vector<double>& directions() const { return directions_; }
    const std::vector<int>& faces() const { return faces_; }
    double max_triangle_edge_degrees() const { return max_triangle_edge_degrees_; }

    // Barycentric interpolation weights of a query direction: an exact measurement
    // returns one id with weight 1, otherwise the containing triangle's vertices
    // with non-negative weights summing to one.
    Status weights(const double direction[3], std::vector<int>* ids,
                   std::vector<double>* weights) const;

    // Spherical Voronoi area of every vertex, normalized to sum to one.
    void voronoi_areas(std::vector<double>* areas) const;

    // Nearest measurement direction (exact search; the vertex count is small).
    std::size_t nearest(const double direction[3], double* distance) const;

private:
    void build_faces();
    void compute_face_inverses();

    std::vector<double> directions_;      // [V][3] unit
    std::vector<int> faces_;              // [F][3]
    std::vector<double> inverse_;         // [F][9] row-major inverse of the vertex matrix
    double max_triangle_edge_degrees_ = 0.0;
};

}  // namespace joc::hrtf
