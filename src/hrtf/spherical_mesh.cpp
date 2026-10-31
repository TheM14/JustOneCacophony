
#include "hrtf/spherical_mesh.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace joc::hrtf {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

Status mesh_error(const char* message) {
    return Status::fail(JOC_ERR_INVALID_ARGUMENT, joc::stage::kRender, message);
}

void normalize(double vector[3]) {
    const double length = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] +
                                    vector[2] * vector[2]);
    if (length > 0.0) {
        vector[0] /= length;
        vector[1] /= length;
        vector[2] /= length;
    }
}

void cross(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

double dot(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Solid angle of the spherical triangle (a, b, c), Van Oosterom & Strackee.
double triangle_area(const double a[3], const double b[3], const double c[3]) {
    const double determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) -
                               a[1] * (b[0] * c[2] - b[2] * c[0]) +
                               a[2] * (b[0] * c[1] - b[1] * c[0]);
    const double denominator = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
    return 2.0 * std::atan2(std::fabs(determinant), denominator);
}

// In-place inverse of a row-major 3x3 matrix.  Returns false when singular.
bool invert3(const double matrix[9], double out[9]) {
    const double c00 = matrix[4] * matrix[8] - matrix[5] * matrix[7];
    const double c01 = matrix[5] * matrix[6] - matrix[3] * matrix[8];
    const double c02 = matrix[3] * matrix[7] - matrix[4] * matrix[6];
    const double determinant = matrix[0] * c00 + matrix[1] * c01 + matrix[2] * c02;
    if (!std::isfinite(determinant) || std::fabs(determinant) < 1e-12) {
        return false;
    }
    const double inverse = 1.0 / determinant;
    out[0] = c00 * inverse;
    out[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse;
    out[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse;
    out[3] = c01 * inverse;
    out[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse;
    out[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse;
    out[6] = c02 * inverse;
    out[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inverse;
    out[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse;
    return true;
}

struct HullFace {
    int vertex[3] = {0, 0, 0};
    double normal[3] = {0.0, 0.0, 0.0};
    bool alive = true;
};

}  // namespace

Status SphericalMesh::build(const std::vector<double>& directions, std::size_t count) {
    if (count < 4 || directions.size() < count * 3) {
        return mesh_error("spherical mesh needs at least four directions");
    }
    directions_.clear();
    faces_.clear();
    inverse_.clear();
    max_triangle_edge_degrees_ = 0.0;

    // Merge coincident directions, keeping the first occurrence (reference order).
    std::vector<double> rounded(count * 3);
    for (std::size_t index = 0; index < count * 3; ++index) {
        rounded[index] = std::round(directions[index] * 1e10) / 1e10;
    }
    std::vector<std::size_t> kept;
    std::size_t slot_count = 0;
    for (std::size_t index = 0; index < count; ++index) {
        std::size_t found = slot_count;
        for (std::size_t slot = 0; slot < kept.size(); ++slot) {
            const std::size_t first = kept[slot];
            if (rounded[first * 3] == rounded[index * 3] &&
                rounded[first * 3 + 1] == rounded[index * 3 + 1] &&
                rounded[first * 3 + 2] == rounded[index * 3 + 2]) {
                found = slot;
                break;
            }
        }
        if (found == slot_count) {
            kept.push_back(index);
            ++slot_count;
        }
    }
    if (kept.size() < 4) {
        return mesh_error("spherical mesh needs four distinct directions");
    }
    directions_.resize(kept.size() * 3);
    for (std::size_t slot = 0; slot < kept.size(); ++slot) {
        double value[3] = {directions[kept[slot] * 3], directions[kept[slot] * 3 + 1],
                           directions[kept[slot] * 3 + 2]};
        if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2])) {
            return mesh_error("measurement direction is not finite");
        }
        normalize(value);
        std::copy(value, value + 3, directions_.begin() + static_cast<std::ptrdiff_t>(slot * 3));
    }
    build_faces();
    compute_face_inverses();
    return Status::success();
}

void SphericalMesh::build_faces() {
    const std::size_t count = vertex_count();
    const auto point = [this](std::size_t index) {
        return directions_.data() + index * 3;
    };

    // Initial tetrahedron: the farthest pair, then the farthest point from that
    // line, then the farthest point from that plane.
    std::size_t i0 = 0;
    std::size_t i1 = 0;
    double best = -1.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double dx = point(index)[0] - point(i0)[0];
        const double dy = point(index)[1] - point(i0)[1];
        const double dz = point(index)[2] - point(i0)[2];
        const double distance = dx * dx + dy * dy + dz * dz;
        if (distance > best) {
            best = distance;
            i1 = index;
        }
    }
    if (best <= 0.0) {
        return;
    }
    std::size_t i2 = 0;
    best = -1.0;
    double edge[3];
    for (std::size_t index = 0; index < count; ++index) {
        double a[3];
        double b[3];
        for (int axis = 0; axis < 3; ++axis) {
            a[axis] = point(i1)[axis] - point(i0)[axis];
            b[axis] = point(index)[axis] - point(i0)[axis];
        }
        cross(a, b, edge);
        const double area = dot(edge, edge);
        if (area > best) {
            best = area;
            i2 = index;
        }
    }
    if (best <= 1e-24) {
        return;
    }
    std::size_t i3 = 0;
    best = -1.0;
    double plane[3];
    {
        double a[3];
        double b[3];
        for (int axis = 0; axis < 3; ++axis) {
            a[axis] = point(i1)[axis] - point(i0)[axis];
            b[axis] = point(i2)[axis] - point(i0)[axis];
        }
        cross(a, b, plane);
    }
    for (std::size_t index = 0; index < count; ++index) {
        double delta[3];
        for (int axis = 0; axis < 3; ++axis) {
            delta[axis] = point(index)[axis] - point(i0)[axis];
        }
        const double volume = std::fabs(dot(plane, delta));
        if (volume > best) {
            best = volume;
            i3 = index;
        }
    }
    if (best <= 1e-24) {
        return;
    }

    // Centroid of the tetrahedron is strictly inside the hull, so a face normal is
    // outward when it points away from it.
    double centroid[3] = {0.0, 0.0, 0.0};
    const std::size_t seed[4] = {i0, i1, i2, i3};
    for (std::size_t index = 0; index < 4; ++index) {
        for (int axis = 0; axis < 3; ++axis) {
            centroid[axis] += point(seed[index])[axis] * 0.25;
        }
    }
    std::vector<HullFace> faces;
    faces.reserve(count * 2);
    const int seed_index[4] = {static_cast<int>(i0), static_cast<int>(i1),
                               static_cast<int>(i2), static_cast<int>(i3)};
    const int initial[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    for (const int* triple : initial) {
        HullFace face;
        for (int corner = 0; corner < 3; ++corner) {
            face.vertex[corner] = seed_index[triple[corner]];
        }
        double a[3];
        double b[3];
        for (int axis = 0; axis < 3; ++axis) {
            a[axis] = point(face.vertex[1])[axis] - point(face.vertex[0])[axis];
            b[axis] = point(face.vertex[2])[axis] - point(face.vertex[0])[axis];
        }
        cross(a, b, face.normal);
        double to_center[3];
        for (int axis = 0; axis < 3; ++axis) {
            to_center[axis] = centroid[axis] - point(face.vertex[0])[axis];
        }
        if (dot(face.normal, to_center) > 0.0) {
            std::swap(face.vertex[1], face.vertex[2]);
            for (int axis = 0; axis < 3; ++axis) {
                face.normal[axis] = -face.normal[axis];
            }
        }
        faces.push_back(face);
    }

    struct Edge {
        int a;
        int b;
        int count;
    };
    std::vector<Edge> boundary;
    std::vector<std::size_t> visible;
    for (std::size_t index = 0; index < count; ++index) {
        const double* p = point(index);
        bool seed_point = false;
        for (int corner = 0; corner < 4; ++corner) {
            if (seed_index[corner] == static_cast<int>(index)) {
                seed_point = true;
                break;
            }
        }
        if (seed_point) {
            continue;
        }
        visible.clear();
        for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
            HullFace& face = faces[face_index];
            if (!face.alive) {
                continue;
            }
            double delta[3];
            for (int axis = 0; axis < 3; ++axis) {
                delta[axis] = p[axis] - point(face.vertex[0])[axis];
            }
            if (dot(face.normal, delta) > 1e-12) {
                visible.push_back(face_index);
            }
        }
        if (visible.empty()) {
            continue;
        }
        boundary.clear();
        const auto add_edge = [&boundary](int a, int b) {
            for (Edge& edge : boundary) {
                if ((edge.a == a && edge.b == b) || (edge.a == b && edge.b == a)) {
                    ++edge.count;
                    return;
                }
            }
            boundary.push_back(Edge{a, b, 1});
        };
        for (const std::size_t face_index : visible) {
            const HullFace& face = faces[face_index];
            add_edge(face.vertex[0], face.vertex[1]);
            add_edge(face.vertex[1], face.vertex[2]);
            add_edge(face.vertex[2], face.vertex[0]);
        }
        for (const std::size_t face_index : visible) {
            faces[face_index].alive = false;
        }
        for (const Edge& edge : boundary) {
            if (edge.count != 1) {
                continue;
            }
            HullFace face;
            face.vertex[0] = edge.a;
            face.vertex[1] = edge.b;
            face.vertex[2] = static_cast<int>(index);
            double a[3];
            double b[3];
            for (int axis = 0; axis < 3; ++axis) {
                a[axis] = point(face.vertex[1])[axis] - point(face.vertex[0])[axis];
                b[axis] = point(face.vertex[2])[axis] - point(face.vertex[0])[axis];
            }
            cross(a, b, face.normal);
            double inside[3];
            for (int axis = 0; axis < 3; ++axis) {
                inside[axis] = centroid[axis] - point(face.vertex[0])[axis];
            }
            if (dot(face.normal, inside) > 0.0) {
                std::swap(face.vertex[1], face.vertex[2]);
                for (int axis = 0; axis < 3; ++axis) {
                    face.normal[axis] = -face.normal[axis];
                }
            }
            faces.push_back(face);
        }
    }

    faces_.clear();
    max_triangle_edge_degrees_ = 0.0;
    for (const HullFace& face : faces) {
        if (!face.alive) {
            continue;
        }
        faces_.push_back(face.vertex[0]);
        faces_.push_back(face.vertex[1]);
        faces_.push_back(face.vertex[2]);
        const double* a = point(face.vertex[0]);
        const double* b = point(face.vertex[1]);
        const double* c = point(face.vertex[2]);
        for (int first = 0; first < 3; ++first) {
            for (int second = first + 1; second < 3; ++second) {
                const double* u = point(face.vertex[first]);
                const double* v = point(face.vertex[second]);
                const double edge_dot = std::clamp(dot(u, v), -1.0, 1.0);
                const double degrees = std::acos(edge_dot) * 180.0 / kPi;
                max_triangle_edge_degrees_ = std::max(max_triangle_edge_degrees_, degrees);
            }
        }
    }
}

void SphericalMesh::compute_face_inverses() {
    inverse_.assign(face_count() * 9, 0.0);
    std::vector<int> kept;
    for (std::size_t face_index = 0; face_index < face_count(); ++face_index) {
        const int* ids = faces_.data() + face_index * 3;
        double matrix[9];
        // Columns are the vertices: the barycentric weights of q are the solution
        // of matrix * w = q, so the inverse is applied to the query afterwards.
        for (int corner = 0; corner < 3; ++corner) {
            const double* vertex = directions_.data() + ids[corner] * 3;
            for (int axis = 0; axis < 3; ++axis) {
                matrix[axis * 3 + corner] = vertex[axis];
            }
        }
        if (!invert3(matrix, inverse_.data() + face_index * 9)) {
            continue;
        }
        kept.push_back(static_cast<int>(face_index));
    }
    if (kept.size() != face_count()) {
        // Drop singular faces: they cannot contain a query direction strictly.
        std::vector<int> compact;
        std::vector<double> compact_inverse;
        for (const int face_index : kept) {
            for (int corner = 0; corner < 3; ++corner) {
                compact.push_back(faces_[face_index * 3 + corner]);
            }
            for (int value = 0; value < 9; ++value) {
                compact_inverse.push_back(inverse_[face_index * 9 + value]);
            }
        }
        faces_ = compact;
        inverse_ = compact_inverse;
    }
}

std::size_t SphericalMesh::nearest(const double direction[3], double* distance) const {
    const std::size_t count = vertex_count();
    std::size_t best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < count; ++index) {
        const double* vertex = directions_.data() + index * 3;
        const double dx = vertex[0] - direction[0];
        const double dy = vertex[1] - direction[1];
        const double dz = vertex[2] - direction[2];
        const double squared = dx * dx + dy * dy + dz * dz;
        if (squared < best_distance) {
            best_distance = squared;
            best = index;
        }
    }
    if (distance != nullptr) {
        *distance = std::sqrt(best_distance);
    }
    return best;
}

Status SphericalMesh::weights(const double direction[3], std::vector<int>* ids,
                              std::vector<double>* weights) const {
    if (ids == nullptr || weights == nullptr) {
        return mesh_error("weight output is null");
    }
    double query[3] = {direction[0], direction[1], direction[2]};
    const double length = std::sqrt(dot(query, query));
    if (!std::isfinite(length) || length <= 1e-12) {
        return mesh_error("query direction is degenerate");
    }
    normalize(query);
    double distance = 0.0;
    const std::size_t nearest_index = nearest(query, &distance);
    ids->clear();
    weights->clear();
    if (distance < 1e-10) {
        ids->push_back(static_cast<int>(nearest_index));
        weights->push_back(1.0);
        return Status::success();
    }
    for (std::size_t face_index = 0; face_index < face_count(); ++face_index) {
        const double* inverse = inverse_.data() + face_index * 9;
        const double barycentric[3] = {
            inverse[0] * query[0] + inverse[1] * query[1] + inverse[2] * query[2],
            inverse[3] * query[0] + inverse[4] * query[1] + inverse[5] * query[2],
            inverse[6] * query[0] + inverse[7] * query[1] + inverse[8] * query[2]};
        if (std::min({barycentric[0], barycentric[1], barycentric[2]}) < -1e-10) {
            continue;
        }
        double total = 0.0;
        double clamped[3];
        for (int corner = 0; corner < 3; ++corner) {
            clamped[corner] = std::max(barycentric[corner], 0.0);
            total += clamped[corner];
        }
        if (total <= 0.0) {
            continue;
        }
        for (int corner = 0; corner < 3; ++corner) {
            ids->push_back(faces_[face_index * 3 + corner]);
            weights->push_back(clamped[corner] / total);
        }
        return Status::success();
    }
    // The query fell outside every triangle (only possible on numerical margins).
    ids->push_back(static_cast<int>(nearest_index));
    weights->push_back(1.0);
    return Status::success();
}

void SphericalMesh::voronoi_areas(std::vector<double>* areas) const {
    const std::size_t vertices = vertex_count();
    const std::size_t face_total = face_count();
    areas->assign(vertices, 0.0);
    if (vertices == 0 || face_total == 0) {
        return;
    }
    // Circumcenter of a triangle of unit vectors is its normalized face normal.
    std::vector<double> circumcenters(face_total * 3, 0.0);
    for (std::size_t face_index = 0; face_index < face_total; ++face_index) {
        const int* ids = faces_.data() + face_index * 3;
        const double* a = directions_.data() + ids[0] * 3;
        const double* b = directions_.data() + ids[1] * 3;
        const double* c = directions_.data() + ids[2] * 3;
        double normal[3];
        double edge_one[3];
        double edge_two[3];
        for (int axis = 0; axis < 3; ++axis) {
            edge_one[axis] = b[axis] - a[axis];
            edge_two[axis] = c[axis] - a[axis];
        }
        cross(edge_one, edge_two, normal);
        normalize(normal);
        std::copy(normal, normal + 3,
                  circumcenters.begin() + static_cast<std::ptrdiff_t>(face_index * 3));
    }
    std::vector<std::vector<std::size_t>> incident(vertices);
    for (std::size_t face_index = 0; face_index < face_total; ++face_index) {
        for (int corner = 0; corner < 3; ++corner) {
            incident[static_cast<std::size_t>(faces_[face_index * 3 + corner])]
                .push_back(face_index);
        }
    }
    std::vector<std::pair<double, std::size_t>> ordered;
    double total = 0.0;
    for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
        const double* centre = directions_.data() + vertex * 3;
        if (incident[vertex].size() < 3) {
            continue;
        }
        // Tangent basis at the vertex.
        double tangent[3] = {0.0, 0.0, 0.0};
        if (std::fabs(centre[0]) < 0.9) {
            tangent[0] = 1.0;
        } else {
            tangent[1] = 1.0;
        }
        const double projection = dot(tangent, centre);
        double axis_one[3];
        double axis_two[3];
        for (int axis = 0; axis < 3; ++axis) {
            axis_one[axis] = tangent[axis] - projection * centre[axis];
        }
        normalize(axis_one);
        cross(centre, axis_one, axis_two);
        ordered.clear();
        for (const std::size_t face_index : incident[vertex]) {
            const double* point = circumcenters.data() + face_index * 3;
            const double angle = std::atan2(dot(point, axis_two), dot(point, axis_one));
            ordered.emplace_back(angle, face_index);
        }
        std::sort(ordered.begin(), ordered.end());
        double area = 0.0;
        for (std::size_t index = 0; index < ordered.size(); ++index) {
            const double* first = circumcenters.data() + ordered[index].second * 3;
            const double* second =
                circumcenters.data() + ordered[(index + 1) % ordered.size()].second * 3;
            area += triangle_area(centre, first, second);
        }
        (*areas)[vertex] = area;
        total += area;
    }
    if (total > 0.0) {
        for (double& value : *areas) {
            value /= total;
        }
    }
}

}  // namespace joc::hrtf
