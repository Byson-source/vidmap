#pragma once

#include <algorithm>
#include <Eigen/Core>
#include <ceres/ceres.h>

namespace vidmap {
// All coordinates and segment endpoints are metric. The fixed 2x3 map has
// gravity in its nullspace; no camera rotation or height measurement is added.
struct FloorplanWallError {
  Eigen::Matrix<double, 2, 3> projection;
  Eigen::Vector2d offset, start, end;
  double sigma;

  template <typename T>
  bool operator()(const T* point, T* residual) const {
    const Eigen::Matrix<T, 2, 1> q =
        projection.cast<T>() * Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point) + offset.cast<T>();
    const Eigen::Matrix<T, 2, 1> direction = (end - start).cast<T>();
    const T along = (q - start.cast<T>()).dot(direction) / direction.squaredNorm();
    const T t = std::min(T(1), std::max(T(0), along));
    Eigen::Map<Eigen::Matrix<T, 2, 1>> out(residual);
    out = (q - start.cast<T>() - t * direction) / T(sigma);
    return true;
  }
};

// Fix only the arbitrary height datum of one camera, allowing both horizontal
// coordinates to move. The two rows of the projection are orthogonal/equal norm.
class FloorplanHeightManifold final : public ceres::Manifold {
 public:
  explicit FloorplanHeightManifold(const Eigen::Matrix<double, 2, 3>& projection)
      : basis_(projection.transpose() / projection.row(0).norm()) {}
  int AmbientSize() const override { return 3; }
  int TangentSize() const override { return 2; }
  bool Plus(const double* x, const double* delta, double* out) const override {
    Eigen::Map<Eigen::Vector3d> result(out);
    result = Eigen::Map<const Eigen::Vector3d>(x) + basis_ * Eigen::Map<const Eigen::Vector2d>(delta);
    return true;
  }
  bool PlusJacobian(const double*, double* out) const override {
    Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> result(out);
    result = basis_;
    return true;
  }
  bool Minus(const double* y, const double* x, double* out) const override {
    Eigen::Map<Eigen::Vector2d> result(out);
    result = basis_.transpose() * (Eigen::Map<const Eigen::Vector3d>(y) - Eigen::Map<const Eigen::Vector3d>(x));
    return true;
  }
  bool MinusJacobian(const double*, double* out) const override {
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> result(out);
    result = basis_.transpose();
    return true;
  }
 private:
  Eigen::Matrix<double, 3, 2> basis_;
};
}  // namespace vidmap
