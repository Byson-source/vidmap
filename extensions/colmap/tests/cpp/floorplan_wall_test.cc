#include "stages/floorplan_wall.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

void Check(bool ok) { if (!ok) throw std::runtime_error("wall check failed"); }
int main() {
  using namespace vidmap;
  Eigen::Matrix<double,2,3> p; p << 1,0,0,0,0,1;
  FloorplanWallError f{p, Eigen::Vector2d::Zero(), {0,0}, {2,0}, .5};
  ceres::AutoDiffCostFunction<FloorplanWallError,2,3> cost(new FloorplanWallError(f));
  for (const Eigen::Vector3d point : {Eigen::Vector3d(1,7,.3), Eigen::Vector3d(3,7,.3), Eigen::Vector3d(-1,7,.3)}) {
    double r[2],jac[6]; const double* params[]={point.data()}; double* jacobians[]={jac};
    Check(cost.Evaluate(params,r,jacobians));
    for (int j=0;j<3;++j) {
      auto a=point,b=point;a[j]+=1e-6;b[j]-=1e-6;
      double ra[2],rb[2];f(a.data(),ra);f(b.data(),rb);
      for (int i=0;i<2;++i) Check(std::abs(jac[3*i+j]-(ra[i]-rb[i])/2e-6)<1e-7);
    }
    Check(jac[1]==0 && jac[4]==0);
  }
  double point[]={1,123,0},r[2]; f(point,r);Check(r[0]==0 && r[1]==0);
  FloorplanHeightManifold m(p);double delta[]={.2,.3},out[3];m.Plus(point,delta,out);
  Check(out[1]==point[1] && std::abs(out[0]-1.2)<1e-12 && std::abs(out[2]-.3)<1e-12);
  FloorplanPositionError anchor{p, {0,999,0}, .5};
  ceres::AutoDiffCostFunction<FloorplanPositionError,2,3> anchor_cost(new FloorplanPositionError(anchor));
  double ar[2], aj[6]; const double* ap[]={point}; double* ajs[]={aj};
  Check(anchor_cost.Evaluate(ap,ar,ajs));
  Check(ar[0]==2 && ar[1]==0 && aj[0]==2 && aj[5]==2 && aj[1]==0 && aj[4]==0);
  ceres::HuberLoss huber(2.);double rho[3];huber.Evaluate(400,rho);Check(std::abs(rho[1]-.1)<1e-12);
  std::cout << "wall segment, endpoints, Jacobian, height nullspace and Huber PASS\n";
}
