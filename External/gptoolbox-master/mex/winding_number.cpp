#ifdef MEX
#ifdef WITH_EMBREE
#  include "winding_number_ray.h"
#endif
#include "parse_rhs.h"
#include "prepare_lhs.h"

#include <igl/winding_number.h>
#include <igl/WindingNumberAABB.h>
#include <igl/matlab/MexStream.h>
#include <igl/matlab/mexErrMsgTxt.h>
#include <igl/writeOBJ.h>
#include <igl/get_seconds.h>
#include <igl/per_face_normals.h>
#ifdef WITH_EMBREE
#  include <igl/embree/EmbreeIntersector.h>
#endif
#include <igl/random_dir.h>

#include <mex.h>
#include <Eigen/Dense>
#include <iostream>

void mexFunction(
  int nlhs, mxArray *plhs[], 
  int nrhs, const mxArray *prhs[])
{
  using namespace std;
  using namespace Eigen;
  using namespace igl;

  igl::matlab::MexStream mout;        
  std::streambuf *outbuf = cout.rdbuf(&mout);
  //mexPrintf("Compiled at %s on %s\n",__TIME__,__DATE__);

  // vertex position list
  double * V;
  // face list
  double * Ftemp;
  // origin list
  double * O;
  // number of dimensions (and simplex size in F)
  int dim;
  // number of mesh vertices
  int n;
  // number of mesh faces
  int m;
  // number of origins
  int no;
  winding_number_params params;
  parse_rhs(nrhs,prhs,dim,V,n,Ftemp,m,O,no,params);

  double * F = new double[dim*m];
  // convert to 0-index
  //transform(Ftemp,Ftemp+m*dim,F,bind2nd(plus<double>(),-1.0));
  for(int i = 0;i<m;i++)
  {
    for(int d = 0;d<dim;d++)
    {
      F[d*m + i]=Ftemp[d*m+i]-1;
    }
  }

  //// Set up openmp
  //int nProcessors=omp_get_max_threads();
  ////std::cout<<nProcessors<<std::endl;
  //omp_set_num_threads(min(nProcessors,2));

  // List of winding numbers
  double * W = NULL;
  if(nlhs == 0)
  {
    return;
  }
  prepare_lhs(nlhs,plhs,no,W);

  double start_sec = igl::get_seconds();
  if(params.ray_cast)
  {
#ifdef WITH_EMBREE
    // Prepare eigen versions of input
    MatrixXd MV(n,dim);
    copy(V,V+n*dim,MV.data());
    MatrixXi MF(m,dim);
    copy(F,F+m*dim,MF.data());

    // Initialize Embree
    //cout<<"Flipping faces..."<<endl;
    MatrixXi FF;
    FF.resize(MF.rows()*2,MF.cols());
    FF << MF, MF.rowwise().reverse().eval();
    //cout<<"Computing normals..."<<endl;
    Eigen::MatrixXd N;
    per_face_normals(MV,MF,N);
    //cout<<"Initializing Embree..."<<endl;
    // Initialize intersector
    igl::embree::EmbreeIntersector ei;
    ei.init(MV.cast<float>(),FF);

    // loop over origins
#   pragma omp parallel for if (no>IGL_WINDING_NUMBER_OMP_MIN_VALUE)
    for(int o = 0;o<no;o++)
    {
      Vector3d p(O[0*no+o], O[1*no+o], O[2*no+o]);
      for(int r = 0;r<params.num_rays;r++)
      {
        int num_rays_shot = -1;
        Vector3d dir = random_dir();
        if(params.twod_rays)
        {
          dir[2] = 0;
          dir.normalize();
        }
        double w = 0;
        if(params.ray_parity)
        {
          winding_number_ray_parity(ei,N,p,dir,w,num_rays_shot);
        }else
        {
          winding_number_ray(ei,N,p,dir,w,num_rays_shot);
        }
        W[o] += w;
      }
      W[o] /= (double)params.num_rays;
    }


#else
    igl::matlab::mexErrMsgTxt(false,"Recompile with WITH_EMBREE defined");
#endif
  }else
  {
    if(params.hierarchical && dim == 3)
    {
      // Prepare eigen versions of input
      MatrixXd MV(n,dim);
      copy(V,V+n*dim,MV.data());
      MatrixXi MF(m,dim);
      copy(F,F+m*dim,MF.data());
      // Initialize hierarchy
      WindingNumberAABB<Vector3d> hier(MV,MF);
      // Build hierarchy
      hier.grow();
      // loop over origins
#     pragma omp parallel for if (no>IGL_WINDING_NUMBER_OMP_MIN_VALUE)
      for(int o = 0;o<no;o++)
      {
        Vector3d p(O[0*no+o], O[1*no+o], O[2*no+o]);
        W[o] = hier.winding_number(p);
      }
    }else
    {
      switch(dim)
      {
        case 3:
          winding_number_3(V,n,F,m,O,no,W);
          break;
        case 2:
          winding_number_2(V,n,F,m,O,no,W);
          break;
      }
    }
  }
  //mexPrintf("Elapsed time is %g seconds\n",igl::get_seconds()-start_sec);

  // Restore the std stream buffer Important!
  delete[] F;
  std::cout.rdbuf(outbuf);
}

#endif
