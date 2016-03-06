#pragma once

//////////////////////////////////////////////////////
// MatUtils contains abusive math operations between
// different types
//////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include "/usr/local/cuda/samples/common/inc/helper_math.h"
#include "cumath.h"

//namespace roo {

//////////////////////////////////////////////////////
// Cross type operations
//////////////////////////////////////////////////////

inline __host__ __device__ float3 operator*(float b, uchar3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float3 operator*(uchar3 a, float b)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float1 operator*(float b, uchar1 a)
{
    return make_float1(b * a.x);
}

inline __host__ __device__ float1 operator*(uchar1 a, float b)
{
    return make_float1(b * a.x);
}

//////////////////////////////////////////////////////
// Construct Mat types
//////////////////////////////////////////////////////

__host__ __device__ inline
Mat<float,3> make_mat(float x, float y, float z)
{
    Mat<float,3> m;
    m(0) = x; m(1) = y; m(2) = z;
    return m;
}

__host__ __device__ inline
Mat<float,4> make_mat(float x, float y, float z, float w)
{
    Mat<float,4> m;
    m(0) = x; m(1) = y; m(2) = z; m(3) = w;
    return m;
}

//////////////////////////////////////////////////////
// Mat homegeneous multiplication with and Mat
//////////////////////////////////////////////////////

__host__ __device__ inline
Mat<float,3> operator*(const Mat<float,3,4>& T_ba, const Mat<float,3>& p_a)
{
    Mat<float,3> m;
    m(0)= T_ba(0,0) * p_a(0) + T_ba(0,1) * p_a(1) + T_ba(0,2) * p_a(2) + T_ba(0,3);
    m(1)= T_ba(1,0) * p_a(0) + T_ba(1,1) * p_a(1) + T_ba(1,2) * p_a(2) + T_ba(1,3);
    m(2)= T_ba(2,0) * p_a(0) + T_ba(2,1) * p_a(1) + T_ba(2,2) * p_a(2) + T_ba(2,3);
    return m;
}

__host__ __device__ inline
Mat<float,3> operator*(const Mat<float,3,4>& T_ba, const Mat<float,4>& p_a)
{
    Mat<float,3> m;
    m(0)= T_ba(0,0) * p_a(0) + T_ba(0,1) * p_a(1) + T_ba(0,2) * p_a(2) + T_ba(0,3);
    m(1)= T_ba(1,0) * p_a(0) + T_ba(1,1) * p_a(1) + T_ba(1,2) * p_a(2) + T_ba(1,3);
    m(2)= T_ba(2,0) * p_a(0) + T_ba(2,1) * p_a(1) + T_ba(2,2) * p_a(2) + T_ba(2,3);
    return m;
}

__host__ __device__ inline
Mat<float,3,4> operator*(const Mat<float,3,4>& T_cb, const Mat<float,3,4>& T_ba)
{
    Mat<float,3,4> m;
    /*
    m(0,0)= T_cb(0,0) * T_ba(0,0) + T_cb(0,1) * T_ba(1,0) + T_cb(0,2) * T_ba(2,0) + T_cb(0,3);
    m(1,0)= T_cb(1,0) * T_ba(0,0) + T_cb(1,1) * T_ba(1,0) + T_cb(1,2) * T_ba(2,0) + T_cb(1,3);
    m(2,0)= T_cb(2,0) * T_ba(0,0) + T_cb(2,1) * T_ba(1,0) + T_cb(2,2) * T_ba(2,0) + T_cb(2,3);
    m(0,1)= T_cb(0,0) * T_ba(0,1) + T_cb(0,1) * T_ba(1,1) + T_cb(0,2) * T_ba(2,1) + T_cb(0,3);
    m(1,1)= T_cb(1,0) * T_ba(0,1) + T_cb(1,1) * T_ba(1,1) + T_cb(1,2) * T_ba(2,1) + T_cb(1,3);
    m(2,1)= T_cb(2,0) * T_ba(0,1) + T_cb(2,1) * T_ba(1,1) + T_cb(2,2) * T_ba(2,1) + T_cb(2,3);
    m(0,2)= T_cb(0,0) * T_ba(0,2) + T_cb(0,1) * T_ba(1,2) + T_cb(0,2) * T_ba(2,2) + T_cb(0,3);
    m(1,2)= T_cb(1,0) * T_ba(0,2) + T_cb(1,1) * T_ba(1,2) + T_cb(1,2) * T_ba(2,2) + T_cb(1,3);
    m(2,2)= T_cb(2,0) * T_ba(0,2) + T_cb(2,1) * T_ba(1,2) + T_cb(2,2) * T_ba(2,2) + T_cb(2,3);
    */
    m(0,0)= T_cb(0,0) * T_ba(0,0) + T_cb(0,1) * T_ba(1,0) + T_cb(0,2) * T_ba(2,0);
    m(1,0)= T_cb(1,0) * T_ba(0,0) + T_cb(1,1) * T_ba(1,0) + T_cb(1,2) * T_ba(2,0);
    m(2,0)= T_cb(2,0) * T_ba(0,0) + T_cb(2,1) * T_ba(1,0) + T_cb(2,2) * T_ba(2,0);
    m(0,1)= T_cb(0,0) * T_ba(0,1) + T_cb(0,1) * T_ba(1,1) + T_cb(0,2) * T_ba(2,1);
    m(1,1)= T_cb(1,0) * T_ba(0,1) + T_cb(1,1) * T_ba(1,1) + T_cb(1,2) * T_ba(2,1);
    m(2,1)= T_cb(2,0) * T_ba(0,1) + T_cb(2,1) * T_ba(1,1) + T_cb(2,2) * T_ba(2,1);
    m(0,2)= T_cb(0,0) * T_ba(0,2) + T_cb(0,1) * T_ba(1,2) + T_cb(0,2) * T_ba(2,2);
    m(1,2)= T_cb(1,0) * T_ba(0,2) + T_cb(1,1) * T_ba(1,2) + T_cb(1,2) * T_ba(2,2);
    m(2,2)= T_cb(2,0) * T_ba(0,2) + T_cb(2,1) * T_ba(1,2) + T_cb(2,2) * T_ba(2,2);
    m(0,3)= T_cb(0,3) + T_cb(0,0) * T_ba(0,3) + T_cb(0,1) * T_ba(1,3) + T_cb(0,2) * T_ba(2,3);
    m(1,3)= T_cb(1,3) + T_cb(1,0) * T_ba(0,3) + T_cb(1,1) * T_ba(1,3) + T_cb(1,2) * T_ba(2,3);
    m(2,3)= T_cb(2,3) + T_cb(2,0) * T_ba(0,3) + T_cb(2,1) * T_ba(1,3) + T_cb(2,2) * T_ba(2,3);
    return m;
}


//////////////////////////////////////////////////////
// Mat homegeneous multiplication with Mat and float
//////////////////////////////////////////////////////

__host__ __device__ inline
float3 operator*(const Mat<float,3,4>& T_ba, const float3& p_a)
{
    return make_float3(
            T_ba(0,0) * p_a.x + T_ba(0,1) * p_a.y + T_ba(0,2) * p_a.z + T_ba(0,3),
            T_ba(1,0) * p_a.x + T_ba(1,1) * p_a.y + T_ba(1,2) * p_a.z + T_ba(1,3),
            T_ba(2,0) * p_a.x + T_ba(2,1) * p_a.y + T_ba(2,2) * p_a.z + T_ba(2,3)
    );
}

//__host__ __device__ inline
//float3 operator*(const Mat<float,3,4>& T_ba, const float4& p_a)
//{
//    return make_float3(
//            T_ba(0,0) * p_a.x + T_ba(0,1) * p_a.y + T_ba(0,2) * p_a.z + T_ba(0,3),
//            T_ba(1,0) * p_a.x + T_ba(1,1) * p_a.y + T_ba(1,2) * p_a.z + T_ba(1,3),
//            T_ba(2,0) * p_a.x + T_ba(2,1) * p_a.y + T_ba(2,2) * p_a.z + T_ba(2,3)
//    );
//}


/// Changed by Ankur ---
__host__ __device__ inline
float3 operator*(const Mat<float,3,4>& T_ba, const float4& p_a)
{
    return make_float3(
            T_ba(0,0) * p_a.x + T_ba(0,1) * p_a.y + T_ba(0,2) * p_a.z + T_ba(0,3)*p_a.w,
            T_ba(1,0) * p_a.x + T_ba(1,1) * p_a.y + T_ba(1,2) * p_a.z + T_ba(1,3)*p_a.w,
            T_ba(2,0) * p_a.x + T_ba(2,1) * p_a.y + T_ba(2,2) * p_a.z + T_ba(2,3)*p_a.w
    );
}

__host__ __device__ inline
float3 mulSO3(const Mat<float,3,3>& R_ba, const float3& r_a)
{
    return make_float3(
            R_ba(0,0) * r_a.x + R_ba(0,1) * r_a.y + R_ba(0,2) * r_a.z,
            R_ba(1,0) * r_a.x + R_ba(1,1) * r_a.y + R_ba(1,2) * r_a.z,
            R_ba(2,0) * r_a.x + R_ba(2,1) * r_a.y + R_ba(2,2) * r_a.z
    );
}

__host__ __device__ inline
float3 mulSO3(const Mat<float,3,4>& T_ab, const float3& r_a)
{
    return make_float3(
            T_ab(0,0) * r_a.x + T_ab(0,1) * r_a.y + T_ab(0,2) * r_a.z,
            T_ab(1,0) * r_a.x + T_ab(1,1) * r_a.y + T_ab(1,2) * r_a.z,
            T_ab(2,0) * r_a.x + T_ab(2,1) * r_a.y + T_ab(2,2) * r_a.z
    );
}

__host__ __device__ inline
float3 mulSO3(const Mat<float,3,4>& T_ab, const float4& r_a)
{
    return make_float3(
            T_ab(0,0) * r_a.x + T_ab(0,1) * r_a.y + T_ab(0,2) * r_a.z,
            T_ab(1,0) * r_a.x + T_ab(1,1) * r_a.y + T_ab(1,2) * r_a.z,
            T_ab(2,0) * r_a.x + T_ab(2,1) * r_a.y + T_ab(2,2) * r_a.z
    );
}

__host__ __device__ inline
float3 mulSO3inv(const Mat<float,3,3>& R_ab, const float3& r_a)
{
    return make_float3(
            R_ab(0,0) * r_a.x + R_ab(1,0) * r_a.y + R_ab(2,0) * r_a.z,
            R_ab(0,1) * r_a.x + R_ab(1,1) * r_a.y + R_ab(2,1) * r_a.z,
            R_ab(0,2) * r_a.x + R_ab(1,2) * r_a.y + R_ab(2,2) * r_a.z
    );
}

__host__ __device__ inline
float3 mulSO3inv(const Mat<float,3,4>& T_ba, const float3& r_a)
{
    return make_float3(
            T_ba(0,0) * r_a.x + T_ba(1,0) * r_a.y + T_ba(2,0) * r_a.z,
            T_ba(0,1) * r_a.x + T_ba(1,1) * r_a.y + T_ba(2,1) * r_a.z,
            T_ba(0,2) * r_a.x + T_ba(1,2) * r_a.y + T_ba(2,2) * r_a.z
    );
}

__host__ __device__ inline
float3 mulSE3(const Mat<float,3,4>& T_ab, const float3& r_a)
{
    return make_float3(
            T_ab(0,0) * r_a.x + T_ab(0,1) * r_a.y + T_ab(0,2) * r_a.z + T_ab(0,3),
            T_ab(1,0) * r_a.x + T_ab(1,1) * r_a.y + T_ab(1,2) * r_a.z + T_ab(1,3),
            T_ab(2,0) * r_a.x + T_ab(2,1) * r_a.y + T_ab(2,2) * r_a.z + T_ab(2,3)
    );
}

__host__ __device__ inline
float3 mulSE3inv(const Mat<float,3,4>& T_ab, const float3& r_a)
{
    return make_float3(
            T_ab(0,0) * (r_a.x-T_ab(0,3)) + T_ab(1,0) * (r_a.y-T_ab(1,3)) + T_ab(2,0) * (r_a.z-T_ab(2,3)),
            T_ab(0,1) * (r_a.x-T_ab(0,3)) + T_ab(1,1) * (r_a.y-T_ab(1,3)) + T_ab(2,1) * (r_a.z-T_ab(2,3)),
            T_ab(0,2) * (r_a.x-T_ab(0,3)) + T_ab(1,2) * (r_a.y-T_ab(1,3)) + T_ab(2,2) * (r_a.z-T_ab(2,3))
    );
}

__device__ __host__ inline
Mat<float,3,4> SE3inv(const Mat<float,3,4>& T_ba)
{
    Mat<float,3,4> T_ab;
    T_ab(0,0) = T_ba(0,0); T_ab(0,1) = T_ba(1,0); T_ab(0,2) = T_ba(2,0);
    T_ab(1,0) = T_ba(0,1); T_ab(1,1) = T_ba(1,1); T_ab(1,2) = T_ba(2,1);
    T_ab(2,0) = T_ba(0,2); T_ab(2,1) = T_ba(1,2); T_ab(2,2) = T_ba(2,2);
    T_ab(0,3) = - (T_ab(0,0)* T_ba(0,3) + T_ab(0,1)* T_ba(1,3) + T_ab(0,2)* T_ba(2,3) );
    T_ab(1,3) = - (T_ab(1,0)* T_ba(0,3) + T_ab(1,1)* T_ba(1,3) + T_ab(1,2)* T_ba(2,3) );
    T_ab(2,3) = - (T_ab(2,0)* T_ba(0,3) + T_ab(2,1)* T_ba(1,3) + T_ab(2,2)* T_ba(2,3) );
    return T_ab;
}

__host__ __device__ inline
float3 SE3Translation(const Mat<float,3,4>& T_ba)
{
    return make_float3( T_ba(0,3), T_ba(1,3), T_ba(2,3) );
}

//////////////////////////////////////////////////////
// Mat homegeneous multiplication with and floatx, convert to mat
//////////////////////////////////////////////////////

__host__ __device__ inline
Mat<float,3> mulSE3Mat(const Mat<float,3,4>& T_ba, const float3& p_a)
{
    Mat<float,3> m;
    m(0)= T_ba(0,0) * p_a.x + T_ba(0,1) * p_a.y + T_ba(0,2) * p_a.z + T_ba(0,3);
    m(1)= T_ba(1,0) * p_a.x + T_ba(1,1) * p_a.y + T_ba(1,2) * p_a.z + T_ba(1,3);
    m(2)= T_ba(2,0) * p_a.x + T_ba(2,1) * p_a.y + T_ba(2,2) * p_a.z + T_ba(2,3);
    return m;
}

__host__ __device__ inline
Mat<float,3> mulSE3Mat(const Mat<float,3,4>& T_ba, const float4& p_a)
{
    Mat<float,3> m;
    m(0)= T_ba(0,0) * p_a.x + T_ba(0,1) * p_a.y + T_ba(0,2) * p_a.z + T_ba(0,3);
    m(1)= T_ba(1,0) * p_a.x + T_ba(1,1) * p_a.y + T_ba(1,2) * p_a.z + T_ba(1,3);
    m(2)= T_ba(2,0) * p_a.x + T_ba(2,1) * p_a.y + T_ba(2,2) * p_a.z + T_ba(2,3);
    return m;
}

//////////////////////////////////////////////////////
// Mat, float3/float4 subtraction
//////////////////////////////////////////////////////

__host__ __device__ inline
Mat<float,3> operator-(const Mat<float,3>& lhs, const float3& rhs)
{
    Mat<float,3> m;
    m(0) = lhs(0) - rhs.x;
    m(1) = lhs(1) - rhs.y;
    m(2) = lhs(2) - rhs.z;
    return m;
}

__host__ __device__ inline
Mat<float,3> operator-(const Mat<float,3>& lhs, const float4& rhs)
{
    Mat<float,3> m;
    m(0) = lhs(0) - rhs.x;
    m(1) = lhs(1) - rhs.y;
    m(2) = lhs(2) - rhs.z;
    return m;
}

//////////////////////////////////////////////////////
// Operations between cuda vector types with
// incompatible dimensions
//////////////////////////////////////////////////////

__host__ __device__ inline
float3 operator-(const float3& lhs, const float4& rhs)
{
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__host__ __device__ inline
float3 operator-(const float4& lhs, const float3& rhs)
{
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

//////////////////////////////////////////////////////

__host__ __device__ inline
float dot(const float3& lhs, const float4& rhs)
{
    return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}

__host__ __device__ inline
float dot(const float4& lhs, const float3& rhs)
{
    return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}

__host__ __device__ inline
float dot3(const float4& lhs, const float4& rhs)
{
    return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}

__host__ __device__ inline
float length3(float4 r)
{
    return sqrtf(dot3(r, r));
}

//////////////////////////////////////////////////////
// SE3 Generator sparse multiplied by Mat
// gen_i * p
//////////////////////////////////////////////////////

__host__ __device__ inline
Mat<float,3> SE3gen0mul(const Mat<float,3>& /*p*/) {
    return make_mat(1,0,0);
}
__host__ __device__ inline
Mat<float,3> SE3gen1mul(const Mat<float,3>& /*p*/) {
    return make_mat(0,1,0);
}
__host__ __device__ inline
Mat<float,3> SE3gen2mul(const Mat<float,3>& /*p*/) {
    return make_mat(0,0,1);
}
__host__ __device__ inline
Mat<float,3> SE3gen3mul(const Mat<float,3>& p) {
    return make_mat(0,-p(2),p(1));
}
__host__ __device__ inline
Mat<float,3> SE3gen4mul(const Mat<float,3>& p) {
    return make_mat(p(2),0,-p(0));
}
__host__ __device__ inline
Mat<float,3> SE3gen5mul(const Mat<float,3>& p) {
    return make_mat(-p(1),p(0),0);
}

//////////////////////////////////////////////////////
// SE3 Generator sparse multiplied by cuda vector
// gen_i * p
//////////////////////////////////////////////////////

__host__ __device__ inline
float3 SE3gen0mul(const float3& /*p*/) {
    return make_float3(1,0,0);
}
__host__ __device__ inline
float3 SE3gen1mul(const float3& /*p*/) {
    return make_float3(0,1,0);
}
__host__ __device__ inline
float3 SE3gen2mul(const float3& /*p*/) {
    return make_float3(0,0,1);
}
__host__ __device__ inline
float3 SE3gen3mul(const float3& p) {
    return make_float3(0,-p.z,p.y);
}
__host__ __device__ inline
float3 SE3gen4mul(const float3& p) {
    return make_float3(p.z,0,-p.x);
}
__host__ __device__ inline
float3 SE3gen5mul(const float3& p) {
    return make_float3(-p.y,p.x,0);
}

//////////////////////////////////////////////////////

__host__ __device__ inline
float3 SE3gen0mul(const float4& /*p*/) {
    return make_float3(1,0,0);
}
__host__ __device__ inline
float3 SE3gen1mul(const float4& /*p*/) {
    return make_float3(0,1,0);
}
__host__ __device__ inline
float3 SE3gen2mul(const float4& /*p*/) {
    return make_float3(0,0,1);
}
__host__ __device__ inline
float3 SE3gen3mul(const float4& p) {
    return make_float3(0,-p.z,p.y);
}
__host__ __device__ inline
float3 SE3gen4mul(const float4& p) {
    return make_float3(p.z,0,-p.x);
}
__host__ __device__ inline
float3 SE3gen5mul(const float4& p) {
    return make_float3(-p.y,p.x,0);
}

//////////////////////////////////////////////////////
// Mat of float3
//////////////////////////////////////////////////////

template<unsigned CR, unsigned C>
inline __device__ __host__
Mat<float3,1,C> operator*(const Mat<float3, 1,CR>& lhs, const Mat<float,CR,C>& rhs)
{
    Mat<float3,1,C> ret;

    for( unsigned c=0; c<C; ++c) {
        ret(0,c) = lhs(0,0) * rhs(0,c);
#pragma unroll
        for( unsigned k=1; k<CR; ++k)  {
            ret(0,c) += lhs(0,k) * rhs(k,c);
        }
    }
    return ret;
}

// Homogeneous multiplication 3x4 * 3x1
inline __device__ __host__
float3 operator*(const Mat<float3, 1,4>& T_ba, const float3& p_a)
{
    return make_float3(
            T_ba(0).x * p_a.x + T_ba(1).x * p_a.y + T_ba(2).x * p_a.z + T_ba(3).x,
            T_ba(0).y * p_a.x + T_ba(1).y * p_a.y + T_ba(2).y * p_a.z + T_ba(3).y,
            T_ba(0).z * p_a.x + T_ba(1).z * p_a.y + T_ba(2).z * p_a.z + T_ba(3).z
    );
}

//////////////////////////////////////////////////////
// Outer Product Mat of float3
//////////////////////////////////////////////////////

template<unsigned R, unsigned C>
inline __device__ __host__
SymMat<float,R*C> OuterProduct(const Mat<float3,R,C>& M, const float weight)
{
    const unsigned N = R*C;
    SymMat<float,N> ret;
    int i=0;
    for( int r=0; r<N; ++r )
#pragma unroll
        for( int c=0; c<=r; ++c ) {
            ret.m[i++] = weight * (
                        M(r).x * M(c).x +
                        M(r).y * M(c).y +
                        M(r).z * M(c).z
                    );
        }
    return ret;
}

template<unsigned R>
inline __device__ __host__ Mat<float,R,1> mul_aTb(const Mat<float3,1,R>& a, const float3 b)
{
    Mat<float,R,1> ret;

#pragma unroll
    for( unsigned r=0; r<R; ++r) {
        ret(r,0) = dot(a(0,r), b);
    }
    return ret;
}

//////////////////////////////////////////////////////
// Transform plane params
//////////////////////////////////////////////////////

inline __device__ __host__
float3 Plane_b_from_a(const Mat<float,3,4> T_ab, const float3 n_a)
{
    const float ba_dot_na_p1 =
            T_ab(0,3) * n_a.x +
            T_ab(1,3) * n_a.y +
            T_ab(2,3) * n_a.z + 1.0f;

    // n_b
    return make_float3(
        (T_ab(0,0) * n_a.x + T_ab(1,0) * n_a.y + T_ab(2,0) * n_a.z) / ba_dot_na_p1,
        (T_ab(0,1) * n_a.x + T_ab(1,1) * n_a.y + T_ab(2,1) * n_a.z) / ba_dot_na_p1,
        (T_ab(0,2) * n_a.x + T_ab(1,2) * n_a.y + T_ab(2,2) * n_a.z) / ba_dot_na_p1
    );
}

//////////////////////////////////////////////////////
// L1 norm
//////////////////////////////////////////////////////

inline __device__ __host__
float L1(float val)
{
    return abs(val);
}

inline __device__ __host__
float L1(float2 val)
{
    return abs(val.x) + abs(val.y);
}

inline __device__ __host__
float L1(float3 val)
{
    return abs(val.x) + abs(val.y) + abs(val.z);
}

inline __device__ __host__
float L1(float4 val)
{
    return abs(val.x) + abs(val.y) + abs(val.z) + abs(val.w);
}

#ifdef USE_EIGEN
inline __host__
Eigen::Vector3d ToEigen(const float3 v)
{
    return Eigen::Vector3d(v.x, v.y, v.z);
}

inline __host__
float3 ToCuda(const Eigen::Vector3d& v)
{
    return make_float3(v(0), v(1), v(2));
}

#endif // USE_EIGEN

//}

//////////////////////////////////////////////////////
// Cuda vector type Stream Overloads
//////////////////////////////////////////////////////

inline std::ostream& operator<<( std::ostream& os, const float3& v)
{
    os << v.x << " " << v.y << " " << v.z;
    return os;
}

inline std::istream& operator>>( std::istream& is, float3& v)
{
    is >> v.x;
    is >> v.y;
    is >> v.z;
    return is;
}
