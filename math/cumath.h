
#pragma once

#include <vector_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>

#include "/usr/local/cuda/samples/common/inc/helper_math.h"


/// Steve Lovegrove's Kangaroo Mat.h


//namespace roo
//{

///////////////////////////////////////////
// Matrix Class
///////////////////////////////////////////

// R x C Real Matrix. Internally row-major.
template<typename P, unsigned R, unsigned C = 1>
struct Mat
{
    inline __device__ __host__ P operator()(int r, int c) const {
        return m[r*C + c];
    }

    inline __device__ __host__ P& operator()(int r, int c) {
        return m[r*C + c];
    }

    inline __device__ __host__ P operator()(int r) const {
        return m[r];
    }

    inline __device__ __host__ P& operator()(int r) {
        return m[r];
    }

    inline __device__ __host__ P operator[](int r) const {
        return m[r];
    }

    inline __device__ __host__ P& operator[](int r) {
        return m[r];
    }

    inline __device__ __host__ unsigned Rows() {
        return R;
    }

    inline __device__ __host__ unsigned Cols() {
        return C;
    }

    inline __device__ __host__ void operator+=(const Mat<P,R,C>& rhs) {
        #pragma unroll
        for( size_t i=0; i<R*C; ++i )
            m[i] += rhs.m[i];
    }

    inline __device__ __host__ void Fill(P val) {
        #pragma unroll
        for( size_t i=0; i<R*C; ++i )
            m[i] = val;
    }

    inline __device__ __host__ void SetZero() {
        Fill(0);
    }

    inline __device__ __host__ P Length() const {
        P sum = 0;
        for( size_t i=0; i<R*C; ++i )
            sum += m[i] * m[i];
        return sqrt(sum);
    }

    template<unsigned NR>
    inline __device__ __host__ Mat<P,NR,1> Head() {
        // TODO: static assert NR <= R;
        Mat<P,NR,1> ret;
        #pragma unroll
        for( size_t i=0; i<R; ++i )
            ret[i] = m[i];
        return ret;
    }

    template<unsigned NR, unsigned NC>
    inline __device__ __host__ Mat<P,NR,NC> Block(unsigned rs, unsigned cs) {
        // TODO: static assert NR <= R, NC <= C;
        Mat<P,NR,NC> ret;

        for( size_t r=0; r<NR; ++r )
            #pragma unroll
            for( size_t c=0; c<NC; ++c )
                ret[r*NC + c] = m[(rs+r)*NC + cs+c];
        return ret;
    }

#ifdef USE_EIGEN

    inline __host__ Mat() {
    }

    template<typename PF>
    inline __host__ Mat(const Eigen::Matrix<PF,R,C>& em) {
        for( size_t r=0; r<R; ++r )
            for( size_t c=0; c<C; ++c )
                m[r*C + c] = (P)em(r,c);
    }

    template<typename PT>
    inline __host__ operator Eigen::Matrix<PT,R,C>() const {
        Eigen::Matrix<PT,R,C> ret;
        for( size_t r=0; r<R; ++r )
            for( size_t c=0; c<C; ++c )
                ret(r,c) = (PT)m[r*C + c];
        return ret;
    }
#endif // USE_EIGEN

#ifdef USE_TOON
    inline __host__ operator TooN::Matrix<R,C,P>() const {
        TooN::Matrix<R,C,P> ret;
        for( size_t r=0; r<R; ++r )
            for( size_t c=0; c<C; ++c )
                ret(r,c) = m[r*C + c];
        return ret;
    }

    inline __host__ operator TooN::Vector<R*C,P>() const {
        TooN::Vector<R*C,P> ret;
        for( size_t i=0; i< R*C; ++i )
            ret[i] = m[i];
        return ret;
    }
#endif // USE_TOON

    P m[R*C];
};

///////////////////////////////////////////
// Construct from Zero / Identity
///////////////////////////////////////////

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ Mat<P,R,C> MatZero()
{
    Mat<P,R,C> ret;
#pragma unroll
    for( size_t i=0; i<R*C; ++i )
        ret(i) = 0.0f;
    return ret;
}

template<typename P, unsigned R>
inline __device__ __host__ Mat<P,R,R> MatId()
{
    Mat<P,R,R> ret = MatZero<P,R,R>();
#pragma unroll
    for( size_t i=0; i<R; ++i )
        ret(i,i) = 1.0f;
    return ret;
}

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ Mat<P,R,C> MatFill(P val)
{
    Mat<P,R,C> ret;
#pragma unroll
    for( size_t i=0; i<R*C; ++i )
        ret(i) = val;
    return ret;
}

///////////////////////////////////////////
// Matrix Matrix operations
///////////////////////////////////////////

template<typename P, unsigned R, unsigned CR, unsigned C>
inline __device__ __host__ Mat<P,R,C> operator*(const Mat<P,R,CR>& lhs, const Mat<P,CR,C>& rhs)
{
    Mat<P,R,C> ret;

    for( size_t r=0; r<R; ++r) {
        for( size_t c=0; c<C; ++c) {
            ret(r,c) = 0;
#pragma unroll
            for( size_t k=0; k<CR; ++k)  {
                ret(r,c) += lhs(r,k) * rhs(k,c);
            }
        }
    }
    return ret;
}

// Specialisation for scalar product
template<typename P, unsigned CR>
inline __device__ __host__ P operator*(const Mat<P,1,CR>& lhs, const Mat<P,CR,1>& rhs)
{
    return dot(lhs,rhs);
}

// Specialisation for scalar product
template<typename P, unsigned R>
inline __device__ __host__ P operator*(const Mat<P,R,1>& lhs, const Mat<P,R,1>& rhs)
{
    return dot(lhs,rhs);
}

// Dot Product
// TODO Check sizes are compatible
template<typename P, unsigned R1, unsigned C1, unsigned R2, unsigned C2>
inline __device__ __host__ P dot(const Mat<P,R1,C1>& lhs, const Mat<P,R2,C2>& rhs)
{
    P ret = lhs(0) * rhs(0);
    #pragma unroll
    for( size_t i=1; i<R1*C1; ++i)
        ret += lhs(i) * rhs(i);
    return ret;
}

template<typename P, unsigned R, unsigned CR, unsigned C>
inline __device__ __host__ Mat<P,R,C> mul_aTb(const Mat<P,CR,R>& a, const Mat<P,CR,C>& b)
{
    Mat<P,R,C> ret;

    for( size_t r=0; r<R; ++r) {
        for( size_t c=0; c<C; ++c) {
            ret(r,c) = 0;
            #pragma unroll
            for( size_t k=0; k<CR; ++k)  {
                ret(r,c) += a(k,r) * b(k,c);
            }
        }
    }
    return ret;
}

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ Mat<P,R,C> mul_aTb(const Mat<P,C,R>& a, const P b)
{
    Mat<P,R,C> ret;

    for( size_t r=0; r<R; ++r) {
#pragma unroll
        for( size_t c=0; c<C; ++c) {
            ret(r,c) = a(c,r) * b;
        }
    }
    return ret;
}

template<typename P, unsigned R, unsigned CR, unsigned C>
inline __device__ __host__ Mat<P,R,C> mul_abT(const Mat<P,R,CR>& a, const Mat<P,C,CR>& b)
{
    Mat<P,R,C> ret;

    for( size_t r=0; r<R; ++r) {
        for( size_t c=0; c<C; ++c) {
            ret(r,c) = 0;
            for( size_t k=0; k<CR; ++k)  {
                ret(r,c) += a(r,k) * b(c,k);
            }
        }
    }
    return ret;
}

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ Mat<P,R,C> operator+(const Mat<P,R,C>& lhs, const Mat<P,R,C>& rhs)
{
    Mat<P,R,C> ret;
#pragma unroll
    for( size_t i=0; i<R*C; ++i )
        ret(i) = lhs(i) + rhs(i);
    return ret;
}

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ Mat<P,R,C> operator-(const Mat<P,R,C>& lhs, const Mat<P,R,C>& rhs)
{
    Mat<P,R,C> ret;
#pragma unroll
    for( size_t i=0; i<R*C; ++i )
        ret(i) = lhs(i) - rhs(i);
    return ret;
}

///////////////////////////////////////////
// Matrix Scalar operations
///////////////////////////////////////////

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ Mat<P,R,C> operator*(const Mat<P,R,C>& lhs, const P rhs)
{
    Mat<P,R,C> ret;
#pragma unroll
    for( size_t i=0; i<R*C; ++i )
        ret(i) = lhs(i) * rhs;
    return ret;
}

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ Mat<P,R,C> operator*(const P lhs, const Mat<P,R,C>& rhs)
{
    Mat<P,R,C> ret;
#pragma unroll
    for( size_t i=0; i<R*C; ++i )
        ret(i) = lhs * rhs(i);
    return ret;
}

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ Mat<P,R,C> operator/(const Mat<P,R,C>& lhs, const P rhs)
{
    Mat<P,R,C> ret;
#pragma unroll
    for( size_t i=0; i<R*C; ++i )
        ret(i) = lhs(i) / rhs;
    return ret;
}

///////////////////////////////////////////
// Symmetric Matrix class & utilities
///////////////////////////////////////////

// N x N Real symetric matrix. Internally row major for
// each unique element
template<typename P, unsigned N>
struct SymMat
{
    static const int unique = N*(N+1)/2;

    template<typename PT>
    inline __device__ __host__ operator Mat<PT,N,N>()
    {
        Mat<PT,N,N> ret;
        size_t i = 0;
        for( size_t r=0; r<N; ++r ) {
            for( size_t c=0; c<=r; ++c ) {
                const PT elem = (PT)m[i++];
                ret(r,c) = elem;
                ret(c,r) = elem;
            }
        }
        return ret;
    }

    inline __device__ __host__ void SetZero() {
        #pragma unroll
        for( size_t i=0; i<unique; ++i )
            m[i] = 0;
    }

#ifdef USE_EIGEN
    template<typename PT>
    inline __host__ operator Eigen::Matrix<PT,N,N>()
    {
        Eigen::Matrix<PT,N,N> ret;
        size_t i = 0;
        for( size_t r=0; r<N; ++r ) {
            for( size_t c=0; c<=r; ++c ) {
                const PT elem = (PT)m[i++];
                ret(r,c) = elem;
                ret(c,r) = elem;
            }
        }
        return ret;
    }

#endif // USE_EIGEN

#ifdef USE_TOON
    template<typename PT>
    inline __host__ operator TooN::Matrix<N,N,PT>()
    {
        TooN::Matrix<N,N,PT> ret;
        size_t i = 0;
        for( size_t r=0; r<N; ++r ) {
            for( size_t c=0; c<=r; ++c ) {
                const PT elem = (PT)m[i++];
                ret(r,c) = elem;
                ret(c,r) = elem;
            }
        }
        return ret;
    }
#endif // USE_TOON

    inline __device__ __host__ void operator+=(const SymMat<P,N>& rhs)
    {
        #pragma unroll
        for( size_t i=0; i<unique; ++i )
            m[i] += rhs.m[i];
    }

    inline __device__ __host__ void operator*=(const P w)
    {
        #pragma unroll
        for( size_t i=0; i<unique; ++i )
            m[i] *= w;
    }

    P m[unique];
};

template<typename P, unsigned N>
inline __device__ __host__ SymMat<P,N> operator+(const SymMat<P,N>& lhs, const SymMat<P,N>& rhs)
{
    SymMat<P,N> ret;
#pragma unroll
    for( size_t i=0; i<SymMat<P,N>::unique; ++i )
        ret.m[i] = lhs.m[i] + rhs.m[i];
    return ret;
}

template<typename P, unsigned N>
inline __device__ __host__ SymMat<P,N> operator*(const SymMat<P,N>& lhs, const float rhs)
{
    SymMat<P,N> ret;
#pragma unroll
    for( size_t i=0; i<SymMat<P,N>::unique; ++i )
        ret.m[i] = lhs.m[i] * rhs;
    return ret;
}

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ SymMat<P,R*C> OuterProduct(const Mat<P,R,C>& M)
{
    const size_t N = R*C;
    SymMat<P,N> ret;
    size_t i=0;
    for( size_t r=0; r<N; ++r )
#pragma unroll
        for( size_t c=0; c<=r; ++c )
            ret.m[i++] = M(r) * M(c);
    return ret;
}

template<typename P, unsigned R, unsigned C>
inline __device__ __host__ SymMat<P,R*C> OuterProduct(const Mat<P,R,C>& M, const P weight)
{
    const unsigned N = R*C;
    SymMat<P,N> ret;
    size_t i=0;
    for( size_t r=0; r<N; ++r )
#pragma unroll
        for( size_t c=0; c<=r; ++c )
            ret.m[i++] = M(r) * M(c) * weight;
    return ret;
}

template<typename P, unsigned N>
inline __device__ __host__ SymMat<P,N> SymMat_zero()
{
    SymMat<P,N> ret;
#pragma unroll
    for( size_t i=0; i< SymMat<P,N>::unique; ++i )
        ret.m[i] = 0.0f;
    return ret;
}

///////////////////////////////////////////
// Least Squares Linear System
///////////////////////////////////////////

template<typename P, unsigned N>
struct LeastSquaresSystem
{
  Mat<P,N,1> JTy;
  SymMat<P,N> JTJ;
  P sqErr;
  unsigned obs;

  inline __device__ __host__ void SetZero() {
      JTJ.SetZero();
      JTy.SetZero();
      sqErr = 0;
      obs = 0;
  }

  inline __device__ __host__ void operator+=(const LeastSquaresSystem<P,N>& rhs)
  {
    JTy += rhs.JTy;
    JTJ += rhs.JTJ;
    sqErr += rhs.sqErr;
    obs += rhs.obs;
  }
};

template<typename P, unsigned N>
inline __device__ __host__ LeastSquaresSystem<P,N> operator+(const LeastSquaresSystem<P,N>& lhs, const LeastSquaresSystem<P,N>& rhs)
{
  return (LeastSquaresSystem<P,N>){lhs.JTy+rhs.JTy, lhs.JTJ+rhs.JTJ, lhs.sqErr + rhs.sqErr, lhs.obs + rhs.obs };
}

///////////////////////////////////////////
// Vector project / unproject
///////////////////////////////////////////

template<typename P>
inline __device__ __host__ Mat<P,3> up( const Mat<P,2>& x )
{
    return (Mat<P,3>){x(0),x(1),1};
}

template<typename P>
inline __device__ __host__ Mat<P,4> up( const Mat<P,3>& x )
{
    return (Mat<P,4>){x(0),x(1),x(2),1};
}

template<typename P>
inline __device__ __host__ Mat<P,2> dn( const Mat<P,3>& x )
{
    return (Mat<P,2>){x(0)/x(2), x(1)/x(2)};
}

template<typename P>
inline __device__ __host__ Mat<P,3> dn( const Mat<P,4>& x )
{
    return (Mat<P,3>){x(0)/x(3), x(1)/x(3), x(2)/x(3)};
}


///////////////////////////////////////////
// Primitive project / unproject
///////////////////////////////////////////

inline __device__ __host__ float3 up( const float2& x )
{
    return (float3){x.x, x.y, 1};
}

inline __device__ __host__ float4 up( const float3& x )
{
    return (float4){x.x, x.y, x.z, 1};
}

inline __device__ __host__ float2 dn( const float3& x )
{
    return (float2){x.x/x.z, x.y/x.z};
}

inline __device__ __host__ float3 dn( const float4& x )
{
    return (float3){x.x/x.w, x.y/x.w, x.z/x.w};
}

///////////////////////////////////////////
// IO
///////////////////////////////////////////

template<typename P, unsigned R, unsigned C>
inline __host__ std::ostream& operator<<( std::ostream& os, const Mat<P,R,C>& m )
{
    for( size_t r=0; r<R; ++r)
    {
        for( size_t c=0; c<C; ++c)
            std::cout << m(r,c) << " ";
        std::cout << std::endl;
    }
    return os;
}

template<typename P, unsigned N>
inline __host__ std::ostream& operator<<( std::ostream& os, const SymMat<P,N>& m )
{
    for(size_t i=0; i < SymMat<P,N>::unique; ++i ) {
        std::cout << m.m[i] << " ";
    }
    return os;
}

//}
