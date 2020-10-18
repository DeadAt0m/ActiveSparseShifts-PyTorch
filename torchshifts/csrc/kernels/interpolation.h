#include "../global_scope.h"

template<typename scalar_t>
API_INCLUDE scalar_t interp1D(scalar_t v1, scalar_t v2, scalar_t x)
{
    return v1*(1 - x) + v2*x;
}

template<typename scalar_t>
API_INCLUDE scalar_t interp1D_dx(scalar_t v1, scalar_t v2)
{
    return v2 - v1;
}

template<typename scalar_t>
API_INCLUDE scalar_t interp2D(scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4, scalar_t x, scalar_t y)
{
    return interp1D(interp1D(v1, v2, x), interp1D(v3, v4, x), y);
}

template<typename scalar_t>
API_INCLUDE scalar_t interp2D_dx(scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4, scalar_t y)
{
    return interp1D(interp1D_dx(v1, v3), interp1D_dx(v2, v4), y);
}

template<typename scalar_t>
API_INCLUDE scalar_t interp2D_dy(scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4, scalar_t x)
{
    return interp1D_dx(interp1D(v1, v2, x), interp1D(v3, v4, x));
}

template<typename scalar_t>
API_INCLUDE scalar_t interp3D(scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4,
                                             scalar_t v5, scalar_t v6, scalar_t v7, scalar_t v8,
                                             scalar_t x, scalar_t y, scalar_t z){
    return interp1D(interp2D(v1, v2, v3, v4, x, y), interp2D(v5, v6, v7, v8, x, y), z);
}

template<typename scalar_t>
API_INCLUDE scalar_t interp3D_dx(scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4,
                                                scalar_t v5, scalar_t v6, scalar_t v7, scalar_t v8,
                                                scalar_t y, scalar_t z)
{
    return interp1D(interp2D_dx(v1, v2, v3, v4, y), interp2D_dx(v5, v6, v7, v8, y), z);
}

template<typename scalar_t>
API_INCLUDE scalar_t interp3D_dy(scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4,
                                                scalar_t v5, scalar_t v6, scalar_t v7, scalar_t v8,
                                                scalar_t x, scalar_t z)
{
    return interp1D(interp2D_dy(v1, v2, v3, v4, x), interp2D_dy(v5, v6, v7, v8, x), z);
}

template<typename scalar_t>
API_INCLUDE scalar_t interp3D_dz(scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4,
                                                scalar_t v5, scalar_t v6, scalar_t v7, scalar_t v8,
                                                scalar_t x, scalar_t y)
{
    return interp1D_dx(interp2D(v1, v2, v3, v4, x, y), interp2D(v5, v6, v7, v8, x, y));
}
