#ifndef TENSOR_CUH
#define TENSOR_CUH

namespace uammd{

    struct tensor3{

        real xx,xy,xz;
        real yx,yy,yz;
        real zx,zy,zz;

        VECATTR tensor3(real xx,real xy,real xz,
                        real yx,real yy,real yz,
                        real zx,real zy,real zz):xx(xx),xy(xy),xz(xz),
                                                 yx(yx),yy(yy),yz(yz),
                                                 zx(zx),zy(zy),zz(zz){}

        VECATTR tensor3(real init):xx(init),xy(init),xz(init),
                                   yx(init),yy(init),yz(init),
                                   zx(init),zy(init),zz(init){}

        VECATTR tensor3():tensor3(real(0.0)){};

        VECATTR tensor3(const tensor3& t){

            this->xx = t.xx;
            this->yx = t.yx;
            this->zx = t.zx;

            this->xy = t.xy;
            this->yy = t.yy;
            this->zy = t.zy;

            this->xz = t.xz;
            this->yz = t.yz;
            this->zz = t.zz;


        }

        VECATTR real3 diag(){return {xx,yy,zz};}

        VECATTR real trace(){return xx+yy+zz;}

        friend std::ostream& operator<<(std::ostream& os, const tensor3& t);

    };

    std::ostream& operator<<(std::ostream& os, const tensor3& t){

        os << t.xx << " " << t.xy << " " << t.xz << " "  
           << t.yx << " " << t.yy << " " << t.yz << " "
           << t.zx << " " << t.zy << " " << t.zz;

        return os;
    }

    ///////////////////////TENSOR3///////////////////////////////

    VECATTR  tensor3 operator +(const tensor3 &a, const tensor3 &b){

        tensor3 sum;

        sum.xx = a.xx + b.xx;
        sum.yx = a.yx + b.yx;
        sum.zx = a.zx + b.zx;

        sum.xy = a.xy + b.xy;
        sum.yy = a.yy + b.yy;
        sum.zy = a.zy + b.zy;

        sum.xz = a.xz + b.xz;
        sum.yz = a.yz + b.yz;
        sum.zz = a.zz + b.zz;

        return sum;
    }

    VECATTR  void operator +=(tensor3 &a, const tensor3 &b){

        a.xx += b.xx;
        a.yx += b.yx;
        a.zx += b.zx;

        a.xy += b.xy;
        a.yy += b.yy;
        a.zy += b.zy;

        a.xz += b.xz;
        a.yz += b.yz;
        a.zz += b.zz;
    }

    VECATTR  tensor3 operator +(const tensor3 &a, const real &b){

        tensor3 sum;

        sum.xx = a.xx + b;
        sum.yx = a.yx + b;
        sum.zx = a.zx + b;

        sum.xy = a.xy + b;
        sum.yy = a.yy + b;
        sum.zy = a.zy + b;

        sum.xz = a.xz + b;
        sum.yz = a.yz + b;
        sum.zz = a.zz + b;

        return sum;
    }

    VECATTR  tensor3 operator +(const real &b, const tensor3 &a){return a+b;}

    VECATTR  void operator +=(tensor3 &a, const real &b){

        a.xx += b;
        a.yx += b;
        a.zx += b;

        a.xy += b;
        a.yy += b;
        a.zy += b;

        a.xz += b;
        a.yz += b;
        a.zz += b;
    }

    VECATTR  tensor3 operator -(const tensor3 &a, const tensor3 &b){

        tensor3 sub;

        sub.xx = a.xx - b.xx;
        sub.yx = a.yx - b.yx;
        sub.zx = a.zx - b.zx;

        sub.xy = a.xy - b.xy;
        sub.yy = a.yy - b.yy;
        sub.zy = a.zy - b.zy;

        sub.xz = a.xz - b.xz;
        sub.yz = a.yz - b.yz;
        sub.zz = a.zz - b.zz;

        return sub;
    }

    VECATTR  void operator -=(tensor3 &a, const tensor3 &b){

        a.xx -= b.xx;
        a.yx -= b.yx;
        a.zx -= b.zx;

        a.xy -= b.xy;
        a.yy -= b.yy;
        a.zy -= b.zy;

        a.xz -= b.xz;
        a.yz -= b.yz;
        a.zz -= b.zz;
    }

    VECATTR  tensor3 operator -(const tensor3 &a, const real &b){

        tensor3 sub;

        sub.xx = a.xx - b;
        sub.yx = a.yx - b;
        sub.zx = a.zx - b;

        sub.xy = a.xy - b;
        sub.yy = a.yy - b;
        sub.zy = a.zy - b;

        sub.xz = a.xz - b;
        sub.yz = a.yz - b;
        sub.zz = a.zz - b;

        return sub;
    }

    VECATTR  tensor3 operator -(const real &a, const tensor3 &b){

        tensor3 sub;

        sub.xx = a - b.xx;
        sub.yx = a - b.yx;
        sub.zx = a - b.zx;

        sub.xy = a - b.xy;
        sub.yy = a - b.yy;
        sub.zy = a - b.zy;

        sub.xz = a - b.xz;
        sub.yz = a - b.yz;
        sub.zz = a - b.zz;

        return sub;
    }

    VECATTR  void operator -=(tensor3 &a, const real &b){
        a.xx -= b;
        a.yx -= b;
        a.zx -= b;

        a.xy -= b;
        a.yy -= b;
        a.zy -= b;

        a.xz -= b;
        a.yz -= b;
        a.zz -= b;
    }

    VECATTR  tensor3 operator *(const tensor3 &a, const tensor3 &b){

        tensor3 pro;

        pro.xx = a.xx * b.xx;
        pro.yx = a.yx * b.yx;
        pro.zx = a.zx * b.zx;

        pro.xy = a.xy * b.xy;
        pro.yy = a.yy * b.yy;
        pro.zy = a.zy * b.zy;

        pro.xz = a.xz * b.xz;
        pro.yz = a.yz * b.yz;
        pro.zz = a.zz * b.zz;

        return pro;
    }

    VECATTR  void operator *=(tensor3 &a, const tensor3 &b){

        a.xx = a.xx * b.xx;
        a.yx = a.yx * b.yx;
        a.zx = a.zx * b.zx;

        a.xy = a.xy * b.xy;
        a.yy = a.yy * b.yy;
        a.zy = a.zy * b.zy;

        a.xz = a.xz * b.xz;
        a.yz = a.yz * b.yz;
        a.zz = a.zz * b.zz;

    }

    VECATTR  tensor3 operator *(const tensor3 &a, const real &b){

        tensor3 pro;

        pro.xx = a.xx * b;
        pro.yx = a.yx * b;
        pro.zx = a.zx * b;

        pro.xy = a.xy * b;
        pro.yy = a.yy * b;
        pro.zy = a.zy * b;

        pro.xz = a.xz * b;
        pro.yz = a.yz * b;
        pro.zz = a.zz * b;

        return pro;
    }

    VECATTR  tensor3 operator *(const real &a, const tensor3 &b){

        tensor3 pro;

        pro.xx = a * b.xx;
        pro.yx = a * b.yx;
        pro.zx = a * b.zx;

        pro.xy = a * b.xy;
        pro.yy = a * b.yy;
        pro.zy = a * b.zy;

        pro.xz = a * b.xz;
        pro.yz = a * b.yz;
        pro.zz = a * b.zz;

        return pro;
    }

    VECATTR  void operator *=(tensor3 &a, const real &b){
        a.xx = a.xx * b;
        a.yx = a.yx * b;
        a.zx = a.zx * b;

        a.xy = a.xy * b;
        a.yy = a.yy * b;
        a.zy = a.zy * b;

        a.xz = a.xz * b;
        a.yz = a.yz * b;
        a.zz = a.zz * b;
    }

    VECATTR  tensor3 operator /(const tensor3 &a, const tensor3 &b){

        tensor3 div;

        div.xx = a.xx / b.xx;
        div.yx = a.yx / b.yx;
        div.zx = a.zx / b.zx;

        div.xy = a.xy / b.xy;
        div.yy = a.yy / b.yy;
        div.zy = a.zy / b.zy;

        div.xz = a.xz / b.xz;
        div.yz = a.yz / b.yz;
        div.zz = a.zz / b.zz;

        return div;
    }

    VECATTR  void operator /=(tensor3 &a, const tensor3 &b){

        a.xx /= b.xx;
        a.yx /= b.yx;
        a.zx /= b.zx;

        a.xy /= b.xy;
        a.yy /= b.yy;
        a.zy /= b.zy;

        a.xz /= b.xz;
        a.yz /= b.yz;
        a.zz /= b.zz;

    }

    VECATTR  tensor3 operator /(const tensor3 &a, const real &b){
        return (real(1.0)/b)*a;
    }

    VECATTR  tensor3 operator /(const real &a, const tensor3 &b){

        tensor3 div;

        div.xx = a / b.xx;
        div.yx = a / b.yx;
        div.zx = a / b.zx;

        div.xy = a / b.xy;
        div.yy = a / b.yy;
        div.zy = a / b.zy;

        div.xz = a / b.xz;
        div.yz = a / b.yz;
        div.zz = a / b.zz;

        return div;
    }

    VECATTR void operator /=(tensor3 &a, const real &b){
        a *= real(1.0)/b;
    }

    VECATTR tensor3 outer(const real3 &a, const real3 &b){

        tensor3 result;

        result.xx = a.x * b.x;
        result.yx = a.y * b.x;
        result.zx = a.z * b.x;

        result.xy = a.x * b.y;
        result.yy = a.y * b.y;
        result.zy = a.z * b.y;

        result.xz = a.x * b.z;
        result.yz = a.y * b.z;
        result.zz = a.z * b.z;

        return result;
    }

    struct tensor4{

        real xx,xy,xz,xw;
        real yx,yy,yz,yw;
        real zx,zy,zz,zw;
        real wx,wy,wz,ww;

        VECATTR tensor4(real xx,real xy,real xz,real xw,
                real yx,real yy,real yz,real yw,
                real zx,real zy,real zz,real zw,
                real wx,real wy,real wz,real ww):xx(xx),xy(xy),xz(xz),xw(xw),
        yx(yx),yy(yy),yz(yz),yw(yw),
        zx(zx),zy(zy),zz(zz),zw(zw),
        wx(wx),wy(wy),wz(wz),ww(ww){}

        VECATTR tensor4(real init):xx(init),xy(init),xz(init),xw(init),
        yx(init),yy(init),yz(init),yw(init),
        zx(init),zy(init),zz(init),zw(init),
        wx(init),wy(init),wz(init),ww(init){}

        VECATTR tensor4():tensor4(real(0.0)){};

        /*
        VECATTR tensor4(real4 v1,real4 v2){

            //v1v2T-v2Tv1

            this->xx = real(0.0);
            this->yx = v1.y*v2.x-v2.y*v1.x;
            this->zx = v1.z*v2.x-v2.z*v1.x;
            this->wx = v1.w*v2.x-v2.w*v1.x;

            this->xy = v1.x*v2.y-v2.x*v1.y;
            this->yy = real(0.0);
            this->zy = v1.z*v2.y-v2.z*v1.y;
            this->wy = v1.w*v2.y-v2.w*v1.y;

            this->xz = v1.x*v2.z-v2.x*v1.z;
            this->yz = v1.y*v2.z-v2.y*v1.z;
            this->zz = real(0.0);
            this->wz = v1.w*v2.z-v2.w*v1.z;

            this->xw = v1.x*v2.w-v2.x*v1.w;
            this->yw = v1.y*v2.w-v2.y*v1.w;
            this->zw = v1.z*v2.w-v2.z*v1.w;
            this->ww = real(0.0);
        }*/

        VECATTR tensor4(const tensor4& t){

            this->xx = t.xx;
            this->yx = t.yx;
            this->zx = t.zx;
            this->wx = t.wx;

            this->xy = t.xy;
            this->yy = t.yy;
            this->zy = t.zy;
            this->wy = t.wy;

            this->xz = t.xz;
            this->yz = t.yz;
            this->zz = t.zz;
            this->wz = t.wz;

            this->xw = t.xw;
            this->yw = t.yw;
            this->zw = t.zw;
            this->ww = t.ww;

        }

    };

    VECATTR  tensor4 operator +(const tensor4 &a, const tensor4 &b){

        tensor4 sum;

        sum.xx = a.xx + b.xx;
        sum.yx = a.yx + b.yx;
        sum.zx = a.zx + b.zx;
        sum.wx = a.wx + b.wx;

        sum.xy = a.xy + b.xy;
        sum.yy = a.yy + b.yy;
        sum.zy = a.zy + b.zy;
        sum.wy = a.wy + b.wy;

        sum.xz = a.xz + b.xz;
        sum.yz = a.yz + b.yz;
        sum.zz = a.zz + b.zz;
        sum.wz = a.wz + b.wz;

        sum.xw = a.xw + b.xw;
        sum.yw = a.yw + b.yw;
        sum.zw = a.zw + b.zw;
        sum.ww = a.ww + b.ww;

        return sum;
    }

    VECATTR  tensor4 operator *(const real &a, const tensor4 &b){

        tensor4 pro;

        pro.xx = a * b.xx;
        pro.yx = a * b.yx;
        pro.zx = a * b.zx;
        pro.wx = a * b.wx;

        pro.xy = a * b.xy;
        pro.yy = a * b.yy;
        pro.zy = a * b.zy;
        pro.wy = a * b.wy;

        pro.xz = a * b.xz;
        pro.yz = a * b.yz;
        pro.zz = a * b.zz;
        pro.wz = a * b.wz;

        pro.xw = a * b.xw;
        pro.yw = a * b.yw;
        pro.zw = a * b.zw;
        pro.ww = a * b.ww;

        return pro;
    }

    VECATTR  real4 operator *(const tensor4 &A, const real4 &v){

        real4 R;

        R.x = A.xx*v.x + A.xy*v.y + A.xz*v.z + A.xw*v.w;
        R.y = A.yx*v.x + A.yy*v.y + A.yz*v.z + A.yw*v.w;
        R.z = A.zx*v.x + A.zy*v.y + A.zz*v.z + A.zw*v.w;
        R.w = A.wx*v.x + A.wy*v.y + A.wz*v.z + A.ww*v.w;

        return R;
    }

    VECATTR  tensor4 operator *(const tensor4 &A, const tensor4 &B){

        tensor4 R;

        R.xx = A.xx*B.xx + A.xy*B.yx + A.xz*B.zx + A.xw*B.wx;
        R.xy = A.xx*B.xy + A.xy*B.yy + A.xz*B.zy + A.xw*B.wy;
        R.xz = A.xx*B.xz + A.xy*B.yz + A.xz*B.zz + A.xw*B.wz;
        R.xw = A.xx*B.xw + A.xy*B.yw + A.xz*B.zw + A.xw*B.ww;
        R.yx = A.yx*B.xx + A.yy*B.yx + A.yz*B.zx + A.yw*B.wx;
        R.yy = A.yx*B.xy + A.yy*B.yy + A.yz*B.zy + A.yw*B.wy;
        R.yz = A.yx*B.xz + A.yy*B.yz + A.yz*B.zz + A.yw*B.wz;
        R.yw = A.yx*B.xw + A.yy*B.yw + A.yz*B.zw + A.yw*B.ww;
        R.zx = A.zx*B.xx + A.zy*B.yx + A.zz*B.zx + A.zw*B.wx;
        R.zy = A.zx*B.xy + A.zy*B.yy + A.zz*B.zy + A.zw*B.wy;
        R.zz = A.zx*B.xz + A.zy*B.yz + A.zz*B.zz + A.zw*B.wz;
        R.zw = A.zx*B.xw + A.zy*B.yw + A.zz*B.zw + A.zw*B.ww;
        R.wx = A.wx*B.xx + A.wy*B.yx + A.wz*B.zx + A.ww*B.wx;
        R.wy = A.wx*B.xy + A.wy*B.yy + A.wz*B.zy + A.ww*B.wy;
        R.wz = A.wx*B.xz + A.wy*B.yz + A.wz*B.zz + A.ww*B.wz;
        R.ww = A.wx*B.xw + A.wy*B.yw + A.wz*B.zw + A.ww*B.ww;

        return R;
    }

}

#endif
