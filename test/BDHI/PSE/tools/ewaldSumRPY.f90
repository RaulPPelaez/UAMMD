  !This code is based in the paper:

  !Ewald sum of the Rotne-Prager tensor
  !C. W. J. Beenakker
  !J. Chem. Phys. 85(3), 1581 (1986)

  !However, we use the RPY tensor with corrections
  !for overlapping particles, see eq. 8 and 9 of
  !R. Kekre et al. Phys. Rev. E 82, 011802, 2010.

  !The code computes the hydrodynamic force
  !between two particles at different distances

  !F_i0 = 6*pi*eta*a*S_i0 / (g0 + g1 + g2) / F_Stokes

  !The terms g0, g1, g2 are given (with different notation)
  !in the Eq. 4 of Beenaker's paper.

  !HOW TO USE:
  !modify variables (line 42), compile and run.




  program ewaldSum
    implicit none
    integer :: i, l, nPoints, n, fourierModes
    integer :: ix, iy, iz

    double precision :: lx, ly, lz
    double precision :: dx, a
    double precision :: x0, y0, z0, x1, y1, z1, x2, y2, z2
    double precision :: u, eta
    double precision :: kx, ky, kz, k
    double precision :: m2Value
    double precision :: rCell, rPart, r
    double precision :: g0, g1, g2
    double precision :: volume, zeta, pi
    double precision :: m1, m2
    double precision :: x
    pi = 4.d0*datan(1.d0)
    
    !DEFINE VARIBALES
    a = 1.0 !Particle radius
    lx = 64.d0 !Box size
    ly = 64.d0 !Box size
    lz = 64.d0 !Box size
    volume = lx * ly * lz
    nPoints = 1000 !
    eta = 1.0d0 !viscosity is not used
    fourierModes = 50
    !END DEFINE VARIABLES

    dx = lx/(2.d0 * nPoints)

    zeta = dsqrt(pi) /(volume**(1.d0/3.d0)) 

    !Coordinates particle 1
    x0 = 0.d0
    y0 = 0.d0
    z0 = 0.d0
    !Coordinates particle 2
    x2 = 0.d0
    y2 = 0.d0
    z2 = 0.d0

    !g0
    g0 = 1.d0 - 6.d0 * zeta * a/dsqrt(pi) + 40.d0*((zeta*a)**3)/(3.d0*dsqrt(pi))
    
    do n = 0, nPoints-1

       !g1
       g1 = 0.d0
       !Particle 2 is moving
       x2 = lx/2.d0 - n*dx
       do ix = -10, 10
          do iy = -10, 10
             do iz = -10, 10
                if(.true.) then
                !if((ix**2+iy**2+iz**2).le.fourierModes) then
                   !write(*,*) "#bucle 1", ix**2+iy**2+iz**2
                   x1 = x2 + ix * lx
                   y1 = y2 + iy * ly
                   z1 = z2 + iz * lz
                   r = dsqrt(x1**2 + y1**2 + z1**2)
                   x = x1/r
                   g1 = g1 - m1(r,zeta,a,pi,x)
                   
                   if((ix.ne.0).or.(iy.ne.0).or.(iz.ne.0)) then
                      x1 = ix * lx
                      y1 = iy * ly
                      z1 = iz * lz
                      r = dsqrt(x1**2 + y1**2 + z1**2)
                      x = x1/r
                      g1 = g1 + m1(r,zeta,a,pi,x)
                   end if
                end if
             end do
          end do
       end do

       !g2
       g2=0.d0
       do ix = -10, 10
          do iy = -10, 10
             do iz = -10, 10
                if( (ix.ne.0).or.(iy.ne.0).or.(iz.ne.0) ) then
                   !if(((ix.ne.0).or.(iy.ne.0).or.(iz.ne.0)).and.((ix**2+iy**2+iz**2).le.fourierModes)) then
                   kx = 2.d0*pi/lx * ix
                   ky = 2.d0*pi/ly * iy
                   kz = 2.d0*pi/lz * iz
                   k = dsqrt(kx**2+ky**2+kz**2)
                   m2Value = m2(k,kx,a,zeta,pi)
                   g2 = g2 + m2Value*(1.d0-dcos(kx*x2+ky*y2*kz*z2))/volume
                end if
             end do
          end do
       end do
       !write(*,*) x2, 6.d0*pi*eta*a*u/(g0+g1+g2)

       !if r<2a, rest RPY without overlap correction and 
       !add RPY with overlap correction
       if(x2.lt.2*a) then
          !rest RP no overlap correction
          g1 = g1 + 1.5*a/x2 - (a/x2)**3 !Sign - because the force is -F in the second particle
          !add RPY with overlap correction
          g1 = g1 - 1 + 6*x2 / (a*32.d0)
       end if

       !Divided by the Stokes force
       ! write(*,*) x2, 1.d0/(g0)
       write(*,*) x2, 1.d0/(g0+g1+g2)*(1-2.837297d0/lx)
    end do



    write(*,*) "#END", zeta
  end program ewaldSum



double precision function m1(r,zeta,a,pi,x)
  double precision :: r, zeta, a, pi, x1, x
  m1 = ( &
       (0.75d0 * a/r + 0.5d0 * (a/r)**3) * derfc(zeta*r) + &
       (4.d0*zeta**7*a**3*r**4 + &
       3.d0*zeta**3*a*r**2 - &
       20.d0*zeta**5*a**3*r**2 - &
       4.5d0*zeta*a + &
       14.d0*(zeta*a)**3 + &
       zeta*a**3/(r**2)) * dexp(-(zeta*r)**2)/dsqrt(pi)) + &
       x*x*( &
       (0.75d0*a/r - 1.5*(a/r)**3)*derfc(zeta*r) + &
       (-4.d0*zeta**7*a**3*r**4 - &
       3.d0*zeta**3*a*r**2 + &
       16.d0*zeta**5*a**3*r**2 + &
       1.5d0*zeta*a - &
       2.d0*(zeta*a)**3 - &
       3.d0*zeta*a**3/r**2) * dexp(-(zeta*r)**2)/dsqrt(pi) &
       )
       

end function m1


double precision function m2(k,kx,a,zeta,pi)
  double precision :: k, kx, a, zeta, pi
  double precision :: kxn
  kxn = kx/k
  m2 = (1-kxn*kxn) * &
       (a-(a**3*k**2)/3.d0) * &
       (1.d0+0.25d0*(k/zeta)**2 + 0.125d0*(k/zeta)**4) * &
       6.d0*pi*dexp(-0.25*(k/zeta)**2)/k**2

end function m2
