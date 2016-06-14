      program lennard_jones
      implicit none
      include 'param_lj.inc'
      double precision dens,temp,u,p,cv
      double precision coef_adia,cs,eta,thc,bulkv,sa
      logical lrc !long range corrections

      lrc=.false.

      rcutfluid=2.5

      read(*,*) dens,temp

c      call lj(temp,dens,lrc,u,p,cv,
c     &       coef_adia,cs,eta,thc,bulkv,sa)

      call lj_p_u(temp,dens,lrc,u,p)

      write(*,*) dens,temp,u,p
      end
* este programa proporciona la equacion de estado 
* calorica u=u(rho,t) y la ecuacion de estado termica p=p(rho,t)
* de un gas lennard jones, para un amplio rango de temperaturas y densidades.
* la fuente es kj. karl johnson, john a. zollweg and keith e gubbins, mol. phys v78, no3, 591-618

      subroutine lj_p_u(t,dens,lrc,u,p)
      implicit none
      include 'param_lj.inc'
      double precision a(10), b(10),c(10),d(10),g(10),f,p,u,gamm,t1
      double precision t2,t3,t4,st,dens
      double precision tempinit,tempfin,densinit,densfin,t
      double precision plrc,ulrc,ssr3
      double precision x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
      double precision x13,x14,x15,x16,x17
      double precision x18,x19, x20,x21,x22,x23,x24,x25,x26,x27
      double precision x28,x29,x30,x31,x32,sigma
      integer i,j,k
      logical lrc

      sigma=1.0
c      lrc = true => se inluyen correciones de largo alcanze a p y u
      ssr3=(sigma/rcutfluid)**3
      plrc=32.*pi*dens**2*(ssr3**3-1.5*ssr3)/9.0
      ulrc=8.*pi*dens**2*(ssr3**3-3*ssr3)/9.0
      


c     parameters of the eq. 
      x1=0.862308509
      x2=2.9762187
      x3=-8.402230115
      x4=0.105413
      x5=-0.856458
      x6=1.582759
      x7=0.763942
      x8=1.753173
      x9=2.798291772e+3
      x10=-4.839422023e-2
      x11=0.996326519
      x12=-3.698000291e+1
      x13=2.084012e+1
      x14=8.305402e+1
      x15=-9.574799e+2
      x16=-1.477746e+2
      x17=6.398607852e+1
      x18=1.603993673e+1
      x19=6.805916e+1
      x20=-2.791293578e+3
      x21=-6.245128304
      x22=-8.116836104e+3
      x23=1.488735e+1
      x24=-1.059346e+4
      x25=-1.131607e+2
      x26=-8.867771e+3
      x27=-3.986982e+1
      x28=-4.689270e+3
      x29=2.593535e+2
      x30=-2.694523e+3
      x31=-7.218487e+2
      x32=1.721802063e+2


c     temperature functions

      st=sqrt(t)
      t1=1.0/t
      t2=1.0/t**2
      t3=1.0/t**3
      t4=1.0/t**4

c     more parameters
      a(1)=x1 *t + x2 *st +x3 + x4 *t1 + x5 *t2
      a(2)=x6 *t + x7 + x8 *t1 + x9 *t2
      a(3)=x10 *t + x11 + x12 *t1
      a(4)=x13
      a(5)=x14 *t1 +x15 *t2
      a(6)=x16 *t1
      a(7)=x17 *t1 + x18 *t2
      a(8)=x19 *t2

      b(1)=x20 *t2 + x21 *t3
      b(2)=x22 *t2 + x23 *t4
      b(3)=x24 *t2 + x25 *t3
      b(4)=x26 *t2 + x27 *t4
      b(5)=x28 *t2 + x29 *t3
      b(6)=x30 *t2 + x31 *t3 + x32 *t4

      c(1)=x2*st/2.  + x3 + 2*x4 *t1 + 3*x5*t2
      c(2)=x7 +2*x8*t1 +3*x9*t2
      c(3)=x11 +2*x12*t1
      c(4)=x13
      c(5)= 2*x14*t1 +3*x15*t2
      c(6)= 2*x16*t1
      c(7)= 2*x17*t1 +3*x18*t2
      c(8)= 3*x19*t2

      d(1)=   3 *x20*t2 +4*x21*t3
      d(2)=   3*x22*t2 +5*x23*t4
      d(3 )=  3*x24*t2 +4*x25*t3
      d(4)=   3*x26*t2 +5*x27*t4
      d(5)=   3*x28*t2 +4*x29*t3
      d(6)=   3*x30*t2 +4*x31*t3 +5*x32*t4


c     more parameters
      gamm=3.0
      f=exp(-gamm*dens**2)
      g(1)=      (1.-f)/(2*gamm)
      g(2)=-(f*dens**2-2*g(1) )/(2*gamm)
      g(3)=-(f*dens**4-4*g(2) )/(2*gamm)
      g(4)=-(f*dens**6-6*g(3) )/(2*gamm)
      g(5)=-(f*dens**8-8*g(4) )/(2*gamm)
      g(6)=-(f*dens**10-10*g(5) )/(2*gamm)



c   calculate  internal energy=u, pressure = p

      p=0.
      u=0.

      do i=1,8
         p=p+a(i)*dens**(i+1)
         u=u+c(i)*dens**i/float(i)
      end do
      



      do j=1,6
         u=u+ d(j)*g(j)
         p=p+f*b(j)*dens**(2*j+1)
      end do



      p=p+dens*t




      if(.not.lrc) then
      u=u-ulrc
      p=p-plrc
      end if



      return
      end 


