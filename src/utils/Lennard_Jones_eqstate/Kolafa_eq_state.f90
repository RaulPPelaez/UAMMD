program llamada

print*,alj(0.4,0.2)

contains
c===================================================================
C      Package supplying the thermodynamic properties of the
C      LENNARD-JONES fluid
c
c      J. Kolafa, I. Nezbeda, Fluid Phase Equil. 100 (1994), 1
c
c      ALJ(T,rho)...Helmholtz free energy (including the ideal term)
c      PLJ(T,rho)...Pressure
c      ULJ(T,rho)...Internal energy
c===================================================================
      DOUBLE PRECISION FUNCTION ALJ(T,rho)
C      Helmholtz free energy (including the ideal term)
c
      implicit double precision (a-h,o-z)
      data pi /3.141592654d0/
      eta = PI/6.*rho * (dC(T))**3
      ALJ =  (dlog(rho)+betaAHS(eta)
     +  +rho*BC(T)/exp(gammaBH(T)*rho**2))*T
     +  +DALJ(T,rho)
      RETURN
      END
C/* Helmholtz free energy (without ideal term) */
      DOUBLE PRECISION FUNCTION ALJres(T,rho)
      implicit double precision (a-h,o-z)
      data pi /3.141592654d0/
      eta = PI/6. *rho*(dC(T))**3
      ALJres = (betaAHS(eta)
     + +rho*BC(T)/exp(gammaBH(T)*rho**2))*T
     + +DALJ(T,rho)
      RETURN
      END
C/* pressure */
      DOUBLE PRECISION FUNCTION PLJ(T,rho)
      implicit double precision (a-h,o-z)
      data pi /3.141592654d0/
      eta=PI/6. *rho*(dC(T))**3
      sum=((2.01546797*2+rho*(
     + (-28.17881636)*3+rho*(
     + 28.28313847*4+rho*
     + (-10.42402873)*5)))
     + +((-19.58371655)*2+rho*(
     + +75.62340289*3+rho*(
     + (-120.70586598)*4+rho*(
     + +93.92740328*5+rho*
     + (-27.37737354)*6))))/dsqrt(T)
     + + ((29.34470520*2+rho*(
     + (-112.35356937)*3+rho*(
     + +170.64908980*4+rho*(
     + (-123.06669187)*5+rho*
     + 34.42288969*6))))+
     + ((-13.37031968)*2+rho*(
     + 65.38059570*3+rho*(
     + (-115.09233113)*4+rho*(
     + 88.91973082*5+rho*
     + (-25.62099890)*6))))/T)/T)*rho**2
       PLJ = ((zHS(eta)
     +  + BC(T)/exp(gammaBH(T)*rho**2)
     +  *rho*(1-2*gammaBH(T)*rho**2))*T
     +  +sum )*rho
       RETURN
       END
C/* internal energy */
      DOUBLE PRECISION FUNCTION ULJ( T, rho)
      implicit double precision (a-h,o-z)
       data pi /3.141592654d0/
      dBHdT=dCdT(T)
      dB2BHdT=BCdT(T)
      d=dC(T)
      eta=PI/6. *rho*d**3
      sum= ((2.01546797+rho*(
     + (-28.17881636)+rho*(
     + +28.28313847+rho*
     + (-10.42402873))))
     + + (-19.58371655*1.5+rho*(
     + 75.62340289*1.5+rho*(
     + (-120.70586598)*1.5+rho*(
     + 93.92740328*1.5+rho*
     + (-27.37737354)*1.5))))/dsqrt(T)
     + + ((29.34470520*2+rho*(
     + -112.35356937*2+rho*(
     +  170.64908980*2+rho*(
     + -123.06669187*2+rho*
     + 34.42288969*2)))) +
     + (-13.37031968*3+rho*(
     +  65.38059570*3+rho*(
     +  -115.09233113*3+rho*(
     + 88.91973082*3+rho*
     + (-25.62099890)*3))))/T)/T) *rho*rho
      ULJ = 3*(zHS(eta)-1)*dBHdT/d
     + +rho*dB2BHdT/exp(gammaBH(T)*rho**2) +sum
      RETURN
      END
      DOUBLE PRECISION FUNCTION zHS(eta)
      implicit double precision (a-h,o-z)
      zHS = (1+eta*(1+eta*(1-eta/1.5*(1+eta)))) / (1-eta)**3
      RETURN
      END
      DOUBLE PRECISION FUNCTION betaAHS( eta )
      implicit double precision (a-h,o-z)
      betaAHS = dlog(1-eta)/0.6
     +  + eta*( (4.0/6*eta-33.0/6)*eta+34.0/6 ) /(1.-eta)**2
      RETURN
      END
C /* hBH diameter */
      DOUBLE PRECISION FUNCTION dLJ(T)
      implicit double precision (a-h,o-z)
      DOUBLE PRECISION IST
      isT=1/dsqrt(T)
      dLJ = ((( 0.011117524191338 *isT-0.076383859168060)
     + *isT)*isT+0.000693129033539)/isT+1.080142247540047
     + +0.127841935018828*dlog(isT)
      RETURN
      END
      DOUBLE PRECISION FUNCTION dC(T)
      implicit double precision (a-h,o-z)
      sT=dsqrt(T)
      dC = -0.063920968*dlog(T)+0.011117524/T
     +     -0.076383859/sT+1.080142248+0.000693129*sT
      RETURN
      END
      DOUBLE PRECISION FUNCTION dCdT( T)
      implicit double precision (a-h,o-z)
      sT=dsqrt(T)
      dCdT =   0.063920968*T+0.011117524+(-0.5*0.076383859
     +   -0.5*0.000693129*T)*sT
      RETURN
      END
      DOUBLE PRECISION FUNCTION BC( T)
      implicit double precision (a-h,o-z)
      DOUBLE PRECISION isT
      isT=1/dsqrt(T)
      BC = (((((-0.58544978*isT+0.43102052)*isT
     +  +.87361369)*isT-4.13749995)*isT+2.90616279)*isT
     +  -7.02181962)/T+0.02459877
      RETURN
      END
      DOUBLE PRECISION FUNCTION BCdT( T)
      implicit double precision (a-h,o-z)
      DOUBLE PRECISION iST
      isT=1/dsqrt(T)
      BCdT = ((((-0.58544978*3.5*isT+0.43102052*3)*isT
     +  +0.87361369*2.5)*isT-4.13749995*2)*isT
     +  +2.90616279*1.5)*isT-7.02181962
      RETURN
      END
      DOUBLE PRECISION FUNCTION gammaBH(X)
      implicit double precision (a-h,o-z)
      gammaBH=1.92907278
      RETURN
      END
      DOUBLE PRECISION FUNCTION DALJ(T,rho)
      implicit double precision (a-h,o-z)
      DALJ = ((+2.01546797+rho*(-28.17881636
     + +rho*(+28.28313847+rho*(-10.42402873))))
     + +(-19.58371655+rho*(75.62340289+rho*((-120.70586598)
     + +rho*(93.92740328+rho*(-27.37737354)))))/dsqrt(T)
     + + ( (29.34470520+rho*((-112.35356937)
     + +rho*(+170.64908980+rho*((-123.06669187)
     + +rho*34.42288969))))
     + +(-13.37031968+rho*(65.38059570+
     + rho*((-115.09233113)+rho*(88.91973082
     + +rho* (-25.62099890)))))/T)/T) *rho*rho
      RETURN
      END

    end program 
