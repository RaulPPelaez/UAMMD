/***************************  bravais.c  ****************************

  Generation of Bravais lattices. The program calculates positions

         R = i a1 + j a2 + k a3

  within a simulation box of side length L and inserts the basis at
  each point of the lattice.
  Author: Marc Meléndez Schofield. */

/***** Pre-defined Bravais lattice types *****/
#ifndef BRAVAIS_H
#define BRAVAIS_H
/***** Pre-defined Bravais lattice types *****/
typedef enum {sc, bcc, fcc, dia, hcp, sq, tri} BRAVAISLAT;
#define MAXBASIS 1000 /* Maximum number of user basis elements */
#ifdef __cplusplus
#else
typedef enum {false=0, true} bool;
#endif


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/***** Create Bravais lattice *****/
void Bravais(float * pos,  /* Pointer to positions array */
             BRAVAISLAT type, /* Lattice (sc, bcc, fcc, dia, hcp, sq, tri) */
             int N,        /* Number of nodes ( = particles) */
             float Lx, float Ly, float Lz, /* Box dimensions */
             float colour, /* Colour */
             FILE * basisfile, FILE * vectorsfile, /* Basis and vectors files */
             bool keepaspect) /* Keep aspect ratio (true or false) */
{
  int nx, ny, nz; /* Number of lattice nodes on sides */
  int node = 0; /* Lattice node number */
  float V; /* Box volume */
  int ncells; /* Number of unit cells */
  int i, j, k, l, d, n; /* Indices */
  float stretchfactor[3], minstretch; /* Stretch factors */
  float L[3] = {Lx, Ly, Lz}; /* Box dimensions */
  float e[3][3]; /* Lattice vectors */
  float basis[8][3]; /* Basis positions */
  int nbasis = 1; /* Number of elements in the basis */
  float pbasis[MAXBASIS][5]; /* User basis elements */
  int npbasis = -1; /* Number of user basis elements */
  int nvectors = -1; /* Number of user lattice vectors */
  float r[3]; /* Position vector */
  char buffer[250]; /* Buffer to store file data */
  int count; /* Number of elements read */

  /* Clear all the lattice vector components */
  for(i = 0; i < 3; i++)
    for(j = 0; j < 3; j++)
      e[i][j] = 0;

  /* Clear personalised basis elements */
  for(j = 0; j < MAXBASIS; j++)
    for(k = 0; k < 5; k++)
      pbasis[j][k] = 0;

  /* Get basis from file */
  if(basisfile != NULL) {
    /* Read data from file */
    npbasis = 0;
    while(fgets(buffer, 250, basisfile) && npbasis < MAXBASIS) {
      count = sscanf(buffer, "%f %f %f %f", &pbasis[npbasis][0],
                             &pbasis[npbasis][1], &pbasis[npbasis][2],
                             &pbasis[npbasis][3]);
      if(count > 0) npbasis++;
    }
    fclose(basisfile);
  }

  /* Get vectors from file */
  if(vectorsfile != NULL) {
    /* Read data from file */
    nvectors = 0;
    while(fgets(buffer, 250, vectorsfile) && nvectors < 3) {
      count = sscanf(buffer, "%f %f %f", &e[nvectors][0], &e[nvectors][1],
                                         &e[nvectors][2]);
      if(count > 0) nvectors++;
    }
    fclose(vectorsfile);
  }

  /*** Definition of the lattice vectors (if undefined by user) ***/
  if(nvectors < 0) {
    /* Write non-zero components of lattice vectors */
    switch(type) {
      case sc:
      case bcc:
      case fcc:
      case dia:
        e[0][0] = e[1][1] = e[2][2] = 1;
        break;
      case hcp:
        e[0][0] = 1;
        e[1][0] = 0.5;  e[1][1] = sqrt(3)/2;
        e[2][2] = 2*sqrt(6)/3;
        break;
      case sq:
        e[0][0] = e[1][1] = 1;
        L[2] = 1; /* Lz does not contribute to the volume */
        break;
      case tri:
        e[0][0] = 1;
        e[1][0] = 0.5;  e[1][1] = sqrt(3)/2;
        L[2] = 1; /* Lz does not contribute to the volume */
        break;
    }
  }

  /*** Definition of the basis positions ***/
  for(i = 0; i < 8; i++)
    for(j = 0; j < 3; j++)
      basis[i][j] = 0; /* Clear all the basis positions */

  /* Non-zero basis positions */
  switch(type)
  {
    case bcc:
      nbasis = 2;
      basis[1][0] = basis[1][1] = basis[1][2] = 0.5;
      break;
    case fcc:
      nbasis = 4;
      basis[1][0] = basis[1][1] = 0.5;
      basis[2][0] = basis[2][2] = 0.5;
      basis[3][1] = basis[3][2] = 0.5;
      break;
    case dia:
      nbasis = 8;
      basis[1][0] = basis[1][1] = 0.5;
      basis[2][0] = basis[2][2] = 0.5;
      basis[3][1] = basis[3][2] = 0.5;
      basis[4][0] = basis[4][1] = basis[4][2] = 0.25;
      basis[5][0] = basis[5][1] = 0.75; basis[5][2] = 0.25;
      basis[6][0] = basis[6][2] = 0.75; basis[6][1] = 0.25;
      basis[7][1] = basis[7][2] = 0.75; basis[6][0] = 0.25;
      break;
    case hcp:
      nbasis = 2;
      basis[1][0] = .5; basis[1][1] = 0.25; basis[1][2] = sqrt(6.)/3;
      break;
    default:
      break;
  }

  /* Total number of unit cells */
  if(npbasis > 0)
    ncells = ceil(N/(1.f*nbasis*npbasis));
  else
    ncells = ceil(N/(1.f*nbasis));
  /* Box volume */
  V = L[0]*L[1]*L[2];

  /* Unit cells on the side of the box */
  if(type == sq || type == tri) { /* 2D lattices */
    nx = ceil(sqrt(ncells/V)*L[0]);
    ny = ceil(ncells/(1.0f*nx));
    nz = 1;
  }
  else { /* 3D lattices */
    nx = ceil(pow(ncells/V, 1/3.)*L[0]);
    ny = ceil(pow(ncells/V, 1/3.)*L[1]);
    nz = ceil(ncells/(1.0f*nx*ny));
  }
  /* Stretch factors */
  stretchfactor[0] = L[0]/(nx*e[0][0]);
  stretchfactor[1] = L[1]/(ny*e[1][1]);
  stretchfactor[2] = L[2]/(nz*e[2][2]);

  if(keepaspect) {
    minstretch = (stretchfactor[0] < stretchfactor[1])?
                     stretchfactor[0]:
                     stretchfactor[1];
    if(type != sq || type != tri)
      minstretch = (minstretch < stretchfactor[2])?
                     minstretch:
                     stretchfactor[2];
      for(i = 0; i < 3; i++) stretchfactor[i] = minstretch;
  }


  // printf("%d %d %d %.3f %.3f %.3f\n", nx,ny, nz, pow(ncells/V, 1/3.)*L[0], pow(ncells/V, 1/3.)*L[1], pow(ncells/V, 1/3.)*L[2]);

  //     printf("%.3f %.3f %.3f\n", stretchfactor[0], stretchfactor[1],stretchfactor[2]);

  
  /* Positions in the Bravais lattice */
  for(i = 0; i < nx; i++) {
    for(j = 0; j < ny; j++) {
      for(k = 0; k < nz; k++) {
        for(l = 0; l < nbasis; l++) { /* Loop over the elements in the basis */
          if(node >= N) return;
          for(d = 0; d < 3; d++) { /* Loop over dimensions */
            r[d] = -L[d]/2. + stretchfactor[d]*(i*e[0][d] + j*e[1][d]
                                              + k*e[2][d] + basis[l][d]);
          }

          /* Wrap-around (triangular and hexagonal closed-packed lattices) */
          if(type == tri || type == hcp)
            if(r[0] > L[0]/2) r[0] -= L[0];

          /* No third dimension for 2D lattices */
          if(type == sq || type == tri)
            r[2] = 0;

          /* Write positions */
          if(npbasis > 0) {
            if(node >= N) return;
            for(n = 0; n < npbasis; n++) { /* Write user basis */
              for(d = 0; d < 3; d++)
                pos[d] = r[d] + stretchfactor[d]*pbasis[n][d];

              pos[3] = pbasis[n][3]; /* Particle colour */

              pos += 4;/* Next position */
              node++; /* Next node */
            }
          }
          else { /* Output lattice node */
            for(d = 0; d < 3; d++) pos[d] = r[d];
            pos[3] = colour; /* Particle colour */
            pos += 4; /* Next position */
            node++; /* Next node */
          }
        }
      }
    }
  }

  return;
}



#endif
