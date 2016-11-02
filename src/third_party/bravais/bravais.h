/***************************  bravais.c  ****************************

  Generation of Bravais lattices. The program calculates positions

         R = i a1 + j a2 + k a3

  within a simulation box of side length L and inserts the basis at
  each point of the lattice.
  Author: Marc Meléndez Schofield. */

/***** Pre-defined Bravais lattice types *****/
typedef enum {sc, bcc, fcc, dia, hcp, sq, tri} BRAVAISLAT;
#define MAXBASIS 1000 /* Maximum number of user basis elements */
#ifdef __cplusplus
extern "C"{
#else
typedef enum {false=0, true} bool;
#endif
/***** Create Bravais lattice *****/
void Bravais(float * pos,  /* Pointer to positions array */
             BRAVAISLAT type, /* Lattice (sc, bcc, fcc, dia, hcp, sq, tri) */
             int N,        /* Number of nodes ( = particles) */
             float Lx, float Ly, float Lz, /* Box dimensions */
             float colour, /* Colour */
             FILE * basisfile, FILE * vectorsfile, /* Basis and vectors files */
             bool keepaspect); /* Keep aspect ratio (true or false) */

#ifdef __cplusplus
};
#endif
