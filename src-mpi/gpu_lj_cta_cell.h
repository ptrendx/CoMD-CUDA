/*************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

__global__
__launch_bounds__(CTA_CELL_CTA, CTA_CELL_ACTIVE_CTAS)
void LJ_Force_cta_cell(SimGpu sim, int * cells_list, real_t rCut2, real_t s6)
{
  __shared__ real_t otherX[SHARED_SIZE_CTA_CELL];
  __shared__ real_t otherY[SHARED_SIZE_CTA_CELL];
  __shared__ real_t otherZ[SHARED_SIZE_CTA_CELL];
  __shared__ real_t otherR2[SHARED_SIZE_CTA_CELL];

  // compute box ID and local atom ID
  const int iBox = (cells_list == NULL)? blockIdx.x: cells_list[blockIdx.x]; 
  const int nAtoms = sim.boxes.nAtoms[iBox];
  
  // common constants for LJ potential
  const real_t epsilon = sim.lj_pot.epsilon;

  const real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
  const real_t eShift = rCut6 * (rCut6 - 1.0f);

  for(int iAtom = threadIdx.x; iAtom < MAXATOMS; iAtom += blockDim.x)
  {

      // zero out forces and energy
      real_t ifx = 0;
      real_t ify = 0;
      real_t ifz = 0;
      real_t ie = 0;
      
      real_t fr_sum = 0;

      // fetch position
      const int iOff = iBox * MAXATOMS + iAtom;

      const real_t irx = sim.atoms.r.x[iOff];
      const real_t iry = sim.atoms.r.y[iOff];
      const real_t irz = sim.atoms.r.z[iOff];

      const real_t myR2 = 0.5f*(irx*irx + iry*iry + irz*irz);

      // loop over my neighbor cells
      for (int j = 0; j < N_MAX_NEIGHBORS; j++) 
      {
          const int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
          for(int base = 0; base < MAXATOMS; base += SHARED_SIZE_CTA_CELL)
          {
              __syncthreads();
              //Fetch atom positions
#pragma unroll
              for(int i = 0; i < SHARED_SIZE_CTA_CELL; i += CTA_CELL_CTA)
              {
                  otherX[i+threadIdx.x] = sim.atoms.r.x[jBox*MAXATOMS + i + threadIdx.x + base];
                  otherY[i+threadIdx.x] = sim.atoms.r.y[jBox*MAXATOMS + i + threadIdx.x + base];
                  otherZ[i+threadIdx.x] = sim.atoms.r.z[jBox*MAXATOMS + i + threadIdx.x + base];
                  otherR2[i+threadIdx.x] = sim.atoms.r2[jBox*MAXATOMS + i + threadIdx.x + base];
              }
              __syncthreads();
              if(iAtom >= nAtoms)
                  continue;
              int maxN = SHARED_SIZE_CTA_CELL + base < sim.boxes.nAtoms[jBox]?SHARED_SIZE_CTA_CELL:sim.boxes.nAtoms[jBox]-base;
              // loop over all atoms in the neighbor cell 
              for (int jAtom = 0; jAtom < maxN; jAtom++) 
              {  
                  // distance^2
                  real_t r2 = 2.f*(myR2 - irx*otherX[jAtom] - iry*otherY[jAtom] - irz*otherZ[jAtom]) + otherR2[jAtom];
                  // no divide by zero
                  if (r2 <= rCut2 && r2 > 1e-2f)  
                  {
                      r2 = 1.0f/r2;
                      real_t r6 = s6 * (r2*r2*r2);
                      real_t eLocal = r6 * (r6 - 1.0f) - eShift;

                      // update energy
                      ie += 0.5f * eLocal;
                      // different formulation to avoid sqrt computation
                      real_t fr = r6*r2*(48.0f*r6 - 24.0f);

                      // update forces
                      ifx -= fr * otherX[jAtom];
                      ify -= fr * otherY[jAtom];
                      ifz -= fr * otherZ[jAtom];

                      fr_sum += fr;
                  }
              } // loop over all atoms
          }
      } // loop over neighbor cells

      ifx += fr_sum * irx;
      ify += fr_sum * iry;
      ifz += fr_sum * irz;

      sim.atoms.f.x[iOff] = ifx * epsilon;
      sim.atoms.f.y[iOff] = ify * epsilon;
      sim.atoms.f.z[iOff] = ifz * epsilon;

      sim.atoms.e[iOff] = ie * 4 * epsilon;
  }
}

