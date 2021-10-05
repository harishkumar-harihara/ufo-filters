/*
 * Copyright (C) 2011-2013 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

 constant sampler_t volumeSampler_single = CLK_NORMALIZED_COORDS_FALSE |
                                           CLK_ADDRESS_CLAMP |
                                           CLK_FILTER_NEAREST;

 kernel void
 interleave_single ( global float *slices,
                     write_only image2d_array_t interleaved_slices)
 {
     const int idx = get_global_id(0);
     const int idy = get_global_id(1);
     const int idz = get_global_id(2);
     const int sizex = get_global_size(0);
     const int sizey = get_global_size(1);

     int slice_offset = idz*2;

     float x = slices[idx + idy * sizex + (slice_offset) * sizex * sizey];
     float y = slices[idx + idy * sizex + (slice_offset+1) * sizex * sizey];

     write_imagef(interleaved_slices, (int4)(idx, idy, idz, 0),(float4)(x,y,0.0f,0.0f));
 }

 kernel void
 texture_single (
         read_only image2d_array_t slices,
         global float2 *reconstructed_buffer,
         float axis_pos,
         float angle_step,
         unsigned long size){

         const int local_idx = get_local_id(0);
         const int local_idy = get_local_id(1);

         const int global_idx = get_global_id(0);
         const int global_idy = get_global_id(1);
         const int idz = get_global_id(2);

         int local_sizex = get_local_size(0);
         int local_sizey = get_local_size(1);

         int global_sizex = get_global_size(0);
         int global_sizey = get_global_size(1);

         // Computing sequential numbers of 4x4 square, quadrant, and pixel within quadrant
         int square = local_idy%4;
         int quadrant = local_idx/4;
         int pixel = local_idx%4;

         // Computing projection and pixel offsets
         int projection_index = local_idy/4;

         int2 remapped_index_local   = {(4*square + 2*(quadrant%2) + (pixel%2)),
                                        (2* (quadrant/2) + (pixel/2))};

         int2 remapped_index_global  = {(get_group_id(0)*get_local_size(0)+remapped_index_local.x),
                                         (get_group_id(1)*get_local_size(1)+remapped_index_local.y)};

         const float angle = remapped_index_global.y * angle_step;
         const float r = fmin (axis_pos, size - axis_pos);          // radius of object circle
         const float d = remapped_index_global.x - axis_pos + 0.5f; // positive/negative distance from detector center
         const float l = sqrt(4.0f*r*r - 4.0f*d*d);                 // length of the cut through the circle

         float2 D = (float2) (cos(angle), sin(angle)); //vector in detector direction
         D = normalize(D);

         const float2 N = (float2) (D.y, -D.x); //vector perpendicular to the detector

         float2 sample = d * D - l/2.0f * N + ((float2) (axis_pos, axis_pos));

         float2 sum[4] = {0.0f,0.0f};
         __local float2 shared_mem[64][4];
         __local float2 reconstructed_cache[16][16];

         for(int proj = projection_index; proj < l; proj+=4) {
             for(int q=0; q<4; q+=1){
                    sum[q] += read_imagef(slices, volumeSampler_single, (float4)((float2)sample+q*N,idz,0.0f)).xy;
                    //sample += N;
            }
         }

         if(get_global_id(0)==200 && get_global_id(1)==200 && get_global_id(2)==0){
             printf("\n %f %f %f %f",sum[0],sum[1],sum[2],sum[3]);
         }

         int2 remapped_index = {(local_idx%4), (4*local_idy + (local_idx/4))};

         for(int q=0; q<4;q+=1){
             // Moving partial sums to shared memory
             shared_mem[(local_sizex*remapped_index_local.y + remapped_index_local.x)][projection_index] = sum[q];

             barrier(CLK_LOCAL_MEM_FENCE); // syncthreads

             for(int i=2; i>=1; i/=2){
                 if(remapped_index.x <=i){
                     shared_mem[remapped_index.y][remapped_index.x] += shared_mem[remapped_index.y][remapped_index.x+i];
                 }
                 barrier(CLK_GLOBAL_MEM_FENCE); // syncthreads
             }

             if(remapped_index.x == 0){
                 reconstructed_cache[4*q+remapped_index.y/16][remapped_index.y%16] = shared_mem[remapped_index.y][0];
             }
             barrier(CLK_LOCAL_MEM_FENCE); // syncthreads
         }

         if(get_global_id(0)==200 && get_global_id(1)==200 && get_global_id(2)==0){
            printf("\n %f ",reconstructed_cache[local_idy][local_idx]);
         }

         reconstructed_buffer[global_idx + global_idy*size + idz*size*size] = (reconstructed_cache[local_idy][local_idx]);
 }

 kernel void
 uninterleave_single (global float2 *reconstructed_buffer,
                 global float *output)
 {
     const int idx = get_global_id(0);
     const int idy = get_global_id(1);
     const int idz = get_global_id(2);
     const int sizex = get_global_size(0);
     const int sizey = get_global_size(1);
     int output_offset = idz*2;

     output[idx + idy*sizex + (output_offset)*sizex*sizey] = reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].x;
     output[idx + idy*sizex + (output_offset+1)*sizex*sizey] = reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].y;
 }