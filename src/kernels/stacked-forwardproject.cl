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

 constant sampler_t volumeSampler = CLK_NORMALIZED_COORDS_FALSE |
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

        const int idx = get_global_id(0);
        const int idy = get_global_id(1);
        const int idz = get_global_id(2);

        float angle = idy * angle_step;
        float r = fmin (axis_pos, size - axis_pos);
        float d = idx - axis_pos + 0.5f;
        float l = sqrt(4.0f*r*r - 4.0f*d*d);

        float2 D = (float2) (cos(angle), sin(angle));

        D = normalize(D);

        float2 N = (float2) (D.y, -D.x);

        float2 sample = d * D - l/2.0f * N + ((float2) (axis_pos, axis_pos));

        float2 sum = {0.0f,0.0f};

        for (int i = 0; i < l; i++) {
            sum += read_imagef(slices, volumeSampler, (float4)((float2)sample,idz,0.0f)).xy;
            sample += N;
        }

        reconstructed_buffer[idx + idy*size + idz*size*size] = sum;
  }


 /*kernel void
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
          int l_index = local_idy/4;

          int2 remapped_index_local   = {(4*square + 2*(quadrant%2) + (pixel%2)),
                                         (2* (quadrant/2) + (pixel/2))};

          int2 remapped_index_global  = {(get_group_id(0)*get_local_size(0)+remapped_index_local.x),
                                          (get_group_id(1)*get_local_size(1)+remapped_index_local.y)};

          float angle = remapped_index_global.y * angle_step;
          float r = fmin (axis_pos, size - axis_pos);           // radius of object circle
          float d = remapped_index_global.x - axis_pos + 0.5f;  // positive/negative distance from detector center
          float l = sqrt(4.0f*r*r - 4.0f*d*d);                  // length of the cut through the circle

          float2 D = (float2) (cos(angle), sin(angle)); //vector in detector direction
          D = normalize(D);

          float2 N = (float2) (D.y, -D.x); //vector perpendicular to the detector

          float2 sample = d * D - l/2.0f * N + ((float2) (axis_pos, axis_pos));

          float2 sum[4] = {0.0f,0.0f};
          __local float2 shared_mem[64][4];
          __local float2 reconstructed_cache[16][16];

          sample += l_index*N;

          for(int len = l_index; len < l; len+=4) {
              for(int q=0; q<4; q+=1){
                 sum[q] += read_imagef(slices, volumeSampler, (float4)((float2)sample+q*N,idz,0.0f)).xy;
              }
              sample += 4*N;
          }

          int2 remapped_index = {(local_idx%4), (4*local_idy + (local_idx/4))};

          for(int q=0; q<4;q+=1){
              // Moving partial sums to shared memory
              shared_mem[(local_sizex*remapped_index_local.y + remapped_index_local.x)][l_index] = sum[q];

              barrier(CLK_LOCAL_MEM_FENCE); // syncthreads

              for(int i=2; i>=1; i/=2){
                  if(remapped_index.x <i){
                      shared_mem[remapped_index.y][remapped_index.x] += shared_mem[remapped_index.y][remapped_index.x+i];
                  }
                  barrier(CLK_GLOBAL_MEM_FENCE); // syncthreads
              }

              if(remapped_index.x == 0){
                  reconstructed_cache[4*q+remapped_index.y/16][remapped_index.y%16] = shared_mem[remapped_index.y][0];
              }
              barrier(CLK_LOCAL_MEM_FENCE); // syncthreads
          }

          reconstructed_buffer[global_idx + global_idy*size + idz*size*size] = reconstructed_cache[local_idy][local_idx];
  }*/

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

  kernel void
  interleave_half ( global float *slices,
                    write_only image2d_array_t interleaved_slices)
  {
      const int idx = get_global_id(0);
      const int idy = get_global_id(1);
      const int idz = get_global_id(2);

      const int sizex = get_global_size(0);
      const int sizey = get_global_size(1);

      int slice_offset = idz*4;

      float4 b = {slices[idx + idy * sizex + (slice_offset) * sizex * sizey],
                  slices[idx + idy * sizex + (slice_offset+1) * sizex * sizey],
                  slices[idx + idy * sizex + (slice_offset+2) * sizex * sizey],
                  slices[idx + idy * sizex + (slice_offset+3) * sizex * sizey]};

      write_imagef(interleaved_slices, (int4)(idx, idy, idz, 0),(float4)(b));
  }

kernel void
texture_half (
        read_only image2d_array_t slices,
        global float4 *reconstructed_buffer,
        float axis_pos,
        float angle_step,
        unsigned long size){

      const int idx = get_global_id(0);
      const int idy = get_global_id(1);
      const int idz = get_global_id(2);

      float angle = idy * angle_step;
      float r = fmin (axis_pos, size - axis_pos);
      float d = idx - axis_pos + 0.5f;
      float l = sqrt(4.0f*r*r - 4.0f*d*d);

      float2 D = (float2) (cos(angle), sin(angle));

      D = normalize(D);

      float2 N = (float2) (D.y, -D.x);

      float2 sample = d * D - l/2.0f * N + ((float2) (axis_pos, axis_pos));

      float4 sum = {0.0f,0.0f,0.0f,0.0f};

      for (int i = 0; i < l; i++) {
          sum += read_imagef(slices, volumeSampler, (float4)((float2)sample,idz,0.0f));
          sample += N;
      }

      reconstructed_buffer[idx + idy*size + idz*size*size] = sum;
}

 kernel void
 uninterleave_half (global float4 *reconstructed_buffer,
                 global float *output)
 {
     const int idx = get_global_id(0);
     const int idy = get_global_id(1);
     const int idz = get_global_id(2);

     const int sizex = get_global_size(0);
     const int sizey = get_global_size(1);

     int output_offset = idz*4;

     output[idx + idy*sizex + (output_offset)*sizex*sizey] = reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].x;
     output[idx + idy*sizex + (output_offset+1)*sizex*sizey] = reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].y;
     output[idx + idy*sizex + (output_offset+2)*sizex*sizey] = reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].z;
     output[idx + idy*sizex + (output_offset+3)*sizex*sizey] = reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].w;
 }

 union converter {
     uint2 storage;
     uchar8 a;
 };

  kernel void
  interleave_uint ( global float *slices,
                    write_only image2d_array_t interleaved_slices,
                    const float min,
                    const float max)
  {
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);

    int slice_offset = idz*8;

    const float scale = 255.0f / (max - min);

    union converter il;
    il.a.s0 = (slices[idx + idy * sizex + (slice_offset) * sizex * sizey] - min)*scale;
    il.a.s1 = (slices[idx + idy * sizex + (slice_offset+1) * sizex * sizey] - min)*scale;
    il.a.s2 = (slices[idx + idy * sizex + (slice_offset+2) * sizex * sizey] - min)*scale;
    il.a.s3 = (slices[idx + idy * sizex + (slice_offset+3) * sizex * sizey] - min)*scale;
    il.a.s4 = (slices[idx + idy * sizex + (slice_offset+4) * sizex * sizey] - min)*scale;
    il.a.s5 = (slices[idx + idy * sizex + (slice_offset+5) * sizex * sizey] - min)*scale;
    il.a.s6 = (slices[idx + idy * sizex + (slice_offset+6) * sizex * sizey] - min)*scale;
    il.a.s7 = (slices[idx + idy * sizex + (slice_offset+7) * sizex * sizey] - min)*scale;

    write_imageui(interleaved_slices, (int4)(idx, idy, idz, 0),(uint4)((uint)il.storage.x,(uint)il.storage.y,0,0));
  }

kernel void
texture_uint (
        read_only image2d_array_t slices,
        global uint8 *reconstructed_buffer,
        float axis_pos,
        float angle_step,
        unsigned long size){

      const int idx = get_global_id(0);
      const int idy = get_global_id(1);
      const int idz = get_global_id(2);

      float angle = idy * angle_step;
      float r = fmin (axis_pos, size - axis_pos);
      float d = idx - axis_pos + 0.5f;
      float l = sqrt(4.0f*r*r - 4.0f*d*d);

      float2 D = (float2) (cos(angle), sin(angle));

      D = normalize(D);

      float2 N = (float2) (D.y, -D.x);

      float2 sample = d * D - l/2.0f * N + ((float2) (axis_pos, axis_pos));

      uint8 sum = {0,0,0,0,0,0,0,0};

      union converter tex;
      for (int i = 0; i < l; i++) {
          tex.storage = read_imageui(slices, volumeSampler, (float4)((float2)sample,idz,0.0f)).xy;

         sum.s0 += (uint)tex.a.s0;
         sum.s1 += (uint)tex.a.s1;
         sum.s2 += (uint)tex.a.s2;
         sum.s3 += (uint)tex.a.s3;
         sum.s4 += (uint)tex.a.s4;
         sum.s5 += (uint)tex.a.s5;
         sum.s6 += (uint)tex.a.s6;
         sum.s7 += (uint)tex.a.s7;

         sample += N;
      }

      reconstructed_buffer[idx + idy*size + idz*size*size] = sum;
}

 kernel void
 uninterleave_uint (global uint8 *reconstructed_buffer,
                    global float *output,
                    const float min,
                    const float max)
 {
     const int idx = get_global_id(0);
     const int idy = get_global_id(1);
     const int idz = get_global_id(2);

     const int sizex = get_global_size(0);
     const int sizey = get_global_size(1);

     int output_offset = idz*8;
     float scale = (max-min)/255.0f;

     output[idx + idy*sizex + (output_offset)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].s0)*scale+min;
     output[idx + idy*sizex + (output_offset+1)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].s1)*scale+min;
     output[idx + idy*sizex + (output_offset+2)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].s2)*scale+min;
     output[idx + idy*sizex + (output_offset+3)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].s3)*scale+min;
     output[idx + idy*sizex + (output_offset+4)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].s4)*scale+min;
     output[idx + idy*sizex + (output_offset+5)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].s5)*scale+min;
     output[idx + idy*sizex + (output_offset+6)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].s6)*scale+min;
     output[idx + idy*sizex + (output_offset+7)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizex + idz*sizex*sizey].s7)*scale+min;
 }