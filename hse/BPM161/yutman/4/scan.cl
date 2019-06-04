__kernel void scan_blelloch(__global double * a, __global double * r, __global double * t, __local double * b, int N)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint group_id = get_group_id(0);

    if (gid < N) {
        uint dp = 1;

        b[lid] = a[gid];

        for(uint s = block_size>>1; s > 0; s >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lid < s)
            {
                uint i = dp*(2*lid+1)-1;
                uint j = dp*(2*lid+2)-1;
                b[j] += b[i];
            }

            dp <<= 1;
        }

        if(lid == 0) b[block_size - 1] = 0;

        for(uint s = 1; s < block_size; s <<= 1)
        {
            dp >>= 1;
            barrier(CLK_LOCAL_MEM_FENCE);

            if(lid < s)
            {
                uint i = dp*(2*lid+1)-1;
                uint j = dp*(2*lid+2)-1;

                int t = b[j];
                b[j] += b[i];
                b[i] = t;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == block_size - 1) {
            t[group_id] = b[lid] + a[gid];
        }
        r[gid] = b[lid];
    }
}

__kernel void add(__global double * r, __global double * t, int N) {
    int gid = get_global_id(0);
    int group_id = get_group_id(0);
    if (gid < N) {
        r[gid] += t[group_id];
    }
}