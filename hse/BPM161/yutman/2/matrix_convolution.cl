__kernel void convolve(__global double * a, __global double * b, __global double * c, int M, int N)
{
   int row = get_global_id(0);
   int col = get_global_id(1);
   if (row < N && col < N) {
      int HM = (M - 1) / 2;
      double ans = 0;
      for (int j = -HM; j <= HM; j++) {
         for (int k = -HM; k <= HM; k++) {
            if (row + j >= 0 && row + j < N && col + k >= 0 && col + k < N) {
               ans += a[(row + j) * N + col + k] * b[(HM + j) * M + HM + k];
            }
         }
      }
      c[row * N + col] = ans;
   }
}