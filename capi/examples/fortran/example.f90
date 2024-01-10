! This file is part of pdfflow
program example

   use, intrinsic :: ISO_C_BINDING, only: C_ptr, c_char
   implicit none

   integer, parameter :: dp = kind(1.d0)

   integer :: pid(0:10), i
   real(dp) :: alpha_s, q2(0:1), x(0:1), xfx(0:11)
   real(dp) :: q2_vectorized(0:2), as_vectorized(0:2)

   character(kind=c_char, len=23) :: ss = "NNPDF40_nnlo_as_01180/0"
   character(kind=c_char, len=24) :: pp = "/usr/share/lhapdf/LHAPDF/"

   call mkPDF(ss, pp)

   do i = 0, 10
      pid(i) = i - 5
   enddo
   x(0) = 0.1
   q2(0) = 1.65

   call xfxq2(pid, 11, x, 1, q2, 1, xfx)

   do i = 0, 10
      write(*, fmt=200)"Flavour: ", i - 5, " value: ", xfx(i)
   enddo

   q2_vectorized(0) = 1.65
   q2_vectorized(1) = 10.65
   call alphasq2(q2_vectorized, 2, as_vectorized);

   write(*, fmt=100)"alphas(q2=",q2_vectorized(0),") = ", as_vectorized(0)
   write(*, fmt=100)"alphas(q2=",q2_vectorized(1),") = ", as_vectorized(1)

100 format (A, F10.7, A, F10.7)
200 format ("    ", A, I0, A, F10.7)

end program
