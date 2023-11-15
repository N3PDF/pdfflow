! This file is part of pdfflow
program example

   use, intrinsic :: ISO_C_BINDING, only: C_ptr, c_char
   implicit none

   integer, parameter :: dp = kind(1.d0)

   integer :: pid(0:10), i
   real(dp) :: alpha_s, q2(0:1), x(0:1), xfx(0:10)

   character(kind=c_char, len=21) :: ss = "NNPDF31_nlo_as_0118/0"
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

100 format (A, F6.2, A, F4.2)
200 format ("    ", A, I0, A, F10.7)

end program
