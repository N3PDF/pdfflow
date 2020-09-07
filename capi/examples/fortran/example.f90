! This file is part of pdfflow
program example

   use, intrinsic :: ISO_C_BINDING, only: C_ptr, c_char
   implicit none

   integer, parameter :: dp = kind(1.d0)

   integer :: pid
   real(dp) :: alpha_s, q2, x, xfx

   character(kind=c_char, len=21) :: ss = "NNPDF31_nlo_as_0118/0"
   character(kind=c_char, len=24) :: pp = "/usr/share/lhapdf/LHAPDF/"

   call mkPDF(ss, pp)

   q2 = 1.65
   x = 0.1
   call alphasq2(q2, alpha_s)

   write(*, '(A, F7.5)')"The value of alpha_s is: ", alpha_s

   write(*, fmt=100)"Value of x*f(x) at q2=", q2, " x=",x
   do pid = -5, 5
      call xfxq2(pid, x, q2, xfx)
      write(*, fmt=200)"Flavour: ", pid, " value: ", xfx
   enddo

100 format (A, F6.2, A, F4.2)
200 format ("    ", A, I0, A, F10.7)

end program
