! This file is part of pdfflow
program example

   use, intrinsic :: ISO_C_BINDING, only: C_ptr, c_char, C_NULL_ptr
   implicit none

   type(C_ptr), save :: thePDF = C_NULL_ptr

   character(kind=c_char, len=21) :: ss = "NNPDF31_nlo_as_0118/0"
   character(kind=c_char, len=24) :: pp = "/usr/share/lhapdf/LHAPDF/"

   call mkPDF(ss, pp)

end program
