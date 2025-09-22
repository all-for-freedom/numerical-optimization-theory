!======================================================================
! File:        gs.f90
! Description: solve e.g. 2.3.1 using golden section search
! 
! Created on:  Mon Sep 22 2025 18:56:10
! Author:      Shen Yang
! University:  Hunan Normal University
! Email:       yangshen@hunnu.edu.cn
!======================================================================

program gs
    use iso_fortran_env, only: real64
    implicit none
    integer, parameter :: dp = real64

    real(dp) :: a = 0.0_dp
    real(dp) :: b = 3.0_dp
    
    real(dp) :: golden_ratio = (sqrt(5.0_dp)-1.0_dp)/2.0_dp

    real(dp) :: u
    real(dp) :: v

    real(dp) :: fu
    real(dp) :: fv

    real(dp) :: seps = 1.0e-12_dp
    real(dp) :: atol = 1.0e-12_dp
    real(dp) :: rtol = 1.0e-12_dp

    integer :: iter = 0
    integer :: max_iter = 1000

    u = b - golden_ratio * (b-a)
    v = a + golden_ratio * (b-a) 
    
    fu = f(u)
    fv = f(v)

    do while ( (abs(b-a) > seps) .and. (iter <= max_iter) )

        iter = iter + 1

        if ( abs(fu-fv) < (atol + rtol*max(1.0_dp,max(abs(fu),abs(fv)))) ) then
            a = u
            b = v
            u = b - golden_ratio * (b-a)
            v = a + golden_ratio * (b-a)
            fu = f(u)
            fv = f(v)
        else if (fu < fv) then
            b = v
            v = u
            u = b - golden_ratio * (b-a)
            fv = fu
            fu = f(u)
        else
            a = u
            u = v
            v = a + golden_ratio * (b-a)
            fu = fv
            fv = f(v)
        end if

    end do

    print *, 'iter =', iter
    print *, 'x =', (a+b)/2.0_dp    
    print *, 'f(x) =', f((a+b)/2.0_dp)

contains
    pure real(dp) function f(x)
        real(dp), intent(in) :: x
        f = 3.0_dp*x**4 - 16.0_dp*x**3 + 30.0_dp*x**2 - 24.0_dp*x + 8.0_dp
    end function f

end program gs