!======================================================================
! File:        armijo.f90
! Description: solve e.g. 2.3.2 using Armijo rule 
! 
! Created on:  Mon Sep 22 2025 19:25:58
! Author:      Shen Yang
! University:  Hunan Normal University
! Email:       yangshen@hunnu.edu.cn
!======================================================================

program armijo
    use iso_fortran_env, only: real64
    implicit none
    integer, parameter :: dp = real64

    real(dp) :: alpha = 1.0_dp
    real(dp) :: beta = 0.5_dp
    real(dp) :: rho = 0.5_dp
    real(dp) :: sigma = 0.9_dp

    real(dp) :: x(2) = [1.0_dp, 1.0_dp]
    real(dp) :: d(2) = [1.0_dp, -1.0_dp]
    real(dp) :: f0, f1

    integer :: iter = 0
    integer :: max_iter = 1000

    f0 = f(x)
    f1 = f(x + alpha*d)

    if (f1 <= f0 + sigma*alpha*grad_f_d(x,d)) then
        print *, "alpha = ", alpha
    else
        alpha = beta
        do while (f1 > f0 + sigma*alpha*grad_f_d(x,d) .and. iter < max_iter)
            alpha = alpha * rho
            f1 = f(x + alpha*d)
            iter = iter + 1
        end do
    end if

    print *, "Number of iterations: ", iter
    print *, "alpha = ", alpha

contains
    pure real(dp) function f(z)
        real(dp), intent(in) :: z(:)
        f = 0.5_dp*z(1)**2 + z(2)**2
    end function f

    pure real(dp) function grad_f_d(z1, z2)
        real(dp), intent(in) :: z1(:)
        real(dp), intent(in) :: z2(:)
        grad_f_d = z1(1)*z2(1) + 2.0_dp*z1(2)*z2(2)
    end function grad_f_d

end program armijo

