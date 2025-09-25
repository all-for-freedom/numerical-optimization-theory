!======================================================================
! File:        wolfe_powell.f90
! Description: 
!
! Created on:  Thu Sep 25 2025
! Author:      Shen Yang
! University:  Hunan Normal University
! Email:       yangshen@hunnu.edu.cn
!======================================================================

program wp
    use iso_fortran_env, only: real64
    implicit none
    integer, parameter :: dp = real64

    real(dp) :: a, b, x
    real(dp) :: beta, rho, rho1, sigma1, sigma2
    real(dp) :: tol
    integer  :: iter, iter_max
    real(dp) :: d, alpha

    a = 0.0_dp;  b = 3.0_dp
    x = 0.5_dp * (a + b)

    beta   = 0.5_dp
    rho    = 0.5_dp
    rho1   = 0.5_dp
    sigma1 = 0.1_dp
    sigma2 = 0.9_dp

    tol      = 1.0e-8_dp
    iter     = 0
    iter_max = 1000

    do while (abs(df(x)) > tol .and. iter < iter_max)
        iter  = iter + 1
        d     = -df(x)
        alpha = wolfe_powell(x, d, beta, rho, rho1, sigma1, sigma2)
        x     = x + alpha*d
    end do

    write(*,'(A,1X,ES24.16)') 'x:',    x
    write(*,'(A,1X,ES24.16)') 'f(x):', f(x)

contains
    pure real(dp) function f(x) result(val)
        real(dp), intent(in) :: x
        val = 3.0_dp*x**4 - 16.0_dp*x**3 + 30.0_dp*x**2 - 24.0_dp*x + 8.0_dp
    end function f

    pure real(dp) function df(x) result(val)
        real(dp), intent(in) :: x
        val = 12.0_dp*x**3 - 48.0_dp*x**2 + 60.0_dp*x - 24.0_dp
    end function df

    pure real(dp) function wolfe_powell(x, d, beta, rho, rho1, sigma1, sigma2) result(alpha)
        real(dp), intent(in) :: x, d
        real(dp), intent(in) :: beta, rho, rho1, sigma1, sigma2
        real(dp) :: fx, sigma1_df_d, sigma2_df_d
        real(dp) :: alpha_loc, beta_k, temp
        real(dp), parameter :: tol_alpha = 1.0e-16_dp

        fx          = f(x)
        sigma1_df_d = sigma1 * df(x) * d
        sigma2_df_d = sigma2 * df(x) * d

        ! Step 0
        if ( f(x + d) <= fx + sigma1_df_d .and. df(x + d)*d >= sigma2_df_d ) then
            alpha = 1.0_dp
            return
        end if

        ! Step 1
        alpha_loc = beta
        if ( f(x + alpha_loc*d) > fx + alpha_loc*sigma1_df_d ) then
            do
                alpha_loc = rho * alpha_loc
                if ( f(x + alpha_loc*d) <= fx + alpha_loc*sigma1_df_d ) exit
            end do
        else
            do
                if ( f(x + (alpha_loc/rho)*d) <= fx + (alpha_loc/rho)*sigma1_df_d ) then
                    alpha_loc = alpha_loc / rho
                else
                    exit
                end if
            end do
            alpha_loc = rho * alpha_loc
        end if

        ! Step 2 & 3
        do while ( df(x + alpha_loc*d)*d < sigma2_df_d )
            beta_k = alpha_loc / rho
            temp   = beta_k
            do
                if ( f(x + temp*d) <= fx + temp*sigma1_df_d ) exit
                temp = (temp - alpha_loc)*rho1 + alpha_loc
                if ( (temp - alpha_loc) < tol_alpha ) then
                    alpha = alpha_loc
                    return
                end if
            end do
            alpha_loc = temp
        end do

        alpha = alpha_loc
    end function wolfe_powell
end program wp