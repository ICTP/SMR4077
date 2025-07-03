program loops
      use openacc
      implicit none
      
      integer,allocatable :: a(:),b(:),c(:) 
      integer(kind=8)     :: n
      integer             :: i,j,k
      n=10_8**9

      allocate(a(n))
      allocate(b(n))
      allocate(c(n))
!$acc data copyin(a(1:n),b(1:n)) copyout(c(1:n))
!$acc parallel
     do i=1,n
        a(i)=3
        b(i)=4
     end do

!$acc parallel
     do j=1,n
        a(j) = 2*a(j)
        b(j) = b(j) +1      
     end do 

!$acc parallel
     do k=1,n
        c(k) = a(k) +b(k)
     end do   
!$acc end data
     print *, c(1:10)
end program  
