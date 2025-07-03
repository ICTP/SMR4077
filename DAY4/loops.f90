program loops
      use openacc
      implicit none
      
      integer,allocatable :: a(:),b(:),c(:) 
      integer(kind=8)     :: n
      integer             :: i,j,k
      n=10_8**10

      allocate(a(n))
      allocate(b(n))
      allocate(c(n))

     do i=1,n
        a(i)=3
        b(i)=4
     end do

     do j=1,n
        a(j) = 2*a(j)
        b(j) = b(j) +1      
     end do 

     do k=1,n
        c(k) = a(k) +b(k)
     end do   

     print *, c(1:10)
end program  
